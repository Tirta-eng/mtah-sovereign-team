"""Unit tests for SoccerTrack visual feature extractor.

Verifies:
  - Feature contract (8 visual features) and units
  - Homography projection from centred metres to pixel space
  - Per-frame visual feature computation on synthetic image patches
  - Synthetic end-to-end extraction workflow (SQLite + NPZ)

No official competition data, models, or network calls are accessed.
"""

from contextlib import closing
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "extract_vis",
    Path(__file__).resolve().parents[1]
    / "tools/extract_soccertrack_visual_features_117093.py",
)
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)


class ExtractSoccertrackVisualFeaturesTests(unittest.TestCase):
    def test_feature_contract(self):
        self.assertEqual(len(V.VISUAL_FEATURE_NAMES), 8)
        self.assertEqual(len(V.VISUAL_FEATURE_UNITS), 8)
        self.assertIn("vis_global_motion", V.VISUAL_FEATURE_NAMES)
        self.assertIn("vis_ball_vicinity_motion", V.VISUAL_FEATURE_NAMES)
        self.assertIn("vis_home_color_score", V.VISUAL_FEATURE_NAMES)
        self.assertIn("vis_away_color_score", V.VISUAL_FEATURE_NAMES)

    def test_pitch_to_pixel_projection(self):
        # Identity-like homography mapping pitch [0..105, 0..68] directly to [0..1050, 0..680]
        H_simple = np.array([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        # Centred (0, 0) -> pitch (52.5, 34.0) -> (525, 340)
        u, v = V.project_pitch_to_pixel(0.0, 0.0, H_simple, img_width=1050, img_height=680)
        self.assertEqual(u, 525)
        self.assertEqual(v, 340)

        # None coordinates -> fallback to pitch center
        u, v = V.project_pitch_to_pixel(None, None, H_simple, img_width=1050, img_height=680)
        self.assertEqual(u, 525)
        self.assertEqual(v, 340)

    def test_compute_visual_features_synthetic(self):
        # Create 2 synthetic frames
        h, w = 1080, 4096
        f0 = np.full((h, w, 3), 100, dtype=np.uint8)  # uniform gray
        f1 = f0.copy()

        # In f1, create blue patch at (1000, 500)
        f1[450:550, 950:1050, 0] = 220  # Blue channel high
        f1[450:550, 950:1050, 1] = 50   # Green low
        f1[450:550, 950:1050, 2] = 50   # Red low

        # Frame 0 (first frame, no previous)
        feats0, g0, crop0 = V.compute_visual_features_for_frame(
            frame=f0,
            prev_small_gray=None,
            prev_crop128=None,
            u=1000,
            v=500,
            img_width=w,
            img_height=h,
        )
        self.assertEqual(len(feats0), 8)
        self.assertEqual(feats0[0], 0.0)  # global motion is 0 on frame 0
        self.assertEqual(feats0[2], 0.0)  # ball motion is 0 on frame 0

        # Frame 1 (motion and blue color should be detected)
        feats1, g1, crop1 = V.compute_visual_features_for_frame(
            frame=f1,
            prev_small_gray=g0,
            prev_crop128=crop0,
            u=1000,
            v=500,
            img_width=w,
            img_height=h,
        )
        self.assertGreater(feats1[0], 0.0)  # global motion > 0
        self.assertGreater(feats1[2], 0.0)  # ball vicinity motion > 0
        self.assertGreater(feats1[5], 50.0) # home blue color score high!

    def test_synthetic_end_to_end_extraction(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_path = tmp_path / "synthetic_video.mp4"
            homography_path = tmp_path / "homography.npy"
            tracking_sqlite = tmp_path / "features.sqlite"
            out_dir = tmp_path / "visual_out"

            # 1. Write synthetic homography
            H = np.eye(3)
            np.save(homography_path, H)

            # 2. Write synthetic tracking sqlite
            with closing(sqlite3.connect(tracking_sqlite)) as conn:
                conn.execute(
                    """CREATE TABLE features (
                        match_id TEXT, period INTEGER, video_frame INTEGER PRIMARY KEY,
                        raw_frame INTEGER, raw_time_ms INTEGER, ball_x_m REAL, ball_y_m REAL
                    )"""
                )
                conn.execute("INSERT INTO features VALUES ('117093', 1, 0, 277, 1040, 0.0, 0.0)")
                conn.execute("INSERT INTO features VALUES ('117093', 1, 1, 278, 1080, 1.0, 0.0)")
                conn.execute("INSERT INTO features VALUES ('117093', 1, 2, 279, 1120, NULL, NULL)")
                conn.commit()

            # 3. Create synthetic 3-frame video
            fourcc = V.cv2.VideoWriter_fourcc(*"mp4v")
            writer = V.cv2.VideoWriter(str(video_path), fourcc, 25.0, (1024, 540))
            for i in range(3):
                frame = np.full((540, 1024, 3), 50 * (i + 1), dtype=np.uint8)
                writer.write(frame)
            writer.release()

            # 4. Run extractor
            summary = V.extract_and_save_visual_features(
                video_path=video_path,
                homography_path=homography_path,
                features_sqlite_path=tracking_sqlite,
                output_dir=out_dir,
                skip_hash_check=True,
            )

            self.assertEqual(summary["total_video_frames"], 3)
            self.assertEqual(summary["feature_count"], 8)
            self.assertTrue((out_dir / "visual_features.sqlite").exists())
            self.assertTrue((out_dir / "visual_features.npz").exists())
            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "manifest.json").exists())

            # Check SQLite
            with closing(sqlite3.connect(out_dir / "visual_features.sqlite")) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT video_frame, proj_u, proj_v, vis_global_motion FROM visual_features ORDER BY video_frame"
                )
                rows = cursor.fetchall()
                self.assertEqual(len(rows), 3)
                self.assertEqual(rows[0][0], 0)
                self.assertEqual(rows[0][3], 0.0) # Frame 0 global motion is 0
                self.assertGreater(rows[1][3], 0.0) # Frame 1 has motion!

            # Check NPZ
            with np.load(out_dir / "visual_features.npz") as pz:
                self.assertEqual(pz["visual_features"].shape, (3, 8))
                self.assertEqual(len(pz["video_frames"]), 3)


if __name__ == "__main__":
    unittest.main()
