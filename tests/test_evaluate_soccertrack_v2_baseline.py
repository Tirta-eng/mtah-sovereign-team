"""Unit tests for SoccerTrack V2 baseline evaluation tool.

Verifies:
  - Target loading and label encoding logic
  - Metric report serialization
  - Synthetic end-to-end evaluation workflow

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
    "eval_v2",
    Path(__file__).resolve().parents[1]
    / "tools/evaluate_soccertrack_v2_baseline.py",
)
EV = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EV)


class MockCoverageRouter:
    def __init__(self):
        self.classes_ = np.array(["A", "D", "H"])

    def predict_proba(self, X):
        n = len(X)
        probs = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            probs[i] = [0.2, 0.1, 0.7]  # A=0.2, D=0.1, H=0.7 -> H is argmax
        return probs


class EvaluateSoccertrackV2BaselineTests(unittest.TestCase):
    def test_label_mappings(self):
        self.assertEqual(EV.LABEL_MAP["H"], 0)
        self.assertEqual(EV.LABEL_MAP["A"], 1)
        self.assertEqual(EV.LABEL_MAP["D"], 2)
        self.assertEqual(EV.LABEL_MAP["X"], 3)
        self.assertEqual(EV.INV_LABEL_MAP[0], "H")
        self.assertEqual(EV.INV_LABEL_MAP[1], "A")
        self.assertEqual(EV.INV_LABEL_MAP[2], "D")
        self.assertEqual(EV.INV_LABEL_MAP[3], "X")

    def test_target_loading(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "targets.sqlite"
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute(
                    "CREATE TABLE targets (video_frame INTEGER PRIMARY KEY, label TEXT, valid_target INTEGER)"
                )
                conn.execute("INSERT INTO targets VALUES (0, 'H', 1)")
                conn.execute("INSERT INTO targets VALUES (1, 'A', 1)")
                conn.execute("INSERT INTO targets VALUES (2, NULL, 0)")
                conn.commit()

            # Test with custom total frames
            saved_total = EV.TOTAL_VIDEO_FRAMES
            EV.TOTAL_VIDEO_FRAMES = 3
            try:
                labels, valid = EV.load_ground_truth_targets(db_path)
                self.assertEqual(labels, ["H", "A", None])
                self.assertEqual(valid, [True, True, False])
            finally:
                EV.TOTAL_VIDEO_FRAMES = saved_total

    def test_synthetic_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            targets_db = tmp_path / "targets.sqlite"
            features_npz = tmp_path / "features.npz"
            out_dir = tmp_path / "eval_out"

            # Create synthetic targets (3 frames)
            with closing(sqlite3.connect(targets_db)) as conn:
                conn.execute(
                    """CREATE TABLE targets (
                        match_id TEXT, period INTEGER, video_frame INTEGER PRIMARY KEY,
                        raw_frame INTEGER, raw_time_ms INTEGER, ball_status TEXT,
                        label TEXT, valid_target INTEGER, invalid_reason TEXT
                    )"""
                )
                conn.execute("INSERT INTO targets VALUES ('117093', 1, 0, 277, 1040, 'HOME', 'H', 1, NULL)")
                conn.execute("INSERT INTO targets VALUES ('117093', 1, 1, 278, 1080, 'HOME', 'H', 1, NULL)")
                conn.execute("INSERT INTO targets VALUES ('117093', 1, 2, 279, 1120, 'NEUTRAL', NULL, 0, 'masked')")
                conn.commit()

            # Create synthetic features.npz
            np.savez_compressed(
                features_npz,
                video_frames=np.array([0, 1, 2], dtype=np.int32),
                raw_frames=np.array([277, 278, 279], dtype=np.int32),
                raw_times_ms=np.array([1040, 1080, 1120], dtype=np.int32),
                features=np.ones((3, 18), dtype=np.float64),
            )

            # Mock bundle loader
            mock_model = MockCoverageRouter()
            mock_params = {
                "decoder_version": 1,
                "classes": ["H", "A", "D"],
                "fps": 25,
                "state_prior": [0.33, 0.33, 0.34],
                "initial_prior": [0.33, 0.33, 0.34],
                "transition_matrix": [
                    [0.98, 0.01, 0.01],
                    [0.01, 0.98, 0.01],
                    [0.01, 0.01, 0.98],
                ],
                "emission": "log classifier posterior - log training state prior",
                "emission_weight": 1.0,
                "transition_weight": 1.0,
            }

            original_load_bundle = EV.load_bundle
            EV.load_bundle = lambda _: (mock_model, mock_params)

            saved_total = EV.TOTAL_VIDEO_FRAMES
            EV.TOTAL_VIDEO_FRAMES = 3
            try:
                report = EV.run_evaluation(
                    bundle_dir=tmp_path,
                    features_npz_path=features_npz,
                    targets_db_path=targets_db,
                    output_dir=out_dir,
                    bootstrap_n=0,
                )

                self.assertTrue((out_dir / "predictions.sqlite").exists())
                self.assertTrue((out_dir / "predictions.npz").exists())
                self.assertTrue((out_dir / "metrics.json").exists())
                self.assertTrue((out_dir / "manifest.json").exists())

                self.assertEqual(report["total_video_frames"], 3)
                self.assertEqual(report["graded_frames"], 2)
                self.assertEqual(report["masked_frames"], 1)

                # Check SQLite predictions
                with closing(sqlite3.connect(out_dir / "predictions.sqlite")) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT video_frame, raw_label, decoded_label, gt_label, valid_target FROM predictions ORDER BY video_frame"
                    )
                    rows = cursor.fetchall()
                    self.assertEqual(len(rows), 3)
                    self.assertEqual(rows[0], (0, "H", "H", "H", 1))
                    self.assertEqual(rows[1], (1, "H", "H", "H", 1))
                    self.assertEqual(rows[2], (2, "H", "H", None, 0))

                # Check NPZ predictions
                with np.load(out_dir / "predictions.npz") as pz:
                    self.assertEqual(pz["probabilities"].shape, (3, 3))
                    self.assertEqual(pz["raw_labels"].tolist(), ["H", "H", "H"])
                    self.assertEqual(pz["decoded_labels"].tolist(), ["H", "H", "H"])
                    self.assertEqual(pz["valid_targets"].tolist(), [True, True, False])

            finally:
                EV.load_bundle = original_load_bundle
                EV.TOTAL_VIDEO_FRAMES = saved_total


if __name__ == "__main__":
    unittest.main()
