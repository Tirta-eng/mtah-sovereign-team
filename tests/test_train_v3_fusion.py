"""Unit tests for Multimodal Fusion Model (V3) training script.

Verifies:
  - Feature concatenation (18 tracking + 8 visual = 26 total)
  - Dataset synchronization and target alignment
  - Synthetic end-to-end model training, Markov estimation, and Viterbi evaluation

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
    "train_v3",
    Path(__file__).resolve().parents[1]
    / "tools/train_soccertrack_v3_fusion.py",
)
TF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TF)


class TrainV3FusionTests(unittest.TestCase):
    def test_load_dataset_concatenation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            track_npz = tmp_path / "track.npz"
            vis_npz = tmp_path / "vis.npz"
            targets_db = tmp_path / "targets.sqlite"

            n = 6
            # 18 tracking features
            np.savez_compressed(
                track_npz,
                video_frames=np.arange(n, dtype=np.int32),
                raw_frames=np.arange(277, 277 + n, dtype=np.int32),
                raw_times_ms=np.arange(1040, 1040 + n * 40, 40, dtype=np.int32),
                features=np.ones((n, 18), dtype=np.float64),
                model_features=np.array([f"t_{i}" for i in range(18)]),
            )

            # 8 visual features
            np.savez_compressed(
                vis_npz,
                visual_features=np.full((n, 8), 2.0, dtype=np.float64),
                feature_names=np.array([f"v_{i}" for i in range(8)]),
            )

            # Targets sqlite
            with closing(sqlite3.connect(targets_db)) as conn:
                conn.execute(
                    "CREATE TABLE targets (video_frame INTEGER PRIMARY KEY, label TEXT, valid_target INTEGER)"
                )
                for i in range(n):
                    lbl = "H" if i % 3 == 0 else ("A" if i % 3 == 1 else "D")
                    conn.execute("INSERT INTO targets VALUES (?, ?, 1)", (i, lbl))
                conn.commit()

            X, v_frames, r_frames, r_times, names, valid, labels = TF.load_dataset(
                tracking_npz_path=track_npz,
                visual_npz_path=vis_npz,
                targets_sqlite_path=targets_db,
            )

            self.assertEqual(X.shape, (6, 26))
            self.assertEqual(len(names), 26)
            self.assertEqual(len(v_frames), 6)
            self.assertEqual(len(labels), 6)
            self.assertTrue(all(valid))
            # Verify feature values: first 18 are 1.0, last 8 are 2.0
            np.testing.assert_array_equal(X[:, :18], 1.0)
            np.testing.assert_array_equal(X[:, 18:], 2.0)

    def test_synthetic_end_to_end_training(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            track_npz = tmp_path / "track.npz"
            vis_npz = tmp_path / "vis.npz"
            targets_db = tmp_path / "targets.sqlite"
            out_dir = tmp_path / "fusion_out"

            # 30 frames: alternating H, A, D
            n = 30
            np.savez_compressed(
                track_npz,
                video_frames=np.arange(n, dtype=np.int32),
                raw_frames=np.arange(277, 277 + n, dtype=np.int32),
                raw_times_ms=np.arange(1040, 1040 + n * 40, 40, dtype=np.int32),
                features=np.random.RandomState(42).randn(n, 18),
                model_features=np.array([f"t_{i}" for i in range(18)]),
            )

            np.savez_compressed(
                vis_npz,
                visual_features=np.random.RandomState(43).randn(n, 8),
                feature_names=np.array([f"v_{i}" for i in range(8)]),
            )

            with closing(sqlite3.connect(targets_db)) as conn:
                conn.execute(
                    "CREATE TABLE targets (video_frame INTEGER PRIMARY KEY, label TEXT, valid_target INTEGER)"
                )
                for i in range(n):
                    lbl = ["H", "A", "D"][i % 3]
                    conn.execute("INSERT INTO targets VALUES (?, ?, 1)", (i, lbl))
                conn.commit()

            report = TF.train_and_evaluate_fusion(
                tracking_npz_path=track_npz,
                visual_npz_path=vis_npz,
                targets_sqlite_path=targets_db,
                output_dir=out_dir,
                train_split_ratio=0.67,
            )

            self.assertTrue((out_dir / "model.joblib").exists())
            self.assertTrue((out_dir / "predictions.sqlite").exists())
            self.assertTrue((out_dir / "predictions.npz").exists())
            self.assertTrue((out_dir / "metrics.json").exists())
            self.assertTrue((out_dir / "manifest.json").exists())

            self.assertEqual(report["features"]["total_count"], 26)
            self.assertEqual(report["splits"]["train_frames"], 20)
            self.assertEqual(report["splits"]["val_frames"], 10)

            # Check predictions database
            with closing(sqlite3.connect(out_dir / "predictions.sqlite")) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM predictions")
                self.assertEqual(cursor.fetchone()[0], 30)

            # Check predictions npz
            with np.load(out_dir / "predictions.npz") as pz:
                self.assertEqual(pz["probabilities"].shape, (30, 3))
                self.assertEqual(len(pz["raw_labels"]), 30)
                self.assertEqual(len(pz["decoded_labels"]), 30)


if __name__ == "__main__":
    unittest.main()
