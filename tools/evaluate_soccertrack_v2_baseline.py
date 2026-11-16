"""Evaluate frozen V2 baseline model on SoccerTrack 117093 tracking features.

Runs inference using the certified frozen V2 Model C (CoverageRouter + Viterbi decoder)
from external_artifacts/v2/inference-bundle-v2-r1 on the 18 tracking features extracted
from SoccerTrack 117093 Babak 1, evaluating against certified external targets.

Outputs:
  - predictions.sqlite: frame-by-frame raw/decoded predictions, probabilities, and GT
  - predictions.npz: compressed arrays of probabilities and labels
  - metrics.json: comprehensive performance evaluation (Spell F1, Macro F1, Transitions)
  - manifest.json: hashes, provenance, and configuration metadata

No official competition data, models, or predictions are accessed or modified.
"""

import argparse
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import possession_metrics as pm
from external_baseline.contracts import FrameKey
from external_baseline.inference_bundle import load_bundle
from external_baseline.training import predict_raw
from external_baseline.temporal import decode_probabilities

MATCH_ID = "117093"
PERIOD = 1
TOTAL_VIDEO_FRAMES = 67625
LABEL_MAP = {"H": 0, "A": 1, "D": 2, "X": 3}
INV_LABEL_MAP = {0: "H", 1: "A", 2: "D", 3: "X"}


def sha256_file(path: Path) -> str:
    """Compute SHA-256 digest of a file in 1 MB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ground_truth_targets(targets_db_path: Path) -> Tuple[List[Optional[str]], List[bool]]:
    """Load video frame labels and valid_target flags from targets.sqlite."""
    with closing(sqlite3.connect(targets_db_path)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT video_frame, label, valid_target FROM targets ORDER BY video_frame"
        )
        rows = cursor.fetchall()

    if len(rows) != TOTAL_VIDEO_FRAMES:
        raise ValueError(f"Expected {TOTAL_VIDEO_FRAMES} targets, got {len(rows)}")

    labels: List[Optional[str]] = [r[1] for r in rows]
    valid: List[bool] = [bool(r[2]) for r in rows]
    return labels, valid


def run_evaluation(
    bundle_dir: Path,
    features_npz_path: Path,
    targets_db_path: Path,
    output_dir: Path,
    bootstrap_n: int = 0,
) -> Dict[str, object]:
    """Execute prediction, decoding, and evaluation on SoccerTrack 117093."""
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    # 1. Load frozen V2 bundle
    model, decoder_params = load_bundle(bundle_dir)

    # 2. Load tracking features
    with np.load(features_npz_path) as npz_data:
        features = npz_data["features"]
        video_frames = npz_data["video_frames"]
        raw_frames = npz_data["raw_frames"]
        raw_times_ms = npz_data["raw_times_ms"]

    n_frames = len(video_frames)
    if n_frames != TOTAL_VIDEO_FRAMES:
        raise ValueError(f"Expected {TOTAL_VIDEO_FRAMES} frames, got {n_frames}")

    # 3. Model inference: raw probabilities and hard predictions
    t_inf_start = time.perf_counter()
    raw_labels, probs = predict_raw(model, features)
    t_inf_elapsed = time.perf_counter() - t_inf_start

    # 4. Temporal decoding: Viterbi offline sequence
    keys = [FrameKey(MATCH_ID, PERIOD, int(v)) for v in video_frames]
    stamps = [int(t) * 1000 for t in raw_times_ms]

    t_dec_start = time.perf_counter()
    decoded_labels = decode_probabilities(keys, stamps, probs, decoder_params)
    t_dec_elapsed = time.perf_counter() - t_dec_start

    # 5. Load certified targets
    gt_labels, valid_mask = load_ground_truth_targets(targets_db_path)

    # Convert to numerical codes for evaluation metrics
    gt_codes = np.array(
        [LABEL_MAP[l] if v and l in LABEL_MAP else pm.X for l, v in zip(gt_labels, valid_mask)],
        dtype=np.int8,
    )
    raw_codes = np.array([LABEL_MAP[l] for l in raw_labels], dtype=np.int8)
    decoded_codes = np.array([LABEL_MAP[l] for l in decoded_labels], dtype=np.int8)

    # 6. Evaluate metrics
    eval_raw = pm.evaluate(gt_codes, raw_codes, bootstrap_n=bootstrap_n)
    eval_dec = pm.evaluate(gt_codes, decoded_codes, bootstrap_n=bootstrap_n)

    # Clean non-serializable objects (e.g. numpy arrays in confusion matrix)
    def serialize_metrics(m: Dict[str, object]) -> Dict[str, object]:
        cleaned = {}
        for k, v in m.items():
            if isinstance(v, np.ndarray):
                cleaned[k] = v.tolist()
            elif isinstance(v, dict):
                cleaned[k] = serialize_metrics(v)
            elif isinstance(v, (np.floating, float)):
                cleaned[k] = None if not math.isfinite(v) else round(float(v), 5)
            elif isinstance(v, (np.integer, int)):
                cleaned[k] = int(v)
            else:
                cleaned[k] = v
        return cleaned

    metrics_raw_clean = serialize_metrics(eval_raw)
    metrics_dec_clean = serialize_metrics(eval_dec)

    metrics_report = {
        "dataset": "SoccerTrack v1.2 match 117093 First Half",
        "total_video_frames": n_frames,
        "graded_frames": int(eval_raw["n_graded"]),
        "masked_frames": int(n_frames - eval_raw["n_graded"]),
        "timing": {
            "inference_seconds": round(t_inf_elapsed, 3),
            "decoding_seconds": round(t_dec_elapsed, 3),
            "total_elapsed_seconds": round(time.perf_counter() - started, 3),
        },
        "raw_predictions": metrics_raw_clean,
        "decoded_predictions": metrics_dec_clean,
        "comparison": {
            "spell_f1_mean_delta": round(float(eval_dec["spell_f1_mean"] - eval_raw["spell_f1_mean"]), 5),
            "spell_f1_headline_delta": round(float(eval_dec["spell_f1_at_headline"] - eval_raw["spell_f1_at_headline"]), 5),
            "macro_f1_delta": round(float(eval_dec["macro_f1"] - eval_raw["macro_f1"]), 5),
            "alive_dead_f1_delta": round(float(eval_dec["alive_dead_f1"] - eval_raw["alive_dead_f1"]), 5),
            "accuracy_delta": round(float(eval_dec["accuracy"] - eval_raw["accuracy"]), 5),
        },
    }

    # 7. Write predictions.sqlite
    sqlite_path = output_dir / "predictions.sqlite"
    with closing(sqlite3.connect(sqlite_path)) as db:
        db.execute("PRAGMA synchronous=OFF")
        db.execute("PRAGMA journal_mode=MEMORY")
        db.execute(
            """CREATE TABLE predictions (
                match_id TEXT NOT NULL,
                period INTEGER NOT NULL,
                video_frame INTEGER PRIMARY KEY,
                raw_frame INTEGER NOT NULL,
                raw_time_ms INTEGER NOT NULL,
                raw_label TEXT NOT NULL,
                decoded_label TEXT NOT NULL,
                p_H REAL NOT NULL,
                p_A REAL NOT NULL,
                p_D REAL NOT NULL,
                gt_label TEXT,
                valid_target INTEGER NOT NULL
            ) WITHOUT ROWID"""
        )

        batch = []
        for i in range(n_frames):
            batch.append((
                MATCH_ID,
                PERIOD,
                int(video_frames[i]),
                int(raw_frames[i]),
                int(raw_times_ms[i]),
                str(raw_labels[i]),
                str(decoded_labels[i]),
                float(probs[i, 0]),
                float(probs[i, 1]),
                float(probs[i, 2]),
                gt_labels[i],
                int(valid_mask[i]),
            ))
            if len(batch) >= 4096:
                db.executemany("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", batch)
                db.commit()
                batch.clear()

        if batch:
            db.executemany("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", batch)
            db.commit()
            batch.clear()

        db.execute("CREATE TABLE evaluation_summary (json TEXT NOT NULL)")
        db.execute(
            "INSERT INTO evaluation_summary VALUES (?)",
            (json.dumps(metrics_report, sort_keys=True),),
        )
        db.commit()

    # 8. Write predictions.npz
    npz_path = output_dir / "predictions.npz"
    np.savez_compressed(
        npz_path,
        video_frames=video_frames,
        raw_frames=raw_frames,
        raw_times_ms=raw_times_ms,
        probabilities=probs,
        raw_labels=np.array(raw_labels),
        decoded_labels=np.array(decoded_labels),
        gt_labels=np.array([l if l is not None else "X" for l in gt_labels]),
        valid_targets=np.array(valid_mask, dtype=bool),
    )

    # 9. Write metrics.json and manifest.json
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_report, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "kind": "soccertrack_117093_v2_baseline_evaluation",
        "dataset": "atomscott/soccertrack-v2",
        "match_id": MATCH_ID,
        "period": PERIOD,
        "total_video_frames": n_frames,
        "graded_frames": int(eval_raw["n_graded"]),
        "masked_frames": int(n_frames - eval_raw["n_graded"]),
        "inputs": {
            "bundle": str(bundle_dir.relative_to(ROOT) if bundle_dir.is_relative_to(ROOT) else bundle_dir),
            "features_npz": str(features_npz_path.relative_to(ROOT) if features_npz_path.is_relative_to(ROOT) else features_npz_path),
            "features_npz_sha256": sha256_file(features_npz_path),
            "targets_sqlite": str(targets_db_path.relative_to(ROOT) if targets_db_path.is_relative_to(ROOT) else targets_db_path),
            "targets_sqlite_sha256": sha256_file(targets_db_path),
        },
        "output_sha256": {
            "predictions.sqlite": sha256_file(sqlite_path),
            "predictions.npz": sha256_file(npz_path),
            "metrics.json": sha256_file(metrics_path),
        },
        "status": "ready",
        "summary": {
            "raw_macro_f1": metrics_raw_clean["macro_f1"],
            "raw_spell_f1_mean": metrics_raw_clean["spell_f1_mean"],
            "raw_spell_f1_at_headline": metrics_raw_clean["spell_f1_at_headline"],
            "decoded_macro_f1": metrics_dec_clean["macro_f1"],
            "decoded_spell_f1_mean": metrics_dec_clean["spell_f1_mean"],
            "decoded_spell_f1_at_headline": metrics_dec_clean["spell_f1_at_headline"],
            "alive_dead_f1": metrics_dec_clean["alive_dead_f1"],
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return metrics_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=ROOT / "external_artifacts/v2/inference-bundle-v2-r1",
        help="Path to frozen V2 inference bundle",
    )
    parser.add_argument(
        "--features-npz",
        type=Path,
        default=ROOT / "external_artifacts/v3/117093-tracking-features-20261116/features.npz",
        help="Path to extracted features.npz",
    )
    parser.add_argument(
        "--targets-sqlite",
        type=Path,
        default=ROOT / "external_artifacts/v3/117093-possession-targets-20261116/targets.sqlite",
        help="Path to certified targets.sqlite",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "external_artifacts/v3/117093-v2-baseline-eval-20261116",
        help="Directory to write evaluation outputs",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Bootstrap resamples for confidence intervals (0=disabled)",
    )
    args = parser.parse_args()

    report = run_evaluation(
        bundle_dir=args.bundle,
        features_npz_path=args.features_npz,
        targets_db_path=args.targets_sqlite,
        output_dir=args.output_dir,
        bootstrap_n=args.bootstrap,
    )
    raw = report["raw_predictions"]
    dec = report["decoded_predictions"]
    print(json.dumps({
        "status": "ready",
        "output_dir": str(args.output_dir),
        "raw_macro_f1": raw["macro_f1"],
        "raw_spell_f1_mean": raw["spell_f1_mean"],
        "decoded_macro_f1": dec["macro_f1"],
        "decoded_spell_f1_mean": dec["spell_f1_mean"],
        "decoded_spell_f1@0.5": dec["spell_f1_at_headline"],
        "alive_dead_f1": dec["alive_dead_f1"],
        "accuracy": dec["accuracy"],
    }, indent=2))


if __name__ == "__main__":
    main()
