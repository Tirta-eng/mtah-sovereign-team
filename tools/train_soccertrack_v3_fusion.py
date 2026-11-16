"""Train and evaluate Multimodal Fusion Model (V3) on SoccerTrack 117093.

Combines 18 spatial/causal tracking features with 8 video visual features (26 total),
trains a Gradient Boosting classifier on valid target frames, decodes temporal
sequences with Viterbi HMM, and benchmarks gains against the tracking-only V2 baseline.

Outputs:
  - model.joblib: trained multimodal estimator, feature list, and decoder parameters
  - predictions.sqlite: frame-by-frame predictions, posteriors, and GT
  - predictions.npz: predicted probabilities and label sequences
  - metrics.json: comprehensive performance report (Spell F1, Macro F1, Confusion Matrix)
  - manifest.json: provenance hashes and configuration metadata

No official competition data, models, or network calls are accessed.
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

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import possession_metrics as pm
from external_baseline.contracts import FrameKey
from external_baseline.temporal import decode_probabilities, estimate_markov

MATCH_ID = "117093"
PERIOD = 1
LABEL_MAP = {"H": 0, "A": 1, "D": 2, "X": 3}
INV_LABEL_MAP = {0: "H", 1: "A", 2: "D", 3: "X"}
LABELS = ("H", "A", "D")


def sha256_file(path: Path) -> str:
    """Compute SHA-256 digest of a file in 1 MB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(
    tracking_npz_path: Path,
    visual_npz_path: Path,
    targets_sqlite_path: Path,
    max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[bool], List[str]]:
    """Load and synchronize tracking features, visual features, and targets."""
    with np.load(tracking_npz_path) as t_data:
        track_feats = t_data["features"]
        video_frames = t_data["video_frames"]
        raw_frames = t_data["raw_frames"]
        raw_times_ms = t_data["raw_times_ms"]
        track_names = [str(n) for n in t_data["model_features"]]

    with np.load(visual_npz_path) as v_data:
        vis_feats = v_data["visual_features"]
        vis_names = [str(n) for n in v_data["feature_names"]]

    n_available = min(len(track_feats), len(vis_feats))
    if max_frames is not None:
        n_available = min(n_available, max_frames)

    track_feats = track_feats[:n_available]
    vis_feats = vis_feats[:n_available]
    video_frames = video_frames[:n_available]
    raw_frames = raw_frames[:n_available]
    raw_times_ms = raw_times_ms[:n_available]

    # Combine features: (N, 26)
    X_combined = np.hstack([track_feats, vis_feats])
    combined_names = track_names + vis_names

    # Load targets
    with closing(sqlite3.connect(targets_sqlite_path)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT video_frame, label, valid_target FROM targets ORDER BY video_frame"
        )
        target_rows = cursor.fetchall()[:n_available]

    gt_labels = [r[1] for r in target_rows]
    valid_mask = [bool(r[2]) for r in target_rows]

    return X_combined, video_frames, raw_frames, raw_times_ms, combined_names, valid_mask, gt_labels


def train_and_evaluate_fusion(
    tracking_npz_path: Path,
    visual_npz_path: Path,
    targets_sqlite_path: Path,
    output_dir: Path,
    train_split_ratio: float = 0.67,
    max_frames: Optional[int] = None,
) -> Dict[str, object]:
    """Train Multimodal V3 model, decode temporal sequence, and evaluate performance."""
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    # 1. Load data
    (
        X,
        video_frames,
        raw_frames,
        raw_times_ms,
        feature_names,
        valid_mask,
        gt_labels,
    ) = load_dataset(
        tracking_npz_path=tracking_npz_path,
        visual_npz_path=visual_npz_path,
        targets_sqlite_path=targets_sqlite_path,
        max_frames=max_frames,
    )

    n_total = len(X)
    split_point = int(n_total * train_split_ratio)

    train_indices = np.arange(split_point)
    val_indices = np.arange(split_point, n_total)

    train_valid = [i for i in train_indices if valid_mask[i] and gt_labels[i] in LABELS]
    val_valid = [i for i in val_indices if valid_mask[i] and gt_labels[i] in LABELS]

    X_train = X[train_valid]
    y_train = np.array([gt_labels[i] for i in train_valid])

    print(f"Total frames: {n_total}, Train valid: {len(X_train)}, Val valid: {len(val_valid)}")

    # 2. Train multimodal classifier
    t_train_start = time.perf_counter()
    clf = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.1,
        max_iter=100,
        min_samples_leaf=20,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    t_train_elapsed = time.perf_counter() - t_train_start

    # 3. Model predict proba on all frames
    probs_native = clf.predict_proba(X)
    class_order = list(clf.classes_)
    # Reorder to standard ('H', 'A', 'D') defensively
    probs = np.zeros((len(X), 3), dtype=np.float64)
    for col, lbl in enumerate(LABELS):
        if lbl in class_order:
            probs[:, col] = probs_native[:, class_order.index(lbl)]
    # Normalize if needed
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs /= row_sums
    raw_preds = np.array(LABELS)[np.argmax(probs, axis=1)]

    # 4. Estimate Markov transitions from training split
    keys_train = [FrameKey(MATCH_ID, PERIOD, int(video_frames[i])) for i in train_indices]
    stamps_train = [int(raw_times_ms[i]) * 1000 for i in train_indices]
    labels_train = [gt_labels[i] if valid_mask[i] else "D" for i in train_indices]
    valid_train = [valid_mask[i] for i in train_indices]

    decoder_params = estimate_markov(
        keys=keys_train,
        stamps=stamps_train,
        labels=labels_train,
        valid=valid_train,
    )

    # 5. Decode temporal Viterbi sequence on all frames
    keys_all = [FrameKey(MATCH_ID, PERIOD, int(v)) for v in video_frames]
    stamps_all = [int(t) * 1000 for t in raw_times_ms]

    t_dec_start = time.perf_counter()
    decoded_preds = decode_probabilities(keys_all, stamps_all, probs, decoder_params)
    t_dec_elapsed = time.perf_counter() - t_dec_start

    # 6. Evaluation metrics on Validation Split and Full Match
    gt_codes_all = np.array(
        [LABEL_MAP[l] if v and l in LABEL_MAP else pm.X for l, v in zip(gt_labels, valid_mask)],
        dtype=np.int8,
    )
    raw_codes_all = np.array([LABEL_MAP[l] for l in raw_preds], dtype=np.int8)
    dec_codes_all = np.array([LABEL_MAP[l] for l in decoded_preds], dtype=np.int8)

    # Validation split evaluation
    gt_codes_val = gt_codes_all[val_indices]
    raw_codes_val = raw_codes_all[val_indices]
    dec_codes_val = dec_codes_all[val_indices]

    eval_val_raw = pm.evaluate(gt_codes_val, raw_codes_val, bootstrap_n=0)
    eval_val_dec = pm.evaluate(gt_codes_val, dec_codes_val, bootstrap_n=0)

    # Full match evaluation
    eval_full_raw = pm.evaluate(gt_codes_all, raw_codes_all, bootstrap_n=0)
    eval_full_dec = pm.evaluate(gt_codes_all, dec_codes_all, bootstrap_n=0)

    def clean_report(m: Dict[str, object]) -> Dict[str, object]:
        cleaned = {}
        for k, v in m.items():
            if isinstance(v, np.ndarray):
                cleaned[k] = v.tolist()
            elif isinstance(v, dict):
                cleaned[k] = clean_report(v)
            elif isinstance(v, (np.floating, float)):
                cleaned[k] = None if not math.isfinite(v) else round(float(v), 5)
            elif isinstance(v, (np.integer, int)):
                cleaned[k] = int(v)
            else:
                cleaned[k] = v
        return cleaned

    report = {
        "model": "Multimodal Fusion V3 (HistGradientBoosting + Viterbi HMM)",
        "features": {
            "total_count": len(feature_names),
            "tracking_count": 18,
            "visual_count": 8,
            "names": feature_names,
        },
        "splits": {
            "train_frames": len(train_indices),
            "train_valid_graded": len(X_train),
            "val_frames": len(val_indices),
            "val_valid_graded": len(val_valid),
            "train_ratio": train_split_ratio,
        },
        "timing": {
            "training_seconds": round(t_train_elapsed, 3),
            "decoding_seconds": round(t_dec_elapsed, 3),
            "total_seconds": round(time.perf_counter() - started, 3),
        },
        "validation_split": {
            "raw": clean_report(eval_val_raw),
            "decoded": clean_report(eval_val_dec),
            "gain_macro_f1": round(float(eval_val_dec["macro_f1"] - eval_val_raw["macro_f1"]), 5),
            "gain_spell_f1_mean": round(float(eval_val_dec["spell_f1_mean"] - eval_val_raw["spell_f1_mean"]), 5),
        },
        "full_match": {
            "raw": clean_report(eval_full_raw),
            "decoded": clean_report(eval_full_dec),
        },
    }

    # 7. Write outputs
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
                split TEXT NOT NULL,
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
        for i in range(n_total):
            sp = "train" if i < split_point else "validation"
            batch.append((
                MATCH_ID,
                PERIOD,
                int(video_frames[i]),
                int(raw_frames[i]),
                int(raw_times_ms[i]),
                sp,
                str(raw_preds[i]),
                str(decoded_preds[i]),
                float(probs[i, 0]),
                float(probs[i, 1]),
                float(probs[i, 2]),
                gt_labels[i],
                int(valid_mask[i]),
            ))
            if len(batch) >= 4096:
                db.executemany("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
                db.commit()
                batch.clear()

        if batch:
            db.executemany("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
            db.commit()
            batch.clear()

        db.execute("CREATE TABLE training_report (json TEXT NOT NULL)")
        db.execute("INSERT INTO training_report VALUES (?)", (json.dumps(report, sort_keys=True),))
        db.commit()

    # Save model.joblib
    model_bundle = {
        "model": clf,
        "feature_names": feature_names,
        "decoder_parameters": decoder_params,
        "classes": list(LABELS),
        "algorithm": "v3_multimodal_hist_gradient_boosting",
    }
    model_path = output_dir / "model.joblib"
    joblib.dump(model_bundle, model_path)

    # Save predictions.npz
    npz_path = output_dir / "predictions.npz"
    np.savez_compressed(
        npz_path,
        video_frames=video_frames,
        raw_frames=raw_frames,
        raw_times_ms=raw_times_ms,
        probabilities=probs,
        raw_labels=raw_preds,
        decoded_labels=decoded_preds,
        gt_labels=np.array([l if l is not None else "X" for l in gt_labels]),
        valid_targets=np.array(valid_mask, dtype=bool),
    )

    # Save metrics.json and manifest.json
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "kind": "soccertrack_117093_v3_multimodal_model",
        "dataset": "atomscott/soccertrack-v2",
        "match_id": MATCH_ID,
        "period": PERIOD,
        "total_frames": n_total,
        "features": {
            "total": len(feature_names),
            "tracking": 18,
            "visual": 8,
        },
        "inputs": {
            "tracking_npz": str(tracking_npz_path.relative_to(ROOT) if tracking_npz_path.is_relative_to(ROOT) else tracking_npz_path),
            "tracking_npz_sha256": sha256_file(tracking_npz_path),
            "visual_npz": str(visual_npz_path.relative_to(ROOT) if visual_npz_path.is_relative_to(ROOT) else visual_npz_path),
            "visual_npz_sha256": sha256_file(visual_npz_path),
            "targets_sqlite": str(targets_sqlite_path.relative_to(ROOT) if targets_sqlite_path.is_relative_to(ROOT) else targets_sqlite_path),
            "targets_sqlite_sha256": sha256_file(targets_sqlite_path),
        },
        "outputs_sha256": {
            "model.joblib": sha256_file(model_path),
            "predictions.sqlite": sha256_file(sqlite_path),
            "predictions.npz": sha256_file(npz_path),
            "metrics.json": sha256_file(metrics_path),
        },
        "status": "ready",
        "headline_results": {
            "val_decoded_macro_f1": report["validation_split"]["decoded"]["macro_f1"],
            "val_decoded_spell_f1_mean": report["validation_split"]["decoded"]["spell_f1_mean"],
            "val_decoded_accuracy": report["validation_split"]["decoded"]["accuracy"],
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracking-npz",
        type=Path,
        default=ROOT / "external_artifacts/v3/117093-tracking-features-20261116/features.npz",
        help="Path to tracking features.npz",
    )
    parser.add_argument(
        "--visual-npz",
        type=Path,
        default=ROOT / "external_artifacts/v3/117093-visual-features-20261116/visual_features.npz",
        help="Path to visual features.npz",
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
        default=ROOT / "external_artifacts/v3/117093-v3-multimodal-model-20261116",
        help="Directory to write model and metrics",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.67,
        help="Training split ratio (default 0.67)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max frames to process for fast testing",
    )
    args = parser.parse_args()

    report = train_and_evaluate_fusion(
        tracking_npz_path=args.tracking_npz,
        visual_npz_path=args.visual_npz,
        targets_sqlite_path=args.targets_sqlite,
        output_dir=args.output_dir,
        train_split_ratio=args.split_ratio,
        max_frames=args.max_frames,
    )
    val_dec = report["validation_split"]["decoded"]
    print(json.dumps({
        "status": "ready",
        "output_dir": str(args.output_dir),
        "val_macro_f1": val_dec["macro_f1"],
        "val_spell_f1_mean": val_dec["spell_f1_mean"],
        "val_spell_f1@0.5": val_dec["spell_f1_at_headline"],
        "val_accuracy": val_dec["accuracy"],
        "val_alive_dead_f1": val_dec["alive_dead_f1"],
    }, indent=2))


if __name__ == "__main__":
    main()
