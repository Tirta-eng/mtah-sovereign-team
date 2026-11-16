"""End-to-end multimodal inference pipeline for Multimodal Bundle V3."""

from collections import Counter
from contextlib import closing
import csv
import json
from pathlib import Path
import sqlite3
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np

# Robust import of cv2
try:
    import cv2
except ImportError:
    for parent in Path(__file__).resolve().parents:
        geom = parent / "runtime/v3-geometry-venv/Lib/site-packages"
        if geom.exists() and str(geom) not in sys.path:
            sys.path.insert(0, str(geom))
            break
    import cv2

from .atd_parser import frames, metadata, organizer_period_scope
from .bundle_verifier import sha256_file, verify_bundle
from .contracts import ALL_FEATURES_26, FrameKey, LABELS, TRACKING_FEATURES, VISUAL_FEATURES
from .temporal_decoder import decode_probabilities
from .tracking_features import TrackingFeatureExtractor
from .visual_features import VisualFeatureExtractor


def load_homography_file(path: Path) -> Tuple[Union[np.ndarray, Dict[int, Optional[np.ndarray]]], bool]:
    """Load homography from either .npy (static 3x3) or .csv (per-frame 3x3)."""
    p = Path(path)
    if p.suffix.lower() == ".npy":
        mat = np.load(p)
        if mat.shape != (3, 3):
            raise ValueError(f"Expected 3x3 homography matrix in .npy, got {mat.shape}")
        return mat, False
    elif p.suffix.lower() == ".csv":
        matrices: Dict[int, Optional[np.ndarray]] = {}
        with p.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            h_indices = [header.index(f"h{r}{c}") for r in range(3) for c in range(3)]
            valid_idx = header.index("valid") if "valid" in header else None
            frame_idx = header.index("frame") if "frame" in header else 0
            for row in reader:
                if not row:
                    continue
                f_num = int(float(row[frame_idx]))
                if valid_idx is not None and int(float(row[valid_idx])) == 0:
                    matrices[f_num] = None
                else:
                    h_vals = [float(row[i]) for i in h_indices]
                    matrices[f_num] = np.array(h_vals, dtype=np.float64).reshape((3, 3))
        return matrices, True
    else:
        raise ValueError(f"Unsupported homography format: {p.suffix}")


def run_multimodal_inference(
    bundle_dir: Path,
    metadata_path: Path,
    tracking_path: Path,
    video_path: Path,
    homography_path: Path,
    output_dir: Path,
    expected_frames: int = 199319,
    period_source: str = "organizer-public",
    game: Optional[str] = None,
) -> Dict[str, object]:
    """Execute complete multimodal inference pipeline and produce official submission.csv."""
    t_start = time.perf_counter()
    bundle_dir = Path(bundle_dir).resolve(strict=True)
    metadata_path = Path(metadata_path).resolve(strict=True)
    tracking_path = Path(tracking_path).resolve(strict=True)
    video_path = Path(video_path).resolve(strict=True)
    homography_path = Path(homography_path).resolve(strict=True)
    output_dir = Path(output_dir).resolve()

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify bundle
    bundle_manifest = verify_bundle(bundle_dir, check_runtime=True)

    # 2. Compute input hashes
    input_hashes = {
        "metadata": sha256_file(metadata_path),
        "tracking": sha256_file(tracking_path),
        "video": sha256_file(video_path),
        "homography": sha256_file(homography_path),
        "bundle_manifest": sha256_file(bundle_dir / "bundle_manifest.json"),
    }

    # 3. Resolve period scope
    if period_source == "organizer-public":
        public_meta_file = bundle_dir / "src/organizer_game_metadata.py"
        period_scope = organizer_period_scope(public_meta_file, game, expected_frames)
    elif period_source == "provider":
        period_scope = None
    else:
        raise ValueError(f"Unsupported period_source: {period_source}")

    meta = metadata(metadata_path, period_scope)

    # 4. Load trained model & decoder parameters
    model_bundle = joblib.load(bundle_dir / "model/model.joblib")
    clf = model_bundle["model"]
    decoder_params = json.loads((bundle_dir / "model/decoder_parameters.json").read_text(encoding="utf-8"))

    # Load homography (supports both .npy and official .csv)
    homography_source, is_normalised = load_homography_file(homography_path)

    # 5. Initialize extractors
    track_extractor = TrackingFeatureExtractor(meta["length_m"], meta["width_m"])
    vis_extractor = VisualFeatureExtractor(
        homography_source=homography_source,
        is_normalised_coords=is_normalised,
        pitch_length=meta["length_m"],
        pitch_width=meta["width_m"],
    )

    # 6. Open video stream
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    # Preallocate output arrays
    labels = np.full(expected_frames, "X", dtype="U1")
    raw_labels = np.full(expected_frames, "X", dtype="U1")

    # Set up SQLite storage
    sqlite_features = output_dir / "features.sqlite"
    sqlite_preds = output_dir / "predictions.sqlite"

    with closing(sqlite3.connect(sqlite_features)) as feat_db, closing(sqlite3.connect(sqlite_preds)) as pred_db:
        # Create tables
        cols_sql = ", ".join(f"{name} REAL" for name in ALL_FEATURES_26)
        feat_db.execute(f"CREATE TABLE features (frame INTEGER PRIMARY KEY, period INTEGER, time_us INTEGER, {cols_sql})")
        pred_db.execute(
            "CREATE TABLE predictions (frame INTEGER PRIMARY KEY, period INTEGER, raw_label TEXT, label TEXT, p_H REAL, p_A REAL, p_D REAL)"
        )

        batch_feats = []
        video_frame_idx = 0

        # Stream tracking frames
        for frame_idx, period, stamp_us, entities in frames(tracking_path, meta, expected_frames):
            if period is None:
                continue

            # Extract tracking features
            key = FrameKey("match", period, frame_idx)
            t_feats = track_extractor.extract(key, stamp_us, entities)

            # Sync video reading
            if video_frame_idx != frame_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                video_frame_idx = frame_idx

            ret, frame_bgr = cap.read()
            video_frame_idx += 1
            if not ret or frame_bgr is None:
                # If frame read fails, fallback to zero visual features
                v_feats = [0.0] * len(VISUAL_FEATURES)
            else:
                ball_entity = next((e for e in entities if e.team == "BALL"), None)
                bx = ball_entity.x_m if ball_entity is not None else None
                by = ball_entity.y_m if ball_entity is not None else None
                v_feats = vis_extractor.extract_frame(frame_bgr, bx, by, frame_idx=frame_idx)

            row_values = [t_feats[name] for name in TRACKING_FEATURES] + v_feats
            batch_feats.append((frame_idx, period, stamp_us, *row_values))

            if len(batch_feats) >= 4096:
                feat_db.executemany("INSERT INTO features VALUES (" + ",".join("?" for _ in batch_feats[0]) + ")", batch_feats)
                feat_db.commit()
                batch_feats.clear()

        if batch_feats:
            feat_db.executemany("INSERT INTO features VALUES (" + ",".join("?" for _ in batch_feats[0]) + ")", batch_feats)
            feat_db.commit()

        cap.release()

        # Run inference per period
        cols_query = ", ".join(ALL_FEATURES_26)
        for period, start, end in meta["periods"]:
            period_rows = feat_db.execute(
                f"SELECT frame, time_us, {cols_query} FROM features WHERE period = ? ORDER BY frame",
                (period,),
            ).fetchall()

            if not period_rows:
                continue

            keys = [FrameKey("match", period, r[0]) for r in period_rows]
            stamps = [r[1] for r in period_rows]
            X_period = np.array([r[2:] for r in period_rows], dtype=np.float64)

            # Model prediction
            probs_native = clf.predict_proba(X_period)
            class_order = list(clf.classes_)
            # Map into ('H', 'A', 'D') order
            probs = np.zeros((len(X_period), 3), dtype=np.float64)
            for idx_lbl, lbl in enumerate(LABELS):
                if lbl in class_order:
                    probs[:, idx_lbl] = probs_native[:, class_order.index(lbl)]

            raw_idx = np.argmax(probs, axis=1)
            raw_period_labels = [LABELS[i] for i in raw_idx]

            # Viterbi decoding
            decoded_period_labels = decode_probabilities(keys, stamps, probs, decoder_params)

            # Populate submission arrays and DB
            db_preds = []
            for k, raw_l, dec_l, p in zip(keys, raw_period_labels, decoded_period_labels, probs):
                labels[k.source_frame] = dec_l
                raw_labels[k.source_frame] = raw_l
                db_preds.append((k.source_frame, period, str(raw_l), str(dec_l), float(p[0]), float(p[1]), float(p[2])))

            pred_db.executemany("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?)", db_preds)
            pred_db.commit()

    # 7. Write submission.csv and submission_raw.csv
    sub_path = output_dir / "submission.csv"
    with sub_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["frame", "label"])
        writer.writerows((i, str(val)) for i, val in enumerate(labels))

    raw_sub_path = output_dir / "submission_raw.csv"
    with raw_sub_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["frame", "label"])
        writer.writerows((i, str(val)) for i, val in enumerate(raw_labels))

    # Contract assertions
    if len(labels) != expected_frames:
        raise ValueError(f"Submission row count {len(labels)} != expected {expected_frames}")
    if set(labels) - set("HADX"):
        raise ValueError(f"Invalid labels in submission: {set(labels) - set('HADX')}")

    elapsed_s = time.perf_counter() - t_start

    # 8. Write report and manifest
    report = {
        "status": "ready",
        "bundle": "inference-bundle-v3-r1",
        "expected_frames": expected_frames,
        "elapsed_seconds": round(elapsed_s, 3),
        "fps": round(expected_frames / elapsed_s, 2) if elapsed_s > 0 else None,
        "label_counts": dict(Counter(labels)),
        "raw_label_counts": dict(Counter(raw_labels)),
        "periods": meta["periods"],
        "pitch_length_m": meta["length_m"],
        "pitch_width_m": meta["width_m"],
        "period_source": period_source,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    output_hashes = {p.name: sha256_file(p) for p in output_dir.iterdir() if p.is_file()}
    manifest = {
        "status": "ready",
        "pipeline": "v3_multimodal_inference",
        "inputs_sha256": input_hashes,
        "outputs_sha256": output_hashes,
        "submission_file": str(sub_path.name),
        "submission_rows": expected_frames,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return report
