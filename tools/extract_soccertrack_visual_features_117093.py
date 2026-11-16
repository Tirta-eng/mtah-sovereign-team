"""Extract 8 certified visual features from SoccerTrack 117093 video.

Uses homography projection and local/global image differential processing to
extract temporal motion, ball vicinity appearance, team jersey color signatures,
and player edge energy from 117093_panorama_1st_half.mp4 aligned with tracking features.

Outputs:
  - visual_features.sqlite: table visual_features and visual_metadata
  - visual_features.npz: compressed (67625, 8) numpy float64 matrix
  - summary.json: feature coverage and descriptive statistics
  - manifest.json: SHA256 integrity digests and projection parameters

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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GEOM_SITE = ROOT / "runtime/v3-geometry-venv/Lib/site-packages"
if GEOM_SITE.exists() and str(GEOM_SITE.resolve()) not in sys.path:
    sys.path.insert(0, str(GEOM_SITE.resolve()))

import cv2

MATCH_ID = "117093"
PERIOD = 1
TOTAL_VIDEO_FRAMES = 67625
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

VISUAL_FEATURE_NAMES: Tuple[str, ...] = (
    "vis_global_motion",
    "vis_pitch_motion",
    "vis_ball_vicinity_motion",
    "vis_ball_crop_std",
    "vis_ball_white_ratio",
    "vis_home_color_score",
    "vis_away_color_score",
    "vis_player_edge_energy",
)

VISUAL_FEATURE_UNITS: Dict[str, str] = {
    "vis_global_motion": "mean_abs_diff",
    "vis_pitch_motion": "mean_abs_diff",
    "vis_ball_vicinity_motion": "mean_abs_diff",
    "vis_ball_crop_std": "std_intensity",
    "vis_ball_white_ratio": "fraction",
    "vis_home_color_score": "channel_excess",
    "vis_away_color_score": "fraction",
    "vis_player_edge_energy": "gradient_magnitude",
}


def sha256_file(path: Path) -> str:
    """Compute SHA-256 digest of a file in 1 MB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_pitch_to_pixel(
    x_m: Optional[float],
    y_m: Optional[float],
    H_mat: np.ndarray,
    img_width: int = 4096,
    img_height: int = 1080,
) -> Tuple[int, int]:
    """Project centred metres coordinate (x_m, y_m) to pixel space (u, v)."""
    if x_m is None or y_m is None or not math.isfinite(x_m) or not math.isfinite(y_m):
        xp, yp = PITCH_LENGTH_M / 2.0, PITCH_WIDTH_M / 2.0
    else:
        xp = x_m + PITCH_LENGTH_M / 2.0
        yp = y_m + PITCH_WIDTH_M / 2.0

    proj = H_mat @ np.array([xp, yp, 1.0])
    w = proj[2]
    if abs(w) < 1e-6:
        u = img_width // 2
        v = img_height // 2
    else:
        u = int(np.clip(round(proj[0] / w), 0, img_width - 1))
        v = int(np.clip(round(proj[1] / w), 0, img_height - 1))
    return u, v


def compute_visual_features_for_frame(
    frame: np.ndarray,
    prev_small_gray: Optional[np.ndarray],
    prev_crop128: Optional[np.ndarray],
    u: int,
    v: int,
    img_width: int = 4096,
    img_height: int = 1080,
) -> Tuple[List[float], np.ndarray, np.ndarray]:
    """Extract 8 visual features from current frame and previous state."""
    # 1. Downsample 4x (1024x270) for global & pitch motion
    small = cv2.resize(frame, (1024, 270), interpolation=cv2.INTER_NEAREST)
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    if prev_small_gray is not None:
        diff_small = cv2.absdiff(gray_small, prev_small_gray)
        global_motion = float(diff_small.mean())
        # Pitch active region in 1024x270: y in [75..190], x in [350..900]
        pitch_motion = float(diff_small[75:190, 350:900].mean())
    else:
        global_motion = 0.0
        pitch_motion = 0.0

    # 2. Crops around projected ball
    # Crop 128x128 for vicinity motion, color, and edge energy
    u0_128, u1_128 = max(0, u - 64), min(img_width, u + 64)
    v0_128, v1_128 = max(0, v - 64), min(img_height, v + 64)
    crop128 = frame[v0_128:v1_128, u0_128:u1_128]

    # Crop 64x64 for ball appearance & white ratio
    u0_64, u1_64 = max(0, u - 32), min(img_width, u + 32)
    v0_64, v1_64 = max(0, v - 32), min(img_height, v + 32)
    crop64 = frame[v0_64:v1_64, u0_64:u1_64]

    # Ball vicinity motion
    if prev_crop128 is not None and crop128.shape == prev_crop128.shape:
        ball_motion = float(cv2.absdiff(crop128, prev_crop128).mean())
    else:
        ball_motion = 0.0

    # Ball crop standard deviation & white ratio
    if crop64.size > 0:
        ball_crop_std = float(crop64.std())
        b64 = crop64[:, :, 0].astype(float)
        g64 = crop64[:, :, 1].astype(float)
        r64 = crop64[:, :, 2].astype(float)
        white_ratio = float(np.mean((b64 > 180) & (g64 > 180) & (r64 > 180)))
    else:
        ball_crop_std = 0.0
        white_ratio = 0.0

    # Team jersey color scores in 128x128 crop
    if crop128.size > 0:
        b128 = crop128[:, :, 0].astype(float)
        g128 = crop128[:, :, 1].astype(float)
        r128 = crop128[:, :, 2].astype(float)

        # Home team (9701, Blue): excess blue over red/green
        home_color = float(np.mean(np.maximum(0.0, b128 - (r128 + g128) / 2.0)))

        # Away team (9834, White): bright neutral pixels
        away_mask = (b128 > 140) & (g128 > 140) & (r128 > 140) & (np.std([r128, g128, b128], axis=0) < 15.0)
        away_color = float(np.mean(away_mask))

        # Edge energy via Sobel gradient
        gray128 = cv2.cvtColor(crop128, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray128, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray128, cv2.CV_32F, 0, 1, ksize=3)
        edge_energy = float(np.mean(np.sqrt(sobelx**2 + sobely**2)))
    else:
        home_color = 0.0
        away_color = 0.0
        edge_energy = 0.0

    features = [
        global_motion,
        pitch_motion,
        ball_motion,
        ball_crop_std,
        white_ratio,
        home_color,
        away_color,
        edge_energy,
    ]
    return features, gray_small, crop128


def extract_and_save_visual_features(
    video_path: Path,
    homography_path: Path,
    features_sqlite_path: Path,
    output_dir: Path,
    max_frames: Optional[int] = None,
    skip_hash_check: bool = False,
    resume: bool = False,
) -> Dict[str, object]:
    """Execute streaming visual feature extraction and write persistent artifacts."""
    output_dir.mkdir(parents=True, exist_ok=resume)
    started = time.perf_counter()

    video_sha256 = sha256_file(video_path) if not skip_hash_check else "skipped"
    homography_sha256 = sha256_file(homography_path) if not skip_hash_check else "skipped"

    H_mat = np.load(homography_path)
    if H_mat.shape != (3, 3):
        raise ValueError(f"Expected 3x3 homography matrix, got {H_mat.shape}")

    # Load ball coordinates from features.sqlite
    with closing(sqlite3.connect(features_sqlite_path)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT video_frame, raw_frame, raw_time_ms, ball_x_m, ball_y_m FROM features ORDER BY video_frame"
        )
        tracking_rows = cursor.fetchall()

    total_frames = len(tracking_rows)
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
        tracking_rows = tracking_rows[:total_frames]

    sqlite_path = output_dir / "visual_features.sqlite"
    npz_path = output_dir / "visual_features.npz"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"

    # Preallocate numpy arrays
    video_frames_arr = np.zeros(total_frames, dtype=np.int32)
    raw_frames_arr = np.zeros(total_frames, dtype=np.int32)
    raw_times_ms_arr = np.zeros(total_frames, dtype=np.int32)
    visual_features_arr = np.zeros((total_frames, len(VISUAL_FEATURE_NAMES)), dtype=np.float64)

    # Initialize statistics
    stats = {
        name: {
            "defined": 0,
            "min": float("inf"),
            "max": float("-inf"),
            "sum": 0.0,
        }
        for name in VISUAL_FEATURE_NAMES
    }

    start_idx = 0
    if resume and sqlite_path.exists():
        with closing(sqlite3.connect(sqlite_path)) as db:
            cols = ", ".join(VISUAL_FEATURE_NAMES)
            rows = db.execute(f"SELECT video_frame, raw_frame, raw_time_ms, {cols} FROM visual_features ORDER BY video_frame").fetchall()
            for r in rows:
                idx = int(r[0])
                if idx < total_frames:
                    video_frames_arr[idx] = idx
                    raw_frames_arr[idx] = int(r[1])
                    raw_times_ms_arr[idx] = int(r[2])
                    feats = [float(v) for v in r[3:]]
                    visual_features_arr[idx, :] = feats
                    for j, name in enumerate(VISUAL_FEATURE_NAMES):
                        val = feats[j]
                        stats[name]["defined"] += 1
                        stats[name]["sum"] += val
                        if val < stats[name]["min"]:
                            stats[name]["min"] = val
                        if val > stats[name]["max"]:
                            stats[name]["max"] = val
            if rows:
                start_idx = int(rows[-1][0]) + 1
                print(f"Resuming from frame {start_idx} (loaded {len(rows)} existing frames)...")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if start_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    prev_small_gray: Optional[np.ndarray] = None
    prev_crop128: Optional[np.ndarray] = None

    with closing(sqlite3.connect(sqlite_path)) as db:
        db.execute("PRAGMA synchronous=OFF")
        db.execute("PRAGMA journal_mode=MEMORY")

        col_defs = ", ".join(f"{name} REAL NOT NULL" for name in VISUAL_FEATURE_NAMES)
        db.execute(
            f"""CREATE TABLE IF NOT EXISTS visual_features (
                match_id TEXT NOT NULL,
                period INTEGER NOT NULL,
                video_frame INTEGER PRIMARY KEY,
                raw_frame INTEGER NOT NULL,
                raw_time_ms INTEGER NOT NULL,
                proj_u INTEGER NOT NULL,
                proj_v INTEGER NOT NULL,
                {col_defs}
            ) WITHOUT ROWID"""
        )

        placeholders = ", ".join("?" for _ in range(7 + len(VISUAL_FEATURE_NAMES)))
        insert_sql = f"INSERT OR REPLACE INTO visual_features VALUES ({placeholders})"

        batch = []
        for i in range(start_idx, total_frames):
            v_idx, raw_f, raw_t, bx_m, by_m = tracking_rows[i]
            ret, frame = cap.read()
            if not ret:
                # Video ended prematurely or frame 67624 beyond video: reuse blank frame
                frame = np.zeros((img_height, img_width, 3), dtype=np.uint8)

            u, v = project_pitch_to_pixel(bx_m, by_m, H_mat, img_width, img_height)

            feats, prev_small_gray, prev_crop128 = compute_visual_features_for_frame(
                frame=frame,
                prev_small_gray=prev_small_gray,
                prev_crop128=prev_crop128,
                u=u,
                v=v,
                img_width=img_width,
                img_height=img_height,
            )

            video_frames_arr[i] = v_idx
            raw_frames_arr[i] = raw_f
            raw_times_ms_arr[i] = raw_t
            visual_features_arr[i, :] = feats

            for j, name in enumerate(VISUAL_FEATURE_NAMES):
                val = feats[j]
                stats[name]["defined"] += 1
                stats[name]["sum"] += val
                if val < stats[name]["min"]:
                    stats[name]["min"] = val
                if val > stats[name]["max"]:
                    stats[name]["max"] = val

            batch.append((MATCH_ID, PERIOD, v_idx, raw_f, raw_t, u, v, *feats))

            if len(batch) >= 2048:
                db.executemany(insert_sql, batch)
                db.commit()
                batch.clear()

        if batch:
            db.executemany(insert_sql, batch)
            db.commit()
            batch.clear()

        # Metadata table
        db.execute("CREATE TABLE IF NOT EXISTS visual_metadata (json TEXT NOT NULL)")
        meta_json = json.dumps(
            {
                "version": 1,
                "match_id": MATCH_ID,
                "period": PERIOD,
                "total_video_frames": total_frames,
                "fps": 25,
                "feature_names": VISUAL_FEATURE_NAMES,
                "units": VISUAL_FEATURE_UNITS,
                "resolution": f"{img_width}x{img_height}",
            },
            sort_keys=True,
        )
        db.execute("DELETE FROM visual_metadata")
        db.execute("INSERT INTO visual_metadata VALUES (?)", (meta_json,))
        db.commit()

    cap.release()
    elapsed_s = round(time.perf_counter() - started, 3)

    # Save NPZ
    np.savez_compressed(
        npz_path,
        video_frames=video_frames_arr,
        raw_frames=raw_frames_arr,
        raw_times_ms=raw_times_ms_arr,
        visual_features=visual_features_arr,
        feature_names=np.array(VISUAL_FEATURE_NAMES),
    )

    # Finalize statistics
    summary_features = {}
    for name, s in stats.items():
        count_def = s["defined"]
        summary_features[name] = {
            "defined": count_def,
            "coverage": round(count_def / total_frames, 4),
            "min": round(s["min"], 4) if count_def > 0 else None,
            "max": round(s["max"], 4) if count_def > 0 else None,
            "mean": round(s["sum"] / count_def, 4) if count_def > 0 else None,
            "unit": VISUAL_FEATURE_UNITS.get(name, ""),
        }

    summary = {
        "match_id": MATCH_ID,
        "period": PERIOD,
        "total_video_frames": total_frames,
        "elapsed_seconds": elapsed_s,
        "fps_rate": round(total_frames / elapsed_s, 1) if elapsed_s > 0 else 0.0,
        "feature_count": len(VISUAL_FEATURE_NAMES),
        "features": summary_features,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "kind": "soccertrack_117093_visual_features",
        "dataset": "atomscott/soccertrack-v2",
        "match_id": MATCH_ID,
        "period": PERIOD,
        "total_video_frames": total_frames,
        "fps": 25,
        "resolution": f"{img_width}x{img_height}",
        "feature_names": list(VISUAL_FEATURE_NAMES),
        "units": VISUAL_FEATURE_UNITS,
        "inputs": {
            "video_path": str(video_path.relative_to(ROOT) if video_path.is_relative_to(ROOT) else video_path),
            "video_sha256": video_sha256,
            "homography_path": str(homography_path.relative_to(ROOT) if homography_path.is_relative_to(ROOT) else homography_path),
            "homography_sha256": homography_sha256,
            "features_sqlite": str(features_sqlite_path.relative_to(ROOT) if features_sqlite_path.is_relative_to(ROOT) else features_sqlite_path),
        },
        "outputs_sha256": {
            "visual_features.sqlite": sha256_file(sqlite_path),
            "visual_features.npz": sha256_file(npz_path),
            "summary.json": sha256_file(summary_path),
        },
        "status": "ready",
        "provenance": {
            "method": "OpenCV temporal differential motion + homography ball crop analysis",
            "protocol": "docs/protokol_fitur_visual_v3_117093.md",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=ROOT / "Soccertrack V2/videos/117093/117093_panorama_1st_half.mp4",
        help="Path to 117093_panorama_1st_half.mp4",
    )
    parser.add_argument(
        "--homography",
        type=Path,
        default=ROOT / "Soccertrack V2/raw/117093/117093_homography.npy",
        help="Path to 117093_homography.npy",
    )
    parser.add_argument(
        "--features-sqlite",
        type=Path,
        default=ROOT / "external_artifacts/v3/117093-tracking-features-20261116/features.sqlite",
        help="Path to tracking features.sqlite",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "external_artifacts/v3/117093-visual-features-20261116",
        help="Directory to write visual feature outputs",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional max frames to process for testing",
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Skip 3.3 GB video SHA256 check for fast startup",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume extraction from existing visual_features.sqlite progress",
    )
    args = parser.parse_args()

    summary = extract_and_save_visual_features(
        video_path=args.video,
        homography_path=args.homography,
        features_sqlite_path=args.features_sqlite,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        skip_hash_check=args.skip_hash_check,
        resume=args.resume,
    )
    print(json.dumps({
        "status": "ready",
        "total_video_frames": summary["total_video_frames"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "fps_rate": summary["fps_rate"],
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
