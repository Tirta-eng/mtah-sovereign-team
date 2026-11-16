"""Extract certified tracking features for SoccerTrack 117093.

Extracts planar spatial and causal context features (20 features total, 18 model
features dropping signed position) from SoccerTrack v1.2 match 117093 raw tracker
box XML data aligned with video frames [0..67624] and project possession targets.

Outputs:
  - features.sqlite: SQLite database containing table 'features' and 'feature_metadata'
  - features.npz: Compressed numpy archive containing (67625, 18) model features
  - summary.json: Summary statistics and feature coverage
  - manifest.json: Integrity hashes, contracts, and provenance metadata

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
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_baseline.contracts import FrameKey, STEP_US
from external_baseline.features import (
    ContextFeatureExtractor,
    EntityPosition,
    FEATURE_NAMES as BASE_FEATURE_NAMES,
    CONTEXT_NAMES,
)
MATCH_ID = "117093"
PERIOD = 1
TOTAL_VIDEO_FRAMES = 67625  # 0-indexed: 0..67624
CLOCK_OFFSET_MS = 1040
RAW_FRAME_OFFSET = 277  # 251 + 26
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# 20 full features (16 base + 4 causal context)
FEATURE_NAMES: Tuple[str, ...] = BASE_FEATURE_NAMES + CONTEXT_NAMES

# 18 model features (dropping signed coordinates ball_x_m and ball_y_m to prevent orientation bias)
MODEL_FEATURES: Tuple[str, ...] = tuple(
    name for name in FEATURE_NAMES if name not in ("ball_x_m", "ball_y_m")
)

FEATURE_UNITS: Dict[str, str] = {
    "ball_record_present": "binary",
    "ball_x_observed": "binary",
    "ball_y_observed": "binary",
    "ball_xy_observed": "binary",
    "ball_x_m": "m",
    "ball_y_m": "m",
    "ball_speed_mps": "m/s",
    "home_records": "count",
    "home_xy_observed": "count",
    "away_records": "count",
    "away_xy_observed": "count",
    "nearest_home_m": "m",
    "nearest_away_m": "m",
    "ball_longitudinal_margin_m": "m",
    "ball_lateral_margin_m": "m",
    "ball_inside_pitch": "binary",
    "ball_xy_fraction_1s": "fraction",
    "ball_xy_age_s": "s",
    "nearest_home_median_1s": "m",
    "nearest_away_median_1s": "m",
}


def sha256_file(path: Path) -> str:
    """Compute SHA-256 digest of a file in 1 MB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_metadata(metadata_path: Path) -> Dict[str, object]:
    """Parse SoccerTrack metadata XML for pitch size and player team mappings."""
    tree = ET.parse(metadata_path)
    root = tree.getroot()

    pitch_elem = root.find("pitch")
    if pitch_elem is not None:
        length_m = float(pitch_elem.attrib.get("width", PITCH_LENGTH_M))
        width_m = float(pitch_elem.attrib.get("height", PITCH_WIDTH_M))
    else:
        length_m = PITCH_LENGTH_M
        width_m = PITCH_WIDTH_M

    teams = {}
    for team_elem in root.findall(".//teams/team"):
        tid = team_elem.attrib["id"]
        side = team_elem.attrib.get("side", "").lower()
        if side == "home" or tid == "9701":
            teams[tid] = "H"
        elif side == "away" or tid == "9834":
            teams[tid] = "A"

    player_to_team = {}
    for p_elem in root.findall(".//players/player"):
        pid = p_elem.attrib["id"]
        tid = p_elem.attrib.get("teamId")
        if tid in teams:
            player_to_team[pid] = teams[tid]
        elif tid == "9701":
            player_to_team[pid] = "H"
        elif tid == "9834":
            player_to_team[pid] = "A"

    return {
        "length_m": length_m,
        "width_m": width_m,
        "teams": teams,
        "player_to_team": player_to_team,
    }


def normalized_to_centred_metres(
    loc_str: Optional[str], length_m: float, width_m: float
) -> Tuple[Optional[float], Optional[float]]:
    """Convert normalized [x, y] in [0, 1] to centered metres.

    Centred coordinate system:
      x_m = (x_norm - 0.5) * length_m
      y_m = (y_norm - 0.5) * width_m
    """
    if not loc_str:
        return None, None
    s = loc_str.strip("[] \t\r\n")
    if not s or s.lower() == "na":
        return None, None
    parts = s.split(",")
    if len(parts) != 2:
        return None, None
    try:
        x_raw = float(parts[0].strip())
        y_raw = float(parts[1].strip())
    except ValueError:
        return None, None

    x_m = (x_raw - 0.5) * length_m if math.isfinite(x_raw) else None
    y_m = (y_raw - 0.5) * width_m if math.isfinite(y_raw) else None
    return x_m, y_m


def extract_features_stream(
    data_xml_path: Path,
    metadata: Dict[str, object],
    total_video_frames: int = TOTAL_VIDEO_FRAMES,
    clock_offset_ms: int = CLOCK_OFFSET_MS,
    raw_frame_offset: int = RAW_FRAME_OFFSET,
):
    """Generator yielding (video_frame, raw_frame, raw_time_ms, features_dict).

    Iterates through XML sequentially, mapping raw frames to video frame indices,
    filling any missing frames or outside-coverage frames with empty entity lists.
    """
    length_m = float(metadata["length_m"])
    width_m = float(metadata["width_m"])
    player_to_team: Dict[str, str] = metadata["player_to_team"]

    extractor = ContextFeatureExtractor(length_m, width_m)
    current_v_idx = 0

    parser = ET.iterparse(data_xml_path, events=("start", "end"))
    _, root = next(parser)

    for event, elem in parser:
        if event == "end" and elem.tag == "frame":
            if elem.attrib.get("eventPeriod") == "FIRST_HALF":
                raw_f = int(elem.attrib["frameNumber"])
                target_v_idx = raw_f - raw_frame_offset

                # Before video frame 0, skip
                if target_v_idx < 0:
                    root.clear()
                    continue

                # Stop if beyond video scope
                if target_v_idx >= total_video_frames:
                    root.clear()
                    break

                # If there are any gaps between current_v_idx and target_v_idx, fill with empty frames
                while current_v_idx < target_v_idx:
                    raw_time_ms = current_v_idx * 40 + clock_offset_ms
                    stamp_us = raw_time_ms * 1000
                    expected_raw_f = raw_frame_offset + current_v_idx
                    key = FrameKey(MATCH_ID, PERIOD, current_v_idx)
                    feats = extractor.extract(key, stamp_us, entities=[])
                    yield current_v_idx, expected_raw_f, raw_time_ms, feats
                    current_v_idx += 1

                # Parse entities for current frame
                entities: List[EntityPosition] = []
                for child in elem:
                    if child.tag == "player":
                        pid = child.attrib["playerId"]
                        team = player_to_team.get(pid)
                        if team is None:
                            continue
                    elif child.tag == "ball":
                        pid = child.attrib.get("playerId", "ball")
                        team = "BALL"
                    else:
                        continue

                    loc_str = child.attrib.get("loc")
                    x_m, y_m = normalized_to_centred_metres(loc_str, length_m, width_m)
                    entities.append(EntityPosition(entity_id=pid, team=team, x_m=x_m, y_m=y_m))

                raw_time_ms = current_v_idx * 40 + clock_offset_ms
                stamp_us = raw_time_ms * 1000
                key = FrameKey(MATCH_ID, PERIOD, current_v_idx)
                feats = extractor.extract(key, stamp_us, entities)
                yield current_v_idx, raw_f, raw_time_ms, feats
                current_v_idx += 1

            root.clear()

    # Fill any remaining frames up to total_video_frames (e.g. frame 67624)
    while current_v_idx < total_video_frames:
        raw_time_ms = current_v_idx * 40 + clock_offset_ms
        stamp_us = raw_time_ms * 1000
        expected_raw_f = raw_frame_offset + current_v_idx
        key = FrameKey(MATCH_ID, PERIOD, current_v_idx)
        feats = extractor.extract(key, stamp_us, entities=[])
        yield current_v_idx, expected_raw_f, raw_time_ms, feats
        current_v_idx += 1


def extract_and_save(
    data_xml_path: Path,
    metadata_xml_path: Path,
    output_dir: Path,
    total_video_frames: int = TOTAL_VIDEO_FRAMES,
    skip_hash_check: bool = False,
) -> Dict[str, object]:
    """Execute complete feature extraction and persist artifacts."""
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    data_xml_sha256 = sha256_file(data_xml_path) if not skip_hash_check else "skipped"
    meta_xml_sha256 = sha256_file(metadata_xml_path) if not skip_hash_check else "skipped"

    metadata = parse_metadata(metadata_xml_path)

    sqlite_path = output_dir / "features.sqlite"
    npz_path = output_dir / "features.npz"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"

    # Preallocate numpy matrices
    video_frames_arr = np.zeros(total_video_frames, dtype=np.int32)
    raw_frames_arr = np.zeros(total_video_frames, dtype=np.int32)
    raw_times_ms_arr = np.zeros(total_video_frames, dtype=np.int32)
    features_all_arr = np.full((total_video_frames, len(FEATURE_NAMES)), np.nan, dtype=np.float64)
    features_model_arr = np.full((total_video_frames, len(MODEL_FEATURES)), np.nan, dtype=np.float64)

    model_indices = [FEATURE_NAMES.index(n) for n in MODEL_FEATURES]

    # Initialize statistics
    stats = {
        name: {
            "defined": 0,
            "missing": 0,
            "min": float("inf"),
            "max": float("-inf"),
            "sum": 0.0,
        }
        for name in FEATURE_NAMES
    }

    # Setup SQLite database
    with closing(sqlite3.connect(sqlite_path)) as db:
        db.execute("PRAGMA synchronous=OFF")
        db.execute("PRAGMA journal_mode=MEMORY")

        col_defs = ", ".join(f"{name} REAL" for name in FEATURE_NAMES)
        db.execute(
            f"""CREATE TABLE features (
                match_id TEXT NOT NULL,
                period INTEGER NOT NULL,
                video_frame INTEGER NOT NULL,
                raw_frame INTEGER NOT NULL,
                raw_time_ms INTEGER NOT NULL,
                {col_defs},
                PRIMARY KEY (match_id, period, video_frame)
            ) WITHOUT ROWID"""
        )

        placeholders = ", ".join("?" for _ in range(5 + len(FEATURE_NAMES)))
        insert_sql = f"INSERT INTO features VALUES ({placeholders})"

        batch = []
        stream = extract_features_stream(
            data_xml_path=data_xml_path,
            metadata=metadata,
            total_video_frames=total_video_frames,
        )

        for v_idx, raw_f, raw_t, feats in stream:
            video_frames_arr[v_idx] = v_idx
            raw_frames_arr[v_idx] = raw_f
            raw_times_ms_arr[v_idx] = raw_t

            feat_vals = [feats[name] for name in FEATURE_NAMES]

            for i, name in enumerate(FEATURE_NAMES):
                v = feat_vals[i]
                if v is None or not math.isfinite(v):
                    stats[name]["missing"] += 1
                else:
                    stats[name]["defined"] += 1
                    stats[name]["sum"] += v
                    if v < stats[name]["min"]:
                        stats[name]["min"] = v
                    if v > stats[name]["max"]:
                        stats[name]["max"] = v
                    features_all_arr[v_idx, i] = v

            features_model_arr[v_idx, :] = features_all_arr[v_idx, model_indices]

            batch.append((MATCH_ID, PERIOD, v_idx, raw_f, raw_t, *feat_vals))

            if len(batch) >= 4096:
                db.executemany(insert_sql, batch)
                db.commit()
                batch.clear()

        if batch:
            db.executemany(insert_sql, batch)
            db.commit()
            batch.clear()

        # Metadata table
        db.execute("CREATE TABLE feature_metadata (json TEXT NOT NULL)")
        meta_json = json.dumps(
            {
                "feature_version": 2,
                "match_id": MATCH_ID,
                "period": PERIOD,
                "total_video_frames": total_video_frames,
                "fps": 25,
                "feature_names": FEATURE_NAMES,
                "model_features": MODEL_FEATURES,
                "units": FEATURE_UNITS,
                "pitch_dimensions": {
                    "length_m": float(metadata["length_m"]),
                    "width_m": float(metadata["width_m"]),
                },
            },
            sort_keys=True,
        )
        db.execute("INSERT INTO feature_metadata VALUES (?)", (meta_json,))
        db.commit()

    # Save NPZ
    np.savez_compressed(
        npz_path,
        video_frames=video_frames_arr,
        raw_frames=raw_frames_arr,
        raw_times_ms=raw_times_ms_arr,
        features=features_model_arr,
        features_all=features_all_arr,
        feature_names=np.array(FEATURE_NAMES),
        model_features=np.array(MODEL_FEATURES),
    )

    elapsed_s = round(time.perf_counter() - started, 3)

    # Finalize statistics
    summary_features = {}
    for name, s in stats.items():
        count_def = s["defined"]
        summary_features[name] = {
            "defined": count_def,
            "missing": s["missing"],
            "coverage": round(count_def / total_video_frames, 4),
            "min": round(s["min"], 4) if count_def > 0 else None,
            "max": round(s["max"], 4) if count_def > 0 else None,
            "mean": round(s["sum"] / count_def, 4) if count_def > 0 else None,
            "unit": FEATURE_UNITS.get(name, ""),
        }

    summary = {
        "match_id": MATCH_ID,
        "period": PERIOD,
        "total_video_frames": total_video_frames,
        "clock_offset_ms": CLOCK_OFFSET_MS,
        "raw_frame_offset": RAW_FRAME_OFFSET,
        "elapsed_seconds": elapsed_s,
        "feature_count_total": len(FEATURE_NAMES),
        "feature_count_model": len(MODEL_FEATURES),
        "features": summary_features,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "kind": "soccertrack_117093_tracking_features",
        "dataset": "atomscott/soccertrack-v2",
        "feature_version": 2,
        "match_id": MATCH_ID,
        "period": PERIOD,
        "total_video_frames": total_video_frames,
        "fps": 25,
        "clock_offset_ms": CLOCK_OFFSET_MS,
        "raw_frame_offset": RAW_FRAME_OFFSET,
        "pitch_dimensions": {
            "length_m": float(metadata["length_m"]),
            "width_m": float(metadata["width_m"]),
        },
        "feature_names": list(FEATURE_NAMES),
        "model_features": list(MODEL_FEATURES),
        "units": FEATURE_UNITS,
        "input": {
            "data_xml": str(data_xml_path.relative_to(ROOT) if data_xml_path.is_relative_to(ROOT) else data_xml_path),
            "data_xml_sha256": data_xml_sha256,
            "metadata_xml": str(metadata_xml_path.relative_to(ROOT) if metadata_xml_path.is_relative_to(ROOT) else metadata_xml_path),
            "metadata_xml_sha256": meta_xml_sha256,
        },
        "output_sha256": {
            "features.sqlite": sha256_file(sqlite_path),
            "features.npz": sha256_file(npz_path),
            "summary.json": sha256_file(summary_path),
        },
        "status": "ready",
        "provenance": {
            "extractor": "external_baseline.features.ContextFeatureExtractor",
            "model_feature_policy": "V2 Model C: drops signed ball_x_m and ball_y_m (18 features)",
            "alignment": "video_frame * 40 + 1040 ms, raw_frame = 277 + video_frame",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-xml",
        type=Path,
        default=ROOT / "Soccertrack V2/raw/117093/117093_tracker_box_data.xml",
        help="Path to 117093_tracker_box_data.xml",
    )
    parser.add_argument(
        "--meta-xml",
        type=Path,
        default=ROOT / "Soccertrack V2/raw/117093/117093_tracker_box_metadata.xml",
        help="Path to 117093_tracker_box_metadata.xml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "external_artifacts/v3/117093-tracking-features-20261116",
        help="Target output directory",
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Skip SHA-256 calculation for inputs",
    )
    args = parser.parse_args()

    summary = extract_and_save(
        data_xml_path=args.data_xml,
        metadata_xml_path=args.meta_xml,
        output_dir=args.output_dir,
        skip_hash_check=args.skip_hash_check,
    )
    print(json.dumps({
        "status": "ready",
        "total_video_frames": summary["total_video_frames"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
