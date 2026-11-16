"""Generate certified external possession targets for SoccerTrack 117093.

Maps raw XML ballStatus attributes from SoccerTrack v1.2 match 117093 to project
target conventions (H, A, D) with deterministic clock alignment (+1040 ms offset)
and explicit masking of ambiguous NEUTRAL transitions, in accordance with
docs/protokol_pengesahan_possession_117093.md and docs/hasil_pengesahan_possession_117093.md.

No official competition data, models, or predictions are accessed or modified.
"""

import argparse
from contextlib import closing
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Dict, Optional, Tuple
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
MATCH_ID = "117093"
PERIOD = 1
TOTAL_VIDEO_FRAMES = 67625  # 0-indexed: 0..67624
CLOCK_OFFSET_MS = 1040
RAW_FRAME_OFFSET = 277  # 251 (start raw frame) + 26 (1040 ms / 40 ms)
EXPECTED_XML_SHA256 = "43f45382e667959f2fc5904f87f65b597de103eb225616c0e6309b2144a9a7c1"
EXPECTED_XML_BYTES = 216067470


def sha256_file(path: Path) -> str:
    """Compute SHA-256 digest of a file in 1 MB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def map_status(ball_status: Optional[str]) -> Tuple[Optional[str], int, Optional[str]]:
    """Map SoccerTrack ballStatus to competition labels with masking.

    Returns:
        (label, valid_target, invalid_reason)
    """
    if ball_status == "HOME":
        return ("H", 1, None)
    if ball_status == "AWAY":
        return ("A", 1, None)
    if ball_status == "BALLOUT":
        return ("D", 1, None)
    if ball_status == "NEUTRAL":
        return (None, 0, "neutral_transition_masked")
    if ball_status in ("OUT_OF_COVERAGE", None):
        return (None, 0, "outside_raw_coverage")
    return (None, 0, f"unknown_status_{ball_status}")


def calculate_alignment(video_frame_idx: int) -> Tuple[int, int]:
    """Calculate deterministic raw frame number and raw matchTime (ms).

    Formula:
        raw_time_ms = video_frame_idx * 40 + CLOCK_OFFSET_MS
        raw_frame_number = RAW_FRAME_OFFSET + video_frame_idx
    """
    if video_frame_idx < 0:
        raise ValueError(f"video_frame_idx must be non-negative, got {video_frame_idx}")
    raw_time_ms = video_frame_idx * 40 + CLOCK_OFFSET_MS
    raw_frame_number = RAW_FRAME_OFFSET + video_frame_idx
    return raw_frame_number, raw_time_ms


def stream_first_half_raw_xml(xml_path: Path) -> Dict[int, Tuple[int, str]]:
    """Stream raw XML and extract {raw_frameNumber: (matchTime, ballStatus)} for FIRST_HALF."""
    first_half_data: Dict[int, Tuple[int, str]] = {}
    parser = ET.iterparse(xml_path, events=("start", "end"))
    _, root = next(parser)
    if root.tag != "data":
        raise ValueError(f"Unexpected XML root element: {root.tag}")

    for event, elem in parser:
        if event == "end" and elem.tag == "frame":
            if elem.attrib.get("eventPeriod") == "FIRST_HALF":
                raw_f = int(elem.attrib["frameNumber"])
                raw_t = int(elem.attrib["matchTime"])
                status = elem.attrib.get("ballStatus", "")
                first_half_data[raw_f] = (raw_t, status)
            root.clear()
    return first_half_data


def generate_targets(
    xml_path: Path,
    output_dir: Path,
    total_video_frames: int = TOTAL_VIDEO_FRAMES,
    skip_hash_check: bool = False,
) -> Dict[str, object]:
    """Generate targets database, CSV, and manifest for SoccerTrack 117093."""
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    if not xml_path.exists():
        raise FileNotFoundError(f"XML input not found: {xml_path}")

    # Integrity verification
    actual_size = xml_path.stat().st_size
    if not skip_hash_check:
        if actual_size != EXPECTED_XML_BYTES:
            raise ValueError(
                f"XML file size mismatch: expected {EXPECTED_XML_BYTES}, got {actual_size}"
            )
        actual_sha = sha256_file(xml_path)
        if actual_sha != EXPECTED_XML_SHA256:
            raise ValueError(
                f"XML SHA-256 mismatch: expected {EXPECTED_XML_SHA256}, got {actual_sha}"
            )
    else:
        actual_sha = "skipped_hash_check"

    t0 = time.time()
    raw_data = stream_first_half_raw_xml(xml_path)
    output_dir.mkdir(parents=True)

    db_path = output_dir / "targets.sqlite"
    csv_path = output_dir / "targets.csv"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"

    counts: Dict[str, int] = {
        "H": 0,
        "A": 0,
        "D": 0,
        "NEUTRAL_MASKED": 0,
        "OUT_OF_COVERAGE": 0,
        "UNKNOWN": 0,
        "TOTAL": total_video_frames,
        "VALID_TARGETS": 0,
        "INVALID_TARGETS": 0,
    }

    rows_to_insert = []
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file, lineterminator="\n")
        csv_writer.writerow(
            [
                "match_id",
                "period",
                "video_frame",
                "raw_frame",
                "raw_time_ms",
                "ball_status",
                "label",
                "valid_target",
                "invalid_reason",
            ]
        )

        for v_idx in range(total_video_frames):
            raw_f, raw_t = calculate_alignment(v_idx)
            if raw_f in raw_data:
                xml_t, ball_status = raw_data[raw_f]
                # Consistency check on time grid
                if xml_t != (raw_f - 251) * 40:
                    raise ValueError(
                        f"XML matchTime grid desync at frame {raw_f}: expected {(raw_f - 251) * 40}, got {xml_t}"
                    )
            else:
                ball_status = "OUT_OF_COVERAGE"

            label, valid_target, invalid_reason = map_status(ball_status)

            # Update distribution counts
            if valid_target == 1:
                counts["VALID_TARGETS"] += 1
                if label in counts:
                    counts[label] += 1
            else:
                counts["INVALID_TARGETS"] += 1
                if ball_status == "NEUTRAL":
                    counts["NEUTRAL_MASKED"] += 1
                elif ball_status == "OUT_OF_COVERAGE":
                    counts["OUT_OF_COVERAGE"] += 1
                else:
                    counts["UNKNOWN"] += 1

            row = (
                MATCH_ID,
                PERIOD,
                v_idx,
                raw_f,
                raw_t,
                ball_status,
                label,
                valid_target,
                invalid_reason,
            )
            rows_to_insert.append(row)
            csv_writer.writerow(row)

    # Insert into SQLite database matching project conventions
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("PRAGMA cache_size=-8192")
        conn.execute("PRAGMA temp_store=FILE")
        with conn:
            conn.execute(
                """CREATE TABLE targets (
                    match_id TEXT NOT NULL,
                    period INTEGER NOT NULL,
                    video_frame INTEGER NOT NULL,
                    raw_frame INTEGER NOT NULL,
                    raw_time_ms INTEGER NOT NULL,
                    ball_status TEXT NOT NULL,
                    label TEXT,
                    valid_target INTEGER NOT NULL,
                    invalid_reason TEXT,
                    PRIMARY KEY (match_id, period, video_frame)
                ) WITHOUT ROWID"""
            )
            conn.execute(
                "CREATE INDEX idx_targets_valid ON targets (valid_target, label)"
            )
            conn.executemany(
                """INSERT INTO targets (
                    match_id, period, video_frame, raw_frame, raw_time_ms,
                    ball_status, label, valid_target, invalid_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows_to_insert,
            )

    duration_s = round(time.time() - t0, 3)

    summary = {
        "match_id": MATCH_ID,
        "period": PERIOD,
        "total_video_frames": total_video_frames,
        "clock_offset_ms": CLOCK_OFFSET_MS,
        "raw_frame_offset": RAW_FRAME_OFFSET,
        "counts": counts,
        "coverage_percentage": round(
            counts["VALID_TARGETS"] / total_video_frames * 100, 2
        ),
        "neutral_masked_percentage": round(
            counts["NEUTRAL_MASKED"] / total_video_frames * 100, 2
        ),
        "duration_seconds": duration_s,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "kind": "soccertrack_117093_possession_targets",
        "dataset": "atomscott/soccertrack-v2",
        "dataset_revision": "325bd4c57ee8e135ab652f3921d274526ada496b",
        "protocol": "docs/protokol_pengesahan_possession_117093.md",
        "findings": "docs/hasil_pengesahan_possession_117093.md",
        "source_xml_sha256": actual_sha,
        "source_xml_bytes": actual_size,
        "database_sha256": sha256_file(db_path),
        "csv_sha256": sha256_file(csv_path),
        "summary": summary,
        "adjudication_rules": {
            "HOME": "H (Team 9701, Blue jersey)",
            "AWAY": "A (Team 9834, White jersey)",
            "BALLOUT": "D (Boundary dead-ball)",
            "NEUTRAL": "Masked (valid_target=0, label=NULL, transition ambiguity)",
            "OUT_OF_COVERAGE": "Masked (valid_target=0, frame 67624 beyond raw coverage)",
            "goalkeeper_open_play_rule": "Law 12 IFAB: open-play keeper possession is live play",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output directory (must not exist)",
    )
    parser.add_argument(
        "--xml-path",
        type=Path,
        default=ROOT / "Soccertrack V2/raw/117093/117093_tracker_box_data.xml",
        help="Path to raw XML tracker file",
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Skip strict file hash verification (for testing/synthetic runs)",
    )
    args = parser.parse_args()

    output_dir = args.output.resolve()
    xml_path = args.xml_path.resolve()

    manifest = generate_targets(
        xml_path=xml_path,
        output_dir=output_dir,
        skip_hash_check=args.skip_hash_check,
    )
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
