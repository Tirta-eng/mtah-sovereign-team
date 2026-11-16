"""Unit tests for SoccerTrack possession target generator.

Verifies deterministic clock alignment, status mapping contracts (H, A, D, masking),
coverage bounds handling, and SQLite database integrity using synthetic fixtures.
No official data, models, or network calls are used.
"""

import importlib.util
from pathlib import Path
import sqlite3
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "gen_targets",
    Path(__file__).resolve().parents[1]
    / "tools/generate_soccertrack_possession_targets.py",
)
G = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(G)


class GeneratePossessionTargetsTests(unittest.TestCase):
    def test_status_mapping_contract(self):
        self.assertEqual(G.map_status("HOME"), ("H", 1, None))
        self.assertEqual(G.map_status("AWAY"), ("A", 1, None))
        self.assertEqual(G.map_status("BALLOUT"), ("D", 1, None))
        self.assertEqual(
            G.map_status("NEUTRAL"), (None, 0, "neutral_transition_masked")
        )
        self.assertEqual(
            G.map_status("OUT_OF_COVERAGE"), (None, 0, "outside_raw_coverage")
        )
        self.assertEqual(G.map_status(None), (None, 0, "outside_raw_coverage"))
        self.assertEqual(
            G.map_status("INVALID_STATUS"),
            (None, 0, "unknown_status_INVALID_STATUS"),
        )

    def test_calculate_alignment_formula(self):
        # video 0 -> raw_frame 277 (251 + 26), raw_time 1040 ms
        self.assertEqual(G.calculate_alignment(0), (277, 1040))
        # video 1 -> raw_frame 278, raw_time 1080 ms
        self.assertEqual(G.calculate_alignment(1), (278, 1080))
        # video 100 -> raw_frame 377, raw_time 5040 ms
        self.assertEqual(G.calculate_alignment(100), (377, 5040))
        with self.assertRaises(ValueError):
            G.calculate_alignment(-1)

    def test_synthetic_xml_generation_and_database(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            synthetic_xml = tmp_path / "synthetic_tracker.xml"
            output_dir = tmp_path / "targets_out"

            # Minimal synthetic XML covering offset +26 (raw 277..279)
            xml_content = (
                '<?xml version="1.0" encoding="utf-8"?>\n'
                "<data>\n"
                '  <frame eventPeriod="FIRST_HALF" frameNumber="251" matchTime="0" ballStatus="BALLOUT"/>\n'
                '  <frame eventPeriod="FIRST_HALF" frameNumber="277" matchTime="1040" ballStatus="HOME"/>\n'
                '  <frame eventPeriod="FIRST_HALF" frameNumber="278" matchTime="1080" ballStatus="AWAY"/>\n'
                '  <frame eventPeriod="FIRST_HALF" frameNumber="279" matchTime="1120" ballStatus="NEUTRAL"/>\n'
                '  <frame eventPeriod="SECOND_HALF" frameNumber="251" matchTime="2699967" ballStatus="HOME"/>\n'
                "</data>\n"
            )
            synthetic_xml.write_text(xml_content, encoding="utf-8")

            # Run generation for 4 video frames: 0 (HOME), 1 (AWAY), 2 (NEUTRAL), 3 (OUT_OF_COVERAGE)
            manifest = G.generate_targets(
                xml_path=synthetic_xml,
                output_dir=output_dir,
                total_video_frames=4,
                skip_hash_check=True,
            )

            self.assertTrue(output_dir.exists())
            self.assertTrue((output_dir / "targets.sqlite").exists())
            self.assertTrue((output_dir / "targets.csv").exists())
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "manifest.json").exists())

            summary = manifest["summary"]
            self.assertEqual(summary["total_video_frames"], 4)
            self.assertEqual(summary["counts"]["VALID_TARGETS"], 2)  # H and A
            self.assertEqual(summary["counts"]["INVALID_TARGETS"], 2)  # NEUTRAL and OUT_OF_COVERAGE
            self.assertEqual(summary["counts"]["H"], 1)
            self.assertEqual(summary["counts"]["A"], 1)
            self.assertEqual(summary["counts"]["D"], 0)
            self.assertEqual(summary["counts"]["NEUTRAL_MASKED"], 1)
            self.assertEqual(summary["counts"]["OUT_OF_COVERAGE"], 1)

            # Query database directly
            conn = sqlite3.connect(output_dir / "targets.sqlite")
            try:
                rows = conn.execute(
                    "SELECT video_frame, raw_frame, raw_time_ms, ball_status, label, valid_target, invalid_reason "
                    "FROM targets ORDER BY video_frame"
                ).fetchall()

                self.assertEqual(len(rows), 4)
                # Frame 0: HOME -> H
                self.assertEqual(rows[0], (0, 277, 1040, "HOME", "H", 1, None))
                # Frame 1: AWAY -> A
                self.assertEqual(rows[1], (1, 278, 1080, "AWAY", "A", 1, None))
                # Frame 2: NEUTRAL -> Masked
                self.assertEqual(
                    rows[2],
                    (2, 279, 1120, "NEUTRAL", None, 0, "neutral_transition_masked"),
                )
                # Frame 3: Outside raw coverage
                self.assertEqual(
                    rows[3],
                    (3, 280, 1160, "OUT_OF_COVERAGE", None, 0, "outside_raw_coverage"),
                )

                # Primary key constraint check
                with self.assertRaises(sqlite3.IntegrityError):
                    conn.execute(
                        "INSERT INTO targets (match_id, period, video_frame, raw_frame, raw_time_ms, ball_status, label, valid_target, invalid_reason) "
                        "VALUES ('117093', 1, 0, 277, 1040, 'HOME', 'H', 1, NULL)"
                    )
            finally:
                conn.close()

    def test_existing_output_directory_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            existing = Path(tmp_dir) / "already_exists"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                G.generate_targets(
                    xml_path=Path("non_existent.xml"),
                    output_dir=existing,
                    skip_hash_check=True,
                )

    def test_hash_mismatch_rejected_without_skip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_file = Path(tmp_dir) / "dummy.xml"
            dummy_file.write_text("<data></data>", encoding="utf-8")
            out = Path(tmp_dir) / "out"
            with self.assertRaises(ValueError):
                G.generate_targets(
                    xml_path=dummy_file,
                    output_dir=out,
                    skip_hash_check=False,
                )


if __name__ == "__main__":
    unittest.main()
