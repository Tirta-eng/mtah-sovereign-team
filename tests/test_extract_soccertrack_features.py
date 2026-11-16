"""Unit tests for SoccerTrack feature extractor.

Verifies:
  - Feature contract (20 full features, 18 model features)
  - Coordinate normalization to centered metres
  - Metadata parsing for pitch dimensions and team mappings
  - Sequential streaming, gap handling, and boundary padding
  - Synthetic end-to-end feature extraction into SQLite and NPZ

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
    "extract_features",
    Path(__file__).resolve().parents[1]
    / "tools/extract_soccertrack_features_117093.py",
)
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)


class ExtractSoccertrackFeaturesTests(unittest.TestCase):
    def test_feature_contracts_match_v2_bundle(self):
        bundle_contract_path = (
            Path(__file__).resolve().parents[1]
            / "external_artifacts/v2/inference-bundle-v2-r1/feature_contract.json"
        )
        if bundle_contract_path.exists():
            contract = json.loads(bundle_contract_path.read_text(encoding="utf-8"))
            self.assertEqual(list(E.FEATURE_NAMES), contract["feature_names"])
            self.assertEqual(list(E.MODEL_FEATURES), contract["model_features"])
        self.assertEqual(len(E.FEATURE_NAMES), 20)
        self.assertEqual(len(E.MODEL_FEATURES), 18)
        self.assertNotIn("ball_x_m", E.MODEL_FEATURES)
        self.assertNotIn("ball_y_m", E.MODEL_FEATURES)

    def test_coordinate_normalization(self):
        # Center kickoff
        x, y = E.normalized_to_centred_metres("[0.5, 0.5]", 105.0, 68.0)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 0.0)

        # Top-left corner
        x, y = E.normalized_to_centred_metres("[0.0, 0.0]", 105.0, 68.0)
        self.assertAlmostEqual(x, -52.5)
        self.assertAlmostEqual(y, -34.0)

        # Bottom-right corner
        x, y = E.normalized_to_centred_metres("[1.0, 1.0]", 105.0, 68.0)
        self.assertAlmostEqual(x, 52.5)
        self.assertAlmostEqual(y, 34.0)

        # Invalid or NA strings
        x, y = E.normalized_to_centred_metres("NA", 105.0, 68.0)
        self.assertIsNone(x)
        self.assertIsNone(y)

        x, y = E.normalized_to_centred_metres("", 105.0, 68.0)
        self.assertIsNone(x)
        self.assertIsNone(y)

    def test_metadata_parsing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            meta_xml = tmp_path / "meta.xml"
            meta_xml.write_text(
                '<?xml version="1.0" encoding="utf-8"?>\n'
                "<metadata>\n"
                '  <pitch width="105" height="68"/>\n'
                "  <teams>\n"
                '    <team id="9701" name="HomeTeam" side="home"/>\n'
                '    <team id="9834" name="AwayTeam" side="away"/>\n'
                "  </teams>\n"
                "  <players>\n"
                '    <player id="101" teamId="9701"/>\n'
                '    <player id="102" teamId="9834"/>\n'
                "  </players>\n"
                "</metadata>\n",
                encoding="utf-8",
            )

            meta = E.parse_metadata(meta_xml)
            self.assertEqual(meta["length_m"], 105.0)
            self.assertEqual(meta["width_m"], 68.0)
            self.assertEqual(meta["teams"], {"9701": "H", "9834": "A"})
            self.assertEqual(meta["player_to_team"], {"101": "H", "102": "A"})

    def test_synthetic_end_to_end_extraction(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            meta_xml = tmp_path / "meta.xml"
            data_xml = tmp_path / "data.xml"
            out_dir = tmp_path / "features_out"

            meta_xml.write_text(
                '<?xml version="1.0" encoding="utf-8"?>\n'
                "<metadata>\n"
                '  <pitch width="105" height="68"/>\n'
                "  <teams>\n"
                '    <team id="9701" name="HomeTeam" side="home"/>\n'
                '    <team id="9834" name="AwayTeam" side="away"/>\n'
                "  </teams>\n"
                "  <players>\n"
                '    <player id="101" teamId="9701"/>\n'
                '    <player id="102" teamId="9834"/>\n'
                "  </players>\n"
                "</metadata>\n",
                encoding="utf-8",
            )

            # Data covers raw frames 277 (v_idx=0) and 278 (v_idx=1)
            # Frame 2 (raw 279) is missing from XML, testing gap/boundary padding
            data_xml.write_text(
                '<?xml version="1.0" encoding="utf-8"?>\n'
                "<data>\n"
                '  <frame eventPeriod="FIRST_HALF" frameNumber="251" matchTime="0" ballStatus="BALLOUT">\n'
                '    <player playerId="101" loc="[0.5, 0.5]"/>\n'
                "  </frame>\n"
                '  <frame eventPeriod="FIRST_HALF" frameNumber="277" matchTime="1040" ballStatus="HOME">\n'
                '    <player playerId="101" loc="[0.5, 0.5]"/>\n'
                '    <player playerId="102" loc="[0.6, 0.5]"/>\n'
                '    <ball playerId="ball" loc="[0.5, 0.5]"/>\n'
                "  </frame>\n"
                '  <frame eventPeriod="FIRST_HALF" frameNumber="278" matchTime="1080" ballStatus="HOME">\n'
                '    <player playerId="101" loc="[0.5, 0.5]"/>\n'
                '    <player playerId="102" loc="[0.6, 0.5]"/>\n'
                '    <ball playerId="ball" loc="[0.50038, 0.5]"/>\n'
                "  </frame>\n"
                "</data>\n",
                encoding="utf-8",
            )

            summary = E.extract_and_save(
                data_xml_path=data_xml,
                metadata_xml_path=meta_xml,
                output_dir=out_dir,
                total_video_frames=3,
                skip_hash_check=True,
            )

            self.assertEqual(summary["total_video_frames"], 3)
            self.assertTrue((out_dir / "features.sqlite").exists())
            self.assertTrue((out_dir / "features.npz").exists())
            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "manifest.json").exists())

            # Verify SQLite contents
            with closing(sqlite3.connect(out_dir / "features.sqlite")) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT video_frame, raw_frame, raw_time_ms, ball_xy_observed, nearest_home_m FROM features ORDER BY video_frame"
                )
                rows = cursor.fetchall()
                self.assertEqual(len(rows), 3)

                # Frame 0: ball present at (0, 0), home player 101 at (0, 0) -> nearest_home_m == 0.0
                self.assertEqual(rows[0][0], 0)
                self.assertEqual(rows[0][1], 277)
                self.assertEqual(rows[0][2], 1040)
                self.assertEqual(rows[0][3], 1.0)
                self.assertAlmostEqual(rows[0][4], 0.0, places=2)

                # Frame 1: ball present
                self.assertEqual(rows[1][0], 1)
                self.assertEqual(rows[1][1], 278)
                self.assertEqual(rows[1][2], 1080)
                self.assertEqual(rows[1][3], 1.0)

                # Frame 2: padded/missing -> ball_xy_observed == 0.0, nearest_home_m is None
                self.assertEqual(rows[2][0], 2)
                self.assertEqual(rows[2][1], 279)
                self.assertEqual(rows[2][2], 1120)
                self.assertEqual(rows[2][3], 0.0)
                self.assertIsNone(rows[2][4])

            # Verify NPZ contents
            with np.load(out_dir / "features.npz") as npz_data:
                self.assertEqual(npz_data["features"].shape, (3, 18))
                self.assertEqual(npz_data["features_all"].shape, (3, 20))
                self.assertEqual(len(npz_data["video_frames"]), 3)
                # Frame 2 is padded/missing: ball_record_present is 0.0, nearest_home_m is NaN
                ball_rec_idx = list(E.MODEL_FEATURES).index("ball_record_present")
                nearest_h_idx = list(E.MODEL_FEATURES).index("nearest_home_m")
                self.assertEqual(npz_data["features"][2, ball_rec_idx], 0.0)
                self.assertTrue(np.isnan(npz_data["features"][2, nearest_h_idx]))


if __name__ == "__main__":
    unittest.main()
