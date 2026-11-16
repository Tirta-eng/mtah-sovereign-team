"""Unit tests for Multimodal Inference Bundle V3 (Revision 1)."""

from contextlib import closing
import csv
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = ROOT / "external_artifacts/v3/inference-bundle-v3-r1"
sys.path.insert(0, str(BUNDLE_DIR / "src"))

try:
    import cv2
except ImportError:
    geom_site = ROOT / "runtime/v3-geometry-venv/Lib/site-packages"
    if geom_site.exists() and str(geom_site) not in sys.path:
        sys.path.insert(0, str(geom_site))
    import cv2

from multimodal_bundle.bundle_verifier import verify_bundle
from multimodal_bundle.contracts import ALL_FEATURES_26
from multimodal_bundle.pipeline import run_multimodal_inference


class TestInferenceBundleV3(unittest.TestCase):
    """Test suite for Bundle V3 verification and end-to-end synthetic inference."""

    def test_bundle_verification(self):
        manifest = verify_bundle(BUNDLE_DIR, check_runtime=True)
        self.assertEqual(manifest["bundle_version"], 3)
        self.assertEqual(manifest["bundle_name"], "inference-bundle-v3-r1")
        self.assertEqual(len(manifest["member_sha256"]), 16)

    def test_bundle_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Copy bundle manifest
            orig_manifest = json.loads((BUNDLE_DIR / "bundle_manifest.json").read_text(encoding="utf-8"))
            # Point to empty dir
            with self.assertRaises((FileNotFoundError, ValueError)):
                verify_bundle(tmp_path, check_runtime=False)

    def test_synthetic_multimodal_inference(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            work_dir = Path(tmp_dir)
            out_dir = work_dir / "output"

            # 1. Synthetic 3x3 homography
            homography_path = work_dir / "homography.npy"
            np.save(homography_path, np.eye(3, dtype=np.float64))

            # 2. Synthetic 10-frame video
            video_path = work_dir / "synthetic_panorama.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(str(video_path), fourcc, 25.0, (512, 256))
            for i in range(10):
                frame = np.full((256, 512, 3), (i * 25) % 255, dtype=np.uint8)
                vw.write(frame)
            vw.release()

            # 3. Synthetic FIFA metadata XML
            meta_xml = """<?xml version="1.0" encoding="utf-8"?>
<Metadata>
  <GlobalConfig>
    <FrameRate>25</FrameRate>
    <ProviderGlobalParameters>
      <ProviderParameter><Name>first_half_start</Name><Value>1</Value></ProviderParameter>
      <ProviderParameter><Name>first_half_end</Name><Value>8</Value></ProviderParameter>
      <ProviderParameter><Name>second_half_start</Name><Value>9</Value></ProviderParameter>
      <ProviderParameter><Name>second_half_end</Name><Value>10</Value></ProviderParameter>
    </ProviderGlobalParameters>
  </GlobalConfig>
  <FieldSize><Width>105</Width><Height>68</Height></FieldSize>
  <Teams>
    <Team id="HOME_TEAM"/>
    <Team id="AWAY_TEAM"/>
  </Teams>
  <MatchParameters>
    <Score idLocalTeam="HOME_TEAM" idVisitingTeam="AWAY_TEAM"/>
  </MatchParameters>
  <Players>
    <Player id="P1" teamId="HOME_TEAM"/>
    <Player id="P2" teamId="AWAY_TEAM"/>
  </Players>
  <PlayerChannels>
    <PlayerChannel id="C1_X" playerId="P1" channelId="x"/>
    <PlayerChannel id="C1_Y" playerId="P1" channelId="y"/>
    <PlayerChannel id="C2_X" playerId="P2" channelId="x"/>
    <PlayerChannel id="C2_Y" playerId="P2" channelId="y"/>
  </PlayerChannels>
  <Channels>
    <Channel id="x"><Unit>normalized</Unit></Channel>
    <Channel id="y"><Unit>normalized</Unit></Channel>
  </Channels>
  <DataFormatSpecifications>
    <DataFormatSpecification startFrame="1" endFrame="10" separator=":">
      <StringRegister name="frameCount"/>
      <SplitRegister separator=";">
        <SplitRegister separator=",">
          <PlayerChannelRef playerChannelId="C1_X"/>
          <PlayerChannelRef playerChannelId="C1_Y"/>
        </SplitRegister>
        <SplitRegister separator=",">
          <PlayerChannelRef playerChannelId="C2_X"/>
          <PlayerChannelRef playerChannelId="C2_Y"/>
        </SplitRegister>
      </SplitRegister>
      <SplitRegister separator=",">
        <BallChannelRef channelId="x"/>
        <BallChannelRef channelId="y"/>
      </SplitRegister>
    </DataFormatSpecification>
  </DataFormatSpecifications>
</Metadata>
"""
            meta_path = work_dir / "metadata.xml"
            meta_path.write_text(meta_xml, encoding="utf-8")

            # 4. Synthetic tracking text
            # Format: frameCount:p1_x,p1_y;p2_x,p2_y:ball_x,ball_y
            tracking_lines = []
            for f in range(1, 11):
                tracking_lines.append(f"{f}:0.50,0.50;0.52,0.52:0.51,0.51")
            tracking_path = work_dir / "tracking.txt"
            tracking_path.write_text("\n".join(tracking_lines) + "\n", encoding="utf-8")

            # 5. Run pipeline in provider mode with 10 expected frames
            report = run_multimodal_inference(
                bundle_dir=BUNDLE_DIR,
                metadata_path=meta_path,
                tracking_path=tracking_path,
                video_path=video_path,
                homography_path=homography_path,
                output_dir=out_dir,
                expected_frames=10,
                period_source="provider",
            )

            # Check report
            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["expected_frames"], 10)

            # Check submission.csv
            sub_csv = out_dir / "submission.csv"
            self.assertTrue(sub_csv.exists())
            with sub_csv.open("r", encoding="utf-8") as f:
                reader = list(csv.reader(f))
                self.assertEqual(reader[0], ["frame", "label"])
                self.assertEqual(len(reader), 11)  # header + 10 rows
                for idx, row in enumerate(reader[1:]):
                    self.assertEqual(int(row[0]), idx)
                    self.assertIn(row[1], ("H", "A", "D", "X"))

            # Check SQLite databases
            with closing(sqlite3.connect(out_dir / "features.sqlite")) as conn:
                count = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
                self.assertEqual(count, 10)

            with closing(sqlite3.connect(out_dir / "predictions.sqlite")) as conn:
                count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
                self.assertEqual(count, 10)

    def test_synthetic_multimodal_inference_with_csv_homography(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            work_dir = Path(tmp_dir)
            out_dir = work_dir / "output_csv"

            # 1. Synthetic homography CSV (10 frames)
            homography_csv_path = work_dir / "homography.csv"
            with homography_csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["frame", "h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22", "reprojection_error", "valid"])
                for i in range(10):
                    writer.writerow([i, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.1, 1])

            # 2. Synthetic 10-frame video
            video_path = work_dir / "synthetic_panorama.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(str(video_path), fourcc, 25.0, (512, 256))
            for i in range(10):
                frame = np.full((256, 512, 3), (i * 25) % 255, dtype=np.uint8)
                vw.write(frame)
            vw.release()

            # 3. Synthetic FIFA metadata XML
            meta_xml = """<?xml version="1.0" encoding="utf-8"?>
<Metadata>
  <GlobalConfig>
    <FrameRate>25</FrameRate>
    <ProviderGlobalParameters>
      <ProviderParameter><Name>first_half_start</Name><Value>1</Value></ProviderParameter>
      <ProviderParameter><Name>first_half_end</Name><Value>8</Value></ProviderParameter>
      <ProviderParameter><Name>second_half_start</Name><Value>9</Value></ProviderParameter>
      <ProviderParameter><Name>second_half_end</Name><Value>10</Value></ProviderParameter>
    </ProviderGlobalParameters>
  </GlobalConfig>
  <FieldSize><Width>105</Width><Height>68</Height></FieldSize>
  <Teams>
    <Team id="HOME_TEAM"/>
    <Team id="AWAY_TEAM"/>
  </Teams>
  <MatchParameters>
    <Score idLocalTeam="HOME_TEAM" idVisitingTeam="AWAY_TEAM"/>
  </MatchParameters>
  <Players>
    <Player id="P1" teamId="HOME_TEAM"/>
    <Player id="P2" teamId="AWAY_TEAM"/>
  </Players>
  <PlayerChannels>
    <PlayerChannel id="C1_X" playerId="P1" channelId="x"/>
    <PlayerChannel id="C1_Y" playerId="P1" channelId="y"/>
    <PlayerChannel id="C2_X" playerId="P2" channelId="x"/>
    <PlayerChannel id="C2_Y" playerId="P2" channelId="y"/>
  </PlayerChannels>
  <Channels>
    <Channel id="x"><Unit>normalized</Unit></Channel>
    <Channel id="y"><Unit>normalized</Unit></Channel>
  </Channels>
  <DataFormatSpecifications>
    <DataFormatSpecification startFrame="1" endFrame="10" separator=":">
      <StringRegister name="frameCount"/>
      <SplitRegister separator=";">
        <SplitRegister separator=",">
          <PlayerChannelRef playerChannelId="C1_X"/>
          <PlayerChannelRef playerChannelId="C1_Y"/>
        </SplitRegister>
        <SplitRegister separator=",">
          <PlayerChannelRef playerChannelId="C2_X"/>
          <PlayerChannelRef playerChannelId="C2_Y"/>
        </SplitRegister>
      </SplitRegister>
      <SplitRegister separator=",">
        <BallChannelRef channelId="x"/>
        <BallChannelRef channelId="y"/>
      </SplitRegister>
    </DataFormatSpecification>
  </DataFormatSpecifications>
</Metadata>
"""
            meta_path = work_dir / "metadata.xml"
            meta_path.write_text(meta_xml, encoding="utf-8")

            # 4. Synthetic tracking text
            tracking_lines = []
            for f in range(1, 11):
                tracking_lines.append(f"{f}:0.50,0.50;0.52,0.52:0.51,0.51")
            tracking_path = work_dir / "tracking.txt"
            tracking_path.write_text("\n".join(tracking_lines) + "\n", encoding="utf-8")

            # 5. Run pipeline with CSV homography
            report = run_multimodal_inference(
                bundle_dir=BUNDLE_DIR,
                metadata_path=meta_path,
                tracking_path=tracking_path,
                video_path=video_path,
                homography_path=homography_csv_path,
                output_dir=out_dir,
                expected_frames=10,
                period_source="provider",
            )

            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["expected_frames"], 10)
            self.assertTrue((out_dir / "submission.csv").exists())


if __name__ == "__main__":
    unittest.main()
