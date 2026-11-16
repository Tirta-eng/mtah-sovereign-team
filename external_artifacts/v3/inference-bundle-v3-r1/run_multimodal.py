"""Local-operator Multimodal ATD+Video inference for MTAH 2027.

Combines 18 kinematic tracking features and 8 video visual features with
Viterbi HMM temporal decoding to produce compliant submission.csv.

Operator-only command:
python -B run_multimodal.py --bundle . --metadata PATH_XML --tracking PATH_TXT --video PATH_MP4 --homography PATH_NPY --output OUTPUT_DIR --period-source organizer-public --game BAR-ATH --operator-local
"""

import argparse
from pathlib import Path
import sys

# Ensure geometry runtime (with opencv) is in sys.path if not present
root = Path(__file__).resolve().parent
for parent in (root, root.parent, root.parent.parent, root.parent.parent.parent):
    geom = parent / "runtime/v3-geometry-venv/Lib/site-packages"
    if geom.exists() and str(geom) not in sys.path:
        sys.path.insert(0, str(geom))
        break

sys.path.insert(0, str(root / "src"))

from multimodal_bundle.pipeline import run_multimodal_inference

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=root, help="Path to bundle directory (default: current bundle)")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to Metrica/FIFA metadata XML")
    parser.add_argument("--tracking", type=Path, required=True, help="Path to ATD tracking text file")
    parser.add_argument("--video", type=Path, required=True, help="Path to panorama video MP4 file")
    parser.add_argument("--homography", type=Path, required=True, help="Path to 3x3 homography matrix .npy")
    parser.add_argument("--output", type=Path, required=True, help="Path to output directory")
    parser.add_argument("--expected-frames", type=int, default=199319, help="Total match frames (default: 199319)")
    parser.add_argument("--period-source", choices=("organizer-public", "provider"), default="organizer-public",
                        help="organizer-public: documented per-half video scope; provider: XML provider parameters")
    parser.add_argument("--game", help="Organizer game identifier (e.g. BAR-ATH); required for organizer-public")
    parser.add_argument("--operator-local", action="store_true", required=True,
                        help="Operator-only explicit confirmation flag; required")

    args = parser.parse_args()

    print(f"Starting Multimodal Bundle V3 inference on game '{args.game}'...")
    report = run_multimodal_inference(
        bundle_dir=args.bundle,
        metadata_path=args.metadata,
        tracking_path=args.tracking,
        video_path=args.video,
        homography_path=args.homography,
        output_dir=args.output,
        expected_frames=args.expected_frames,
        period_source=args.period_source,
        game=args.game,
    )
    print("Inference completed successfully!")
    print(f"Submission saved to: {args.output / 'submission.csv'}")
    print(f"Total rows: {report['expected_frames']}, Execution time: {report['elapsed_seconds']}s")

if __name__ == "__main__":
    main()
