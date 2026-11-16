"""Verification runner for Multimodal Bundle V3."""

import argparse
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "src"))

from multimodal_bundle.bundle_verifier import verify_bundle

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true", help="Perform model load and runtime forward pass check")
    args = parser.parse_args()

    manifest = verify_bundle(root, check_runtime=args.runtime)
    print("Bundle V3 verified successfully!")
    print(f"Bundle version: {manifest.get('bundle_version')}")
    print(f"Verified files count: {len(manifest.get('member_sha256', {}))}")

if __name__ == "__main__":
    main()
