"""Cryptographic integrity and runtime verifier for Multimodal Bundle V3."""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

def sha256_file(path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def verify_bundle(bundle_dir: Path, check_runtime: bool = False) -> Dict[str, object]:
    """Verify cryptographic checksums and runtime compatibility of Bundle V3."""
    bundle_path = Path(bundle_dir).resolve(strict=True)
    manifest_file = bundle_path / "bundle_manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Missing bundle_manifest.json in {bundle_path}")

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("bundle_version") != 3:
        raise ValueError(f"Expected bundle_version 3, got {manifest.get('bundle_version')}")

    # Check member files
    member_hashes = manifest.get("member_sha256", {})
    if not member_hashes:
        raise ValueError("bundle_manifest.json has no member_sha256 entries")

    for rel_path, expected_hash in member_hashes.items():
        file_path = bundle_path / rel_path
        if not file_path.exists():
            raise FileNotFoundError(f"Missing bundle member file: {rel_path}")
        actual_hash = sha256_file(file_path)
        if actual_hash != expected_hash:
            raise ValueError(f"Checksum mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}")

    if check_runtime:
        import joblib
        import numpy as np

        model_path = bundle_path / "model/model.joblib"
        loaded = joblib.load(model_path)
        if not isinstance(loaded, dict) or "model" not in loaded:
            raise ValueError("model.joblib does not contain valid bundle dictionary")
        clf = loaded["model"]
        if not hasattr(clf, "predict_proba"):
            raise ValueError("Model does not implement predict_proba")

        # Test dummy forward pass
        dummy_x = np.zeros((2, 26), dtype=np.float64)
        dummy_probs = clf.predict_proba(dummy_x)
        if dummy_probs.shape != (2, 3):
            raise ValueError(f"Expected model prediction shape (2, 3), got {dummy_probs.shape}")

        decoder_path = bundle_path / "model/decoder_parameters.json"
        decoder_params = json.loads(decoder_path.read_text(encoding="utf-8"))
        if "transition_matrix" not in decoder_params or "state_prior" not in decoder_params:
            raise ValueError("decoder_parameters.json missing required HMM fields")

    return manifest
