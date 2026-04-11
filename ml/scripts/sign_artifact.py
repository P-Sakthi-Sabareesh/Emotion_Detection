#!/usr/bin/env python3
"""Sign a trained model artifact so the runtime loader can trust it.

Usage:
    FER_MODEL_SIGNING_KEY=<secret> python ml/scripts/sign_artifact.py \
        --manifest ml/artifacts/artifact_manifest.json

Re-computes each file's sha256, rewrites the manifest, then attaches an
HMAC-SHA256 signature over the canonical ``files`` block.  The runtime loader
(`inference/model_loader.py`) refuses artifacts whose signature does not match.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(files_section: dict) -> bytes:
    return json.dumps(files_section, sort_keys=True, separators=(",", ":")).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sign a FER model artifact manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--key-env", default="FER_MODEL_SIGNING_KEY")
    parser.add_argument("--model-version", default=None)
    args = parser.parse_args()

    key = os.environ.get(args.key_env, "")
    if not key:
        print(f"ERROR: {args.key_env} must be set in the environment", file=sys.stderr)
        return 2

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files") or {}
    if not files:
        print("ERROR: manifest has no files section", file=sys.stderr)
        return 2

    manifest_dir = manifest_path.parent
    for name, entry in files.items():
        file_path = Path(entry.get("path", manifest_dir / name)).resolve()
        if not file_path.exists():
            alt = (manifest_dir / Path(entry.get("path", name)).name).resolve()
            if alt.exists():
                file_path = alt
            else:
                print(f"ERROR: file missing for manifest entry '{name}': {file_path}", file=sys.stderr)
                return 2
        entry["path"] = str(file_path)
        entry["sha256"] = sha256_file(file_path)
        entry["size_bytes"] = file_path.stat().st_size

    if args.model_version:
        manifest["model_version"] = args.model_version

    signature = hmac.new(
        key.encode("utf-8"), canonical_bytes(files), hashlib.sha256
    ).hexdigest()
    manifest["signature_hex"] = signature
    manifest["signature_algorithm"] = "HMAC-SHA256"

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Signed manifest: {manifest_path}")
    print(f"Signature: {signature}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
