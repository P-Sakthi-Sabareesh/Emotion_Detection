#!/usr/bin/env python3
"""One-command bootstrap for the FER ViT ONNX artifact.

What this does:
  1. Checks whether a valid signed ONNX artifact already exists
     (sha256 in the committed manifest matches the file on disk).
     If yes, exits 0 with a skip notice — idempotent.
  2. Otherwise, downloads ``dima806/facial_emotions_image_detection``
     from HuggingFace Hub, exports it to ONNX via torch.onnx.export,
     writes the runtime manifest including the canonical class
     permutation, and signs the manifest with FER_MODEL_SIGNING_KEY
     (HMAC-SHA256).
  3. Verifies the signed artifact loads in onnxruntime and produces a
     valid 7-class output on a dummy input.
  4. Reports sha256 + size + signature so users can confirm.

Run once after `git clone`, before `docker compose up`:

    export FER_MODEL_SIGNING_KEY=$(python -c 'import secrets;print(secrets.token_urlsafe(48))')
    pip install -r requirements-bootstrap.txt
    python scripts/bootstrap_model.py

The script is intentionally self-contained — it imports nothing from
the Django project so it can be run in a minimal build stage.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

CANONICAL = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
DEFAULT_MODEL_ID = "dima806/facial_emotions_image_detection"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(files_section: dict) -> bytes:
    return json.dumps(files_section, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonicalize_label(label: str) -> str:
    normalized = label.strip().lower()
    remap = {
        "anger": "angry",
        "happiness": "happy",
        "sadness": "sad",
        "disgusted": "disgust",
        "fearful": "fear",
        "surprised": "surprise",
        "surprize": "surprise",
    }
    return remap.get(normalized, normalized)


def _resolve_class_permutation(id2label: dict) -> list[int]:
    lookup: dict[str, int] = {}
    for idx, label in id2label.items():
        lookup[_canonicalize_label(str(label))] = int(idx)
    missing = [c for c in CANONICAL if c not in lookup]
    if missing:
        raise SystemExit(
            f"Source model is missing canonical FER classes: {missing}. "
            f"Available labels: {list(id2label.values())}"
        )
    return [lookup[c] for c in CANONICAL]


def _existing_artifact_is_valid(onnx_path: Path, manifest_path: Path, signing_key: str) -> bool:
    """Return True if the cached artifact matches the manifest signature."""
    if not onnx_path.exists() or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    files = manifest.get("files") or {}
    model_entry = files.get("model") or {}
    expected_sha = (model_entry.get("sha256") or "").lower()
    if not expected_sha:
        return False
    actual_sha = _sha256_file(onnx_path).lower()
    if not hmac.compare_digest(expected_sha, actual_sha):
        return False
    signature = manifest.get("signature_hex")
    if not signature or not signing_key:
        return False
    expected_sig = hmac.new(
        signing_key.encode("utf-8"), _canonical_bytes(files), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected_sig, signature)


def _download_and_export(model_id: str, out_dir: Path, opset: int) -> tuple[Path, dict, list[int], int, list[float], list[float]]:
    """Download from HuggingFace and export to ONNX.

    Returns (onnx_path, id2label, canonical_perm, image_size, mean, std).
    """
    print(f"[bootstrap] downloading {model_id} from HuggingFace Hub ...", flush=True)
    # Heavy imports only happen on the actual download path so the
    # idempotency check doesn't require torch/transformers.
    import torch  # type: ignore
    from transformers import AutoImageProcessor, AutoModelForImageClassification  # type: ignore

    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.train(False)  # inference mode
    processor = AutoImageProcessor.from_pretrained(model_id)

    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    perm = _resolve_class_permutation(id2label)

    image_size = 224
    size_cfg = getattr(processor, "size", None)
    if isinstance(size_cfg, dict):
        image_size = int(size_cfg.get("height") or size_cfg.get("shortest_edge") or 224)
    image_mean = list(getattr(processor, "image_mean", [0.5, 0.5, 0.5]))
    image_std = list(getattr(processor, "image_std", [0.5, 0.5, 0.5]))

    onnx_path = out_dir / "emotion_classifier.onnx"
    print(f"[bootstrap] exporting ONNX to {onnx_path} (opset={opset}) ...", flush=True)
    dummy = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            str(onnx_path),
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=opset,
            do_constant_folding=True,
        )

    return onnx_path, id2label, perm, image_size, image_mean, image_std


def _write_manifest(
    out_dir: Path,
    onnx_path: Path,
    model_id: str,
    id2label: dict,
    perm: list[int],
    image_size: int,
    image_mean: list[float],
    image_std: list[float],
) -> Path:
    model_version = f"hf-{model_id.replace('/', '_')}-onnx"

    metadata = {
        "contract_schema_version": "2.0.0",
        "model_version": model_version,
        "source_model": model_id,
        "exported_at_utc": datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat(),
        "labels": list(CANONICAL),
        "source_labels": [id2label[i] for i in sorted(id2label)],
        "class_permutation": perm,
        "input_spec": {
            "image_mode": "RGB",
            "width": image_size,
            "height": image_size,
            "normalize": "imagenet",
            "image_mean": image_mean,
            "image_std": image_std,
            "layout": "NCHW",
            "value_range": [0.0, 1.0],
        },
        "output_spec": {"logits_dim": 7, "softmax_applied": False},
    }
    metadata_path = out_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "model_version": model_version,
                "source": model_id,
                "note": "metrics not computed on bootstrap; see HuggingFace model card",
                "labels": list(CANONICAL),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = {
        "artifact_contract_version": "2",
        "model_version": model_version,
        "input_spec": metadata["input_spec"],
        "labels": list(CANONICAL),
        "class_permutation": perm,
        "files": {
            "model": {
                "path": str(onnx_path),
                "sha256": _sha256_file(onnx_path),
                "size_bytes": onnx_path.stat().st_size,
            },
            "metadata": {
                "path": str(metadata_path),
                "sha256": _sha256_file(metadata_path),
                "size_bytes": metadata_path.stat().st_size,
            },
            "metrics": {
                "path": str(metrics_path),
                "sha256": _sha256_file(metrics_path),
                "size_bytes": metrics_path.stat().st_size,
            },
        },
    }
    manifest_path = out_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _sign_manifest(manifest_path: Path, signing_key: str) -> str:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files") or {}
    signature = hmac.new(
        signing_key.encode("utf-8"), _canonical_bytes(files), hashlib.sha256
    ).hexdigest()
    manifest["signature_hex"] = signature
    manifest["signature_algorithm"] = "HMAC-SHA256"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return signature


def _verify_loads_and_predicts(onnx_path: Path, image_size: int) -> None:
    print("[bootstrap] verifying onnxruntime load + inference ...", flush=True)
    import numpy as np  # type: ignore
    import onnxruntime as ort  # type: ignore

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    dummy = np.zeros((1, 3, image_size, image_size), dtype=np.float32)
    outputs = sess.run(None, {input_name: dummy})
    shape = tuple(outputs[0].shape)
    if shape != (1, 7):
        raise SystemExit(f"expected output shape (1,7), got {shape}")
    print(f"[bootstrap] OK — output shape {shape}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("FER_BOOTSTRAP_MODEL", DEFAULT_MODEL_ID))
    parser.add_argument("--output-dir", default="ml/artifacts")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--force", action="store_true", help="re-export even if cached artifact is valid")
    args = parser.parse_args()

    signing_key = os.getenv("FER_MODEL_SIGNING_KEY", "").strip()
    if not signing_key:
        print(
            "ERROR: FER_MODEL_SIGNING_KEY is not set.\n\n"
            "Generate one and export it before running bootstrap:\n"
            "    python -c 'import secrets;print(secrets.token_urlsafe(48))'\n"
            "    export FER_MODEL_SIGNING_KEY=<that-value>\n",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "emotion_classifier.onnx"
    manifest_path = out_dir / "artifact_manifest.json"

    if not args.force and _existing_artifact_is_valid(onnx_path, manifest_path, signing_key):
        print(
            f"[bootstrap] cached artifact at {onnx_path} already matches signed "
            f"manifest; skipping download. Pass --force to re-export.",
            flush=True,
        )
        return 0

    onnx_path, id2label, perm, size, mean, std = _download_and_export(
        args.model, out_dir, args.opset
    )
    manifest_path = _write_manifest(out_dir, onnx_path, args.model, id2label, perm, size, mean, std)
    signature = _sign_manifest(manifest_path, signing_key)
    _verify_loads_and_predicts(onnx_path, size)

    sha = _sha256_file(onnx_path)
    size_mb = onnx_path.stat().st_size / 1024 / 1024
    print()
    print("[bootstrap] SUCCESS")
    print(f"  artifact : {onnx_path}")
    print(f"  size     : {size_mb:.1f} MB")
    print(f"  sha256   : {sha}")
    print(f"  signature: {signature}")
    print(f"  manifest : {manifest_path}")
    print()
    print(
        "Start the server:\n"
        "  python manage.py runserver   (local dev)\n"
        "  docker compose up --build    (containerized)\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
