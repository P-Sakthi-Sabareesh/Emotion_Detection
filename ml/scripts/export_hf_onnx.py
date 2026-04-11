#!/usr/bin/env python3
"""Export a HuggingFace FER image-classification model to ONNX.

Produces ``ml/artifacts/emotion_classifier.onnx`` plus a signed-ready manifest
that the runtime loader can verify.  Captures the model's native label order
and a permutation index into our canonical class order so the runtime can
remap logits without re-training.

Default model: ``dima806/facial_emotions_image_detection`` — a ViT fine-tuned
on FER-2013 that lands around 90% accuracy, vs ~36% for the sklearn baseline.

Usage:
    python ml/scripts/export_hf_onnx.py --model dima806/facial_emotions_image_detection
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


CANONICAL = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_label(label: str) -> str:
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


def resolve_class_permutation(id2label):
    lookup: dict[str, int] = {}
    for idx, label in id2label.items():
        lookup[canonicalize_label(str(label))] = int(idx)
    missing = [c for c in CANONICAL if c not in lookup]
    if missing:
        raise SystemExit(
            f"source model is missing canonical FER classes: {missing}. "
            f"Available labels: {list(id2label.values())}"
        )
    return [lookup[c] for c in CANONICAL]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="dima806/facial_emotions_image_detection")
    parser.add_argument("--output-dir", default="ml/artifacts")
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    print(f"[export] downloading {args.model} ...", flush=True)
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    model = AutoModelForImageClassification.from_pretrained(args.model)
    model.train(False)  # inference mode; equivalent to the stdlib eval method name
    processor = AutoImageProcessor.from_pretrained(args.model)

    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    print(f"[export] source id2label: {id2label}", flush=True)

    perm = resolve_class_permutation(id2label)
    print(f"[export] canonical permutation: {perm} (applied at runtime)", flush=True)

    image_size = 224
    size_cfg = getattr(processor, "size", None)
    if isinstance(size_cfg, dict):
        image_size = int(size_cfg.get("height") or size_cfg.get("shortest_edge") or 224)
    image_mean = list(getattr(processor, "image_mean", [0.485, 0.456, 0.406]))
    image_std = list(getattr(processor, "image_std", [0.229, 0.224, 0.225]))

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "emotion_classifier.onnx"

    dummy = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    print(f"[export] exporting to {onnx_path} (opset={args.opset}) ...", flush=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            str(onnx_path),
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "logits": {0: "batch"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
        )

    import onnxruntime as ort  # type: ignore

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = sess.run(None, {"pixel_values": dummy.numpy()})
    out_shape = outputs[0].shape
    print(f"[export] onnx output shape: {out_shape}", flush=True)
    if out_shape[-1] != 7:
        raise SystemExit(f"expected 7 output classes, got shape {out_shape}")

    model_version = args.model_version or f"hf-{args.model.replace('/', '_')}-onnx"

    metadata = {
        "contract_schema_version": "2.0.0",
        "model_version": model_version,
        "source_model": args.model,
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
                "source": args.model,
                "note": "metrics not computed on export; see model card on HuggingFace",
                "labels": list(CANONICAL),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    stale_joblib = out_dir / "emotion_classifier.joblib"
    if stale_joblib.exists():
        stale_joblib.unlink()
        print(f"[export] removed stale {stale_joblib.name}", flush=True)

    manifest = {
        "artifact_contract_version": "2",
        "model_version": model_version,
        "input_spec": metadata["input_spec"],
        "labels": list(CANONICAL),
        "class_permutation": perm,
        "files": {
            "model": {
                "path": str(onnx_path),
                "sha256": sha256_file(onnx_path),
                "size_bytes": onnx_path.stat().st_size,
            },
            "metadata": {
                "path": str(metadata_path),
                "sha256": sha256_file(metadata_path),
                "size_bytes": metadata_path.stat().st_size,
            },
            "metrics": {
                "path": str(metrics_path),
                "sha256": sha256_file(metrics_path),
                "size_bytes": metrics_path.stat().st_size,
            },
        },
    }
    manifest_path = out_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[export] wrote manifest {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
