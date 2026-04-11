#!/usr/bin/env python3
"""Benchmark multiple ONNX artifacts against the same sample set.

Reports top-1 class, confidence, and delta vs the first (reference) model.
Meant for A/B/C comparisons between FP32, dynamic-quantized, and
static-quantized models.  Loads each model in the same process via
onnxruntime.InferenceSession so there is no Docker image cache to get in
the way.

Usage:
    python ml/scripts/bench_onnx_models.py \\
        fp32=ml/artifacts/emotion_classifier.onnx \\
        q8dyn=ml/artifacts/emotion_classifier.q8dyn.onnx
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort  # type: ignore
from PIL import Image

CANONICAL = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def load_manifest(artifacts_dir: Path) -> dict:
    mp = artifacts_dir / "artifact_manifest.json"
    if not mp.exists():
        raise SystemExit(f"manifest not found: {mp}")
    return json.loads(mp.read_text())


def preprocess(image_bytes: bytes, mean, std, size: int) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as im:
        rgb = im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    arr = np.transpose(arr, (2, 0, 1))
    return arr[None, ...].astype(np.float32)


def predict(sess: ort.InferenceSession, x: np.ndarray, perm: list[int]) -> np.ndarray:
    input_name = sess.get_inputs()[0].name
    logits = sess.run(None, {input_name: x})[0][0]
    logits = logits[np.asarray(perm, dtype=np.int64)]
    shifted = logits - float(np.max(logits))
    e = np.exp(shifted)
    return (e / float(np.sum(e))).astype(np.float32)


def run_bench(models: dict[str, Path], samples: list[Path], manifest: dict) -> dict:
    spec = manifest["input_spec"]
    mean = spec["image_mean"]
    std = spec["image_std"]
    size = int(spec["width"])
    perm = manifest["class_permutation"]

    # Preload all images
    images: dict[str, bytes] = {}
    for p in samples:
        images[p.name] = p.read_bytes()

    # Load all models
    sessions: dict[str, ort.InferenceSession] = {}
    for name, path in models.items():
        if not path.exists():
            raise SystemExit(f"model artifact missing: {path}")
        print(f"[bench] loading {name} from {path.name} ({path.stat().st_size/1024/1024:.1f} MB)", flush=True)
        sessions[name] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    results: dict[str, dict[str, dict]] = {}
    for name, sess in sessions.items():
        per_sample: dict[str, dict] = {}
        for sample_name, raw in images.items():
            x = preprocess(raw, mean, std, size)
            # warmup
            _ = predict(sess, x, perm)
            t0 = time.perf_counter()
            probs = predict(sess, x, perm)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            scores = {CANONICAL[i]: float(probs[i]) for i in range(len(CANONICAL))}
            top_idx = int(np.argmax(probs))
            per_sample[sample_name] = {
                "top": CANONICAL[top_idx],
                "confidence": float(probs[top_idx]),
                "latency_ms": round(dt_ms, 1),
                "scores": scores,
            }
        results[name] = per_sample

    return results


def render_table(results: dict[str, dict[str, dict]]) -> None:
    model_names = list(results.keys())
    if not model_names:
        return
    baseline = model_names[0]
    sample_names = sorted(results[baseline].keys())

    header = ["sample"] + [f"{n}:class" for n in model_names] + [f"{n}:conf" for n in model_names] + ["match?"]
    widths = [max(len(h), 12) for h in header]
    print()
    print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
    print("-+-".join("-" * w for w in widths))

    matches = 0
    total = 0
    max_conf_drop = 0.0
    for s in sample_names:
        row = [s]
        classes = []
        confs = []
        for n in model_names:
            e = results[n][s]
            classes.append(e["top"])
            confs.append(e["confidence"])
        match_ok = all(c == classes[0] for c in classes)
        if match_ok:
            matches += 1
        total += 1
        for c in classes:
            row.append(c)
        for c in confs:
            row.append(f"{c:.2%}")
        row.append("OK" if match_ok else "FAIL")
        print(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))

        # confidence drop vs baseline
        for i, n in enumerate(model_names[1:], start=1):
            if classes[i] == classes[0]:
                drop = confs[0] - confs[i]
                if drop > max_conf_drop:
                    max_conf_drop = drop

    print()
    print(f"[bench] {matches}/{total} samples top-1 match baseline ({model_names[0]})")
    print(f"[bench] max confidence drop on same-class predictions: {max_conf_drop:.2%}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specs", nargs="+", help="name=path pairs")
    parser.add_argument("--samples-dir", default="static/samples")
    parser.add_argument("--artifacts-dir", default="ml/artifacts")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    models: dict[str, Path] = {}
    for spec in args.specs:
        if "=" not in spec:
            raise SystemExit(f"invalid spec (expected name=path): {spec}")
        name, path = spec.split("=", 1)
        models[name] = Path(path).resolve()

    samples_dir = Path(args.samples_dir).resolve()
    samples = sorted(samples_dir.glob("sample-*.jpg"))
    if not samples:
        raise SystemExit(f"no samples found in {samples_dir}")

    manifest = load_manifest(Path(args.artifacts_dir).resolve())

    results = run_bench(models, samples, manifest)
    render_table(results)

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"[bench] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
