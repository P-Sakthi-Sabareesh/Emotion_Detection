#!/usr/bin/env python3
"""ONNX quantization for the FER ViT artifact.

Supports two strategies — always picked deliberately, never silently:

* ``dynamic``   — ``quantize_dynamic`` with the shape-inference
  preprocessing step the ONNX Runtime docs explicitly require for
  transformer models. Cheap to run, no calibration data needed.
* ``static``    — ``quantize_static`` with a real calibration dataset
  of FER-2013 faces. Significantly better accuracy preservation for
  ViT attention layers. Requires a calibration corpus.

The script never touches the input file — you always write to an
explicit output path so a bad quantization cannot clobber the FP32
baseline.

Usage:
    python ml/scripts/quantize_onnx.py \\
        --input  ml/artifacts/emotion_classifier.onnx \\
        --output ml/artifacts/emotion_classifier.q8dyn.onnx \\
        --mode   dynamic

    python ml/scripts/quantize_onnx.py \\
        --input  ml/artifacts/emotion_classifier.onnx \\
        --output ml/artifacts/emotion_classifier.q8stat.onnx \\
        --mode   static \\
        --calibration-dir data/raw/train \\
        --calibration-samples 128
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _load_input_spec(artifacts_dir: Path) -> dict:
    manifest = artifacts_dir / "artifact_manifest.json"
    if not manifest.exists():
        raise SystemExit(f"manifest missing, cannot derive input_spec: {manifest}")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    spec = data.get("input_spec") or {}
    if not spec:
        raise SystemExit("manifest.input_spec is empty")
    return spec


def run_preprocess(src: Path, dst: Path) -> None:
    """Run onnxruntime shape inference + graph cleanup before quantization.

    The ORT quantizer prints a warning that this step is required for
    transformer models; skipping it is the bug that produced a class flip
    on one of our FER samples when I first shipped dynamic INT8.
    """
    from onnxruntime.quantization.shape_inference import quant_pre_process  # type: ignore

    print(f"[quantize] shape_inference.quant_pre_process: {src.name} -> {dst.name}", flush=True)
    quant_pre_process(
        input_model_path=str(src),
        output_model_path=str(dst),
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=False,
        auto_merge=False,
        int_max=2**31 - 1,
        guess_output_rank=False,
        verbose=1,
        save_as_external_data=False,
    )


def run_dynamic(preprocessed: Path, dst: Path, weight_type_name: str) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore

    weight_type = QuantType.QInt8 if weight_type_name == "QInt8" else QuantType.QUInt8
    # Restrict to MatMul/Gemm ops: the ViT patch embedding is a Conv and
    # per-channel ConvInteger is not implemented on the default CPU EP.
    # Leaving the patch embedding in FP32 is also the standard
    # recommendation for preserving ViT accuracy under dynamic INT8.
    op_types = ["MatMul", "Gemm"]
    print(
        f"[quantize] quantize_dynamic weight_type={weight_type_name} "
        f"op_types={op_types}",
        flush=True,
    )
    quantize_dynamic(
        model_input=str(preprocessed),
        model_output=str(dst),
        weight_type=weight_type,
        op_types_to_quantize=op_types,
        per_channel=False,
        reduce_range=False,
    )


class FerCalibrationReader:
    """CalibrationDataReader that yields preprocessed face tensors."""

    def __init__(self, root: Path, input_name: str, spec: dict, max_samples: int, seed: int = 42):
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        self._np = np
        self._Image = Image
        self._input_name = input_name
        self._mean = np.asarray(spec.get("image_mean") or [0.5, 0.5, 0.5], dtype=np.float32)
        self._std = np.asarray(spec.get("image_std") or [0.5, 0.5, 0.5], dtype=np.float32)
        self._size = int(spec.get("width") or 224)

        if not root.exists():
            raise SystemExit(f"calibration dir not found: {root}")
        all_images = [p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not all_images:
            raise SystemExit(f"no calibration images under {root}")
        rng = random.Random(seed)
        rng.shuffle(all_images)
        self._paths = all_images[:max_samples]
        self._iter = iter(self._paths)
        print(f"[calibration] using {len(self._paths)} images from {root}", flush=True)

    def _load(self, path: Path):
        with self._Image.open(path) as im:
            rgb = im.convert("RGB").resize((self._size, self._size), self._Image.Resampling.BILINEAR)
        arr = self._np.asarray(rgb, dtype=self._np.float32) / 255.0
        arr = (arr - self._mean) / self._std
        arr = self._np.transpose(arr, (2, 0, 1))
        return arr[None, ...].astype(self._np.float32)

    def get_next(self):
        try:
            path = next(self._iter)
        except StopIteration:
            return None
        return {self._input_name: self._load(path)}

    def rewind(self):
        self._iter = iter(self._paths)


def run_static(
    preprocessed: Path,
    dst: Path,
    weight_type_name: str,
    calibration_dir: Path,
    max_samples: int,
    input_name: str,
    spec: dict,
) -> None:
    from onnxruntime.quantization import (  # type: ignore
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    reader = FerCalibrationReader(calibration_dir, input_name, spec, max_samples)
    weight_type = QuantType.QInt8 if weight_type_name == "QInt8" else QuantType.QUInt8

    print(
        f"[quantize] quantize_static weight_type={weight_type_name} "
        f"calibration={max_samples} samples",
        flush=True,
    )
    quantize_static(
        model_input=str(preprocessed),
        model_output=str(dst),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        calibrate_method=CalibrationMethod.MinMax,
        weight_type=weight_type,
        activation_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )


def _onnx_first_input_name(path: Path) -> str:
    import onnxruntime as ort  # type: ignore

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess.get_inputs()[0].name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["dynamic", "static"], required=True)
    parser.add_argument("--weight-type", choices=["QInt8", "QUInt8"], default="QInt8")
    parser.add_argument("--calibration-dir", default="data/raw/train")
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--artifacts-dir", default="ml/artifacts")
    parser.add_argument("--keep-preprocessed", action="store_true")
    args = parser.parse_args()

    src = Path(args.input).resolve()
    dst = Path(args.output).resolve()
    if not src.exists():
        raise SystemExit(f"input not found: {src}")
    if dst.resolve() == src.resolve():
        raise SystemExit("refusing to quantize in-place; pick a distinct --output path")
    dst.parent.mkdir(parents=True, exist_ok=True)

    pre_path = dst.with_suffix(dst.suffix + ".prep")

    src_size = src.stat().st_size
    print(f"[quantize] input: {src.name} ({src_size/1024/1024:.1f} MB)", flush=True)
    print(f"[quantize] output: {dst.name} (mode={args.mode})", flush=True)

    run_preprocess(src, pre_path)

    try:
        if args.mode == "dynamic":
            run_dynamic(pre_path, dst, args.weight_type)
        else:
            spec = _load_input_spec(Path(args.artifacts_dir).resolve())
            input_name = _onnx_first_input_name(pre_path)
            run_static(
                pre_path,
                dst,
                args.weight_type,
                Path(args.calibration_dir).resolve(),
                args.calibration_samples,
                input_name,
                spec,
            )
    finally:
        if not args.keep_preprocessed and pre_path.exists():
            pre_path.unlink()

    if not dst.exists():
        raise SystemExit("quantization produced no output")

    dst_size = dst.stat().st_size
    ratio = dst_size / src_size if src_size else 0.0
    print(
        f"[quantize] {args.mode} done: {dst.name} = "
        f"{dst_size/1024/1024:.1f} MB ({ratio:.1%} of {src.name})",
        flush=True,
    )
    if dst_size > 100 * 1024 * 1024:
        print(
            f"[quantize] WARNING: {dst_size/1024/1024:.1f} MB exceeds GitHub's 100 MB hard limit. "
            "Use Git LFS or a release asset.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
