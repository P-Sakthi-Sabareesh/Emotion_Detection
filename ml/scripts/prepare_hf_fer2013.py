#!/usr/bin/env python3
"""Materialize a FER-2013 dataset from HuggingFace into the folder layout
expected by ``ml/scripts/train_fer2013.py`` (``<root>/{train,test}/<emotion>/``).

We try a small ordered list of candidate HF datasets so that the first one
that is reachable wins.  The existing trainer only cares about class folder
names, so we translate whatever label mapping the dataset ships with to the
canonical set::

    angry, disgust, fear, happy, neutral, sad, surprise

Usage:
    python ml/scripts/prepare_hf_fer2013.py --output-dir data/raw --max-per-class 1500
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
from pathlib import Path

CANONICAL = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CANONICAL_SET = set(CANONICAL)

# In preference order.  Each tuple is (repo_id, (train_split, test_split)).
# `clip-benchmark/wds_fer2013` is the first reliable candidate on the HF Hub
# today — it ships the raw FER-2013 grid as a webdataset with `jpg` + `cls`.
CANDIDATES: list[tuple[str, tuple[str, str]]] = [
    ("clip-benchmark/wds_fer2013", ("train", "test")),
    ("Jeneral/fer-2013", ("train", "test")),
]

# FER-2013 Kaggle canonical label-index → name mapping.
# Used only when the dataset does not ship ClassLabel names.
FALLBACK_LABEL_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def _resolve_label_names(info) -> list[str]:
    features = getattr(info, "features", None) or {}
    feat = None
    for key in ("label", "labels", "cls", "class"):
        if features and key in features:
            feat = features[key]
            break
    if feat is not None and hasattr(feat, "names") and feat.names:
        names = [str(n).strip().lower() for n in feat.names]
        remap = {
            "anger": "angry",
            "happiness": "happy",
            "sadness": "sad",
            "disgusted": "disgust",
            "fearful": "fear",
            "surprised": "surprise",
        }
        normalized = [remap.get(n, n) for n in names]
        if all(n in CANONICAL_SET for n in normalized):
            return normalized
    return FALLBACK_LABEL_NAMES


def _open_pil(example):
    # HuggingFace datasets use several conventions: `image`, `img`, `png`,
    # or webdataset fields such as `jpg`.  Accept whichever exists.
    for key in ("image", "img", "jpg", "png", "webp"):
        img = example.get(key)
        if img is None:
            continue
        if hasattr(img, "convert"):
            return img
        try:
            from PIL import Image

            if isinstance(img, dict) and img.get("bytes") is not None:
                return Image.open(io.BytesIO(img["bytes"]))
            if isinstance(img, (bytes, bytearray)):
                return Image.open(io.BytesIO(img))
        except Exception:
            continue
    return None


def _extract_label_idx(example):
    for key in ("label", "labels", "cls", "class"):
        if key in example and example[key] is not None:
            return example[key]
    return None


def materialize(
    repo_id: str,
    split_names: tuple[str, str],
    out_root: Path,
    max_per_class: int | None,
) -> int:
    from datasets import load_dataset  # type: ignore

    print(f"[data] downloading {repo_id} ...", flush=True)
    ds = load_dataset(repo_id)

    saved = 0
    for dest_split, src_split in zip(("train", "test"), split_names):
        if src_split not in ds:
            candidate_splits = list(ds.keys())
            if not candidate_splits:
                raise RuntimeError(f"dataset {repo_id} has no splits")
            src_split = candidate_splits[0] if dest_split == "train" else candidate_splits[-1]
        split = ds[src_split]
        label_names = _resolve_label_names(split.info)
        print(
            f"[data] {repo_id}:{src_split} -> {dest_split} (labels={label_names})",
            flush=True,
        )

        per_class_counts: dict[str, int] = {c: 0 for c in CANONICAL}
        for example in split:
            label_idx = _extract_label_idx(example)
            if label_idx is None:
                continue
            try:
                label_name = label_names[int(label_idx)]
            except (IndexError, ValueError, TypeError):
                continue
            if label_name not in CANONICAL_SET:
                continue
            if max_per_class and per_class_counts[label_name] >= max_per_class:
                continue
            img = _open_pil(example)
            if img is None:
                continue
            class_dir = out_root / dest_split / label_name
            class_dir.mkdir(parents=True, exist_ok=True)
            filename = class_dir / f"{per_class_counts[label_name]:05d}.png"
            img.convert("L").resize((48, 48)).save(filename, format="PNG")
            per_class_counts[label_name] += 1
            saved += 1
        print(
            f"[data] {repo_id}:{src_split} wrote "
            f"{sum(per_class_counts.values())} images "
            f"(per-class: {per_class_counts})",
            flush=True,
        )
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--max-per-class", type=int, default=1500)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.output_dir).resolve()
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    last_error: Exception | None = None
    for repo_id, split_names in CANDIDATES:
        try:
            total = materialize(repo_id, split_names, out_root, args.max_per_class)
            print(f"[data] done: {total} images from {repo_id}", flush=True)
            return 0
        except Exception as exc:
            print(f"[data] candidate {repo_id} failed: {type(exc).__name__}: {exc}", flush=True)
            last_error = exc
    print(
        f"[data] ERROR: no candidate dataset succeeded: {last_error}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
