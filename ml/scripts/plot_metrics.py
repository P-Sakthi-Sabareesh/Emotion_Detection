#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot confusion matrix from FER metrics.")
    parser.add_argument("--metrics", default="ml/artifacts/metrics.json")
    parser.add_argument("--out", default="ml/artifacts/confusion_matrix.png")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    metrics_path = Path(args.metrics).resolve()
    if not metrics_path.exists():
        raise SystemExit(f"Missing metrics file: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    matrix_source = metrics.get("test_metrics", metrics)
    conf = np.array(matrix_source["confusion_matrix"], dtype=np.float64)
    labels = metrics["class_labels"]
    title = "FER-2013 Confusion Matrix"
    if args.normalize:
        row_sums = conf.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        conf = conf / row_sums
        title = "FER-2013 Confusion Matrix (Normalized)"

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(conf, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            text = f"{conf[i, j]:.2f}" if args.normalize else f"{int(conf[i, j])}"
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
