#!/usr/bin/env python3
import argparse

# subprocess is used to invoke the vetted Kaggle CLI below; only an argv list
# resolved via shutil.which is ever passed, never a shell string.
import subprocess  # nosec B404
import sys
import zipfile
from pathlib import Path
import shutil


def run() -> None:
    parser = argparse.ArgumentParser(description="Download FER-2013 from Kaggle.")
    parser.add_argument("--dataset", default="msambare/fer2013")
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    venv_kaggle = Path(sys.executable).parent / "kaggle"
    kaggle_bin = str(venv_kaggle) if venv_kaggle.exists() else shutil.which("kaggle")
    if not kaggle_bin:
        raise SystemExit("Kaggle CLI executable not found. Install with: pip install kaggle")

    print(f"Downloading {args.dataset} into {output_dir}...")
    # shell=False (default) is the secure form; bandit B603 flags any
    # subprocess.run even when it is being called correctly with a list
    # argv. kaggle_bin is resolved via shutil.which, not user input.
    subprocess.run(  # nosec B603
        [
            kaggle_bin,
            "datasets",
            "download",
            "-d",
            args.dataset,
            "-p",
            str(output_dir),
            "--force",
        ],
        check=True,
    )

    zip_path = output_dir / "fer2013.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Extracted {zip_path}")
    else:
        print("Download finished; expected fer2013.zip not found. Check Kaggle output.")


if __name__ == "__main__":
    run()
