"""
Download datasets to data/ before submitting training jobs.
Run once from the project root:
    python scripts/download_data.py
"""

import argparse
from pathlib import Path
from torchvision import datasets

DATASETS = {
    "stl10": [
        ("unlabeled", "STL-10 unlabeled (100k, used for QIT training)"),
        ("train",     "STL-10 labeled train (5k, used for probe)"),
        ("test",      "STL-10 labeled test (8k, used for probe)"),
    ],
    "cifar10": [
        ("train", "CIFAR-10 train"),
        ("test",  "CIFAR-10 test"),
    ],
}


def download(dataset: str, data_root: Path):
    splits = DATASETS[dataset]
    out = data_root / dataset
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n--- {dataset.upper()} → {out} ---")
    for split, desc in splits:
        print(f"  {desc}...")
        if dataset == "stl10":
            datasets.STL10(out, split=split, download=True)
        elif dataset == "cifar10":
            train = split == "train"
            datasets.CIFAR10(out, train=train, download=True)
    print(f"  Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["stl10"],
                        choices=list(DATASETS), help="Datasets to download.")
    parser.add_argument("--data_dir", default="data",
                        help="Root data directory (default: data/).")
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    for ds in args.datasets:
        download(ds, data_root)
    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
