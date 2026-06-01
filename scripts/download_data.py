"""
Download datasets and model weights to data/ before submitting training jobs.
Run once from the project root:
    python scripts/download_data.py                      # STL-10 only
    python scripts/download_data.py --datasets stl10 sst2
"""

import argparse
from pathlib import Path
from torchvision import datasets

VISION_DATASETS = {
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

TEXT_DATASETS = {
    "sst2": ("nyu-mll/glue", "sst2", "distilbert-base-uncased"),
}


def download_vision(dataset: str, data_root: Path):
    out = data_root / dataset
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n--- {dataset.upper()} → {out} ---")
    for split, desc in VISION_DATASETS[dataset]:
        print(f"  {desc}...")
        if dataset == "stl10":
            datasets.STL10(out, split=split, download=True)
        elif dataset == "cifar10":
            train = split == "train"
            datasets.CIFAR10(out, train=train, download=True)
    print("  Done.")


def download_text(dataset: str, data_root: Path):
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModel

    hf_name, hf_config, model_name = TEXT_DATASETS[dataset]
    out = data_root / dataset
    hf_cache = str(out / "hf_cache")
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n--- {dataset.upper()} → {out} ---")
    print(f"  Dataset ({hf_name}/{hf_config})...")
    load_dataset(hf_name, hf_config, cache_dir=hf_cache)

    print(f"  Tokenizer ({model_name})...")
    AutoTokenizer.from_pretrained(model_name, cache_dir=hf_cache)

    print(f"  Model weights ({model_name})...")
    AutoModel.from_pretrained(model_name, cache_dir=hf_cache)

    print("  Done.")


def main():
    all_datasets = list(VISION_DATASETS) + list(TEXT_DATASETS)
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["stl10"],
                        choices=all_datasets)
    parser.add_argument("--data_dir", default="data")
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    for ds in args.datasets:
        if ds in VISION_DATASETS:
            download_vision(ds, data_root)
        else:
            download_text(ds, data_root)
    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
