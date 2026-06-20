#!/usr/bin/env python3
import argparse
import random

from mn5_segmentation.data import find_dataset_source, list_pairs
from mn5_segmentation.utils import write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Create reproducible Kvasir-SEG train/val/test splits.")
    parser.add_argument("--data-root", default="data", help="Dataset folder or location containing kvasir-seg.zip.")
    parser.add_argument("--output", default="outputs/splits/kvasir_split.json", help="Output split JSON path.")
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    source = find_dataset_source(args.data_root)
    keys = list_pairs(source)
    rng = random.Random(args.seed)
    rng.shuffle(keys)

    train_end = int(len(keys) * args.train_ratio)
    val_end = train_end + int(len(keys) * args.val_ratio)
    split = {
        "seed": args.seed,
        "data_root": args.data_root,
        "dataset": {"kind": source["kind"], "path": source["path"]},
        "counts": {
            "total": len(keys),
            "train": train_end,
            "val": val_end - train_end,
            "test": len(keys) - val_end,
        },
        "splits": {
            "train": keys[:train_end],
            "val": keys[train_end:val_end],
            "test": keys[val_end:],
        },
    }
    write_json(split, args.output)
    print(f"Wrote {args.output}")
    print(split["counts"])


if __name__ == "__main__":
    main()

