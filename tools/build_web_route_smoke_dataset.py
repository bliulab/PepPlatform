#!/usr/bin/env python3
"""Build a tiny uploaded dataset for real Web route smoke tests."""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path


SPLITS = {
    "train": [
        ("TrainA", "AAAAAAAAAAAA", (1, 0)),
        ("TrainC", "CCCCCCCCCCCC", (0, 1)),
        ("TrainD", "DDDDDDDDDDDD", (1, 0)),
        ("TrainE", "EEEEEEEEEEEE", (0, 1)),
    ],
    "val": [
        ("ValG", "GGGGGGGGGGGG", (1, 0)),
        ("ValH", "HHHHHHHHHHHH", (0, 1)),
    ],
    "test": [
        ("TestI", "IIIIIIIIIIII", (1, 0)),
        ("TestK", "KKKKKKKKKKKK", (0, 1)),
    ],
}


def _fasta(records):
    return "".join(f">{sample_id}\n{sequence}\n" for sample_id, sequence, _labels in records)


def _labels(records):
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["class_a", "class_b"])
    writer.writerows(labels for _sample_id, _sequence, labels in records)
    return output.getvalue()


def build_archive(output_path):
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for split, records in SPLITS.items():
            archive.writestr(f"{split}/seqs.fasta", _fasta(records))
            archive.writestr(f"{split}/labels.csv", _labels(records))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = build_archive(args.output)
    print(output)


if __name__ == "__main__":
    main()
