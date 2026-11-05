import os
import sys
import re
import csv
import argparse
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pepplatform.hlai_input import hla_allele
from pepplatform.task_compat import LEGACY_HLAI_COLUMN

COL_RE = re.compile(r"^(\d+)_([A-Z])$")


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def to_int01(x: str) -> int:
    try:
        return int(float(str(x).strip()))
    except Exception:
        return 0


def main(argv=None, *, defaults_root=None, output_column="hla"):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        default="datasets/task_peptide_HLAI_interaction/pseudoseqs.csv",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="datasets/task_peptide_HLAI_interaction/pseudoseqs_decoded.csv",
    )
    if defaults_root:
        parser.set_defaults(
            input_csv=defaults_root + "/pseudoseqs.csv",
            output_csv=defaults_root + "/pseudoseqs_decoded.csv",
        )
    args = parser.parse_args(argv)

    input_csv = resolve_path(args.input_csv)
    output_csv = resolve_path(args.output_csv)

    if not os.path.exists(input_csv):
        raise SystemExit(f"输入文件不存在: {input_csv}")

    with open(input_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not ({"hla", LEGACY_HLAI_COLUMN} & set(reader.fieldnames)):
            raise SystemExit("输入CSV缺少 hla 列")

        pos_to_cols = defaultdict(list)
        for c in reader.fieldnames:
            m = COL_RE.match(c)
            if not m:
                continue
            pos = int(m.group(1))
            aa = m.group(2)
            pos_to_cols[pos].append((aa, c))

        if not pos_to_cols:
            raise SystemExit("未检测到形如 2_A 的one-hot列")

        positions = sorted(pos_to_cols.keys())
        for p in positions:
            pos_to_cols[p].sort(key=lambda x: x[0])

        rows_out = []
        total = 0
        ok_cnt = 0
        amb_cnt = 0

        for row in reader:
            total += 1
            hla = hla_allele(row)
            seq_chars = []
            bad = False

            for p in positions:
                hits = []
                for aa, col in pos_to_cols[p]:
                    if to_int01(row.get(col, "0")) == 1:
                        hits.append(aa)

                if len(hits) == 1:
                    seq_chars.append(hits[0])
                elif len(hits) == 0:
                    seq_chars.append("X")
                    bad = True
                else:
                    seq_chars.append(sorted(hits)[0])
                    bad = True

            pseudoseq = "".join(seq_chars)
            status = "ok" if not bad else "ambiguous"
            if status == "ok":
                ok_cnt += 1
            else:
                amb_cnt += 1

            rows_out.append({output_column: hla, "pseudoseq": pseudoseq, "status": status})

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[output_column, "pseudoseq", "status"])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"输入: {input_csv}")
    print(f"输出: {output_csv}")
    print(f"总数: {total}")
    print(f"ok: {ok_cnt}")
    print(f"ambiguous: {amb_cnt}")


if __name__ == "__main__":
    main()
