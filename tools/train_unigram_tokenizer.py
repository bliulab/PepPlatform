import os
import sys
import argparse
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from pepplatform.tokenizers import UnigramTokenizerTrainer
from pepplatform.unigram import (
    UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
    validate_unigram_max_raw_length,
    validate_unigram_vocab_size,
)


def resolve_path(root, path):
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


def load_fasta_sequences(path):
    sequences = []
    current = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
                continue
            current.append(line)
    if current:
        sequences.append("".join(current))
    return sequences


def main():
    parser = argparse.ArgumentParser(description="Train Unigram tokenizer")
    parser.add_argument("--data_dir", type=str, default="datasets/task_peptide_function/data")
    parser.add_argument(
        "--save_dir", type=str, default="datasets/task_peptide_function/unigram_tokenizers"
    )
    parser.add_argument(
        "--vocab_size",
        type=validate_unigram_vocab_size,
        default=1000,
    )
    parser.add_argument(
        "--max_raw_length",
        type=validate_unigram_max_raw_length,
        default=UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
    )
    parser.add_argument("--output_name", type=str, default="unigram_tokenizer.json")
    args = parser.parse_args()

    data_dir = resolve_path(project_root, args.data_dir)
    save_dir = resolve_path(project_root, args.save_dir)
    save_path = os.path.join(save_dir, args.output_name)

    train_fasta = os.path.join(data_dir, "train", "seqs.fasta")
    if not os.path.exists(train_fasta):
        raise SystemExit(f"Missing training FASTA: {train_fasta}")
    sequences = load_fasta_sequences(train_fasta)
    if not sequences:
        raise SystemExit(f"No sequences found in training FASTA: {train_fasta}")

    pbar = tqdm(total=3, desc="Building unigram tokenizer", leave=False)
    trainer = UnigramTokenizerTrainer(
        vocab_size=args.vocab_size,
        max_raw_length=args.max_raw_length,
    )
    pbar.set_postfix_str("training model")
    pbar.update(1)
    trainer.ensure_trained(
        sequences,
        save_path,
        sequence_role="sequence",
    )
    pbar.set_postfix_str("writing file")
    pbar.update(2)
    pbar.close()
    print(f"Saved unigram tokenizer to: {save_path}")


if __name__ == "__main__":
    main()
