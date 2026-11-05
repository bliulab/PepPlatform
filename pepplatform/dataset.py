import csv

import torch
from torch.utils.data import Dataset

# IDs: 0 = padding, 1-20 = canonical residues, 21 = unknown.
# Ambiguous and noncanonical residues retain the aliases below.
AA_VOCAB = {
    "<PAD>": 0,
    "A": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "Y": 20,
    "X": 21,
    "B": 3,
    "Z": 4,
    "J": 10,
    "U": 2,
    "O": 9,
}


class PeptideDataset(Dataset):
    def __init__(self, seq_fasta_path, label_csv_path, *, featurizer):
        if featurizer is None:
            raise ValueError("PeptideDataset requires an explicit featurizer")
        self.sequence_ids, self.sequences = self._load_sequences(seq_fasta_path)
        self.label_names, self.labels = self._load_labels(label_csv_path)
        self.featurizer = featurizer

        assert len(self.sequences) == len(self.labels), "Sequences and labels count mismatch"
        self._unigram_sampling = None
        self._sampling_epoch = 0

    def enable_unigram_sampling(
        self,
        *,
        seed,
        nbest_size,
        alpha,
        role="sequence",
    ):
        if not bool(getattr(self.featurizer, "supports_unigram_sampling", False)):
            raise ValueError("dataset featurizer does not support Unigram sampling")
        self._unigram_sampling = {
            "seed": int(seed),
            "nbest_size": int(nbest_size),
            "alpha": float(alpha),
            "role": str(role or "sequence"),
        }

    def set_epoch(self, epoch):
        self._sampling_epoch = int(epoch)

    def _load_sequences(self, path):
        ids = []
        sequences = []
        current = []
        current_id = None
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current:
                        ids.append(current_id if current_id is not None else str(len(ids)))
                        sequences.append("".join(current))
                    current_id = line[1:].strip()
                    current = []
                else:
                    current.append(line)
            if current:
                ids.append(current_id if current_id is not None else str(len(ids)))
                sequences.append("".join(current))
        return ids, sequences

    def _load_labels(self, path):
        with open(path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
        header = rows[0]
        labels = [[float(x) for x in row] for row in rows[1:]]
        return header, labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_id = self.sequence_ids[idx] if idx < len(self.sequence_ids) else None
        label = self.labels[idx]

        if self._unigram_sampling is not None:
            encoded = self.featurizer.encode_sampled(
                seq,
                seed=self._unigram_sampling["seed"],
                epoch=self._sampling_epoch,
                sample_id=seq_id if seq_id is not None else str(idx),
                role=self._unigram_sampling["role"],
                nbest_size=self._unigram_sampling["nbest_size"],
                alpha=self._unigram_sampling["alpha"],
            )
        else:
            try:
                encoded = self.featurizer.encode(seq, sample_id=seq_id)
            except TypeError:
                encoded = self.featurizer.encode(seq)
        if isinstance(encoded, torch.Tensor) or (
            hasattr(encoded, "features")
            and hasattr(encoded, "mask")
            and isinstance(encoded.features, torch.Tensor)
            and isinstance(encoded.mask, torch.Tensor)
        ):
            input_features = encoded
        else:
            input_features = torch.tensor(encoded, dtype=torch.long)

        label_tensor = torch.tensor(label, dtype=torch.float)

        return input_features, label_tensor

    def get_num_classes(self):
        return len(self.label_names)
