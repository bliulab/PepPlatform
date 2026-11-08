import pytest
import torch

from pepplatform.dataset import PeptideDataset


class RecordingFeaturizer:
    def __init__(self):
        self.calls = []

    def encode(self, sequence, sample_id=None):
        self.calls.append((sequence, sample_id))
        return [1, 2, 0]


def _write_dataset(tmp_path):
    fasta = tmp_path / "seqs.fasta"
    labels = tmp_path / "labels.csv"
    fasta.write_text(">sample-a\nACD\n", encoding="utf-8")
    labels.write_text("label\n1\n", encoding="utf-8")
    return fasta, labels


def test_peptide_dataset_requires_an_explicit_featurizer(tmp_path):
    fasta, labels = _write_dataset(tmp_path)

    with pytest.raises(TypeError, match="featurizer"):
        PeptideDataset(fasta, labels)
    with pytest.raises(ValueError, match="explicit featurizer"):
        PeptideDataset(fasta, labels, featurizer=None)


def test_peptide_dataset_forwards_sequence_and_sample_id(tmp_path):
    fasta, labels = _write_dataset(tmp_path)
    featurizer = RecordingFeaturizer()
    dataset = PeptideDataset(fasta, labels, featurizer=featurizer)

    features, label = dataset[0]

    assert featurizer.calls == [("ACD", "sample-a")]
    assert torch.equal(features, torch.tensor([1, 2, 0], dtype=torch.long))
    assert torch.equal(label, torch.tensor([1.0]))
