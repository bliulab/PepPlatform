import csv
import hashlib
import random
import subprocess
import sys
from pathlib import Path

import pytest

from pepplatform.mhci_input import MHCIInputError, require_mhci_partner_sequence


ALLELE = "HLA-A*02:01"
SEQUENCE = "ALVYSAGKVAHSVDLVRHYVLATTVLGAAR"
READERS = ("train", "profile", "lengths", "web", "legacy")


@pytest.fixture(params=READERS)
def reader(request, tmp_path, monkeypatch):
    def read(mapping, *, explicit=None):
        path = tmp_path / "data/train/pairs.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            fields = ["sample_id", "mhc", "pep", "tgt"] + (
                ["pair_pseudo"] if explicit is not None else []
            )
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            row = dict(sample_id="s1", mhc=ALLELE, pep="SIINFEKL", tgt=1)
            if explicit is not None:
                row["pair_pseudo"] = explicit
            writer.writerow(row)
        pseudo_path = tmp_path / "pseudoseqs_decoded.csv"
        with pseudo_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["mhc", "pseudoseq", "status"])
            writer.writerows((key, value, "ok") for key, value in mapping.items())
        with path.open() as handle:
            row = next(csv.DictReader(handle))
        if request.param == "train":
            from scripts import train_pair_pipeline as p

            return p.read_ppi_csv(path, mhc_pseudo_map=mapping, pseudoseq_path=pseudo_path)[0][2]
        if request.param == "profile":
            from tools import build_ppi_profile_cache as p

            return p.read_ppi_csv(path, mhc_pseudo_map=mapping, pseudoseq_path=pseudo_path)[0][2]
        if request.param == "lengths":
            from pepplatform.sequence_lengths import pair_sequences

            return pair_sequences(
                row, mhci_pseudosequences=mapping, source_path=path, row_number=1
            )[1]
        if request.param == "web":
            from pepplatform.web.datasets import _pair_partner

            return _pair_partner(row, mapping, source_path=path, row_number=1)
        from pepplatform.web.inference import ModelService

        # Legacy bundles resolve mapping files relative to task_root.
        metadata = {
            "args": {"task_type": "pep_mhci", "task_root": str(tmp_path), "train_csv": str(path)}
        }
        return ModelService._legacy_pair_rows(metadata, "train_csv")[1][0][2]

    return request.param, read


@pytest.mark.parametrize("mapping", [{}, {"HLA-B*07:02": SEQUENCE}])
def test_missing_mapping_never_encodes_allele(reader, mapping):
    _, read = reader
    # InferenceUnsupportedError also preserves the actionable diagnostic.
    with pytest.raises(
        Exception, match=r"pairs\.csv:row 1.*HLA-A\*02:01.*no usable pseudosequence"
    ):
        read(mapping)


def test_complete_mapping_preserves_sequence(reader):
    _, read = reader
    assert read({ALLELE: SEQUENCE}) == SEQUENCE


def test_explicit_sequence_does_not_require_mapping(reader):
    _, read = reader
    assert read({}, explicit=SEQUENCE) == SEQUENCE


def test_explicit_allele_identifier_is_rejected(reader):
    _, read = reader
    with pytest.raises(Exception, match="invalid partner sequence"):
        read({}, explicit=ALLELE)


def test_existing_field_precedence_is_preserved(reader):
    name, read = reader
    # Web sequence views historically prefer inline sequences; other readers
    # historically prefer a matching mapping over pair_pseudo.
    expected = "C" * 30 if name == "web" or name == "semantic" else SEQUENCE
    assert read({ALLELE: SEQUENCE}, explicit="C" * 30) == expected


@pytest.mark.parametrize(
    "content",
    [None, "mhc,pseudoseq,status\n", f"mhc,pseudoseq,status\n{ALLELE},{SEQUENCE},missing\n"],
)
def test_missing_empty_or_non_ok_mapping_is_rejected(tmp_path, content):
    from scripts import train_pair_pipeline as p

    path = tmp_path / "pairs.csv"
    path.write_text(f"mhc,pep,tgt\n{ALLELE},SIINFEKL,1\n")
    pseudo = tmp_path / "pseudoseqs_decoded.csv"
    if content is not None:
        pseudo.write_text(content)
    with pytest.raises(MHCIInputError, match="pseudoseqs_decoded.csv"):
        p.read_ppi_csv(
            path, mhc_pseudo_map=p.load_mhci_pseudoseq_map(tmp_path), pseudoseq_path=pseudo
        )


def test_other_pair_tasks_and_contexts_are_unchanged(tmp_path):
    from scripts import train_pair_pipeline as p

    path = tmp_path / "pairs.csv"
    path.write_text("sample_id,peptide,protein,label\ns1,acde,wxyz,1\n")
    assert p.read_ppi_csv(path, return_contexts=True) == ([("s1", "ACDE", "wxyz", 1)], ["wxyz"])
    path.write_text("sample_id,peptide,hla_pair,pair_pseudo,label\ns1,acde,HLA-DRB1*01:01,AAAA,1\n")
    assert p.read_ppi_csv(path, return_contexts=True) == (
        [("s1", "ACDE", "AAAA", 1)],
        ["HLA-DRB1*01:01"],
    )
    path.write_text(f"sample_id,mhc,pep,tgt\ns1,{ALLELE},acde,1\n")
    assert p.read_ppi_csv(path, {ALLELE: SEQUENCE}, True) == (
        [("s1", "ACDE", SEQUENCE, 1)],
        [ALLELE],
    )


def test_web_ensemble_accepts_inline_sequence_and_rejects_missing_mapping(tmp_path):
    from pepplatform.web.ensemble_data import read_split
    from pepplatform.web.ensemble_math import EnsembleError

    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    path = root / "train/pairs.csv"
    dataset = {"root_path": str(root), "task_type": "pep_mhci"}
    path.write_text(f"sample_id,mhc,pep,pair_pseudo,tgt\ns1,{ALLELE},SIINFEKL,{SEQUENCE},1\n")
    assert read_split(dataset, "train")["inputs"]["pairs"][0]["protein"] == SEQUENCE
    path.write_text(f"sample_id,mhc,pep,tgt\ns1,{ALLELE},SIINFEKL,1\n")
    with pytest.raises(EnsembleError) as error:
        read_split(dataset, "train")
    assert "no usable pseudosequence" in str(error.value.__cause__)


def test_guard_preserves_random_states_and_sequence_spelling():
    import numpy as np
    import torch

    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    partner = " acdX "
    assert (
        require_mhci_partner_sequence(
            partner, allele=ALLELE, resolved=True, source_path="pairs.csv"
        )
        == partner
    )
    assert random.getstate() == before[0]
    after = np.random.get_state()
    assert (
        after[0] == before[1][0]
        and np.array_equal(after[1], before[1][1])
        and after[2:] == before[1][2:]
    )
    assert torch.equal(torch.get_rng_state(), before[2])


def test_length_scan_finds_existing_task_root_mapping(tmp_path):
    from pepplatform.sequence_lengths import scan_dataset_length_stats

    data = tmp_path / "data/train"
    data.mkdir(parents=True)
    (data / "pairs.csv").write_text(f"mhc,pep,tgt\n{ALLELE},SIINFEKL,1\n")
    (tmp_path / "pseudoseqs_decoded.csv").write_text(
        f"mhc,pseudoseq,status\n{ALLELE},{SEQUENCE},ok\n"
    )
    stats = scan_dataset_length_stats(tmp_path / "data", "pair_split_dir")
    assert stats["overall"]["partner"]["max"] == len(SEQUENCE)
