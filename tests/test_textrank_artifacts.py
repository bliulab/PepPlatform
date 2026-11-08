from argparse import Namespace
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from pepplatform.featurizers import FeaturizerFactory
from pepplatform.web.inference import InferenceUnsupportedError, ModelService
from scripts.train_pair_pipeline import (
    _classic_extra_artifact_info,
    _stream_artifact_metadata,
    build_classic_featurizer,
)
from scripts.train_sequence_pipeline import SequencePipeline


def _kmer_textrank_contract():
    featurizer = FeaturizerFactory(bpe_path="").create(
        "kmer",
        "textrank",
        max_length=5,
        k=1,
    )
    return dict(featurizer.textrank_contract)


def _inference_metadata(contract):
    return {
        "classic": {
            "tokenizer": "kmer",
            "encoder": "textrank",
            "max_length": 5,
            "k": 1,
            "textrank_contract": contract,
        }
    }


def test_sequence_artifact_exports_textrank_contract():
    contract = _kmer_textrank_contract()
    pipeline = object.__new__(SequencePipeline)
    pipeline.args = SimpleNamespace(
        tokenizer="kmer",
        encoder="textrank",
        max_length=5,
        k=1,
        artifact_dir="",
        _textrank_contract=contract,
    )

    info = pipeline._sequence_feature_artifact_info(SimpleNamespace(label_names=["positive"]))

    assert info["classic"]["textrank_contract"] == contract


def test_pair_artifact_exports_textrank_contract():
    contract = _kmer_textrank_contract()

    info = _classic_extra_artifact_info(
        Namespace(
            tokenizer="kmer",
            encoder="textrank",
            k=1,
            _textrank_contract=contract,
        )
    )

    assert info["textrank_contract"] == contract


def test_pair_featurizer_records_contract_on_runtime_args(tmp_path):
    args = Namespace(
        tokenizer="kmer",
        encoder="textrank",
        max_length=5,
        k=1,
    )
    rows = [("sample", "ACD", "GGA", 1)]

    featurizer = build_classic_featurizer(args, str(tmp_path), rows)

    assert args._textrank_contract == featurizer.textrank_contract


def test_stream_artifact_exports_contract_and_legacy_adapter_preserves_it():
    contract = _kmer_textrank_contract()
    args = Namespace(
        streams={
            "peptide": {
                "tokenizer": "kmer",
                "encoder": "textrank",
                "max_length": 5,
                "params": {"k": 1},
            }
        },
        _stream_runtime={"peptide": {"textrank_contract": contract}},
        artifact_dir="",
    )

    stream = _stream_artifact_metadata(
        args,
        "peptide",
        torch.ones((2, 28)),
        "vector",
    )
    prepared = ModelService._stream_legacy_metadata(
        {"streams": {"peptide": stream}, "args": {}},
        "peptide",
    )

    assert stream["textrank_contract"] == contract
    assert prepared["classic"]["textrank_contract"] == contract


def test_inference_accepts_exact_textrank_contract():
    contract = _kmer_textrank_contract()

    featurizer, encoder = ModelService._build_classic_featurizer(_inference_metadata(contract))

    assert encoder == "textrank"
    assert featurizer.textrank_contract == contract


@pytest.mark.parametrize("mode", ["missing", "tampered"])
def test_inference_rejects_legacy_or_tampered_textrank_contract(mode):
    contract = _kmer_textrank_contract()
    metadata = _inference_metadata(contract)
    if mode == "missing":
        metadata["classic"].pop("textrank_contract")
    else:
        metadata = deepcopy(metadata)
        metadata["classic"]["textrank_contract"]["damping"] = 0.9

    with pytest.raises(
        InferenceUnsupportedError,
        match="strict weighted TextRank contract",
    ) as exc:
        ModelService._build_classic_featurizer(metadata)

    assert exc.value.code == "textrank_contract_unsupported"
    assert exc.value.stage == "bundle"


@pytest.mark.parametrize("encoder", ["property", "physchem"])
def test_inference_explicitly_rejects_removed_property_artifacts(encoder):
    with pytest.raises(
        InferenceUnsupportedError,
        match="Property encoder has been removed",
    ) as exc:
        ModelService._build_classic_featurizer(
            {
                "classic": {
                    "tokenizer": "kmer",
                    "encoder": encoder,
                    "max_length": 5,
                    "k": 1,
                }
            }
        )

    assert exc.value.code == "property_encoder_removed"
    assert exc.value.stage == "bundle"
