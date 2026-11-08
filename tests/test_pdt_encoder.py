from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pepplatform.encoders import PDTEncoder
from pepplatform.featurizers import FeaturizerFactory
from pepplatform.pdt import (
    PDT_DATA_SNAPSHOT_SHA256,
    PDT_DEFAULT_LAG,
    PDT_DEFAULT_MAX_RAW_LENGTH,
    PDT_NORMALIZED_PROPERTY_TABLE,
    PDT_NORMALIZED_PROPERTY_TABLE_SHA256,
    PDT_PROPERTY_ACCESSIONS,
    PDT_PROPERTY_COUNT,
    PDT_RAW_PROPERTY_TABLE,
    PDT_RAW_PROPERTY_TABLE_SHA256,
    build_pdt_encoder_contract,
    compute_pdt_features,
    pdt_feature_names,
    standardize_pdt_property_table,
)
from pepplatform.web.inference import InferenceUnsupportedError, ModelService
from pepplatform.web.preflight import preflight_experiment
from scripts.train_pair_pipeline import (
    _classic_extra_artifact_info,
    _stream_artifact_metadata,
)
from scripts.train_sequence_pipeline import SequencePipeline


REFERENCE = Path(__file__).with_name("fixtures") / "pdt_reference_beta8.tsv"


def _contract(lag=2, max_raw_length=100):
    return build_pdt_encoder_contract(
        lag=lag,
        max_raw_length=max_raw_length,
    )


def _inference_metadata(lag=2, max_raw_length=100):
    return {
        "classic": {
            "tokenizer": "kmer",
            "encoder": "pdt",
            "max_length": 3,
            "k": 1,
            "pdt_lag": lag,
            "pdt_max_raw_length": max_raw_length,
            "pdt_encoder_contract": _contract(lag, max_raw_length),
        }
    }


def test_pdt_snapshot_provenance_and_standardization():
    assert PDT_DATA_SNAPSHOT_SHA256 == (
        "84490233ec4f98885d7e29edeb489001542fefd1f6dca3826d4f574f3ae6b01f"
    )
    assert PDT_RAW_PROPERTY_TABLE_SHA256 == (
        "ab585790767bb5266778274bc7a820ed9f956651218bc78e9a1680f13000b77c"
    )
    assert PDT_NORMALIZED_PROPERTY_TABLE_SHA256 == (
        "973ded2cbcc4b37cd610b10d0c37c7222e49aacf4d949a7e1f27d85b25bfd8f9"
    )
    assert len(PDT_PROPERTY_ACCESSIONS) == PDT_PROPERTY_COUNT == 531
    np.testing.assert_allclose(
        PDT_NORMALIZED_PROPERTY_TABLE.mean(axis=1),
        np.zeros(PDT_PROPERTY_COUNT),
        atol=5e-15,
    )
    np.testing.assert_allclose(
        PDT_NORMALIZED_PROPERTY_TABLE.std(axis=1, ddof=0),
        np.ones(PDT_PROPERTY_COUNT),
        atol=5e-15,
    )


def test_pdt_matches_author_reference_vector():
    expected = np.loadtxt(REFERENCE)
    actual = compute_pdt_features(
        "ACDEFGHIKLMNPQRSTVWY",
        lag=PDT_DEFAULT_LAG,
    )
    assert actual.shape == (531 * 8,)
    # Text S2 writes six decimal places; this tolerance covers that output
    # quantization while still detecting absolute-distance or ordering errors.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-5)


def test_pdt_formula_and_property_major_ordering():
    table = np.zeros((PDT_PROPERTY_COUNT, 20), dtype=np.float64)
    table[0] = np.arange(20, dtype=np.float64)
    table[1] = np.arange(20, dtype=np.float64) * 2
    actual = compute_pdt_features(
        "ARND",
        lag=2,
        max_raw_length=100,
        property_table=table,
    )
    assert actual[:4].tolist() == pytest.approx([1.0, 4.0, 4.0, 16.0])
    assert np.count_nonzero(actual[4:]) == 0


def test_encoder_contract_dimensions_names_and_dtype():
    encoder = PDTEncoder(lag=2, max_raw_length=100)
    encoded = encoder.encode_sequence(" acdefghik ")

    assert encoded.shape == (1062,)
    assert encoded.dtype == torch.float32
    assert encoder.feature_names == pdt_feature_names(2)
    assert encoder.pdt_encoder_contract == _contract(2, 100)
    torch.testing.assert_close(encoded, encoder.encode("ACDEFGHIK"))


def test_factory_uses_complete_raw_sequence_and_requires_k1():
    factory = FeaturizerFactory(
        bpe_path="",
        pdt_lag=2,
        pdt_max_raw_length=100,
    )
    with pytest.raises(ValueError, match="requires k=1"):
        factory.create("kmer", "pdt", max_length=3, k=2)

    featurizer = factory.create("kmer", "pdt", max_length=3, k=1)
    first = featurizer.encode("ACDEFGHIK")
    second = featurizer.encode("ACDEFGHIA")
    assert first.shape == (1062,)
    assert not torch.equal(first, second)
    assert featurizer.pdt_encoder_contract == _contract(2, 100)


@pytest.mark.parametrize(
    ("sequence", "message"),
    [
        ("", "non-empty"),
        ("   ", "non-empty"),
        ("AC DEFG", "internal whitespace"),
        ("ACDX", "non-standard"),
        ("AC", "length > lag"),
        ("ACDEFG", "exceeds pdt_max_raw_length"),
    ],
)
def test_pdt_rejects_invalid_sequences(sequence, message):
    encoder = PDTEncoder(lag=2, max_raw_length=5)
    with pytest.raises(ValueError, match=message):
        encoder.encode_sequence(sequence)


@pytest.mark.parametrize("value", [True, False, "2", 0, -1, 1.5])
def test_pdt_rejects_invalid_lag(value):
    with pytest.raises(ValueError, match="positive integer"):
        PDTEncoder(lag=value)


@pytest.mark.parametrize("value", [True, "10", 0, -1, 100001])
def test_pdt_rejects_invalid_max_raw_length(value):
    with pytest.raises(ValueError, match="pdt_max_raw_length"):
        PDTEncoder(max_raw_length=value)


def test_pdt_rejects_non_string_input():
    with pytest.raises(ValueError, match="sequence string"):
        PDTEncoder(lag=1).encode_sequence(["A", "C"])


def test_sequence_and_pair_artifacts_persist_pdt_contract():
    contract = _contract(lag=4, max_raw_length=321)
    pipeline = object.__new__(SequencePipeline)
    pipeline.args = SimpleNamespace(
        tokenizer="kmer",
        encoder="pdt",
        max_length=3,
        k=1,
        pdt_lag=4,
        pdt_max_raw_length=321,
        ac_lag=10,
        cc_lag=10,
        acc_lag=10,
        covariance_max_raw_length=100000,
        artifact_dir="",
        _pdt_encoder_contract=contract,
    )
    sequence_info = pipeline._sequence_feature_artifact_info(
        SimpleNamespace(label_names=["positive"])
    )["classic"]
    pair_info = _classic_extra_artifact_info(
        Namespace(
            tokenizer="kmer",
            encoder="pdt",
            k=1,
            pdt_lag=4,
            pdt_max_raw_length=321,
            _pdt_encoder_contract=contract,
        )
    )

    assert sequence_info["pdt_lag"] == 4
    assert sequence_info["pdt_max_raw_length"] == 321
    assert sequence_info["pdt_encoder_contract"] == contract
    assert pair_info["pdt_encoder_contract"] == contract
    assert pair_info["pdt_max_raw_length"] == 321


def test_inference_reconstructs_nondefault_pdt_contract():
    metadata = _inference_metadata(lag=3, max_raw_length=321)
    featurizer, encoder = ModelService._build_classic_featurizer(metadata)

    assert encoder == "pdt"
    assert featurizer.feature_dim == 1593
    assert featurizer.encoder.lag == 3
    assert featurizer.encoder.max_raw_length == 321


def test_pair_stream_metadata_preserves_pdt_contract():
    contract = _contract(lag=2, max_raw_length=100)
    args = Namespace(
        streams={
            "peptide": {
                "tokenizer": "kmer",
                "encoder": "pdt",
                "max_length": 3,
                "params": {
                    "k": 1,
                    "pdt_lag": 2,
                    "pdt_max_raw_length": 100,
                },
            }
        },
        _stream_runtime={"peptide": {"pdt_encoder_contract": contract}},
        artifact_dir="",
    )
    stream = _stream_artifact_metadata(
        args,
        "peptide",
        torch.ones((2, 1062)),
        "vector",
    )
    prepared = ModelService._stream_legacy_metadata(
        {"streams": {"peptide": stream}, "args": {}},
        "peptide",
    )
    restored, encoder = ModelService._build_classic_featurizer(prepared)

    assert stream["pdt_encoder_contract"] == contract
    assert prepared["classic"]["pdt_encoder_contract"] == contract
    assert encoder == "pdt"
    assert restored.feature_dim == 1062


def test_inference_rejects_missing_and_tampered_pdt_contracts():
    missing = _inference_metadata()
    del missing["classic"]["pdt_encoder_contract"]
    with pytest.raises(InferenceUnsupportedError) as missing_error:
        ModelService._build_classic_featurizer(missing)
    assert missing_error.value.code == "pdt_contract_unsupported"

    tampered = deepcopy(_inference_metadata())
    tampered["classic"]["pdt_encoder_contract"]["output_dimension"] = 999
    with pytest.raises(InferenceUnsupportedError) as tampered_error:
        ModelService._build_classic_featurizer(tampered)
    assert tampered_error.value.code == "pdt_contract_unsupported"


def test_standardization_rejects_nonfinite_and_zero_variance():
    nonfinite = PDT_RAW_PROPERTY_TABLE.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        standardize_pdt_property_table(nonfinite)

    zero_variance = PDT_RAW_PROPERTY_TABLE.copy()
    zero_variance[0] = 1.0
    with pytest.raises(ValueError, match="non-zero variance"):
        standardize_pdt_property_table(zero_variance)


def _sequence_dataset():
    return {
        "id": "dataset",
        "name": "sequences",
        "task_type": "pep_function",
        "format": "sequence_dir",
        "metadata": {
            "splits": {"train": 10, "val": 2},
            "length_stats": {
                "version": 1,
                "overall": {
                    "sequence": {
                        "count": 12,
                        "min": 8,
                        "max": 60,
                        "mean": 20.0,
                    }
                },
                "by_split": {},
            },
        },
    }


def test_pdt_preflight_derives_k1_and_checks_dataset_lengths():
    base = {
        "tokenizer": "kmer",
        "encoder": "pdt",
        "model": "mlp",
        "classifier": "mlp",
        "pdt_lag": 7,
        "pdt_max_raw_length": 60,
    }
    valid = preflight_experiment(_sequence_dataset(), "pep_function", base)
    assert valid["normalized_config"]["k"] == 1
    assert valid["valid"] is True

    wrong_k = preflight_experiment(
        _sequence_dataset(),
        "pep_function",
        {**base, "k": 2},
    )
    assert "pdt_encoder_requires_k1" in wrong_k["error_codes"]

    short_lag = preflight_experiment(
        _sequence_dataset(),
        "pep_function",
        {**base, "pdt_lag": 8},
    )
    assert "pdt_lag_too_large" in short_lag["error_codes"]

    short_limit = preflight_experiment(
        _sequence_dataset(),
        "pep_function",
        {**base, "pdt_max_raw_length": 59},
    )
    assert "pdt_raw_length_exceeded" in short_limit["error_codes"]
