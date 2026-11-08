from argparse import Namespace
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pepplatform.covariance import (
    COVARIANCE_DEFAULT_MAX_RAW_LENGTH,
    GUO_NORMALIZED_PROPERTY_TABLE,
    GUO_NORMALIZED_PROPERTY_TABLE_SHA256,
    GUO_PROPERTY_NAMES,
    GUO_RAW_PROPERTY_TABLE,
    GUO_RAW_PROPERTY_TABLE_SHA256,
    GUO_SUPPLEMENTARY_DOC_SHA256,
    build_covariance_encoder_contract,
    compute_covariance_features,
    covariance_feature_names,
    standardize_property_table,
)
from pepplatform.encoders import (
    AutoCovarianceEncoder,
    AutoCrossCovarianceEncoder,
    CrossCovarianceEncoder,
)
from pepplatform.featurizers import FeaturizerFactory
from pepplatform.web.inference import InferenceUnsupportedError, ModelService
from pepplatform.web.preflight import preflight_experiment
from scripts.train_pair_pipeline import (
    _classic_extra_artifact_info,
    _stream_artifact_metadata,
    build_classic_featurizer,
)
from scripts.train_sequence_pipeline import SequencePipeline


def _contract(encoder="ac", lag=2, max_raw_length=1000):
    return build_covariance_encoder_contract(
        encoder=encoder,
        lag=lag,
        max_raw_length=max_raw_length,
    )


def _inference_metadata(encoder="ac", lag=2, max_raw_length=1000):
    return {
        "classic": {
            "tokenizer": "kmer",
            "encoder": encoder,
            "max_length": 3,
            "k": 1,
            f"{encoder}_lag": lag,
            "covariance_max_raw_length": max_raw_length,
            "covariance_encoder_contract": _contract(
                encoder,
                lag,
                max_raw_length,
            ),
        }
    }


def test_guo_property_table_provenance_and_population_standardization():
    assert GUO_SUPPLEMENTARY_DOC_SHA256 == (
        "f98f4b5fa182ce6053e5d46364c5a16fcfde4c2f75fedcb45503d50e7b647a59"
    )
    assert GUO_RAW_PROPERTY_TABLE_SHA256 == (
        "1af30c54e54b10140de41e204762ed37d92627334d83daf7d77731cccbe18699"
    )
    assert GUO_NORMALIZED_PROPERTY_TABLE_SHA256 == (
        "eed2a0d5661a2cbe34dd0338f9a250ecbded2d44174da60410d5f24fefd66a7a"
    )
    np.testing.assert_allclose(
        GUO_NORMALIZED_PROPERTY_TABLE.mean(axis=0),
        np.zeros(7),
        atol=5e-16,
    )
    np.testing.assert_allclose(
        GUO_NORMALIZED_PROPERTY_TABLE.std(axis=0, ddof=0),
        np.ones(7),
        atol=5e-16,
    )


def test_ac_cc_match_explicit_formula_and_keep_cc_direction():
    table = np.arange(20 * 7, dtype=np.float64).reshape(20, 7)
    table = table + np.arange(7, dtype=np.float64) ** 2
    sequence = "ACDE"
    values = table[:4]
    centered = values - values.mean(axis=0)

    expected_ac = np.asarray(
        [sum(centered[i, prop] * centered[i + 1, prop] for i in range(3)) / 3 for prop in range(7)]
    )
    expected_cc = np.asarray(
        [
            sum(centered[i, source] * centered[i + 1, target] for i in range(3)) / 3
            for source in range(7)
            for target in range(7)
            if source != target
        ]
    )

    actual_ac = compute_covariance_features(
        sequence,
        encoder="ac",
        lag=1,
        max_raw_length=100,
        property_table=table,
    )
    actual_cc = compute_covariance_features(
        sequence,
        encoder="cc",
        lag=1,
        max_raw_length=100,
        property_table=table,
    )
    np.testing.assert_allclose(actual_ac, expected_ac, rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual_cc, expected_cc, rtol=0, atol=1e-12)

    names = covariance_feature_names("cc", 1)
    forward = names.index("cc_lag1_H1_to_H2")
    reverse = names.index("cc_lag1_H2_to_H1")
    assert actual_cc[forward] == pytest.approx(expected_cc[forward])
    assert actual_cc[reverse] == pytest.approx(expected_cc[reverse])


def test_real_sequence_acc_golden_vector_and_exact_decomposition():
    expected = np.asarray(
        [
            -0.027641320199,
            0.052800005614,
            0.128975973023,
            -0.114423339089,
            -0.010100579683,
            0.002814904826,
            -0.006469259720,
            -0.058112293460,
            0.160606087771,
            0.032717081840,
            0.202213048978,
            0.157284852631,
            -0.063653233680,
            0.070496194730,
            -0.106727498553,
            -0.009295534280,
            -0.125706814251,
            -0.102722729019,
            0.077828911123,
            0.109240337070,
            -0.114023106294,
            -0.218564142861,
            0.056947165196,
            0.069137447741,
            0.231476382926,
            0.099716557588,
            -0.022536002621,
            -0.011270220922,
            -0.027618859967,
            -0.017612793418,
            0.147068055525,
            0.111138674336,
            -0.110948242415,
            0.058478490513,
            -0.157358794241,
            0.000486686127,
            0.205001767875,
            0.149487173794,
            -0.117998979274,
            0.069376595034,
            -0.218336388959,
            -0.013497481906,
            0.203519285603,
            -0.207890794149,
            -0.133255798195,
            0.173965926223,
            0.127245119724,
            0.212196887858,
            0.212239148094,
        ],
        dtype=np.float64,
    )
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    ac = compute_covariance_features(sequence, encoder="ac", lag=1, max_raw_length=100)
    cc = compute_covariance_features(sequence, encoder="cc", lag=1, max_raw_length=100)
    acc = compute_covariance_features(sequence, encoder="acc", lag=1, max_raw_length=100)

    np.testing.assert_allclose(acc, expected, rtol=0, atol=5e-13)
    np.testing.assert_array_equal(acc, np.concatenate((ac, cc)))


@pytest.mark.parametrize(
    ("encoder_class", "dimension"),
    [
        (AutoCovarianceEncoder, 21),
        (CrossCovarianceEncoder, 126),
        (AutoCrossCovarianceEncoder, 147),
    ],
)
def test_encoder_dimensions_names_dtype_and_normalization(encoder_class, dimension):
    encoder = encoder_class(lag=3, max_raw_length=100)
    result = encoder.encode_sequence("  acdefghik  ")

    assert result.shape == (dimension,)
    assert result.dtype == torch.float32
    assert len(encoder.feature_names) == dimension
    assert encoder.covariance_encoder_contract["output_dimension"] == dimension
    torch.testing.assert_close(result, encoder.encode("ACDEFGHIK"))


def test_acc_uses_complete_raw_sequence_independent_of_tokenizer_max_length():
    featurizer = FeaturizerFactory(
        bpe_path="",
        acc_lag=2,
        covariance_max_raw_length=100,
    ).create("kmer", "acc", max_length=3, k=1)

    first = featurizer.encode("ACDEFGHIK")
    second = featurizer.encode("ACDEFGHIA")

    assert first.shape == (98,)
    assert not torch.equal(first, second)
    assert featurizer.covariance_encoder_contract["full_length_policy"] == (
        "complete_raw_sequence_no_tokenizer_truncation"
    )


@pytest.mark.parametrize(
    "sequence, message",
    [
        ("", "non-empty"),
        ("   ", "non-empty"),
        ("AC DE", "internal whitespace"),
        ("ACDX", "non-standard"),
        ("AC", "length > lag"),
        ("ACDEFG", "exceeds covariance_max_raw_length"),
    ],
)
def test_invalid_sequences_are_rejected(sequence, message):
    encoder = AutoCovarianceEncoder(lag=2, max_raw_length=5)
    with pytest.raises(ValueError, match=message):
        encoder.encode_sequence(sequence)


@pytest.mark.parametrize("sequence", [None, ["A", "C"], b"ACD"])
def test_covariance_input_must_be_a_string(sequence):
    with pytest.raises(ValueError, match="sequence string"):
        AutoCovarianceEncoder(lag=1).encode_sequence(sequence)


@pytest.mark.parametrize("lag", [True, False, "1", 0, -1, 1.5])
def test_invalid_lag_is_rejected(lag):
    with pytest.raises(ValueError, match="positive integer"):
        AutoCovarianceEncoder(lag=lag)


@pytest.mark.parametrize("max_raw_length", [True, "10", 0, -1, 100001])
def test_invalid_max_raw_length_is_rejected(max_raw_length):
    with pytest.raises(ValueError, match="covariance_max_raw_length"):
        AutoCovarianceEncoder(max_raw_length=max_raw_length)


def test_factory_requires_residue_k_for_covariance_encoders():
    with pytest.raises(ValueError, match="requires k=1"):
        FeaturizerFactory(bpe_path="").create("kmer", "ac", max_length=10, k=2)


def test_web_preflight_derives_k1_and_rejects_explicit_non_residue_k():
    dataset = {
        "id": "dataset",
        "name": "sequences",
        "task_type": "pep_function",
        "format": "sequence_dir",
        "metadata": {"splits": {"train": 10, "val": 2, "test": 2}},
    }
    derived = preflight_experiment(
        dataset,
        "pep_function",
        {"tokenizer": "kmer", "encoder": "ac", "model": "mlp"},
    )
    rejected = preflight_experiment(
        dataset,
        "pep_function",
        {"tokenizer": "kmer", "encoder": "ac", "model": "mlp", "k": 2},
    )

    assert derived["normalized_config"]["k"] == 1
    assert derived["parameter_sources"]["k"] == "derived"
    assert "covariance_encoder_requires_k1" in rejected["error_codes"]


def test_property_table_validation_rejects_nonfinite_and_zero_variance():
    nonfinite = GUO_RAW_PROPERTY_TABLE.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        standardize_property_table(nonfinite)

    zero_variance = GUO_RAW_PROPERTY_TABLE.copy()
    zero_variance[:, 0] = 1.0
    with pytest.raises(ValueError, match="non-zero variance"):
        standardize_property_table(zero_variance)

    with pytest.raises(ValueError, match="non-zero variance"):
        compute_covariance_features(
            "ACDE",
            encoder="ac",
            lag=1,
            max_raw_length=100,
            property_table=zero_variance,
        )


def test_sequence_and_pair_artifacts_persist_covariance_contract_and_lag():
    contract = _contract("cc", lag=4, max_raw_length=321)
    pipeline = object.__new__(SequencePipeline)
    pipeline.args = SimpleNamespace(
        tokenizer="kmer",
        encoder="cc",
        max_length=3,
        k=1,
        ac_lag=10,
        cc_lag=4,
        acc_lag=10,
        covariance_max_raw_length=321,
        artifact_dir="",
        _covariance_encoder_contract=contract,
    )
    sequence_info = pipeline._sequence_feature_artifact_info(
        SimpleNamespace(label_names=["positive"])
    )["classic"]
    pair_info = _classic_extra_artifact_info(
        Namespace(
            tokenizer="kmer",
            encoder="cc",
            k=1,
            covariance_max_raw_length=321,
            _covariance_encoder_contract=contract,
        )
    )

    assert sequence_info["cc_lag"] == 4
    assert sequence_info["covariance_encoder_contract"] == contract
    assert pair_info["covariance_encoder_contract"] == contract
    assert pair_info["covariance_max_raw_length"] == 321


def test_pair_featurizer_and_stream_adapter_preserve_nondefault_contract(tmp_path):
    args = Namespace(
        tokenizer="kmer",
        encoder="acc",
        max_length=3,
        k=1,
        acc_lag=3,
        covariance_max_raw_length=321,
    )
    rows = [("sample", "ACDEFG", "HIKLMN", 1)]
    featurizer = build_classic_featurizer(args, str(tmp_path), rows)
    contract = dict(featurizer.covariance_encoder_contract)

    stream_args = Namespace(
        streams={
            "peptide": {
                "tokenizer": "kmer",
                "encoder": "acc",
                "max_length": 3,
                "params": {
                    "k": 1,
                    "acc_lag": 3,
                    "covariance_max_raw_length": 321,
                },
            }
        },
        _stream_runtime={"peptide": {"covariance_encoder_contract": contract}},
        artifact_dir="",
    )
    stream = _stream_artifact_metadata(
        stream_args,
        "peptide",
        torch.ones((2, 147)),
        "vector",
    )
    prepared = ModelService._stream_legacy_metadata(
        {"streams": {"peptide": stream}, "args": {}},
        "peptide",
    )
    restored, encoder = ModelService._build_classic_featurizer(prepared)

    assert args._covariance_encoder_contract == contract
    assert stream["covariance_encoder_contract"] == contract
    assert prepared["classic"]["covariance_encoder_contract"] == contract
    assert encoder == "acc"
    assert restored.feature_dim == 147


def test_inference_reconstructs_nondefault_covariance_contract():
    metadata = _inference_metadata("acc", lag=3, max_raw_length=321)
    featurizer, encoder = ModelService._build_classic_featurizer(metadata)

    assert encoder == "acc"
    assert featurizer.feature_dim == 147
    assert featurizer.encoder.lag == 3
    assert featurizer.encoder.max_raw_length == 321


def test_inference_rejects_missing_legacy_and_tampered_covariance_contracts():
    missing = _inference_metadata("ac", lag=2, max_raw_length=321)
    del missing["classic"]["covariance_encoder_contract"]
    with pytest.raises(InferenceUnsupportedError) as missing_error:
        ModelService._build_classic_featurizer(missing)
    assert missing_error.value.code == "covariance_contract_unsupported"

    tampered = deepcopy(_inference_metadata("ac", lag=2, max_raw_length=321))
    tampered["classic"]["covariance_encoder_contract"]["output_dimension"] = 999
    with pytest.raises(InferenceUnsupportedError) as tampered_error:
        ModelService._build_classic_featurizer(tampered)
    assert tampered_error.value.code == "covariance_contract_unsupported"


def test_contract_has_fixed_guo7_property_order_and_acc_block_order():
    contract = _contract("acc", lag=2)
    names = covariance_feature_names("acc", 2)

    assert contract["property_names"] == list(GUO_PROPERTY_NAMES)
    assert names[:7] == tuple(f"ac_lag1_{name}" for name in GUO_PROPERTY_NAMES)
    assert names[14] == "cc_lag1_H1_to_H2"
    assert contract["output_dimension"] == 98
    assert contract["property_normalization"]["ddof"] == 0
    assert contract["denominator"] == "n-lag"
