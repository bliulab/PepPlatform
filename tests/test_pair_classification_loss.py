from argparse import Namespace

import pytest
import torch

from scripts import train_pair_pipeline


def test_pair_classification_loss_parser_defaults_to_historical_wbce():
    args = train_pair_pipeline.build_argument_parser().parse_args([])

    assert args.loss_type == "wbce"
    assert args.pos_weight_max == 50.0


def test_pair_classification_loss_bce_has_no_positive_weight():
    args = Namespace(loss_type="bce", pos_weight_max=50.0)
    criterion = train_pair_pipeline.build_pair_classification_criterion(
        args,
        torch.tensor([[1.0], [0.0], [0.0]]),
        torch.device("cpu"),
    )

    assert criterion.pos_weight is None
    assert args._classification_loss_provenance == {
        "schema_version": 1,
        "loss_type": "bce",
        "pos_weight_max": 50.0,
        "positive_count": 1,
        "negative_count": 2,
        "raw_pos_weight": pytest.approx(1.999998),
        "effective_pos_weight": None,
    }


def test_pair_classification_loss_wbce_uses_training_counts_and_custom_cap():
    args = Namespace(loss_type="wbce", pos_weight_max=1.5)
    criterion = train_pair_pipeline.build_pair_classification_criterion(
        args,
        torch.tensor([[1.0], [0.0], [0.0], [0.0]]),
        torch.device("cpu"),
    )

    assert criterion.pos_weight.tolist() == pytest.approx([1.5])
    provenance = args._classification_loss_provenance
    assert provenance["positive_count"] == 1
    assert provenance["negative_count"] == 3
    assert provenance["raw_pos_weight"] == pytest.approx(2.999997)
    assert provenance["effective_pos_weight"] == pytest.approx(1.5)


@pytest.mark.parametrize("raw", ["0", "0.99", "nan", "inf", "-inf"])
def test_pair_classification_loss_parser_rejects_invalid_cap(raw):
    with pytest.raises(SystemExit):
        train_pair_pipeline.build_argument_parser().parse_args(["--pos_weight_max", raw])


def test_pair_classification_loss_legacy_args_resolve_to_wbce_defaults():
    args = Namespace()
    criterion = train_pair_pipeline.build_pair_classification_criterion(
        args,
        torch.tensor([[1.0], [0.0]]),
        torch.device("cpu"),
    )

    assert args.loss_type == "wbce"
    assert args.pos_weight_max == 50.0
    assert criterion.pos_weight.tolist() == pytest.approx([1.0])


def test_pair_protocol_metadata_keeps_classification_and_semantic_contracts_separate():
    args = Namespace(
        predictor_type="classifier",
        seed=42,
        _classification_loss_provenance={
            "schema_version": 1,
            "loss_type": "wbce",
            "pos_weight_max": 50.0,
            "positive_count": 2,
            "negative_count": 2,
            "raw_pos_weight": 1.0,
            "effective_pos_weight": 1.0,
        },
        _semantic_provenance={"relation_type": "jaccard", "source_split": "train"},
    )

    metadata = train_pair_pipeline._pair_protocol_metadata(args)

    assert metadata["classification_loss"]["loss_type"] == "wbce"
    assert metadata["semantic"]["relation_type"] == "jaccard"
