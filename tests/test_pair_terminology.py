from argparse import Namespace
from copy import deepcopy

import pytest

from pepplatform.pair_terminology import normalize_pair_strategy, public_pair_strategy
from pepplatform.route_registry import resolve_effective_route
from pepplatform.web.config import normalize_experiment_config
from pepplatform.workflow import normalize_pair_workflow_args


ROUTES = [
    dict(
        pair_strategy="single_stream",
        tokenizer="kmer",
        encoder="index",
        model="cnn",
        training_mode="end_to_end",
        classifier="mlp",
    ),
    dict(
        pair_strategy="single_stream",
        tokenizer="kmer",
        encoder="onehot",
        model="gcn",
        sequence_model="gcn",
        training_mode="export_embeddings",
        classifier="rf",
    ),
    dict(
        pair_strategy="single_stream",
        tokenizer="esm2",
        encoder="esm2",
        model="cnn",
        sequence_model="cnn",
        training_mode="end_to_end",
        classifier="mlp",
    ),
    dict(
        pair_strategy="single_stream",
        tokenizer="prott5",
        encoder="prott5",
        model="gru",
        sequence_model="gru",
        training_mode="export_embeddings",
        classifier="rf",
    ),
    dict(
        pair_strategy="two_stream",
        tokenizer="esm2",
        model="mlp",
        interaction_level="vector",
        fusion_mode="concat_all",
    ),
    dict(
        pair_strategy="two_stream",
        tokenizer="kmer",
        encoder="onehot",
        model="cross_attn",
        interaction_level="token",
        interaction_model="cross_attn",
        training_mode="end_to_end",
        classifier="mlp",
    ),
    dict(
        pair_strategy="two_stream",
        tokenizer="esm2",
        model="bi_cross_attn",
        interaction_level="token",
        interaction_model="bi_cross_attn",
        training_mode="export_embeddings",
        classifier="rf",
        partner_tokenizer="prott5",
        partner_encoder="prott5",
    ),
    dict(
        pair_strategy="two_stream",
        tokenizer="esm2",
        model="aligned_gated",
        interaction_level="vector",
        fusion_mode="aligned_gated",
        partner_tokenizer="prott5",
        partner_encoder="prott5",
    ),
    dict(
        pair_strategy="two_stream",
        tokenizer="esm2",
        model="sage",
        interaction_level="vector",
        predictor_type="sample_graph",
        graph_predictor="sage",
    ),
    dict(
        pair_strategy="two_stream",
        tokenizer="esm2",
        model="hgt",
        interaction_level="vector",
        predictor_type="sample_graph",
        graph_predictor="hgt",
    ),
]


@pytest.mark.parametrize("config", ROUTES)
@pytest.mark.parametrize("task", ["pep_hlai", "pep_hlaii", "pep_protein_interaction"])
@pytest.mark.parametrize("hyphenated", [False, True])
def test_public_aliases_preserve_full_effective_config_and_route(config, task, hyphenated):
    original = deepcopy(config)
    renamed = deepcopy(config)
    value = public_pair_strategy(config["pair_strategy"])
    renamed["pair_strategy"] = value.replace("_", "-") if hyphenated else value
    old = normalize_experiment_config(task, original)
    new = normalize_experiment_config(task, renamed)
    assert old == new  # All parameters used by contracts and cache identities.
    assert resolve_effective_route(original, task) == resolve_effective_route(renamed, task)
    old_args, new_args = Namespace(task_type=task, **original), Namespace(task_type=task, **renamed)
    normalize_pair_workflow_args(old_args)
    normalize_pair_workflow_args(new_args)
    assert vars(old_args) == vars(new_args)
    assert config == original  # Caller-owned input was not rewritten.


def test_cli_accepts_both_generations_and_rejects_unknown():
    from scripts.train_pair_pipeline import build_argument_parser

    parser = build_argument_parser()
    for value in (
        "single_stream",
        "single-stream",
        "single_stream",
        "two_stream",
        "two-stream",
        "two_stream",
    ):
        args = parser.parse_args(["--pair_strategy", value])
        assert args.pair_strategy == normalize_pair_strategy(value)
    assert parser.parse_args([]).pair_strategy is None
    with pytest.raises(SystemExit):
        parser.parse_args(["--pair_strategy", "three_stream"])
