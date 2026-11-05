from argparse import Namespace

import pytest

from pepplatform.workflow import (
    CHAIN_GRAPH_ARCHITECTURES,
    CHAIN_GRAPH_COMPATIBLE_ENCODERS,
    CLASSIC_ENCODER_OPTIONS,
    END_TO_END,
    EXPORT_EMBEDDINGS,
    SEQUENCE_ARCHITECTURES,
    SEQUENCE_MODELS,
    encoder_family,
    normalize_pair_workflow_args,
    normalize_workflow_args,
    supports_classic_pair_vector_inference,
    transformation_options_for_encoder,
    validate_index_route_semantics,
)


def test_modeling_architecture_taxonomy_keeps_legacy_combined_list():
    assert SEQUENCE_ARCHITECTURES == ["cnn", "rnn", "gru", "lstm", "transformer"]
    assert CHAIN_GRAPH_ARCHITECTURES == ["gcn", "gat"]
    assert SEQUENCE_MODELS == SEQUENCE_ARCHITECTURES + CHAIN_GRAPH_ARCHITECTURES
    assert "onehot" in CHAIN_GRAPH_COMPATIBLE_ENCODERS
    assert "index" not in CHAIN_GRAPH_COMPATIBLE_ENCODERS


def test_encoder_families_and_topic_transformations():
    assert encoder_family("index") == "sequence"
    assert encoder_family("pssm") == "sequence"
    assert encoder_family("bow") == "global"
    assert transformation_options_for_encoder("bow") == ["none", "lsa", "lda", "plsa"]
    assert transformation_options_for_encoder("TF-IDF") == ["none", "lsa"]
    assert transformation_options_for_encoder("textrank") == ["none", "lsa"]


def test_removed_labelled_lda_is_rejected_by_both_workflow_normalizers():
    message = "Labelled LDA feature transformation has been removed"
    with pytest.raises(ValueError, match=message):
        normalize_workflow_args(
            Namespace(
                tokenizer="kmer",
                encoder="bow",
                model="rf",
                reduction="labelled_lda",
            )
        )
    with pytest.raises(ValueError, match=message):
        normalize_pair_workflow_args(
            Namespace(
                tokenizer="kmer",
                encoder="bow",
                model="rf",
                reduction="labelled_lda",
            )
        )


def test_lsa_route_is_single_sequence_only_and_defaults_memory_guard():
    args = normalize_workflow_args(
        Namespace(
            task_type="pep_function",
            tokenizer="kmer",
            encoder="tfidf",
            model="rf",
            reduction="lsa",
            lsa_dim=2,
        )
    )
    assert args.lsa_dense_memory_limit_gib == 4.0

    with pytest.raises(ValueError, match="ml_reduction=tsvd") as exc_info:
        normalize_pair_workflow_args(
            Namespace(
                task_type="pep_protein_interaction",
                tokenizer="kmer",
                encoder="tfidf",
                model="rf",
                reduction="lsa",
                lsa_dim=2,
            )
        )
    assert exc_info.value.code == "lsa_pair_route_unsupported"


def test_pair_workflow_defaults_tsvd_memory_guard():
    args = normalize_pair_workflow_args(
        Namespace(
            task_type="pep_protein_interaction",
            tokenizer="kmer",
            encoder="bow",
            model="rf",
            reduction="none",
        )
    )
    assert args.tsvd_dense_memory_limit_gib == 4.0
    assert args.pca_dense_memory_limit_gib == 4.0
    assert args.kpca_memory_limit_gib == 4.0


def test_workflow_normalizes_kpca_defaults_and_rejects_inapplicable_gamma():
    args = normalize_workflow_args(
        Namespace(
            task_type="pep_function",
            tokenizer="kmer",
            encoder="bow",
            model="rf",
            reduction="none",
            ml_reduction="kpca",
            ml_reduction_dim=2,
        )
    )

    assert args.kpca_kernel == "rbf"
    assert args.kpca_gamma == 0.1
    assert args.kpca_memory_limit_gib == 4.0

    with pytest.raises(ValueError) as exc_info:
        normalize_workflow_args(
            Namespace(
                task_type="pep_function",
                tokenizer="kmer",
                encoder="bow",
                model="rf",
                reduction="none",
                ml_reduction="kpca",
                ml_reduction_dim=2,
                kpca_kernel="linear",
                kpca_gamma=0.1,
            )
        )
    assert exc_info.value.code == "kpca_gamma_not_applicable"


def test_lsa_route_rejects_non_global_encoder_without_silent_noop():
    with pytest.raises(ValueError, match="classic global encoder") as exc_info:
        normalize_workflow_args(
            Namespace(
                task_type="pep_function",
                tokenizer="kmer",
                encoder="onehot",
                model="rf",
                reduction="lsa",
                lsa_dim=2,
            )
        )
    assert exc_info.value.code == "lsa_encoder_unsupported"


def test_lda_route_accepts_strict_pair_bow_and_defaults_memory_guard():
    args = normalize_workflow_args(
        Namespace(
            task_type="pep_function",
            tokenizer="kmer",
            encoder="bow",
            model="rf",
            reduction="lda",
            lda_topics=5,
        )
    )
    assert args.lda_dense_memory_limit_gib == 4.0

    pair_args = normalize_pair_workflow_args(
        Namespace(
            task_type="pep_protein_interaction",
            tokenizer="kmer",
            encoder="bow",
            model="rf",
            classifier="rf",
            pair_strategy="two_stream",
            interaction_level="vector",
            predictor_type="classifier",
            fusion_mode="concat",
            reduction="lda",
            lda_topics=5,
        )
    )
    assert pair_args.reduction == "lda"
    assert pair_args.fusion_mode == "concat"

    with pytest.raises(ValueError, match="fusion_mode=concat") as exc_info:
        normalize_pair_workflow_args(
            Namespace(
                task_type="pep_protein_interaction",
                tokenizer="kmer",
                encoder="bow",
                model="rf",
                classifier="rf",
                pair_strategy="two_stream",
                interaction_level="vector",
                predictor_type="classifier",
                fusion_mode="aligned_gated",
                reduction="lda",
                lda_topics=5,
            )
        )
    assert exc_info.value.code == "pair_bow_fusion_unsupported"


def test_lda_route_rejects_weighted_encoder_without_silent_noop():
    with pytest.raises(ValueError, match="encoder=bow") as exc_info:
        normalize_workflow_args(
            Namespace(
                task_type="pep_function",
                tokenizer="kmer",
                encoder="tfidf",
                model="rf",
                reduction="lda",
                lda_topics=2,
            )
        )
    assert exc_info.value.code == "lda_encoder_unsupported"


def test_plsa_route_accepts_strict_pair_bow_and_defaults_contract_controls():
    args = normalize_workflow_args(
        Namespace(
            task_type="pep_function",
            tokenizer="kmer",
            encoder="bow",
            model="rf",
            reduction="plsa",
            plsa_topics=5,
        )
    )
    assert args.plsa_memory_limit_gib == 4.0
    assert args.plsa_contract_version == 2

    pair_args = normalize_pair_workflow_args(
        Namespace(
            task_type="pep_protein_interaction",
            tokenizer="kmer",
            encoder="bow",
            model="rf",
            classifier="rf",
            pair_strategy="two_stream",
            interaction_level="vector",
            predictor_type="classifier",
            fusion_mode="concat",
            reduction="plsa",
            plsa_topics=5,
        )
    )
    assert pair_args.reduction == "plsa"
    assert pair_args.fusion_mode == "concat"

    with pytest.raises(ValueError, match="fusion_mode=concat") as exc_info:
        normalize_pair_workflow_args(
            Namespace(
                task_type="pep_protein_interaction",
                tokenizer="kmer",
                encoder="bow",
                model="rf",
                classifier="rf",
                pair_strategy="two_stream",
                interaction_level="vector",
                predictor_type="classifier",
                fusion_mode="aligned_gated",
                reduction="plsa",
                plsa_topics=5,
            )
        )
    assert exc_info.value.code == "pair_bow_fusion_unsupported"


def test_plsa_route_rejects_weighted_encoder_without_silent_clipping():
    with pytest.raises(ValueError, match="encoder=bow") as exc_info:
        normalize_workflow_args(
            Namespace(
                task_type="pep_function",
                tokenizer="kmer",
                encoder="tfidf",
                model="rf",
                reduction="plsa",
                plsa_topics=2,
            )
        )
    assert exc_info.value.code == "plsa_encoder_unsupported"


def test_removed_encoders_are_rejected_and_not_advertised():
    assert len(CLASSIC_ENCODER_OPTIONS["kmer"]) == 23
    assert "pssm_blosum62_hybrid" in CLASSIC_ENCODER_OPTIONS["kmer"]
    assert "psfm_blosum62_hybrid" in CLASSIC_ENCODER_OPTIONS["kmer"]
    assert {
        "ac_pssm_blosum62_hybrid",
        "cc_pssm_blosum62_hybrid",
        "acc_pssm_blosum62_hybrid",
    }.issubset(CLASSIC_ENCODER_OPTIONS["kmer"])
    assert all(
        {"cs", "property"}.isdisjoint(encoders) for encoders in CLASSIC_ENCODER_OPTIONS.values()
    )
    with pytest.raises(ValueError, match="CS encoder has been removed"):
        normalize_workflow_args(Namespace(tokenizer="kmer", encoder="cs", model="cnn"))
    with pytest.raises(ValueError, match="Property encoder has been removed"):
        normalize_workflow_args(Namespace(tokenizer="kmer", encoder="physchem", model="cnn"))


def test_legacy_sequence_model_maps_to_end_to_end_mlp():
    args = normalize_workflow_args(Namespace(model="cnn"))
    assert args.sequence_model == "cnn"
    assert args.training_mode == END_TO_END
    assert args.classifier == "mlp"
    assert args.model == "cnn"


def test_kmer_workflow_records_token_length_unit_without_leaking_to_other_tokenizers():
    kmer = normalize_workflow_args(Namespace(tokenizer="kmer", encoder="index", model="cnn"))
    other = normalize_workflow_args(
        Namespace(
            tokenizer="dr",
            encoder="index",
            model="cnn",
            max_length_unit="tokens",
        )
    )

    assert kmer.max_length_unit == "tokens"
    assert not hasattr(other, "max_length_unit")


def test_wordpiece_workflow_records_final_token_length_unit():
    wordpiece = normalize_workflow_args(
        Namespace(tokenizer="wordpiece", encoder="index", model="cnn")
    )

    assert wordpiece.max_length_unit == "tokens_including_special_tokens"


def test_explicit_export_route_keeps_sequence_model_and_classifier_roles():
    args = normalize_workflow_args(
        Namespace(
            model="mlp", sequence_model="gru", training_mode=EXPORT_EMBEDDINGS, classifier="svm"
        )
    )
    assert args.sequence_model == "gru"
    assert args.training_mode == EXPORT_EMBEDDINGS
    assert args.classifier == "svm"
    assert args.model == "gru"


@pytest.mark.parametrize(
    ("normalizer", "config", "error_code"),
    [
        (
            normalize_workflow_args,
            Namespace(
                model="cnn",
                sequence_model="cnn",
                training_mode=END_TO_END,
                classifier="rf",
            ),
            "end_to_end_requires_mlp",
        ),
        (
            normalize_workflow_args,
            Namespace(
                model="cnn",
                sequence_model="cnn",
                training_mode=EXPORT_EMBEDDINGS,
                classifier="mlp",
            ),
            "export_embeddings_requires_classical_classifier",
        ),
        (
            normalize_pair_workflow_args,
            Namespace(
                model="cnn",
                pair_strategy="single_stream",
                sequence_model="cnn",
                training_mode=END_TO_END,
                classifier="rf",
            ),
            "end_to_end_requires_mlp",
        ),
        (
            normalize_pair_workflow_args,
            Namespace(
                model="cnn",
                pair_strategy="single_stream",
                sequence_model="cnn",
                training_mode=EXPORT_EMBEDDINGS,
                classifier="mlp",
            ),
            "export_embeddings_requires_classical_classifier",
        ),
        (
            normalize_pair_workflow_args,
            Namespace(
                model="cross_attn",
                pair_strategy="two_stream",
                interaction_level="token",
                interaction_model="cross_attn",
                training_mode=END_TO_END,
                classifier="rf",
            ),
            "end_to_end_requires_mlp",
        ),
        (
            normalize_pair_workflow_args,
            Namespace(
                model="cross_attn",
                pair_strategy="two_stream",
                interaction_level="token",
                interaction_model="cross_attn",
                training_mode=EXPORT_EMBEDDINGS,
                classifier="mlp",
            ),
            "export_embeddings_requires_classical_classifier",
        ),
    ],
)
def test_workflow_rejects_contradictory_training_mode_classifier_pairs(
    normalizer, config, error_code
):
    with pytest.raises(ValueError) as exc_info:
        normalizer(config)
    assert exc_info.value.code == error_code


def test_svm_calibration_normalizes_canonical_and_legacy_configs():
    canonical = normalize_workflow_args(
        Namespace(
            tokenizer="kmer",
            encoder="bow",
            model="svm",
            classifier="svm",
            svm_calibration="sigmoid",
            svm_calibration_cv_folds=3,
        )
    )
    legacy = normalize_workflow_args(
        Namespace(
            tokenizer="kmer",
            encoder="bow",
            model="calibrated_svm",
            classifier="calibrated_svm",
        )
    )

    for args in (canonical, legacy):
        assert args.model == "svm"
        assert args.classifier == "svm"
        assert args.svm_calibration == "sigmoid"
        assert args.svm_calibration_cv_folds == 3


def test_legacy_calibrated_svm_rejects_conflicting_explicit_mode():
    with pytest.raises(ValueError, match="conflicting svm_calibration"):
        normalize_workflow_args(
            Namespace(
                tokenizer="kmer",
                encoder="bow",
                model="calibrated_svm",
                classifier="calibrated_svm",
                svm_calibration="none",
            )
        )


def test_svm_calibration_fold_count_is_fixed():
    with pytest.raises(ValueError, match="fixed at 3"):
        normalize_workflow_args(
            Namespace(
                tokenizer="kmer",
                encoder="bow",
                model="svm",
                classifier="svm",
                svm_calibration="sigmoid",
                svm_calibration_cv_folds=5,
            )
        )


def test_index_requires_a_learned_embedding_route():
    direct = normalize_workflow_args(
        Namespace(task_type="pep_function", tokenizer="kmer", encoder="index", model="svm")
    )
    embedded = normalize_workflow_args(
        Namespace(task_type="pep_function", tokenizer="kmer", encoder="index", model="cnn")
    )
    exported = normalize_workflow_args(
        Namespace(
            task_type="pep_function",
            tokenizer="kmer",
            encoder="index",
            model="cnn",
            sequence_model="cnn",
            training_mode=EXPORT_EMBEDDINGS,
            classifier="svm",
        )
    )

    with pytest.raises(ValueError, match="raw index token IDs require a learned embedding"):
        validate_index_route_semantics(direct)
    validate_index_route_semantics(embedded)
    validate_index_route_semantics(exported)


def test_pair_index_is_token_level_not_vector_level():
    vector = normalize_pair_workflow_args(
        Namespace(
            task_type="pep_protein_interaction",
            tokenizer="kmer",
            encoder="index",
            model="mlp",
            pair_strategy="two_stream",
            interaction_level="vector",
        )
    )
    token = normalize_pair_workflow_args(
        Namespace(
            task_type="pep_protein_interaction",
            tokenizer="kmer",
            encoder="index",
            model="cross_attn",
            pair_strategy="two_stream",
            interaction_level="token",
        )
    )

    with pytest.raises(ValueError, match="affected streams: sequence"):
        validate_index_route_semantics(vector)
    validate_index_route_semantics(token)
    assert supports_classic_pair_vector_inference(vector) is False
    assert (
        supports_classic_pair_vector_inference({"tokenizer": "kmer", "encoder": "onehot"}) is True
    )
    assert (
        supports_classic_pair_vector_inference(
            {"tokenizer": "kmer", "encoder": "psfm_blosum62_hybrid"}
        )
        is True
    )


def test_heterogeneous_pair_vector_rejects_any_index_stream():
    config = {
        "task_type": "pep_protein_interaction",
        "pair_strategy": "two_stream",
        "interaction_level": "vector",
        "streams": {
            "peptide": {"tokenizer": "kmer", "encoder": "onehot"},
            "partner": {"tokenizer": "wordpiece", "encoder": "index"},
        },
    }

    with pytest.raises(ValueError, match="affected streams: partner"):
        validate_index_route_semantics(config)


def test_specialized_models_are_not_reclassified():
    args = normalize_workflow_args(Namespace(model="cross_attn+mlp"))
    assert args.sequence_model == ""
    assert args.training_mode == ""
    assert args.classifier == ""
    assert args.model == "cross_attn+mlp"


def test_legacy_pair_sequence_model_maps_to_single_stream():
    args = normalize_pair_workflow_args(Namespace(model="cnn"))
    assert args.pair_strategy == "single_stream"
    assert args.sequence_model == "cnn"
    assert args.training_mode == END_TO_END
    assert args.classifier == "mlp"
    assert args.predictor_type == "classifier"


def test_legacy_cross_attention_maps_to_two_stream_token_interaction():
    args = normalize_pair_workflow_args(Namespace(model="cross_attn+mlp"))
    assert args.pair_strategy == "two_stream"
    assert args.interaction_level == "token"
    assert args.interaction_model == "cross_attn"
    assert args.predictor_type == "classifier"
    assert args.classifier == "mlp"
    assert args.model == "cross_attn+mlp"


def test_legacy_graph_model_maps_to_sample_graph_predictor():
    args = normalize_pair_workflow_args(Namespace(model="gcn"))
    assert args.pair_strategy == "two_stream"
    assert args.interaction_level == "vector"
    assert args.predictor_type == "sample_graph"
    assert args.graph_predictor == "gcn"
    assert args.model == "gcn"


def test_legacy_contact_alias_maps_to_two_stream_vector_classifier():
    args = normalize_pair_workflow_args(Namespace(model="contact"))
    assert args.pair_strategy == "two_stream"
    assert args.interaction_level == "vector"
    assert args.predictor_type == "classifier"
    assert args.classifier == "mlp"
    assert args.model == "mlp"


def test_explicit_pair_roles_keep_cross_attention_export_route():
    args = normalize_pair_workflow_args(
        Namespace(
            model="mlp",
            pair_strategy="two_stream",
            interaction_level="token",
            interaction_model="bi_cross_attn",
            training_mode=EXPORT_EMBEDDINGS,
            classifier="rf",
        )
    )
    assert args.pair_strategy == "two_stream"
    assert args.interaction_level == "token"
    assert args.interaction_model == "bi_cross_attn"
    assert args.training_mode == EXPORT_EMBEDDINGS
    assert args.classifier == "rf"
    assert args.model == "bi_cross_attn+mlp"
