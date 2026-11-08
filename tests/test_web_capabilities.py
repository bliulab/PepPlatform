from pepplatform.web.capabilities import build_capabilities


def test_capabilities_expose_interaction_semantic_modes():
    capabilities = build_capabilities()

    assert [option["id"] for option in capabilities["pretrained_models"]["esmc"]] == [
        "EvolutionaryScale/esmc-300m-2024-12",
        "EvolutionaryScale/esmc-600m-2024-12",
    ]

    assert "labelled_lda" not in capabilities["feature_transformations"]["bow"]
    assert "lsa" in capabilities["feature_transformations"]["bow"]
    assert all(
        "lsa" not in options for options in capabilities["pair_feature_transformations"].values()
    )
    assert capabilities["pair_feature_transformations"]["bow"] == ["none", "lda", "plsa"]
    assert all(
        "lda" not in options and "plsa" not in options
        for encoder, options in capabilities["pair_feature_transformations"].items()
        if encoder != "bow"
    )
    assert capabilities["feature_transformations"]["tfidf"] == ["none", "lsa"]
    assert capabilities["feature_transformations"]["textrank"] == ["none", "lsa"]
    assert capabilities["lsa_contract"] == {
        "version": 1,
        "variant": "train_only_uncentered_randomized_truncated_svd",
        "route": "single_sequence_classic_global",
        "pair_supported": False,
        "dense_memory_limit_gib_default": 4.0,
    }
    assert capabilities["lda_contract"] == {
        "version": 1,
        "variant": "train_only_batch_variational_bayes_bow_counts",
        "route": "single_sequence_classic_global_bow_v2",
        "pair_route": "pair_two_stream_classic_bow_v2",
        "pair_supported": True,
        "dense_memory_limit_gib_default": 4.0,
    }
    assert capabilities["plsa_contract"] == {
        "version": 2,
        "variant": "train_only_em_bow_counts",
        "route": "single_sequence_classic_global_bow_v2",
        "pair_route": "pair_two_stream_classic_bow_v2",
        "pair_supported": True,
        "memory_limit_gib_default": 4.0,
    }

    assert "calibrated_svm" not in capabilities["classifiers"]
    assert capabilities["svm_calibration"] == {
        "options": ["none", "sigmoid"],
        "default": "none",
        "cv_folds": 3,
    }
    assert "interaction_semantic_modes" not in capabilities
    assert "task_interaction_semantic_modes" not in capabilities
    assert "task_interaction_semantic_defaults" not in capabilities
    assert capabilities["task_semantic_relation_type"]["pep_function"] == "intrinsic_label_jaccard"
    assert capabilities["task_semantic_relation_type"]["pep_hlai"] == "interaction_profile_jaccard"
    assert capabilities["semantic_defaults"]["semantic_zero_weight"] == 0.25
    assert capabilities["semantic_defaults"]["semantic_positive_partners"] == 1
    assert capabilities["semantic_defaults"]["semantic_zero_partners"] == 8
    assert capabilities["semantic_defaults"]["positive_partners_per_anchor"] == 1
    assert capabilities["semantic_defaults"]["zero_partners_per_anchor"] == 1
    assert capabilities["tokenizer_constraints"]["kmer"] == {
        "k": {"min": 1, "max": 3},
        "max_length_unit": "tokens",
    }
    assert capabilities["tokenizer_constraints"]["wordpiece"] == {
        "max_length_unit": "tokens_including_special_tokens",
        "max_raw_length": {
            "default": 50_000,
            "absolute_max": 100_000,
        },
    }
    assert capabilities["frequency_encoder_contract"] == {
        "version": 2,
        "encoders": ["bow", "tfidf"],
        "tokenizers": [
            "kmer",
            "dr",
            "top-n-gram",
            "dt",
        ],
        "max_raw_length": {
            "default": 50_000,
            "absolute_max": 100_000,
        },
        "stream_policy": "full_token_stream",
    }
    pretrained_sequence = capabilities["route_constraints"]["single_sequence"]["pretrained"]
    assert capabilities["modeling_architectures"] == {
        "sequence": ["cnn", "rnn", "gru", "lstm", "transformer"],
        "chain_graph": ["gcn", "gat"],
    }
    assert capabilities["sequence_models"] == [
        "cnn",
        "rnn",
        "gru",
        "lstm",
        "transformer",
        "gcn",
        "gat",
    ]
    assert capabilities["covariance_encoder_contract"] == {
        "version": 1,
        "encoders": ["ac", "cc", "acc"],
        "tokenizer": "kmer",
        "required_k": 1,
        "properties": ["H1", "H2", "V", "P1", "P2", "SASA", "NCI"],
        "max_raw_length": {
            "default": 100_000,
            "absolute_max": 100_000,
        },
        "stream_policy": "complete_raw_sequence",
    }
    assert capabilities["pdt_encoder_contract"] == {
        "version": 1,
        "encoders": ["pdt"],
        "tokenizer": "kmer",
        "required_k": 1,
        "default_lag": 8,
        "dimensions_per_lag": 531,
        "method_source_doi": "10.1371/journal.pone.0046633",
        "max_raw_length": {
            "default": 100_000,
            "absolute_max": 100_000,
        },
        "stream_policy": "complete_raw_sequence",
    }
    assert capabilities["pssm_covariance_encoder_contract"] == {
        "version": 1,
        "encoders": [
            "ac_pssm",
            "cc_pssm",
            "acc_pssm",
            "ac_pssm_blosum62_hybrid",
            "cc_pssm_blosum62_hybrid",
            "acc_pssm_blosum62_hybrid",
        ],
        "strict_encoders": ["ac_pssm", "cc_pssm", "acc_pssm"],
        "hybrid_encoders": [
            "ac_pssm_blosum62_hybrid",
            "cc_pssm_blosum62_hybrid",
            "acc_pssm_blosum62_hybrid",
        ],
        "tokenizer": "kmer",
        "required_k": 1,
        "default_lag": 10,
        "dimensions_per_lag": {
            "ac_pssm": 20,
            "cc_pssm": 380,
            "acc_pssm": 400,
            "ac_pssm_blosum62_hybrid": 20,
            "cc_pssm_blosum62_hybrid": 380,
            "acc_pssm_blosum62_hybrid": 400,
        },
        "lag_keys": {
            "ac_pssm": "ac_pssm_lag",
            "cc_pssm": "cc_pssm_lag",
            "acc_pssm": "acc_pssm_lag",
            "ac_pssm_blosum62_hybrid": "ac_pssm_blosum62_hybrid_lag",
            "cc_pssm_blosum62_hybrid": "cc_pssm_blosum62_hybrid_lag",
            "acc_pssm_blosum62_hybrid": "acc_pssm_blosum62_hybrid_lag",
        },
        "profile_feature_types": {
            "ac_pssm": "pssm",
            "cc_pssm": "pssm",
            "acc_pssm": "pssm",
            "ac_pssm_blosum62_hybrid": "ac_pssm_blosum62_hybrid",
            "cc_pssm_blosum62_hybrid": "cc_pssm_blosum62_hybrid",
            "acc_pssm_blosum62_hybrid": "acc_pssm_blosum62_hybrid",
        },
        "stream_policy": "complete_pssm_profile",
    }
    assert {"ac", "cc", "acc"}.issubset(capabilities["classic_encoder_options"]["kmer"])
    assert (
        capabilities["sequence_profile_contract"]["explicit_hybrid_feature"]
        == "pssm_blosum62_hybrid"
    )
    assert (
        "psfm_blosum62_hybrid"
        in capabilities["sequence_profile_contract"]["explicit_hybrid_encoders"]
    )
    assert (
        capabilities["sequence_profile_contract"]["fallback_policies"]["psfm_blosum62_hybrid"]
        == "blosum62_hybrid"
    )
    assert "mismatch" not in capabilities["classic_tokenizers"]
    assert "mismatch" not in capabilities["classic_encoder_options"]
    assert "cs" not in capabilities["classic_encoder_options"]["kmer"]
    assert (
        "index"
        not in capabilities["route_constraints"]["single_sequence"]["classic"][
            "chain_graph_encoders"
        ]
    )
    expected_tokenizers = {"esm2", "protbert", "prott5", "esmc", "esm3"}
    assert set(pretrained_sequence["residue_architectures"]) == expected_tokenizers
    assert all(
        values == ["cnn", "rnn", "gru", "lstm", "transformer", "gcn", "gat"]
        for values in pretrained_sequence["residue_architectures"].values()
    )
    assert all(
        values == ["end_to_end", "export_embeddings"]
        for values in pretrained_sequence["residue_training_modes"].values()
    )
    assert pretrained_sequence["residue_adapter_dim"] == {
        "default": 128,
        "min": 16,
        "max": 1024,
        "multiple_of": 4,
    }
    assert pretrained_sequence["residue_encoder_frozen"] is True
    assert set(pretrained_sequence["residue_graph_predictors"]) == expected_tokenizers
    assert pretrained_sequence["unsupported_post_encoders"] == []
    assert capabilities["route_schema_version"] == 6
    pair_token = capabilities["route_constraints"]["pair"]["pretrained_token_interaction"]
    assert pair_token["interaction_models"] == ["cross_attn", "bi_cross_attn"]
