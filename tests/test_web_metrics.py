from pepplatform.web.metrics import parse_metric_line, rebuild_metric_summary


def test_macro_and_micro_aupr_remain_distinct_in_final_logs():
    # Actual multilabel logs put the micro field before the aggregate field.
    for prefix, aggregate in [("[tag] Test |", "test"), ("RF Val", "val")]:
        base = (
            f"{prefix} Acc: 0.2803 | SAcc: 0.4912 | F1: 0.3989 | AUC: 0.8391 | "
            "Micro AUC: 0.9136 | Micro AUPR: 0.6253 | "
            "AUPRMethod: trapezoidal_pr_auc | MetricNamespace: 2"
        )
        micro_only = parse_metric_line(base)
        assert f"{aggregate}_aupr" not in micro_only
        assert micro_only[f"{aggregate}_aupr_micro"] == 0.6253
        combined = parse_metric_line(base + " | AUPR: 0.3933")
        assert combined[f"{aggregate}_aupr"] == 0.3933
        assert combined[f"{aggregate}_aupr_micro"] == 0.6253


def test_parse_structured_run_progress_event():
    assert parse_metric_line(
        "[run-progress] stage=classifier_fit status=running current=240 total=800 "
        "feature=lgb unit=tree output=3 outputs=8 iteration=40 iterations=100 "
        "threads=4 elapsed_seconds=12.5 eta_seconds=29.167"
    ) == {
        "kind": "stage",
        "stage": "classifier_fit",
        "status": "running",
        "current": 240,
        "total": 800,
        "output": 3,
        "outputs": 8,
        "iteration": 40,
        "iterations": 100,
        "threads": 4,
        "elapsed_seconds": 12.5,
        "eta_seconds": 29.167,
        "feature": "lgb",
        "unit": "tree",
    }


def test_ignore_invalid_structured_run_progress_event():
    assert (
        parse_metric_line(
            "[run-progress] stage=classifier_fit status=running current=11 total=10 "
            "output=1 outputs=2 iteration=1 iterations=10 threads=4"
        )
        is None
    )
    assert (
        parse_metric_line(
            "[run-progress] stage=classifier_fit status=running current=1 total=10 "
            "output=1 outputs=2 iteration=1 threads=4"
        )
        is None
    )


def test_parse_tuning_progress_event():
    assert parse_metric_line(
        "[tuning-progress] classifier=rf status=running current=6 total=36 candidate=2 candidates=12 folds=3"
    ) == {
        "kind": "stage",
        "stage": "hyperparameter_tuning",
        "status": "running",
        "feature": "rf",
        "current": 6,
        "total": 36,
        "candidate": 2,
        "candidates": 12,
        "folds": 3,
    }


def test_parse_svm_bounded_retry_progress_event():
    assert parse_metric_line(
        "[svm-progress] classifier=svm status=retrying current=2 total=4 "
        "attempt=2 attempts=3 max_iter=10000"
    ) == {
        "kind": "stage",
        "stage": "svm_fit",
        "status": "retrying",
        "feature": "svm",
        "current": 2,
        "total": 4,
        "attempt": 2,
        "attempts": 3,
        "max_iter": 10000,
        "recovered": False,
    }

    completed = parse_metric_line(
        "[svm-progress] classifier=svm status=complete current=2 total=4 "
        "attempt=2 attempts=3 max_iter=10000 recovered=true"
    )
    assert completed["recovered"] is True


def test_ignore_invalid_svm_progress_event():
    assert (
        parse_metric_line(
            "[svm-progress] classifier=svm status=running current=3 total=2 "
            "attempt=1 attempts=3 max_iter=2000"
        )
        is None
    )


def test_parse_profile_progress_stage_line():
    assert parse_metric_line(
        "[profile-progress] stage=profiles status=running current=42 total=12574 "
        "feature=pssm+psfm elapsed_seconds=18.25"
    ) == {
        "kind": "stage",
        "stage": "profiles",
        "status": "running",
        "current": 42,
        "total": 12574,
        "feature": "pssm+psfm",
        "elapsed_seconds": 18.25,
    }


def test_parse_legacy_profile_progress_without_elapsed_time():
    assert parse_metric_line(
        "[profile-progress] stage=database status=complete current=1 total=1 feature=pssm"
    ) == {
        "kind": "stage",
        "stage": "database",
        "status": "complete",
        "current": 1,
        "total": 1,
        "feature": "pssm",
    }


def test_ignore_invalid_profile_progress_stage_line():
    assert (
        parse_metric_line(
            "[profile-progress] stage=profiles status=running current=11 total=10 feature=pssm"
        )
        is None
    )

    assert (
        parse_metric_line(
            "[profile-progress] stage=profiles status=running current=1 total=10 "
            "feature=pssm elapsed_seconds=-1"
        )
        is None
    )


def test_parse_epoch_metric_line():
    event = parse_metric_line(
        "[kmer/index/cnn] Epoch 2/10 | Train Loss: 0.1234 | Val Loss: 0.2222 | "
        "Val Acc: 0.7000 | Val SAcc: 0.6500 | Val F1: 0.6000 | Val AUC: 0.8000"
    )
    assert event == {
        "kind": "epoch",
        "epoch": 2,
        "total": 10,
        "train_loss": 0.1234,
        "val_loss": 0.2222,
        "val_acc": 0.7,
        "val_sacc": 0.65,
        "val_f1": 0.6,
        "val_auc": 0.8,
    }


def test_parse_epoch_metric_line_with_aupr():
    event = parse_metric_line(
        "[kmer/index/cnn] Epoch 3/10 | Train Loss: 0.1000 | Val Loss: 0.2000 | "
        "Val Acc: 0.7000 | Val SAcc: 0.6500 | Val F1: 0.6000 | Val AUC: 0.8000 | Val AUPR: 0.8100"
    )
    assert event["kind"] == "epoch"
    assert event["val_aupr"] == 0.81


def test_parse_multilabel_micro_metrics_without_overwriting_macro_primary():
    event = parse_metric_line(
        "[kmer/index/cnn] Epoch 3/10 | Train Loss: 0.1000 | Val Acc: 0.7000 | "
        "Val SAcc: 0.6500 | Val F1: 0.6000 | Val AUC: 0.8000 | "
        "Val Micro AUC: 0.9000 | Val Micro AUPR: 0.8500 | Val AUPR: 0.7000"
    )
    assert event["val_auc"] == 0.8
    assert event["val_aupr"] == 0.7
    assert event["val_auc_micro"] == 0.9
    assert event["val_aupr_micro"] == 0.85


def test_parse_epoch_metric_line_with_extended_metrics():
    event = parse_metric_line(
        "[kmer/index/cnn] Epoch 3/10 | Train Loss: 0.1000 | Val Loss: 0.2000 | "
        "Val Acc: 0.7000 | Val SAcc: 0.6500 | Val F1: 0.6000 | Val AUC: 0.8000 | "
        "Val Balanced Accuracy: 0.7100 | Val Precision: 0.7200 | "
        "Val Recall (Sensitivity): 0.7300 | Val Specificity: 0.7400 | Val MCC: 0.7500 | Val AUPR: 0.8100"
    )
    assert event["kind"] == "epoch"
    assert event["val_balanced_accuracy"] == 0.71
    assert event["val_precision"] == 0.72
    assert event["val_recall"] == 0.73
    assert event["val_specificity"] == 0.74
    assert event["val_mcc"] == 0.75
    assert event["val_aupr"] == 0.81


def test_parse_graph_epoch_metric_line_without_val_loss():
    event = parse_metric_line(
        "[esm2+ppi_fusion/gcn] Epoch 9/10 | Train Loss: 0.6374 | Val Acc: 0.6250 | "
        "Val SAcc: 0.6250 | Val F1: 0.6969 | Val AUC: 0.7399 | Val AUPR: 0.7246 | "
        "ValThr: 0.42 | GCN=sample_graph"
    )
    assert event == {
        "kind": "epoch",
        "epoch": 9,
        "total": 10,
        "train_loss": 0.6374,
        "val_acc": 0.625,
        "val_sacc": 0.625,
        "val_f1": 0.6969,
        "val_auc": 0.7399,
        "val_aupr": 0.7246,
    }


def test_parse_best_and_test_metric_lines():
    assert parse_metric_line("[tag] Finished. Best Val F1: 0.8123 (Epoch 7)") == {
        "kind": "best",
        "best_val_f1": 0.8123,
        "best_epoch": 7,
    }
    assert parse_metric_line("[tag] Finished. Best Val F1: 0.0000 (Epoch 1)") == {
        "kind": "best",
        "best_val_f1": 0.0,
        "best_epoch": 1,
    }
    assert parse_metric_line("[tag] Finished. Best Val AUC: 0.9012 (Epoch 5)") == {
        "kind": "best",
        "best_val_auc": 0.9012,
        "best_epoch": 5,
    }
    assert parse_metric_line(
        "[pair] Test | TestSet: test.csv | Acc: 0.9000 | SAcc: 0.9000 | "
        "F1: 0.8800 | AUC: 0.9100 | Balanced Accuracy: 0.9300 | Precision: 0.9400 | "
        "Recall (Sensitivity): 0.9500 | Specificity: 0.9600 | MCC: 0.9700 | AUPR: 0.9200"
    ) == {
        "kind": "test",
        "test_set": "test.csv",
        "test_acc": 0.9,
        "test_sacc": 0.9,
        "test_f1": 0.88,
        "test_auc": 0.91,
        "test_balanced_accuracy": 0.93,
        "test_precision": 0.94,
        "test_recall": 0.95,
        "test_specificity": 0.96,
        "test_mcc": 0.97,
        "test_aupr": 0.92,
    }


def test_parse_classical_validation_metric_lines():
    for model_name in ["SVM", "CalibratedSVM", "RF", "ExtraTrees", "HistGB", "LogReg", "XGBoost"]:
        assert parse_metric_line(
            f"{model_name} Val Acc: 0.3079 | SAcc: 0.5039 | F1: 0.4300 | AUC: 0.8812"
        ) == {
            "kind": "validation",
            "validation_model": model_name,
            "val_acc": 0.3079,
            "val_sacc": 0.5039,
            "val_f1": 0.43,
            "val_auc": 0.8812,
        }
    assert parse_metric_line(
        "LightGBM Val Acc: 0.5000 | SAcc: 0.6100 | F1: 0.7200 | AUC: 0.8300 | "
        "Balanced Accuracy: 0.8500 | Precision: 0.8600 | Recall: 0.8700 | "
        "Specificity: 0.8800 | MCC: 0.8900 | AUPR: 0.8400"
    ) == {
        "kind": "validation",
        "validation_model": "LightGBM",
        "val_acc": 0.5,
        "val_sacc": 0.61,
        "val_f1": 0.72,
        "val_auc": 0.83,
        "val_balanced_accuracy": 0.85,
        "val_precision": 0.86,
        "val_recall": 0.87,
        "val_specificity": 0.88,
        "val_mcc": 0.89,
        "val_aupr": 0.84,
    }


def test_rebuild_metric_summary_uses_all_metrics_from_explicit_best_epoch():
    from pepplatform.pr_metrics import metric_definition
    events = [
        {"kind": "epoch", "epoch": 1, "val_f1": 0.7, "val_auc": 0.91, "val_aupr": 0.8},
        {"kind": "epoch", "epoch": 2, "val_f1": 0.8, "val_auc": 0.85, "val_aupr": 0.75},
        {"kind": "best", "best_epoch": 2, "best_val_f1": 0.8},
        {"kind": "test", "test_f1": 0.76},
    ]

    summary = rebuild_metric_summary([{**event, **metric_definition()} for event in events])

    assert summary["best_epoch"] == 2
    assert summary["val_f1"] == 0.8
    assert summary["val_auc"] == 0.85
    assert summary["val_aupr"] == 0.75
    assert summary["test_f1"] == 0.76
    assert summary["kind"] == "test"


def test_rebuild_metric_summary_uses_earliest_best_f1_epoch_until_best_event_arrives():
    summary = rebuild_metric_summary(
        [
            {"kind": "epoch", "epoch": 3, "val_f1": 0.8, "val_auc": 0.7},
            {"kind": "epoch", "epoch": 2, "val_f1": 0.8, "val_auc": 0.75},
            {"kind": "epoch", "epoch": 4, "val_f1": 0.7, "val_auc": 0.9},
        ]
    )

    assert summary["best_epoch"] == 2
    assert summary["best_val_f1"] == 0.8
    assert summary["val_auc"] == 0.75


def test_rebuild_metric_summary_uses_auc_fallback_for_binary_pair_tasks():
    summary = rebuild_metric_summary(
        [
            {"kind": "epoch", "epoch": 1, "val_f1": 0.8, "val_auc": 0.7},
            {"kind": "epoch", "epoch": 2, "val_f1": 0.7, "val_auc": 0.9},
        ],
        existing_summary={"best_epoch": 1, "best_val_f1": 0.8},
        task_type="pep_protein_interaction",
    )

    assert summary["best_epoch"] == 2
    assert summary["best_val_auc"] == 0.9
    assert "best_val_f1" not in summary
    assert summary["val_f1"] == 0.7


def test_rebuild_metric_summary_does_not_invent_binary_checkpoint_without_auc():
    summary = rebuild_metric_summary(
        [{"kind": "epoch", "epoch": 1, "val_f1": 0.8}],
        task_type="pep_hlai",
    )

    assert "best_epoch" not in summary
    assert "best_val_f1" not in summary
    assert "best_val_auc" not in summary
