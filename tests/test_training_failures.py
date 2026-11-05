import json

from pepplatform.sequence_profiles import SequenceProfileError
from pepplatform.ml_trainers import SVMArtifactContractError, SVMConvergenceError
from pepplatform.training_failures import (
    TRAINING_FAILURE_PREFIX,
    failure_from_exception,
    failure_from_process_exit,
    finalize_training_failure,
    parse_training_failure_line,
    prefer_training_failure,
)


def test_structured_profile_failure_round_trips_with_context():
    failure = failure_from_exception(
        SequenceProfileError(
            "no usable homologs",
            code="profile_no_nonself_hits",
            details={
                "feature": "psfm",
                "sample_id": "Train5",
                "sequence_length": 5,
                "zero_nonself_hits": True,
            },
        )
    )
    line = f"{TRAINING_FAILURE_PREFIX} " + json.dumps(failure, ensure_ascii=False)

    parsed = parse_training_failure_line(line)
    finalized = finalize_training_failure(parsed, exit_code=1)

    assert finalized == {
        "schema_version": 1,
        "code": "profile_no_nonself_hits",
        "category": "profile",
        "stage": "profile_generation",
        "message": "no usable homologs",
        "context": {
            "feature": "psfm",
            "sample_id": "Train5",
            "sequence_length": 5,
            "zero_nonself_hits": True,
        },
        "exit_code": 1,
    }


def test_specific_child_failure_wins_over_generic_parent_exit():
    specific = failure_from_exception(
        SequenceProfileError(
            "strict PSSM has no homologs",
            code="profile_no_nonself_hits",
        )
    )
    generic = failure_from_exception(SystemExit("profile cache build failed"))

    selected = prefer_training_failure(specific, generic)

    assert selected["code"] == "profile_no_nonself_hits"


def test_svm_convergence_failure_preserves_optimization_context():
    failure = failure_from_exception(
        SVMConvergenceError(
            "SVM did not converge",
            details={
                "classifier": "svm",
                "output_number": 2,
                "output_count": 5,
                "attempted_max_iters": "2000 → 10000 → 50000",
                "actual_n_iters": "2000 → 10000 → 50000",
                "tol": 0.001,
                "pipeline_stage": "downstream_classifier",
            },
        )
    )

    assert failure["code"] == "svm_convergence_failed"
    assert failure["category"] == "optimization"
    assert failure["stage"] == "training"
    assert failure["context"]["output_number"] == 2
    assert failure["context"]["actual_n_iters"] == "2000 → 10000 → 50000"
    assert failure["context"]["pipeline_stage"] == "downstream_classifier"


def test_svm_artifact_contract_failure_is_not_misclassified_as_data():
    failure = failure_from_exception(
        SVMArtifactContractError("contract could not represent fitted outputs")
    )

    assert failure["code"] == "svm_artifact_contract_failed"
    assert failure["category"] == "runtime"
    assert failure["stage"] == "artifact_export"
    assert failure["context"]["pipeline_stage"] == "artifact_export"


def test_process_exit_fallback_recognizes_cuda_oom_and_unknown_exit():
    oom = failure_from_process_exit(
        1,
        ["RuntimeError: CUDA out of memory. Tried to allocate 1.00 GiB"],
    )
    unknown = failure_from_process_exit(7, ["unexpected native failure"])

    assert oom["code"] == "cuda_out_of_memory"
    assert oom["category"] == "resource"
    assert unknown["code"] == "process_exit_nonzero"
    assert unknown["exit_code"] == 7
