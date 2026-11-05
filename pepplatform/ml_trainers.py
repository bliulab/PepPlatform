from pepplatform.task_compat import normalize_task_type

import json
import hashlib
import os
import sys
import tempfile
import time
import warnings
from itertools import product
from pathlib import Path

import joblib
import numpy as np
from pepplatform.objectives import resolve_selection, selection_fields
import sklearn
import torch
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm as _tqdm
from pepplatform.analysis import build_classical_analysis, save_analysis
from pepplatform.evaluators import EVALUATION_PROTOCOL_VERSION, Evaluator, OperatingPoint
from pepplatform.lsa import (
    LSA_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
    fit_lsa_transform,
    validate_lsa_route,
)
from pepplatform.tsvd import (
    TSVD_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
    fit_tsvd_transform,
)
from pepplatform.pca import (
    PCA_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
    fit_pca_transform,
)
from pepplatform.kpca import (
    KPCA_DEFAULT_MEMORY_LIMIT_GIB,
    fit_kpca_transform,
)
from pepplatform.lda import (
    LDA_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
    fit_lda_transform,
    validate_lda_route,
)
from pepplatform.plsa import (
    PLSA_DEFAULT_MEMORY_LIMIT_GIB,
    fit_plsa_transform,
    validate_plsa_route,
)
from pepplatform.route_registry import ROUTE_SCHEMA_VERSION, resolve_effective_route
from pepplatform.run_progress import emit_run_progress
from pepplatform.workflow import (
    LGB_EXECUTION_POLICY,
    ML_TUNING_BUDGETS,
    ML_TUNING_MAX_CANDIDATES,
    ML_TUNING_SPACE_VERSION,
    PROFILE_DEPENDENT_INFERENCE_REASON,
    SVM_CALIBRATION_OPTIONS,
    SVM_CALIBRATION_CV_FOLDS,
    default_model_selection_objective,
    uses_profile_features,
)


def tqdm(*args, **kwargs):
    """Keep terminal progress bars interactive without polluting piped Web logs."""
    kwargs.setdefault("disable", not sys.stderr.isatty())
    return _tqdm(*args, **kwargs)


try:
    import xgboost
    from xgboost import XGBClassifier
except ImportError:
    xgboost = None
    XGBClassifier = None
try:
    import lightgbm
    from lightgbm import Booster as LightGBMBooster
    from lightgbm import LGBMClassifier
except ImportError:
    lightgbm = None
    LightGBMBooster = None
    LGBMClassifier = None


RF_PREDICTOR_CONTRACT_VERSION = 1
EXTRATREES_PREDICTOR_CONTRACT_VERSION = 1
SVM_PREDICTOR_CONTRACT_VERSION = 1
CALIBRATED_SVM_PREDICTOR_CONTRACT_VERSION = 1
XGB_PREDICTOR_CONTRACT_VERSION = 1
XGB_MANIFEST_SCHEMA_VERSION = 1
LGB_PREDICTOR_CONTRACT_VERSION = 1
LGB_MANIFEST_SCHEMA_VERSION = 1
LOGREG_PREDICTOR_CONTRACT_VERSION = 1
HISTGB_PREDICTOR_CONTRACT_VERSION = 1


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_joblib_dump(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=destination.suffix, dir=destination.parent
    )
    os.close(descriptor)
    try:
        joblib.dump(payload, temporary_name)
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return {
        "path": destination.name,
        "sha256": _sha256_file(destination),
        "size_bytes": int(destination.stat().st_size),
        "publication": "atomic_replace",
    }


def _xgb_effective_config(estimator):
    if XGBClassifier is None or not isinstance(estimator, XGBClassifier):
        raise TypeError("XGB predictor contract requires XGBClassifier estimators")
    config = json.loads(estimator.get_booster().save_config())
    learner = config.get("learner") or {}
    gradient_booster = learner.get("gradient_booster") or {}
    tree_params = gradient_booster.get("tree_train_param") or {}
    objective = learner.get("objective") or {}
    objective_params = objective.get("reg_loss_param") or {}
    return {
        "booster": str(gradient_booster.get("name") or ""),
        "tree_method": str(
            (gradient_booster.get("gbtree_train_param") or {}).get("tree_method") or ""
        ),
        "objective": str(objective.get("name") or ""),
        "num_boosted_rounds": int(estimator.get_booster().num_boosted_rounds()),
        "max_depth": int(tree_params.get("max_depth")),
        "learning_rate": float(tree_params.get("learning_rate")),
        "subsample": float(tree_params.get("subsample")),
        "colsample_bytree": float(tree_params.get("colsample_bytree")),
        "reg_alpha": float(tree_params.get("reg_alpha")),
        "reg_lambda": float(tree_params.get("reg_lambda")),
        "scale_pos_weight": float(objective_params.get("scale_pos_weight")),
    }


def xgb_predictor_contract(estimators, *, input_feature_count):
    """Describe the fitted per-output XGBoost primitives used by new artifacts."""
    if xgboost is None:
        raise ImportError("XGBoost is not installed")
    wrapped = list(estimators or [])
    values = [
        item.get("value")
        for item in wrapped
        if isinstance(item, dict) and item.get("kind") == "sklearn"
    ]
    if len(values) != len(wrapped) or not values:
        raise TypeError(
            "XGB predictor contract requires fitted XGBClassifier outputs without constant fallbacks"
        )
    if not all(isinstance(value, XGBClassifier) for value in values):
        raise TypeError("XGB predictor contract requires homogeneous XGBClassifier outputs")
    effective = [_xgb_effective_config(value) for value in values]
    for item in effective:
        if item["booster"] != "gbtree" or item["tree_method"] != "hist":
            raise ValueError("XGB predictor contract requires the gbtree hist booster")
        if item["objective"] != "binary:logistic":
            raise ValueError("XGB predictor contract requires binary:logistic probability outputs")
    first_params = values[0].get_params(deep=False)
    shared_keys = [
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "objective",
        "eval_metric",
        "tree_method",
        "random_state",
        "n_jobs",
    ]
    for value in values[1:]:
        params = value.get_params(deep=False)
        if any(params.get(key) != first_params.get(key) for key in shared_keys):
            raise ValueError(
                "XGB predictor contract requires shared model parameters across outputs"
            )
    return {
        "contract_version": XGB_PREDICTOR_CONTRACT_VERSION,
        "algorithm": "xgboost",
        "implementation": "xgboost.XGBClassifier",
        "implementation_version": str(xgboost.__version__),
        "sklearn_version": sklearn.__version__,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "status": "verified_platform_variant",
        "input_contract": "finite_fixed_width_global_vector",
        "input_feature_count": int(input_feature_count),
        "architecture": "independent_binary_regularized_gradient_tree_booster_per_output",
        "booster": "gbtree",
        "tree_method": "hist",
        "objective": "binary:logistic",
        "score_kind": "probability",
        "output": "positive_class_probability",
        "calibrated": False,
        "multioutput": "independent_binary_one_vs_rest",
        "num_outputs": len(values),
        "estimator_params": {key: first_params.get(key) for key in shared_keys},
        "effective_params_by_output": effective,
        "serialization": {
            "format": "joblib_preprocessors_plus_xgboost_ubjson",
            "manifest_path": "xgb_manifest.json",
            "manifest_schema_version": XGB_MANIFEST_SCHEMA_VERSION,
            "requires_matching_sklearn_version": True,
            "xgboost_native_model": True,
        },
    }


class XGBTrainingDataError(ValueError):
    """Raised when an XGBoost-labelled run cannot fit an XGBoost model."""

    code = "xgb_training_data_invalid"

    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = {
            "classifier": "xgb",
            "pipeline_stage": "model_training",
            **dict(details or {}),
        }


class XGBArtifactContractError(ValueError):
    """Raised when a fitted XGBoost ensemble cannot be exported safely."""

    code = "xgb_artifact_contract_failed"

    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = {
            "classifier": "xgb",
            "pipeline_stage": "artifact_export",
            **dict(details or {}),
        }


def _lgb_effective_config(estimator):
    if LGBMClassifier is None or not isinstance(estimator, LGBMClassifier):
        raise TypeError("LGB predictor contract requires LGBMClassifier estimators")
    booster = estimator.booster_
    params = dict(booster.params or {})
    return {
        "boosting_type": str(params.get("boosting_type") or ""),
        "objective": str(params.get("objective") or ""),
        "requested_iterations": int(params.get("num_iterations")),
        "fitted_iterations": int(estimator.n_iter_),
        "num_trees": int(booster.num_trees()),
        "num_leaves": int(params.get("num_leaves")),
        "max_depth": int(params.get("max_depth")),
        "learning_rate": float(params.get("learning_rate")),
        "subsample": float(params.get("subsample")),
        "subsample_freq": int(params.get("subsample_freq")),
        "colsample_bytree": float(params.get("colsample_bytree")),
        "reg_alpha": float(params.get("reg_alpha")),
        "reg_lambda": float(params.get("reg_lambda")),
        "scale_pos_weight": float(params.get("scale_pos_weight")),
        "random_state": int(params.get("random_state")),
        "bagging_seed": int(params.get("bagging_seed")),
        "feature_fraction_seed": int(params.get("feature_fraction_seed")),
        "data_random_seed": int(params.get("data_random_seed")),
        "num_threads": int(params.get("num_threads")),
        "deterministic": bool(params.get("deterministic")),
        "force_row_wise": bool(params.get("force_row_wise")),
        "device_type": str(params.get("device_type") or ""),
    }


def lgb_predictor_contract(estimators, *, input_feature_count):
    """Describe fitted per-output LightGBM primitives used by new artifacts."""
    if lightgbm is None:
        raise ImportError("LightGBM is not installed")
    wrapped = list(estimators or [])
    values = [
        item.get("value")
        for item in wrapped
        if isinstance(item, dict) and item.get("kind") == "sklearn"
    ]
    if len(values) != len(wrapped) or not values:
        raise TypeError(
            "LGB predictor contract requires fitted LGBMClassifier outputs "
            "without constant fallbacks"
        )
    if not all(isinstance(value, LGBMClassifier) for value in values):
        raise TypeError("LGB predictor contract requires homogeneous LGBMClassifier outputs")
    effective = [_lgb_effective_config(value) for value in values]
    for item in effective:
        if item["boosting_type"] != "gbdt":
            raise ValueError("LGB predictor contract requires the LightGBM gbdt booster")
        if item["objective"] != "binary":
            raise ValueError("LGB predictor contract requires the binary objective")
        if (
            item["num_threads"] != LGB_EXECUTION_POLICY["lgb_n_jobs"]
            or item["deterministic"] is not True
            or item["force_row_wise"] is not True
            or item["device_type"] != LGB_EXECUTION_POLICY["lgb_device_type"]
        ):
            raise ValueError(
                "LGB predictor contract requires the deterministic CPU execution policy"
            )
    first_params = values[0].get_params(deep=False)
    shared_keys = [
        "boosting_type",
        "num_leaves",
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "subsample_freq",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "objective",
        "random_state",
        "n_jobs",
        "bagging_seed",
        "feature_fraction_seed",
        "data_random_seed",
        "deterministic",
        "force_row_wise",
        "device_type",
    ]
    for value in values[1:]:
        params = value.get_params(deep=False)
        if any(params.get(key) != first_params.get(key) for key in shared_keys):
            raise ValueError(
                "LGB predictor contract requires shared model parameters across outputs"
            )
    return {
        "contract_version": LGB_PREDICTOR_CONTRACT_VERSION,
        "algorithm": "lightgbm",
        "implementation": "lightgbm.LGBMClassifier",
        "implementation_version": str(lightgbm.__version__),
        "sklearn_version": sklearn.__version__,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "status": "verified_platform_variant",
        "input_contract": "finite_fixed_width_global_vector",
        "input_feature_count": int(input_feature_count),
        "architecture": "independent_binary_lightgbm_gbdt_per_output",
        "booster": "gbdt",
        "objective": "binary",
        "score_kind": "probability",
        "output": "positive_class_probability",
        "calibrated": False,
        "multioutput": "independent_binary_one_vs_rest",
        "num_outputs": len(values),
        "execution_policy": {
            "n_jobs": LGB_EXECUTION_POLICY["lgb_n_jobs"],
            "deterministic": LGB_EXECUTION_POLICY["lgb_deterministic"],
            "histogram_mode": LGB_EXECUTION_POLICY["lgb_histogram_mode"],
            "device_type": LGB_EXECUTION_POLICY["lgb_device_type"],
            "seed": int(first_params.get("random_state")),
            "child_seed_policy": "experiment_seed",
        },
        "estimator_params": {key: first_params.get(key) for key in shared_keys},
        "effective_params_by_output": effective,
        "serialization": {
            "format": "joblib_preprocessors_plus_lightgbm_native_text",
            "manifest_path": "lgb_manifest.json",
            "manifest_schema_version": LGB_MANIFEST_SCHEMA_VERSION,
            "requires_matching_sklearn_version": True,
            "lightgbm_native_model": True,
        },
    }


class LGBTrainingDataError(ValueError):
    """Raised when a LightGBM-labelled run cannot fit a LightGBM model."""

    code = "lgb_training_data_invalid"

    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = {
            "classifier": "lgb",
            "pipeline_stage": "model_training",
            **dict(details or {}),
        }


class LGBArtifactContractError(ValueError):
    """Raised when a fitted LightGBM ensemble cannot be exported safely."""

    code = "lgb_artifact_contract_failed"

    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = {
            "classifier": "lgb",
            "pipeline_stage": "artifact_export",
            **dict(details or {}),
        }


class HistGBTrainingDataError(ValueError):
    """Raised when a HistGB-labelled run cannot fit a HistGB model."""

    code = "histgb_training_data_invalid"

    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = {
            "classifier": "histgb",
            "pipeline_stage": "model_training",
            **dict(details or {}),
        }


class HistGBArtifactContractError(ValueError):
    """Raised when fitted HistGB outputs cannot satisfy the artifact contract."""

    code = "histgb_artifact_contract_failed"

    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = {
            "classifier": "histgb",
            "pipeline_stage": "artifact_export",
            **dict(details or {}),
        }


def histgb_predictor_contract(estimators, *, input_feature_count):
    """Describe fitted per-output sklearn histogram gradient boosters."""
    wrapped = list(estimators or [])
    values = [
        item.get("value")
        for item in wrapped
        if isinstance(item, dict) and item.get("kind") == "sklearn"
    ]
    if len(values) != len(wrapped) or not values:
        raise TypeError(
            "HistGB predictor contract requires fitted "
            "HistGradientBoostingClassifier outputs without constant fallbacks"
        )
    if not all(isinstance(value, HistGradientBoostingClassifier) for value in values):
        raise TypeError(
            "HistGB predictor contract requires homogeneous HistGradientBoostingClassifier outputs"
        )
    shared_keys = [
        "loss",
        "learning_rate",
        "max_iter",
        "max_leaf_nodes",
        "max_depth",
        "min_samples_leaf",
        "l2_regularization",
        "max_features",
        "max_bins",
        "categorical_features",
        "monotonic_cst",
        "interaction_cst",
        "warm_start",
        "early_stopping",
        "scoring",
        "validation_fraction",
        "n_iter_no_change",
        "tol",
        "random_state",
        "class_weight",
    ]
    first_params = values[0].get_params(deep=False)
    for value in values[1:]:
        params = value.get_params(deep=False)
        if any(params.get(key) != first_params.get(key) for key in shared_keys):
            raise ValueError(
                "HistGB predictor contract requires shared model parameters across outputs"
            )
    if (
        first_params.get("loss") != "log_loss"
        or first_params.get("early_stopping") is not False
        or first_params.get("warm_start") is not False
        or first_params.get("max_bins") != 255
        or first_params.get("categorical_features") != "from_dtype"
    ):
        raise ValueError(
            "PepPlatform HistGB contract requires fixed log-loss numeric histogram boosting"
        )
    expected_width = int(input_feature_count)
    fitted_iterations = []
    for output_index, value in enumerate(values):
        if int(getattr(value, "n_features_in_", 0) or 0) != expected_width:
            raise ValueError("HistGB fitted feature width does not match the predictor contract")
        if not np.array_equal(np.asarray(value.classes_), np.asarray([0, 1])):
            raise ValueError(
                f"HistGB output {output_index} must be fitted on binary classes 0 and 1"
            )
        if getattr(value, "do_early_stopping_", None) is not False:
            raise ValueError("HistGB predictor contract requires early stopping to be disabled")
        iteration_count = int(getattr(value, "n_iter_", 0) or 0)
        if iteration_count != int(first_params["max_iter"]):
            raise ValueError("HistGB fitted iteration count must equal the configured max_iter")
        if int(getattr(value, "n_trees_per_iteration_", 0) or 0) != 1:
            raise ValueError("HistGB binary outputs require one tree per boosting iteration")
        if getattr(value, "is_categorical_", None) is not None:
            raise ValueError("PepPlatform HistGB contract requires numeric input features")
        fitted_iterations.append(iteration_count)
    return {
        "contract_version": HISTGB_PREDICTOR_CONTRACT_VERSION,
        "algorithm": "histogram_gradient_boosting",
        "implementation": "sklearn.ensemble.HistGradientBoostingClassifier",
        "implementation_version": sklearn.__version__,
        "status": "verified_platform_variant",
        "input_contract": "numeric_fixed_width_global_vector_nan_missing_supported",
        "input_feature_count": expected_width,
        "architecture": "independent_binary_histogram_gradient_boosting_per_output",
        "weak_learner": "histogram_binned_regression_tree",
        "loss": "log_loss",
        "inverse_link": "logistic_sigmoid",
        "binning": "255_non_missing_bins_plus_missing_value_bin",
        "native_missing_values": True,
        "early_stopping": False,
        "score_kind": "probability",
        "output": "positive_class_probability",
        "calibrated": False,
        "multioutput": "independent_binary_one_vs_rest",
        "num_outputs": len(values),
        "fitted_iterations_by_output": fitted_iterations,
        "estimator_params": {key: first_params.get(key) for key in shared_keys},
        "serialization": {
            "format": "joblib",
            "requires_matching_sklearn_version": True,
        },
    }


class LogRegConvergenceError(RuntimeError):
    """Raised when LogisticRegression reaches every allowed iteration budget."""

    code = "logreg_convergence_failed"

    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = {
            "classifier": "logreg",
            "pipeline_stage": "model_training",
            **dict(details or {}),
        }


class LogRegArtifactContractError(ValueError):
    """Raised when fitted logistic regressions cannot satisfy the artifact contract."""

    code = "logreg_artifact_contract_failed"

    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = {
            "classifier": "logreg",
            "pipeline_stage": "artifact_export",
            **dict(details or {}),
        }


def logreg_predictor_contract(estimators, convergence=None, *, input_feature_count):
    """Describe the fitted per-output liblinear logistic regressions."""
    wrapped = list(estimators or [])
    values = [
        item.get("value")
        for item in wrapped
        if isinstance(item, dict) and item.get("kind") == "sklearn"
    ]
    if len(values) != len(wrapped) or not values:
        raise TypeError(
            "LogReg predictor contract requires fitted LogisticRegression outputs without constant fallbacks"
        )
    if not all(isinstance(value, LogisticRegression) for value in values):
        raise TypeError("LogReg predictor contract requires homogeneous LogisticRegression outputs")
    first_params = values[0].get_params(deep=False)
    shared_keys = [
        "C",
        "penalty",
        "solver",
        "class_weight",
        "tol",
        "fit_intercept",
        "intercept_scaling",
        "random_state",
    ]
    for value in values[1:]:
        params = value.get_params(deep=False)
        if any(params.get(key) != first_params.get(key) for key in shared_keys):
            raise ValueError(
                "LogReg predictor contract requires shared model parameters across outputs"
            )
    if first_params.get("solver") != "liblinear" or first_params.get("penalty") != "l2":
        raise ValueError("PepPlatform LogReg contract requires liblinear L2 logistic regression")
    effective_max_iters = [int(value.max_iter) for value in values]
    actual_iterations = [int(np.max(np.asarray(value.n_iter_))) for value in values]
    convergence_contract = {
        "required": True,
        "status": "converged",
        "n_iter_by_output": actual_iterations,
        "effective_max_iter_by_output": effective_max_iters,
    }
    if isinstance(convergence, dict):
        convergence_contract.update(convergence)
        convergence_contract["required"] = True
        convergence_contract["status"] = "converged"
        convergence_contract["n_iter_by_output"] = actual_iterations
        convergence_contract["effective_max_iter_by_output"] = effective_max_iters
    return {
        "contract_version": LOGREG_PREDICTOR_CONTRACT_VERSION,
        "algorithm": "logistic_regression",
        "implementation": "sklearn.linear_model.LogisticRegression",
        "implementation_version": sklearn.__version__,
        "status": "verified_platform_variant",
        "input_contract": "finite_fixed_width_global_vector",
        "input_feature_count": int(input_feature_count),
        "architecture": "independent_binary_l2_regularized_logistic_regression_per_output",
        "solver": "liblinear",
        "link": "logit",
        "penalty": "l2",
        "score_kind": "probability",
        "output": "positive_class_probability",
        "calibrated": False,
        "multioutput": "independent_binary_one_vs_rest",
        "num_outputs": len(values),
        "estimator_params": {
            **{key: first_params.get(key) for key in shared_keys},
            "requested_max_iter": int(
                (convergence or {}).get("requested_max_iter", effective_max_iters[0])
            ),
            "effective_max_iter_by_output": effective_max_iters,
        },
        "convergence": convergence_contract,
        "serialization": {
            "format": "joblib",
            "requires_matching_sklearn_version": True,
        },
    }


def rf_predictor_contract(estimator):
    """Describe the fitted scikit-learn RF primitive used by new artifacts."""
    if not isinstance(estimator, RandomForestClassifier):
        raise TypeError("RF predictor contract requires RandomForestClassifier")
    params = estimator.get_params(deep=False)
    return {
        "contract_version": RF_PREDICTOR_CONTRACT_VERSION,
        "algorithm": "random_forest",
        "implementation": "sklearn.ensemble.RandomForestClassifier",
        "implementation_version": sklearn.__version__,
        "status": "verified",
        "input_contract": "fixed_width_global_vector",
        "architecture": "bootstrap_tree_ensemble_with_random_feature_subsampling",
        "aggregation": "mean_class_probability",
        "score_kind": "probability",
        "calibrated": False,
        "multioutput": "native_sklearn_multi_output",
        "estimator_params": {
            "n_estimators": int(params["n_estimators"]),
            "criterion": str(params["criterion"]),
            "max_depth": params["max_depth"],
            "min_samples_split": params["min_samples_split"],
            "min_samples_leaf": params["min_samples_leaf"],
            "max_features": params["max_features"],
            "bootstrap": bool(params["bootstrap"]),
            "random_state": params["random_state"],
            "n_jobs": params["n_jobs"],
            "warm_start": bool(params["warm_start"]),
        },
        "serialization": {
            "format": "joblib",
            "requires_matching_sklearn_version": True,
        },
    }


def extratrees_predictor_contract(estimator):
    """Describe the fitted scikit-learn Extra Trees primitive used by new artifacts."""
    if not isinstance(estimator, ExtraTreesClassifier):
        raise TypeError("ExtraTrees predictor contract requires ExtraTreesClassifier")
    params = estimator.get_params(deep=False)
    if params.get("bootstrap") is not False:
        raise ValueError(
            "ExtraTrees predictor contract requires whole-sample trees without bootstrap"
        )
    if params.get("max_features") != "sqrt":
        raise ValueError("ExtraTrees predictor contract requires sqrt random feature subsampling")
    base_estimator = getattr(estimator, "estimator_", None)
    if base_estimator is None or getattr(base_estimator, "splitter", None) != "random":
        raise ValueError("ExtraTrees predictor contract requires randomized split thresholds")
    input_feature_count = getattr(estimator, "n_features_in_", None)
    if input_feature_count is None:
        raise ValueError("ExtraTrees predictor contract requires a fitted estimator")
    return {
        "contract_version": EXTRATREES_PREDICTOR_CONTRACT_VERSION,
        "algorithm": "extra_trees",
        "implementation": "sklearn.ensemble.ExtraTreesClassifier",
        "implementation_version": sklearn.__version__,
        "status": "verified",
        "input_contract": "fixed_width_global_vector",
        "input_feature_count": int(input_feature_count),
        "architecture": "whole_sample_extremely_randomized_tree_ensemble",
        "split_rule": "random_threshold_per_candidate_feature_then_best_impurity",
        "aggregation": "mean_class_probability",
        "score_kind": "probability",
        "calibrated": False,
        "multioutput": "native_sklearn_multi_output",
        "estimator_params": {
            "n_estimators": int(params["n_estimators"]),
            "criterion": str(params["criterion"]),
            "max_depth": params["max_depth"],
            "min_samples_split": params["min_samples_split"],
            "min_samples_leaf": params["min_samples_leaf"],
            "max_features": params["max_features"],
            "bootstrap": bool(params["bootstrap"]),
            "random_state": params["random_state"],
            "n_jobs": params["n_jobs"],
            "warm_start": bool(params["warm_start"]),
        },
        "serialization": {
            "format": "joblib",
            "requires_matching_sklearn_version": True,
        },
    }


class SVMConvergenceError(RuntimeError):
    """Raised when an SVM solver reaches its iteration limit without converging."""

    code = "svm_convergence_failed"

    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = dict(details or {})


class SVMArtifactContractError(ValueError):
    """Raised when a fitted SVM cannot be represented by the artifact contract."""

    code = "svm_artifact_contract_failed"

    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = {
            "classifier": "svm",
            "pipeline_stage": "artifact_export",
            **dict(details or {}),
        }


class SVMProbabilityContractError(ValueError):
    """Raised when a calibrated SVM cannot provide auditable probabilities."""

    code = "svm_probability_contract_failed"

    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = {
            "classifier": "svm",
            "pipeline_stage": "probability_calibration",
            **dict(details or {}),
        }


def svm_predictor_contract(estimators, convergence=None):
    """Describe the fitted per-output scikit-learn SVMs used by new artifacts."""
    estimators = list(estimators or [])
    if not estimators:
        raise ValueError("SVM predictor contract requires at least one fitted estimator")
    if all(isinstance(estimator, LinearSVC) for estimator in estimators):
        implementation = "sklearn.svm.LinearSVC"
        kernel = "linear"
        solver = "liblinear"
    elif all(isinstance(estimator, SVC) for estimator in estimators):
        if any(str(estimator.kernel).lower() != "rbf" for estimator in estimators):
            raise TypeError("PepPlatform SVC contracts require the RBF kernel")
        implementation = "sklearn.svm.SVC"
        kernel = "rbf"
        solver = "libsvm"
    else:
        raise TypeError(
            "SVM predictor contract requires homogeneous LinearSVC or RBF SVC estimators"
        )

    first = estimators[0]
    params = first.get_params(deep=False)
    # max_iter is an optimization budget, not part of the converged decision
    # function. Bounded recovery may therefore use a different effective budget
    # for each independent output while all model-defining parameters stay shared.
    shared_keys = ["C", "class_weight", "tol"]
    shared_keys.extend(
        ["penalty", "loss", "dual", "fit_intercept", "intercept_scaling", "random_state"]
        if kernel == "linear"
        else ["kernel", "gamma", "shrinking", "probability"]
    )
    for estimator in estimators[1:]:
        other = estimator.get_params(deep=False)
        if any(other[key] != params[key] for key in shared_keys):
            raise ValueError("SVM predictor contract requires shared parameters across outputs")
        if kernel == "rbf" and not np.isclose(estimator._gamma, first._gamma):
            raise ValueError("SVM predictor contract requires shared fitted gamma across outputs")
    effective_max_iters = [
        int(estimator.get_params(deep=False)["max_iter"]) for estimator in estimators
    ]
    requested_max_iter = int(params["max_iter"])
    if isinstance(convergence, dict):
        candidate = convergence.get("requested_max_iter")
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
            requested_max_iter = candidate
    shared = {
        "C": float(params["C"]),
        "class_weight": params["class_weight"],
        # Retained for contract-v1 consumers; this is the shared requested budget.
        "max_iter": requested_max_iter,
        "requested_max_iter": requested_max_iter,
        "effective_max_iter_by_output": effective_max_iters,
        "tol": float(params["tol"]),
    }
    if kernel == "linear":
        shared.update(
            {
                "penalty": str(params["penalty"]),
                "loss": str(params["loss"]),
                "dual": params["dual"],
                "fit_intercept": bool(params["fit_intercept"]),
                "intercept_scaling": float(params["intercept_scaling"]),
                "random_state": params["random_state"],
            }
        )
    else:
        shared.update(
            {
                "kernel": str(params["kernel"]),
                "gamma": params["gamma"],
                "fitted_gamma": float(first._gamma),
                "shrinking": bool(params["shrinking"]),
                "probability": bool(params["probability"]),
            }
        )
    iterations = [int(np.max(np.asarray(estimator.n_iter_))) for estimator in estimators]
    convergence_contract = {
        "required": True,
        "status": "converged",
        "n_iter_by_output": iterations,
        "effective_max_iter_by_output": effective_max_iters,
    }
    if isinstance(convergence, dict):
        convergence_contract.update(convergence)
        convergence_contract["required"] = True
        convergence_contract["status"] = "converged"
        convergence_contract["n_iter_by_output"] = iterations
        convergence_contract["effective_max_iter_by_output"] = effective_max_iters
    return {
        "contract_version": SVM_PREDICTOR_CONTRACT_VERSION,
        "algorithm": "support_vector_machine",
        "implementation": implementation,
        "implementation_version": sklearn.__version__,
        "status": "verified_platform_variant",
        "input_contract": "fixed_width_global_vector",
        "architecture": "independent_binary_estimator_per_output",
        "kernel": kernel,
        "solver": solver,
        "score_kind": "margin",
        "output": "decision_function_margin",
        "calibrated": False,
        "multioutput": "independent_binary_one_vs_rest",
        "num_outputs": len(estimators),
        "estimator_params": shared,
        "convergence": convergence_contract,
        "serialization": {
            "format": "joblib",
            "requires_matching_sklearn_version": True,
        },
    }


def calibrated_svm_predictor_contract(
    estimators,
    *,
    convergence,
    probability_quality,
    input_feature_count,
):
    """Describe fitted per-output SVMs with cross-validated Platt calibration."""
    wrapped = list(estimators or [])
    values = [
        item.get("value")
        for item in wrapped
        if isinstance(item, dict) and item.get("kind") == "sklearn"
    ]
    if len(values) != len(wrapped) or not values:
        raise TypeError(
            "calibrated SVM contract requires fitted CalibratedClassifierCV "
            "outputs without constant fallbacks"
        )
    if not all(isinstance(value, CalibratedClassifierCV) for value in values):
        raise TypeError(
            "calibrated SVM contract requires homogeneous CalibratedClassifierCV outputs"
        )

    base_estimators = []
    for output_index, value in enumerate(values):
        splitter = value.cv
        if (
            value.method != "sigmoid"
            or value.ensemble is not True
            or not isinstance(splitter, StratifiedKFold)
            or splitter.n_splits != 3
            or splitter.shuffle is not True
        ):
            raise ValueError(
                f"calibrated SVM output {output_index} does not use the fixed "
                "3-fold sigmoid calibration contract"
            )
        calibrated = list(getattr(value, "calibrated_classifiers_", []) or [])
        if len(calibrated) != 3:
            raise ValueError(
                f"calibrated SVM output {output_index} does not contain three fitted folds"
            )
        bases = [getattr(item, "estimator", None) for item in calibrated]
        if any(base is None for base in bases):
            raise ValueError(
                f"calibrated SVM output {output_index} is missing a fitted base estimator"
            )
        base_estimators.append(bases)

    flat_bases = [base for output in base_estimators for base in output]
    if all(isinstance(base, LinearSVC) for base in flat_bases):
        base_implementation = "sklearn.svm.LinearSVC"
        kernel = "linear"
        solver = "liblinear"
    elif all(isinstance(base, SVC) for base in flat_bases):
        if any(str(base.kernel).lower() != "rbf" for base in flat_bases):
            raise TypeError("calibrated SVC contracts require the RBF kernel")
        base_implementation = "sklearn.svm.SVC"
        kernel = "rbf"
        solver = "libsvm"
    else:
        raise TypeError("calibrated SVM contract requires homogeneous LinearSVC or RBF SVC bases")

    if not isinstance(convergence, dict) or convergence.get("status") != "converged":
        raise ValueError("calibrated SVM contract requires complete convergence evidence")
    convergence_outputs = convergence.get("outputs")
    if not isinstance(convergence_outputs, list) or len(convergence_outputs) != len(values):
        raise ValueError("calibrated SVM convergence output count is inconsistent")
    if (
        not isinstance(probability_quality, dict)
        or probability_quality.get("status") != "validated"
        or probability_quality.get("split") != "validation"
        or probability_quality.get("ece_bins") != 10
        or len(probability_quality.get("outputs") or []) != len(values)
    ):
        raise ValueError("calibrated SVM probability-quality evidence is incomplete")
    if not isinstance(input_feature_count, int) or input_feature_count <= 0:
        raise ValueError("calibrated SVM input feature count must be positive")

    first_params = flat_bases[0].get_params(deep=False)
    shared_keys = ["C", "class_weight", "tol"]
    shared_keys.extend(
        ["penalty", "loss", "dual", "fit_intercept", "intercept_scaling", "random_state"]
        if kernel == "linear"
        else ["kernel", "gamma", "shrinking", "probability"]
    )
    for base in flat_bases[1:]:
        params = base.get_params(deep=False)
        if any(params[key] != first_params[key] for key in shared_keys):
            raise ValueError(
                "calibrated SVM contract requires shared model parameters across folds and outputs"
            )
    estimator_params = {key: first_params[key] for key in shared_keys}
    estimator_params.update(
        {
            "requested_max_iter": int(convergence["requested_max_iter"]),
            "effective_max_iter_by_output": [
                int(item["effective_max_iter"]) for item in convergence_outputs
            ],
        }
    )
    return {
        "contract_version": CALIBRATED_SVM_PREDICTOR_CONTRACT_VERSION,
        "algorithm": "support_vector_machine",
        "implementation": "sklearn.calibration.CalibratedClassifierCV",
        "base_implementation": base_implementation,
        "implementation_version": sklearn.__version__,
        "status": "verified_platform_variant",
        "input_contract": "finite_fixed_width_global_vector",
        "input_feature_count": int(input_feature_count),
        "architecture": "independent_binary_cross_validated_platt_svm_per_output",
        "kernel": kernel,
        "solver": solver,
        "score_kind": "probability",
        "output": "calibrated_positive_class_probability",
        "calibrated": True,
        "calibration": {
            "method": "sigmoid_platt",
            "base_score": "decision_function",
            "fit_scope": "train_only_cross_validation",
            "splitter": "StratifiedKFold",
            "cv_folds": 3,
            "shuffle": True,
            "random_state": int(values[0].cv.random_state),
            "ensemble": True,
        },
        "multioutput": "independent_binary_one_vs_rest",
        "num_outputs": len(values),
        "estimator_params": estimator_params,
        "convergence": convergence,
        "probability_quality": probability_quality,
        "serialization": {
            "format": "joblib",
            "requires_matching_sklearn_version": True,
        },
    }


class ConstantDecisionEstimator:
    def __init__(self, value):
        self.value = float(value)

    def decision_function(self, features):
        return np.full(features.shape[0], self.value, dtype=np.float32)


class SigmoidDecisionWrapper:
    def __init__(self, estimator):
        self.estimator = estimator
        self.classes_ = np.array([0, 1])

    def decision_function(self, features):
        return self.estimator.decision_function(features)

    def predict_proba(self, features):
        scores = np.asarray(self.decision_function(features), dtype=np.float64).reshape(-1)
        probs = 1.0 / (1.0 + np.exp(-np.clip(scores, -40.0, 40.0)))
        return np.stack([1.0 - probs, probs], axis=1)


class ClassicalMLTrainer:
    def __init__(self, args):
        self.args = args
        self._task_reducer = None
        self._lsa_contract = None
        self._lda_contract = None
        self._plsa_contract = None
        self._tsvd_contract = None
        self._pca_contract = None
        self._kpca_contract = None
        self._ml_scaler = None
        self._ml_reducer = None
        self._last_tuning_result = None
        self._last_svm_convergence = None
        self._last_svm_probability_quality = None
        self._current_svm_label_names = []
        self._last_logreg_convergence = None
        self._current_logreg_label_names = []

    def _reset_artifacts(self):
        self._task_reducer = None
        self._lsa_contract = None
        self._lda_contract = None
        self._plsa_contract = None
        self._tsvd_contract = None
        self._pca_contract = None
        self._kpca_contract = None
        self._ml_scaler = None
        self._ml_reducer = None
        self._last_tuning_result = None
        self._last_svm_convergence = None
        self._last_svm_probability_quality = None
        self._current_svm_label_names = []
        self._last_logreg_convergence = None
        self._current_logreg_label_names = []

    def _jsonable_args(self):
        payload = {}
        for key, value in vars(self.args).items():
            if str(key).startswith("_"):
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                payload[key] = value
            else:
                payload[key] = str(value)
        return payload

    def _seed(self):
        return int(getattr(self.args, "seed", 42))

    def _label_names(self, train_data):
        return list(getattr(train_data, "label_names", []))

    def _feature_names(self, width):
        prefix = (
            "component"
            if self._task_reducer is not None or self._ml_reducer is not None
            else "feature"
        )
        return [f"{prefix}_{index + 1}" for index in range(int(width or 0))]

    def _write_classical_analysis(
        self,
        train_x,
        train_y,
        val_x,
        val_y,
        val_scores,
        test_x=None,
        test_y=None,
        test_scores=None,
        estimator_bundle=None,
    ):
        artifact_dir = str(getattr(self.args, "artifact_dir", "") or "").strip()
        if not artifact_dir:
            return
        try:
            feature_names = self._feature_names(
                train_x.shape[1] if getattr(train_x, "ndim", 0) > 1 else 1
            )
            operating_point = (getattr(self.args, "_artifact_info", {}) or {}).get(
                "operating_point"
            )
            analysis = build_classical_analysis(
                train_x,
                train_y,
                val_x,
                val_y,
                val_scores,
                test_x,
                test_y,
                test_scores,
                estimator_bundle=estimator_bundle,
                feature_names=feature_names,
                operating_point=operating_point,
            )
            save_analysis(artifact_dir, analysis)
        except Exception as exc:
            print(f"[analysis] skipped classical analysis artifact: {exc}", flush=True)

    def _save_xgb_native_artifact(self, bundle, artifact_dir, label_names, predictor_contract):
        estimators = list(bundle.get("estimators") or [])
        verification_x = np.asarray(bundle.get("_artifact_verification_x"))
        if verification_x.ndim != 2 or verification_x.shape[0] == 0:
            raise XGBArtifactContractError(
                "XGBoost artifact export requires a non-empty two-dimensional verification sample"
            )
        if not np.isfinite(verification_x).all():
            raise XGBArtifactContractError("XGBoost artifact verification features must be finite")
        expected_width = int(predictor_contract["input_feature_count"])
        if verification_x.shape[1] != expected_width:
            raise XGBArtifactContractError(
                "XGBoost artifact verification feature width does not match the predictor contract",
                details={
                    "expected_width": expected_width,
                    "actual_width": int(verification_x.shape[1]),
                },
            )
        names = list(label_names or [])
        if len(names) != len(estimators):
            names = [f"output_{index + 1}" for index in range(len(estimators))]
        model_dir = Path(artifact_dir) / "xgb_models"
        model_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for output_index, (wrapped, label_name) in enumerate(zip(estimators, names)):
            if not isinstance(wrapped, dict) or wrapped.get("kind") != "sklearn":
                raise XGBArtifactContractError(
                    "XGBoost artifacts cannot contain constant or non-XGBoost output estimators",
                    details={"output_index": output_index},
                )
            estimator = wrapped.get("value")
            if XGBClassifier is None or not isinstance(estimator, XGBClassifier):
                raise XGBArtifactContractError(
                    "XGBoost artifact outputs must be fitted XGBClassifier instances",
                    details={"output_index": output_index},
                )
            relative_path = Path("xgb_models") / f"output_{output_index:03d}.ubj"
            destination = Path(artifact_dir) / relative_path
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.stem}.", suffix=".ubj", dir=model_dir
            )
            os.close(descriptor)
            try:
                estimator.save_model(temporary_name)
                with open(temporary_name, "rb") as handle:
                    os.fsync(handle.fileno())
                reloaded = XGBClassifier()
                reloaded.load_model(temporary_name)
                expected = np.asarray(estimator.predict_proba(verification_x)[:, 1])
                actual = np.asarray(reloaded.predict_proba(verification_x)[:, 1])
                if not np.array_equal(expected, actual):
                    raise XGBArtifactContractError(
                        "XGBoost native-model reload changed verification predictions",
                        details={
                            "output_index": output_index,
                            "max_absolute_difference": float(np.max(np.abs(expected - actual))),
                        },
                    )
                os.replace(temporary_name, destination)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
            outputs.append(
                {
                    "output_index": output_index,
                    "label_name": str(label_name),
                    "model_path": relative_path.as_posix(),
                    "sha256": _sha256_file(destination),
                    "size_bytes": int(destination.stat().st_size),
                    "effective_params": _xgb_effective_config(estimator),
                }
            )
        manifest = {
            "schema_version": XGB_MANIFEST_SCHEMA_VERSION,
            "predictor_contract_version": XGB_PREDICTOR_CONTRACT_VERSION,
            "algorithm": "xgboost",
            "implementation": "xgboost.XGBClassifier",
            "implementation_versions": {
                "xgboost": str(xgboost.__version__),
                "scikit_learn": sklearn.__version__,
                "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            },
            "input_feature_count": expected_width,
            "num_outputs": len(outputs),
            "outputs": outputs,
            "reload_verification": {
                "status": "passed",
                "sample_count": int(verification_x.shape[0]),
                "comparison": "exact_positive_class_probability",
            },
        }
        manifest_path = Path(artifact_dir) / "xgb_manifest.json"
        _atomic_write_json(manifest_path, manifest)
        predictor_contract["serialization"]["manifest_sha256"] = _sha256_file(manifest_path)
        predictor_contract["serialization"]["manifest_size_bytes"] = int(
            manifest_path.stat().st_size
        )
        payload = {
            key: value
            for key, value in bundle.items()
            if key not in {"estimators", "_artifact_verification_x"}
        }
        payload.update(
            {
                "classifier": "xgb",
                "estimator_kind": "xgboost_native_ensemble",
                "manifest_path": "xgb_manifest.json",
            }
        )
        return payload

    def _save_lgb_native_artifact(self, bundle, artifact_dir, label_names, predictor_contract):
        estimators = list(bundle.get("estimators") or [])
        verification_x = np.asarray(bundle.get("_artifact_verification_x"))
        if verification_x.ndim != 2 or verification_x.shape[0] == 0:
            raise LGBArtifactContractError(
                "LightGBM artifact export requires a non-empty two-dimensional verification sample"
            )
        if not np.isfinite(verification_x).all():
            raise LGBArtifactContractError("LightGBM artifact verification features must be finite")
        expected_width = int(predictor_contract["input_feature_count"])
        if verification_x.shape[1] != expected_width:
            raise LGBArtifactContractError(
                "LightGBM artifact verification feature width does not match the predictor contract",
                details={
                    "expected_width": expected_width,
                    "actual_width": int(verification_x.shape[1]),
                },
            )
        names = list(label_names or [])
        if len(names) != len(estimators):
            names = [f"output_{index + 1}" for index in range(len(estimators))]
        model_dir = Path(artifact_dir) / "lgb_models"
        model_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for output_index, (wrapped, label_name) in enumerate(zip(estimators, names)):
            if not isinstance(wrapped, dict) or wrapped.get("kind") != "sklearn":
                raise LGBArtifactContractError(
                    "LightGBM artifacts cannot contain constant or non-LightGBM output estimators",
                    details={"output_index": output_index},
                )
            estimator = wrapped.get("value")
            if LGBMClassifier is None or not isinstance(estimator, LGBMClassifier):
                raise LGBArtifactContractError(
                    "LightGBM artifact outputs must be fitted LGBMClassifier instances",
                    details={"output_index": output_index},
                )
            relative_path = Path("lgb_models") / f"output_{output_index:03d}.txt"
            destination = Path(artifact_dir) / relative_path
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.stem}.", suffix=".txt", dir=model_dir
            )
            os.close(descriptor)
            try:
                estimator.booster_.save_model(temporary_name)
                with open(temporary_name, "rb") as handle:
                    os.fsync(handle.fileno())
                reloaded = LightGBMBooster(model_file=temporary_name)
                expected = np.asarray(estimator.predict_proba(verification_x)[:, 1])
                actual = np.asarray(reloaded.predict(verification_x))
                if not np.array_equal(expected, actual):
                    raise LGBArtifactContractError(
                        "LightGBM native-model reload changed verification predictions",
                        details={
                            "output_index": output_index,
                            "max_absolute_difference": float(np.max(np.abs(expected - actual))),
                        },
                    )
                os.replace(temporary_name, destination)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
            outputs.append(
                {
                    "output_index": output_index,
                    "label_name": str(label_name),
                    "model_path": relative_path.as_posix(),
                    "sha256": _sha256_file(destination),
                    "size_bytes": int(destination.stat().st_size),
                    "effective_params": _lgb_effective_config(estimator),
                }
            )
        manifest = {
            "schema_version": LGB_MANIFEST_SCHEMA_VERSION,
            "predictor_contract_version": LGB_PREDICTOR_CONTRACT_VERSION,
            "algorithm": "lightgbm",
            "implementation": "lightgbm.LGBMClassifier",
            "native_loader": "lightgbm.Booster",
            "implementation_versions": {
                "lightgbm": str(lightgbm.__version__),
                "scikit_learn": sklearn.__version__,
                "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            },
            "input_feature_count": expected_width,
            "num_outputs": len(outputs),
            "execution_policy": dict(predictor_contract["execution_policy"]),
            "outputs": outputs,
            "reload_verification": {
                "status": "passed",
                "sample_count": int(verification_x.shape[0]),
                "comparison": "exact_positive_class_probability",
            },
        }
        manifest_path = Path(artifact_dir) / "lgb_manifest.json"
        _atomic_write_json(manifest_path, manifest)
        predictor_contract["serialization"]["manifest_sha256"] = _sha256_file(manifest_path)
        predictor_contract["serialization"]["manifest_size_bytes"] = int(
            manifest_path.stat().st_size
        )
        payload = {
            key: value
            for key, value in bundle.items()
            if key not in {"estimators", "_artifact_verification_x"}
        }
        payload.update(
            {
                "classifier": "lgb",
                "estimator_kind": "lightgbm_native_ensemble",
                "manifest_path": "lgb_manifest.json",
            }
        )
        return payload

    def _artifact_payload(self, bundle, artifact_dir, label_names, predictor_contract):
        if bundle.get("classifier") == "xgb":
            payload = self._save_xgb_native_artifact(
                bundle, artifact_dir, label_names, predictor_contract
            )
        elif bundle.get("classifier") == "lgb":
            payload = self._save_lgb_native_artifact(
                bundle, artifact_dir, label_names, predictor_contract
            )
        else:
            payload = {
                key: value for key, value in bundle.items() if not str(key).startswith("_artifact_")
            }
        return {
            **payload,
            "task_reducer": self._task_reducer,
            "lsa_contract": self._lsa_contract,
            "lda_contract": self._lda_contract,
            "plsa_contract": self._plsa_contract,
            "tsvd_contract": self._tsvd_contract,
            "pca_contract": self._pca_contract,
            "kpca_contract": self._kpca_contract,
            "ml_scaler": self._ml_scaler,
            "ml_reducer": self._ml_reducer,
        }

    def _maybe_save_artifact(self, bundle, train_data):
        artifact_dir = str(getattr(self.args, "artifact_dir", "") or "").strip()
        if not artifact_dir:
            return
        artifact_source = str(getattr(self.args, "_artifact_source", "sequence_features"))
        artifact_info = getattr(self.args, "_artifact_info", {}) or {}
        predictor_contract = None
        if bundle.get("estimator_kind") == "calibrated_svm":
            try:
                predictor_contract = calibrated_svm_predictor_contract(
                    bundle.get("estimators"),
                    convergence=self._last_svm_convergence,
                    probability_quality=self._last_svm_probability_quality,
                    input_feature_count=bundle.get("input_feature_count"),
                )
            except (TypeError, ValueError) as exc:
                raise SVMArtifactContractError(str(exc)) from exc
        elif bundle.get("estimator_kind") == "svm":
            try:
                predictor_contract = svm_predictor_contract(
                    bundle.get("estimators"),
                    convergence=self._last_svm_convergence,
                )
            except (TypeError, ValueError) as exc:
                raise SVMArtifactContractError(str(exc)) from exc
        elif bundle.get("estimator_kind") == "rf":
            predictor_contract = rf_predictor_contract(bundle.get("estimator"))
        elif bundle.get("classifier") == "extratrees":
            predictor_contract = extratrees_predictor_contract(bundle.get("estimator"))
        elif bundle.get("classifier") == "xgb":
            try:
                predictor_contract = xgb_predictor_contract(
                    bundle.get("estimators"),
                    input_feature_count=bundle.get("input_feature_count"),
                )
            except (TypeError, ValueError) as exc:
                raise XGBArtifactContractError(str(exc)) from exc
        elif bundle.get("classifier") == "lgb":
            try:
                predictor_contract = lgb_predictor_contract(
                    bundle.get("estimators"),
                    input_feature_count=bundle.get("input_feature_count"),
                )
            except (TypeError, ValueError) as exc:
                raise LGBArtifactContractError(str(exc)) from exc
        elif bundle.get("classifier") == "histgb":
            try:
                predictor_contract = histgb_predictor_contract(
                    bundle.get("estimators"),
                    input_feature_count=bundle.get("input_feature_count"),
                )
            except (TypeError, ValueError) as exc:
                raise HistGBArtifactContractError(str(exc)) from exc
        elif bundle.get("classifier") == "logreg":
            try:
                predictor_contract = logreg_predictor_contract(
                    bundle.get("estimators"),
                    self._last_logreg_convergence,
                    input_feature_count=bundle.get("input_feature_count"),
                )
            except (TypeError, ValueError) as exc:
                raise LogRegArtifactContractError(str(exc)) from exc
        if artifact_source not in {
            "sequence_features",
            "sequence_embedding_features",
            "pretrained_residue_embedding_features",
            "pretrained_embeddings",
            "pair_vector_features",
            "pair_embedding_features",
        }:
            print(
                "[artifact] reusable classical bundle skipped: embedding-backed route is not reusable"
            )
            return
        if artifact_source in {
            "sequence_embedding_features",
            "pretrained_residue_embedding_features",
        }:
            if normalize_task_type(getattr(self.args, "task_type", "")) != "pep_function":
                print(
                    "[artifact] reusable sequence-embedding classical bundle skipped: currently supported for pep_function only"
                )
                return
            os.makedirs(artifact_dir, exist_ok=True)
            label_names = list(artifact_info.get("label_names") or self._label_names(train_data))
            payload = self._artifact_payload(bundle, artifact_dir, label_names, predictor_contract)
            model_artifact = _atomic_joblib_dump(
                os.path.join(artifact_dir, "model.joblib"), payload
            )
            profile_generation = artifact_info.get("profile_generation") or {}
            pretrained_residue = artifact_source == "pretrained_residue_embedding_features"
            supports_inference = (
                pretrained_residue
                or not uses_profile_features(self.args)
                or bool(profile_generation.get("subject_fasta"))
            )
            metadata = {
                "bundle_type": (
                    "pretrained_residue_embedding_classical"
                    if pretrained_residue
                    else "sequence_embedding_classical"
                ),
                "supports_inference": supports_inference,
                "task_type": normalize_task_type(self.args.task_type),
                "args": self._jsonable_args(),
                "label_names": label_names,
                "route_schema_version": ROUTE_SCHEMA_VERSION,
                "effective_route": resolve_effective_route(vars(self.args), self.args.task_type),
                "model_artifact": model_artifact,
            }
            for key, value in artifact_info.items():
                if key not in {"label_names", "dl_scaler"}:
                    metadata[key] = value
            if predictor_contract is not None:
                metadata["predictor_contract"] = predictor_contract
                if bundle.get("classifier") == "xgb":
                    metadata["xgb_manifest"] = dict(predictor_contract["serialization"])
                elif bundle.get("classifier") == "lgb":
                    metadata["lgb_manifest"] = dict(predictor_contract["serialization"])
            dl_scaler = artifact_info.get("dl_scaler")
            if dl_scaler is not None:
                joblib.dump(
                    {"dl_scaler": dl_scaler}, os.path.join(artifact_dir, "preprocess.joblib")
                )
                metadata["preprocess_path"] = "preprocess.joblib"
            if not metadata["supports_inference"]:
                metadata["reason"] = PROFILE_DEPENDENT_INFERENCE_REASON
                metadata["reason_code"] = "profile_features_require_sample_id"
                metadata["inference_unavailable_reason"] = PROFILE_DEPENDENT_INFERENCE_REASON
                metadata["inference_unavailable_code"] = "profile_features_require_sample_id"
            if self._last_tuning_result:
                metadata["tuning"] = self._last_tuning_result
            _atomic_write_json(os.path.join(artifact_dir, "metadata.json"), metadata)
            print(
                f"[artifact] saved reusable {metadata['bundle_type']} model bundle: {artifact_dir}"
            )
            return
        if artifact_source == "pair_embedding_features":
            if normalize_task_type(getattr(self.args, "task_type", "")) not in {
                "pep_protein_interaction",
                "pep_hlaii",
                "pep_hlai",
            }:
                print(
                    "[artifact] reusable pair-embedding classical bundle skipped: currently supported for pair tasks only"
                )
                return
            os.makedirs(artifact_dir, exist_ok=True)
            label_names = list(artifact_info.get("label_names") or self._label_names(train_data))
            payload = self._artifact_payload(bundle, artifact_dir, label_names, predictor_contract)
            model_artifact = _atomic_joblib_dump(
                os.path.join(artifact_dir, "model.joblib"), payload
            )
            metadata = {
                "bundle_type": "pair_embedding_classical",
                "supports_inference": True,
                "task_type": normalize_task_type(self.args.task_type),
                "args": self._jsonable_args(),
                "label_names": label_names,
                "model_artifact": model_artifact,
            }
            for key, value in artifact_info.items():
                if key != "label_names":
                    metadata[key] = value
            if predictor_contract is not None:
                metadata["predictor_contract"] = predictor_contract
                if bundle.get("classifier") == "xgb":
                    metadata["xgb_manifest"] = dict(predictor_contract["serialization"])
                elif bundle.get("classifier") == "lgb":
                    metadata["lgb_manifest"] = dict(predictor_contract["serialization"])
            if self._last_tuning_result:
                metadata["tuning"] = self._last_tuning_result
            _atomic_write_json(os.path.join(artifact_dir, "metadata.json"), metadata)
            print(
                f"[artifact] saved reusable pair-embedding classical model bundle: {artifact_dir}"
            )
            return
        if artifact_source == "pair_vector_features":
            if normalize_task_type(getattr(self.args, "task_type", "")) not in {
                "pep_protein_interaction",
                "pep_hlaii",
                "pep_hlai",
            }:
                print(
                    "[artifact] reusable pair-vector classical bundle skipped: currently supported for pair tasks only"
                )
                return
            os.makedirs(artifact_dir, exist_ok=True)
            label_names = list(artifact_info.get("label_names") or self._label_names(train_data))
            payload = self._artifact_payload(bundle, artifact_dir, label_names, predictor_contract)
            model_artifact = _atomic_joblib_dump(
                os.path.join(artifact_dir, "model.joblib"), payload
            )
            metadata = {
                "bundle_type": "pair_vector_classical",
                "supports_inference": True,
                "task_type": normalize_task_type(self.args.task_type),
                "args": self._jsonable_args(),
                "label_names": label_names,
                "model_artifact": model_artifact,
            }
            for key, value in artifact_info.items():
                if key != "label_names":
                    metadata[key] = value
            if predictor_contract is not None:
                metadata["predictor_contract"] = predictor_contract
                if bundle.get("classifier") == "xgb":
                    metadata["xgb_manifest"] = dict(predictor_contract["serialization"])
                elif bundle.get("classifier") == "lgb":
                    metadata["lgb_manifest"] = dict(predictor_contract["serialization"])
            if self._last_tuning_result:
                metadata["tuning"] = self._last_tuning_result
            _atomic_write_json(os.path.join(artifact_dir, "metadata.json"), metadata)
            print(f"[artifact] saved reusable pair-vector classical model bundle: {artifact_dir}")
            return
        if normalize_task_type(getattr(self.args, "task_type", "")) != "pep_function":
            print(
                "[artifact] reusable classical bundle skipped: currently supported for pep_function only"
            )
            return
        os.makedirs(artifact_dir, exist_ok=True)
        label_names = list(artifact_info.get("label_names") or self._label_names(train_data))
        payload = self._artifact_payload(bundle, artifact_dir, label_names, predictor_contract)
        model_artifact = _atomic_joblib_dump(os.path.join(artifact_dir, "model.joblib"), payload)
        bundle_type = (
            "pretrained_embedding_classical"
            if artifact_source == "pretrained_embeddings"
            else "sequence_classical"
        )
        profile_generation = artifact_info.get("profile_generation") or {}
        supports_inference = not (
            artifact_source == "sequence_features"
            and uses_profile_features(self.args)
            and not profile_generation.get("subject_fasta")
        )
        metadata = {
            "bundle_type": bundle_type,
            "supports_inference": supports_inference,
            "task_type": normalize_task_type(self.args.task_type),
            "args": self._jsonable_args(),
            "label_names": label_names,
            "model_artifact": model_artifact,
        }
        if not supports_inference:
            metadata["reason"] = PROFILE_DEPENDENT_INFERENCE_REASON
            metadata["reason_code"] = "profile_features_require_sample_id"
            metadata["inference_unavailable_reason"] = PROFILE_DEPENDENT_INFERENCE_REASON
            metadata["inference_unavailable_code"] = "profile_features_require_sample_id"
        if artifact_source == "pretrained_embeddings":
            metadata["pretrained"] = dict(artifact_info.get("pretrained") or {})
        for key, value in artifact_info.items():
            if key not in {"label_names", "pretrained", "dl_scaler"}:
                metadata[key] = value
        if predictor_contract is not None:
            metadata["predictor_contract"] = predictor_contract
            if bundle.get("classifier") == "xgb":
                metadata["xgb_manifest"] = dict(predictor_contract["serialization"])
            elif bundle.get("classifier") == "lgb":
                metadata["lgb_manifest"] = dict(predictor_contract["serialization"])
        if self._last_tuning_result:
            metadata["tuning"] = self._last_tuning_result
        _atomic_write_json(os.path.join(artifact_dir, "metadata.json"), metadata)
        if supports_inference:
            print(f"[artifact] saved reusable {bundle_type} model bundle: {artifact_dir}")
        else:
            print(
                f"[artifact] saved {bundle_type} model bundle without Web raw-sequence inference: {artifact_dir}"
            )

    def _compute_aupr(self, y_true, scores):
        if y_true is None or scores is None:
            return None
        return Evaluator.ranking_metrics(y_true, scores).get("aupr")

    def _operating_point_objective(self):
        return resolve_selection(self.args).threshold

    def _checkpoint_metric(self):
        return resolve_selection(self.args).checkpoint.checkpoint_metric

    def _record_pair_vector_threshold(self, y_true, scores):
        score_kind = (
            self._svm_score_kind()
            if str(getattr(self.args, "model", "")).lower() == "svm"
            or str(getattr(self.args, "classifier", "")).lower() == "svm"
            else "probability"
        )
        operating_point = Evaluator.fit_operating_point(
            y_true,
            scores,
            score_kind=score_kind,
            objective=self._operating_point_objective(),
            search_profile=resolve_selection(self.args).search_profile,
        )
        self._record_operating_point(operating_point)
        return operating_point.threshold if operating_point.is_binary else operating_point.values

    def _record_operating_point(self, operating_point):
        operating_point = OperatingPoint.from_dict(operating_point)
        artifact_info = dict(getattr(self.args, "_artifact_info", {}) or {})
        point_payload = operating_point.to_dict()
        artifact_info["evaluation_protocol_version"] = EVALUATION_PROTOCOL_VERSION
        artifact_info["checkpoint_metric"] = self._checkpoint_metric()
        artifact_info.update(selection_fields(self.args))
        artifact_info["feature_fit_scope"] = "train_only"
        artifact_info["evaluation_mode"] = "independent_iid"
        artifact_info["operating_point"] = point_payload
        for key in ["threshold", "thresholds", "threshold_by_label", "threshold_source"]:
            if key in point_payload:
                artifact_info[key] = point_payload[key]
        self.args._artifact_info = artifact_info

    def _fit_validation_metrics(
        self, y_true, scores, *, score_kind="probability", label_names=None
    ):
        metrics, operating_point = Evaluator.fit_and_score(
            y_true,
            scores,
            score_kind=score_kind,
            label_names=label_names,
            objective=self._operating_point_objective(),
            search_profile=resolve_selection(self.args).search_profile,
        )
        self._record_operating_point(operating_point)
        return metrics, operating_point

    def _save_protocol_evaluation(
        self, val_y, val_scores, operating_point, test_y=None, test_scores=None
    ):
        artifact_dir = str(getattr(self.args, "artifact_dir", "") or "").strip()
        if not artifact_dir:
            return
        os.makedirs(artifact_dir, exist_ok=True)
        validation_labels = np.asarray(val_y, dtype=np.int32)
        validation_scores = np.asarray(val_scores, dtype=np.float64)
        payload = {
            "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
            "checkpoint_metric": self._checkpoint_metric(),
            **selection_fields(self.args),
            "feature_fit_scope": "train_only",
            "evaluation_mode": "independent_iid",
            "operating_point": operating_point.to_dict(),
            "validation": Evaluator.score_metrics(
                validation_labels,
                validation_scores,
                operating_point,
            ),
        }
        if self._last_svm_probability_quality is not None:
            payload["probability_quality"] = self._last_svm_probability_quality
        arrays = {
            "validation_labels": validation_labels,
            "validation_scores": validation_scores,
        }
        if test_y is not None and test_scores is not None:
            persisted_test_labels = np.asarray(test_y, dtype=np.int32)
            persisted_test_scores = np.asarray(test_scores, dtype=np.float64)
            payload["test"] = Evaluator.score_metrics(
                persisted_test_labels,
                persisted_test_scores,
                operating_point,
            )
            arrays["test_labels"] = persisted_test_labels
            arrays["test_scores"] = persisted_test_scores
        with open(os.path.join(artifact_dir, "evaluation.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        np.savez_compressed(os.path.join(artifact_dir, "evaluation_scores.npz"), **arrays)

    def collect_xy(self, dataset, split_name="dataset"):
        tensors = getattr(dataset, "tensors", None)
        if tensors is not None and len(tensors) >= 2:
            x_tensor, y_tensor = tensors[:2]
            if x_tensor.ndim > 2:
                x_tensor = x_tensor.reshape(x_tensor.shape[0], -1)
            X = x_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            Y = y_tensor.detach().cpu().numpy().astype(int, copy=False)
            return X, Y
        X, Y = [], []
        for i in tqdm(range(len(dataset)), desc=f"Collecting {split_name} features"):
            x, y = dataset[i]
            if hasattr(x, "features") and hasattr(x, "mask"):
                x = x.features
            X.append(x.view(-1) if x.ndim > 1 else x)
            Y.append(y)
        X = torch.stack(X).cpu().numpy().astype(np.float32)
        Y = torch.stack(Y).cpu().numpy().astype(int)
        return X, Y

    def apply_reduction(self, train_x, val_x, train_y=None, test_x=None):
        self._task_reducer = None
        self._lsa_contract = None
        self._lda_contract = None
        self._plsa_contract = None
        self._tsvd_contract = None
        raw_encoder = str(getattr(self.args, "encoder", "")).strip().lower()
        encoder = "tfidf" if raw_encoder == "tf-idf" else raw_encoder
        reduction = getattr(self.args, "reduction", "none")

        if reduction == "none":
            return train_x, val_x, test_x

        if reduction == "lsa":
            validate_lsa_route(self.args)
            artifact_info = dict(getattr(self.args, "_artifact_info", {}) or {})
            upstream_contract = artifact_info.get("classic") or {
                "encoder": encoder,
                "input_feature_count": int(train_x.shape[1]),
            }
            train_x, val_x, test_x, svd, contract = fit_lsa_transform(
                train_x,
                val_x,
                test_x,
                encoder=encoder,
                n_components=getattr(self.args, "lsa_dim", 128),
                seed=self._seed(),
                dense_memory_limit_gib=getattr(
                    self.args,
                    "lsa_dense_memory_limit_gib",
                    LSA_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
                ),
                upstream_feature_contract=upstream_contract,
            )
            self._task_reducer = {"kind": "sklearn", "value": svd}
            self._lsa_contract = contract
            artifact_info["lsa_contract"] = contract
            self.args._artifact_info = artifact_info
        elif reduction == "lda":
            validate_lda_route(self.args)
            artifact_info = dict(getattr(self.args, "_artifact_info", {}) or {})
            is_pair = normalize_task_type(getattr(self.args, "task_type", "") or "") in {
                "pep_protein_interaction",
                "pep_hlaii",
                "pep_hlai",
            }
            upstream_contract = (
                artifact_info.get("pair_bow_joint_contract")
                if is_pair
                else artifact_info.get("classic")
            )
            train_x, val_x, test_x, lda, contract = fit_lda_transform(
                train_x,
                val_x,
                test_x,
                n_components=getattr(self.args, "lda_topics", 50),
                seed=self._seed(),
                dense_memory_limit_gib=getattr(
                    self.args,
                    "lda_dense_memory_limit_gib",
                    LDA_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
                ),
                upstream_feature_contract=upstream_contract,
            )
            self._task_reducer = {"kind": "sklearn", "value": lda}
            self._lda_contract = contract
            artifact_info["lda_contract"] = contract
            self.args._artifact_info = artifact_info
        elif reduction == "plsa":
            validate_plsa_route(self.args)
            artifact_info = dict(getattr(self.args, "_artifact_info", {}) or {})
            is_pair = normalize_task_type(getattr(self.args, "task_type", "") or "") in {
                "pep_protein_interaction",
                "pep_hlaii",
                "pep_hlai",
            }
            upstream_contract = (
                artifact_info.get("pair_bow_joint_contract")
                if is_pair
                else artifact_info.get("classic")
            )
            train_x, val_x, test_x, plsa, contract = fit_plsa_transform(
                train_x,
                val_x,
                test_x,
                n_components=getattr(self.args, "plsa_topics", 50),
                seed=self._seed(),
                memory_limit_gib=getattr(
                    self.args,
                    "plsa_memory_limit_gib",
                    PLSA_DEFAULT_MEMORY_LIMIT_GIB,
                ),
                upstream_feature_contract=upstream_contract,
            )
            self._task_reducer = {"kind": "pepplatform_plsa", "value": plsa}
            self._plsa_contract = contract
            artifact_info["plsa_contract"] = contract
            self.args._artifact_info = artifact_info

        return train_x, val_x, test_x

    def apply_ml_scaler(self, train_x, val_x, test_x=None):
        self._ml_scaler = None
        ml_scaler = str(getattr(self.args, "ml_scaler", "standard")).lower()
        if ml_scaler == "none":
            return train_x, val_x, test_x
        if ml_scaler == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        val_x = scaler.transform(val_x)
        if test_x is not None:
            test_x = scaler.transform(test_x)
        self._ml_scaler = scaler
        return train_x, val_x, test_x

    def apply_ml_reduction(self, train_x, val_x, test_x=None):
        self._ml_reducer = None
        self._tsvd_contract = None
        self._pca_contract = None
        self._kpca_contract = None
        ml_reduction = str(getattr(self.args, "ml_reduction", "none")).lower()
        if ml_reduction == "none":
            return train_x, val_x, test_x
        if ml_reduction == "pca":
            train_x, val_x, test_x, reducer, contract = fit_pca_transform(
                train_x,
                val_x,
                test_x,
                n_components=getattr(self.args, "ml_reduction_dim", 128),
                seed=self._seed(),
                dense_memory_limit_gib=getattr(
                    self.args,
                    "pca_dense_memory_limit_gib",
                    PCA_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
                ),
            )
            self._ml_reducer = reducer
            self._pca_contract = contract
            artifact_info = dict(getattr(self.args, "_artifact_info", {}) or {})
            artifact_info["pca_contract"] = contract
            self.args._artifact_info = artifact_info
            return train_x, val_x, test_x
        elif ml_reduction == "kpca":
            train_x, val_x, test_x, reducer, contract = fit_kpca_transform(
                train_x,
                val_x,
                test_x,
                n_components=getattr(self.args, "ml_reduction_dim", 128),
                kernel=getattr(self.args, "kpca_kernel", "rbf"),
                gamma=getattr(self.args, "kpca_gamma", None),
                seed=self._seed(),
                memory_limit_gib=getattr(
                    self.args,
                    "kpca_memory_limit_gib",
                    KPCA_DEFAULT_MEMORY_LIMIT_GIB,
                ),
            )
            self._ml_reducer = reducer
            self._kpca_contract = contract
            artifact_info = dict(getattr(self.args, "_artifact_info", {}) or {})
            artifact_info["kpca_contract"] = contract
            self.args._artifact_info = artifact_info
            return train_x, val_x, test_x
        elif ml_reduction == "tsvd":
            train_x, val_x, test_x, reducer, contract = fit_tsvd_transform(
                train_x,
                val_x,
                test_x,
                n_components=getattr(self.args, "ml_reduction_dim", 128),
                seed=self._seed(),
                dense_memory_limit_gib=getattr(
                    self.args,
                    "tsvd_dense_memory_limit_gib",
                    TSVD_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
                ),
            )
            self._ml_reducer = reducer
            self._tsvd_contract = contract
            artifact_info = dict(getattr(self.args, "_artifact_info", {}) or {})
            artifact_info["tsvd_contract"] = contract
            self.args._artifact_info = artifact_info
            return train_x, val_x, test_x
        else:
            raise ValueError(f"unsupported ml_reduction: {ml_reduction}")

    def _preprocess_tuning_split(self, train_x, train_y, val_x):
        saved = (
            self._task_reducer,
            self._lsa_contract,
            self._lda_contract,
            self._plsa_contract,
            self._tsvd_contract,
            self._pca_contract,
            self._kpca_contract,
            self._ml_scaler,
            self._ml_reducer,
            getattr(self.args, "_artifact_info", None),
        )
        try:
            train_x, val_x, _ = self.apply_reduction(train_x, val_x, train_y, None)
            train_x, val_x, _ = self.apply_ml_scaler(train_x, val_x, None)
            train_x, val_x, _ = self.apply_ml_reduction(train_x, val_x, None)
            return train_x, val_x
        finally:
            (
                self._task_reducer,
                self._lsa_contract,
                self._lda_contract,
                self._plsa_contract,
                self._tsvd_contract,
                self._pca_contract,
                self._kpca_contract,
                self._ml_scaler,
                self._ml_reducer,
                artifact_info,
            ) = saved
            if artifact_info is None:
                if hasattr(self.args, "_artifact_info"):
                    delattr(self.args, "_artifact_info")
            else:
                self.args._artifact_info = artifact_info

    def _auto_tune_enabled(self):
        return bool(getattr(self.args, "ml_auto_tune", False))

    def _tuning_objective(self):
        objective = resolve_selection(self.args).tuning.id
        return "f1" if objective == "macro_f1" else objective

    def _tuning_mode(self):
        mode = str(getattr(self.args, "ml_tuning_mode", "cv")).strip().lower()
        return mode if mode in {"holdout", "cv"} else "cv"

    def _tuning_max_candidates(self, mode):
        budget_name = str(getattr(self.args, "ml_tuning_budget", "") or "").strip().lower()
        budget = ML_TUNING_BUDGETS.get(budget_name)
        if budget and mode == "cv":
            return int(budget["max_candidates"])
        default = 6 if mode == "holdout" else ML_TUNING_MAX_CANDIDATES
        try:
            value = int(getattr(self.args, "ml_tuning_max_candidates", default))
        except (TypeError, ValueError):
            return default
        return min(ML_TUNING_MAX_CANDIDATES, max(1, value))

    def _tuning_folds(self):
        budget_name = str(getattr(self.args, "ml_tuning_budget", "") or "").strip().lower()
        budget = ML_TUNING_BUDGETS.get(budget_name)
        if budget:
            return int(budget["cv_folds"])
        return min(5, max(2, int(getattr(self.args, "ml_cv_folds", 2))))

    def _dedupe_candidates(self, candidates):
        seen = set()
        unique = []
        for candidate in candidates:
            key = tuple(sorted(candidate.items()))
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    def _ordered_grid_candidates(self, classifier, current, keys, values):
        grid = [dict(zip(keys, combination)) for combination in product(*values)]
        current_key = tuple(sorted(current.items()))
        remainder = [
            item
            for item in self._dedupe_candidates(grid)
            if tuple(sorted(item.items())) != current_key
        ]

        def stable_rank(item):
            payload = json.dumps(
                {"classifier": classifier, "params": item, "seed": self._seed()},
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()

        remainder.sort(key=stable_rank)
        return [dict(current), *remainder]

    def _rf_candidates(self, mode):
        current = {
            "rf_estimators": int(getattr(self.args, "rf_estimators", 300)),
            "rf_max_depth": int(getattr(self.args, "rf_max_depth", 0)),
            "rf_min_samples_leaf": int(getattr(self.args, "rf_min_samples_leaf", 1)),
        }
        return self._ordered_grid_candidates(
            "rf",
            current,
            ["rf_estimators", "rf_max_depth", "rf_min_samples_leaf"],
            [[100, 300, 500], [0, 10, 30], [1, 2]],
        )

    def _svm_candidates(self, mode):
        current = {
            "svm_c": float(getattr(self.args, "svm_c", 1.0)),
            "svm_class_weight": str(getattr(self.args, "svm_class_weight", "balanced")),
            "svm_kernel": str(getattr(self.args, "svm_kernel", "linear")).lower(),
            "svm_gamma": str(getattr(self.args, "svm_gamma", "scale")),
        }
        candidates = []
        for kernel, c_value, class_weight in product(
            ["linear", "rbf"], [0.01, 0.1, 1.0, 10.0], ["balanced", "none"]
        ):
            gammas = ["scale"] if kernel == "linear" else ["scale", 0.01, 0.1]
            for gamma in gammas:
                candidates.append(
                    {
                        "svm_c": c_value,
                        "svm_class_weight": class_weight,
                        "svm_kernel": kernel,
                        "svm_gamma": gamma,
                    }
                )
        current_key = tuple(sorted(current.items()))
        remainder = [
            item
            for item in self._dedupe_candidates(candidates)
            if tuple(sorted(item.items())) != current_key
        ]
        remainder.sort(
            key=lambda item: hashlib.sha256(
                json.dumps(
                    {"classifier": "svm", "params": item, "seed": self._seed()}, sort_keys=True
                ).encode()
            ).hexdigest()
        )
        return [current, *remainder]

    def _generic_candidates(self, classifier):
        specs = {
            "logreg": (
                {
                    "logreg_c": float(getattr(self.args, "logreg_c", 1.0)),
                    "logreg_class_weight": str(
                        getattr(self.args, "logreg_class_weight", "balanced")
                    ),
                },
                ["logreg_c", "logreg_class_weight"],
                [[0.01, 0.1, 1.0, 10.0, 100.0], ["balanced", "none"]],
            ),
            "extratrees": (
                {
                    "et_estimators": int(getattr(self.args, "et_estimators", 300)),
                    "et_max_depth": int(getattr(self.args, "et_max_depth", 0)),
                    "et_min_samples_leaf": int(getattr(self.args, "et_min_samples_leaf", 1)),
                },
                ["et_estimators", "et_max_depth", "et_min_samples_leaf"],
                [[100, 300, 500], [0, 10, 30], [1, 2]],
            ),
            "histgb": (
                {
                    "histgb_max_iter": int(getattr(self.args, "histgb_max_iter", 300)),
                    "histgb_max_leaf_nodes": int(getattr(self.args, "histgb_max_leaf_nodes", 31)),
                    "histgb_lr": float(getattr(self.args, "histgb_lr", 0.1)),
                    "histgb_l2_regularization": float(
                        getattr(self.args, "histgb_l2_regularization", 0.0)
                    ),
                },
                [
                    "histgb_max_iter",
                    "histgb_max_leaf_nodes",
                    "histgb_lr",
                    "histgb_l2_regularization",
                ],
                [[100, 300, 500], [15, 31, 63], [0.03, 0.1], [0.0, 1.0]],
            ),
            "xgb": (
                {
                    key: getattr(self.args, key)
                    for key in [
                        "xgb_estimators",
                        "xgb_max_depth",
                        "xgb_lr",
                        "xgb_subsample",
                        "xgb_colsample",
                        "xgb_reg_alpha",
                        "xgb_reg_lambda",
                    ]
                },
                [
                    "xgb_estimators",
                    "xgb_max_depth",
                    "xgb_lr",
                    "xgb_subsample",
                    "xgb_colsample",
                    "xgb_reg_alpha",
                    "xgb_reg_lambda",
                ],
                [
                    [100, 300, 500],
                    [3, 6, 10],
                    [0.03, 0.1],
                    [0.8, 1.0],
                    [0.8, 1.0],
                    [0.0, 0.1],
                    [1.0, 5.0],
                ],
            ),
            "lgb": (
                {
                    key: getattr(self.args, key)
                    for key in [
                        "lgb_estimators",
                        "lgb_max_depth",
                        "lgb_lr",
                        "lgb_subsample",
                        "lgb_colsample",
                        "lgb_reg_alpha",
                        "lgb_reg_lambda",
                    ]
                },
                [
                    "lgb_estimators",
                    "lgb_max_depth",
                    "lgb_lr",
                    "lgb_subsample",
                    "lgb_colsample",
                    "lgb_reg_alpha",
                    "lgb_reg_lambda",
                ],
                [
                    [100, 300, 500],
                    [0, 6, 12],
                    [0.03, 0.1],
                    [0.8, 1.0],
                    [0.8, 1.0],
                    [0.0, 0.1],
                    [0.0, 1.0],
                ],
            ),
        }
        current, keys, values = specs[classifier]
        return self._ordered_grid_candidates(classifier, current, keys, values)

    def _candidate_params(self, classifier, mode):
        if classifier == "rf":
            return self._rf_candidates(mode)
        if classifier == "svm":
            return self._svm_candidates(mode)
        if classifier in {"logreg", "extratrees", "histgb", "xgb", "lgb"}:
            return self._generic_candidates(classifier)
        return []

    def _format_params(self, params):
        return ", ".join(f"{key}={value}" for key, value in sorted(params.items()))

    def _svm_gamma_value(self, params=None):
        raw_value = (params or {}).get("svm_gamma", getattr(self.args, "svm_gamma", "scale"))
        if isinstance(raw_value, str):
            value = raw_value.strip().lower()
            if value in {"scale", "auto"}:
                return value
            return float(value)
        return float(raw_value)

    def _svm_calibration(self):
        value = str(getattr(self.args, "svm_calibration", "none") or "none").strip().lower()
        if value not in SVM_CALIBRATION_OPTIONS:
            raise ValueError(
                f"unsupported svm_calibration: {value}; expected one of {SVM_CALIBRATION_OPTIONS}"
            )
        return value

    def _svm_score_kind(self):
        return "probability" if self._svm_calibration() == "sigmoid" else "margin"

    @staticmethod
    def _validate_svm_training_data(train_x, train_y):
        features = np.asarray(train_x)
        labels = np.asarray(train_y)
        if labels.ndim == 1:
            labels = labels[:, None]
        if features.ndim != 2 or labels.ndim != 2:
            raise ValueError("SVM requires 2D feature and label matrices")
        if features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("SVM requires non-empty training features")
        if labels.shape[0] != features.shape[0] or labels.shape[1] == 0:
            raise ValueError("SVM feature and label shapes are inconsistent")
        try:
            features_finite = bool(np.isfinite(features).all())
            labels_finite = bool(np.isfinite(labels).all())
        except TypeError as exc:
            raise ValueError("SVM features and labels must be numeric") from exc
        if not features_finite or not labels_finite:
            raise ValueError("SVM features and labels must be finite")
        for output_index in range(labels.shape[1]):
            values = np.unique(labels[:, output_index])
            if not np.array_equal(values, np.asarray([0, 1])):
                raise ValueError(
                    "SVM output "
                    f"{output_index} requires both binary classes 0 and 1; observed {values.tolist()}"
                )
        return features, labels

    def _svm_iteration_budgets(self, initial_max_iter):
        initial = max(1, int(initial_max_iter))
        policy = (
            str(getattr(self.args, "svm_convergence_policy", "strict") or "strict").strip().lower()
        )
        if policy != "bounded_retry":
            return "strict", [initial]
        ceiling = max(initial, 50_000)
        budgets = []
        for multiplier in (1, 5, 25):
            budget = min(initial * multiplier, ceiling)
            if budget not in budgets:
                budgets.append(budget)
        return "bounded_retry_v1", budgets

    @staticmethod
    def _svm_actual_iterations(estimator):
        value = getattr(estimator, "n_iter_", None)
        if value is None:
            return None
        array = np.asarray(value)
        return int(np.max(array)) if array.size else None

    def _svm_failure_details(
        self,
        *,
        estimator,
        output_index,
        output_count,
        policy,
        budgets,
        attempts,
    ):
        label_name = (
            self._current_svm_label_names[output_index]
            if output_index < len(self._current_svm_label_names)
            else ""
        )
        artifact_source = str(getattr(self.args, "_artifact_source", "sequence_features") or "")
        details = {
            "classifier": "svm",
            "solver": "libsvm" if isinstance(estimator, SVC) else "liblinear",
            "kernel": str(getattr(estimator, "kernel", "linear") or "linear"),
            "output_index": int(output_index),
            "output_number": int(output_index) + 1,
            "output_count": int(output_count),
            "max_iter": int(budgets[-1]),
            "initial_max_iter": int(budgets[0]),
            "attempt_count": int(len(attempts)),
            "attempted_max_iters": " → ".join(str(value) for value in budgets),
            "actual_n_iters": " → ".join(
                "unknown" if item.get("n_iter") is None else str(item["n_iter"])
                for item in attempts
            ),
            "tol": float(getattr(estimator, "tol", 0.0)),
            "scaler": str(getattr(self.args, "ml_scaler", "none") or "none"),
            "convergence_policy": policy,
            "pipeline_stage": (
                "downstream_classifier"
                if artifact_source
                in {
                    "sequence_embedding_features",
                    "pretrained_residue_embedding_features",
                    "pair_embedding_features",
                }
                else "classifier"
            ),
        }
        if label_name:
            details["label_name"] = str(label_name)
        if attempts and attempts[-1].get("n_iter") is not None:
            details["last_n_iter"] = int(attempts[-1]["n_iter"])
        return details

    def _fit_one_svm(
        self,
        estimator,
        train_x,
        labels,
        output_index,
        output_count,
        *,
        emit_progress=False,
    ):
        policy, budgets = self._svm_iteration_budgets(estimator.max_iter)
        attempts = []
        fitted = None
        for attempt_index, budget in enumerate(budgets, start=1):
            candidate = clone(estimator).set_params(max_iter=int(budget))
            if emit_progress:
                print(
                    f"[svm-progress] classifier=svm status=running "
                    f"current={output_index + 1} total={output_count} "
                    f"attempt={attempt_index} attempts={len(budgets)} max_iter={budget}",
                    flush=True,
                )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                candidate.fit(train_x, labels)
            did_warn = any(issubclass(item.category, ConvergenceWarning) for item in caught)
            bad_fit_status = isinstance(candidate, SVC) and int(candidate.fit_status_) != 0
            actual_iterations = self._svm_actual_iterations(candidate)
            converged = not did_warn and not bad_fit_status
            attempts.append(
                {
                    "max_iter": int(budget),
                    "n_iter": actual_iterations,
                    "status": "converged" if converged else "not_converged",
                }
            )
            if converged:
                fitted = candidate
                if emit_progress:
                    print(
                        f"[svm-progress] classifier=svm status=complete "
                        f"current={output_index + 1} total={output_count} "
                        f"attempt={attempt_index} attempts={len(budgets)} "
                        f"max_iter={budget} recovered={'true' if attempt_index > 1 else 'false'}",
                        flush=True,
                    )
                break
            if emit_progress and attempt_index < len(budgets):
                print(
                    f"[svm-progress] classifier=svm status=retrying "
                    f"current={output_index + 1} total={output_count} "
                    f"attempt={attempt_index + 1} attempts={len(budgets)} "
                    f"max_iter={budgets[attempt_index]}",
                    flush=True,
                )
        if fitted is None:
            details = self._svm_failure_details(
                estimator=candidate,
                output_index=output_index,
                output_count=output_count,
                policy=policy,
                budgets=budgets,
                attempts=attempts,
            )
            if emit_progress:
                print(
                    f"[svm-progress] classifier=svm status=failed "
                    f"current={output_index + 1} total={output_count} "
                    f"attempt={len(attempts)} attempts={len(budgets)} "
                    f"max_iter={budgets[-1]}",
                    flush=True,
                )
            raise SVMConvergenceError(
                f"SVM output {output_index} did not converge after "
                f"max_iter budgets {details['attempted_max_iters']} with "
                f"tol={candidate.tol}; increase max_iter, relax tol, or improve "
                "feature scaling",
                details=details,
            )
        return fitted, {
            "output_index": int(output_index),
            "output_number": int(output_index) + 1,
            "attempts": attempts,
            "effective_max_iter": int(fitted.max_iter),
            "recovered": len(attempts) > 1,
        }

    def _fit_svm_estimators(
        self,
        train_x,
        train_y,
        params=None,
        progress=False,
        *,
        record_convergence=True,
    ):
        params = params or {}
        train_x, train_y = self._validate_svm_training_data(train_x, train_y)
        num_classes = train_y.shape[1]
        class_weight = params.get(
            "svm_class_weight", getattr(self.args, "svm_class_weight", "balanced")
        )
        if class_weight == "none":
            class_weight = None
        kernel = str(params.get("svm_kernel", getattr(self.args, "svm_kernel", "linear"))).lower()
        estimators = []
        convergence_outputs = []
        class_iter = (
            tqdm(range(num_classes), desc="Training SVM") if progress else range(num_classes)
        )
        for c in class_iter:
            y = train_y[:, c]
            if kernel == "rbf":
                clf = SVC(
                    C=float(params.get("svm_c", getattr(self.args, "svm_c", 1.0))),
                    kernel="rbf",
                    gamma=self._svm_gamma_value(params),
                    class_weight=class_weight,
                    max_iter=int(getattr(self.args, "svm_max_iter", 2000)),
                    tol=float(getattr(self.args, "svm_tol", 1e-3)),
                )
            else:
                clf = LinearSVC(
                    C=float(params.get("svm_c", getattr(self.args, "svm_c", 1.0))),
                    class_weight=class_weight,
                    max_iter=int(getattr(self.args, "svm_max_iter", 2000)),
                    tol=float(getattr(self.args, "svm_tol", 1e-3)),
                    dual="auto",
                    random_state=self._seed(),
                )
            fitted, convergence = self._fit_one_svm(
                clf,
                train_x,
                y,
                c,
                num_classes,
                emit_progress=progress,
            )
            estimators.append(fitted)
            convergence_outputs.append(convergence)
        if record_convergence:
            policy, budgets = self._svm_iteration_budgets(
                int(getattr(self.args, "svm_max_iter", 2000))
            )
            self._last_svm_convergence = {
                "policy": policy,
                "requested_max_iter": int(budgets[0]),
                "automatic_ceiling": max(int(budgets[0]), 50_000),
                "recovered": any(item["recovered"] for item in convergence_outputs),
                "outputs": convergence_outputs,
            }
        return estimators

    def _svm_scores(self, estimators, x):
        scores = np.stack(
            [np.ravel(estimator.decision_function(x)) for estimator in estimators], axis=1
        )
        if not np.isfinite(scores).all():
            raise ValueError("SVM decision_function returned non-finite margins")
        return scores

    def _calibrated_svm_base(self, params, max_iter):
        class_weight = params.get(
            "svm_class_weight", getattr(self.args, "svm_class_weight", "balanced")
        )
        if class_weight == "none":
            class_weight = None
        kernel = str(params.get("svm_kernel", getattr(self.args, "svm_kernel", "linear"))).lower()
        common = {
            "C": float(params.get("svm_c", getattr(self.args, "svm_c", 1.0))),
            "class_weight": class_weight,
            "max_iter": int(max_iter),
            "tol": float(getattr(self.args, "svm_tol", 1e-3)),
        }
        if kernel == "rbf":
            return SVC(
                **common,
                kernel="rbf",
                gamma=self._svm_gamma_value(params),
                probability=False,
            )
        if kernel != "linear":
            raise ValueError(f"unsupported SVM kernel: {kernel}")
        return LinearSVC(
            **common,
            dual="auto",
            random_state=self._seed(),
        )

    @staticmethod
    def _calibrated_svm_fold_iterations(estimator):
        iterations = []
        for calibrated in getattr(estimator, "calibrated_classifiers_", []) or []:
            base = getattr(calibrated, "estimator", None)
            value = getattr(base, "n_iter_", None)
            array = np.asarray(value) if value is not None else np.asarray([])
            iterations.append(int(np.max(array)) if array.size else None)
        return iterations

    def _fit_one_calibrated_svm(
        self,
        train_x,
        labels,
        params,
        output_index,
        output_count,
        *,
        emit_progress=False,
    ):
        requested_max_iter = int(getattr(self.args, "svm_max_iter", 2000))
        policy, budgets = self._svm_iteration_budgets(requested_max_iter)
        attempts = []
        fitted = None
        for attempt_index, budget in enumerate(budgets, start=1):
            base = self._calibrated_svm_base(params, budget)
            splitter = StratifiedKFold(
                n_splits=SVM_CALIBRATION_CV_FOLDS,
                shuffle=True,
                random_state=self._seed(),
            )
            candidate = CalibratedClassifierCV(
                estimator=base,
                method="sigmoid",
                cv=splitter,
                ensemble=True,
            )
            if emit_progress:
                print(
                    f"[svm-progress] classifier=svm calibration=sigmoid status=running "
                    f"current={output_index + 1} total={output_count} "
                    f"attempt={attempt_index} attempts={len(budgets)} max_iter={budget}",
                    flush=True,
                )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                candidate.fit(train_x, labels)
            did_warn = any(issubclass(item.category, ConvergenceWarning) for item in caught)
            calibrated = list(getattr(candidate, "calibrated_classifiers_", []) or [])
            bases = [getattr(item, "estimator", None) for item in calibrated]
            bad_fit_status = any(
                isinstance(item, SVC) and int(getattr(item, "fit_status_", 1)) != 0
                for item in bases
            )
            fold_iterations = self._calibrated_svm_fold_iterations(candidate)
            converged = (
                not did_warn
                and not bad_fit_status
                and len(bases) == SVM_CALIBRATION_CV_FOLDS
                and all(item is not None for item in bases)
            )
            attempts.append(
                {
                    "max_iter": int(budget),
                    "n_iter_by_fold": fold_iterations,
                    "status": "converged" if converged else "not_converged",
                }
            )
            if converged:
                fitted = candidate
                break
        if fitted is None:
            raise SVMConvergenceError(
                f"calibrated SVM output {output_index} did not converge after "
                f"max_iter budgets {' -> '.join(str(value) for value in budgets)}",
                details={
                    "classifier": "svm",
                    "svm_calibration": "sigmoid",
                    "output_index": int(output_index),
                    "output_count": int(output_count),
                    "attempts": attempts,
                    "pipeline_stage": "probability_calibration",
                },
            )
        if emit_progress:
            print(
                f"[svm-progress] classifier=svm calibration=sigmoid status=complete "
                f"current={output_index + 1} total={output_count} "
                f"attempt={len(attempts)} attempts={len(budgets)} "
                f"max_iter={fitted.estimator.max_iter}",
                flush=True,
            )
        return fitted, {
            "output_index": int(output_index),
            "output_number": int(output_index) + 1,
            "attempts": attempts,
            "effective_max_iter": int(fitted.estimator.max_iter),
            "recovered": len(attempts) > 1,
        }

    def _fit_calibrated_svm_estimators(
        self,
        train_x,
        train_y,
        params=None,
        progress=False,
        *,
        record_convergence=True,
    ):
        params = params or {}
        train_x, train_y = self._validate_svm_training_data(train_x, train_y)
        class_counts = [
            np.bincount(train_y[:, column].astype(int), minlength=2)
            for column in range(train_y.shape[1])
        ]
        for output_index, counts in enumerate(class_counts):
            if np.any(counts < SVM_CALIBRATION_CV_FOLDS):
                raise SVMProbabilityContractError(
                    f"calibrated SVM output {output_index} requires at least "
                    f"{SVM_CALIBRATION_CV_FOLDS} training samples from each binary class; "
                    f"observed {counts.tolist()}",
                    details={
                        "output_index": int(output_index),
                        "class_counts": counts.tolist(),
                        "required_per_class": SVM_CALIBRATION_CV_FOLDS,
                    },
                )
        estimators = []
        convergence_outputs = []
        columns = range(train_y.shape[1])
        class_iter = tqdm(columns, desc="Training Calibrated SVM") if progress else columns
        for output_index in class_iter:
            fitted, convergence = self._fit_one_calibrated_svm(
                train_x,
                train_y[:, output_index],
                params,
                output_index,
                train_y.shape[1],
                emit_progress=progress,
            )
            estimators.append({"kind": "sklearn", "value": fitted})
            convergence_outputs.append(convergence)
        if record_convergence:
            policy, budgets = self._svm_iteration_budgets(
                int(getattr(self.args, "svm_max_iter", 2000))
            )
            self._last_svm_convergence = {
                "required": True,
                "status": "converged",
                "policy": policy,
                "requested_max_iter": int(budgets[0]),
                "automatic_ceiling": max(int(budgets[0]), 50_000),
                "recovered": any(item["recovered"] for item in convergence_outputs),
                "outputs": convergence_outputs,
            }
        return estimators

    def _probability_ensemble_scores(self, estimators, x):
        scores = []
        for estimator in estimators:
            if estimator["kind"] == "constant":
                scores.append(np.full(x.shape[0], float(estimator["value"]), dtype=np.float32))
            else:
                scores.append(estimator["value"].predict_proba(x)[:, 1])
        matrix = np.stack(scores, axis=1)
        if not np.isfinite(matrix).all():
            raise SVMProbabilityContractError("probability scores contain non-finite values")
        if np.any(matrix < 0.0) or np.any(matrix > 1.0):
            raise SVMProbabilityContractError("probability scores must remain within [0, 1]")
        return matrix

    @staticmethod
    def _expected_calibration_error(labels, probabilities, bins=10):
        labels = np.asarray(labels, dtype=np.float64).reshape(-1)
        probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        assignments = np.digitize(probabilities, np.linspace(0.0, 1.0, bins + 1)[1:-1])
        error = 0.0
        for index in range(bins):
            mask = assignments == index
            if np.any(mask):
                error += float(mask.mean()) * abs(
                    float(probabilities[mask].mean()) - float(labels[mask].mean())
                )
        return float(error)

    def _svm_probability_quality(self, labels, probabilities):
        labels = np.asarray(labels)
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if labels.ndim == 1:
            labels = labels[:, None]
        if probabilities.ndim == 1:
            probabilities = probabilities[:, None]
        if labels.ndim != 2 or probabilities.ndim != 2 or not labels.shape[1]:
            raise SVMProbabilityContractError(
                "probability-quality labels and scores must be non-empty 2D matrices"
            )
        if labels.shape != probabilities.shape:
            raise SVMProbabilityContractError(
                "probability-quality labels and scores have inconsistent shapes"
            )
        if (
            not np.isfinite(probabilities).all()
            or np.any(probabilities < 0.0)
            or np.any(probabilities > 1.0)
        ):
            raise SVMProbabilityContractError(
                "probability-quality scores must be finite and within [0, 1]"
            )
        try:
            labels_finite = bool(np.isfinite(labels).all())
        except TypeError as exc:
            raise SVMProbabilityContractError(
                "probability-quality labels must be finite and binary"
            ) from exc
        if not labels_finite:
            raise SVMProbabilityContractError(
                "probability-quality labels must be finite and binary"
            )
        outputs = []
        for output_index in range(labels.shape[1]):
            raw_labels = labels[:, output_index]
            values = np.unique(raw_labels)
            if not np.all(np.isin(values, [0, 1])):
                raise SVMProbabilityContractError(
                    f"validation output {output_index} requires binary labels; "
                    f"observed {values.tolist()}",
                    details={"output_index": int(output_index)},
                )
            output_labels = raw_labels.astype(int)
            if not np.array_equal(values, np.asarray([0, 1])):
                raise SVMProbabilityContractError(
                    f"validation output {output_index} requires both classes for "
                    f"probability-quality evidence; observed {values.tolist()}",
                    details={"output_index": int(output_index)},
                )
            output_probabilities = probabilities[:, output_index]
            outputs.append(
                {
                    "output_index": int(output_index),
                    "sample_count": int(output_labels.shape[0]),
                    "positive_count": int(output_labels.sum()),
                    "negative_count": int((output_labels == 0).sum()),
                    "brier": float(brier_score_loss(output_labels, output_probabilities)),
                    "log_loss": float(log_loss(output_labels, output_probabilities, labels=[0, 1])),
                    "ece": self._expected_calibration_error(
                        output_labels, output_probabilities, bins=10
                    ),
                    "probability_min": float(output_probabilities.min()),
                    "probability_max": float(output_probabilities.max()),
                }
            )
        quality = {
            "status": "validated",
            "split": "validation",
            "acceptance": "record_and_validate_no_dataset_threshold",
            "ece_bins": 10,
            "outputs": outputs,
            "macro": {
                key: float(np.mean([item[key] for item in outputs]))
                for key in ("brier", "log_loss", "ece")
            },
        }
        self._last_svm_probability_quality = quality
        return quality

    def _fit_rf_model(self, train_x, train_y, params=None, progress=False):
        params = params or {}
        max_depth_value = int(params.get("rf_max_depth", getattr(self.args, "rf_max_depth", 0)))
        max_depth = max_depth_value if max_depth_value > 0 else None
        total_estimators = int(
            params.get("rf_estimators", getattr(self.args, "rf_estimators", 300))
        )
        min_leaf = int(
            params.get("rf_min_samples_leaf", getattr(self.args, "rf_min_samples_leaf", 1))
        )
        y_for_fit = train_y.ravel() if train_y.ndim == 2 and train_y.shape[1] == 1 else train_y
        rf = RandomForestClassifier(
            n_estimators=1 if progress else total_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            n_jobs=-1,
            random_state=self._seed(),
            warm_start=progress,
        )
        if not progress:
            rf.fit(train_x, y_for_fit)
            return rf
        step = min(10, total_estimators)
        trained = 0
        with tqdm(total=total_estimators, desc="Training RF") as pbar:
            while trained < total_estimators:
                next_n = min(trained + step, total_estimators)
                rf.set_params(n_estimators=next_n)
                rf.fit(train_x, y_for_fit)
                pbar.update(next_n - trained)
                trained = next_n
        return rf

    def _fit_extra_trees_model(self, train_x, train_y, params=None, progress=False):
        params = params or {}
        max_depth_value = int(params.get("et_max_depth", getattr(self.args, "et_max_depth", 0)))
        max_depth = max_depth_value if max_depth_value > 0 else None
        total_estimators = int(
            params.get("et_estimators", getattr(self.args, "et_estimators", 300))
        )
        min_leaf = int(
            params.get("et_min_samples_leaf", getattr(self.args, "et_min_samples_leaf", 1))
        )
        y_for_fit = train_y.ravel() if train_y.ndim == 2 and train_y.shape[1] == 1 else train_y
        et = ExtraTreesClassifier(
            n_estimators=1 if progress else total_estimators,
            criterion="gini",
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            max_features="sqrt",
            bootstrap=False,
            n_jobs=-1,
            random_state=self._seed(),
            warm_start=progress,
        )
        if not progress:
            et.fit(train_x, y_for_fit)
            return et
        step = min(10, total_estimators)
        trained = 0
        with tqdm(total=total_estimators, desc="Training ExtraTrees") as pbar:
            while trained < total_estimators:
                next_n = min(trained + step, total_estimators)
                et.set_params(n_estimators=next_n)
                et.fit(train_x, y_for_fit)
                pbar.update(next_n - trained)
                trained = next_n
        return et

    def _positive_proba_column(self, proba, classes):
        classes_list = list(classes)
        if 1 in classes_list:
            return proba[:, classes_list.index(1)]
        return np.zeros(proba.shape[0], dtype=np.float32)

    def _rf_scores(self, rf, x):
        proba = rf.predict_proba(x)
        classes = rf.classes_
        if isinstance(proba, list):
            return np.stack(
                [
                    self._positive_proba_column(item, classes[index])
                    for index, item in enumerate(proba)
                ],
                axis=1,
            )
        return self._positive_proba_column(proba, classes)[:, None]

    def _fit_histgb_estimators(self, train_x, train_y, params=None, progress=False):
        params = params or {}
        features, labels = self._validate_histgb_training_data(train_x, train_y)

        def integer_parameter(name, default):
            raw_value = params.get(name, getattr(self.args, name, default))
            if isinstance(raw_value, bool):
                raise HistGBTrainingDataError(f"{name} must be an integer")
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise HistGBTrainingDataError(f"{name} must be an integer") from exc
            if not np.isfinite(numeric_value) or not numeric_value.is_integer():
                raise HistGBTrainingDataError(f"{name} must be an integer")
            return int(numeric_value)

        def float_parameter(name, default):
            raw_value = params.get(name, getattr(self.args, name, default))
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise HistGBTrainingDataError(f"{name} must be numeric") from exc
            if not np.isfinite(value):
                raise HistGBTrainingDataError(f"{name} must be finite")
            return value

        max_iter = integer_parameter("histgb_max_iter", 300)
        max_leaf_nodes = integer_parameter("histgb_max_leaf_nodes", 31)
        learning_rate = float_parameter("histgb_lr", 0.1)
        l2_regularization = float_parameter("histgb_l2_regularization", 0.0)
        if max_iter < 1:
            raise HistGBTrainingDataError("histgb_max_iter must be at least 1")
        if max_leaf_nodes == 1 or max_leaf_nodes < 0:
            raise HistGBTrainingDataError(
                "histgb_max_leaf_nodes must be 0 (unlimited) or at least 2"
            )
        if learning_rate <= 0.0:
            raise HistGBTrainingDataError("histgb_lr must be greater than 0")
        if l2_regularization < 0.0:
            raise HistGBTrainingDataError(
                "histgb_l2_regularization must be greater than or equal to 0"
            )
        max_leaf_nodes = max_leaf_nodes if max_leaf_nodes > 0 else None
        num_classes = labels.shape[1]
        estimators = []
        class_iter = (
            tqdm(range(num_classes), desc="Training HistGradientBoosting")
            if progress
            else range(num_classes)
        )
        for c in class_iter:
            y = labels[:, c]
            clf = HistGradientBoostingClassifier(
                loss="log_loss",
                max_iter=max_iter,
                max_leaf_nodes=max_leaf_nodes,
                learning_rate=learning_rate,
                l2_regularization=l2_regularization,
                categorical_features="from_dtype",
                warm_start=False,
                early_stopping=False,
                random_state=self._seed(),
            )
            clf.fit(features, y)
            estimators.append({"kind": "sklearn", "value": clf})
        return estimators

    @staticmethod
    def _validate_histgb_training_data(train_x, train_y):
        features = np.asarray(train_x)
        labels = np.asarray(train_y)
        if labels.ndim == 1:
            labels = labels[:, None]
        if features.ndim != 2 or labels.ndim != 2:
            raise HistGBTrainingDataError(
                "HistGB requires two-dimensional feature and label matrices"
            )
        if features.shape[0] == 0 or features.shape[1] == 0:
            raise HistGBTrainingDataError(
                "HistGB requires at least one training sample and one feature"
            )
        if labels.shape[0] != features.shape[0] or labels.shape[1] == 0:
            raise HistGBTrainingDataError("HistGB feature and label shapes are inconsistent")
        if not np.issubdtype(features.dtype, np.number) or np.iscomplexobj(features):
            raise HistGBTrainingDataError("HistGB training features must be numeric")
        if not np.issubdtype(labels.dtype, np.number) or np.iscomplexobj(labels):
            raise HistGBTrainingDataError("HistGB training labels must be numeric")
        if np.isinf(features).any():
            raise HistGBTrainingDataError(
                "HistGB training features may contain NaN missing values but not infinity"
            )
        if not np.isfinite(labels).all():
            raise HistGBTrainingDataError("HistGB training labels must be finite")
        for output_index in range(labels.shape[1]):
            values = np.unique(labels[:, output_index])
            if not np.array_equal(values, np.asarray([0, 1])):
                raise HistGBTrainingDataError(
                    f"HistGB output {output_index} requires both binary classes 0 and 1; "
                    f"observed {values.tolist()}",
                    details={"output_index": output_index},
                )
        return features, labels.astype(int, copy=False)

    @staticmethod
    def _validate_logreg_training_data(train_x, train_y):
        features = np.asarray(train_x)
        labels = np.asarray(train_y)
        if labels.ndim == 1:
            labels = labels[:, None]
        if features.ndim != 2 or labels.ndim != 2:
            raise ValueError("LogReg requires 2D feature and label matrices")
        if features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("LogReg requires non-empty training features")
        if labels.shape[0] != features.shape[0] or labels.shape[1] == 0:
            raise ValueError("LogReg feature and label shapes are inconsistent")
        try:
            finite = bool(np.isfinite(features).all() and np.isfinite(labels).all())
        except TypeError as exc:
            raise ValueError("LogReg features and labels must be numeric") from exc
        if not finite:
            raise ValueError("LogReg features and labels must be finite")
        for output_index in range(labels.shape[1]):
            values = np.unique(labels[:, output_index])
            if not np.array_equal(values, np.asarray([0, 1])):
                raise ValueError(
                    f"LogReg output {output_index} requires both binary classes 0 and 1; "
                    f"observed {values.tolist()}"
                )
        return features, labels

    def _logreg_iteration_budgets(self, initial_max_iter):
        initial = max(1, int(initial_max_iter))
        policy = (
            str(getattr(self.args, "logreg_convergence_policy", "strict") or "strict")
            .strip()
            .lower()
        )
        if policy != "bounded_retry":
            return "strict", [initial]
        ceiling = max(initial, 50_000)
        budgets = []
        for multiplier in (1, 5, 25):
            budget = min(initial * multiplier, ceiling)
            if budget not in budgets:
                budgets.append(budget)
        return "bounded_retry_v1", budgets

    @staticmethod
    def _logreg_actual_iterations(estimator):
        values = np.asarray(getattr(estimator, "n_iter_", []))
        return int(np.max(values)) if values.size else None

    def _fit_one_logreg(
        self, estimator, train_x, labels, output_index, output_count, *, emit_progress=False
    ):
        policy, budgets = self._logreg_iteration_budgets(estimator.max_iter)
        attempts = []
        fitted = None
        for attempt_index, budget in enumerate(budgets, start=1):
            candidate = clone(estimator).set_params(max_iter=int(budget))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                candidate.fit(train_x, labels)
            did_warn = any(issubclass(item.category, ConvergenceWarning) for item in caught)
            actual_iterations = self._logreg_actual_iterations(candidate)
            converged = not did_warn
            attempts.append(
                {
                    "max_iter": int(budget),
                    "n_iter": actual_iterations,
                    "status": "converged" if converged else "not_converged",
                }
            )
            if emit_progress:
                progress_status = (
                    "complete"
                    if converged
                    else "retrying"
                    if attempt_index < len(budgets)
                    else "failed"
                )
                print(
                    f"[logreg-progress] classifier=logreg status="
                    f"{progress_status} "
                    f"current={output_index + 1} total={output_count} "
                    f"attempt={attempt_index} attempts={len(budgets)} max_iter={budget}",
                    flush=True,
                )
            if converged:
                fitted = candidate
                break
        if fitted is None:
            label_name = (
                self._current_logreg_label_names[output_index]
                if output_index < len(self._current_logreg_label_names)
                else ""
            )
            details = {
                "solver": "liblinear",
                "output_index": int(output_index),
                "output_number": int(output_index) + 1,
                "output_count": int(output_count),
                "initial_max_iter": int(budgets[0]),
                "max_iter": int(budgets[-1]),
                "attempt_count": len(attempts),
                "attempted_max_iters": " → ".join(str(value) for value in budgets),
                "actual_n_iters": " → ".join(
                    "unknown" if item["n_iter"] is None else str(item["n_iter"])
                    for item in attempts
                ),
                "convergence_policy": policy,
                "scaler": str(getattr(self.args, "ml_scaler", "none") or "none"),
            }
            if label_name:
                details["label_name"] = str(label_name)
            raise LogRegConvergenceError(
                f"LogReg output {output_index} did not converge after max_iter budgets "
                f"{details['attempted_max_iters']}; increase max_iter or improve feature scaling",
                details=details,
            )
        return fitted, {
            "output_index": int(output_index),
            "output_number": int(output_index) + 1,
            "attempts": attempts,
            "effective_max_iter": int(fitted.max_iter),
            "recovered": len(attempts) > 1,
        }

    def _fit_logreg_estimators(
        self, train_x, train_y, params=None, progress=False, *, record_convergence=True
    ):
        params = params or {}
        train_x, train_y = self._validate_logreg_training_data(train_x, train_y)
        c_value = float(params.get("logreg_c", getattr(self.args, "logreg_c", 1.0)))
        if not np.isfinite(c_value) or c_value <= 0:
            raise ValueError("logreg_c must be finite and greater than 0")
        class_weight = params.get(
            "logreg_class_weight", getattr(self.args, "logreg_class_weight", "balanced")
        )
        class_weight = None if class_weight == "none" else class_weight
        estimators = []
        convergence_outputs = []
        columns = (
            tqdm(range(train_y.shape[1]), desc="Training LogisticRegression")
            if progress
            else range(train_y.shape[1])
        )
        for column in columns:
            estimator = LogisticRegression(
                C=c_value,
                max_iter=int(getattr(self.args, "logreg_max_iter", 2000)),
                class_weight=class_weight,
                solver="liblinear",
                penalty="l2",
                random_state=self._seed(),
            )
            fitted, convergence = self._fit_one_logreg(
                estimator,
                train_x,
                train_y[:, column],
                column,
                train_y.shape[1],
                emit_progress=progress,
            )
            estimators.append({"kind": "sklearn", "value": fitted})
            convergence_outputs.append(convergence)
        if record_convergence:
            self._last_logreg_convergence = {
                "policy": str(getattr(self.args, "logreg_convergence_policy", "strict") or "strict")
                .strip()
                .lower(),
                "requested_max_iter": int(getattr(self.args, "logreg_max_iter", 2000)),
                "outputs": convergence_outputs,
            }
        return estimators

    def _validate_xgb_training_request(self, train_x, train_y, params):
        features = np.asarray(train_x)
        labels = np.asarray(train_y)
        if features.ndim != 2 or labels.ndim != 2:
            raise XGBTrainingDataError(
                "XGBoost requires two-dimensional feature and label matrices"
            )
        if features.shape[0] == 0 or labels.shape[1] == 0:
            raise XGBTrainingDataError("XGBoost requires at least one sample and one output")
        if features.shape[0] != labels.shape[0]:
            raise XGBTrainingDataError(
                "XGBoost feature and label sample counts differ",
                details={
                    "feature_rows": int(features.shape[0]),
                    "label_rows": int(labels.shape[0]),
                },
            )
        if not np.isfinite(features).all() or not np.isfinite(labels).all():
            raise XGBTrainingDataError("XGBoost training features and labels must be finite")
        invalid_outputs = []
        for output_index in range(labels.shape[1]):
            values = np.unique(labels[:, output_index])
            if not np.array_equal(values, np.array([0, 1], dtype=values.dtype)):
                invalid_outputs.append(
                    {"output_index": output_index, "observed_labels": values.tolist()}
                )
        if invalid_outputs:
            raise XGBTrainingDataError(
                "Every XGBoost output requires both binary classes 0 and 1; "
                "constant-output fallbacks are not XGBoost models",
                details={"invalid_outputs": invalid_outputs},
            )
        requested = {
            "n_estimators": int(
                params.get("xgb_estimators", getattr(self.args, "xgb_estimators", 300))
            ),
            "max_depth": int(params.get("xgb_max_depth", getattr(self.args, "xgb_max_depth", 6))),
            "learning_rate": float(params.get("xgb_lr", getattr(self.args, "xgb_lr", 0.1))),
            "subsample": float(
                params.get("xgb_subsample", getattr(self.args, "xgb_subsample", 1.0))
            ),
            "colsample_bytree": float(
                params.get("xgb_colsample", getattr(self.args, "xgb_colsample", 1.0))
            ),
            "reg_alpha": float(
                params.get("xgb_reg_alpha", getattr(self.args, "xgb_reg_alpha", 0.0))
            ),
            "reg_lambda": float(
                params.get("xgb_reg_lambda", getattr(self.args, "xgb_reg_lambda", 1.0))
            ),
        }
        invalid = []
        if requested["n_estimators"] <= 0:
            invalid.append("xgb_estimators must be greater than 0")
        if requested["max_depth"] < 0:
            invalid.append("xgb_max_depth must be 0 (unlimited) or greater")
        if requested["learning_rate"] <= 0:
            invalid.append("xgb_lr must be greater than 0")
        if not 0 < requested["subsample"] <= 1:
            invalid.append("xgb_subsample must be in (0, 1]")
        if not 0 < requested["colsample_bytree"] <= 1:
            invalid.append("xgb_colsample must be in (0, 1]")
        if requested["reg_alpha"] < 0 or requested["reg_lambda"] < 0:
            invalid.append("XGBoost regularization strengths must be non-negative")
        if invalid:
            raise XGBTrainingDataError("; ".join(invalid), details={"requested_params": requested})
        return requested

    def _validate_lgb_training_request(self, train_x, train_y, params):
        features = np.asarray(train_x)
        labels = np.asarray(train_y)
        if features.ndim != 2 or labels.ndim != 2:
            raise LGBTrainingDataError(
                "LightGBM requires two-dimensional feature and label matrices"
            )
        if features.shape[0] == 0 or features.shape[1] == 0 or labels.shape[1] == 0:
            raise LGBTrainingDataError(
                "LightGBM requires at least one sample, one feature, and one output"
            )
        if features.shape[0] != labels.shape[0]:
            raise LGBTrainingDataError(
                "LightGBM feature and label sample counts differ",
                details={
                    "feature_rows": int(features.shape[0]),
                    "label_rows": int(labels.shape[0]),
                },
            )
        try:
            features_finite = bool(np.isfinite(features).all())
            labels_finite = bool(np.isfinite(labels).all())
        except TypeError as exc:
            raise LGBTrainingDataError(
                "LightGBM training features and labels must be numeric"
            ) from exc
        if not features_finite or not labels_finite:
            raise LGBTrainingDataError("LightGBM training features and labels must be finite")
        invalid_outputs = []
        for output_index in range(labels.shape[1]):
            values = np.unique(labels[:, output_index])
            if not np.array_equal(values, np.array([0, 1], dtype=values.dtype)):
                invalid_outputs.append(
                    {"output_index": output_index, "observed_labels": values.tolist()}
                )
        if invalid_outputs:
            raise LGBTrainingDataError(
                "Every LightGBM output requires both binary classes 0 and 1; "
                "constant-output fallbacks are not LightGBM models",
                details={"invalid_outputs": invalid_outputs},
            )
        try:
            requested = {
                "n_estimators": int(
                    params.get("lgb_estimators", getattr(self.args, "lgb_estimators", 300))
                ),
                "max_depth": int(
                    params.get("lgb_max_depth", getattr(self.args, "lgb_max_depth", 6))
                ),
                "learning_rate": float(params.get("lgb_lr", getattr(self.args, "lgb_lr", 0.1))),
                "subsample": float(
                    params.get("lgb_subsample", getattr(self.args, "lgb_subsample", 1.0))
                ),
                "colsample_bytree": float(
                    params.get("lgb_colsample", getattr(self.args, "lgb_colsample", 1.0))
                ),
                "reg_alpha": float(
                    params.get("lgb_reg_alpha", getattr(self.args, "lgb_reg_alpha", 0.0))
                ),
                "reg_lambda": float(
                    params.get("lgb_reg_lambda", getattr(self.args, "lgb_reg_lambda", 0.0))
                ),
            }
        except (TypeError, ValueError, OverflowError) as exc:
            raise LGBTrainingDataError(
                "LightGBM training parameters must be finite numeric values"
            ) from exc
        invalid = []
        if not all(np.isfinite(value) for value in requested.values()):
            invalid.append("LightGBM training parameters must be finite")
        if requested["n_estimators"] <= 0:
            invalid.append("lgb_estimators must be greater than 0")
        if requested["max_depth"] < 0:
            invalid.append("lgb_max_depth must be 0 (unlimited) or greater")
        if requested["learning_rate"] <= 0:
            invalid.append("lgb_lr must be greater than 0")
        if not 0 < requested["subsample"] <= 1:
            invalid.append("lgb_subsample must be in (0, 1]")
        if not 0 < requested["colsample_bytree"] <= 1:
            invalid.append("lgb_colsample must be in (0, 1]")
        if requested["reg_alpha"] < 0 or requested["reg_lambda"] < 0:
            invalid.append("LightGBM regularization strengths must be non-negative")
        if invalid:
            raise LGBTrainingDataError("; ".join(invalid), details={"requested_params": requested})
        return requested

    def _fit_boosting_estimators(self, classifier, train_x, train_y, params=None, progress=False):
        params = params or {}
        if classifier == "xgb" and XGBClassifier is None:
            raise ImportError("XGBoost requires 'xgboost' library. Run `pip install xgboost`.")
        if classifier == "lgb" and LGBMClassifier is None:
            raise ImportError("LightGBM requires 'lightgbm' library. Run `pip install lightgbm`.")
        xgb_params = (
            self._validate_xgb_training_request(train_x, train_y, params)
            if classifier == "xgb"
            else None
        )
        lgb_params = (
            self._validate_lgb_training_request(train_x, train_y, params)
            if classifier == "lgb"
            else None
        )
        estimators = []
        label = "XGBoost" if classifier == "xgb" else "LightGBM"
        columns = (
            tqdm(range(train_y.shape[1]), desc=f"Training {label}")
            if progress and sys.stderr.isatty()
            else range(train_y.shape[1])
        )
        output_count = int(train_y.shape[1])
        lgb_iterations = int(lgb_params["n_estimators"]) if lgb_params else 0
        lgb_total_iterations = output_count * lgb_iterations
        lgb_started_at = time.monotonic()
        for column in columns:
            y = train_y[:, column]
            pos = y.sum()
            neg = len(y) - pos
            scale_pos_weight = float(neg / pos) if pos > 0 else 1.0
            if classifier == "xgb":
                estimator = XGBClassifier(
                    n_estimators=xgb_params["n_estimators"],
                    max_depth=xgb_params["max_depth"],
                    learning_rate=xgb_params["learning_rate"],
                    subsample=xgb_params["subsample"],
                    colsample_bytree=xgb_params["colsample_bytree"],
                    reg_alpha=xgb_params["reg_alpha"],
                    reg_lambda=xgb_params["reg_lambda"],
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_jobs=-1,
                    random_state=self._seed(),
                    scale_pos_weight=scale_pos_weight,
                    tree_method="hist",
                )
            else:
                depth = lgb_params["max_depth"]
                subsample = lgb_params["subsample"]
                if progress:
                    elapsed = time.monotonic() - lgb_started_at
                    emit_run_progress(
                        "classifier_fit",
                        "running",
                        column * lgb_iterations,
                        lgb_total_iterations,
                        feature="lgb",
                        unit="tree",
                        output=column + 1,
                        outputs=output_count,
                        iteration=0,
                        iterations=lgb_iterations,
                        threads=LGB_EXECUTION_POLICY["lgb_n_jobs"],
                        elapsed_seconds=elapsed,
                    )
                estimator = LGBMClassifier(
                    boosting_type="gbdt",
                    num_leaves=31,
                    n_estimators=lgb_params["n_estimators"],
                    max_depth=depth if depth > 0 else -1,
                    learning_rate=lgb_params["learning_rate"],
                    subsample=subsample,
                    subsample_freq=1 if subsample < 1.0 else 0,
                    colsample_bytree=lgb_params["colsample_bytree"],
                    reg_alpha=lgb_params["reg_alpha"],
                    reg_lambda=lgb_params["reg_lambda"],
                    objective="binary",
                    n_jobs=LGB_EXECUTION_POLICY["lgb_n_jobs"],
                    random_state=self._seed(),
                    bagging_seed=self._seed(),
                    feature_fraction_seed=self._seed(),
                    data_random_seed=self._seed(),
                    deterministic=LGB_EXECUTION_POLICY["lgb_deterministic"],
                    force_row_wise=True,
                    device_type=LGB_EXECUTION_POLICY["lgb_device_type"],
                    scale_pos_weight=scale_pos_weight,
                    verbosity=-1,
                )
            if classifier == "lgb" and progress:
                last_emitted_at = time.monotonic()

                def progress_callback(environment):
                    nonlocal last_emitted_at
                    now = time.monotonic()
                    if now - last_emitted_at < 2.0:
                        return
                    completed_iteration = int(
                        environment.iteration - environment.begin_iteration + 1
                    )
                    global_current = column * lgb_iterations + completed_iteration
                    elapsed = now - lgb_started_at
                    eta = None
                    if column >= 2 and global_current > 0:
                        eta = elapsed * (lgb_total_iterations - global_current) / global_current
                    emit_run_progress(
                        "classifier_fit",
                        "running",
                        global_current,
                        lgb_total_iterations,
                        feature="lgb",
                        unit="tree",
                        output=column + 1,
                        outputs=output_count,
                        iteration=completed_iteration,
                        iterations=lgb_iterations,
                        threads=LGB_EXECUTION_POLICY["lgb_n_jobs"],
                        elapsed_seconds=elapsed,
                        eta_seconds=eta,
                    )
                    last_emitted_at = now

                progress_callback.order = 50
                progress_callback.before_iteration = False
                estimator.fit(train_x, y, callbacks=[progress_callback])
            else:
                estimator.fit(train_x, y)
            estimators.append({"kind": "sklearn", "value": estimator})
            if classifier == "lgb" and progress:
                completed = (column + 1) * lgb_iterations
                elapsed = time.monotonic() - lgb_started_at
                eta = None
                if column + 1 >= 2 and completed > 0:
                    eta = elapsed * (lgb_total_iterations - completed) / completed
                emit_run_progress(
                    "classifier_fit",
                    "complete" if column + 1 == output_count else "running",
                    completed,
                    lgb_total_iterations,
                    feature="lgb",
                    unit="tree",
                    output=column + 1,
                    outputs=output_count,
                    iteration=lgb_iterations,
                    iterations=lgb_iterations,
                    threads=LGB_EXECUTION_POLICY["lgb_n_jobs"],
                    elapsed_seconds=elapsed,
                    eta_seconds=eta,
                )
        return estimators

    def _fit_tuning_model(self, classifier, train_x, train_y, params):
        if classifier == "rf":
            return self._fit_rf_model(train_x, train_y, params, progress=False)
        if classifier == "svm":
            if self._svm_calibration() == "sigmoid":
                return self._fit_calibrated_svm_estimators(
                    train_x,
                    train_y,
                    params,
                    record_convergence=False,
                )
            return self._fit_svm_estimators(
                train_x,
                train_y,
                params,
                record_convergence=False,
            )
        if classifier == "logreg":
            return self._fit_logreg_estimators(train_x, train_y, params, record_convergence=False)
        if classifier == "extratrees":
            return self._fit_extra_trees_model(train_x, train_y, params, progress=False)
        if classifier == "histgb":
            return self._fit_histgb_estimators(train_x, train_y, params)
        if classifier in {"xgb", "lgb"}:
            return self._fit_boosting_estimators(classifier, train_x, train_y, params)
        raise ValueError(f"unsupported auto-tune classifier: {classifier}")

    def _tuning_scores(self, classifier, model, x):
        if classifier == "rf":
            return self._rf_scores(model, x)
        if classifier == "svm":
            if self._svm_calibration() == "sigmoid":
                return self._probability_ensemble_scores(model, x)
            return self._svm_scores(model, x)
        if classifier == "extratrees":
            return self._rf_scores(model, x)
        if classifier in {"logreg", "histgb", "xgb", "lgb"}:
            return self._probability_ensemble_scores(model, x)
        raise ValueError(f"unsupported auto-tune classifier: {classifier}")

    def _evaluate_score_matrix(self, scores, y, *, score_kind):
        config = resolve_selection(self.args)
        metrics, _point = Evaluator.fit_and_score(
            y,
            scores,
            score_kind=score_kind,
            objective=config.threshold,
            search_profile=config.search_profile,
        )
        tuning_key = config.tuning.key(metrics)
        if config.tuning.id == "dual_f1_sacc":
            metrics["dual_score"] = tuning_key[0]
        scalar_keys = [
            "acc",
            "sacc",
            "f1",
            "auc",
            "aupr",
            "dual_score",
            *Evaluator.EXTRA_METRIC_KEYS,
        ]
        return {
            key: None if metrics.get(key) is None else float(metrics[key]) for key in scalar_keys
        }

    def _mean_metrics(self, metrics):
        result = {}
        for key in ["acc", "sacc", "f1", "auc", "aupr", "dual_score", *Evaluator.EXTRA_METRIC_KEYS]:
            values = [
                fold_metrics[key] for fold_metrics in metrics if fold_metrics[key] is not None
            ]
            result[key] = float(np.mean(values)) if values else None
        return result

    def _std_metrics(self, metrics):
        result = {}
        for key in ["acc", "sacc", "f1", "auc", "aupr", "dual_score", *Evaluator.EXTRA_METRIC_KEYS]:
            values = [
                fold_metrics[key] for fold_metrics in metrics if fold_metrics[key] is not None
            ]
            result[key] = float(np.std(values)) if values else None
        return result

    def _score_candidate_holdout(self, classifier, train_x, train_y, val_x, val_y, params):
        fold_train_x, fold_val_x = self._preprocess_tuning_split(train_x, train_y, val_x)
        model = self._fit_tuning_model(classifier, fold_train_x, train_y, params)
        scores = self._tuning_scores(classifier, model, fold_val_x)
        score_kind = self._svm_score_kind() if classifier == "svm" else "probability"
        return {
            "metrics": self._evaluate_score_matrix(scores, val_y, score_kind=score_kind),
            "fold_metrics": [],
        }

    def _score_candidate_cv(self, classifier, train_x, train_y, params, folds):
        metrics = []
        score_kind = self._svm_score_kind() if classifier == "svm" else "probability"
        stratified_binary = (
            classifier in {"svm", "logreg"} and train_y.ndim == 2 and train_y.shape[1] == 1
        )
        splitter = (
            StratifiedKFold(n_splits=folds, shuffle=True, random_state=self._seed())
            if stratified_binary
            else KFold(n_splits=folds, shuffle=True, random_state=self._seed())
        )
        split_labels = train_y[:, 0] if stratified_binary else None
        for fold_index, (train_index, val_index) in enumerate(
            splitter.split(train_x, split_labels), start=1
        ):
            fold_train_x, fold_val_x = self._preprocess_tuning_split(
                train_x[train_index], train_y[train_index], train_x[val_index]
            )
            model = self._fit_tuning_model(classifier, fold_train_x, train_y[train_index], params)
            scores = self._tuning_scores(classifier, model, fold_val_x)
            metrics.append(
                {
                    "fold": fold_index,
                    **self._evaluate_score_matrix(
                        scores,
                        train_y[val_index],
                        score_kind=score_kind,
                    ),
                }
            )
        return {
            "metrics": self._mean_metrics(metrics),
            "metrics_std": self._std_metrics(metrics),
            "fold_metrics": metrics,
        }

    def _select_better_tuning_result(self, candidate, best, objective):
        if best is None:
            return True
        if objective == "dual_f1_sacc":
            candidate_metrics = candidate["metrics"]
            best_metrics = best["metrics"]
            spec = resolve_selection(self.args).tuning
            return spec.key(candidate_metrics) > spec.key(best_metrics)
        secondary = "auc" if objective == "f1" else "f1"
        candidate_key = tuple(
            self._tuning_sort_value(candidate["metrics"][key]) for key in (objective, secondary)
        )
        best_key = tuple(
            self._tuning_sort_value(best["metrics"][key]) for key in (objective, secondary)
        )
        return candidate_key > best_key

    @staticmethod
    def _tuning_sort_value(value):
        return float("-inf") if value is None or not np.isfinite(value) else float(value)

    @staticmethod
    def _format_tuning_metric(value):
        return "NA" if value is None or not np.isfinite(value) else f"{value:.4f}"

    def _apply_tuning_params(self, params):
        for key, value in params.items():
            setattr(self.args, key, value)

    def _rank_tuning_candidates(self, candidates, objective):
        if objective == "dual_f1_sacc":
            ranked = sorted(
                candidates,
                key=lambda candidate: (
                    resolve_selection(self.args).tuning.key(candidate["metrics"])
                    if candidate["metrics"]["f1"] is not None
                    and candidate["metrics"]["sacc"] is not None
                    else (float("-inf"),) * 4
                ),
                reverse=True,
            )
            for rank, candidate in enumerate(ranked, start=1):
                candidate["rank"] = rank
            return candidates
        secondary = "auc" if objective == "f1" else "f1"
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                self._tuning_sort_value(candidate["metrics"][objective]),
                self._tuning_sort_value(candidate["metrics"][secondary]),
            ),
            reverse=True,
        )
        for rank, candidate in enumerate(ranked, start=1):
            candidate["rank"] = rank
        return candidates

    def _maybe_auto_tune(self, classifier, train_x, train_y, val_x, val_y):
        if not self._auto_tune_enabled():
            return
        mode = self._tuning_mode()
        objective = self._tuning_objective()
        candidates = self._candidate_params(classifier, mode)[: self._tuning_max_candidates(mode)]
        if not candidates:
            return
        baseline_params = dict(candidates[0])
        folds = 0
        if mode == "cv":
            requested_folds = self._tuning_folds()
            folds = min(max(2, requested_folds), train_x.shape[0])
            if classifier in {"svm", "logreg"} and train_y.ndim == 2 and train_y.shape[1] == 1:
                class_counts = np.bincount(train_y[:, 0].astype(int), minlength=2)
                if np.any(class_counts < 2):
                    raise ValueError(
                        f"{classifier.upper()} cross-validation requires at least two training samples "
                        "from each binary class"
                    )
                folds = min(folds, int(class_counts.min()))
            if folds < 2:
                mode = "holdout"
                folds = 0
        label = classifier.upper()
        print(
            f"[auto-tune] {label}: mode={mode} objective={objective} candidates={len(candidates)}"
            + (f" folds={folds}" if folds else ""),
            flush=True,
        )
        best = None
        candidate_results = []
        for index, params in enumerate(candidates, start=1):
            candidate_error = None
            try:
                if mode == "cv":
                    result = self._score_candidate_cv(classifier, train_x, train_y, params, folds)
                    metric_label = "CV"
                else:
                    result = self._score_candidate_holdout(
                        classifier, train_x, train_y, val_x, val_y, params
                    )
                    metric_label = "Val"
            except (
                SVMConvergenceError,
                SVMProbabilityContractError,
                LogRegConvergenceError,
            ) as exc:
                expected = (
                    classifier == "svm"
                    and isinstance(exc, (SVMConvergenceError, SVMProbabilityContractError))
                ) or (classifier == "logreg" and isinstance(exc, LogRegConvergenceError))
                if not expected:
                    raise
                candidate_error = str(exc)
                result = {
                    "metrics": {
                        key: None
                        for key in [
                            "acc",
                            "sacc",
                            "f1",
                            "auc",
                            "dual_score",
                            *Evaluator.EXTRA_METRIC_KEYS,
                        ]
                    },
                    "metrics_std": {},
                    "fold_metrics": [],
                }
                metric_label = "CV" if mode == "cv" else "Val"
            metrics = result["metrics"]
            candidate = {
                "index": index,
                "params": dict(params),
                "metrics": metrics,
                "metrics_std": result.get("metrics_std", {}),
                "fold_metrics": result.get("fold_metrics", []),
                "status": "failed" if candidate_error else "completed",
            }
            if candidate_error:
                candidate["error"] = candidate_error
            candidate_results.append(candidate)
            print(
                f"[auto-tune] {label} candidate {index}/{len(candidates)} | "
                f"{self._format_params(params)} | {metric_label} F1: "
                f"{self._format_tuning_metric(metrics['f1'])} | {metric_label} SAcc: "
                f"{self._format_tuning_metric(metrics['sacc'])} | {metric_label} Dual: "
                f"{self._format_tuning_metric(metrics.get('dual_score'))} | {metric_label} AUC: "
                f"{self._format_tuning_metric(metrics['auc'])}",
                flush=True,
            )
            if candidate_error:
                print(
                    f"[auto-tune] {label} candidate {index} rejected: {candidate_error}",
                    flush=True,
                )
            progress_total = len(candidates) * (folds or 1)
            progress_current = index * (folds or 1)
            print(
                f"[tuning-progress] classifier={classifier} status=running "
                f"current={progress_current} total={progress_total} candidate={index} "
                f"candidates={len(candidates)} folds={folds or 1}",
                flush=True,
            )
            if not candidate_error and self._select_better_tuning_result(
                candidate, best, objective
            ):
                best = candidate
        if best is None:
            if classifier == "svm":
                raise SVMConvergenceError(
                    "all SVM tuning candidates failed; increase svm_max_iter, relax "
                    "svm_tol, improve feature scaling, or provide enough samples for calibration"
                )
            if classifier == "logreg":
                raise LogRegConvergenceError(
                    "all LogReg tuning candidates failed to converge; increase "
                    "logreg_max_iter or improve feature scaling"
                )
            return
        self._apply_tuning_params(best["params"])
        self._rank_tuning_candidates(candidate_results, objective)
        for candidate in candidate_results:
            candidate["selected"] = candidate["index"] == best["index"]
        self._last_tuning_result = {
            "objective_definition": resolve_selection(self.args).tuning.to_dict(),
            "classifier": classifier,
            "mode": mode,
            "objective": objective,
            "budget": str(getattr(self.args, "ml_tuning_budget", "") or "legacy"),
            "space_version": ML_TUNING_SPACE_VERSION,
            "candidate_space_size": len(self._candidate_params(classifier, mode)),
            "cv_folds": folds,
            "baseline_params": baseline_params,
            "best_params": dict(best["params"]),
            "best_score": (
                resolve_selection(self.args).tuning.key(best["metrics"])[0]
                if objective == "dual_f1_sacc"
                else best["metrics"][objective]
            ),
            "best_metrics": dict(best["metrics"]),
            "candidates_evaluated": len(candidates),
            "candidate_fold_evaluations": len(candidates) * (folds or 1),
            "candidates_succeeded": sum(
                candidate["status"] == "completed" for candidate in candidate_results
            ),
            "candidates_failed": sum(
                candidate["status"] == "failed" for candidate in candidate_results
            ),
            "selected_matches_baseline": dict(best["params"]) == baseline_params,
            "candidates": candidate_results,
        }
        print(
            f"[tuning-progress] classifier={classifier} status=complete current="
            f"{len(candidates) * (folds or 1)} total={len(candidates) * (folds or 1)} "
            f"candidate={best['index']} candidates={len(candidates)} folds={folds or 1}",
            flush=True,
        )
        selected_score = (
            resolve_selection(self.args).tuning.key(best["metrics"])[0]
            if objective == "dual_f1_sacc"
            else best["metrics"][objective]
        )
        print(
            f"[auto-tune] {label} selected | {self._format_params(best['params'])} | "
            f"{objective.upper()}: {self._format_tuning_metric(selected_score)}",
            flush=True,
        )

    def _format_split_metrics(self, label, metrics, aupr=None):
        auc_text = "NA" if metrics.get("auc") is None else f"{metrics['auc']:.4f}"
        message = (
            f"{label} Acc: {metrics['acc']:.4f} | SAcc: {metrics['sacc']:.4f} | "
            f"F1: {metrics['f1']:.4f} | AUC: {auc_text}"
            f"{Evaluator.format_extra_metrics(metrics)}"
        )
        if aupr is not None:
            message += f" | AUPR: {aupr:.4f}"
        return message

    def run_svm(self, train_data, val_data, test_data=None):
        self._reset_artifacts()
        self.args.model = "svm"
        self.args.classifier = "svm"
        self.args.svm_calibration = self._svm_calibration()
        self.args.svm_calibration_cv_folds = SVM_CALIBRATION_CV_FOLDS
        self._current_svm_label_names = list(
            (getattr(self.args, "_artifact_info", {}) or {}).get("label_names")
            or self._label_names(train_data)
        )
        train_x, train_y = self.collect_xy(train_data, "train")
        val_x, val_y = self.collect_xy(val_data, "val")
        test_x, test_y = (
            self.collect_xy(test_data, "test") if test_data is not None else (None, None)
        )
        self._maybe_auto_tune("svm", train_x, train_y, val_x, val_y)
        calibrated = self._svm_calibration() == "sigmoid"
        display_name = "Calibrated SVM" if calibrated else "SVM"
        print(f"{display_name}: applying reduction/scaler...", flush=True)
        train_x, val_x, test_x = self.apply_reduction(train_x, val_x, train_y, test_x)
        train_x, val_x, test_x = self.apply_ml_scaler(train_x, val_x, test_x)
        train_x, val_x, test_x = self.apply_ml_reduction(train_x, val_x, test_x)
        if calibrated:
            clfs = self._fit_calibrated_svm_estimators(train_x, train_y, progress=True)
            val_scores = self._probability_ensemble_scores(clfs, val_x)
            self._svm_probability_quality(val_y, val_scores)
        else:
            clfs = self._fit_svm_estimators(train_x, train_y, progress=True)
            val_scores = self._svm_scores(clfs, val_x)
        score_kind = "probability" if calibrated else "margin"
        metrics, operating_point = self._fit_validation_metrics(
            val_y,
            val_scores,
            score_kind=score_kind,
            label_names=self._label_names(train_data),
        )
        if getattr(self.args, "report_aupr", False):
            aupr = self._compute_aupr(val_y, val_scores)
            print(self._format_split_metrics(f"{display_name} Val", metrics, aupr=aupr))
        else:
            print(self._format_split_metrics(f"{display_name} Val", metrics))
        test_scores = None
        if test_x is not None:
            test_scores = (
                self._probability_ensemble_scores(clfs, test_x)
                if calibrated
                else self._svm_scores(clfs, test_x)
            )
            test_metrics = Evaluator.score_metrics(test_y, test_scores, operating_point)
            if getattr(self.args, "report_aupr", False):
                test_aupr = self._compute_aupr(test_y, test_scores)
                print(
                    self._format_split_metrics(f"{display_name} Test", test_metrics, aupr=test_aupr)
                )
            else:
                print(self._format_split_metrics(f"{display_name} Test", test_metrics))
        bundle = {
            "classifier": "svm",
            "estimator_kind": "calibrated_svm" if calibrated else "svm",
            "estimators": clfs,
            "input_feature_count": int(train_x.shape[1]),
        }
        self._write_classical_analysis(
            train_x, train_y, val_x, val_y, val_scores, test_x, test_y, test_scores, bundle
        )
        self._save_protocol_evaluation(val_y, val_scores, operating_point, test_y, test_scores)
        self._maybe_save_artifact(bundle, train_data)
        return metrics["f1"]

    def run_calibrated_svm(self, train_data, val_data, test_data=None):
        self.args.model = "svm"
        self.args.classifier = "svm"
        self.args.svm_calibration = "sigmoid"
        self.args.svm_calibration_cv_folds = SVM_CALIBRATION_CV_FOLDS
        return self.run_svm(train_data, val_data, test_data)

    def run_logreg(self, train_data, val_data, test_data=None):
        self._reset_artifacts()
        self._current_logreg_label_names = list(
            (getattr(self.args, "_artifact_info", {}) or {}).get("label_names")
            or self._label_names(train_data)
        )
        train_x, train_y = self.collect_xy(train_data, "train")
        val_x, val_y = self.collect_xy(val_data, "val")
        test_x, test_y = (
            self.collect_xy(test_data, "test") if test_data is not None else (None, None)
        )
        self._maybe_auto_tune("logreg", train_x, train_y, val_x, val_y)
        print("LogisticRegression: applying reduction/scaler...", flush=True)
        train_x, val_x, test_x = self.apply_reduction(train_x, val_x, train_y, test_x)
        train_x, val_x, test_x = self.apply_ml_scaler(train_x, val_x, test_x)
        train_x, val_x, test_x = self.apply_ml_reduction(train_x, val_x, test_x)
        estimators = self._fit_logreg_estimators(train_x, train_y, progress=True)
        val_scores = self._probability_ensemble_scores(estimators, val_x)
        metrics, operating_point = self._fit_validation_metrics(
            val_y, val_scores, score_kind="probability", label_names=self._label_names(train_data)
        )
        if getattr(self.args, "report_aupr", False):
            aupr = self._compute_aupr(val_y, val_scores)
            print(self._format_split_metrics("LogReg Val", metrics, aupr=aupr))
        else:
            print(self._format_split_metrics("LogReg Val", metrics))
        test_score_matrix = None
        if test_x is not None:
            test_score_matrix = self._probability_ensemble_scores(estimators, test_x)
            test_metrics = Evaluator.score_metrics(test_y, test_score_matrix, operating_point)
            if getattr(self.args, "report_aupr", False):
                test_aupr = self._compute_aupr(test_y, test_score_matrix)
                print(self._format_split_metrics("LogReg Test", test_metrics, aupr=test_aupr))
            else:
                print(self._format_split_metrics("LogReg Test", test_metrics))
        bundle = {
            "classifier": "logreg",
            "estimator_kind": "probability_ensemble",
            "estimators": estimators,
            "input_feature_count": int(train_x.shape[1]),
        }
        self._write_classical_analysis(
            train_x, train_y, val_x, val_y, val_scores, test_x, test_y, test_score_matrix, bundle
        )
        self._save_protocol_evaluation(
            val_y, val_scores, operating_point, test_y, test_score_matrix
        )
        self._maybe_save_artifact(bundle, train_data)
        return metrics["f1"]

    def run_rf(self, train_data, val_data, test_data=None):
        self._reset_artifacts()
        train_x, train_y = self.collect_xy(train_data, "train")
        val_x, val_y = self.collect_xy(val_data, "val")
        test_x, test_y = (
            self.collect_xy(test_data, "test") if test_data is not None else (None, None)
        )
        self._maybe_auto_tune("rf", train_x, train_y, val_x, val_y)
        print("RF: applying task reduction...", flush=True)
        train_x, val_x, test_x = self.apply_reduction(train_x, val_x, train_y, test_x)
        print("RF: applying ML scaler...", flush=True)
        train_x, val_x, test_x = self.apply_ml_scaler(train_x, val_x, test_x)
        print("RF: applying ML reduction...", flush=True)
        train_x, val_x, test_x = self.apply_ml_reduction(train_x, val_x, test_x)
        rf = self._fit_rf_model(train_x, train_y, progress=True)
        val_scores = self._rf_scores(rf, val_x)
        metrics, operating_point = self._fit_validation_metrics(
            val_y, val_scores, score_kind="probability", label_names=self._label_names(train_data)
        )
        if getattr(self.args, "report_aupr", False):
            aupr = self._compute_aupr(val_y, val_scores)
            print(self._format_split_metrics("RF Val", metrics, aupr=aupr))
        else:
            print(self._format_split_metrics("RF Val", metrics))
        test_scores = None
        if test_x is not None:
            test_scores = self._rf_scores(rf, test_x)
            test_metrics = Evaluator.score_metrics(test_y, test_scores, operating_point)
            if getattr(self.args, "report_aupr", False):
                test_aupr = self._compute_aupr(test_y, test_scores)
                print(self._format_split_metrics("RF Test", test_metrics, aupr=test_aupr))
            else:
                print(self._format_split_metrics("RF Test", test_metrics))
        bundle = {"estimator_kind": "rf", "estimator": rf}
        self._write_classical_analysis(
            train_x, train_y, val_x, val_y, val_scores, test_x, test_y, test_scores, bundle
        )
        self._save_protocol_evaluation(val_y, val_scores, operating_point, test_y, test_scores)
        self._maybe_save_artifact(bundle, train_data)
        return metrics["f1"]

    def run_extratrees(self, train_data, val_data, test_data=None):
        self._reset_artifacts()
        train_x, train_y = self.collect_xy(train_data, "train")
        val_x, val_y = self.collect_xy(val_data, "val")
        test_x, test_y = (
            self.collect_xy(test_data, "test") if test_data is not None else (None, None)
        )
        self._maybe_auto_tune("extratrees", train_x, train_y, val_x, val_y)
        print("ExtraTrees: applying task reduction...", flush=True)
        train_x, val_x, test_x = self.apply_reduction(train_x, val_x, train_y, test_x)
        print("ExtraTrees: applying ML scaler...", flush=True)
        train_x, val_x, test_x = self.apply_ml_scaler(train_x, val_x, test_x)
        print("ExtraTrees: applying ML reduction...", flush=True)
        train_x, val_x, test_x = self.apply_ml_reduction(train_x, val_x, test_x)
        et = self._fit_extra_trees_model(train_x, train_y, progress=True)
        val_scores = self._rf_scores(et, val_x)
        metrics, operating_point = self._fit_validation_metrics(
            val_y, val_scores, score_kind="probability", label_names=self._label_names(train_data)
        )
        if getattr(self.args, "report_aupr", False):
            aupr = self._compute_aupr(val_y, val_scores)
            print(self._format_split_metrics("ExtraTrees Val", metrics, aupr=aupr))
        else:
            print(self._format_split_metrics("ExtraTrees Val", metrics))
        test_scores = None
        if test_x is not None:
            test_scores = self._rf_scores(et, test_x)
            test_metrics = Evaluator.score_metrics(test_y, test_scores, operating_point)
            if getattr(self.args, "report_aupr", False):
                test_aupr = self._compute_aupr(test_y, test_scores)
                print(self._format_split_metrics("ExtraTrees Test", test_metrics, aupr=test_aupr))
            else:
                print(self._format_split_metrics("ExtraTrees Test", test_metrics))
        bundle = {"estimator_kind": "tree_ensemble", "estimator": et, "classifier": "extratrees"}
        self._write_classical_analysis(
            train_x, train_y, val_x, val_y, val_scores, test_x, test_y, test_scores, bundle
        )
        self._save_protocol_evaluation(val_y, val_scores, operating_point, test_y, test_scores)
        self._maybe_save_artifact(bundle, train_data)
        return metrics["f1"]

    def run_histgb(self, train_data, val_data, test_data=None):
        self._reset_artifacts()
        train_x, train_y = self.collect_xy(train_data, "train")
        val_x, val_y = self.collect_xy(val_data, "val")
        test_x, test_y = (
            self.collect_xy(test_data, "test") if test_data is not None else (None, None)
        )
        self._maybe_auto_tune("histgb", train_x, train_y, val_x, val_y)
        print("HistGradientBoosting: applying reduction/scaler...", flush=True)
        train_x, val_x, test_x = self.apply_reduction(train_x, val_x, train_y, test_x)
        train_x, val_x, test_x = self.apply_ml_scaler(train_x, val_x, test_x)
        train_x, val_x, test_x = self.apply_ml_reduction(train_x, val_x, test_x)
        estimators = self._fit_histgb_estimators(train_x, train_y, progress=True)
        val_scores = self._probability_ensemble_scores(estimators, val_x)
        metrics, operating_point = self._fit_validation_metrics(
            val_y, val_scores, score_kind="probability", label_names=self._label_names(train_data)
        )
        if getattr(self.args, "report_aupr", False):
            aupr = self._compute_aupr(val_y, val_scores)
            print(self._format_split_metrics("HistGB Val", metrics, aupr=aupr))
        else:
            print(self._format_split_metrics("HistGB Val", metrics))
        test_scores = None
        if test_x is not None:
            test_scores = self._probability_ensemble_scores(estimators, test_x)
            test_metrics = Evaluator.score_metrics(test_y, test_scores, operating_point)
            if getattr(self.args, "report_aupr", False):
                test_aupr = self._compute_aupr(test_y, test_scores)
                print(self._format_split_metrics("HistGB Test", test_metrics, aupr=test_aupr))
            else:
                print(self._format_split_metrics("HistGB Test", test_metrics))
        bundle = {
            "estimator_kind": "probability_ensemble",
            "estimators": estimators,
            "classifier": "histgb",
            "input_feature_count": int(train_x.shape[1]),
        }
        self._write_classical_analysis(
            train_x, train_y, val_x, val_y, val_scores, test_x, test_y, test_scores, bundle
        )
        self._save_protocol_evaluation(val_y, val_scores, operating_point, test_y, test_scores)
        self._maybe_save_artifact(bundle, train_data)
        return metrics["f1"]

    def run_xgb(self, train_data, val_data, test_data=None):
        self._reset_artifacts()
        if XGBClassifier is None:
            raise ImportError("XGBoost requires 'xgboost' library. Run `pip install xgboost`.")
        train_x, train_y = self.collect_xy(train_data, "train")
        val_x, val_y = self.collect_xy(val_data, "val")
        test_x, test_y = (
            self.collect_xy(test_data, "test") if test_data is not None else (None, None)
        )
        self._maybe_auto_tune("xgb", train_x, train_y, val_x, val_y)
        print("XGBoost: applying reduction/scaler...", flush=True)
        train_x, val_x, test_x = self.apply_reduction(train_x, val_x, train_y, test_x)
        train_x, val_x, test_x = self.apply_ml_scaler(train_x, val_x, test_x)
        train_x, val_x, test_x = self.apply_ml_reduction(train_x, val_x, test_x)
        estimators = self._fit_boosting_estimators("xgb", train_x, train_y, progress=True)
        val_scores = self._probability_ensemble_scores(estimators, val_x)
        metrics, operating_point = self._fit_validation_metrics(
            val_y, val_scores, score_kind="probability", label_names=self._label_names(train_data)
        )
        if getattr(self.args, "report_aupr", False):
            aupr = self._compute_aupr(val_y, val_scores)
            print(self._format_split_metrics("XGBoost Val", metrics, aupr=aupr))
        else:
            print(self._format_split_metrics("XGBoost Val", metrics))
        test_score_matrix = None
        if test_x is not None:
            test_score_matrix = self._probability_ensemble_scores(estimators, test_x)
            test_metrics = Evaluator.score_metrics(test_y, test_score_matrix, operating_point)
            if getattr(self.args, "report_aupr", False):
                test_aupr = self._compute_aupr(test_y, test_score_matrix)
                print(self._format_split_metrics("XGBoost Test", test_metrics, aupr=test_aupr))
            else:
                print(self._format_split_metrics("XGBoost Test", test_metrics))
        bundle = {
            "classifier": "xgb",
            "estimator_kind": "probability_ensemble",
            "estimators": estimators,
            "input_feature_count": int(train_x.shape[1]),
            "_artifact_verification_x": np.asarray(val_x[: min(32, len(val_x))]).copy(),
        }
        self._write_classical_analysis(
            train_x, train_y, val_x, val_y, val_scores, test_x, test_y, test_score_matrix, bundle
        )
        self._save_protocol_evaluation(
            val_y, val_scores, operating_point, test_y, test_score_matrix
        )
        self._maybe_save_artifact(bundle, train_data)
        return metrics["f1"]

    def run_lgb(self, train_data, val_data, test_data=None):
        self._reset_artifacts()
        if LGBMClassifier is None:
            raise ImportError("LightGBM requires 'lightgbm' library. Run `pip install lightgbm`.")
        train_x, train_y = self.collect_xy(train_data, "train")
        val_x, val_y = self.collect_xy(val_data, "val")
        test_x, test_y = (
            self.collect_xy(test_data, "test") if test_data is not None else (None, None)
        )
        self._maybe_auto_tune("lgb", train_x, train_y, val_x, val_y)
        print("LightGBM: applying reduction/scaler...", flush=True)
        train_x, val_x, test_x = self.apply_reduction(train_x, val_x, train_y, test_x)
        train_x, val_x, test_x = self.apply_ml_scaler(train_x, val_x, test_x)
        train_x, val_x, test_x = self.apply_ml_reduction(train_x, val_x, test_x)
        estimators = self._fit_boosting_estimators("lgb", train_x, train_y, progress=True)
        emit_run_progress("finalization", "running", 0, 3, feature="lgb", unit="step")
        val_scores = self._probability_ensemble_scores(estimators, val_x)
        metrics, operating_point = self._fit_validation_metrics(
            val_y, val_scores, score_kind="probability", label_names=self._label_names(train_data)
        )
        if getattr(self.args, "report_aupr", False):
            aupr = self._compute_aupr(val_y, val_scores)
            print(self._format_split_metrics("LightGBM Val", metrics, aupr=aupr))
        else:
            print(self._format_split_metrics("LightGBM Val", metrics))
        test_score_matrix = None
        if test_x is not None:
            test_score_matrix = self._probability_ensemble_scores(estimators, test_x)
            test_metrics = Evaluator.score_metrics(test_y, test_score_matrix, operating_point)
            if getattr(self.args, "report_aupr", False):
                test_aupr = self._compute_aupr(test_y, test_score_matrix)
                print(self._format_split_metrics("LightGBM Test", test_metrics, aupr=test_aupr))
            else:
                print(self._format_split_metrics("LightGBM Test", test_metrics))
        emit_run_progress("finalization", "running", 1, 3, feature="lgb", unit="step")
        bundle = {
            "classifier": "lgb",
            "estimator_kind": "probability_ensemble",
            "estimators": estimators,
            "input_feature_count": int(train_x.shape[1]),
            "_artifact_verification_x": np.asarray(val_x[: min(32, len(val_x))]).copy(),
        }
        self._write_classical_analysis(
            train_x, train_y, val_x, val_y, val_scores, test_x, test_y, test_score_matrix, bundle
        )
        self._save_protocol_evaluation(
            val_y, val_scores, operating_point, test_y, test_score_matrix
        )
        emit_run_progress("finalization", "running", 2, 3, feature="lgb", unit="step")
        self._maybe_save_artifact(bundle, train_data)
        emit_run_progress("finalization", "complete", 3, 3, feature="lgb", unit="step")
        return metrics["f1"]
