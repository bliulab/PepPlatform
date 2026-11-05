import shutil
from pathlib import Path


def is_deletable_model_artifact(path, paths):
    candidate = Path(path).resolve(strict=False)
    allowed_roots = [paths.experiments.resolve(), paths.models.resolve()]
    return any(candidate != root and candidate.is_relative_to(root) for root in allowed_roots)


def delete_model_artifact(path, paths):
    candidate = Path(path)
    if not candidate.exists() or not is_deletable_model_artifact(candidate, paths):
        return False
    if candidate.is_dir():
        shutil.rmtree(candidate)
    else:
        candidate.unlink()
    return True


def is_deletable_experiment_artifact(path, paths):
    candidate = Path(path).resolve(strict=False)
    experiments_root = paths.experiments.resolve()
    return candidate != experiments_root and candidate.is_relative_to(experiments_root)


def delete_experiment_artifact(experiment_id, paths):
    candidate = paths.experiments / experiment_id
    if not candidate.exists() or not is_deletable_experiment_artifact(candidate, paths):
        return False
    if candidate.is_dir():
        shutil.rmtree(candidate)
    else:
        candidate.unlink()
    return True


def is_deletable_dataset_artifact(path, dataset_id, paths):
    candidate = Path(path).resolve(strict=False)
    dataset_root = (paths.datasets / dataset_id).resolve(strict=False)
    datasets_root = paths.datasets.resolve()
    return (
        dataset_root != datasets_root
        and dataset_root.is_relative_to(datasets_root)
        and candidate.is_relative_to(dataset_root)
    )


def delete_dataset_artifact(dataset, paths):
    if not dataset:
        return False
    if dataset.get("source") != "upload":
        return False
    candidate = Path(dataset["root_path"])
    dataset_root = paths.datasets / dataset["id"]
    if not candidate.exists() or not is_deletable_dataset_artifact(candidate, dataset["id"], paths):
        return False
    if dataset_root.is_dir():
        shutil.rmtree(dataset_root)
    else:
        dataset_root.unlink()
    return True
