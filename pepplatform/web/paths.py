from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class RuntimePaths:
    project_root: Path
    root: Path
    datasets: Path
    experiments: Path
    models: Path
    database: Path

    @classmethod
    def from_project_root(cls, project_root, runtime_root=None):
        project_root = Path(project_root).resolve()
        runtime_root = runtime_root or os.environ.get("PEP_RUNTIME_ROOT")
        root = Path(runtime_root).resolve() if runtime_root else project_root / ".pepplatform_web"
        return cls(
            project_root=project_root,
            root=root,
            datasets=root / "datasets",
            experiments=root / "experiments",
            models=root / "models",
            database=root / "metadata.sqlite3",
        )

    def ensure(self):
        for path in [self.root, self.datasets, self.experiments, self.models]:
            path.mkdir(parents=True, exist_ok=True)
        return self
