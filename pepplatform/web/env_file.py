"""Minimal ``.env`` loader shared by the Web app and operational scripts.

Values already present in ``os.environ`` win, so an operator can override a
``.env`` entry from the process environment without editing the file.
"""

import os
import shlex
from pathlib import Path


def load_env_file(env_path):
    """Load ``KEY=VALUE`` lines from ``env_path`` into ``os.environ``."""
    env_path = Path(env_path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        try:
            parsed = shlex.split(value, comments=True, posix=True)
        except ValueError:
            parsed = [value]
        os.environ[key] = parsed[0] if parsed else ""
