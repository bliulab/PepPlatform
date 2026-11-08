"""Content identities used to reject stale derived-feature caches."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable


CACHE_PROVENANCE_VERSION = 5
MODEL_IDENTITY_SUFFIXES = {
    ".bin",
    ".json",
    ".model",
    ".pt",
    ".pth",
    ".safetensors",
    ".spm",
    ".txt",
    ".vocab",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_text_items(items: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for item in items:
        encoded = str(item).encode("utf-8", errors="replace")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)
        digest.update(b"\x1e")
    return digest.hexdigest()


@lru_cache(maxsize=32)
def model_source_fingerprint(raw_source: str) -> str:
    """Hash local model content or the exact remote model identifier."""
    source = str(raw_source or "").strip()
    path = Path(os.path.expanduser(source))
    if not path.exists():
        return "remote:" + hashlib.sha256(source.encode("utf-8")).hexdigest()
    if path.is_file():
        return "file:" + sha256_file(path)
    digest = hashlib.sha256()
    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and candidate.suffix.lower() in MODEL_IDENTITY_SUFFIXES
        and ".cache" not in candidate.relative_to(path).parts
    )
    if not files:
        raise ValueError(f"model source has no identity-bearing files: {path}")
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(candidate)))
    return "tree:" + digest.hexdigest()
