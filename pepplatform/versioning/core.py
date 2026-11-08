"""Immutable content objects and append-only run bindings.

No training modules are imported here. Snapshot publication precedes database
publication, so interrupted writes can leave unreferenced objects, never a
registered partial object. There is intentionally no automatic garbage collector.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

from sqlalchemy import Column, MetaData, Table, Text, insert, select, update
from sqlalchemy.exc import IntegrityError


SCHEMA = 1
ID_PATTERN = re.compile(r"^[a-z_]+:[0-9a-f]{64}$")
VERSION_METADATA = MetaData()
records = Table(
    "content_versions",
    VERSION_METADATA,
    Column("id", Text, primary_key=True),
    Column("kind", Text, nullable=False),
    Column("identity_json", Text, nullable=False),
    Column("created_at", Text, nullable=False),
)
runs = Table(
    "versioning_runs",
    VERSION_METADATA,
    Column("run_id", Text, primary_key=True),
    Column("owner_user_id", Text),
    Column("status", Text, nullable=False),
    Column("details_json", Text, nullable=False),
    Column("created_at", Text, nullable=False),
    Column("updated_at", Text, nullable=False),
)
bindings = Table(
    "run_version_bindings",
    VERSION_METADATA,
    Column("run_id", Text, primary_key=True),
    Column("role", Text, primary_key=True),
    Column("version_id", Text, nullable=False),
)
relations = Table(
    "version_relations",
    VERSION_METADATA,
    Column("run_id", Text, primary_key=True),
    Column("relation", Text, primary_key=True),
    Column("target", Text, primary_key=True),
)


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def version_id(kind, identity):
    if not re.fullmatch("[a-z_]+", kind):
        raise ValueError("invalid version kind")
    return kind + ":" + hashlib.sha256(canonical(identity)).hexdigest()


def relative_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or str(path) != name:
        raise ValueError("snapshot member must be a canonical relative path")
    return name


def tree_files(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"input directory missing: {root}")
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in {".git", "__pycache__", ".cache"} for part in relative.parts):
            continue
        if path.is_symlink() and path.is_dir():
            raise ValueError(f"directory symlink cannot be silently omitted: {relative}")
        if path.is_file():
            result[relative.as_posix()] = path
    if not result:
        raise ValueError(f"input directory is empty: {root}")
    return result


def copy_file(source, target):
    """Reflink when supported, independent byte copy otherwise; never hardlink."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open("rb") as incoming, target.open("xb") as outgoing:
        try:
            fcntl.ioctl(outgoing.fileno(), 0x40049409, incoming.fileno())  # Linux FICLONE
        except OSError:
            incoming.seek(0)
            outgoing.seek(0)
            outgoing.truncate()
            shutil.copyfileobj(incoming, outgoing, 4 * 1024 * 1024)
        outgoing.flush()
        os.fsync(outgoing.fileno())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp = tempfile.mkstemp(dir=path.parent, prefix=".writing-")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class VersionRepository:
    def __init__(self, engine, root):
        self.engine = engine
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        VERSION_METADATA.create_all(engine)

    @contextmanager
    def locked(self):
        with (self.root / ".publish.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def object_path(self, identifier):
        if not ID_PATTERN.fullmatch(identifier):
            raise ValueError("invalid version ID")
        kind, digest = identifier.split(":")
        return self.root / "objects" / kind / digest

    def get(self, identifier):
        with self.engine.connect() as conn:
            row = conn.execute(select(records).where(records.c.id == identifier)).mappings().first()
        if row is None:
            raise KeyError(identifier)
        return {
            "id": row["id"],
            "kind": row["kind"],
            "identity": json.loads(row["identity_json"]),
            "created_at": row["created_at"],
        }

    def register(self, kind, value):
        identity = {"schema_version": SCHEMA, "kind": kind, **value}
        identifier = version_id(kind, identity)
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    insert(records).values(
                        id=identifier,
                        kind=kind,
                        identity_json=canonical(identity).decode(),
                        created_at=now(),
                    )
                )
        except IntegrityError:
            if self.get(identifier)["identity"] != identity:
                raise ValueError("version identity conflict")
        return self.get(identifier)

    def snapshot(self, kind, files, extra=None):
        if not files:
            raise ValueError("cannot snapshot an empty file set")
        staging_root = self.root / "staging"
        staging_root.mkdir(exist_ok=True)
        temporary = Path(tempfile.mkdtemp(dir=staging_root))
        try:
            manifest = {}
            for name, source in sorted(files.items()):
                relative_name(name)
                source = Path(source)
                before = source.stat()
                target = temporary / "files" / name
                copy_file(source, target)
                copied = sha256(target)
                observed = sha256(source)
                after = source.stat()
                if copied != observed or (before.st_size, before.st_mtime_ns) != (
                    after.st_size,
                    after.st_mtime_ns,
                ):
                    raise ValueError(f"input changed during snapshot: {name}")
                manifest[name] = {"sha256": copied, "bytes": target.stat().st_size}
            identity = {"schema_version": SCHEMA, "kind": kind, **(extra or {}), "files": manifest}
            identifier = version_id(kind, identity)
            destination = self.object_path(identifier)
            atomic_json(temporary / "manifest.json", identity)
            with self.locked():
                if destination.exists():
                    self._verify_object(identifier, identity)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    for member in temporary.rglob("*"):
                        if member.is_file():
                            member.chmod(0o444)
                    os.rename(temporary, destination)
                record = self.register(
                    kind, {k: v for k, v in identity.items() if k not in {"schema_version", "kind"}}
                )
            return record
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def _verify_object(self, identifier, identity):
        root = self.object_path(identifier)
        if json.loads((root / "manifest.json").read_text()) != identity:
            raise ValueError("snapshot manifest changed")
        actual_files = set(tree_files(root / "files"))
        if actual_files != set(identity["files"]):
            raise ValueError("snapshot file set changed")
        for name, expected in identity["files"].items():
            relative_name(name)
            path = root / "files" / name
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != expected["bytes"]
                or sha256(path) != expected["sha256"]
            ):
                raise ValueError(f"snapshot integrity mismatch: {name}")

    def verify(self, identifier):
        record = self.get(identifier)
        if version_id(record["kind"], record["identity"]) != identifier:
            raise ValueError("registered version identity changed")
        if "files" in record["identity"]:
            self._verify_object(identifier, record["identity"])
        return {"id": identifier, "status": "verified", "checked_at": now()}

    def materialize(self, identifier, destination):
        self.verify(identifier)
        record = self.get(identifier)
        destination = Path(destination)
        if destination.exists():
            raise ValueError("materialization destination already exists")
        for name in record["identity"]["files"]:
            copy_file(self.object_path(identifier) / "files" / name, destination / name)
        return destination

    def start_run(self, run_id, owner_user_id=None):
        with self.engine.begin() as conn:
            conn.execute(
                insert(runs).values(
                    run_id=run_id,
                    owner_user_id=owner_user_id,
                    status="preparing",
                    details_json="{}",
                    created_at=now(),
                    updated_at=now(),
                )
            )

    def update_run(self, run_id, status=None, **details):
        with self.engine.begin() as conn:
            row = conn.execute(select(runs).where(runs.c.run_id == run_id)).mappings().one()
            payload = json.loads(row["details_json"])
            payload.update(details)
            conn.execute(
                update(runs)
                .where(runs.c.run_id == run_id)
                .values(
                    status=status or row["status"],
                    details_json=canonical(payload).decode(),
                    updated_at=now(),
                )
            )

    def bind(self, run_id, role, identifier):
        self.get(identifier)
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    insert(bindings).values(run_id=run_id, role=role, version_id=identifier)
                )
        except IntegrityError:
            with self.engine.connect() as conn:
                existing = conn.execute(
                    select(bindings.c.version_id).where(
                        bindings.c.run_id == run_id, bindings.c.role == role
                    )
                ).scalar_one()
            if existing != identifier:
                raise ValueError("run version binding is immutable")

    def relate(self, run_id, relation, target):
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    insert(relations).values(run_id=run_id, relation=relation, target=target)
                )
        except IntegrityError:
            pass

    def run(self, run_id):
        with self.engine.connect() as conn:
            row = conn.execute(select(runs).where(runs.c.run_id == run_id)).mappings().first()
            if row is None:
                return {
                    "schema_version": SCHEMA,
                    "run_id": run_id,
                    "status": "not_recorded",
                    "versions": [],
                    "relations": [],
                }
            refs = (
                conn.execute(
                    select(bindings).where(bindings.c.run_id == run_id).order_by(bindings.c.role)
                )
                .mappings()
                .all()
            )
            edges = (
                conn.execute(select(relations).where(relations.c.run_id == run_id)).mappings().all()
            )
        return {
            "schema_version": SCHEMA,
            "run_id": run_id,
            "status": row["status"],
            "owner_user_id": row["owner_user_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "details": json.loads(row["details_json"]),
            "versions": [{"role": ref["role"], **self.get(ref["version_id"])} for ref in refs],
            "relations": [
                {"relation": edge["relation"], "target": edge["target"]} for edge in edges
            ],
        }

    def referencing_runs(self, identifier):
        with self.engine.connect() as conn:
            return list(
                conn.execute(
                    select(bindings.c.run_id).where(bindings.c.version_id == identifier).distinct()
                ).scalars()
            )

    def export(self, run_id, path):
        atomic_json(path, self.run(run_id))
