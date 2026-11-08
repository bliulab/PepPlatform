"""Upload boundaries and cleanup, without changing training input semantics."""

import asyncio
import io
import stat
import struct
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from pepplatform.web.dataset_uploads import (
    MIB,
    DatasetArchiveError,
    DatasetUploadBodyLimitMiddleware,
    DatasetUploadLimitError,
    DatasetUploadLimits,
)
from pepplatform.web.datasets import DatasetService
from pepplatform.web.paths import RuntimePaths
from pepplatform.web.store import MetadataStore


def _files():
    return {
        f"{split}/{name}": data
        for split in ("train", "val")
        for name, data in (
            ("seqs.fasta", b">s1\naCd\n>s2\nEFG\n"),
            ("labels.csv", b"label\r\n1\r\n0\r\n"),
        )
    }


def _zip(files, compression=zipfile.ZIP_DEFLATED):
    buf = io.BytesIO()
    items = files.items() if isinstance(files, dict) else files
    with zipfile.ZipFile(buf, "w", compression=compression) as archive:
        for name, data in items:
            archive.writestr(name, data)
    return buf.getvalue()


@pytest.fixture
def service(tmp_path):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    return DatasetService(MetadataStore(tmp_path / "metadata.sqlite"), paths)


def _assert_clean(service):
    assert list(service.paths.datasets.iterdir()) == []
    assert service.store.list_datasets() == []


def test_default_limits_and_environment_overrides(monkeypatch):
    for suffix in ("ARCHIVE_MIB", "EXTRACTED_MIB", "FILE_MIB", "ENTRIES"):
        monkeypatch.delenv(f"PEP_DATASET_UPLOAD_MAX_{suffix}", raising=False)
    limits = DatasetUploadLimits.from_env()
    assert limits == DatasetUploadLimits(64 * MIB, 512 * MIB, 256 * MIB, 1000)
    assert limits.max_request_bytes == 65 * MIB
    for suffix, value in (
        ("ARCHIVE_MIB", 3),
        ("EXTRACTED_MIB", 7),
        ("FILE_MIB", 2),
        ("ENTRIES", 11),
    ):
        monkeypatch.setenv(f"PEP_DATASET_UPLOAD_MAX_{suffix}", str(value))
    assert DatasetUploadLimits.from_env() == DatasetUploadLimits(3 * MIB, 7 * MIB, 2 * MIB, 11)


@pytest.mark.parametrize("suffix", ["ARCHIVE_MIB", "EXTRACTED_MIB", "FILE_MIB", "ENTRIES"])
@pytest.mark.parametrize("value", ["", "0", "-1", "1.5", "unlimited"])
def test_invalid_limits_are_rejected(monkeypatch, suffix, value):
    key = f"PEP_DATASET_UPLOAD_MAX_{suffix}"
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=key):
        DatasetUploadLimits.from_env()


def test_exact_boundaries_and_seekable_files_preserve_bytes(service):
    files = _files()
    raw = _zip(files)
    service.upload_limits = DatasetUploadLimits(
        max_archive_bytes=len(raw),
        max_extracted_bytes=sum(map(len, files.values())),
        max_file_bytes=max(map(len, files.values())),
        max_entries=len(files),
    )
    stream = io.BytesIO(raw)
    stream.seek(len(raw))  # The caller's current position must not change the archive size check.
    result = service.upload_archive("pep_function", "Exact limits", "data.zip", stream)
    assert not stream.closed
    for name, data in files.items():
        assert (Path(result["root_path"]) / name).read_bytes() == data


@pytest.mark.parametrize(
    "field,limit_type",
    [
        ("max_archive_bytes", "archive_bytes"),
        ("max_extracted_bytes", "extracted_bytes"),
        ("max_file_bytes", "file_bytes"),
        ("max_entries", "entries"),
    ],
)
def test_direct_service_rejects_one_byte_or_entry_over_limit_and_cleans_up(
    service, field, limit_type
):
    files = _files()
    raw = _zip(files)
    exact = {
        "max_archive_bytes": len(raw),
        "max_extracted_bytes": sum(map(len, files.values())),
        "max_file_bytes": max(map(len, files.values())),
        "max_entries": len(files),
    }
    exact[field] -= 1
    service.upload_limits = DatasetUploadLimits(**exact)
    with pytest.raises(DatasetUploadLimitError) as exc:
        service.upload_archive("pep_function", "Oversized", "data.zip", raw)
    assert exc.value.limit_type == limit_type
    assert exc.value.requested > exc.value.limit
    _assert_clean(service)


def test_high_compression_ratio_is_allowed_within_absolute_limits(service):
    files = {**_files(), "notes.bin": b"0" * (128 * 1024)}
    raw = _zip(files)
    assert sum(map(len, files.values())) / len(raw) > 100
    result = service.upload_archive("pep_function", "Repetitive", "data.zip", raw)
    assert (Path(result["root_path"]) / "notes.bin").read_bytes() == files["notes.bin"]


def test_directory_objects_are_bounded_even_with_forged_end_record_counts(service, monkeypatch):
    import pepplatform.web.dataset_uploads as uploads

    raw = bytearray(_zip(_files()))
    end_record = raw.rindex(b"PK\x05\x06")
    struct.pack_into("<HH", raw, end_record + 8, 0, 0)
    service.upload_limits = replace(service.upload_limits, max_entries=2)

    def unexpected(*args):
        pytest.fail("Too many ZIP entries must fail while parsing metadata")

    monkeypatch.setattr(uploads, "_checked_members", unexpected)
    with pytest.raises(DatasetUploadLimitError) as exc:
        service.upload_archive("pep_function", "Forged directory", "data.zip", raw)
    assert exc.value.limit_type == "entries"
    _assert_clean(service)


def test_ordinary_directory_entries_are_counted_and_preserved(service):
    files = {"./": b"", "./train/": b"", "val/": b"", **_files()}
    service.upload_limits = replace(service.upload_limits, max_entries=len(files))
    result = service.upload_archive("pep_function", "Directories", "data.zip", _zip(files))
    for name, data in _files().items():
        assert (Path(result["root_path"]) / name).read_bytes() == data


@pytest.mark.parametrize("limit_type", ["file_bytes", "extracted_bytes"])
def test_actual_decoded_bytes_are_limited_even_if_metadata_understates_size(
    service, monkeypatch, limit_type
):
    raw = _zip({"notes.bin": b"x"})
    service.upload_limits = DatasetUploadLimits(max_file_bytes=8, max_extracted_bytes=16)
    if limit_type == "extracted_bytes":
        service.upload_limits = replace(
            service.upload_limits, max_file_bytes=16, max_extracted_bytes=8
        )
    # Simulate a decoder producing more than its declared size. Do not allocate a real ZIP bomb.
    monkeypatch.setattr(zipfile.ZipFile, "open", lambda *args, **kwargs: io.BytesIO(b"x" * 32))
    with pytest.raises(DatasetUploadLimitError) as exc:
        service.upload_archive("pep_function", "Decoded overflow", "data.zip", raw)
    assert exc.value.limit_type == limit_type
    _assert_clean(service)


@pytest.mark.parametrize(
    "name",
    ["../outside", "train/../../outside", "/outside", "C:/outside", "a\\outside", "bad\x00name"],
)
def test_unsafe_paths_fail_before_writing_entries(service, name):
    archive_name = name.replace("\x00", "!")
    raw = _zip({**_files(), archive_name: b"x"})
    if "\x00" in name:
        # ZipFile.writestr truncates NUL names; put the invalid bytes in both ZIP headers.
        raw = raw.replace(archive_name.encode(), name.encode())
    with pytest.raises(DatasetArchiveError):
        service.upload_archive("pep_function", "Unsafe", "data.zip", raw)
    _assert_clean(service)
    assert not (service.paths.datasets.parent / "outside").exists()


@pytest.mark.parametrize("kind", [stat.S_IFLNK, stat.S_IFIFO, stat.S_IFCHR])
def test_links_and_special_files_are_rejected(service, kind):
    info = zipfile.ZipInfo("special")
    info.create_system = 3
    info.external_attr = (kind | 0o777) << 16
    with pytest.raises(DatasetArchiveError):
        service.upload_archive("pep_function", "Special", "data.zip", _zip([(info, b"../outside")]))
    _assert_clean(service)


@pytest.mark.parametrize("reverse", [False, True])
def test_conflicting_file_and_directory_paths_are_rejected(service, reverse):
    items = [("train", b"x"), ("train/seqs.fasta", b">s1\nACD\n")]
    if reverse:
        items.reverse()
    with pytest.raises(DatasetArchiveError):
        service.upload_archive("pep_function", "Conflicting", "data.zip", _zip(items))
    _assert_clean(service)


def test_duplicate_file_entries_are_rejected(service):
    with pytest.warns(UserWarning, match="Duplicate name"):
        raw = _zip([("train/seqs.fasta", b"first"), ("train/seqs.fasta", b"second")])
    with pytest.raises(DatasetArchiveError):
        service.upload_archive("pep_function", "Duplicate", "data.zip", raw)
    _assert_clean(service)


@pytest.mark.parametrize(
    "damage", ["not_zip", "truncated", "crc", "unsupported", "bzip2", "lzma", "encrypted"]
)
def test_invalid_archives_are_validation_errors_and_leave_no_files_or_rows(service, damage):
    raw = bytearray(_zip(_files()))
    central = raw.index(b"PK\x01\x02")
    if damage == "not_zip":
        raw = b"not a ZIP"
    elif damage == "truncated":
        raw = raw[:-30]
    elif damage == "crc":
        struct.pack_into("<L", raw, central + 16, 0)
    elif damage in {"unsupported", "bzip2", "lzma"}:
        method = {"unsupported": 99, "bzip2": zipfile.ZIP_BZIP2, "lzma": zipfile.ZIP_LZMA}[damage]
        struct.pack_into("<H", raw, central + 10, method)
    else:
        flags = struct.unpack_from("<H", raw, central + 8)[0]
        struct.pack_into("<H", raw, central + 8, flags | 1)
    with pytest.raises(DatasetArchiveError):
        service.upload_archive("pep_function", "Damaged", "data.zip", raw)
    _assert_clean(service)


@pytest.fixture
def client(tmp_path, monkeypatch, request):
    from fastapi.testclient import TestClient

    from pepplatform.web.app import create_app

    monkeypatch.setenv("PEP_AUTH_ENABLED", "1" if getattr(request, "param", False) else "0")
    for suffix, value in (
        ("ARCHIVE_MIB", 1),
        ("EXTRACTED_MIB", 1),
        ("FILE_MIB", 1),
        ("ENTRIES", 6),
    ):
        monkeypatch.setenv(f"PEP_DATASET_UPLOAD_MAX_{suffix}", str(value))
    root = tmp_path / "project"
    root.mkdir()
    app = create_app(project_root=root, runtime_root=tmp_path / "runtime")
    with TestClient(app) as result:
        yield result


def _post(client, raw, **kwargs):
    return client.post(
        "/api/datasets/upload",
        data={"task_type": "pep_function", "name": "HTTP upload"},
        files={"archive": ("dataset.zip", raw, "application/zip")},
        **kwargs,
    )


def test_http_upload_preserves_bytes_and_owner_file_is_closed(client, monkeypatch):
    from starlette.formparsers import MultiPartParser

    # Capture the actual multipart-upload object without changing parser behavior.
    original = MultiPartParser.on_headers_finished
    uploads = []

    def record(parser):
        original(parser)
        if parser._current_part.file:
            uploads.append(parser._current_part.file)

    monkeypatch.setattr(MultiPartParser, "on_headers_finished", record)
    response = _post(client, _zip(_files()))
    assert response.status_code == 200
    for name, data in _files().items():
        assert (Path(response.json()["root_path"]) / name).read_bytes() == data
    assert uploads and all(upload.file.closed for upload in uploads)


@pytest.mark.parametrize("client", [True], indirect=True)
def test_upload_still_requires_authentication(client):
    response = _post(client, _zip(_files()))
    assert response.status_code == 401
    assert list(client.app.state.paths.datasets.iterdir()) == []
    assert client.app.state.store.list_datasets() == []


@pytest.mark.parametrize("dimension", ["archive", "file", "extracted", "entries", "invalid"])
def test_http_upload_returns_specific_error_and_cleans_up(client, dimension):
    files = _files()
    compression = zipfile.ZIP_DEFLATED
    codes = {
        "archive": "dataset_upload_archive_too_large",
        "file": "dataset_upload_file_too_large",
        "extracted": "dataset_upload_extracted_too_large",
        "entries": "dataset_upload_too_many_entries",
        "invalid": "dataset_upload_invalid_archive",
    }
    if dimension == "archive":
        files["padding.bin"] = b"x" * MIB
        compression = zipfile.ZIP_STORED
    elif dimension == "file":
        files["padding.bin"] = b"x" * (MIB + 1)
    elif dimension == "extracted":
        files.update({"first.bin": b"x" * 600_000, "second.bin": b"x" * 600_000})
    elif dimension == "entries":
        files.update({f"extra-{i}": b"" for i in range(3)})
    raw = b"not a ZIP" if dimension == "invalid" else _zip(files, compression)
    response = _post(client, raw)
    assert response.status_code == (400 if dimension == "invalid" else 413)
    assert response.json()["detail"]["code"] == codes[dimension]
    assert list(client.app.state.paths.datasets.iterdir()) == []
    assert client.app.state.store.list_datasets() == []


def test_declared_oversized_request_is_rejected_before_parsing(client, monkeypatch):
    import starlette.formparsers

    def unexpected(*args, **kwargs):
        pytest.fail("An oversized declared request must not create multipart files")

    monkeypatch.setattr(starlette.formparsers, "SpooledTemporaryFile", unexpected)
    response = _post(client, _zip(_files()), headers={"Content-Length": str(2 * MIB + 1)})
    assert response.status_code == 413
    assert response.json()["detail"]["limit_type"] == "request_bytes"


@pytest.mark.parametrize("length_header", [None, "1"])
def test_chunked_or_understated_request_is_bounded_and_spooled_files_are_closed(
    client, monkeypatch, length_header
):
    import starlette.formparsers

    original = starlette.formparsers.SpooledTemporaryFile
    spooled = []

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        spooled.append(result)
        return result

    monkeypatch.setattr(starlette.formparsers, "SpooledTemporaryFile", record)
    consumed = []

    def chunks():
        yield (
            b'--boundary\r\nContent-Disposition: form-data; name="archive"; '
            b'filename="dataset.zip"\r\nContent-Type: application/zip\r\n\r\n'
        )
        yield b"x" * MIB
        yield b"x" * MIB
        consumed.append("past-limit")
        yield b"\r\n--boundary--\r\n"

    headers = {"Content-Type": "multipart/form-data; boundary=boundary"}
    if length_header is not None:
        headers["Content-Length"] = length_header
    # TestClient's transport can combine generator chunks; use ASGITransport to retain streaming.
    import httpx

    async def send():
        async def body():
            for chunk in chunks():
                yield chunk

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=client.app), base_url="http://testserver"
        ) as http:
            return await http.post("/api/datasets/upload", content=body(), headers=headers)

    response = asyncio.run(send())
    assert response.status_code == 413
    assert response.json()["detail"]["limit_type"] == "request_bytes"
    assert not consumed
    assert spooled and all(file.closed for file in spooled)
    assert list(client.app.state.paths.datasets.iterdir()) == []
    assert client.app.state.store.list_datasets() == []


@pytest.mark.parametrize(
    "path,method,extra,status",
    [
        ("/api/datasets/upload", "POST", 0, 200),
        ("/api/datasets/upload", "POST", 1, 413),
        ("/api/datasets/upload/", "POST", 1, 413),
        ("/api/other", "POST", 1, 200),
        ("/api/datasets/upload", "PUT", 1, 200),
    ],
)
def test_body_limit_boundary_and_endpoint_scope(monkeypatch, path, method, extra, status):
    import pepplatform.web.dataset_uploads as uploads

    monkeypatch.setattr(uploads, "MULTIPART_ALLOWANCE_BYTES", 8)
    limits = DatasetUploadLimits(max_archive_bytes=4)
    messages = []
    requests = iter([{"type": "http.request", "body": b"x" * (12 + extra), "more_body": False}])

    async def receive():
        return next(requests)

    async def send(message):
        messages.append(message)

    async def app(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = DatasetUploadBodyLimitMiddleware(app, limits)
    asyncio.run(
        middleware({"type": "http", "method": method, "path": path, "headers": []}, receive, send)
    )
    assert [m["status"] for m in messages if m["type"] == "http.response.start"] == [status]
