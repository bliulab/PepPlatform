"""Resource bounds for new dataset uploads, independent of training data readers."""

import io
import os
import stat
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

MIB = 1024 * 1024
UPLOAD_CHUNK_BYTES = 64 * 1024
MULTIPART_ALLOWANCE_BYTES = MIB


class DatasetValidationError(ValueError):
    pass


class DatasetArchiveError(DatasetValidationError):
    @property
    def detail(self):
        return {"code": "dataset_upload_invalid_archive", "message": str(self)}


class DatasetUploadLimitError(DatasetValidationError):
    def __init__(self, limit_type, limit, requested):
        self.limit_type = limit_type
        self.limit = limit
        self.requested = requested
        super().__init__(f"Dataset upload exceeds {limit_type} limit ({requested} > {limit})")

    @property
    def detail(self):
        codes = {
            "request_bytes": "dataset_upload_request_too_large",
            "archive_bytes": "dataset_upload_archive_too_large",
            "extracted_bytes": "dataset_upload_extracted_too_large",
            "file_bytes": "dataset_upload_file_too_large",
            "entries": "dataset_upload_too_many_entries",
        }
        return {
            "code": codes[self.limit_type],
            "message": str(self),
            "limit_type": self.limit_type,
            "limit": self.limit,
            "requested": self.requested,
            "params": {"limit_mib": self.limit / MIB, "limit_entries": self.limit},
        }


@dataclass(frozen=True)
class DatasetUploadLimits:
    max_archive_bytes: int = 64 * MIB
    max_extracted_bytes: int = 512 * MIB
    max_file_bytes: int = 256 * MIB
    max_entries: int = 1000

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_env(cls):
        values = {}
        for field, suffix, default, multiplier in (
            ("max_archive_bytes", "ARCHIVE_MIB", 64, MIB),
            ("max_extracted_bytes", "EXTRACTED_MIB", 512, MIB),
            ("max_file_bytes", "FILE_MIB", 256, MIB),
            ("max_entries", "ENTRIES", 1000, 1),
        ):
            key = f"PEP_DATASET_UPLOAD_MAX_{suffix}"
            raw = os.environ.get(key, str(default))
            try:
                value = int(raw)
            except ValueError as exc:
                raise ValueError(f"{key} must be a positive integer") from exc
            if value <= 0:
                raise ValueError(f"{key} must be a positive integer")
            values[field] = value * multiplier
        return cls(**values)

    @property
    def max_request_bytes(self):
        return self.max_archive_bytes + MULTIPART_ALLOWANCE_BYTES

    @staticmethod
    def check(limit_type, limit, actual):
        if actual > limit:
            raise DatasetUploadLimitError(limit_type, limit, actual)


def _member_parts(info):
    name = info.orig_filename
    parts = PurePosixPath(name).parts
    kind = stat.S_IFMT(info.external_attr >> 16)
    if (
        (not parts and (not name or not info.is_dir()))
        or "\x00" in name
        or "\\" in name
        or name.startswith("/")
        or PureWindowsPath(name).drive
        or ".." in parts
        or kind not in {0, stat.S_IFREG, stat.S_IFDIR}
        or (kind == stat.S_IFDIR and not info.is_dir())
        or (info.is_dir() and (kind == stat.S_IFREG or info.file_size != 0))
        or info.flag_bits & 1
        or info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
    ):
        raise DatasetArchiveError("ZIP contains an unsafe entry or unsupported compression method")
    return parts


def _checked_members(archive, limits):
    members = archive.infolist()
    limits.check("entries", limits.max_entries, len(members))
    paths = {}
    checked = []
    declared_total = 0
    for info in members:
        parts = _member_parts(info)
        if parts in paths and not (paths[parts] and info.is_dir()):
            raise DatasetArchiveError("ZIP contains conflicting entry paths")
        paths[parts] = info.is_dir()
        limits.check("file_bytes", limits.max_file_bytes, info.file_size)
        declared_total += info.file_size
        limits.check("extracted_bytes", limits.max_extracted_bytes, declared_total)
        checked.append((info, parts))
    for _info, parts in checked:
        if any(paths.get(parts[:index]) is False for index in range(1, len(parts))):
            raise DatasetArchiveError("ZIP contains conflicting file and directory paths")
    return checked


class _SeekableUploadReader:
    """Python 3.10's SpooledTemporaryFile supports seek but omits seekable()."""

    def __init__(self, stream):
        self.stream = stream

    def seekable(self):
        return True

    def __getattr__(self, name):
        return getattr(self.stream, name)


class _LimitedZipInfoList(list):
    def __init__(self, limits):
        super().__init__()
        self.limits = limits

    def append(self, item):
        self.limits.check("entries", self.limits.max_entries, len(self) + 1)
        super().append(item)


class _LimitedZipFile(zipfile.ZipFile):
    def __init__(self, stream, limits):
        self.upload_limits = limits
        super().__init__(stream)

    def _RealGetContents(self):
        # Bound metadata objects as Python 3.10's ZIP reader builds its directory,
        # including when the ZIP end record understates the actual entry count.
        self.filelist = _LimitedZipInfoList(self.upload_limits)
        super()._RealGetContents()


def extract_dataset_archive(content, target_dir, limits):
    """Accept bytes or a caller-owned seekable file; preserve extracted file bytes."""
    owned_stream = isinstance(content, (bytes, bytearray, memoryview))
    if owned_stream:
        limits.check("archive_bytes", limits.max_archive_bytes, len(content))
        stream = io.BytesIO(content)
    else:
        stream = content
    try:
        stream.seek(0, os.SEEK_END)
        limits.check("archive_bytes", limits.max_archive_bytes, stream.tell())
        stream.seek(0)
        root = Path(target_dir).resolve()
        actual_total = 0
        zip_stream = stream if hasattr(stream, "seekable") else _SeekableUploadReader(stream)
        with _LimitedZipFile(zip_stream, limits) as archive:
            for info, parts in _checked_members(archive, limits):
                target = root.joinpath(*parts)
                if not target.resolve().is_relative_to(root):
                    raise DatasetArchiveError("ZIP entry escapes the upload directory")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                actual_file = 0
                with archive.open(info) as source, target.open("xb") as output:
                    while True:
                        chunk_size = min(
                            UPLOAD_CHUNK_BYTES,
                            limits.max_file_bytes - actual_file + 1,
                            limits.max_extracted_bytes - actual_total + 1,
                        )
                        # Decode errors belong to the archive; filesystem write errors do not.
                        try:
                            chunk = source.read(chunk_size)
                        except (OSError, EOFError, ValueError) as exc:
                            raise DatasetArchiveError("ZIP entry could not be decoded") from exc
                        if not chunk:
                            break
                        actual_file += len(chunk)
                        actual_total += len(chunk)
                        limits.check("file_bytes", limits.max_file_bytes, actual_file)
                        limits.check("extracted_bytes", limits.max_extracted_bytes, actual_total)
                        output.write(chunk)
                if actual_file != info.file_size:
                    raise DatasetArchiveError("ZIP entry size does not match its declared size")
    except (
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
        RuntimeError,
        NotImplementedError,
        UnicodeError,
        EOFError,
        zlib.error,
    ) as exc:
        raise DatasetArchiveError("Invalid, damaged, or unsupported ZIP archive") from exc
    finally:
        if owned_stream:
            stream.close()


class DatasetUploadBodyLimitMiddleware:
    """Bound the upload request before multipart parsing, including chunked bodies."""

    def __init__(self, app, limits):
        self.app = app
        self.limits = limits

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"].rstrip("/") != "/api/datasets/upload"
        ):
            return await self.app(scope, receive, send)

        # Import optional Web dependencies only when serving an HTTP upload.
        from starlette.formparsers import MultiPartException
        from starlette.responses import JSONResponse

        limit = self.limits.max_request_bytes
        for key, value in scope.get("headers", []):
            if key.lower() == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    continue
                if declared > limit:
                    error = DatasetUploadLimitError("request_bytes", limit, declared)
                    return await JSONResponse({"detail": error.detail}, status_code=413)(
                        scope, receive, send
                    )

        received = 0
        limit_error = None
        response_started = False

        async def limited_receive():
            nonlocal received, limit_error
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    limit_error = DatasetUploadLimitError("request_bytes", limit, received)
                    # Multipart parsers close partially spooled uploads on this exception.
                    raise MultiPartException(str(limit_error))
            return message

        async def limit_response():
            nonlocal response_started
            response_started = True
            await JSONResponse({"detail": limit_error.detail}, status_code=413)(
                scope, receive, send
            )

        async def limited_send(message):
            nonlocal response_started
            if limit_error is not None:
                # FastAPI may translate a multipart parsing failure to 400; retain our 413.
                if not response_started:
                    await limit_response()
                return
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, limited_send)
        except Exception:
            if limit_error is None:
                raise
        if limit_error is not None and not response_started:
            await limit_response()
