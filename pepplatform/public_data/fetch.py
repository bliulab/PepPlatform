"""Network access is confined to this module; builds never call it."""

from __future__ import annotations

import json
import platform
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .common import (
    VERSION,
    DataError,
    file_inventory,
    load_recipe,
    new_directory,
    source_identity,
    write_json,
)


class HTTPClient:
    def __init__(self, root, timeout=45, attempts=3, interval=0.5):
        self.root = Path(root)
        (self.root / "responses").mkdir()
        self.timeout, self.attempts, self.interval = timeout, attempts, interval
        self.requests = []
        self.last_request = 0.0

    def get(self, url):
        for attempt in range(self.attempts):
            time.sleep(max(0, self.interval - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            try:
                request = urllib.request.Request(
                    url,
                    headers={
                        "Accept": "application/json",
                        "User-Agent": "PepPlatform-public-data/1",
                    },
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    content = response.read(32 * 1024 * 1024 + 1)
                    if len(content) > 32 * 1024 * 1024:
                        raise DataError("response exceeds 32 MiB; reduce page_size")
                    name = f"responses/{len(self.requests):06d}.json"
                    (self.root / name).write_bytes(content)
                    self.requests.append(
                        {
                            "url": url,
                            "response": name,
                            "status": response.status,
                            "fetched_at": datetime.now(timezone.utc).isoformat(),
                            "headers": {
                                k.lower(): v
                                for k, v in response.headers.items()
                                if k.lower()
                                in {
                                    "date",
                                    "etag",
                                    "last-modified",
                                    "content-range",
                                    "x-uniprot-release",
                                    "x-uniprot-release-date",
                                }
                            },
                        }
                    )
                try:
                    return json.loads(content)
                except (ValueError, UnicodeError) as exc:
                    raise DataError(f"source returned invalid JSON: {url}") from exc
            except urllib.error.HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt + 1 == self.attempts:
                    raise DataError(f"HTTP {exc.code}: {url}") from exc
                retry = exc.headers.get("Retry-After", "")
                time.sleep(min(30, int(retry)) if retry.isdigit() else 2**attempt)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt + 1 == self.attempts:
                    raise DataError(f"request failed: {url}: {exc}") from exc
                time.sleep(2**attempt)
        raise DataError("request retries exhausted")


def url(base, params):
    return base + "?" + urllib.parse.urlencode(params)


def collect(recipe, client):
    source, query = recipe["source"], recipe["query"]
    size, cap = recipe["page_size"], recipe["max_records"]
    records, seen, expected = [], set(), None
    page = 0
    while True:
        if source == "dbaasp":
            response = client.get(
                url("https://dbaasp.org/peptides", dict(query, limit=size, offset=len(records)))
            )
            if (
                not isinstance(response, dict)
                or not isinstance(response.get("data"), list)
                or type(response.get("totalCount")) is not int
            ):
                raise DataError("DBAASP search schema changed")
            rows, total = response["data"], response["totalCount"]
        elif source == "iedb":
            response = client.get(
                url(
                    "https://query-api.iedb.org/mhc_search",
                    dict(
                        query,
                        limit=size,
                        offset=len(records),
                        order="elution_id.asc",
                        select="elution_id,mhc_export(*)",
                    ),
                )
            )
            if not isinstance(response, list):
                raise DataError("IEDB search schema changed")
            rows = []
            for item in response:
                exports = item.get("mhc_export") if isinstance(item, dict) else None
                if (
                    not isinstance(exports, list)
                    or len(exports) != 1
                    or exports[0].get("assay_id") != item.get("elution_id")
                ):
                    raise DataError("IEDB search/export relationship changed")
                rows.append(exports[0])
            total = None
        else:
            response = client.get(
                url(
                    "https://www.ebi.ac.uk/intact/ws/interaction/findInteractions/"
                    + urllib.parse.quote(query["query"], safe=""),
                    {"page": page, "pageSize": size, "sort": "ac,asc"},
                )
            )
            if (
                not isinstance(response, dict)
                or not isinstance(response.get("content"), list)
                or type(response.get("totalElements")) is not int
            ):
                raise DataError("IntAct search schema changed")
            if response.get("size", size) != size or response.get("number", page) != page:
                raise DataError("IntAct server changed pagination parameters")
            rows, total = response["content"], response["totalElements"]
        if page and total != expected:
            raise DataError("source total changed during pagination; fetch a new snapshot")
        expected = total
        if len(rows) > size:
            raise DataError("server ignored requested page size")
        for row in rows:
            if not isinstance(row, dict):
                raise DataError("source row is not an object")
            key = row.get(
                {"dbaasp": "id", "iedb": "assay_id", "intact": "binaryInteractionId"}[source]
            )
            if key is None or str(key) in seen:
                raise DataError("missing or repeated source record ID during pagination")
            if len(records) >= cap:
                raise DataError(
                    "max_records reached before query exhausted; narrow query or increase cap"
                )
            seen.add(str(key))
            if source == "dbaasp":
                row = client.get(f"https://dbaasp.org/peptides/{int(key)}")
                if not isinstance(row, dict) or str(row.get("id")) != str(key):
                    raise DataError("DBAASP detail identity mismatch")
            records.append({"source_id": str(key), "record": row})
        if total is not None:
            if len(records) == total:
                break
            if len(records) > total or not rows:
                raise DataError("pagination ended before reported total")
        elif not rows:
            break
        page += 1
    return records


def fetch(recipe_path, out, *, client_factory=HTTPClient):
    recipe = load_recipe(recipe_path)
    root = new_directory(out)
    manifest = {
        "version": VERSION,
        "kind": "snapshot",
        "complete": False,
        "recipe": recipe,
        "requests": [],
        "resources": {},
        "python": platform.python_version(),
        "fetcher_identity": source_identity(),
        "upstream_version": "recorded in response headers where provided; otherwise not reported",
    }
    client = client_factory(root)
    try:
        (root / "resources").mkdir()
        for kind, source in sorted(recipe["resources"].items()):
            source = Path(source).expanduser()
            if not source.is_absolute():
                source = Path(recipe_path).resolve().parent / source
            target = f"resources/{kind}{'.csv' if kind == 'pseudosequences' else '.json'}"
            shutil.copyfile(source, root / target)
            manifest["resources"][kind] = target
        if "benchmarks" not in manifest["resources"]:
            target = "resources/benchmarks.json"
            write_json(
                root / target,
                {
                    "status": "not_available",
                    "reason": "reference benchmark index not supplied",
                    "task": recipe["task"],
                    "files": [],
                },
            )
            manifest["resources"]["benchmarks"] = target
        records = collect(recipe, client)
        write_json(root / "records.json", records)
        manifest.update(complete=True, record_count=len(records))
    except Exception as exc:
        manifest["error"] = str(exc)
        raise
    finally:
        manifest["requests"] = client.requests
        manifest["files"] = file_inventory(root, {"snapshot.json"})
        write_json(root / "snapshot.json", manifest)
    return manifest
