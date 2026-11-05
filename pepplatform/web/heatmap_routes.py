"""Authenticated Web heatmap endpoints, independent of legacy analysis polling."""

from fastapi import HTTPException, Query, Request
from fastapi.responses import Response

from .heatmap_data import HeatmapError, capabilities
from .heatmap_jobs import csv_export, page_result


def register_heatmap_routes(app, jobs, current_user, require_experiment):
    def context(experiment_id, request):
        experiment = require_experiment(experiment_id, current_user(request))
        dataset = app.state.store.get_dataset(experiment["dataset_id"])
        if not dataset:
            raise HTTPException(
                404, detail={"code": "dataset_missing", "message": "dataset_missing"}
            )
        return {
            "experiment": experiment,
            "dataset": dataset,
            "runtime_root": str(app.state.paths.root),
        }

    def call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except HeatmapError as exc:
            status = (
                404 if exc.code == "job_missing" else (429 if exc.code == "queue_full" else 422)
            )
            raise HTTPException(status, detail={"code": exc.code, "message": exc.code}) from exc
        except (OSError, ValueError, KeyError, TypeError):
            raise HTTPException(
                422, detail={"code": "evidence_invalid", "message": "evidence_invalid"}
            )

    base = "/api/experiments/{experiment_id}/visualizations"

    @app.get(base + "/capabilities")
    def heatmap_capabilities(experiment_id: str, request: Request):
        return call(capabilities, context(experiment_id, request))

    @app.post(base + "/jobs")
    def heatmap_submit(experiment_id: str, payload: dict, request: Request):
        return call(jobs.submit, context(experiment_id, request), payload)

    @app.get(base + "/jobs/{job_id}")
    def heatmap_status(experiment_id: str, job_id: str, request: Request):
        context(experiment_id, request)
        return call(jobs.get, experiment_id, job_id)

    @app.post(base + "/jobs/{job_id}/cancel")
    def heatmap_cancel(experiment_id: str, job_id: str, request: Request):
        context(experiment_id, request)
        return call(jobs.cancel, experiment_id, job_id)

    @app.get(base + "/jobs/{job_id}/result")
    def heatmap_result(
        experiment_id: str,
        job_id: str,
        request: Request,
        row_offset: int = Query(0, ge=0),
        column_offset: int = Query(0, ge=0),
        search: str = Query("", max_length=200),
    ):
        context(experiment_id, request)
        result = call(jobs.result, experiment_id, job_id)
        return page_result(result, row_offset, column_offset, search)

    @app.get(base + "/jobs/{job_id}/export")
    def heatmap_export(
        experiment_id: str,
        job_id: str,
        request: Request,
        format: str = Query("csv", pattern="^(csv|json)$"),
    ):
        context(experiment_id, request)
        result = call(jobs.result, experiment_id, job_id)
        if format == "json":
            return {k: v for k, v in result.items() if k not in ("cells", "rows", "columns")}
        return Response(
            csv_export(result),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="heatmap.csv"'},
        )
