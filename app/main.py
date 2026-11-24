"""FastAPI surface for the piano-take key-set classifier."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from . import store as store_mod

app = FastAPI(
    title="Piano take key-set classifier",
    version="1.0.0",
    description=(
        "Decide whether two recorded attempts use the same set of piano keys, "
        "report a witness for each feasible case, and let an editor confirm or "
        "correct individual key presses against a versioned report."
    ),
)

STORE = store_mod.Store()


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key_names: list[str] = Field(..., description="Up to 24 candidate key labels, in template order.")
    band_names: list[str] = Field(..., description="Frequency band labels, in template column order.")
    templates: list[list[float]] = Field(
        ..., description="Per-key spectral template; rows match key_names, columns match band_names."
    )
    band_energy_1: list[float] = Field(..., description="Measured band energies for take 1.")
    band_error_1: list[float] = Field(..., description="Allowed absolute error per band for take 1.")
    band_energy_2: list[float] = Field(..., description="Measured band energies for take 2.")
    band_error_2: list[float] = Field(..., description="Allowed absolute error per band for take 2.")
    gain_lower: list[float] = Field(..., description="Closed-interval lower gain bound per key.")
    gain_upper: list[float] = Field(..., description="Closed-interval upper gain bound per key.")
    time_limit: float = Field(
        10.0, gt=0, le=120, description="Per-model solver time limit in seconds."
    )


class ConfirmationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment: Literal[0, 1] = Field(..., description="Which take (0 or 1).")
    key: str = Field(..., description="Key label from key_names.")
    pressed: bool = Field(..., description="Confirmed state: played (true) or not played (false).")


class ConfirmationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_version: int = Field(..., ge=1, description="Report version the editor is looking at.")
    confirmations: list[ConfirmationItem] = Field(..., min_length=1)


@app.exception_handler(store_mod.ApiError)
async def _api_error_handler(_request: Request, exc: store_mod.ApiError) -> JSONResponse:
    body: dict = {"error": {"code": exc.code, "message": exc.message}}
    if exc.detail is not None:
        body["error"]["detail"] = exc.detail
    return JSONResponse(status_code=exc.status, content=body)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyses", status_code=201)
def create_analysis(req: AnalysisRequest) -> dict:
    payload = store_mod.AnalysisInput(
        key_names=list(req.key_names),
        templates=[list(row) for row in req.templates],
        band_names=list(req.band_names),
        band_energy_1=list(req.band_energy_1),
        band_error_1=list(req.band_error_1),
        band_energy_2=list(req.band_energy_2),
        band_error_2=list(req.band_error_2),
        gain_lower=list(req.gain_lower),
        gain_upper=list(req.gain_upper),
        time_limit=req.time_limit,
    )
    analysis = STORE.create(payload)
    return store_mod.serialize_version(analysis, analysis.current)


@app.get("/analyses/{analysis_id}")
def get_analysis(analysis_id: str, version: int | None = None) -> dict:
    analysis, report = STORE.snapshot(analysis_id, version)
    return store_mod.serialize_version(analysis, report)


@app.get("/analyses/{analysis_id}/versions")
def list_versions(analysis_id: str) -> dict:
    current_version, rows = STORE.version_index(analysis_id)
    return {
        "analysis_id": analysis_id,
        "current_version": current_version,
        "versions": [
            {"version": v, "kind": kind, "created_at": created} for v, kind, created in rows
        ],
    }


@app.get("/analyses/{analysis_id}/confirmations")
def get_confirmations(analysis_id: str) -> dict:
    analysis, version, confirmations = STORE.confirmation_snapshot(analysis_id)
    return store_mod.serialize_confirmations(analysis, version, confirmations)


@app.post("/analyses/{analysis_id}/confirmations")
def post_confirmations(analysis_id: str, req: ConfirmationRequest) -> dict:
    analysis = STORE.get(analysis_id)
    index = {name: i for i, name in enumerate(analysis.payload.key_names)}
    incoming: dict[int, dict[int, bool]] = {}
    for item in req.confirmations:
        if item.key not in index:
            raise store_mod.ApiError(
                "invalid_input",
                f"unknown key {item.key!r}",
                detail={"known_keys": analysis.payload.key_names},
            )
        segment = int(item.segment)
        key_index = index[item.key]
        target = incoming.setdefault(segment, {})
        if key_index in target and target[key_index] != item.pressed:
            raise store_mod.ApiError(
                "invalid_input",
                f"contradictory confirmations in one request for segment {segment} "
                f"key {item.key!r}",
            )
        target[key_index] = item.pressed
    updated = STORE.confirm(analysis_id, req.expected_version, incoming)
    return store_mod.serialize_version(updated, updated.current)
