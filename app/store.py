"""Versioned reports, editor confirmations, and transactional recomputation.

Semantics
---------
* Every analysis holds an ordered list of immutable report versions.
* An editor submits confirmations together with the report version they are
  looking at.  The service recomputes from the *full* confirmation set
  (previously confirmed state merged with this request's edits) and the new
  report version and the confirmations become visible together, atomically.
* A stale ``expected_version`` or an infeasible recomputation rejects the whole
  request; the prior confirmations and reports remain unchanged and queryable.
* A confirmation for a key may be *repeated* with the same value (idempotent)
  or changed, which is the editor's correction; both are accepted.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from . import solver


class ApiError(Exception):
    """Domain error carrying an HTTP status, a stable code, and optional detail."""

    def __init__(self, code: str, message: str, status: int = 400, detail: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class AnalysisInput:
    key_names: list[str]
    templates: list[list[float]]
    band_names: list[str]
    band_energy_1: list[float]
    band_error_1: list[float]
    band_energy_2: list[float]
    band_error_2: list[float]
    gain_lower: list[float]
    gain_upper: list[float]
    time_limit: float = 10.0

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "templates": np.asarray(self.templates, dtype=float),
            "b1": np.asarray(self.band_energy_1, dtype=float),
            "e1": np.asarray(self.band_error_1, dtype=float),
            "b2": np.asarray(self.band_energy_2, dtype=float),
            "e2": np.asarray(self.band_error_2, dtype=float),
            "lo": np.asarray(self.gain_lower, dtype=float),
            "hi": np.asarray(self.gain_upper, dtype=float),
        }


@dataclass(frozen=True)
class TakeView:
    pressed_keys: list[str]
    gains: dict[str, float]


@dataclass(frozen=True)
class ReportVersion:
    version: int
    kind: str
    # One entry per reported hypothesis ("same"/"different"), each a pair of takes.
    possibilities: list[tuple[str, list[TakeView]]]
    evidence: dict[str, Any]
    contributions: dict[int, dict[int, bool]]  # segment -> key index -> pressed
    created_at: str


@dataclass
class Analysis:
    analysis_id: str
    payload: AnalysisInput
    reports: list[ReportVersion] = field(default_factory=list)
    confirmations: dict[int, dict[int, bool]] = field(default_factory=dict)

    @property
    def current(self) -> ReportVersion:
        return self.reports[-1]


class Store:
    """Thread-safe in-memory store; deliberately no external database."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._analyses: dict[str, Analysis] = {}

    # -- input validation -------------------------------------------------
    @staticmethod
    def _validate(payload: AnalysisInput) -> None:
        n = len(payload.key_names)
        m = len(payload.band_names)
        if n == 0:
            raise ApiError("invalid_input", "key_names must not be empty")
        if n > 24:
            raise ApiError("invalid_input", "at most 24 candidate keys are supported")
        if m == 0:
            raise ApiError("invalid_input", "band_names must not be empty")
        if len(set(payload.key_names)) != n:
            raise ApiError("invalid_input", "key_names must be unique")
        if len(set(payload.band_names)) != m:
            raise ApiError("invalid_input", "band_names must be unique")
        if len(payload.templates) != n:
            raise ApiError("invalid_input", f"templates must have {n} rows (one per key)")
        if len(payload.gain_lower) != n or len(payload.gain_upper) != n:
            raise ApiError("invalid_input", "gain_lower and gain_upper must have one entry per key")
        for label, band in (
            ("band_energy_1", payload.band_energy_1),
            ("band_error_1", payload.band_error_1),
            ("band_energy_2", payload.band_energy_2),
            ("band_error_2", payload.band_error_2),
        ):
            if len(band) != m:
                raise ApiError("invalid_input", f"{label} must have {m} entries")
        for i, row in enumerate(payload.templates):
            if len(row) != m:
                raise ApiError("invalid_input", f"templates row {i} must have {m} entries")
        if payload.time_limit <= 0:
            raise ApiError("invalid_input", "time_limit must be positive")

        arrays = payload.arrays()
        try:
            solver._validate_templates(arrays["templates"])
            solver._validate_energy("band_energy_1", arrays["b1"], m)
            solver._validate_energy("band_energy_2", arrays["b2"], m)
            solver._validate_error("band_error_1", arrays["e1"], m)
            solver._validate_error("band_error_2", arrays["e2"], m)
            solver._validate_gains(arrays["lo"], arrays["hi"], n)
        except solver.SolverError as exc:
            raise ApiError("invalid_input", str(exc)) from exc

    # -- rendering --------------------------------------------------------
    @staticmethod
    def _take_views(payload: AnalysisInput, takes: tuple[solver.Solution, solver.Solution]) -> list[TakeView]:
        return [
            TakeView(
                pressed_keys=[payload.key_names[k] for k in sol.pressed],
                gains={payload.key_names[k]: float(sol.gains[k]) for k in sol.pressed},
            )
            for sol in takes
        ]

    def _version_from_verdict(
        self,
        payload: AnalysisInput,
        verdict: solver.Verdict,
        contributions: dict[int, dict[int, bool]],
        version: int,
    ) -> ReportVersion:
        possibilities = [
            (name, self._take_views(payload, takes))
            for name, takes in verdict.possibilities.items()
        ]
        return ReportVersion(
            version=version,
            kind=verdict.kind,
            possibilities=possibilities,
            evidence=verdict.evidence,
            contributions={seg: dict(keys) for seg, keys in contributions.items()},
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    # -- public API -------------------------------------------------------
    def create(self, payload: AnalysisInput) -> Analysis:
        self._validate(payload)
        arrays = payload.arrays()
        try:
            verdict = solver.classify(
                arrays["templates"],
                (arrays["b1"], arrays["b2"]),
                (arrays["e1"], arrays["e2"]),
                arrays["lo"],
                arrays["hi"],
                confirmations={},
                time_limit=payload.time_limit,
            )
        except solver.SolverError as exc:
            raise ApiError("invalid_input", str(exc)) from exc
        except solver.SolverInconclusive as exc:
            raise ApiError("solver_inconclusive", str(exc), status=503) from exc

        analysis = Analysis(analysis_id=str(uuid.uuid4()), payload=payload)
        report = self._version_from_verdict(payload, verdict, {}, version=1)
        with self._lock:
            analysis.reports.append(report)
            self._analyses[analysis.analysis_id] = analysis
            return analysis

    def get(self, analysis_id: str) -> Analysis:
        with self._lock:
            analysis = self._analyses.get(analysis_id)
            if analysis is None:
                raise ApiError("not_found", "analysis not found", status=404)
            return analysis

    def snapshot(self, analysis_id: str, version: int | None = None) -> tuple[Analysis, ReportVersion]:
        """Return the analysis and one immutable report, read under the lock."""
        with self._lock:
            analysis = self.get(analysis_id)
            if version is None:
                return analysis, analysis.current
            for report in analysis.reports:
                if report.version == version:
                    return analysis, report
            raise ApiError(
                "not_found",
                f"version {version} not found for analysis {analysis_id}",
                status=404,
                detail={"available_versions": [r.version for r in analysis.reports]},
            )

    def version_index(self, analysis_id: str) -> tuple[int, list[tuple[int, str, str]]]:
        with self._lock:
            analysis = self.get(analysis_id)
            return analysis.current.version, [
                (r.version, r.kind, r.created_at) for r in analysis.reports
            ]

    def confirmation_snapshot(self, analysis_id: str) -> tuple[Analysis, int, dict[int, dict[int, bool]]]:
        with self._lock:
            analysis = self.get(analysis_id)
            return (
                analysis,
                analysis.current.version,
                {seg: dict(keys) for seg, keys in analysis.confirmations.items()},
            )

    def confirm(
        self,
        analysis_id: str,
        expected_version: int,
        incoming: dict[int, dict[int, bool]],
    ) -> Analysis:
        """Apply confirmations atomically against an expected report version.

        Rejected wholesale (prior state untouched) when the version is stale or
        recomputation from the merged confirmation set has no solution.  The
        solver runs outside the lock; a version check inside a second locked
        section guards the commit so two concurrent editors cannot both advance
        the same version.
        """
        with self._lock:
            analysis = self.get(analysis_id)
            current = analysis.current
            if not incoming:
                raise ApiError("invalid_input", "at least one confirmation is required")

            n = len(analysis.payload.key_names)
            for seg, keys in incoming.items():
                if seg not in (0, 1):
                    raise ApiError("invalid_input", f"segment {seg} must be 0 or 1")
                for key in keys:
                    if not 0 <= key < n:
                        raise ApiError("invalid_input", f"key index {key} out of range for {n} keys")

            if expected_version != current.version:
                raise ApiError(
                    "stale_version",
                    f"report has advanced to version {current.version}; request targeted "
                    f"version {expected_version}",
                    status=409,
                    detail={"current_version": current.version, "expected_version": expected_version},
                )

            # Snapshot state and merge: this request may repeat or correct
            # earlier confirmations.
            snapshot = {seg: dict(keys) for seg, keys in analysis.confirmations.items()}
            observed_version = current.version
            merged = {seg: dict(keys) for seg, keys in snapshot.items()}
            for seg, keys in incoming.items():
                target = merged.setdefault(seg, {})
                target.update(keys)

        flat: solver.Confirmations = {
            (seg, key): pressed for seg, keys in merged.items() for key, pressed in keys.items()
        }
        arrays = analysis.payload.arrays()
        try:
            verdict = solver.classify(
                arrays["templates"],
                (arrays["b1"], arrays["b2"]),
                (arrays["e1"], arrays["e2"]),
                arrays["lo"],
                arrays["hi"],
                confirmations=flat,
                time_limit=analysis.payload.time_limit,
            )
        except solver.SolverError as exc:
            raise ApiError("invalid_input", str(exc)) from exc
        except solver.SolverInconclusive as exc:
            raise ApiError("solver_inconclusive", str(exc), status=503) from exc

        if verdict.kind == "impossible":
            raise ApiError(
                "infeasible_after_confirmations",
                "no reconstruction fits both takes under the confirmed edits; "
                "the confirmation was rolled back",
                status=409,
                detail={
                    "rejected_confirmations": {
                        str(seg): {analysis.payload.key_names[k]: v for k, v in keys.items()}
                        for seg, keys in incoming.items()
                    },
                    "evidence": verdict.evidence,
                    "current_version": observed_version,
                    "current_confirmations": {
                        str(seg): {analysis.payload.key_names[k]: v for k, v in keys.items()}
                        for seg, keys in snapshot.items()
                    },
                },
            )

        report = self._version_from_verdict(analysis.payload, verdict, merged, observed_version + 1)
        with self._lock:
            # Re-check under the lock: another editor may have advanced the report
            # while the solver was running.  Reject rather than append out of order.
            if analysis.current.version != observed_version:
                raise ApiError(
                    "stale_version",
                    f"report advanced to version {analysis.current.version} during recomputation",
                    status=409,
                    detail={
                        "current_version": analysis.current.version,
                        "expected_version": expected_version,
                    },
                )
            analysis.confirmations = merged
            analysis.reports.append(report)
            return analysis


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------
def serialize_version(analysis: Analysis, report: ReportVersion) -> dict[str, Any]:
    payload = analysis.payload

    def confirmations(contrib: dict[int, dict[int, bool]]) -> dict[str, dict[str, bool]]:
        return {
            str(seg): {payload.key_names[k]: pressed for k, pressed in sorted(keys.items())}
            for seg, keys in sorted(contrib.items())
        }

    return {
        "analysis_id": analysis.analysis_id,
        "version": report.version,
        "kind": report.kind,
        "verdict": _verdict_text(report.kind),
        "created_at": report.created_at,
        "key_names": payload.key_names,
        "band_names": payload.band_names,
        "possibilities": [
            {
                "hypothesis": name,
                "takes": [
                    {"segment": idx, "pressed_keys": take.pressed_keys, "gains": take.gains}
                    for idx, take in enumerate(takes)
                ],
            }
            for name, takes in report.possibilities
        ],
        "evidence": report.evidence,
        "confirmations": confirmations(report.contributions),
    }


def _verdict_text(kind: str) -> str:
    return {
        "same": "The two takes certainly use the same key set.",
        "different": "The two takes certainly use different key sets.",
        "both": "Either case is possible: both a shared and a differing key set fit the evidence.",
        "impossible": "No reconstruction exists; the recorded evidence conflicts.",
    }.get(kind, kind)


def serialize_confirmations(
    analysis: Analysis,
    current_version: int,
    confirmations: dict[int, dict[int, bool]],
) -> dict[str, Any]:
    payload = analysis.payload
    return {
        "analysis_id": analysis.analysis_id,
        "current_version": current_version,
        "confirmations": {
            str(seg): {payload.key_names[k]: pressed for k, pressed in sorted(keys.items())}
            for seg, keys in sorted(confirmations.items())
        },
    }
