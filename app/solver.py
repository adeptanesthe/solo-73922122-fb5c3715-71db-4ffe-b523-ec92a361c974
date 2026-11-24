"""Scientific core: classify and reconstruct two piano takes with SciPy MILP.

Model
-----
A candidate key ``k`` has a non-negative spectral template ``T[k, :]`` over the
measured frequency bands.  A take chooses a gain ``g[k] >= 0`` for every key.
An unplayed key has gain exactly zero; a played key has a gain in the closed
interval ``[lo[k], hi[k]]``.  The take's band energies are
``sum_k g[k] * T[k, :]`` and every band must lie in ``[b - e, b + e]``.

Classification uses two mixed-integer programs, one per hypothesis:

* ``same``: both takes share one binary pressed-vector ``x``.
* ``different``: the two takes have distinct pressed-vectors, enforced with
  ``d[k] = |x1[k] - x2[k]|`` and ``sum(d) >= 1``.

Neither model enumerates key subsets; HiGHS branches on the binaries.  The
three report outcomes follow directly from feasibility:

* same feasible, different infeasible -> key sets are certainly identical;
* same infeasible, different feasible -> key sets are certainly different;
* both feasible                        -> either case is possible;
* both infeasible                      -> no reconstruction, evidence conflict.

A strict pin ``lo == hi`` is handled by the same linearisation
``lo * x <= g <= hi * x``; it remains exact for closed intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

# A confirmation maps (segment index, key index) -> pressed (True/False).
Confirmations = dict[tuple[int, int], bool]

_INF = np.inf
_OPTIMAL = 0
_INFEASIBLE = 2
# Absolute slack used only to absorb floating-point noise when re-checking a
# witness; it is far below any meaningful spectral tolerance.
_FEAS_TOL = 1e-7


class SolverError(Exception):
    """Raised for malformed numerical input (distinct from infeasibility)."""


class SolverInconclusive(Exception):
    """The solver stopped without proving feasibility or infeasibility."""


@dataclass(frozen=True)
class Solution:
    """One feasible reconstruction for a take."""

    pressed: list[int]
    gains: dict[int, float]


@dataclass(frozen=True)
class Outcome:
    """Result of one hypothesis model."""

    feasible: bool
    takes: tuple[Solution, Solution] | None
    evidence: dict[str, object]


@dataclass(frozen=True)
class Verdict:
    """Three-way classification plus witnesses for every feasible hypothesis."""

    kind: Literal["same", "different", "both", "impossible"]
    # Keys "same" and/or "different" mapped to a witness pair for that hypothesis.
    possibilities: dict[str, tuple[Solution, Solution]]
    evidence: dict[str, object]


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def _validate_templates(templates: np.ndarray) -> None:
    if templates.ndim != 2:
        raise SolverError("templates must be a 2-D array (keys x bands)")
    if templates.size == 0:
        raise SolverError("templates must not be empty")
    if not np.all(np.isfinite(templates)):
        raise SolverError("templates contain non-finite values")
    if np.any(templates < 0):
        raise SolverError("templates must be non-negative")


def _validate_energy(name: str, values: np.ndarray, bands: int) -> None:
    if values.shape != (bands,):
        raise SolverError(f"{name} must have {bands} bands, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise SolverError(f"{name} contains non-finite values")


def _validate_error(name: str, values: np.ndarray, bands: int) -> None:
    if values.shape != (bands,):
        raise SolverError(f"{name} must have {bands} bands, got shape {values.shape}")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise SolverError(f"{name} must be finite and non-negative")


def _validate_gains(lo: np.ndarray, hi: np.ndarray, keys: int) -> None:
    for name, arr in (("gain_lower", lo), ("gain_upper", hi)):
        if arr.shape != (keys,):
            raise SolverError(f"{name} must have {keys} entries, got shape {arr.shape}")
        if not np.all(np.isfinite(arr)) or np.any(arr < 0):
            raise SolverError(f"{name} must be finite and non-negative")
    # A played key must scale its template by a *positive* gain.  Allowing
    # gain_lower == 0 would make "played with gain 0" indistinguishable from
    # "not played", which contradicts the problem statement.
    if np.any(lo <= 0):
        raise SolverError("gain_lower must be strictly positive for every key")
    if np.any(lo > hi):
        raise SolverError("gain_lower must not exceed gain_upper for any key")


def _check_confirmations(conf: Confirmations, keys: int) -> None:
    for (seg, key), _pressed in conf.items():
        if seg not in (0, 1):
            raise SolverError(f"confirmation segment {seg} must be 0 or 1")
        if not 0 <= key < keys:
            raise SolverError(f"confirmation key index {key} out of range")


# ---------------------------------------------------------------------------
# MILP construction
# ---------------------------------------------------------------------------
def _add_gain_block(
    n: int,
    m: int,
    templates: np.ndarray,
    energy: np.ndarray,
    error: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    x_off: int,
    g_off: int,
    nv: int,
    lb: np.ndarray,
    ub: np.ndarray,
    rows: list[np.ndarray],
    rlb: list[float],
    rub: list[float],
) -> None:
    """Link one take's binaries to gains and add its band constraints."""
    for k in range(n):
        ub[g_off + k] = hi[k]
        # g - hi*x <= 0
        row = np.zeros(nv)
        row[g_off + k] = 1.0
        row[x_off + k] = -hi[k]
        rows.append(row)
        rlb.append(-_INF)
        rub.append(0.0)
        # lo*x - g <= 0
        row = np.zeros(nv)
        row[x_off + k] = lo[k]
        row[g_off + k] = -1.0
        rows.append(row)
        rlb.append(-_INF)
        rub.append(0.0)
    for j in range(m):
        row = np.zeros(nv)
        row[g_off : g_off + n] = templates[:, j]
        rows.append(row)
        rlb.append(energy[j] - error[j])
        rub.append(energy[j] + error[j])


def _run_milp(
    c: np.ndarray,
    integrality: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    rows: list[np.ndarray],
    rlb: list[float],
    rub: list[float],
    time_limit: float,
) -> object:
    matrix = np.vstack(rows)
    return milp(
        c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(matrix, np.asarray(rlb), np.asarray(rub)),
        options={"time_limit": time_limit, "mip_rel_gap": 0.0, "disp": False},
    )


def _solver_diagnostics(model: str, result: object, conf: Confirmations) -> dict[str, object]:
    return {
        "model": model,
        "solver_status": int(getattr(result, "status", -1)),
        "solver_message": str(getattr(result, "message", "unknown solver status")),
        "mip_node_count": getattr(result, "mip_node_count", None),
        "confirmations": {f"{seg}:{key}": pressed for (seg, key), pressed in sorted(conf.items())},
    }


# ---------------------------------------------------------------------------
# witness extraction and verification
# ---------------------------------------------------------------------------
def _witness(
    templates: np.ndarray,
    n: int,
    bits: np.ndarray,
    raw_gains: np.ndarray,
    energy: np.ndarray,
    error: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> Solution | None:
    """Turn solver output into a verified witness, or ``None`` if unusable."""
    pressed_bits = np.rint(bits).astype(int)
    pressed = [k for k in range(n) if pressed_bits[k] == 1]
    gains = np.zeros(n)
    for k in pressed:
        gains[k] = float(raw_gains[k])
        # Clip only floating-point drift back onto the closed interval.
        if gains[k] < float(lo[k]) and float(lo[k]) - gains[k] <= _FEAS_TOL:
            gains[k] = float(lo[k])
        if gains[k] > float(hi[k]) and gains[k] - float(hi[k]) <= _FEAS_TOL:
            gains[k] = float(hi[k])

    if not _verify(templates, gains, pressed, energy, error, lo, hi):
        # Fallback: re-solve gains for this fixed pressed set.
        recovered = _feasible_gains(templates, pressed, energy, error, lo, hi)
        if recovered is None:
            return None
        gains = recovered
        if not _verify(templates, gains, pressed, energy, error, lo, hi):
            return None

    gain_map = {k: float(gains[k]) for k in pressed}
    return Solution(pressed=pressed, gains=gain_map)


def _verify(
    templates: np.ndarray,
    gains: np.ndarray,
    pressed: list[int],
    energy: np.ndarray,
    error: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> bool:
    allowed = set(pressed)
    for k in range(len(gains)):
        if k in allowed:
            if gains[k] < lo[k] - _FEAS_TOL or gains[k] > hi[k] + _FEAS_TOL:
                return False
        elif gains[k] != 0.0:
            return False
    produced = gains @ templates
    if not np.all(np.isfinite(produced)):
        return False
    return bool(
        np.all(produced <= energy + error + _FEAS_TOL)
        and np.all(produced >= energy - error - _FEAS_TOL)
    )


def _feasible_gains(
    templates: np.ndarray,
    pressed: list[int],
    energy: np.ndarray,
    error: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> np.ndarray | None:
    """Find gains for a fixed pressed set with a small LP (feasibility only)."""
    gains = np.zeros(templates.shape[0])
    if not pressed:
        if np.all(np.abs(energy) <= error + _FEAS_TOL):
            return gains
        return None
    idx = np.asarray(pressed, dtype=int)
    sub = templates[idx, :]
    # sub^T g <= b + e  and  -(sub^T g) <= -(b - e)
    a_ub = np.vstack([sub.T, -sub.T])
    b_ub = np.concatenate([energy + error, -(energy - error)])
    bounds = [(float(lo[k]), float(hi[k])) for k in idx]
    res = linprog(np.zeros(len(idx)), A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return None
    for pos, k in enumerate(idx):
        gains[k] = float(res.x[pos])
    return gains


# ---------------------------------------------------------------------------
# hypothesis models
# ---------------------------------------------------------------------------
def _same_set_model(
    templates: np.ndarray,
    b1: np.ndarray,
    e1: np.ndarray,
    b2: np.ndarray,
    e2: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    conf: Confirmations,
    time_limit: float,
) -> Outcome:
    n, m = templates.shape
    nv = 3 * n
    c = np.zeros(nv)
    c[:n] = 1.0  # prefer small witnesses; any feasible one is valid
    integrality = np.zeros(nv)
    integrality[:n] = 1
    lb = np.zeros(nv)
    ub = np.zeros(nv)
    ub[:n] = 1.0
    rows: list[np.ndarray] = []
    rlb: list[float] = []
    rub: list[float] = []

    _add_gain_block(n, m, templates, b1, e1, lo, hi, 0, n, nv, lb, ub, rows, rlb, rub)
    _add_gain_block(n, m, templates, b2, e2, lo, hi, 0, 2 * n, nv, lb, ub, rows, rlb, rub)

    # Same-set hypothesis: a key's pressed state is shared, so confirmations
    # for the two segments must agree.  Disagreement makes this model
    # infeasible (it is evidence for "different"), not an input error.
    pinned: dict[int, bool] = {}
    pinned_segment: dict[int, int] = {}
    for (seg, key), pressed in conf.items():
        if key in pinned and pinned[key] != pressed:
            return Outcome(
                feasible=False,
                takes=None,
                evidence={
                    "model": "same_set",
                    "reason": "confirmations_imply_different_press_state",
                    "key_index": key,
                    "confirmed": {
                        "segment": pinned_segment[key],
                        "pressed": pinned[key],
                    },
                    "conflicting": {"segment": seg, "pressed": pressed},
                },
            )
        pinned[key] = pressed
        pinned_segment[key] = seg
    for key, pressed in pinned.items():
        lb[key] = ub[key] = 1.0 if pressed else 0.0

    result = _run_milp(c, integrality, lb, ub, rows, rlb, rub, time_limit)
    status = int(getattr(result, "status", -1))
    if status == _OPTIMAL:
        x = np.asarray(result.x)
        sol1 = _witness(templates, n, x[:n], x[n : 2 * n], b1, e1, lo, hi)
        sol2 = _witness(templates, n, x[:n], x[2 * n : 3 * n], b2, e2, lo, hi)
        if sol1 is None or sol2 is None or sol1.pressed != sol2.pressed:
            raise SolverInconclusive("same-set witness failed verification")
        return Outcome(feasible=True, takes=(sol1, sol2), evidence={"model": "same_set", "status": "optimal"})
    if status == _INFEASIBLE:
        return Outcome(feasible=False, takes=None, evidence=_solver_diagnostics("same_set", result, conf))
    raise SolverInconclusive(f"same-set model inconclusive: {getattr(result, 'message', status)}")


def _different_set_model(
    templates: np.ndarray,
    b1: np.ndarray,
    e1: np.ndarray,
    b2: np.ndarray,
    e2: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    conf: Confirmations,
    time_limit: float,
) -> Outcome:
    n, m = templates.shape
    # Layout: x1 | x2 | g1 | g2 | d
    nv = 5 * n
    c = np.zeros(nv)
    c[: 2 * n] = 1.0
    integrality = np.zeros(nv)
    integrality[: 2 * n] = 1
    lb = np.zeros(nv)
    ub = np.zeros(nv)
    ub[: 2 * n] = 1.0
    ub[4 * n : 5 * n] = 1.0
    rows: list[np.ndarray] = []
    rlb: list[float] = []
    rub: list[float] = []

    _add_gain_block(n, m, templates, b1, e1, lo, hi, 0, 2 * n, nv, lb, ub, rows, rlb, rub)
    _add_gain_block(n, m, templates, b2, e2, lo, hi, n, 3 * n, nv, lb, ub, rows, rlb, rub)

    for k in range(n):
        d = 4 * n + k
        # d >= x1 - x2
        row = np.zeros(nv)
        row[d], row[k], row[n + k] = 1.0, -1.0, 1.0
        rows.append(row)
        rlb.append(0.0)
        rub.append(_INF)
        # d >= x2 - x1
        row = np.zeros(nv)
        row[d], row[k], row[n + k] = 1.0, 1.0, -1.0
        rows.append(row)
        rlb.append(0.0)
        rub.append(_INF)
        # d <= x1 + x2
        row = np.zeros(nv)
        row[d], row[k], row[n + k] = 1.0, -1.0, -1.0
        rows.append(row)
        rlb.append(-_INF)
        rub.append(0.0)
        # d <= 2 - x1 - x2
        row = np.zeros(nv)
        row[d], row[k], row[n + k] = 1.0, 1.0, 1.0
        rows.append(row)
        rlb.append(-_INF)
        rub.append(2.0)

    # Require at least one different key.
    row = np.zeros(nv)
    row[4 * n : 5 * n] = 1.0
    rows.append(row)
    rlb.append(1.0)
    rub.append(_INF)

    for (seg, key), pressed in conf.items():
        idx = key if seg == 0 else n + key
        lb[idx] = ub[idx] = 1.0 if pressed else 0.0

    result = _run_milp(c, integrality, lb, ub, rows, rlb, rub, time_limit)
    status = int(getattr(result, "status", -1))
    if status == _OPTIMAL:
        x = np.asarray(result.x)
        sol1 = _witness(templates, n, x[:n], x[2 * n : 3 * n], b1, e1, lo, hi)
        sol2 = _witness(templates, n, x[n : 2 * n], x[3 * n : 4 * n], b2, e2, lo, hi)
        if sol1 is None or sol2 is None or sol1.pressed == sol2.pressed:
            raise SolverInconclusive("different-set witness failed verification")
        return Outcome(
            feasible=True, takes=(sol1, sol2), evidence={"model": "different_set", "status": "optimal"}
        )
    if status == _INFEASIBLE:
        return Outcome(feasible=False, takes=None, evidence=_solver_diagnostics("different_set", result, conf))
    raise SolverInconclusive(f"different-set model inconclusive: {getattr(result, 'message', status)}")


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def classify(
    templates: np.ndarray,
    band_energy: tuple[np.ndarray, np.ndarray],
    band_error: tuple[np.ndarray, np.ndarray],
    gain_lower: np.ndarray,
    gain_upper: np.ndarray,
    confirmations: Confirmations | None = None,
    time_limit: float = 10.0,
) -> Verdict:
    """Classify the two takes and return witnesses for each feasible case.

    Raises :class:`SolverError` for malformed input and
    :class:`SolverInconclusive` if a model could not be decided within the
    time limit, since an undecided model cannot support a definite report.
    """
    templates = np.asarray(templates, dtype=float)
    b1, b2 = (np.asarray(v, dtype=float) for v in band_energy)
    e1, e2 = (np.asarray(v, dtype=float) for v in band_error)
    lo = np.asarray(gain_lower, dtype=float)
    hi = np.asarray(gain_upper, dtype=float)

    _validate_templates(templates)
    n, m = templates.shape
    _validate_energy("band_energy[0]", b1, m)
    _validate_energy("band_energy[1]", b2, m)
    _validate_error("band_error[0]", e1, m)
    _validate_error("band_error[1]", e2, m)
    _validate_gains(lo, hi, n)
    conf: Confirmations = dict(confirmations or {})
    _check_confirmations(conf, n)

    same = _same_set_model(templates, b1, e1, b2, e2, lo, hi, conf, time_limit)
    different = _different_set_model(templates, b1, e1, b2, e2, lo, hi, conf, time_limit)

    possibilities: dict[str, tuple[Solution, Solution]] = {}
    if same.feasible and same.takes is not None:
        possibilities["same"] = same.takes
    if different.feasible and different.takes is not None:
        possibilities["different"] = different.takes

    if len(possibilities) == 2:
        kind: Literal["same", "different", "both", "impossible"] = "both"
    elif same.feasible:
        kind = "same"
    elif different.feasible:
        kind = "different"
    else:
        kind = "impossible"

    evidence: dict[str, object] = {
        "same_set_feasible": same.feasible,
        "different_set_feasible": different.feasible,
        "models": {"same": same.evidence, "different": different.evidence},
    }
    if kind == "impossible":
        evidence["reason"] = "conflicting_evidence"
    return Verdict(kind=kind, possibilities=possibilities, evidence=evidence)
