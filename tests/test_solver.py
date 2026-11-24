"""Solver-level tests: closed intervals, strict gains, scale, and brute-force cross-checks."""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from app import solver


def _feasible_set_bruteforce(templates, energy, error, lo, hi):
    """Enumerate pressed subsets for a single take. Assertion helper only."""
    n = templates.shape[0]
    out = []
    for size in range(n + 1):
        for combo in itertools.combinations(range(n), size):
            g = solver._feasible_gains(
                templates, list(combo), energy, error, lo, hi
            )
            if g is not None:
                out.append(frozenset(combo))
    return out


def _random_instance(rng, n, m):
    templates = rng.integers(1, 4, size=(n, m)).astype(float)
    lo = rng.uniform(0.1, 0.4, size=n)
    hi = rng.uniform(0.5, 2.0, size=n)
    return templates, lo, hi


def test_strict_gain_interval() -> None:
    # lo == hi == 1 for the single key: gain is fixed.
    templates = np.array([[1.0, 0.0], [0.0, 1.0]])
    lo = np.array([1.0, 1.0])
    hi = np.array([1.0, 1.0])
    b = np.array([1.0, 0.0])
    e = np.array([0.0, 0.0])
    verdict = solver.classify(templates, (b, b), (e, e), lo, hi)
    assert verdict.kind == "same"
    sol = verdict.possibilities["same"][0]
    assert sol.pressed == [0]
    assert sol.gains[0] == pytest.approx(1.0)


def test_zero_gain_lower_rejected() -> None:
    # gain_lower == 0 would conflate "played with zero gain" and "not played".
    templates = np.array([[1.0]])
    lo = np.array([0.0])
    hi = np.array([2.0])
    b = np.array([1.0])
    e = np.array([0.0])
    with pytest.raises(solver.SolverError, match="strictly positive"):
        solver.classify(templates, (b, b), (e, e), lo, hi)


def test_fixed_gain_midpoint() -> None:
    templates = np.array([[1.0]])
    lo = np.array([0.5])
    hi = np.array([2.0])
    b = np.array([1.0])
    e = np.array([0.0])
    verdict = solver.classify(templates, (b, b), (e, e), lo, hi)
    assert verdict.kind == "same"
    assert verdict.possibilities["same"][0].gains[0] == pytest.approx(1.0)


def test_infeasible_band_energy() -> None:
    templates = np.array([[1.0]])
    lo = np.array([0.5])
    hi = np.array([1.0])
    b = np.array([5.0])
    e = np.array([0.1])
    verdict = solver.classify(templates, (b, b), (e, e), lo, hi)
    assert verdict.kind == "impossible"
    assert verdict.possibilities == {}


def test_validation_errors() -> None:
    templates = np.array([[1.0, 0.0]])
    b = np.array([1.0, 0.0])
    e = np.array([0.0, 0.0])
    with pytest.raises(solver.SolverError):
        solver.classify(templates, (b, b), (e, e), np.array([2.0]), np.array([1.0]))  # lo > hi
    with pytest.raises(solver.SolverError):
        solver.classify(-templates, (b, b), (e, e), np.array([0.5]), np.array([1.0]))  # negative
    with pytest.raises(solver.SolverError):
        solver.classify(templates, (b, b), (np.array([-1.0, 0.0]), e), np.array([0.5]), np.array([1.0]))
    with pytest.raises(solver.SolverError):
        solver.classify(templates, (b, b), (e, e), np.array([0.5]), np.array([1.0]), {(0, 5): True})


@pytest.mark.parametrize("seed", range(6))
def test_classification_matches_bruteforce(seed: int) -> None:
    """Cross-check both MILP hypotheses against exhaustive subset enumeration."""
    rng = np.random.default_rng(seed)
    n, m = 4, 2
    templates, lo, hi = _random_instance(rng, n, m)
    # Build random target energies from a randomly chosen pressed set per take.
    def synth(pressed):
        gains = np.zeros(n)
        for k in pressed:
            gains[k] = rng.uniform(lo[k], hi[k])
        return gains @ templates, gains

    true1 = set(rng.choice(n, size=rng.integers(1, 3), replace=False).tolist())
    true2 = set(rng.choice(n, size=rng.integers(1, 3), replace=False).tolist())
    b1, _ = synth(true1)
    b2, _ = synth(true2)
    e = np.full(m, 1e-9)

    feasible1 = _feasible_set_bruteforce(templates, b1, e, lo, hi)
    feasible2 = _feasible_set_bruteforce(templates, b2, e, lo, hi)
    same_possible = bool(set(feasible1) & set(feasible2))
    diff_possible = any(s1 != s2 for s1 in feasible1 for s2 in feasible2)

    verdict = solver.classify(templates, (b1, b2), (e, e), lo, hi)
    expected = (
        "both" if same_possible and diff_possible
        else "same" if same_possible
        else "different" if diff_possible
        else "impossible"
    )
    assert verdict.kind == expected
    if same_possible:
        assert verdict.possibilities["same"][0].pressed == verdict.possibilities["same"][1].pressed
    if diff_possible:
        assert (
            verdict.possibilities["different"][0].pressed
            != verdict.possibilities["different"][1].pressed
        )
    # Every witness must independently satisfy the band constraints.
    for name, takes in verdict.possibilities.items():
        for take, energy in zip(takes, (b1, b2)):
            gains = np.zeros(n)
            for k in take.pressed:
                gains[k] = take.gains[k]
            produced = gains @ templates
            assert np.allclose(produced, energy, atol=1e-6, rtol=0)


def test_24_keys_within_time_budget() -> None:
    rng = np.random.default_rng(123)
    n, m = 24, 6
    templates, lo, hi = _random_instance(rng, n, m)
    pressed = [2, 9, 17]
    gains = np.zeros(n)
    for k in pressed:
        gains[k] = rng.uniform(lo[k], hi[k])
    b = gains @ templates
    e = np.full(m, 1e-7)
    start = time.monotonic()
    verdict = solver.classify(templates, (b, b), (e, e), lo, hi, time_limit=30.0)
    elapsed = time.monotonic() - start
    assert verdict.kind in {"same", "both"}
    for name, takes in verdict.possibilities.items():
        for take, energy in zip(takes, (b, b)):
            gains = np.zeros(n)
            for k in take.pressed:
                gains[k] = take.gains[k]
            assert np.allclose(gains @ templates, energy, atol=1e-5, rtol=0)
    if verdict.kind == "same":
        assert verdict.possibilities["same"][0].pressed == pressed
    assert elapsed < 30.0


def test_confirmation_conflict_evidence() -> None:
    # Two distinguishable keys so a differing witness exists: energy is 0, so
    # key 0 can only be pressed in a take whose error band admits zero output.
    templates = np.array([[1.0], [1.0]])
    lo = np.array([0.5, 0.5])
    hi = np.array([1.0, 1.0])
    b = np.array([1.0])
    e = np.array([0.0])
    verdict = solver.classify(
        templates, (b, b), (e, e), lo, hi,
        confirmations={(0, 0): True, (1, 0): False},
    )
    assert verdict.kind == "different"
    assert verdict.evidence["models"]["same"]["reason"] == "confirmations_imply_different_press_state"
