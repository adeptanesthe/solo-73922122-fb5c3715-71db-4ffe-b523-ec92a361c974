"""End-to-end tests against the FastAPI app, exercising the real SciPy solver."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import STORE, app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def base_payload() -> dict:
    # Three orthogonal templates: key C, E, G.  Only C is heard in both takes.
    return {
        "key_names": ["C4", "E4", "G4"],
        "band_names": ["f0", "f1", "f2"],
        "templates": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "band_energy_1": [1.0, 0.0, 0.0],
        "band_error_1": [0.02, 0.02, 0.02],
        "band_energy_2": [1.0, 0.0, 0.0],
        "band_error_2": [0.02, 0.02, 0.02],
        "gain_lower": [0.5, 0.5, 0.5],
        "gain_upper": [2.0, 2.0, 2.0],
        "time_limit": 10.0,
    }


def create(client: TestClient, payload: dict) -> dict:
    resp = client.post("/analyses", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_health(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}


def test_same_kind_with_witnesses(client: TestClient) -> None:
    report = create(client, base_payload())
    assert report["kind"] == "same"
    assert len(report["possibilities"]) == 1
    hyp = report["possibilities"][0]
    assert hyp["hypothesis"] == "same"
    for take in hyp["takes"]:
        assert take["pressed_keys"] == ["C4"]
        assert set(take["gains"]) == {"C4"}
        assert 0.5 <= take["gains"]["C4"] <= 2.0


def test_different_kind(client: TestClient) -> None:
    payload = base_payload()
    # Take 1 hears C, take 2 hears E -> certainly different sets.
    payload["band_energy_1"] = [1.0, 0.0, 0.0]
    payload["band_energy_2"] = [0.0, 1.0, 0.0]
    payload["band_error_1"] = [0.01, 0.01, 0.01]
    payload["band_error_2"] = [0.01, 0.01, 0.01]
    report = create(client, payload)
    assert report["kind"] == "different"
    assert len(report["possibilities"]) == 1
    hyp = report["possibilities"][0]
    assert hyp["hypothesis"] == "different"
    sets = [set(t["pressed_keys"]) for t in hyp["takes"]]
    assert sets[0] != sets[1]


def test_both_possible(client: TestClient) -> None:
    # Two keys with identical templates and a mid-level energy: playing either
    # one, or both at half gain, fits equally.  Same set (both) and different
    # set (one in take 1, the other in take 2) are both feasible.
    payload = {
        "key_names": ["A", "B"],
        "band_names": ["f"],
        "templates": [[1.0], [1.0]],
        "band_energy_1": [1.0],
        "band_error_1": [0.0],
        "band_energy_2": [1.0],
        "band_error_2": [0.0],
        "gain_lower": [0.5, 0.5],
        "gain_upper": [1.0, 1.0],
    }
    report = create(client, payload)
    assert report["kind"] == "both"
    hyps = {p["hypothesis"] for p in report["possibilities"]}
    assert hyps == {"same", "different"}
    for possibility in report["possibilities"]:
        assert len(possibility["takes"]) == 2


def test_impossible_reports_conflict(client: TestClient) -> None:
    payload = base_payload()
    payload["band_energy_1"] = [5.0, 0.0, 0.0]  # cannot reach gain 5 with max 2
    payload["band_energy_2"] = [5.0, 0.0, 0.0]
    payload["band_error_1"] = [0.01] * 3
    payload["band_error_2"] = [0.01] * 3
    report = create(client, payload)
    assert report["kind"] == "impossible"
    assert report["possibilities"] == []
    assert report["evidence"]["reason"] == "conflicting_evidence"
    assert report["evidence"]["same_set_feasible"] is False
    assert report["evidence"]["different_set_feasible"] is False


def ambiguous_payload() -> dict:
    """Two keys share one template, so one unit of energy is ambiguous.

    The solver minimises the number of pressed keys and may report either key;
    an editor confirming the other key is therefore a meaningful, feasible
    correction that must be reflected in the recomputed witness.
    """
    return {
        "key_names": ["A", "B"],
        "band_names": ["f"],
        "templates": [[1.0], [1.0]],
        "band_energy_1": [1.0],
        "band_error_1": [0.0],
        "band_energy_2": [1.0],
        "band_error_2": [0.0],
        "gain_lower": [0.5, 0.5],
        "gain_upper": [1.0, 1.0],
    }


def test_confirm_new_key_and_recompute(client: TestClient) -> None:
    payload = base_payload()
    # Make the evidence support C4 and G4 but not E4; the solver may pick a
    # minimal witness, so confirming any consistent key must recompute cleanly.
    payload["band_energy_1"] = [1.0, 0.0, 0.0]
    payload["band_energy_2"] = [1.0, 0.0, 0.0]
    payload["band_error_1"] = [0.02] * 3
    payload["band_error_2"] = [0.02] * 3
    report = create(client, payload)
    assert report["kind"] == "same"
    # Confirm a key that is feasible but not necessarily in the minimal witness:
    # C4 is already heard; additionally confirm G4 is pressed in both takes
    # would contradict its zero band, so instead use the ambiguous A/B case.
    amb = create(client, ambiguous_payload())
    resp = client.post(
        f"/analyses/{amb['analysis_id']}/confirmations",
        json={
            "expected_version": 1,
            "confirmations": [
                {"segment": 0, "key": "B", "pressed": True},
                {"segment": 1, "key": "B", "pressed": True},
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    updated = resp.json()
    assert updated["version"] == 2
    for seg in ("0", "1"):
        assert updated["confirmations"][seg] == {"B": True}
    for possibility in updated["possibilities"]:
        assert all("B" in take["pressed_keys"] for take in possibility["takes"])


def test_refute_absent_key(client: TestClient) -> None:
    payload = base_payload()
    payload["band_energy_1"] = [0.0, 0.0, 0.0]
    payload["band_energy_2"] = [0.0, 0.0, 0.0]
    payload["band_error_1"] = [0.01] * 3
    payload["band_error_2"] = [0.01] * 3
    report = create(client, payload)
    # Nothing is pressed; confirm C4 not pressed in segment 0 is consistent.
    resp = client.post(
        f"/analyses/{report['analysis_id']}/confirmations",
        json={
            "expected_version": 1,
            "confirmations": [{"segment": 0, "key": "C4", "pressed": False}],
        },
    )
    assert resp.status_code == 200, resp.text


def test_correction_overrides_prior_confirmation(client: TestClient) -> None:
    amb = create(client, ambiguous_payload())
    aid = amb["analysis_id"]
    # Confirm B played in take 0: feasible, and must appear in the witness.
    first = client.post(
        f"/analyses/{aid}/confirmations",
        json={
            "expected_version": 1,
            "confirmations": [{"segment": 0, "key": "B", "pressed": True}],
        },
    )
    assert first.status_code == 200, first.text
    assert first.json()["confirmations"]["0"] == {"B": True}
    # Correct it: B was NOT played in take 0.  A then carries the energy.
    second = client.post(
        f"/analyses/{aid}/confirmations",
        json={
            "expected_version": 2,
            "confirmations": [{"segment": 0, "key": "B", "pressed": False}],
        },
    )
    assert second.status_code == 200, second.text
    body = second.json()
    assert body["version"] == 3
    assert body["confirmations"]["0"] == {"B": False}
    # Every reported witness for take 0 must respect the corrected state.
    for possibility in body["possibilities"]:
        take0 = possibility["takes"][0]
        assert "B" not in take0["pressed_keys"]


def test_stale_version_rejected_and_prior_state_intact(client: TestClient) -> None:
    report = create(client, base_payload())
    aid = report["analysis_id"]
    ok = client.post(
        f"/analyses/{aid}/confirmations",
        json={
            "expected_version": 1,
            "confirmations": [{"segment": 0, "key": "C4", "pressed": True}],
        },
    )
    assert ok.status_code == 200
    stale = client.post(
        f"/analyses/{aid}/confirmations",
        json={
            "expected_version": 1,  # stale: current is 2
            "confirmations": [{"segment": 0, "key": "E4", "pressed": True}],
        },
    )
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "stale_version"
    # Prior confirmation still visible.
    state = client.get(f"/analyses/{aid}/confirmations").json()
    assert state["current_version"] == 2
    assert state["confirmations"]["0"] == {"C4": True}
    # Original report still retrievable by version.
    original = client.get(f"/analyses/{aid}", params={"version": 1}).json()
    assert original["kind"] == "same"


def test_infeasible_confirmation_rolls_back(client: TestClient) -> None:
    payload = base_payload()
    payload["band_energy_1"] = [0.0, 0.0, 0.0]
    payload["band_energy_2"] = [0.0, 0.0, 0.0]
    payload["band_error_1"] = [0.01] * 3
    payload["band_error_2"] = [0.01] * 3
    report = create(client, payload)
    aid = report["analysis_id"]
    # Confirm C4 played in both takes, but the energies are zero -> infeasible.
    resp = client.post(
        f"/analyses/{aid}/confirmations",
        json={
            "expected_version": 1,
            "confirmations": [
                {"segment": 0, "key": "C4", "pressed": True},
                {"segment": 1, "key": "C4", "pressed": True},
            ],
        },
    )
    assert resp.status_code == 409
    err = resp.json()["error"]
    assert err["code"] == "infeasible_after_confirmations"
    assert err["detail"]["current_version"] == 1
    assert err["detail"]["evidence"]["reason"] == "conflicting_evidence"
    # No rollback side effects: still version 1, no confirmations.
    state = client.get(f"/analyses/{aid}/confirmations").json()
    assert state["current_version"] == 1
    assert state["confirmations"] == {}
    assert client.get(f"/analyses/{aid}").json()["version"] == 1


def test_confirmation_across_segments_marks_different(client: TestClient) -> None:
    amb = create(client, ambiguous_payload())
    assert amb["kind"] == "both"
    aid = amb["analysis_id"]
    # Confirm B pressed in segment 0 but not in segment 1.  Same-set becomes
    # infeasible; a different-set witness (take 0 uses B, take 1 uses A) exists.
    resp = client.post(
        f"/analyses/{aid}/confirmations",
        json={
            "expected_version": 1,
            "confirmations": [
                {"segment": 0, "key": "B", "pressed": True},
                {"segment": 1, "key": "B", "pressed": False},
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["kind"] == "different"
    assert body["confirmations"]["0"] == {"B": True}
    assert body["confirmations"]["1"] == {"B": False}
    hyp = body["possibilities"][0]
    assert hyp["hypothesis"] == "different"
    assert "B" in hyp["takes"][0]["pressed_keys"]
    assert "B" not in hyp["takes"][1]["pressed_keys"]


def test_versions_endpoint(client: TestClient) -> None:
    report = create(client, base_payload())
    aid = report["analysis_id"]
    client.post(
        f"/analyses/{aid}/confirmations",
        json={
            "expected_version": 1,
            "confirmations": [{"segment": 0, "key": "C4", "pressed": True}],
        },
    )
    versions = client.get(f"/analyses/{aid}/versions").json()
    assert versions["current_version"] == 2
    assert [v["version"] for v in versions["versions"]] == [1, 2]


def test_unknown_key_rejected(client: TestClient) -> None:
    report = create(client, base_payload())
    resp = client.post(
        f"/analyses/{report['analysis_id']}/confirmations",
        json={
            "expected_version": 1,
            "confirmations": [{"segment": 0, "key": "Z9", "pressed": True}],
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_input"


def test_contradictory_request_rejected(client: TestClient) -> None:
    report = create(client, ambiguous_payload())
    resp = client.post(
        f"/analyses/{report['analysis_id']}/confirmations",
        json={
            "expected_version": 1,
            "confirmations": [
                {"segment": 0, "key": "A", "pressed": True},
                {"segment": 0, "key": "A", "pressed": False},
            ],
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_input"
    # Nothing committed.
    state = client.get(f"/analyses/{report['analysis_id']}/confirmations").json()
    assert state["current_version"] == 1
    assert state["confirmations"] == {}


def test_each_possibility_has_both_takes(client: TestClient) -> None:
    report = create(client, ambiguous_payload())
    assert report["kind"] == "both"
    for possibility in report["possibilities"]:
        takes = possibility["takes"]
        assert [t["segment"] for t in takes] == [0, 1]
        for take in takes:
            assert isinstance(take["pressed_keys"], list)
            assert set(take["gains"]) == set(take["pressed_keys"])
    same_sets = [t["pressed_keys"] for t in report["possibilities"][0]["takes"]]
    assert same_sets[0] == same_sets[1]
    diff_sets = [t["pressed_keys"] for t in report["possibilities"][1]["takes"]]
    assert diff_sets[0] != diff_sets[1]


def test_too_many_keys_rejected(client: TestClient) -> None:
    payload = base_payload()
    payload["key_names"] = [f"k{i}" for i in range(25)]
    payload["templates"] = [[1.0, 0.0, 0.0] for _ in range(25)]
    payload["gain_lower"] = [0.0] * 25
    payload["gain_upper"] = [1.0] * 25
    resp = client.post("/analyses", json=payload)
    assert resp.status_code == 400
    assert "24" in resp.json()["error"]["message"]
