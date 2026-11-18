import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.physiology_api import app

client = TestClient(app)
PID = "smoke01"


def bvp(seconds, fs=64):
    t = np.arange(int(seconds * fs)) / fs
    rng = np.random.default_rng(0)
    return (np.sin(2 * np.pi * 1.2 * t) + 0.05 * rng.standard_normal(t.size)).tolist()


def eda(seconds, fs=4, drift=0.0):
    t = np.arange(int(seconds * fs)) / fs
    rng = np.random.default_rng(1)
    return (2.0 + drift * t + 0.02 * rng.standard_normal(t.size)).tolist()


def payload(seconds, **extra):
    return {
        "participant_id": PID,
        "bvp_samples": bvp(seconds),
        "eda_samples": eda(seconds),
        **extra
    }


def new_session(model):
    r = client.post(
        "/api/physiology/sessions",
        json={"participant_id": PID, "model_version": model}
    )
    assert r.status_code == 201, r.text
    return r.json()["session_id"]


def test_health():
    r = client.get("/api/physiology/health")
    assert r.status_code == 200
    assert r.json()["status"] in ("healthy", "degraded")


def test_unknown_session_404():
    assert client.get("/api/physiology/sessions/nope/status").status_code == 404


@pytest.mark.parametrize("model", ["v3", "v5"])
def test_full_flow(model):
    sid = new_session(model)
    base = f"/api/physiology/sessions/{sid}"

    r = client.post(f"{base}/tasks/T2", json=payload(60, task_id="T2"))
    assert r.status_code == 400

    r = client.post(f"{base}/baseline", json=payload(30, task_id="T1"))
    assert r.status_code == 400

    r = client.post(f"{base}/baseline", json=payload(180, task_id="T1"))
    assert r.status_code == 200, r.text

    bad = payload(180, task_id="T1")
    bad["participant_id"] = "someone_else"
    assert client.post(f"{base}/baseline", json=bad).status_code == 400

    assert client.post(
        f"{base}/baseline",
        json=payload(180, task_id="T1")
    ).status_code == 200

    cal = client.post(f"{base}/baseline/calibrate")
    assert cal.status_code == 200, cal.text

    if not cal.json()["is_calibrated"]:
        pytest.skip("synthetic baseline abstained; inference not exercised")

    r = client.post(
        f"{base}/tasks/T2",
        json=payload(60, task_id="T2", bvp_fs=32.0, eda_fs=4.0)
    )
    assert r.status_code == 422

    assert client.post(
        f"{base}/tasks/T2",
        json=payload(10, task_id="T2")
    ).status_code == 400

    for task in ("T2", "T3"):
        r = client.post(
            f"{base}/tasks/{task}",
            json=payload(60, task_id=task, bvp_fs=64.0, eda_fs=4.0)
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["summary"]["probability_calibrated"] is False
        assert body["summary"]["mean_stress_probability"] is None

    s = client.get(f"{base}/summary").json()
    assert set(s["tasks_completed"]) == {"T2", "T3"}
    assert s["reactivity_profile"]["reactivity_class"] == "NOT_COMPUTED"

    assert client.delete(base).status_code == 200
    assert client.delete(base).status_code == 404


def test_reupload_baseline_invalidates_calibration():
    sid = new_session("v5")
    base = f"/api/physiology/sessions/{sid}"

    client.post(f"{base}/baseline", json=payload(180, task_id="T1"))
    cal = client.post(f"{base}/baseline/calibrate").json()

    if not cal["is_calibrated"]:
        pytest.skip("synthetic baseline abstained")

    client.post(f"{base}/baseline", json=payload(180, task_id="T1"))
    assert client.get(f"{base}/status").json()["is_calibrated"] is False
