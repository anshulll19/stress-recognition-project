
from pathlib import Path
from io import BytesIO
import math
import numpy as np

from fastapi.testclient import TestClient
from backend.physiology_api import app

client = TestClient(app)

BVP_DIR = Path("data/ubfc_test_upload/drive-download-20261111T175556Z-1-001")
EDA_DIR = Path("data/ubfc_test_upload/drive-download-20261111T175919Z-1-001")

PARTICIPANTS = ["s1", "s10", "s11", "s12", "s13",
                "s14", "s15", "s16", "s17", "s18", "s19"]

MODELS = ["v3", "v5"]


def csv_bytes(signal):
    buffer = BytesIO()
    np.savetxt(buffer, signal, delimiter=",")
    return buffer.getvalue()


def rest_control(pid, model):
    print(f"\n{'=' * 55}")
    print(f"REST CONTROL | {pid} | {model.upper()}")
    print("=" * 55)

    bvp_path = BVP_DIR / f"bvp_{pid}_T1.csv"
    eda_path = EDA_DIR / f"eda_{pid}_T1.csv"

    bvp = np.loadtxt(bvp_path)
    eda = np.loadtxt(eda_path)

    assert len(bvp) == 180 * 64, f"Unexpected BVP length: {len(bvp)}"
    assert len(eda) == 180 * 4, f"Unexpected EDA length: {len(eda)}"

    # Create session
    response = client.post(
        "/api/physiology/sessions",
        json={
            "participant_id": pid,
            "model_version": model,
        },
    )
    response.raise_for_status()

    sid = response.json()["session_id"]
    base = f"/api/physiology/sessions/{sid}"

    common = {
        "participant_id": (None, pid),
        "bvp_fs": (None, "64"),
        "eda_fs": (None, "4"),
    }

    try:
        # Upload complete T1 recording for calibration
        response = client.post(
            f"{base}/baseline",
            files={
                "bvp_file": (
                    bvp_path.name,
                    bvp_path.read_bytes(),
                    "text/csv",
                ),
                "eda_file": (
                    eda_path.name,
                    eda_path.read_bytes(),
                    "text/csv",
                ),
                **common,
            },
        )
        response.raise_for_status()

        # Calibrate
        response = client.post(f"{base}/baseline/calibrate")
        response.raise_for_status()

        calibration = response.json()

        if not calibration.get("is_calibrated"):
            print("Calibration rejected:", calibration)
            return None

        calibration_duration = calibration["calibration_duration_s"]

        # Start rest-control segment after calibration.
        cut = max(90, math.ceil(calibration_duration) + 5)

        if cut >= 150:
            raise ValueError(
                f"Calibration duration too long for rest control: "
                f"{calibration_duration}"
            )

        print("Calibration duration:", calibration_duration)
        print("Rest-control starts at:", cut, "seconds")

        rest_bvp = bvp[cut * 64:]
        rest_eda = eda[cut * 4:]

        # Upload remaining T1 resting signal as a T2 test input.
        response = client.post(
            f"{base}/tasks/T2",
            files={
                "bvp_file": (
                    f"bvp_{pid}_rest_control.csv",
                    csv_bytes(rest_bvp),
                    "text/csv",
                ),
                "eda_file": (
                    f"eda_{pid}_rest_control.csv",
                    csv_bytes(rest_eda),
                    "text/csv",
                ),
                **common,
            },
        )
        response.raise_for_status()

        result = response.json()
        summary = result.get("summary", {})

        output = {
            "participant": pid,
            "model": model,
            "calibration_duration": calibration_duration,
            "rest_duration": len(rest_bvp) / 64,
            "status": result.get("status"),
            "abstained": result.get("abstained"),
            "total_windows": summary.get("total_windows"),
            "distress_windows": summary.get("stress_windows"),
            "non_stress_windows": summary.get("non_stress_windows"),
            "distress_percentage": summary.get("stress_percentage"),
            "mean_score": summary.get("mean_decision_score"),
            "score_type": summary.get("score_type"),
        }

        print("RESULT:", output)
        return output

    finally:
        client.delete(base)


if __name__ == "__main__":
    results = []

    for pid in PARTICIPANTS:
        for model in MODELS:
            try:
                result = rest_control(pid, model)
                if result:
                    results.append(result)
            except Exception as exc:
                print(f"FAILED: {pid} | {model} | {exc}")

    import pandas as pd

    df = pd.DataFrame(results)
    df.to_csv("rest_control_results.csv", index=False)

    print("\nREST CONTROL TEST FINISHED")
    print(df.to_string(index=False))
    print("\nSaved: rest_control_results.csv")
