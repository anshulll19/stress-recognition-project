
from pathlib import Path
import csv
import json

from fastapi.testclient import TestClient
from backend.physiology_api import app

client = TestClient(app)

BVP_DIR = Path("data/ubfc_test_upload/drive-download-20261111T175556Z-1-001")
EDA_DIR = Path("data/ubfc_test_upload/drive-download-20261111T175919Z-1-001")

PARTICIPANTS = [
    "s1", "s10", "s11", "s12", "s13", "s14",
    "s15", "s16", "s17", "s18", "s19"
]

MODELS = ["v3", "v5"]
OUTPUT_FILE = Path("real_ubfc_api_results.csv")


def upload_files(participant_id, task):
    bvp = BVP_DIR / f"bvp_{participant_id}_{task}.csv"
    eda = EDA_DIR / f"eda_{participant_id}_{task}.csv"

    if not bvp.exists() or not eda.exists():
        raise FileNotFoundError(
            f"Missing files for {participant_id} {task}: "
            f"BVP={bvp.exists()}, EDA={eda.exists()}"
        )

    return {
        "bvp_file": (bvp.name, bvp.read_bytes(), "text/csv"),
        "eda_file": (eda.name, eda.read_bytes(), "text/csv"),
    }


def compact(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)
    return "" if value is None else value


def get_score(summary, result, *keys):
    for source in (summary, result):
        for key in keys:
            if source.get(key) is not None:
                return source[key]
    return ""


def get_warnings(*sources):
    warnings = []

    for source in sources:
        value = source.get("warnings", [])
        if isinstance(value, list):
            warnings.extend(value)
        elif value:
            warnings.append(value)

    return compact(warnings)


def run_model(participant_id, version):
    print(f"\n{'=' * 65}")
    print(f"PARTICIPANT {participant_id.upper()} | MODEL {version.upper()}")
    print("=" * 65)

    common = {
        "participant_id": (None, participant_id),
        "bvp_fs": (None, "64"),
        "eda_fs": (None, "4"),
    }

    base_row = {
        "participant_id": participant_id,
        "model": version,
        "session_status": "",
        "calibration_status": "",
        "calibrated": False,
        "calibration_warnings": "",
    }

    rows = []

    try:
        response = client.post(
            "/api/physiology/sessions",
            json={
                "participant_id": participant_id,
                "model_version": version,
            },
        )
        response.raise_for_status()

        session_id = response.json()["session_id"]
        print("Session created:", session_id)

        # Upload T1 baseline
        response = client.post(
            f"/api/physiology/sessions/{session_id}/baseline",
            files={**upload_files(participant_id, "T1"), **common},
        )
        response.raise_for_status()
        print("T1 uploaded:", response.status_code)

        # Calibrate
        response = client.post(
            f"/api/physiology/sessions/{session_id}/baseline/calibrate"
        )
        response.raise_for_status()

        calibration = response.json()
        calibrated = calibration.get("is_calibrated", False)

        base_row.update({
            "session_status": "created",
            "calibration_status": calibration.get("status", ""),
            "calibrated": calibrated,
            "calibration_warnings": get_warnings(calibration),
        })

        print(
            "Calibration:",
            calibration.get("status"),
            "| calibrated:",
            calibrated,
        )

        if not calibrated:
            print("Calibration rejected. Skipping T2/T3.")
            for task in ("T2", "T3"):
                rows.append({
                    **base_row,
                    "task": task,
                    "inference_status": "SKIPPED_CALIBRATION_FAILED",
                })
            return rows

        # T2 and T3 inference
        for task in ("T2", "T3"):
            row = {
                **base_row,
                "task": task,
                "inference_status": "",
                "abstained": "",
                "total_windows": "",
                "distress_windows": "",
                "non_stress_windows": "",
                "stress_percentage": "",
                "score_type": "",
                "mean_score": "",
                "min_score": "",
                "max_score": "",
                "warnings": "",
                "error": "",
            }

            try:
                response = client.post(
                    f"/api/physiology/sessions/{session_id}/tasks/{task}",
                    files={
                        **upload_files(participant_id, task),
                        **common,
                    },
                )

                if response.status_code >= 400:
                    row["inference_status"] = f"HTTP_{response.status_code}"
                    row["error"] = response.text[:1000]
                    rows.append(row)
                    print(task, "FAILED:", response.status_code)
                    continue

                result = response.json()
                summary = result.get("summary", {})

                row.update({
                    "inference_status": result.get("status", "completed"),
                    "abstained": result.get("abstained", ""),
                    "total_windows": summary.get("total_windows", ""),
                    "distress_windows": summary.get("stress_windows", ""),
                    "non_stress_windows": summary.get("non_stress_windows", ""),
                    "stress_percentage": summary.get("stress_percentage", ""),
                    "score_type": summary.get("score_type", ""),
                    "mean_score": get_score(
                        summary, result, "mean_score", "mean_probability",
                        "mean_decision_score"
                    ),
                    "min_score": get_score(
                        summary, result, "min_score", "min_probability",
                        "min_decision_score"
                    ),
                    "max_score": get_score(
                        summary, result, "max_score", "max_probability",
                        "max_decision_score"
                    ),
                    "warnings": get_warnings(result, summary),
                })

                print(
                    f"{task}: {row['inference_status']} | "
                    f"windows={row['total_windows']} | "
                    f"distress={row['distress_windows']} | "
                    f"stress%={row['stress_percentage']}"
                )

            except Exception as exc:
                row["inference_status"] = "EXCEPTION"
                row["error"] = repr(exc)
                print(task, "ERROR:", repr(exc))

            rows.append(row)

        # Final session summary
        response = client.get(
            f"/api/physiology/sessions/{session_id}/summary"
        )

        if response.status_code == 200:
            final_summary = response.json()
            print(
                "Final summary:",
                final_summary.get("status", "available"),
                "| reactivity:",
                final_summary.get("reactivity_class", "not reported"),
            )
        else:
            print("Final summary failed:", response.status_code)

    except Exception as exc:
        print("MODEL TEST FAILED:", repr(exc))

        for task in ("T2", "T3"):
            rows.append({
                **base_row,
                "task": task,
                "inference_status": "SESSION_OR_CALIBRATION_FAILED",
                "error": repr(exc),
            })

    return rows


def main():
    all_rows = []

    for participant_id in PARTICIPANTS:
        for version in MODELS:
            all_rows.extend(run_model(participant_id, version))

    if not all_rows:
        print("No results generated.")
        return

    fieldnames = list(dict.fromkeys(
        key for row in all_rows for key in row.keys()
    ))

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n" + "=" * 65)
    print("REAL UBFC-PHYS API TEST FINISHED")
    print("Participants:", len(PARTICIPANTS))
    print("Models:", len(MODELS))
    print("Task result rows:", len(all_rows))
    print("CSV saved to:", OUTPUT_FILE)
    print("=" * 65)


if __name__ == "__main__":
    main()
