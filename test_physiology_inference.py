"""
test_physiology_inference.py
============================

Comprehensive unit and integration test suite for:
- src/calibration.py (RestCalibrator)
- src/physiology_inference.py (WESADInferenceEngine, UBFCInferenceEngine, predict_stress)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path("src").resolve()))

from calibration import RestCalibrator
from physiology_inference import (
    WESADInferenceEngine,
    UBFCInferenceEngine,
    predict_stress,
    PredictionResult,
    UBFC_FEATURES,
)


def test_wesad_window_inference():
    print("--> Testing WESAD window inference...")
    engine = WESADInferenceEngine()
    data = np.load("data/wesad/wesad_binary.npz")
    windows = data["windows"]
    labels = data["labels"]

    # 1. Class 0 window (non-stress)
    idx0 = int(np.where(labels == 0)[0][0])
    w0 = windows[idx0]
    res0 = engine.predict_window(w0)
    assert isinstance(res0, PredictionResult)
    assert res0.modality == "WESAD_BVP"
    assert res0.prediction == 0
    assert res0.probability < 0.5
    assert res0.quality_flag == "OK"
    assert res0.latency_ms > 0

    # 2. Class 1 window (stress)
    idx1 = int(np.where(labels == 1)[0][0])
    w1 = windows[idx1]
    res1 = engine.predict_window(w1)
    assert res1.prediction == 1
    assert res1.probability >= 0.5
    assert res1.quality_flag == "OK"

    # 3. Shape handling (3840,), (3840, 1), (1, 3840, 1)
    res_1d = engine.predict_window(w0.flatten())
    assert np.isclose(res_1d.probability, res0.probability, atol=1e-4)

    print("    [PASS] WESAD window inference and shape handling")


def test_wesad_quality_and_errors():
    print("--> Testing WESAD sensor dropout and error handling...")
    engine = WESADInferenceEngine()

    # Flatline / dropout (std == 0)
    flat_w = np.full((3840, 1), 42.0)
    res_flat = engine.predict_window(flat_w)
    assert res_flat.quality_flag == "DROPOUT"
    assert res_flat.probability == 0.0

    # Non-finite values
    nan_w = np.ones((3840, 1))
    nan_w[100] = np.nan
    try:
        engine.predict_window(nan_w)
        assert False, "Failed to raise ValueError on NaN window"
    except ValueError:
        pass

    # Invalid length
    bad_w = np.ones((1000, 1))
    try:
        engine.predict_window(bad_w)
        assert False, "Failed to raise ValueError on bad shape"
    except ValueError:
        pass

    print("    [PASS] WESAD quality and error handling")


def test_rest_calibrator():
    print("--> Testing RestCalibrator...")
    # Synthetic rest baseline
    feats = ["f1", "f2", "f3"]
    df_rest = pd.DataFrame({
        "f1": [10.0, 12.0, 14.0],
        "f2": [100.0, 100.0, 100.0],  # zero variance test
        "f3": [5.0, 7.0, 9.0],
    })

    cal = RestCalibrator(features=feats)
    cal.fit(df_rest)
    assert cal.is_fitted
    assert cal.means["f1"] == 12.0
    assert cal.stds["f1"] == 2.0
    assert cal.means["f2"] == 100.0
    assert cal.stds["f2"] == 1.0  # protected against zero std

    # Transform DataFrame
    df_test = pd.DataFrame({
        "f1": [16.0],
        "f2": [105.0],
        "f3": [11.0],
    })
    transformed = cal.transform(df_test)
    assert np.isclose(transformed["f1"].iloc[0], (16.0 - 12.0) / 2.0)  # +2.0 std
    assert np.isclose(transformed["f2"].iloc[0], (105.0 - 100.0) / 1.0)  # +5.0

    # Serialization test
    tmp_path = Path("/tmp/test_calibrator.json")
    cal.save(tmp_path)
    loaded_cal = RestCalibrator.load(tmp_path)
    assert loaded_cal.is_fitted
    assert loaded_cal.means == cal.means
    assert loaded_cal.stds == cal.stds

    print("    [PASS] RestCalibrator fit, transform, zero-std guard, serialization")


def test_ubfc_inference():
    print("--> Testing UBFC-Phys inference (calibrated & uncalibrated)...")
    df = pd.read_csv("results/physiology/physiology_features_labeled.csv")

    # Fit a calibrator on participant 1's Task 1 (rest) windows
    p1_t1 = df[(df["participant"] == 1) & (df["task"] == 1)]
    cal = RestCalibrator(features=UBFC_FEATURES).fit(p1_t1)

    engine = UBFCInferenceEngine(calibrator=cal)

    # 1. Test uncalibrated prediction
    test_window_rest = df[(df["participant"] == 1) & (df["task"] == 1)].iloc[0]
    res_uncal = engine.predict_features(test_window_rest, use_calibration=False)
    assert res_uncal.modality == "UBFC_PHYS"
    assert not res_uncal.is_calibrated

    # 2. Test calibrated prediction on rest window (should be Class 0)
    res_cal_rest = engine.predict_features(test_window_rest, use_calibration=True)
    assert res_cal_rest.is_calibrated
    assert res_cal_rest.prediction == 0

    # 3. Test calibrated prediction on stress window (Task 2: speech stress)
    test_window_stress = df[(df["participant"] == 1) & (df["task"] == 2)].iloc[0]
    res_cal_stress = engine.predict_features(test_window_stress, use_calibration=True)
    assert res_cal_stress.is_calibrated
    assert res_cal_stress.prediction == 1
    assert res_cal_stress.probability > 0.5

    # 4. Test raw window feature extraction
    raw = np.load("results/physiology/raw_signal_windows.npz")
    bvp_snippet = raw["X_bvp"][0, :, 0]
    eda_snippet = raw["X_eda"][0, :, 0]
    feats = UBFCInferenceEngine.extract_features(bvp_snippet, eda_snippet)
    assert all(k in feats for k in UBFC_FEATURES)

    print("    [PASS] UBFC-Phys inference with rest calibration and feature extraction")


def test_unified_dispatcher():
    print("--> Testing unified predict_stress dispatcher...")
    wesad = np.load("data/wesad/wesad_binary.npz")
    w0 = wesad["windows"][0]

    # WESAD dispatch
    res_w = predict_stress("WESAD", w0)
    assert res_w.modality == "WESAD_BVP"

    # UBFC dispatch
    df = pd.read_csv("results/physiology/physiology_features_labeled.csv")
    row0 = df.iloc[0]
    res_u = predict_stress("UBFC", row0)
    assert res_u.modality == "UBFC_PHYS"

    # CLAS rejection
    try:
        predict_stress("CLAS", w0)
        assert False, "Failed to block CLAS prediction"
    except ValueError as e:
        assert "BLOCKED" in str(e)

    print("    [PASS] Unified dispatcher and CLAS blocker")


if __name__ == "__main__":
    test_wesad_window_inference()
    test_wesad_quality_and_errors()
    test_rest_calibrator()
    test_ubfc_inference()
    test_unified_dispatcher()
    print("\nALL 5 INTEGRATION TESTS PASSED SUCCESSFULLY!")

