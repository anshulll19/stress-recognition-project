"""V3 BVP+EDA inference adapter. Keeps V5 inference untouched."""

import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models/ubfc_deployment/ubfc_phys_bvp_eda_rf_v3.joblib"

BVP_PATH = ROOT / "physiological_pipeline/03_NeuroKit_V2/neurokit_bvp_v2.py"
EDA_PATH = ROOT / "physiological_pipeline/03_NeuroKit_V2/neurokit_eda_v2.py"

BVP_FS = 64.0
EDA_FS = 4.0
WINDOW_SECONDS = 30
STEP_SECONDS = 5
CALIBRATION_END_SECONDS = 85.0

BVP_FEATURES = [
    "bvp_mean", "bvp_std", "bvp_median", "bvp_iqr",
    "bvp_slope_per_sample", "bvp_range", "bvp_peak_count",
    "bvp_heart_rate_mean", "bvp_heart_rate_std",
    "bvp_ibi_mean", "bvp_ibi_std", "bvp_peak_quality",
]

EDA_FEATURES = [
    "eda_mean", "eda_std", "eda_median", "eda_iqr",
    "eda_range", "eda_tonic_mean", "eda_tonic_std",
    "eda_tonic_slope", "eda_phasic_mean", "eda_phasic_std",
]

FEATURES = BVP_FEATURES + EDA_FEATURES


class V3InferenceError(ValueError):
    pass


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise V3InferenceError(f"Cannot load feature extractor: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BVPEDARFV3Inference:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise V3InferenceError(f"V3 model artifact not found: {self.model_path}")

        self.artifact = joblib.load(self.model_path)
        self.model = self.artifact["model"]

        if self.artifact.get("features") != FEATURES:
            raise V3InferenceError("V3 artifact feature order does not match adapter.")

        if self.artifact.get("normalization") != "participant T1 baseline mean/std":
            raise V3InferenceError("Unexpected V3 artifact normalization contract.")

        self.bvp_module = _load_module("neurokit_bvp_v2_inference", BVP_PATH)
        self.eda_module = _load_module("neurokit_eda_v2_inference", EDA_PATH)

        self.means = None
        self.scales = None
        self.calibrated = False

    @staticmethod
    def _validate_sampling_rates(bvp_fs, eda_fs):
        if not np.isclose(float(bvp_fs), BVP_FS):
            raise V3InferenceError(f"V3 requires BVP at {BVP_FS:g} Hz.")
        if not np.isclose(float(eda_fs), EDA_FS):
            raise V3InferenceError(f"V3 requires EDA at {EDA_FS:g} Hz.")

    def _extract_features(self, bvp, eda, bvp_fs, eda_fs):
        self._validate_sampling_rates(bvp_fs, eda_fs)

        bvp = np.asarray(bvp, dtype=float).reshape(-1)
        eda = np.asarray(eda, dtype=float).reshape(-1)

        if not np.isfinite(bvp).all() or not np.isfinite(eda).all():
            raise V3InferenceError("Signals contain non-finite samples.")

        bvp_duration = len(bvp) / BVP_FS
        eda_duration = len(eda) / EDA_FS
        if abs(bvp_duration - eda_duration) > 1.0:
            raise V3InferenceError("BVP and EDA durations differ by more than 1 second.")

        common_duration = min(bvp_duration, eda_duration)
        bvp = bvp[:int(common_duration * BVP_FS)]
        eda = eda[:int(common_duration * EDA_FS)]

        if common_duration < WINDOW_SECONDS:
            raise V3InferenceError("Recording is shorter than one 30-second window.")

        bvp_rows = self.bvp_module.extract_bvp_features(bvp, sampling_rate=int(BVP_FS))
        eda_rows = self.eda_module.extract_eda_features(eda, sampling_rate=int(EDA_FS))

        if len(bvp_rows) != len(eda_rows):
            raise V3InferenceError(
                f"BVP/EDA window mismatch: BVP={len(bvp_rows)}, EDA={len(eda_rows)}. "
                "No windows were paired."
            )

        if not bvp_rows:
            raise V3InferenceError("No valid synchronized feature windows were extracted.")

        frame = pd.DataFrame([
            {**bvp_row, **eda_row}
            for bvp_row, eda_row in zip(bvp_rows, eda_rows)
        ])

        missing = set(FEATURES) - set(frame.columns)
        if missing:
            raise V3InferenceError(f"Extractor omitted required features: {sorted(missing)}")

        frame = frame[FEATURES].astype(float)

        # Missing BVP HR/IBI features are expected and handled by the
        # same median imputer stored in the trained model pipeline.
        non_pulse = [
            feature for feature in FEATURES
            if feature not in {
                "bvp_heart_rate_mean", "bvp_heart_rate_std",
                "bvp_ibi_mean", "bvp_ibi_std",
            }
        ]
        if not np.isfinite(frame[non_pulse].to_numpy()).all():
            raise V3InferenceError("Non-pulse feature extraction produced non-finite values.")

        starts = np.arange(len(frame), dtype=float) * STEP_SECONDS
        frame.insert(0, "end_s", starts + WINDOW_SECONDS)
        frame.insert(0, "start_s", starts)
        return frame

    def calibrate(self, bvp, eda, bvp_fs=BVP_FS, eda_fs=EDA_FS):
        self.calibrated = False
        self.means = None
        self.scales = None

        frame = self._extract_features(bvp, eda, bvp_fs, eda_fs)
        baseline = frame[frame["end_s"] <= CALIBRATION_END_SECONDS]

        if len(baseline) < 2:
            raise V3InferenceError("Insufficient T1 calibration windows ending within 85 seconds.")

        means = baseline[FEATURES].mean()
        scales = baseline[FEATURES].std().fillna(0)
        scales = scales.mask(scales < 1e-8, 1.0)

        if not np.isfinite(means.to_numpy()).all():
            raise V3InferenceError("T1 calibration has features with no valid baseline values.")
        if not np.isfinite(scales.to_numpy()).all():
            raise V3InferenceError("T1 calibration produced invalid normalization scales.")

        self.means = means
        self.scales = scales
        self.calibrated = True

        return {
            "status": "accepted",
            "abstained": False,
            "calibration_windows": int(len(baseline)),
            "calibration_duration_s": CALIBRATION_END_SECONDS,
            "feature_count": len(FEATURES),
            "normalization": "participant T1 baseline mean/std",
            "missing_pulse_feature_values": int(
                baseline[[
                    "bvp_heart_rate_mean", "bvp_heart_rate_std",
                    "bvp_ibi_mean", "bvp_ibi_std",
                ]].isna().sum().sum()
            ),
        }

    def predict_task(self, bvp, eda, bvp_fs=BVP_FS, eda_fs=EDA_FS):
        if not self.calibrated or self.means is None or self.scales is None:
            raise V3InferenceError("V3 model has not been calibrated with T1.")

        frame = self._extract_features(bvp, eda, bvp_fs, eda_fs)
        X = (frame[FEATURES] - self.means) / self.scales

        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        classes = list(self.model.named_steps["classifier"].classes_)
        distress_index = classes.index(1)

        windows = []
        for i, prediction in enumerate(predictions):
            windows.append({
                "window_index": i,
                "start_s": float(frame.iloc[i]["start_s"]),
                "end_s": float(frame.iloc[i]["end_s"]),
                "prediction": int(prediction),
                "class_name": "distress" if int(prediction) == 1 else "no_distress",
                "distress_score": float(probabilities[i, distress_index]),
                "score_type": "uncalibrated_random_forest_probability",
            })

        distress_count = sum(item["prediction"] == 1 for item in windows)

        return {
            "status": "accepted",
            "abstained": False,
            "predictions": windows,
            "diagnostics": {
                "model": "BVP_EDA_RandomForest_V3",
                "feature_count": len(FEATURES),
                "window_count": len(windows),
                "distress_windows": distress_count,
                "no_distress_windows": len(windows) - distress_count,
                "distress_percentage": 100.0 * distress_count / len(windows),
                "probability_calibrated": False,
                "normalization": "participant T1 baseline mean/std",
            },
        }
