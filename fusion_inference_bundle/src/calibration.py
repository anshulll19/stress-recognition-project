"""
calibration.py
==============

Rest calibration module for physiological signals and extracted features.

Subject-level baseline differences in autonomic physiology (resting heart rate,
pulse transit time, baseline skin conductance levels) are frequently larger
than the within-subject effect size of psychological stress. Calibrating
features relative to an individual's resting baseline (e.g. UBFC-Phys Task 1
or an initial rest session) removes static tonic differences, allowing models
to detect dynamic phasic shifts associated with distress.
"""

from typing import Dict, List, Optional, Union
import json
from pathlib import Path
import numpy as np
import pandas as pd


class RestCalibrator:
    """
    Standardizes physiological features using an individual's resting baseline.

    Parameters
    ----------
    features : Optional[List[str]]
        Explicit ordered list of feature names. If None, inferred from DataFrame columns.
    eps : float
        Numerical epsilon to guard against division by zero for static features.
    """

    def __init__(self, features: Optional[List[str]] = None, eps: float = 1e-6):
        self.features = list(features) if features is not None else None
        self.eps = eps
        self.means: Optional[Dict[str, float]] = None
        self.stds: Optional[Dict[str, float]] = None
        self.n_baseline_samples: int = 0
        self.is_fitted: bool = False

    def fit(self, baseline_data: Union[pd.DataFrame, np.ndarray, Dict[str, Union[float, List[float]]]]) -> "RestCalibrator":
        """
        Compute baseline distribution (mean and std) from resting windows.

        Parameters
        ----------
        baseline_data : pd.DataFrame, np.ndarray, or Dict
            Rest period feature data. Must contain finite numbers.

        Returns
        -------
        self : RestCalibrator
        """
        if isinstance(baseline_data, pd.DataFrame):
            df = baseline_data.copy()
            if self.features is None:
                self.features = df.columns.tolist()
            else:
                missing = [f for f in self.features if f not in df.columns]
                if missing:
                    raise ValueError(f"Baseline DataFrame missing required features: {missing}")
                df = df[self.features]

            if df.empty:
                raise ValueError("Cannot fit RestCalibrator on an empty DataFrame.")
            if not np.isfinite(df.to_numpy(dtype=float)).all():
                raise ValueError("Baseline data contains NaN or infinite values.")

            self.n_baseline_samples = len(df)
            means = df.mean().to_dict()
            stds = df.std().to_dict()

        elif isinstance(baseline_data, dict):
            if self.features is None:
                self.features = list(baseline_data.keys())
            
            means = {}
            stds = {}
            sample_lens = []

            for f in self.features:
                if f not in baseline_data:
                    raise ValueError(f"Baseline dictionary missing required feature '{f}'")
                val = baseline_data[f]
                arr = np.asarray(val, dtype=float)
                if arr.ndim == 0:
                    arr = arr.reshape(1)
                if not np.isfinite(arr).all():
                    raise ValueError(f"Baseline values for feature '{f}' contain non-finite numbers.")
                sample_lens.append(len(arr))
                means[f] = float(np.mean(arr))
                stds[f] = float(np.std(arr)) if len(arr) > 1 else 1.0

            self.n_baseline_samples = int(np.mean(sample_lens)) if sample_lens else 1

        elif isinstance(baseline_data, np.ndarray):
            arr = np.asarray(baseline_data, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            elif arr.ndim != 2:
                raise ValueError(f"Expected 2D numpy array for baseline features, got shape {arr.shape}")

            if not np.isfinite(arr).all():
                raise ValueError("Baseline array contains NaN or infinite values.")

            n_samples, n_feats = arr.shape
            if self.features is None:
                self.features = [f"feat_{i}" for i in range(n_feats)]
            elif len(self.features) != n_feats:
                raise ValueError(
                    f"Feature count mismatch: expected {len(self.features)} features, "
                    f"array has {n_feats} columns."
                )

            self.n_baseline_samples = n_samples
            mu_vec = np.mean(arr, axis=0)
            std_vec = np.std(arr, axis=0) if n_samples > 1 else np.ones(n_feats)

            means = {self.features[i]: float(mu_vec[i]) for i in range(n_feats)}
            stds = {self.features[i]: float(std_vec[i]) for i in range(n_feats)}

        else:
            raise TypeError(f"Unsupported baseline data type: {type(baseline_data)}")

        # Guard against zero or near-zero variance
        self.means = means
        self.stds = {}
        for f, s in stds.items():
            if s is None or np.isnan(s) or s < self.eps:
                self.stds[f] = 1.0
            else:
                self.stds[f] = float(s)

        self.is_fitted = True
        return self

    def transform(self, data: Union[pd.DataFrame, np.ndarray, Dict[str, float]]) -> Union[pd.DataFrame, np.ndarray, Dict[str, float]]:
        """
        Z-score standardize features using the fitted baseline parameters:
            z = (x - mu_rest) / std_rest
        """
        if not self.is_fitted or self.means is None or self.stds is None:
            raise RuntimeError("RestCalibrator must be fitted on resting baseline data before transform.")

        if isinstance(data, pd.DataFrame):
            df = data.copy()
            for f in self.features:
                if f in df.columns:
                    mu = self.means[f]
                    sd = self.stds[f]
                    df[f] = (df[f].astype(float) - mu) / sd
            return df

        elif isinstance(data, dict):
            out = dict(data)
            for f in self.features:
                if f in out:
                    mu = self.means[f]
                    sd = self.stds[f]
                    val = float(out[f])
                    out[f] = (val - mu) / sd
            return out

        elif isinstance(data, np.ndarray):
            arr = np.asarray(data, dtype=float).copy()
            orig_shape = arr.shape
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] != len(self.features):
                raise ValueError(
                    f"Transform array shape mismatch: expected {len(self.features)} columns, "
                    f"got {arr.shape[1]}"
                )
            for i, f in enumerate(self.features):
                mu = self.means[f]
                sd = self.stds[f]
                arr[:, i] = (arr[:, i] - mu) / sd

            if len(orig_shape) == 1:
                return arr.reshape(-1)
            return arr

        else:
            raise TypeError(f"Unsupported data type for transform: {type(data)}")

    def fit_transform(self, baseline_data: Union[pd.DataFrame, np.ndarray, Dict[str, float]]) -> Union[pd.DataFrame, np.ndarray, Dict[str, float]]:
        return self.fit(baseline_data).transform(baseline_data)

    def to_dict(self) -> dict:
        """Serialize calibrator state to dictionary."""
        return {
            "features": self.features,
            "means": self.means,
            "stds": self.stds,
            "n_baseline_samples": self.n_baseline_samples,
            "eps": self.eps,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RestCalibrator":
        """Reconstruct calibrator from dictionary."""
        cal = cls(features=d.get("features"), eps=d.get("eps", 1e-6))
        cal.means = d.get("means")
        cal.stds = d.get("stds")
        cal.n_baseline_samples = d.get("n_baseline_samples", 0)
        cal.is_fitted = d.get("is_fitted", False)
        return cal

    def save(self, path: Union[str, Path]) -> None:
        """Save calibrator state to JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RestCalibrator":
        """Load calibrator state from JSON."""
        d = json.loads(Path(path).read_text())
        return cls.from_dict(d)

    def __repr__(self) -> str:
        status = "Fitted" if self.is_fitted else "Unfitted"
        n_feats = len(self.features) if self.features else 0
        return f"<RestCalibrator: status={status}, features={n_feats}, baseline_samples={self.n_baseline_samples}>"

