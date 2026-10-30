"""Temporal trend analysis adapter integrating causal Theil-Sen slope & safeguards.

Adapts the prototype in results/temporal_trend_prototype/trend_prototype.py with
safe hysteresis ensuring stale trends are never presented during insufficient data.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np

# Label Constants
INC, DEC, STABLE, FLUCT, UNCLEAR, INSUFF = range(6)
LABELS = ["increasing", "decreasing", "stable", "fluctuating", "unclear", "insufficient data"]
LAB_IDX = {l: i for i, l in enumerate(LABELS)}

LABEL_COLORS = {
    "increasing": "#FF4B4B",       # Red/Orange - rising distress
    "decreasing": "#00CC96",       # Green - recovering
    "stable": "#1E88E5",           # Blue - steady state
    "fluctuating": "#AB63FA",       # Purple - oscillation
    "unclear": "#FFA15A",          # Amber - indeterminate
    "insufficient data": "#7F7F7F" # Grey - cold start / missing
}

STEP_S = 5
WIN_S = 10
W60 = 60 // STEP_S  # 12 windows per 60 s


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


def theil_sen(t: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Theil-Sen slope (median of pairwise slopes) and median-based intercept."""
    m = len(t)
    ii, jj = np.triu_indices(m, 1)
    s = float(np.median((y[jj] - y[ii]) / (t[jj] - t[ii])))
    b = float(np.median(y - s * t))
    return s, b


@dataclass
class TrendConfig:
    K: int = 12                   # Rolling window length in steps
    min_valid_frac: float = 0.75  # Minimum valid fraction required
    block_len: int = 3            # Bootstrap block length
    n_boot: int = 200             # Bootstrap samples
    ci_level: float = 0.90        # 90% confidence interval on slope
    clip: Tuple[float, float] = (0.001, 0.999)
    delta: float = 0.40           # Slope threshold in logit/60s
    R_high: float = 0.60          # Residual threshold for fluctuation


class TemporalTrendAdapter:
    """Causal temporal trend analysis on windowed model probabilities."""

    def __init__(self, config: Optional[TrendConfig] = None):
        self.cfg = config or TrendConfig()

    def classify_window(self, lo60: float, hi60: float, R: float) -> int:
        """Apply classification rules (design Section 5.5)."""
        d, Rh = self.cfg.delta, self.cfg.R_high
        smooth = (R <= Rh)
        if lo60 > d and smooth:
            return INC
        if hi60 < -d and smooth:
            return DEC
        if lo60 >= -d and hi60 <= d and smooth:
            return STABLE
        if R > Rh and lo60 <= 0.0 <= hi60:
            return FLUCT
        return UNCLEAR

    def evaluate_series(
        self,
        probabilities: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        seed: int = 42
    ) -> Dict[str, np.ndarray]:
        """Causal evaluation across an entire time series of probabilities (past-only)."""
        p = np.asarray(probabilities, dtype=float)
        n = len(p)
        K = self.cfg.K

        if valid_mask is None:
            valid = np.isfinite(p) & (p >= 0.0) & (p <= 1.0)
        else:
            valid = np.asarray(valid_mask, dtype=bool)

        y = np.full(n, np.nan)
        y[valid] = logit(np.clip(p[valid], *self.cfg.clip))

        out = {
            "raw": np.full(n, INSUFF, dtype=int),
            "slope": np.full(n, np.nan),
            "lo": np.full(n, np.nan),
            "hi": np.full(n, np.nan),
            "R": np.full(n, np.nan),
            "n_valid": np.zeros(n, dtype=int),
            "is_sufficient": np.zeros(n, dtype=bool),
        }

        rng = np.random.default_rng(seed)
        L = self.cfg.block_len
        ar = np.arange(L)
        q = (1.0 - self.cfg.ci_level) / 2.0
        k_lo = int(round(q * (self.cfg.n_boot - 1)))
        k_hi = int(round((1.0 - q) * (self.cfg.n_boot - 1)))

        for t in range(n):
            a = max(0, t - K + 1)
            vm = valid[a:t + 1]
            out["n_valid"][t] = int(vm.sum())

            # Cold start or insufficient quality windows
            if t < K - 1 or out["n_valid"][t] < self.cfg.min_valid_frac * K:
                out["is_sufficient"][t] = False
                continue

            out["is_sufficient"][t] = True
            tt = np.arange(a, t + 1, dtype=float)[vm]
            yy = y[a:t + 1][vm]
            m = len(yy)

            s, b = theil_sen(tt, yy)
            fit = s * tt + b
            res = yy - fit
            R = float(np.median(np.abs(np.diff(res)))) if len(res) > 1 else 0.0

            # Bootstrap residuals for slope confidence interval
            ii, jj = np.triu_indices(m, 1)
            dt = tt[jj] - tt[ii]
            nb = int(math.ceil(m / L))
            starts = rng.integers(0, max(1, m - L + 1), size=(self.cfg.n_boot, nb))
            idx = (starts[:, :, None] + ar[None, None, :]).reshape(self.cfg.n_boot, -1)[:, :m]
            Ys = fit[None, :] + res[idx]

            sl = np.median((Ys[:, jj] - Ys[:, ii]) / dt[None, :], axis=1)
            sl = np.partition(sl, [k_lo, k_hi])
            lo60, hi60, s60 = sl[k_lo] * W60, sl[k_hi] * W60, s * W60

            out["slope"][t] = s60
            out["lo"][t] = lo60
            out["hi"][t] = hi60
            out["R"][t] = R
            out["raw"][t] = self.classify_window(lo60, hi60, R)

        # Apply safe anti-flicker hysteresis (transitions to insufficient data are immediate)
        out["displayed"] = self.anti_flicker_safe(out["raw"])
        out["displayed_label"] = [LABELS[idx] for idx in out["displayed"]]
        out["raw_label"] = [LABELS[idx] for idx in out["raw"]]

        return out

    @staticmethod
    def anti_flicker_safe(raw: np.ndarray) -> np.ndarray:
        """Hysteresis filter that immediately reverts to 'insufficient data' when quality drops.

        Ensures a stale trend label is never presented during cold starts or data gaps.
        """
        disp = np.empty_like(raw)
        cur, pending = raw[0], None
        for i, lab in enumerate(raw):
            if lab == INSUFF:
                # Immediate transition on data loss
                cur, pending = lab, None
            elif lab == cur:
                pending = None
            elif pending == lab:
                cur, pending = lab, None
            else:
                pending = lab
            disp[i] = cur
        return disp

