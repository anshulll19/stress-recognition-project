// Mirrors demo/adapters/trend_adapter.py (TrendConfig defaults and LABELS order).
// Index order matches the Python LABELS list, so exported label indexes map directly.
export const TREND_RULES = {
  windowSteps: 12,      // K: rolling window length
  stepSeconds: 5,       // stride between windows
  minValidWindows: 9,   // 0.75 x 12 valid windows needed in the last 12
  ciLevel: 0.9,         // confidence level on the slope
  slopeThreshold: 0.4,  // logit per 60 s
  roughnessThreshold: 0.6,
};

export const TREND_LABELS = [
  { id: "increasing", label: "Increasing", color: "#d63b3b",
    rule: "90% CI of the slope is entirely above +0.40 logit/60s and the fit is smooth." },
  { id: "decreasing", label: "Decreasing", color: "#1fae5b",
    rule: "90% CI of the slope is entirely below −0.40 logit/60s and the fit is smooth." },
  { id: "stable", label: "Stable", color: "#6d57d8",
    rule: "90% CI sits within ±0.40 logit/60s and the fit is smooth. Slow drifts below this threshold can still read as stable." },
  { id: "fluctuating", label: "Fluctuating", color: "#b07cf0",
    rule: "Residual roughness is above 0.60 and the CI includes zero." },
  { id: "unclear", label: "Unclear", color: "#c98206",
    rule: "None of the other rules match." },
  { id: "insufficient data", label: "Insufficient data", color: "#6b6b7b",
    rule: "Fewer than 12 windows so far, or fewer than 9 valid windows in the last 12." },
];