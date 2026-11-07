// All fixed text for the Trends & Research tab.
// Changes from the Streamlit version:
//  - badge no longer says "Validated" (the page's own caveats and table contradict it)
//  - "Prediction" heading changed to "Estimation" to match the subtitle
//  - "Rigorous" and "Independent" dropped from the benchmark heading
export const TRENDS_COPY = {
  title: "Temporal Distress Trend Detection & Research Evaluation",
  subtitle:
    "Causal, past-only Theil-Sen trend estimation and Leave-One-Subject-Out (LOSO) benchmarks",
  badge: "Research benchmarks · prototype, not clinically validated",

  trendTitle: "Temporal Distress Trend Estimation",
  trendSubtitle:
    "Causal, past-only trend estimation using the Theil-Sen slope on model distress probabilities.",

  caveatTitle: "Prototype Status & Research Caveats",
  caveat:
    "This temporal trend detector is an active research prototype. While all mechanical unit checks pass (causality, no future imputation, cold-start handling, and safe anti-flicker hysteresis), several synthetic benchmark acceptance criteria failed during rigorous evaluation (including gradual-ramp sensitivity, fluctuating signal recognition, and abrupt step recovery). The detector is displayed here for experimental inspection with strict safeguards.",

  // Shown next to the trend label so "Stable" is not read as an all-clear.
  stableNote: "Prototype: may miss gradual ramps.",

  sourceLabel: "Trend evaluation source",
  sources: [
    { id: "real", label: "Real UBFC-Phys model trajectory (56 participants)" },
    { id: "synthetic", label: "Synthetic verification case (planted pattern)" },
  ],

  benchmarkTitle: "Research Benchmarking",
  benchmarkSubtitle: "Leave-One-Subject-Out (LOSO) results across datasets and models.",
};