export const FUSION_METHODS = [
  "blend_0.3_uf",
  "blend_0.3",
  "blend_0.2",
  "blend_0.5",
  "quality_aware",
  "stacked_loso",
  "video_upperface",
  "video_only",
  "physio_only",
];

export const METHOD_DESCRIPTIONS = {
  "blend_0.3_uf": "Physio-First (70%) + Upper-Face Video (30% Clean) · AUC: 0.933",
  "blend_0.3": "Fixed Late Fusion (30% Physio + 70% Full-Face Video) · AUC: 0.996 ⚠",
  "blend_0.2": "Fixed Late Fusion (20% Physio + 80% Full-Face Video) · AUC: 0.995",
  "blend_0.5": "Equal Weighting (50% Physio + 50% Video) · AUC: 0.995",
  quality_aware: "Dynamic Quality-Aware Weighting (gated by detection fraction) · AUC: 0.995",
  stacked_loso: "Stacked Logistic Regression Meta-Learner (Inner-LOSO) · AUC: 0.996",
  video_upperface: "Unimodal Upper-Face Video (19 Features, Clean) · AUC: 0.968",
  video_only: "Unimodal Full-Face Video (23 Features, Confounded) · AUC: 0.994",
  physio_only: "Unimodal Physiology Baseline (LSTM) · AUC: 0.814",
};

// Streamlit looks up p_video_only and p_physio_only, which are not columns in the CSV.
// Here those two strategies read the unimodal columns instead.
const COLUMN_OVERRIDE = { video_only: "p_video", physio_only: "p_physio" };
export const columnFor = (method) => COLUMN_OVERRIDE[method] ?? `p_${method}`;

const clip = (v) => Math.min(1, Math.max(0, v));
const whole = (v) => `${Math.round(v * 100)}%`;
const one = (v) => `${(v * 100).toFixed(1)}%`;

// w = [physiology weight, video weight], or null when the meta-learner sets the weights.
export function strategyFor(method, detection) {
  switch (method) {
    case "blend_0.3_uf":
      return {
        w: [0.7, 0.3],
        text: "Physiology-First Fixed Blend (70% Physio + 30% Clean Upper-Face Video, Mouth Confound Removed)",
      };
    case "blend_0.3":
      return { w: [0.3, 0.7], text: "Benchmark Fixed Blend (30% Physio + 70% Full-Face Video)" };
    case "blend_0.2":
      return { w: [0.2, 0.8], text: "Benchmark Fixed Blend (20% Physio + 80% Full-Face Video)" };
    case "blend_0.5":
      return { w: [0.5, 0.5], text: "Equal Weighting Blend (50% Physio + 50% Full-Face Video)" };
    case "quality_aware": {
      const wp = clip(0.2 + (1 - detection) * (0.4 - 0.2));
      return {
        w: [wp, 1 - wp],
        text: `Quality-Aware Dynamic Weighting (Physio: ${one(wp)}, Video: ${one(1 - wp)}, Scaled by Face Detection Fraction: ${whole(detection)})`,
      };
    }
    case "stacked_loso":
      return {
        w: null,
        text: "Stacked Logistic Regression Meta-Learner (Inner-LOSO Trained Coefficients)",
      };
    case "video_upperface":
      return { w: [0, 1], text: "Unimodal Video Only (19 Clean Upper-Face Features)" };
    case "video_only":
      return {
        w: [0, 1],
        text: "Unimodal Video Only (23 Full-Face Features, Confounded by Speech)",
      };
    case "physio_only":
      return { w: [1, 0], text: "Unimodal Physiology Only (LSTM Temporal Model)" };
    default:
      return { w: [0.7, 0.3], text: `Fusion Strategy: ${method}` };
  }
}