// Rows from the "Rigorous Research Benchmarking" table.
// Numbers are percentages except auroc (0 to 1). null means not reported.
// status: complete | dev_only | confound | clean | failed
export const STATUS = {
  complete: { label: "Complete", tone: "good" },
  clean: { label: "Clean", tone: "good" },
  dev_only: { label: "Dev only", tone: "warn" },
  confound: { label: "Confound", tone: "warn" },
  failed: { label: "Failed", tone: "high" },
};

export const BENCHMARKS = [
  { model: "WESAD CNN-TCN-LSTM", protocol: "Corrected LOSO (15 subj)", accuracy: 89.9, balancedAcc: null, auroc: 0.942, status: "complete" },
  { model: "WESAD Dev Checkpoint", protocol: "Dev Split (S10, S4, S6)", accuracy: 92.6, balancedAcc: 90.1, auroc: 0.961, status: "dev_only", devSplit: true },
  { model: "UBFC-Phys LSTM", protocol: "Full LOSO (56 subj)", accuracy: 80.4, balancedAcc: 76.1, auroc: 0.799, status: "complete" },
  { model: "UBFC-Phys TCN", protocol: "Full LOSO (56 subj)", accuracy: 77.2, balancedAcc: 69.3, auroc: 0.796, status: "complete" },
  { model: "UBFC Rest-Calib RF", protocol: "Full LOSO (56 subj)", accuracy: 97.9, balancedAcc: 98.2, auroc: 0.994, status: "complete" },
  { model: "Video RF (Full 23-feat)", protocol: "Subject LOSO (18 subj)", accuracy: 96.6, balancedAcc: 96.7, auroc: 0.994, status: "confound", note: "Speech-motion confound" },
  { model: "Video RF (Upper 19-feat)", protocol: "Subject LOSO (18 subj)", accuracy: 91.4, balancedAcc: 90.6, auroc: 0.968, status: "clean" },
  { model: "Fusion Blend 0.3 (Full)", protocol: "Subject LOSO (18 subj)", accuracy: null, balancedAcc: 97.1, auroc: 0.996, status: "complete" },
  { model: "Fusion Blend 0.3 (Upper-face)", protocol: "Subject LOSO (18 subj)", accuracy: null, balancedAcc: 81.0, auroc: 0.933, status: "complete" },
  { model: "CLAS → WESAD (cross-dataset)", protocol: "Zero-shot transfer", accuracy: 23.9, balancedAcc: null, auroc: 0.223, status: "failed", note: "Below chance" },
  // TODO: one more row is hidden by the scroll box in the Streamlit screenshot
];

export const BENCHMARK_FOOTNOTE =
  "* Development split of three subjects only. Not comparable with the LOSO results.";