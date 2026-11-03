export const OVERVIEW = {
  title: "Multimodal Stress Monitoring Overview",
  subtitle: "Offline benchmark evaluation and synchronized multimodal replay overview",

  highlights: [
    { value: "98.2%", label: "UBFC Rest-Calib RF" },
    { value: "96.8%", label: "Upper-Face RF (19f)" },
    { value: "89.9%", label: "WESAD LOSO accuracy" },
    { value: "71", label: "Benchmark subjects" },
  ],

  engines: [
    { label: "Physiology Engine", value: "Active", note: "BVP (64Hz) + EDA (4Hz) Pipelines" },
    { label: "Video Engine", value: "Active", note: "19-Feature Clean Upper-Face Model" },
    { label: "Decision Engine", value: "Physio-First", note: "70:30 Dynamic Quality Gating" },
    { label: "Benchmark Cohorts", value: "71 Subjects", note: "56 UBFC-Phys + 15 WESAD" },
  ],

  registry: [
    { name: "WESAD CNN-TCN-LSTM", status: "READY", tone: "good" },
    { name: "UBFC Rest-Calib RF", status: "98.2%", tone: "good" },
    { name: "Upper-Face RF (19f)", status: "96.8%", tone: "good" },
    { name: "CLAS Cross-Transfer", status: "BLOCKED", tone: "warn" },
  ],

  architecture: {
    title: "Physiology-First Architecture:",
    text: "Primary stress classification is grounded in cardiovascular (BVP) and sympathetic nervous (EDA) dynamics. Facial video serves as supporting evidence and an automated fallback.",
  },

  status: {
    title: "System Status & Data Input Availability",
    subtitle: "Validation status across offline hardware and benchmark pipelines",
    badge: "Benchmark replay ready",
    intro: [
      { t: "The system operates in a " },
      { t: "physiology-first, video-assisted multimodal configuration", b: true },
      { t: ". All data displayed originate from verified hardware recordings across clinical benchmark datasets:" },
    ],
    bullets: [
      {
        lead: "UBFC-Phys Cohort:",
        text: "56 participants, synchronized 64 Hz BVP, 4 Hz EDA, and high-definition video during Trier Social Stress Test (Rest T1, Speech T2, Math T3).",
      },
      {
        lead: "WESAD Cohort:",
        text: "15 participants, wrist-worn Empatica E4 recordings with corrected LOSO validation (Acc: 89.9%, AUROC: 0.942).",
      },
      {
        lead: "Facial Video Processing:",
        text: "MediaPipe landmark tracking with mouth confound ablation to eliminate speech-motion false positives.",
      },
    ],
  },

  telemetry: {
    title: "Operational Telemetry",
    subtitle: "System timestamp and stream verification",
    rows: [
      { label: "Stream Source:", value: "Hardware Replay / Validated OOF", tone: "info" },
      { label: "Sensor Quality Gate:", value: "ACTIVE (Zero-Std Guard)", tone: "good" },
      { label: "Face Occlusion Gate:", value: "ACTIVE (<30% Face Triggers Fallback)", tone: "good" },
      { label: "CLAS Integrity Guard:", value: "ENFORCED (Incompatible Physics)", tone: "warn" },
    ],
  },

  integrity: {
    title: "Research Integrity & Anti-Fabrication Notice",
    parts: [
      { t: "This application " },
      { t: "never invents or fabricates predictions", b: true },
      { t: ". All displayed physiological values, video landmarks, and trend estimates are grounded in genuine machine learning checkpoints evaluated through subject-independent cross-validation." },
    ],
  },

  trajectory: {
    title: "Recent Session Stress Probability Trajectory",
    subtitle: "Sample multimodal session trajectory across Rest, Speech, and Math tasks",
    badge: "Subject s3",
  },

  notice:
    "Labels reflect standardized experimental stressor protocols (TSST, mental arithmetic, public speaking) as empirical proxies, not formal clinical psychiatric diagnoses.",
};