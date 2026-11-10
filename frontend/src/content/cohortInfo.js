// Values copied from models/deployment_candidate/manifest.json,
// demo/ui/tab_physio.py (CLAS audit) and CLASInferenceAdapter.get_audit_summary().
export const WESAD = {
  rows: [
    ["Model", "CNN-TCN-LSTM (classifier.keras)"],
    ["Input", "3840 samples (60 s at 64 Hz), wrist BVP"],
    ["Training subjects", "12 (S11, S13, S14, S15, S16, S17, S2, S3, S5, S7, S8, S9)"],
    ["Validation subjects", "3 (S10, S4, S6), used for checkpoint selection"],
    ["Windows", "649 training, 163 validation"],
    ["Training class counts", "505 non-stress, 144 stress"],
    ["Best epoch", "27 (validation loss 0.3092)"],
    ["Seed", "42"],
  ],
  note: "Subject-level validation was used for checkpoint selection. This score is not an independent test result and must not be reported as the corrected LOSO performance.",
};

export const CLAS = {
  warning:
    "Inference deliberately disabled. The repository has a self-supervised Masked Autoencoder (MAE) encoder for CLAS, but no trained classification head exists. In line with the project's integrity standards, fabricated predictions are strictly prohibited.",
  findings: [
    ["Dataset present", "false (data/clas/clas_binary.npz)"],
    ["Pretrained encoder weights", "false (models/pretrained_clas_mae_encoder.weights.h5)"],
    ["Architecture", "CNN_TCN_Encoder_Deconv_Decoder"],
    ["Labels used during MAE pretraining", "false (self-supervised masked reconstruction only)"],
    ["Trained classification head", "None (no checkpoint found)"],
  ],
};

export const CROSS_DATASET = [
  {
    experiment: "E2 (WESAD → CLAS)",
    train: "WESAD (Empatica E4 BVP)",
    test: "CLAS (Shimmer3 PPG)",
    accuracy: "62.6%",
    f1: "0.4026",
    auc: "0.3670 (sub-random)",
    outcome: "Severe domain collapse due to sensor and protocol mismatch",
  },
  {
    experiment: "E3 (CLAS → WESAD)",
    train: "CLAS (Shimmer3 PPG)",
    test: "WESAD (Empatica E4 BVP)",
    accuracy: "23.9%",
    f1: "0.2176",
    auc: "0.2226 (inverted ranking)",
    outcome: "Systematic polarity/signal inversion between photoplethysmography hardware",
  },
];
