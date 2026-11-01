import json
import numpy as np
import pandas as pd

from physiology_inference import UBFCInferenceEngine
from multimodal_fusion import (
    UpperFaceVideoEngine,
    PhysiologyFirstFusionEngine,
)

ROOT = "."

PHYS_PATH = f"{ROOT}/results/physiology/physiology_features_labeled.csv"
VIDEO_PATH = f"{ROOT}/results/video/video_windows.csv"
OUTPUT_PATH = f"{ROOT}/results/fusion/deployed_rf_fusion_predictions.csv"

PHYS_FEATURES = [
    "bvp_mean", "bvp_std", "bvp_median", "bvp_iqr",
    "bvp_slope_per_sample", "bvp_range",
    "eda_mean", "eda_std", "eda_median", "eda_iqr",
    "eda_slope_per_sample", "eda_range",
]

print("Loading datasets...")
phys_df = pd.read_csv(PHYS_PATH)
video_df = pd.read_csv(VIDEO_PATH)

with open("models/video_upperface_features.json") as f:
    VIDEO_FEATURES = json.load(f)

# Normalize participant/task identifiers.
phys_df["subject_id"] = "s" + phys_df["participant"].astype(str)
phys_df["session_id"] = "T" + phys_df["task"].astype(str)

print("Loading existing trained models...")
phys_engine = UBFCInferenceEngine()
video_engine = UpperFaceVideoEngine(use_full_features=False)

fusion_engine = PhysiologyFirstFusionEngine(
    physio_weight=0.30,
    video_weight=0.70,
    video_feature_set="upper_face_19",
)

print("Models loaded.")
print("Physiology: uncalibrated RF for this initial integration run")
print("Video: upper-face 19-feature RF")
print("Fusion weights: 30% physiology / 70% video")

results = []

# Process each video window and its overlapping physiology windows.
for idx, vrow in video_df.iterrows():

    subject = str(vrow["subject_id"])
    task = str(vrow["session_id"])
    v_start = float(vrow["window_start_sec"])
    v_end = float(vrow["window_end_sec"])

    candidates = phys_df[
        (phys_df["subject_id"] == subject) &
        (phys_df["session_id"] == task) &
        (phys_df["start_s"] < v_end) &
        (phys_df["end_s"] > v_start)
    ]

    if candidates.empty:
        p_physio = None
        phys_quality = "UNAVAILABLE"
        n_phys_windows = 0
    else:
        phys_probs = []

        for _, prow in candidates.iterrows():
            features = prow[PHYS_FEATURES].to_dict()
            pred = phys_engine.predict_features(
                features,
                use_calibration=False
            )
            phys_probs.append(pred.probability)

        p_physio = float(np.median(phys_probs))
        phys_quality = "OK"
        n_phys_windows = len(phys_probs)

    # Video prediction using the existing 19-feature model.
    video_features = vrow[VIDEO_FEATURES].to_dict()

    video_prob = video_engine.predict(video_features)

    video_quality = (
        "OK"
        if bool(vrow["window_reliable"])
        and float(vrow["detection_fraction"]) >= 0.5
        else "NO_FACE"
    )

    # Perform quality-aware fusion.
    fused = fusion_engine.predict_from_oof(
        physio_prob=p_physio,
        video_prob=video_prob,
        physio_quality=phys_quality,
        video_quality=video_quality,
    )

    results.append({
        "subject_id": subject,
        "task": task,
        "window_idx": int(vrow["window_idx"]),
        "window_start_sec": v_start,
        "window_end_sec": v_end,
        "y_true": int(vrow["label"]),
        "p_physio": p_physio,
        "p_video": video_prob,
        "p_fused": fused.probability,
        "prediction": fused.prediction,
        "class_name": fused.class_name,
        "confidence": fused.confidence,
        "physio_quality": phys_quality,
        "video_quality": video_quality,
        "modality": fused.modality,
        "physio_weight": fused.blend_weights[0],
        "video_weight": fused.blend_weights[1],
        "n_overlapping_physio_windows": n_phys_windows,
    })

    if (idx + 1) % 100 == 0:
        print(f"Processed {idx + 1}/{len(video_df)} video windows")

output = pd.DataFrame(results)

import os
os.makedirs("results/fusion", exist_ok=True)
output.to_csv(OUTPUT_PATH, index=False)

print("\n========== FUSION COMPLETE ==========")
print("Total windows:", len(output))
print("Participants:", output.subject_id.nunique())
print("Tasks:", sorted(output.task.unique()))
print("Modalities:", output.modality.value_counts().to_dict())
print("Mean physiology probability:", round(output.p_physio.mean(), 4))
print("Mean video probability:", round(output.p_video.mean(), 4))
print("Mean fused probability:", round(output.p_fused.mean(), 4))
print("Output:", OUTPUT_PATH)
print("\nSample predictions:")
print(output.head(10).to_string(index=False))
