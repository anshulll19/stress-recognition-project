import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def compute_participant_baselines(df, feature_cols):
    """Computes baseline (T1 Rest) mean and std for each participant."""
    baselines = {}
    t1_df = df[df["session_id"] == "T1"]
    for p in df["subject_id"].unique():
        sub = t1_df[t1_df["subject_id"] == p]
        means = sub[feature_cols].mean()
        stds = sub[feature_cols].std().replace(0, 1e-4).fillna(1.0)
        baselines[p] = {"mean": means.to_dict(), "std": stds.to_dict()}
    return baselines

def get_stress_tier(score):
    if score <= 25:
        return "Baseline / Rest (Tier 1)", "Homeostatic stability; normal autonomic recovery."
    elif score <= 50:
        return "Mild Engagement / Alert (Tier 2)", "Optimal cognitive engagement; no adverse compensatory strain."
    elif score <= 75:
        return "Moderate Distress (Tier 3)", "Elevated sympathetic arousal with early compensatory fatigue."
    else:
        return "Acute Distress / Overload (Tier 4)", "Severe autonomic and neuromuscular overload; immediate intervention recommended."

def diagnose_window(row, baselines, feature_cols):
    sub = row["subject_id"]
    prob_blend = row["prob_blend_0.3"]
    prob_video = row["prob_video_upperface"]
    prob_phys = row["prob_phys_lstm"]
    score = int(round(prob_blend * 100))
    tier_name, tier_desc = get_stress_tier(score)

    # 1. Z-Score Feature Attribution against individual baseline
    b_mean = baselines[sub]["mean"]
    b_std = baselines[sub]["std"]

    subsystem_deviations = {
        "Ocular / Blink Fatigue": [],
        "Neuromuscular (Brow Furrowing)": [],
        "Kinematic Restlessness (Head Pose & Motion)": [],
        "Autonomic Hyperarousal": []
    }

    # Evaluate Video Features
    for feat in feature_cols:
        val = row[feat]
        z = (val - b_mean[feat]) / b_std[feat]
        
        if "ear" in feat:
            subsystem_deviations["Ocular / Blink Fatigue"].append((feat, z))
        elif "brow" in feat:
            subsystem_deviations["Neuromuscular (Brow Furrowing)"].append((feat, z))
        else:
            subsystem_deviations["Kinematic Restlessness (Head Pose & Motion)"].append((feat, z))

    # Evaluate Physiology
    phys_z = (prob_phys - 0.3409) / 0.20  # relative to cohort rest mean
    subsystem_deviations["Autonomic Hyperarousal"].append(("prob_phys_lstm", phys_z))

    # Determine dominant drivers
    ranked_drivers = []
    
    # Check Brow (contraction means lower brow-to-eye distance, negative z is stress)
    brow_z = [z for f, z in subsystem_deviations["Neuromuscular (Brow Furrowing)"] if "mean" in f]
    if brow_z and brow_z[0] < -0.8:
        ranked_drivers.append({
            "subsystem": "Neuromuscular Tension",
            "evidence": f"Corrugator contraction detected (Brow-Eye distance dropped {abs(brow_z[0]):.2f}σ below rest baseline)",
            "severity": "High" if brow_z[0] < -1.8 else "Moderate"
        })

    # Check Autonomic
    if prob_phys >= 0.60:
        ranked_drivers.append({
            "subsystem": "Autonomic Hyperarousal",
            "evidence": f"Sympathetic surge detected via cardiovascular/BVP dynamics (Physiology LSTM probability: {prob_phys:.2f})",
            "severity": "High" if prob_phys >= 0.75 else "Moderate"
        })

    # Check Kinematics
    motion_z = [z for f, z in subsystem_deviations["Kinematic Restlessness (Head Pose & Motion)"] if f == "f_motion_mean"]
    if motion_z and motion_z[0] > 1.0:
        ranked_drivers.append({
            "subsystem": "Motor Restlessness & Postural Agitation",
            "evidence": f"Elevated rigid head velocity and tilt micro-movements (+{motion_z[0]:.2f}σ above rest)",
            "severity": "High" if motion_z[0] > 2.0 else "Moderate"
        })

    # Check Blinking
    ear_z = [z for f, z in subsystem_deviations["Ocular / Blink Fatigue"] if "mean" in f]
    if ear_z and abs(ear_z[0]) > 1.2:
        ranked_drivers.append({
            "subsystem": "Ocular Strain / Blink Irregularity",
            "evidence": f"Eye Aspect Ratio dynamics deviated significantly from baseline ({ear_z[0]:+.2f}σ)",
            "severity": "Moderate"
        })

    if not ranked_drivers and score > 50:
        ranked_drivers.append({
            "subsystem": "Composite Cross-Modal Strain",
            "evidence": f"Combined visual ({prob_video:.2f}) and autonomic ({prob_phys:.2f}) load exceeded thresholds.",
            "severity": "Moderate"
        })
    elif not ranked_drivers:
        ranked_drivers.append({
            "subsystem": "Homeostatic Normalcy",
            "evidence": "All behavioral and physiological metrics aligned with resting equilibrium.",
            "severity": "None"
        })

    # Cross-Modal Discrepancy Check
    discrepancy_alert = None
    if abs(prob_video - prob_phys) >= 0.45:
        if prob_video > prob_phys:
            discrepancy_alert = f"Behavioral-Autonomic Divergence: Visual agitation ({prob_video:.2f}) exceeds autonomic activation ({prob_phys:.2f}). (Potential visual masking or cognitive suppression)."
        else:
            discrepancy_alert = f"Autonomic-Behavioral Divergence: Autonomic arousal ({prob_phys:.2f}) exceeds facial tension ({prob_video:.2f}). (Potential internal somatic distress with neutral facial affect)."

    # 3. Targeted Interventions (Remedies)
    remedies = []
    if score <= 25:
        remedies.append("Status Optimal: Continue workflow; maintain ergonomic posture.")
    elif score <= 50:
        remedies.append("Paced Cognitive Reset: Apply the 20-20-20 rule (20-second distant gaze focus) to prevent ocular strain.")
    elif score <= 75:
        remedies.append("Cognitive Unloading: Micro-break (60-90 seconds). Hydrate and release shoulder/neck posture.")
        if any(d["subsystem"] == "Neuromuscular Tension" for d in ranked_drivers):
            remedies.append("Facial Muscular Release: Consciously relax jaw, forehead, and brow muscles.")
    else:
        remedies.append("Immediate Physiological Regulation: Execute 2-3 physiological sighs (two deep nasal inhales followed by an extended oral exhale) to trigger vagal parasympathetic activation.")
        remedies.append("Task Disengagement: Pause high-intensity stimulus for 3-5 minutes to prevent cognitive burnout.")

    return {
        "subject_id": sub,
        "session_id": row["session_id"],
        "window_start": row["window_start_sec"],
        "window_end": row["window_end_sec"],
        "stress_score": score,
        "tier": tier_name,
        "tier_desc": tier_desc,
        "video_prob": round(prob_video, 4),
        "phys_prob": round(prob_phys, 4),
        "fused_prob": round(prob_blend, 4),
        "drivers": ranked_drivers,
        "discrepancy_alert": discrepancy_alert,
        "remedies": remedies
    }

def print_diagnostic_card(report):
    print("=" * 78)
    print(f"  STRESSSENSE MULTIMODAL DIAGNOSTIC REPORT: {report['subject_id']} [{report['session_id']} | {report['window_start']:.0f}s - {report['window_end']:.0f}s]")
    print("=" * 78)
    print(f"  STRESS SCORE   : {report['stress_score']} / 100")
    print(f"  SEVERITY TIER  : {report['tier']}")
    print(f"  STATUS OVERVIEW: {report['tier_desc']}")
    print("-" * 78)
    print("  CONFIDENCE BREAKDOWN:")
    print(f"    * Fused Multimodal (Blend 0.3) : {report['fused_prob'] * 100:.1f}%")
    print(f"    * Upper-Face Video Dynamics    : {report['video_prob'] * 100:.1f}%")
    print(f"    * Autonomic LSTM Physiology    : {report['phys_prob'] * 100:.1f}%")
    if report["discrepancy_alert"]:
        print(f"    ! ALERT: {report['discrepancy_alert']}")
    print("-" * 78)
    print("  EXPLAINABLE DRIVERS (WHY THIS SCORE?):")
    for idx, d in enumerate(report["drivers"], 1):
        print(f"    {idx}. [{d['severity'].upper()}] {d['subsystem']}: {d['evidence']}")
    print("-" * 78)
    print("  RECOMMENDED ACTIONABLE REMEDIES:")
    for idx, r in enumerate(report["remedies"], 1):
        print(f"    {idx}. {r}")
    print("=" * 78 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StressSense Clinical/Operational Diagnostic Engine")
    parser.add_argument("--demo", action="store_true", help="Run diagnostic demonstrator across protocol states")
    args = parser.parse_args()

    pred_path = Path("results/fusion_upperface/upperface_oof_predictions.csv")
    feat_path = Path("results/fusion_upperface/upperface_features_used.json")

    df = pd.read_csv(pred_path)
    with open(feat_path) as f:
        feature_cols = json.load(f)

    baselines = compute_participant_baselines(df, feature_cols)

    # Pick three representative protocol windows: T1 Rest, T2 Speech, T3 Mental Arithmetic
    demo_samples = [
        df[(df["subject_id"] == "s3") & (df["session_id"] == "T1")].iloc[2],
        df[(df["subject_id"] == "s3") & (df["session_id"] == "T2")].iloc[3],
        df[(df["subject_id"] == "s3") & (df["session_id"] == "T3")].iloc[4],
    ]

    print("\n[+] STRESSSENSE DECISION-SUPPORT & DIAGNOSTIC ENGINE LOADED")
    print(f"[+] Total windows available for diagnostics: {len(df)}")
    print(f"[+] Baseline profiles loaded for {len(baselines)} participants.\n")

    for row in demo_samples:
        diag = diagnose_window(row, baselines, feature_cols)
        print_diagnostic_card(diag)
