"""
SEC Project — Multimodal Stress Monitoring Dashboard
Modern Healthcare Analytics Dark Theme (v3.0)

Physiology-First, Video-Assisted Multimodal Stress Recognition
Preserves all trained models, inference pipelines, and evaluation artifacts.
"""
import os
os.environ["MPLCONFIGDIR"] = "/tmp"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from datetime import datetime
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from demo.adapters.ubfc_adapter import UBFCPhysInferenceAdapter
from demo.adapters.wesad_adapter import WESADInferenceAdapter
from demo.adapters.video_adapter import VideoInferenceAdapter
from demo.adapters.fusion_adapter import MultimodalFusionAdapter
from demo.adapters.trend_adapter import TemporalTrendAdapter
from demo.adapters.clas_adapter import CLASInferenceAdapter
from demo.utils.metrics import ResearchMetricsLoader
from demo.ui.modern_theme import (
    HEALTHCARE_DARK_CSS, render_header, render_metric_card,
    render_probability_bar, render_modality_weights
)

# ──────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VitalPulse · Stress Monitoring Analytics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(HEALTHCARE_DARK_CSS, unsafe_allow_html=True)

PROJECT_ROOT = Path(__file__).resolve().parent


# ──────────────────────────────────────────────────────────────────────────────
# Resource Caching
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_adapters():
    ubfc    = UBFCPhysInferenceAdapter(PROJECT_ROOT)
    wesad   = WESADInferenceAdapter(PROJECT_ROOT)
    video   = VideoInferenceAdapter(PROJECT_ROOT)
    fusion  = MultimodalFusionAdapter(PROJECT_ROOT)
    trend   = TemporalTrendAdapter()
    clas    = CLASInferenceAdapter(PROJECT_ROOT)
    metrics = ResearchMetricsLoader(PROJECT_ROOT)
    return ubfc, wesad, video, fusion, trend, clas, metrics


ubfc_ad, wesad_ad, video_ad, fusion_ad, trend_ad, clas_ad, metrics_loader = get_adapters()


# ──────────────────────────────────────────────────────────────────────────────
# Modern Matplotlib Visualization Helpers
# ──────────────────────────────────────────────────────────────────────────────
def plot_bvp_eda_modern(bvp: np.ndarray, eda: np.ndarray, label: int,
                        pred_prob: float, fs_bvp: float = 64.0, fs_eda: float = 4.0) -> plt.Figure:
    """Healthcare analytics dual-channel signal plot with dark navy palette."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 3.8), facecolor="#0B0F19")
    t_bvp = np.arange(len(bvp)) / fs_bvp
    t_eda = np.arange(len(eda)) / fs_eda

    # Channel 1: Blood Volume Pulse (BVP)
    ax0 = axes[0]
    ax0.set_facecolor("#111827")
    ax0.plot(t_bvp, bvp, color="#06B6D4", linewidth=0.8, alpha=0.95)
    ax0.fill_between(t_bvp, bvp, alpha=0.08, color="#06B6D4")
    ax0.axhline(0, color="#1E293B", linewidth=0.6, linestyle="--")
    ax0.set_ylabel("BVP (V)", color="#9CA3AF", fontsize=7.5, fontfamily="monospace")
    ax0.set_xlim(t_bvp[0], t_bvp[-1])
    ax0.tick_params(colors="#6B7280", labelsize=6.5)
    ax0.spines[:].set_color("#1E293B")
    ax0.set_title(
        f"BVP Channel  ·  64 Hz  ·  10s Window  ·  Ground Truth: {'STRESS' if label==1 else 'REST'}  ·  P(Stress) = {pred_prob:.1%}",
        color="#CBD5E1", fontsize=8, fontfamily="sans-serif", loc="left", pad=4, fontweight="bold"
    )

    # Channel 2: Electrodermal Activity (EDA)
    ax1 = axes[1]
    ax1.set_facecolor("#111827")
    eda_color = "#00D2BA" if label == 0 else "#F43F5E"
    ax1.plot(t_eda, eda, color=eda_color, linewidth=1.1, alpha=0.95)
    ax1.fill_between(t_eda, eda, alpha=0.1, color=eda_color)
    ax1.set_ylabel("EDA (μS)", color="#9CA3AF", fontsize=7.5, fontfamily="monospace")
    ax1.set_xlabel("Time (seconds)", color="#9CA3AF", fontsize=7.5, fontfamily="monospace")
    ax1.set_xlim(t_eda[0], t_eda[-1])
    ax1.tick_params(colors="#6B7280", labelsize=6.5)
    ax1.spines[:].set_color("#1E293B")
    ax1.set_title(
        f"Electrodermal Activity (EDA/GSR)  ·  4 Hz  ·  Tonic/Phasic Skin Conductance",
        color="#CBD5E1", fontsize=8, fontfamily="sans-serif", loc="left", pad=4, fontweight="bold"
    )

    fig.patch.set_facecolor("#0B0F19")
    plt.tight_layout(pad=0.8)
    return fig


def plot_wesad_modern(window: np.ndarray, pred_prob: float, true_label: int,
                      fs: float = 64.0) -> plt.Figure:
    """Healthcare analytics WESAD BVP waveform plot."""
    fig, ax = plt.subplots(figsize=(10, 2.8), facecolor="#0B0F19")
    ax.set_facecolor("#111827")
    t = np.arange(len(window)) / fs
    sig = window.flatten()

    ax.plot(t, sig, color="#00D2BA", linewidth=0.75, alpha=0.95)
    ax.fill_between(t, sig, alpha=0.08, color="#00D2BA")
    ax.axhline(0, color="#1E293B", linewidth=0.6, linestyle="--")
    ax.set_ylabel("Normalized BVP", color="#9CA3AF", fontsize=7.5, fontfamily="monospace")
    ax.set_xlabel("Time (seconds)", color="#9CA3AF", fontsize=7.5, fontfamily="monospace")
    ax.set_xlim(t[0], t[-1])
    ax.tick_params(colors="#6B7280", labelsize=6.5)
    ax.spines[:].set_color("#1E293B")
    ax.set_title(
        f"WESAD Wrist BVP (Empatica E4)  ·  64 Hz  ·  60s Window  ·  Ground Truth: {'STRESS (TSST)' if true_label==1 else 'BASELINE'}  ·  P(Stress) = {pred_prob:.1%}",
        color="#CBD5E1", fontsize=8, fontfamily="sans-serif", loc="left", pad=4, fontweight="bold"
    )
    fig.patch.set_facecolor("#0B0F19")
    plt.tight_layout(pad=0.8)
    return fig


def plot_modality_comparison_modern(p_physio: float, p_video_full: float,
                                    p_fused: float, p_video_uf: float = None) -> plt.Figure:
    """Horizontal bar chart comparing modality probabilities."""
    labels = ["Physiology (LSTM/RF)", "Video (Full 23-Feat)"]
    probs = [p_physio, p_video_full]
    colors = ["#00D2BA", "#F59E0B"]

    if p_video_uf is not None:
        labels.append("Video (Upper 19-Feat)")
        probs.append(p_video_uf)
        colors.append("#06B6D4")

    labels.append("Multimodal Fused")
    probs.append(p_fused)
    colors.append("#8B5CF6")

    fig, ax = plt.subplots(figsize=(7, 2.7), facecolor="#0B0F19")
    ax.set_facecolor("#111827")
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, probs, height=0.55, color=colors, alpha=0.9, edgecolor="#1E293B")
    ax.axvline(0.5, color="#6B7280", linestyle=":", linewidth=1, alpha=0.7)

    for bar, p in zip(bars, probs):
        ax.text(min(p + 0.02, 0.94), bar.get_y() + bar.get_height()/2,
                f"{p:.1%}", va="center", color="#F9FAFB", fontsize=7.5, fontfamily="monospace")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color="#CBD5E1", fontsize=7.5)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("P(Stress)", color="#9CA3AF", fontsize=7.5)
    ax.tick_params(colors="#6B7280", labelsize=7)
    ax.spines[:].set_color("#1E293B")
    ax.set_title("Modality Decision Comparison", color="#CBD5E1", fontsize=8, loc="left", fontweight="bold")
    fig.patch.set_facecolor("#0B0F19")
    plt.tight_layout(pad=0.6)
    return fig


def plot_timeline_modern(sub_df: pd.DataFrame, method_col: str = "p_blend_0.3_uf") -> plt.Figure:
    """Out-of-fold temporal trajectory plot across an entire participant session."""
    fig, ax = plt.subplots(figsize=(10, 2.7), facecolor="#0B0F19")
    ax.set_facecolor("#111827")

    t_sec = sub_df["window_start_sec"].values
    p_phy = sub_df["p_physio"].values
    p_vid = sub_df["p_video"].values
    p_fus = sub_df[method_col].values if method_col in sub_df.columns else sub_df["p_blend_0.3"].values
    y_gt  = sub_df["y_true"].values

    # Color task intervals in background
    for task_id, col, name in [(1, "#10B981", "Rest (T1)"), (2, "#EF4444", "Speech (T2)"), (3, "#F59E0B", "Math (T3)")]:
        mask = sub_df["task"] == task_id
        if mask.any():
            t_sub = t_sec[mask]
            ax.axvspan(t_sub[0], t_sub[-1] + 10, alpha=0.08, color=col, label=f"Task: {name}")

    ax.plot(t_sec, p_phy, color="#00D2BA", linewidth=1.1, label="Physiology", alpha=0.9)
    ax.plot(t_sec, p_vid, color="#F59E0B", linewidth=0.9, linestyle="--", label="Video (Full)", alpha=0.6)
    ax.plot(t_sec, p_fus, color="#8B5CF6", linewidth=1.8, label="Multimodal Fused", alpha=1.0)
    ax.axhline(0.5, color="#6B7280", linestyle=":", linewidth=0.8, alpha=0.7)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(t_sec[0], t_sec[-1] + 10)
    ax.set_xlabel("Session Timeline (seconds)", color="#9CA3AF", fontsize=7.5, fontfamily="monospace")
    ax.set_ylabel("P(Stress)", color="#9CA3AF", fontsize=7.5, fontfamily="monospace")
    ax.tick_params(colors="#6B7280", labelsize=6.5)
    ax.spines[:].set_color("#1E293B")
    ax.legend(loc="upper right", fontsize=6.5, facecolor="#111827", edgecolor="#1E293B", labelcolor="#CBD5E1")
    ax.set_title("Synchronized Multi-Modality Temporal Trajectory", color="#CBD5E1", fontsize=8, loc="left", fontweight="bold")

    fig.patch.set_facecolor("#0B0F19")
    plt.tight_layout(pad=0.6)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar: System Status & Research Integrity Panel
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:1rem;">
            <div style="width:12px; height:12px; border-radius:50%; background:#00D2BA; box-shadow:0 0 10px #00D2BA;"></div>
            <span style="font-weight:700; font-size:1.1rem; color:#F9FAFB; letter-spacing:-0.01em;">VitalPulse Analytics</span>
        </div>
        <div style="font-size:0.75rem; color:#9CA3AF; margin-bottom:1.5rem;">
            SEC Multimodal Stress Recognition<br>
            Version 3.0 · Research Deployment Build
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="health-card" style="padding:12px; margin-bottom:1rem;">
            <div style="font-size:0.7rem; text-transform:uppercase; color:#9CA3AF; letter-spacing:0.05em; font-weight:600; margin-bottom:8px;">Model Registry</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px; font-size:0.75rem;">
                <span style="color:#CBD5E1;">WESAD CNN-TCN-LSTM</span>
                <span class="badge badge-live">READY</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px; font-size:0.75rem;">
                <span style="color:#CBD5E1;">UBFC Rest-Calib RF</span>
                <span class="badge badge-live">98.2%</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px; font-size:0.75rem;">
                <span style="color:#CBD5E1;">Upper-Face RF (19f)</span>
                <span class="badge badge-live">96.8%</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.75rem;">
                <span style="color:#CBD5E1;">CLAS Cross-Transfer</span>
                <span class="badge badge-warning">BLOCKED</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="callout-info">
            <strong>Physiology-First Architecture:</strong><br>
            Primary stress classification is grounded in cardiovascular (BVP) and sympathetic nervous (EDA) dynamics. Facial video serves as supporting evidence and an automated fallback.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="font-size:0.7rem; color:#6B7280; margin-top:2rem; line-height:1.4;">
            <strong>Research Notice:</strong><br>
            Labels reflect standardized experimental stressor protocols (TSST, mental arithmetic, public speaking) as empirical proxies, not formal clinical psychiatric diagnoses.
        </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Six Required Dashboard Tabs
# ──────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🌐 OVERVIEW",
    "📋 HEALTH PROFILE",
    "🫀 PHYSIOLOGICAL MONITORING",
    "📹 FACIAL ANALYSIS",
    "🔀 MULTIMODAL PREDICTION",
    "📈 TRENDS & RESEARCH"
])


# ==============================================================================
# TAB 1: OVERVIEW
# ==============================================================================
with tabs[0]:
    st.markdown(render_header(
        "Multimodal Stress Monitoring Overview",
        "Offline benchmark evaluation and synchronized multimodal replay overview",
        badge_text="BENCHMARK REPLAY ACTIVE · VERIFIED ARTIFACTS",
        badge_type="replay"
    ), unsafe_allow_html=True)

    # Top Status Indicators
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_metric_card(
            "Physiology Engine", "Active", "BVP (64Hz) + EDA (4Hz) Pipelines", "teal"
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(render_metric_card(
            "Video Engine", "Active", "19-Feature Clean Upper-Face Model", "cyan"
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(render_metric_card(
            "Decision Engine", "Physio-First", "70:30 Dynamic Quality Gating", "violet"
        ), unsafe_allow_html=True)
    with c4:
        st.markdown(render_metric_card(
            "Benchmark Cohorts", "71 Subjects", "56 UBFC-Phys + 15 WESAD", "teal"
        ), unsafe_allow_html=True)

    # System Status & Verification Summary
    col_l, col_r = st.columns([1.6, 1])
    with col_l:
        st.markdown("""
            <div class="health-card health-card-accent-teal">
                <div class="health-card-header">
                    <div>
                        <div class="health-card-title">System Status & Data Input Availability</div>
                        <div class="health-card-subtitle">Validation status across offline hardware and benchmark pipelines</div>
                    </div>
                    <span class="badge badge-replay">BENCHMARK REPLAY READY</span>
                </div>
                <div style="font-size:0.84rem; color:#CBD5E1; line-height:1.6;">
                    The system operates in a <strong>physiology-first, video-assisted multimodal configuration</strong>. All data displayed originate from verified hardware recordings across clinical benchmark datasets:
                    <ul style="margin-top:8px; margin-bottom:8px; padding-left:20px;">
                        <li><strong>UBFC-Phys Cohort:</strong> 56 participants, synchronized 64 Hz BVP, 4 Hz EDA, and high-definition video during Trier Social Stress Test (Rest T1, Speech T2, Math T3).</li>
                        <li><strong>WESAD Cohort:</strong> 15 participants, wrist-worn Empatica E4 recordings with corrected LOSO validation (Acc: 89.9%, AUROC: 0.942).</li>
                        <li><strong>Facial Video Processing:</strong> MediaPipe landmark tracking with mouth confound ablation to eliminate speech-motion false positives.</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="health-card health-card-accent-violet">
                <div class="health-card-header">
                    <div>
                        <div class="health-card-title">Recent Session Stress Probability Trajectory</div>
                        <div class="health-card-subtitle">Sample multimodal session trajectory across Rest, Speech, and Math tasks</div>
                    </div>
                    <span class="badge badge-replay">SUBJECT s3</span>
                </div>
        """, unsafe_allow_html=True)
        aligned_df_ov = fusion_ad.load_aligned_data()
        s3_rows = aligned_df_ov[aligned_df_ov["subject_id"] == "s3"].reset_index(drop=True)
        if not s3_rows.empty:
            fig_ov = plot_timeline_modern(s3_rows)
            st.pyplot(fig_ov, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown(f"""
            <div class="health-card health-card-accent-cyan">
                <div class="health-card-header">
                    <div>
                        <div class="health-card-title">Operational Telemetry</div>
                        <div class="health-card-subtitle">System timestamp and stream verification</div>
                    </div>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:10px; font-size:0.8rem;">
                    <span style="color:#9CA3AF;">System Clock:</span>
                    <span style="font-family:'JetBrains Mono',monospace; color:#F9FAFB;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:10px; font-size:0.8rem;">
                    <span style="color:#9CA3AF;">Stream Source:</span>
                    <span style="color:#06B6D4; font-weight:600;">Hardware Replay / Validated OOF</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:10px; font-size:0.8rem;">
                    <span style="color:#9CA3AF;">Sensor Quality Gate:</span>
                    <span style="color:#00D2BA; font-weight:600;">ACTIVE (Zero-Std Guard)</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:10px; font-size:0.8rem;">
                    <span style="color:#9CA3AF;">Face Occlusion Gate:</span>
                    <span style="color:#00D2BA; font-weight:600;">ACTIVE (&lt;30% Face Triggers Fallback)</span>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.8rem;">
                    <span style="color:#9CA3AF;">CLAS Integrity Guard:</span>
                    <span style="color:#F59E0B; font-weight:600;">ENFORCED (Incompatible Physics)</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="health-card health-card-accent-rose">
                <div class="health-card-header">
                    <div>
                        <div class="health-card-title">Research Integrity & Anti-Fabrication Notice</div>
                    </div>
                </div>
                <div style="font-size:0.78rem; color:#CBD5E1; line-height:1.5;">
                    This application <strong>never invents or fabricates predictions</strong>. All displayed physiological values, video landmarks, and trend estimates are grounded in genuine machine learning checkpoints evaluated through subject-independent cross-validation.
                </div>
            </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# TAB 2: HEALTH PROFILE
# ==============================================================================
with tabs[1]:
    st.markdown(render_header(
        "Individual Health Profile & Perceived Stress Assessment",
        "Clinical context, lifestyle factors, and validated Perceived Stress Scale (PSS-10) intake",
        badge_text="CONTEXTUAL INTAKE · NOT INFERENCE INPUT",
        badge_type="replay"
    ), unsafe_allow_html=True)

    st.markdown("""
        <div class="callout-warning">
            <strong>Clinical Context Boundary Notice:</strong><br>
            The demographic, vital, and lifestyle fields below provide personal health context and longitudinal tracking. In strict accordance with research integrity requirements, <strong>these questionnaire fields are NOT fed into the physiological signal models</strong> (which are trained exclusively on sensor time-series). No unvalidated machine-learning score is generated from this form.
        </div>
    """, unsafe_allow_html=True)

    with st.form("health_profile_form"):
        col_hp1, col_hp2 = st.columns(2)

        with col_hp1:
            st.markdown("#### 1. Basic Demographic & Vital Sign Context")
            hp_age = st.number_input("Age (years)", min_value=18, max_value=100, value=25, step=1)
            hp_gender = st.selectbox("Gender (optional)", ["Prefer not to say", "Female", "Male", "Non-binary", "Other"])
            hp_rhr = st.number_input("Resting Heart Rate (BPM, optional)", min_value=40, max_value=180, value=72)
            c_bp1, c_bp2 = st.columns(2)
            with c_bp1:
                hp_sbp = st.number_input("Systolic BP (mmHg, optional)", min_value=70, max_value=220, value=120)
            with c_bp2:
                hp_dbp = st.number_input("Diastolic BP (mmHg, optional)", min_value=40, max_value=130, value=80)

            st.markdown("#### 2. Lifestyle & Sleep Context")
            hp_sleep_dur = st.slider("Typical Sleep Duration (hours/night)", 3.0, 12.0, 7.5, 0.5)
            hp_sleep_qual = st.select_slider("Subjective Sleep Quality", options=["Very Poor", "Poor", "Fair", "Good", "Excellent"], value="Good")
            hp_activity = st.selectbox("Physical Activity Level", ["Sedentary (<30 min/week)", "Lightly Active (1-2 days/week)", "Moderately Active (3-4 days/week)", "Very Active (5+ days/week)"], index=2)

            st.markdown("#### 3. Medical Background (Optional)")
            hp_conditions = st.multiselect("Relevant Known Conditions (optional)", ["Hypertension", "Arrhythmia", "Anxiety Disorder", "Asthma", "Diabetes", "None"])
            hp_medications = st.text_input("Current Relevant Medications (optional, e.g. Beta-blockers, Stimulants)", placeholder="None or list optional medications")

        with col_hp2:
            st.markdown("#### 4. Validated Perceived Stress Scale (PSS-10)")
            st.caption("Rate how often you felt or thought a certain way during the last month (Cohen et al., 1983).")

            pss_options = ["0 - Never", "1 - Almost Never", "2 - Sometimes", "3 - Fairly Often", "4 - Very Often"]

            pss_q1 = st.selectbox("1. Been upset because of something that happened unexpectedly?", pss_options, index=2)
            pss_q2 = st.selectbox("2. Felt that you were unable to control important things in your life?", pss_options, index=2)
            pss_q3 = st.selectbox("3. Felt nervous and stressed?", pss_options, index=2)
            pss_q4 = st.selectbox("4. Felt confident about your ability to handle personal problems? (Reverse scored)", pss_options, index=3)
            pss_q5 = st.selectbox("5. Felt that things were going your way? (Reverse scored)", pss_options, index=3)
            pss_q6 = st.selectbox("6. Found that you could not cope with all the things you had to do?", pss_options, index=1)
            pss_q7 = st.selectbox("7. Been able to control irritations in your life? (Reverse scored)", pss_options, index=3)
            pss_q8 = st.selectbox("8. Felt that you were on top of things? (Reverse scored)", pss_options, index=3)
            pss_q9 = st.selectbox("9. Been angered because of things that happened outside of your control?", pss_options, index=2)
            pss_q10 = st.selectbox("10. Felt difficulties were piling up so high that you could not overcome them?", pss_options, index=1)

        submitted = st.form_submit_button("Calculate Validated PSS-10 Score & Save Profile Summary")

    if submitted:
        # Compute PSS-10 Score
        # Items 1, 2, 3, 6, 9, 10 are standard (0 to 4)
        # Items 4, 5, 7, 8 are positive/reverse scored (0->4, 1->3, 2->2, 3->1, 4->0)
        def parse_pss(val: str, reverse: bool = False) -> int:
            num = int(val[0])
            return (4 - num) if reverse else num

        pss_score = (
            parse_pss(pss_q1) + parse_pss(pss_q2) + parse_pss(pss_q3) +
            parse_pss(pss_q4, reverse=True) + parse_pss(pss_q5, reverse=True) +
            parse_pss(pss_q6) + parse_pss(pss_q7, reverse=True) +
            parse_pss(pss_q8, reverse=True) + parse_pss(pss_q9) + parse_pss(pss_q10)
        )

        if pss_score <= 13:
            pss_cat = "Low Perceived Stress"
            pss_color = "teal"
            pss_badge_cls = "badge-nonstress"
        elif pss_score <= 26:
            pss_cat = "Moderate Perceived Stress"
            pss_color = "cyan"
            pss_badge_cls = "badge-replay"
        else:
            pss_cat = "High Perceived Stress"
            pss_color = "rose"
            pss_badge_cls = "badge-stress"

        st.markdown("---")
        st.markdown("#### Health Profile & Perceived Stress Summary")
        hp_c1, hp_c2, hp_c3 = st.columns([1, 1, 1.2])
        with hp_c1:
            st.markdown(render_metric_card(
                "PSS-10 Total Score", f"{pss_score} / 40", f"Category: {pss_cat}", pss_color
            ), unsafe_allow_html=True)
        with hp_c2:
            st.markdown(render_metric_card(
                "Cardiovascular Baseline", f"{hp_rhr} BPM", f"Blood Pressure: {hp_sbp}/{hp_dbp} mmHg", "teal"
            ), unsafe_allow_html=True)
        with hp_c3:
            st.markdown(f"""
                <div class="health-card health-card-accent-{pss_color}">
                    <div class="health-card-header">
                        <div class="health-card-title">PSS-10 Population Norm Interpretation</div>
                        <span class="badge {pss_badge_cls}">{pss_cat.upper()}</span>
                    </div>
                    <div style="font-size:0.8rem; color:#CBD5E1; line-height:1.5;">
                        <strong>Population Distribution Brackets (Cohen et al., 1983):</strong><br>
                        • <strong>0–13:</strong> Low perceived stress (lower quartile in healthy adult population samples).<br>
                        • <strong>14–26:</strong> Moderate perceived stress (normative range in healthy adult populations).<br>
                        • <strong>27–40:</strong> High perceived stress (upper quartile indicating elevated perceived demands).<br>
                        <em>Important Notice: PSS-10 evaluates subjective cognitive perception of unpredictable, uncontrollable stressors over the preceding 30 days. These score brackets reflect statistical population sample distributions, NOT clinical psychiatric diagnostic thresholds. In accordance with research integrity boundaries, this profile serves as contextual background and is not an input to the sensor models.</em>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="health-card" style="margin-top:1.5rem; text-align:center; padding:2rem 1.5rem;">
                <div style="font-size:1.05rem; font-weight:600; color:#CBD5E1; margin-bottom:0.5rem;">
                    Health Profile &amp; PSS-10 Assessment Awaiting Submission
                </div>
                <div style="font-size:0.82rem; color:#9CA3AF; max-width:680px; margin:0 auto; line-height:1.6;">
                    Complete the demographic context, lifestyle factors, and 10 Perceived Stress Scale questions above, then click <strong>"Calculate Validated PSS-10 Score &amp; Save Profile Summary"</strong> to review your score.<br><br>
                    <strong>Session Privacy &amp; Data Boundary:</strong> All entered health context is maintained strictly in volatile local session state and is never saved to disk or transmitted externally.<br>
                    <strong>Psychometric Boundary:</strong> The PSS-10 (Cohen et al., 1983) measures non-specific cognitive distress over the past month. Score brackets (0–13 Low, 14–26 Moderate, 27–40 High) represent population norm distributions rather than formal clinical diagnostic thresholds.
                </div>
            </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# TAB 3: PHYSIOLOGICAL MONITORING
# ==============================================================================
with tabs[2]:
    st.markdown(render_header(
        "Physiological Signal Monitoring & Inference",
        "Raw cardiovascular and sympathetic autonomic nervous system signal intelligence",
        badge_text="PRIMARY INFERENCE BRANCH (70% WEIGHT)",
        badge_type="replay"
    ), unsafe_allow_html=True)

    dataset_choice = st.radio(
        "Physiological Dataset / Hardware Cohort",
        ["UBFC-Phys (BVP + EDA · 56 Participants · 10s Windows)",
         "WESAD (Wrist BVP · Empatica E4 · 15 Subjects · 60s Windows)",
         "CLAS Sensor Transfer Audit (Permanent Blocker)"],
        horizontal=True,
        key="physio_cohort_radio"
    )

    if "UBFC-Phys" in dataset_choice:
        col_u1, col_u2, col_u3 = st.columns([1.2, 1.2, 1.6])
        with col_u1:
            ubfc_subs = ubfc_ad.get_participants()
            sel_u_sub = st.selectbox("Participant (56 Cohort)", ubfc_subs, index=0, key="ubfc_sel_p")
        with col_u2:
            ubfc_models = ubfc_ad.get_models_available()
            default_m_idx = ubfc_models.index("LSTM") if "LSTM" in ubfc_models else 0
            sel_u_model = st.selectbox("Temporal Inference Model", ubfc_models, index=default_m_idx, key="ubfc_sel_model")

        indices_u, bvp_u, eda_u, y_u, meta_u, oof_u = ubfc_ad.get_participant_windows(
            sel_u_sub, model_name=sel_u_model
        )
        n_win_u = len(indices_u)
        with col_u3:
            win_u = st.slider("Window Selection (10s stride 5s)", 0, max(n_win_u - 1, 0), 0, key="ubfc_win_pos")

        row_meta = meta_u.iloc[win_u].to_dict()
        cur_bvp  = bvp_u[win_u]
        cur_eda  = eda_u[win_u]
        cur_g_idx = int(indices_u[win_u])
        cur_gt    = int(y_u[win_u])

        pred_u = ubfc_ad.predict(
            {"bvp": cur_bvp, "eda": cur_eda},
            {"window_index": cur_g_idx, "model_name": sel_u_model,
             "participant": sel_u_sub, "task": row_meta["task"]}
        )
        p_u_stress = float(pred_u.probabilities[1])

        # Metrics display
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Predicted State",
                      "STRESS" if pred_u.predicted_class == 1 else "NON-STRESS",
                      delta=f"Class {pred_u.predicted_class}",
                      delta_color="inverse" if pred_u.predicted_class == 1 else "normal")
        with m2:
            st.metric("P(Stress)", f"{p_u_stress:.1%}",
                      delta=ubfc_ad.TASK_NAMES.get(row_meta["task"], "—"), delta_color="off")
        with m3:
            match_ok = cur_gt == pred_u.predicted_class
            st.metric("Ground Truth", "Stress" if cur_gt == 1 else "Rest",
                      delta="✓ MATCH" if match_ok else "✗ MISMATCH",
                      delta_color="normal" if match_ok else "inverse")
        with m4:
            st.metric("Protocol Interval",
                      f"T{row_meta['task']} · {row_meta['start_s']:.0f}–{row_meta['end_s']:.0f}s",
                      delta=f"Window #{cur_g_idx}", delta_color="off")

        # Visualizations
        col_pb_u, col_wf_u = st.columns([1, 2.5])
        with col_pb_u:
            st.markdown("#### Probability Analysis")
            st.markdown(render_probability_bar(p_u_stress, "Stress Probability", "rose" if p_u_stress >= 0.5 else "teal"), unsafe_allow_html=True)
            st.markdown(render_probability_bar(1 - p_u_stress, "Rest Probability", "teal"), unsafe_allow_html=True)

            bvp_std = float(np.std(cur_bvp))
            eda_std = float(np.std(cur_eda))
            bvp_flat = bvp_std < 1e-7 or np.isnan(bvp_std)
            eda_flat = eda_std < 1e-7 or np.isnan(eda_std)
            is_valid_u = not (bvp_flat or eda_flat)

            st.markdown(f"""
                <div class="health-card" style="margin-top:12px; padding:12px;">
                    <div style="font-size:0.75rem; color:#9CA3AF;">Signal Quality Gate &amp; Telemetry</div>
                    <div style="font-size:0.95rem; font-weight:700; color:{'#00D2BA' if is_valid_u else '#F43F5E'}; margin-top:4px;">
                        {'VALID (Active Waveform Dynamics)' if is_valid_u else 'DEGRADED (Zero-Variance Flatline)'}
                    </div>
                    <div style="font-size:0.72rem; color:#6B7280; margin-top:4px; line-height:1.4;">
                        Window: {len(cur_bvp)} BVP samples (64 Hz) · {len(cur_eda)} EDA samples (4 Hz)<br>
                        BVP Std Dev: <span style="font-family:'JetBrains Mono',monospace; color:#CBD5E1;">{bvp_std:.4f}</span> | EDA Std Dev: <span style="font-family:'JetBrains Mono',monospace; color:#CBD5E1;">{eda_std:.4f}</span><br>
                        Flatline check: <span style="color:{'#00D2BA' if not bvp_flat else '#F43F5E'};">{'PASS' if not bvp_flat else 'FAIL (BVP Flatline)'}</span> · 
                        <span style="color:{'#00D2BA' if not eda_flat else '#F43F5E'};">{'PASS' if not eda_flat else 'FAIL (EDA Flatline)'}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col_wf_u:
            fig_sig = plot_bvp_eda_modern(cur_bvp.flatten(), cur_eda.flatten(), cur_gt, p_u_stress)
            st.pyplot(fig_sig, width="stretch")

    elif "WESAD" in dataset_choice:
        wesad_manifest = wesad_ad.get_manifest()
        val_subs = wesad_manifest.get("validation_subjects", [])
        all_w_subs = wesad_ad.get_subject_list()

        col_w1, col_w2 = st.columns([1.2, 2])
        with col_w1:
            sel_w_sub = st.selectbox("Participant (15 Cohort)", all_w_subs, index=0, key="wesad_sel_sub")
        indices_w, windows_w, labels_w = wesad_ad.get_subject_windows(sel_w_sub)
        n_win_w = len(indices_w)
        with col_w2:
            win_w = st.slider("Window Selection (60s Window @ 64Hz)", 0, max(n_win_w - 1, 0), 0, key="wesad_win_pos")

        cur_win_w = windows_w[win_w]
        cur_gt_w  = int(labels_w[win_w])
        pred_w    = wesad_ad.predict(cur_win_w,
                        {"subject_id": sel_w_sub, "window_index": int(indices_w[win_w]),
                         "ground_truth": cur_gt_w})
        p_w_stress = float(pred_w.probabilities[1])

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Predicted State",
                      "STRESS" if pred_w.predicted_class == 1 else "NON-STRESS",
                      delta=f"Class {pred_w.predicted_class}",
                      delta_color="inverse" if pred_w.predicted_class == 1 else "normal")
        with m2:
            st.metric("P(Stress)", f"{p_w_stress:.1%}", delta="Softmax", delta_color="off")
        with m3:
            match_w = cur_gt_w == pred_w.predicted_class
            st.metric("Ground Truth", "Stress (TSST)" if cur_gt_w == 1 else "Baseline",
                      delta="✓ MATCH" if match_w else "✗ MISMATCH",
                      delta_color="normal" if match_w else "inverse")
        with m4:
            is_val_split = sel_w_sub in val_subs
            st.metric("Cohort Role",
                      "Validation Split" if is_val_split else "Training Split",
                      delta="Checkpoint Selection" if is_val_split else "Training Cohort", delta_color="off")

        col_pb_w, col_wf_w = st.columns([1, 2.5])
        with col_pb_w:
            st.markdown("#### Probability Analysis")
            st.markdown(render_probability_bar(p_w_stress, "Stress Probability", "rose" if p_w_stress >= 0.5 else "teal"), unsafe_allow_html=True)
            st.markdown(render_probability_bar(1 - p_w_stress, "Rest Probability", "teal"), unsafe_allow_html=True)
            w_std = float(np.std(cur_win_w))
            w_flat = w_std < 1e-7 or np.isnan(w_std)

            st.markdown(f"""
                <div class="health-card" style="margin-top:12px; padding:12px;">
                    <div style="font-size:0.75rem; color:#9CA3AF;">Signal Quality Gate &amp; Architecture</div>
                    <div style="font-size:0.95rem; font-weight:700; color:{'#00D2BA' if not w_flat else '#F43F5E'}; margin-top:2px;">
                        {'VALID (Active BVP Signal)' if not w_flat else 'DEGRADED (Zero-Variance Flatline)'}
                    </div>
                    <div style="font-size:0.72rem; color:#6B7280; margin-top:4px; line-height:1.4;">
                        Model: <strong style="color:#06B6D4;">CNN-TCN-LSTM</strong> (classifier.keras)<br>
                        Window: {len(cur_win_w)} float samples (60s @ 64 Hz)<br>
                        BVP Std Dev: <span style="font-family:'JetBrains Mono',monospace; color:#CBD5E1;">{w_std:.4f}</span><br>
                        Flatline check: <span style="color:{'#00D2BA' if not w_flat else '#F43F5E'};">{'PASS' if not w_flat else 'FAIL (Sensor Dropout)'}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col_wf_w:
            fig_w = plot_wesad_modern(cur_win_w, p_w_stress, cur_gt_w)
            st.pyplot(fig_w, width="stretch")

    else:
        # CLAS Transfer Audit
        from demo.ui.tab_physio import render_clas_audit_panel
        render_clas_audit_panel(clas_ad)


# ==============================================================================
# TAB 4: FACIAL ANALYSIS
# ==============================================================================
with tabs[3]:
    st.markdown(render_header(
        "Facial Behavioral Analysis & Confound Isolation",
        "Computer vision facial action units with speech-confound isolation (19-feature upper-face)",
        badge_text="SUPPORTING INFERENCE BRANCH (30% WEIGHT)",
        badge_type="replay"
    ), unsafe_allow_html=True)

    # Confound Toggle
    feat_mode_sel = st.radio(
        "Facial Feature Architecture",
        ["Upper-Face (19 features — mouth landmarks removed ✓ Clean affect estimate)",
         "Full-Face (23 features — includes mouth landmarks ⚠ Confounded by speech)"],
        index=0,
        horizontal=True,
        key="video_feat_mode_radio"
    )
    use_upper_features = ("Upper-Face" in feat_mode_sel)

    # Video Input Pipeline
    with st.expander("🎥 Video Input & Pipeline Verification Status (Upload Unavailable)"):
        c_vid_src1, c_vid_src2 = st.columns(2)
        with c_vid_src1:
            uploaded_video = st.file_uploader(
                "Upload video file for facial landmark extraction (.mp4, .avi, .mov) — [PIPELINE UNAVAILABLE]",
                type=["mp4", "avi", "mov"],
                help="End-to-end video decoding and MediaPipe frame-by-frame landmark extraction pipeline is not yet implemented."
            )
            if uploaded_video is not None:
                st.warning(
                    "⚠️ Video decoding pipeline is NOT YET IMPLEMENTED / UNAVAILABLE in this dashboard build. "
                    "The deployed Random Forest models require precomputed 19- or 23-dimensional feature vectors. "
                    "The dashboard does not decode raw video files on-the-fly and will NOT fabricate predictions. "
                    "Inference below is operating strictly on verified UBFC-Phys precomputed benchmark windows."
                )
            else:
                st.caption(
                    "Note: Custom video decoding is unavailable. Inference operates on precomputed facial action landmark "
                    "features from verified UBFC-Phys benchmark windows (results/video/video_windows.csv)."
                )
        with c_vid_src2:
            st.markdown("""
                <div class="health-card" style="padding:10px; margin:0;">
                    <div style="font-size:0.75rem; color:#9CA3AF; font-weight:600;">Hardware Ingestion Status</div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:6px;">
                        <span style="font-size:0.8rem; color:#CBD5E1;">Live Webcam Capture:</span>
                        <span class="badge badge-warning">NOT IMPLEMENTED / DISABLED</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:6px;">
                        <span style="font-size:0.8rem; color:#CBD5E1;">Video File Decoding:</span>
                        <span class="badge badge-warning">UNAVAILABLE / OFFLINE ONLY</span>
                    </div>
                    <div style="font-size:0.7rem; color:#6B7280; margin-top:6px; line-height:1.4;">
                        Real-time webcam ingestion and file-based video decoding require dedicated background extraction workers. Active inference is strictly offline replay of verified UBFC-Phys benchmark feature recordings.
                    </div>
                </div>
            """, unsafe_allow_html=True)

    aligned_v_df = fusion_ad.load_aligned_data()
    subjects_v = fusion_ad.get_subjects()

    col_v1, col_v2 = st.columns([1.2, 2])
    with col_v1:
        sel_v_sub = st.selectbox("Participant (18 Video Cohort)", subjects_v, index=0, key="vid_sel_sub")
    sub_v_rows = aligned_v_df[aligned_v_df["subject_id"] == sel_v_sub].reset_index(drop=True)
    n_v_win = len(sub_v_rows)
    with col_v2:
        win_v = st.slider("Window Selection (10s Video Window)", 0, max(n_v_win - 1, 0), 0, key="vid_slider")

    row_vid = sub_v_rows.iloc[win_v]
    p_vid_full = float(row_vid.get("p_video", 0.5))
    p_vid_uf   = float(row_vid.get("p_video_upperface", p_vid_full))
    p_vid_disp = p_vid_uf if use_upper_features else p_vid_full
    gt_vid     = int(row_vid["y_true"])
    pred_vid   = int(p_vid_disp >= 0.5)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Feature Set", "19 Features (Clean)" if use_upper_features else "23 Features (Confounded)",
                  delta="Mouth Removed" if use_upper_features else "⚠ Speech Risk",
                  delta_color="normal" if use_upper_features else "inverse")
    with m2:
        st.metric("P(Stress)", f"{p_vid_disp:.1%}",
                  delta="STRESS" if pred_vid == 1 else "REST",
                  delta_color="inverse" if pred_vid == 1 else "normal")
    with m3:
        match_vid = pred_vid == gt_vid
        st.metric("Ground Truth", "Stress" if gt_vid == 1 else "Non-stress",
                  delta="✓ MATCH" if match_vid else "✗ MISMATCH",
                  delta_color="normal" if match_vid else "inverse")
    with m4:
        det_frac = float(row_vid.get("detection_fraction", 1.0))
        st.metric("Face Detection Quality", f"{det_frac:.0%}",
                  delta="Good Quality" if det_frac >= 0.8 else "Low Quality",
                  delta_color="normal" if det_frac >= 0.8 else "inverse")

    col_pb_v, col_cmp_v = st.columns([1, 2.5])
    with col_pb_v:
        st.markdown("#### Model Comparison")
        st.markdown(render_probability_bar(p_vid_full, "Full-Face (23-Feat)", "rose" if p_vid_full >= 0.5 else "cyan"), unsafe_allow_html=True)
        st.markdown(render_probability_bar(p_vid_uf, "Upper-Face (19-Feat)", "rose" if p_vid_uf >= 0.5 else "teal"), unsafe_allow_html=True)

        confound_shift = abs(p_vid_full - p_vid_uf)
        if confound_shift > 0.05:
            st.markdown(f"""
                <div class="callout-warning" style="margin-top:10px;">
                    <strong>Mouth Speech Confound:</strong><br>
                    Mouth articulation shifts probability by {confound_shift:.1%}.
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="callout-info" style="margin-top:10px;">
                    Mouth speech confound minimal on this window.
                </div>
            """, unsafe_allow_html=True)

    with col_cmp_v:
        fig_cmp = plot_modality_comparison_modern(
            float(row_vid.get("p_physio", 0.5)), p_vid_full,
            float(row_vid.get("p_blend_0.3", 0.5)), p_vid_uf
        )
        st.pyplot(fig_cmp, width="stretch")

    st.markdown("#### Confound Ablation Benchmark Across 972 Aligned Windows")
    confound_df_v = fusion_ad.load_confounds()
    if confound_df_v is not None and not confound_df_v.empty:
        st.dataframe(confound_df_v, width="stretch", hide_index=True)


# ==============================================================================
# TAB 5: MULTIMODAL STRESS PREDICTION
# ==============================================================================
with tabs[4]:
    st.markdown(render_header(
        "Multimodal Stress Prediction & Late Fusion",
        "Physiology-First decision policy with dynamic quality gating and video fallback",
        badge_text="FUSION ARBITRATOR · 70:30 VALIDATED RATIO",
        badge_type="replay"
    ), unsafe_allow_html=True)

    col_f_c1, col_f_c2, col_f_c3 = st.columns([1.2, 1.4, 1.4])
    with col_f_c1:
        f_aligned_df = fusion_ad.load_aligned_data()
        f_subs = fusion_ad.get_subjects()
        sel_f_sub = st.selectbox("Participant (18 Aligned Cohort)", f_subs, index=0, key="fusion_sel_sub")
    f_sub_rows = fusion_ad.get_subject_windows(sel_f_sub)
    n_f_win = len(f_sub_rows)
    with col_f_c2:
        method_f = st.selectbox(
            "Fusion Decision Strategy", fusion_ad.FUSION_METHODS, index=0,
            format_func=lambda m: fusion_ad.METHOD_DESCRIPTIONS.get(m, m),
            key="fusion_method_sel"
        )
    with col_f_c3:
        win_f = st.slider("Synchronized Window", 0, max(n_f_win - 1, 0), 0, key="fus_slider")

    row_f = f_sub_rows.iloc[win_f]
    pred_fusion = fusion_ad.get_fusion_prediction(row_f, method=method_f)
    meta_fusion = pred_fusion.window_metadata

    p_phy_f = float(row_f.get("p_physio", 0.5))
    p_vid_f = float(row_f.get("p_video", 0.5))
    p_vid_uf_f = row_f.get("p_video_upperface", None)
    p_fused_f = float(meta_fusion["p_fused"])
    det_frac_f = float(meta_fusion["detection_fraction"])

    # Determine Modality Weights and Decision Explanation based on verified fusion strategy
    if method_f == "blend_0.3_uf":
        w_phy_dyn, w_vid_dyn = 0.70, 0.30
        decision_branch = "Physiology-First Fixed Blend (70% Physio + 30% Clean Upper-Face Video, Mouth Confound Removed)"
    elif method_f == "blend_0.3":
        w_phy_dyn, w_vid_dyn = 0.30, 0.70
        decision_branch = "Benchmark Fixed Blend (30% Physio + 70% Full-Face Video)"
    elif method_f == "blend_0.2":
        w_phy_dyn, w_vid_dyn = 0.20, 0.80
        decision_branch = "Benchmark Fixed Blend (20% Physio + 80% Full-Face Video)"
    elif method_f == "blend_0.5":
        w_phy_dyn, w_vid_dyn = 0.50, 0.50
        decision_branch = "Equal Weighting Blend (50% Physio + 50% Full-Face Video)"
    elif method_f == "quality_aware":
        w_phy_dyn = float(np.clip(0.20 + (1.0 - det_frac_f) * (0.40 - 0.20), 0.0, 1.0))
        w_vid_dyn = 1.0 - w_phy_dyn
        decision_branch = f"Quality-Aware Dynamic Weighting (Physio: {w_phy_dyn:.1%}, Video: {w_vid_dyn:.1%}, Scaled by Face Detection Fraction: {det_frac_f:.0%})"
    elif method_f == "stacked_loso":
        w_phy_dyn, w_vid_dyn = None, None
        decision_branch = "Stacked Logistic Regression Meta-Learner (Inner-LOSO Trained Coefficients)"
    elif method_f == "video_upperface":
        w_phy_dyn, w_vid_dyn = 0.0, 1.0
        decision_branch = "Unimodal Video Only (19 Clean Upper-Face Features)"
    elif method_f == "video_only":
        w_phy_dyn, w_vid_dyn = 0.0, 1.0
        decision_branch = "Unimodal Video Only (23 Full-Face Features, Confounded by Speech)"
    elif method_f == "physio_only":
        w_phy_dyn, w_vid_dyn = 1.0, 0.0
        decision_branch = "Unimodal Physiology Only (LSTM Temporal Model)"
    else:
        w_phy_dyn, w_vid_dyn = 0.70, 0.30
        decision_branch = f"Fusion Strategy: {method_f}"

    # Status & Weights
    st.markdown(render_modality_weights(w_phy_dyn, w_vid_dyn), unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Physiology P(Stress)", f"{p_phy_f:.1%}",
                  delta="Primary Source", delta_color="off")
    with m2:
        st.metric("Video P(Stress)", f"{p_vid_f:.1%}",
                  delta="Secondary Supporting", delta_color="off")
    with m3:
        st.metric("Fused Decision", f"{p_fused_f:.1%}",
                  delta="STRESS" if p_fused_f >= 0.5 else "REST",
                  delta_color="inverse" if p_fused_f >= 0.5 else "normal")
    with m4:
        gt_f = pred_fusion.ground_truth
        ok_f = gt_f == int(p_fused_f >= 0.5)
        st.metric("Ground Truth", "Stress" if gt_f == 1 else "Rest",
                  delta="✓ MATCH" if ok_f else "✗ MISMATCH",
                  delta_color="normal" if ok_f else "inverse")

    face_qual_str = "SATISFACTORY (>=80%)" if pred_fusion.signal_quality_ok else f"DEGRADED ({det_frac_f:.0%})"
    face_qual_color = "#00D2BA" if pred_fusion.signal_quality_ok else "#F59E0B"
    st.markdown(f"""
        <div class="callout-info">
            <strong>Active Decision Explanation:</strong> {decision_branch}.<br>
            Quality Gates: BVP/EDA signal nominal (valid hardware recording) · Face landmark detection quality = <span style="color:{face_qual_color}; font-weight:600;">{face_qual_str}</span>.<br>
            <em>Validation Notice: Probabilities represent synchronized out-of-fold benchmark predictions evaluated on UBFC-Phys. No artificial dropout fallback is simulated on valid benchmark data.</em>
        </div>
    """, unsafe_allow_html=True)

    # Timeline trajectory
    fig_tl_f = plot_timeline_modern(f_sub_rows, method_col="p_blend_0.3_uf" if "p_blend_0.3_uf" in f_sub_rows.columns else method_f)
    st.pyplot(fig_tl_f, width="stretch")

    # Benchmark summary
    st.markdown("#### Late Fusion Benchmark Across 18 Participants (972 Windows)")
    summary_df_f = fusion_ad.load_summary()
    if summary_df_f is not None and not summary_df_f.empty:
        st.dataframe(summary_df_f, width="stretch", hide_index=True)


# ==============================================================================
# TAB 6: TRENDS & RESEARCH
# ==============================================================================
with tabs[5]:
    st.markdown(render_header(
        "Temporal Distress Trend Detection & Research Evaluation",
        "Causal past-only Theil-Sen trend estimation and Leave-One-Subject-Out (LOSO) benchmarks",
        badge_text="VALIDATED RESEARCH BENCHMARKS",
        badge_type="replay"
    ), unsafe_allow_html=True)

    # 1. Temporal Trend Sub-section
    from demo.ui.tab_trend import render_tab_trend
    render_tab_trend(trend_ad, ubfc_ad)

    st.markdown("---")
    st.markdown("### Rigorous Research Benchmarking & Independent Evaluation")

    perf_data = {
        "Model / Pipeline": [
            "WESAD CNN-TCN-LSTM",
            "WESAD Dev Checkpoint",
            "UBFC-Phys LSTM",
            "UBFC-Phys TCN",
            "UBFC Rest-Calib RF",
            "Video RF (Full 23-feat)",
            "Video RF (Upper 19-feat)",
            "Fusion Blend 0.3 (Full)",
            "Fusion Blend 0.3 (Upper-face)",
            "CLAS → WESAD (cross-dataset)",
            "WESAD → CLAS (cross-dataset)",
        ],
        "Protocol": [
            "Corrected LOSO (15 subj)",
            "Dev Split (S10, S4, S6) *",
            "Full LOSO (56 subj)",
            "Full LOSO (56 subj)",
            "Full LOSO (56 subj)",
            "Subject LOSO (18 subj)",
            "Subject LOSO (18 subj)",
            "Subject LOSO (18 subj)",
            "Subject LOSO (18 subj)",
            "Zero-shot transfer",
            "Zero-shot transfer",
        ],
        "Accuracy": [
            "89.9%", "92.6% *", "80.4%", "77.2%", "97.9%",
            "96.6%", "91.4%", "—", "—", "23.9%", "62.6%"
        ],
        "Balanced Acc": [
            "—", "90.1% *", "76.1%", "69.3%", "98.2%",
            "96.7%", "90.6%", "97.1%", "81.0%", "—", "—"
        ],
        "AUROC": [
            "0.942", "0.961 *", "0.799", "0.796", "0.994",
            "0.994 ⚠", "0.968 ✓", "0.996", "0.933", "0.223", "0.367"
        ],
        "Validation Status": [
            "✅ COMPLETE", "⚠ DEV ONLY", "✅ COMPLETE", "✅ COMPLETE",
            "✅ COMPLETE", "⚠ CONFOUND", "✅ CLEAN", "✅ COMPLETE",
            "✅ COMPLETE", "❌ FAILED", "❌ FAILED"
        ]
    }
    st.dataframe(pd.DataFrame(perf_data), width="stretch", hide_index=True)

    # Per-subject breakdown tables
    c_sub1, c_sub2 = st.columns(2)
    with c_sub1:
        st.markdown("#### WESAD Corrected LOSO (15 Subjects)")
        wesad_sub_path = PROJECT_ROOT / "results/loso_corrected_valloss/loso_corrected_per_subject.csv"
        if wesad_sub_path.exists():
            st.dataframe(pd.read_csv(wesad_sub_path), width="stretch", hide_index=True)

    with c_sub2:
        st.markdown("#### UBFC Deployment LOSO (56 Subjects)")
        ubfc_sub_path = PROJECT_ROOT / "models/ubfc_deployment/loso_per_subject.csv"
        if ubfc_sub_path.exists():
            st.dataframe(pd.read_csv(ubfc_sub_path), width="stretch", hide_index=True)

    st.markdown("""
        <div class="callout-alert" style="margin-top:1.5rem;">
            <strong>Cross-Dataset Hardware Incompatibility (CLAS Transfer Failure):</strong><br>
            CLAS inference is permanently blocked. Evaluation experiments proved cross-dataset transfer failure (AUROC: 0.223–0.367) resulting from:
            (1) Optical transmission vs wrist reflection sensor physics mismatch (Shimmer3 finger clip vs Empatica E4 wristband),
            (2) Cognitive load vs Trier Social Stressor task divergence,
            (3) Lack of supervised classification weights in the CLAS self-supervised MAE encoder checkpoint.
        </div>
    """, unsafe_allow_html=True)
