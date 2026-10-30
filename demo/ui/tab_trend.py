"""Tab 4: Temporal Distress Trend Analysis & Explainability."""
import numpy as np
import pandas as pd
import streamlit as st

from demo.adapters.trend_adapter import TemporalTrendAdapter, TrendConfig, LABELS, LABEL_COLORS
from demo.adapters.ubfc_adapter import UBFCPhysInferenceAdapter
from demo.ui.styles import render_trend_badge
from demo.utils.visualization import plot_trend_series


def render_tab_trend(trend_adapter: TemporalTrendAdapter, ubfc_adapter: UBFCPhysInferenceAdapter):
    """Render Tab 4: Temporal Trend Analysis."""
    st.markdown("### Temporal Distress Trend Prediction")
    st.caption("Causal, past-only trend estimation using Theil-Sen slope on model distress probabilities.")

    st.markdown(
        """
        <div class="research-warning">
            <strong>Prototype Status & Research Caveats:</strong><br>
            This temporal trend detector is an active research prototype. While all mechanical unit checks pass
            (causality, no future imputation, cold-start handling, and safe anti-flicker hysteresis),
            several synthetic benchmark acceptance criteria failed during rigorous evaluation
            (including gradual-ramp sensitivity, fluctuating signal recognition, and abrupt step recovery).
            <strong>The detector is displayed here for experimental inspection with strict safeguards.</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

    data_source = st.radio(
        "Trend Evaluation Source",
        ["Real UBFC-Phys Model Trajectory (56 Participants)", "Synthetic Verification Case (Planted Pattern)"],
        horizontal=True
    )

    if data_source == "Real UBFC-Phys Model Trajectory (56 Participants)":
        render_real_ubfc_trend(trend_adapter, ubfc_adapter)
    else:
        render_synthetic_trend(trend_adapter)


def render_real_ubfc_trend(trend_adapter: TemporalTrendAdapter, ubfc_adapter: UBFCPhysInferenceAdapter):
    """Render trend evaluation on real out-of-fold probability trajectories."""
    participants = ubfc_adapter.get_participants()
    models = ubfc_adapter.get_models_available()

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.2, 1.2, 1.6])
    with col_ctrl1:
        selected_p = st.selectbox("Participant", participants, index=0, key="trend_p")
    with col_ctrl2:
        selected_model = st.selectbox("Model", models, index=0, key="trend_m")

    _, _, _, y_sub, meta_sub, oof_sub = ubfc_adapter.get_participant_windows(
        selected_p, model_name=selected_model
    )

    probs = oof_sub["probability"].to_numpy()
    n_windows = len(probs)

    # Evaluate full trajectory with causal adapter
    trend_results = trend_adapter.evaluate_series(probs)

    with col_ctrl3:
        win_pos = st.slider("Time Step (Window Index)", 0, n_windows - 1, min(15, n_windows - 1))

    current_label = trend_results["displayed_label"][win_pos]
    raw_label = trend_results["raw_label"][win_pos]
    slope_val = trend_results["slope"][win_pos]
    is_suff = trend_results["is_sufficient"][win_pos]

    # Status Cards
    st.markdown("---")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.markdown(f"**Displayed Trend:**<br>{render_trend_badge(current_label)}", unsafe_allow_html=True)

    with m_col2:
        if is_suff and np.isfinite(slope_val):
            st.metric(
                label="Theil-Sen Slope",
                value=f"{slope_val:+.2f} logit/60s",
                delta=f"90% CI: [{trend_results['lo'][win_pos]:+.2f}, {trend_results['hi'][win_pos]:+.2f}]"
            )
        else:
            st.metric(label="Theil-Sen Slope", value="N/A", delta="Cold Start (< 12 windows)")

    with m_col3:
        st.metric(
            label="Data Sufficiency",
            value="Sufficient" if is_suff else "Insufficient Data",
            delta=f"{trend_results['n_valid'][win_pos]} / {trend_adapter.cfg.K} valid windows",
            delta_color="normal" if is_suff else "inverse"
        )

    with m_col4:
        task_id = meta_sub.iloc[win_pos]["task"]
        st.metric(
            label="Protocol Phase",
            value=f"Task {task_id}",
            delta="Rest" if task_id == 1 else "Stress Protocol"
        )

    # Trend Plot
    fig = plot_trend_series(probs, win_pos, trend_results, K=trend_adapter.cfg.K)
    st.pyplot(fig)

    # Diagnostics Expander
    with st.expander("Detailed Trend Diagnostic Trajectory (All Windows)"):
        diag_df = pd.DataFrame({
            "Window": range(n_windows),
            "Task": meta_sub["task"],
            "Probability": [f"{p:.3f}" for p in probs],
            "Data Sufficient": ["Yes" if s else "No" for s in trend_results["is_sufficient"]],
            "Theil-Sen Slope (logit/60s)": [f"{s:+.2f}" if np.isfinite(s) else "N/A" for s in trend_results["slope"]],
            "Raw Classification": trend_results["raw_label"],
            "Displayed Label (Safe Hysteresis)": trend_results["displayed_label"],
        })
        st.dataframe(diag_df, use_container_width=True, hide_index=True)


def render_synthetic_trend(trend_adapter: TemporalTrendAdapter):
    """Render synthetic verification patterns from Phase 2 prototype."""
    st.markdown("#### Synthetic Benchmark Patterns")
    pattern_choice = st.selectbox(
        "Select Planted Pattern",
        ["Strong Increasing Ramp (Distress Onset)", "Strong Decreasing Ramp (Recovery)", "Flat Stationary Series (Non-stress Baseline)"]
    )

    n_points = 35
    t = np.arange(n_points)

    if "Increasing" in pattern_choice:
        # Logistic ramp from 0.1 to 0.9
        logits = -2.0 + 4.0 * (t / n_points)
        probs = 1.0 / (1.0 + np.exp(-logits))
    elif "Decreasing" in pattern_choice:
        logits = 2.0 - 4.0 * (t / n_points)
        probs = 1.0 / (1.0 + np.exp(-logits))
    else:
        probs = np.full(n_points, 0.15) + np.random.default_rng(42).normal(0, 0.02, n_points)
        probs = np.clip(probs, 0.01, 0.99)

    trend_results = trend_adapter.evaluate_series(probs)
    win_pos = st.slider("Select Window Index along Pattern", 0, n_points - 1, 20)

    fig = plot_trend_series(probs, win_pos, trend_results, K=trend_adapter.cfg.K)
    st.pyplot(fig)

    st.markdown(
        f"""
        **Current Window Classification:** {render_trend_badge(trend_results['displayed_label'][win_pos])} &nbsp;|&nbsp;
        **Slope:** `{trend_results['slope'][win_pos]:+.2f} logit/60s` &nbsp;|&nbsp;
        **Sufficient Data:** `{trend_results['is_sufficient'][win_pos]}`
        """,
        unsafe_allow_html=True
    )

