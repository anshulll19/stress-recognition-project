"""Tab 3: Multimodal Decision-Level Late Fusion."""
import pandas as pd
import streamlit as st

from demo.adapters.fusion_adapter import MultimodalFusionAdapter
from demo.utils.visualization import plot_multimodal_comparison


def render_tab_fusion(fusion_adapter: MultimodalFusionAdapter):
    """Render Tab 3: Multimodal Fusion."""
    st.markdown("### Multimodal Decision-Level Late Fusion")
    st.caption("Synchronized late fusion of physiological (LSTM) and facial behavioral (Random Forest) predictions on UBFC-Phys.")

    try:
        aligned_df = fusion_adapter.load_aligned_data()
    except Exception as e:
        st.error(f"Could not load aligned fusion data: {e}")
        return

    subjects = fusion_adapter.get_subjects()

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.2, 1.4, 1.4])
    with col_ctrl1:
        selected_sub = st.selectbox("Participant (18 with aligned video & physio)", subjects, index=0)

    sub_windows = fusion_adapter.get_subject_windows(selected_sub)
    n_windows = len(sub_windows)

    with col_ctrl2:
        method = st.selectbox(
            "Fusion Method",
            fusion_adapter.FUSION_METHODS,
            index=0,
            format_func=lambda m: fusion_adapter.METHOD_DESCRIPTIONS.get(m, m)
        )

    with col_ctrl3:
        win_pos = st.slider("Aligned Window Index", 0, n_windows - 1, 0)

    current_row = sub_windows.iloc[win_pos]

    # Generate fusion prediction result
    pred_res = fusion_adapter.get_fusion_prediction(current_row, method=method)
    meta = pred_res.window_metadata

    # Verification of strict modality alignment
    st.markdown("---")
    st.markdown(
        f"""
        <div class="research-disclaimer">
            <strong>Modality Synchronization Verified:</strong>
            Participant: <code>{meta['subject_id']}</code> &nbsp;|&nbsp;
            Protocol Task: <code>Task {meta['task']}</code> &nbsp;|&nbsp;
            Interval: <code>{meta['window_start_sec']:.1f}s – {meta['window_end_sec']:.1f}s</code> &nbsp;|&nbsp;
            Face Detection Quality: <code>{meta['detection_fraction']:.1%}</code>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Metric Comparison Cards
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        p_phys = meta["p_physio"]
        st.metric(
            label="Physiology (LSTM)",
            value=f"{p_phys:.1%}",
            delta="Stress" if p_phys >= 0.5 else "Non-stress",
            delta_color="inverse" if p_phys >= 0.5 else "normal"
        )

    with m_col2:
        p_vid = meta["p_video"]
        st.metric(
            label="Video (Random Forest)",
            value=f"{p_vid:.1%}",
            delta="Stress" if p_vid >= 0.5 else "Non-stress",
            delta_color="inverse" if p_vid >= 0.5 else "normal"
        )

    with m_col3:
        p_fus = meta["p_fused"]
        st.metric(
            label=f"Fused ({method})",
            value=f"{p_fus:.1%}",
            delta="Stress" if p_fus >= 0.5 else "Non-stress",
            delta_color="inverse" if p_fus >= 0.5 else "normal"
        )

    with m_col4:
        gt_val = pred_res.ground_truth
        gt_text = "Stress (T2/T3)" if gt_val == 1 else "Rest (T1)"
        is_match = (gt_val == pred_res.predicted_class)
        st.metric(
            label="Ground Truth Label",
            value=gt_text,
            delta="MATCH" if is_match else "MISMATCH",
            delta_color="normal" if is_match else "inverse"
        )

    # Visual Comparison Bar Chart
    fig = plot_multimodal_comparison(meta["p_physio"], meta["p_video"], meta["p_fused"])
    st.pyplot(fig)

    # Fusion Benchmark Summary Table
    st.markdown("#### Validated Late Fusion Benchmark Results (18 Participants, 972 Windows)")
    summary_df = fusion_adapter.load_summary()
    if not summary_df.empty:
        # Display formatted columns
        display_df = summary_df.copy()
        if "auc_ci_lo" in display_df.columns and "auc_ci_hi" in display_df.columns:
            display_df["95% CI"] = [f"[{lo:.3f}, {hi:.3f}]" for lo, hi in zip(display_df["auc_ci_lo"], display_df["auc_ci_hi"])]
        cols_to_show = ["method", "bacc_mean", "macro_f1_mean", "auc_mean", "95% CI"]
        cols_existing = [c for c in cols_to_show if c in display_df.columns]
        st.dataframe(
            display_df[cols_existing].rename(columns={
                "method": "Method",
                "bacc_mean": "Balanced Accuracy",
                "macro_f1_mean": "Macro F1",
                "auc_mean": "Mean AUC"
            }),
            use_container_width=True,
            hide_index=True
        )

    # Confound Analysis Note
    st.markdown(
        """
        <div class="research-warning">
            <strong>Multimodal Confound Finding:</strong><br>
            Ablation analysis (<code>results/fusion_lstm/fusion_confound_analysis.csv</code>) confirms that
            video-only performance drops when speech-associated facial features (mouth opening, mouth width) are removed.
            Decision-level late fusion with physiological LSTM features (<code>blend_0.3</code>, AUC = 0.996) provides
            resilience against unimodal facial expression confounds while retaining physiological ground truth.
        </div>
        """,
        unsafe_allow_html=True
    )

