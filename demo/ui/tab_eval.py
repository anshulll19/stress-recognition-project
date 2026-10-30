"""Tab 5: Evaluation and Research Results."""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

from demo.utils.metrics import ResearchMetricsLoader
from demo.utils.visualization import plot_confusion_matrix


def render_tab_eval(metrics_loader: ResearchMetricsLoader):
    """Render Tab 5: Evaluation and Research Results."""
    st.markdown("### Research Benchmarks & Evaluation Audit")
    st.caption("Verifiable performance metrics across datasets, evaluation protocols, and ablation experiments.")

    eval_section = st.selectbox(
        "Select Research Evaluation Suite",
        [
            "1. UBFC-Phys Physiology Models (LOSO)",
            "2. WESAD CNN-TCN-LSTM (Corrected LOSO vs Validation Split)",
            "3. Multimodal Decision Late Fusion (18 Participants)",
            "4. Video Confound & Feature Ablation Analysis",
            "5. Cross-Dataset Generalization (WESAD ↔ CLAS Domain Shift)",
        ]
    )

    if eval_section == "1. UBFC-Phys Physiology Models (LOSO)":
        render_ubfc_eval(metrics_loader)
    elif eval_section == "2. WESAD CNN-TCN-LSTM (Corrected LOSO vs Validation Split)":
        render_wesad_eval(metrics_loader)
    elif eval_section == "3. Multimodal Decision Late Fusion (18 Participants)":
        render_fusion_eval(metrics_loader)
    elif eval_section == "4. Video Confound & Feature Ablation Analysis":
        render_confound_eval(metrics_loader)
    else:
        render_cross_dataset_eval(metrics_loader)


def render_ubfc_eval(metrics_loader: ResearchMetricsLoader):
    """Render UBFC-Phys benchmark metrics and confusion matrices."""
    st.markdown("#### UBFC-Phys Physiological Model Comparison")
    st.markdown(
        """
        All models below were evaluated strictly with Leave-One-Subject-Out (LOSO) cross-validation
        across all 56 participants. No participant was present in both training and test sets.
        """
    )

    metrics_df = metrics_loader.get_ubfc_temporal_metrics()
    if not metrics_df.empty:
        # Format table
        st.dataframe(
            metrics_df.style.format({
                "Accuracy": "{:.2%}",
                "Balanced Accuracy": "{:.2%}",
                "Macro F1": "{:.4f}",
                "ROC-AUC": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")
    st.markdown("#### Out-of-Fold Confusion Matrices (5,880 Windows)")
    c1, c2 = st.columns(2)

    # Actual full confusion matrices from results/temporal_models/tcn_lstm_oof_predictions_full.csv
    # LSTM: [[1242, 718], [436, 3484]]
    # TCN: [[891, 1069], [269, 3651]]
    with c1:
        cm_lstm = np.array([[1242, 718], [436, 3484]])
        fig_l = plot_confusion_matrix(cm_lstm, ["Rest (T1)", "Stress (T2/T3)"], title="UBFC LSTM Full LOSO")
        st.pyplot(fig_l)

    with c2:
        cm_tcn = np.array([[891, 1069], [269, 3651]])
        fig_t = plot_confusion_matrix(cm_tcn, ["Rest (T1)", "Stress (T2/T3)"], title="UBFC TCN Full LOSO")
        st.pyplot(fig_t)

    st.markdown(
        """
        <div class="research-disclaimer">
            <strong>Key Insight:</strong> Rest-calibration of BVP and EDA features (subtracting individual baseline rest)
            achieved the highest balanced accuracy (91.07%) and macro-F1 (0.8811), confirming that physiological
            intersubject baseline variance is a primary source of classification error in raw time-series models.
        </div>
        """,
        unsafe_allow_html=True
    )


def render_wesad_eval(metrics_loader: ResearchMetricsLoader):
    """Render WESAD corrected LOSO performance vs validation split."""
    wesad_data = metrics_loader.get_wesad_metrics()

    st.markdown("#### WESAD Corrected Leave-One-Subject-Out (LOSO) Evaluation")
    st.markdown(
        """
        The WESAD dataset contains 812 60-second BVP windows (632 non-stress, 180 stress) across 15 participants.
        Below is the true independent Leave-One-Subject-Out cross-validation performance:
        """
    )

    loso_summary = wesad_data.get("loso_summary", {}).get("overall", {})
    if loso_summary:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("LOSO Accuracy", f"{loso_summary.get('accuracy', 0.899):.1%}")
        m2.metric("Weighted F1", f"{loso_summary.get('f1_weighted', 0.898):.4f}")
        m3.metric("Stress F1 (Class 1)", f"{loso_summary.get('f1_class1', 0.768):.4f}")
        m4.metric("ROC-AUC", f"{loso_summary.get('auc', 0.942):.4f}")

    st.markdown("---")
    st.markdown("#### Protocol Distinction: Independent LOSO vs. Checkpoint Selection Validation")

    comp_df = pd.DataFrame([
        {
            "Protocol": "Corrected LOSO (Independent Test)",
            "Participants": "15 Subjects (All held out one-by-one)",
            "Windows": "812 Total",
            "Accuracy": "89.90%",
            "Weighted F1": "0.8984",
            "ROC-AUC": "0.9424",
            "Purpose": "Unbiased generalization estimate across the full cohort"
        },
        {
            "Protocol": "Deployment Validation Split (models/deployment_candidate)",
            "Participants": "3 Subjects (S10, S4, S6)",
            "Windows": "163 Validation Windows",
            "Accuracy": "Best Val Loss: 0.3092 (Epoch 27)",
            "Weighted F1": "N/A (Early stopping criterion)",
            "ROC-AUC": "N/A",
            "Purpose": "Used strictly for early stopping & weight selection; NOT an independent test"
        }
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Per-subject breakdown
    per_sub_df = wesad_data.get("loso_per_subject")
    if per_sub_df is not None and not per_sub_df.empty:
        with st.expander("Per-Subject LOSO Metrics (15 Participants)"):
            st.dataframe(per_sub_df, use_container_width=True, hide_index=True)


def render_fusion_eval(metrics_loader: ResearchMetricsLoader):
    """Render multimodal decision fusion results."""
    st.markdown("#### Multimodal Decision-Level Late Fusion Benchmark")
    st.markdown(
        """
        Evaluated on 18 UBFC-Phys participants (972 windows) with synchronized physiological (LSTM)
        and facial video (Random Forest) out-of-fold predictions.
        """
    )

    fusion_dict = metrics_loader.get_fusion_metrics()
    summary_df = fusion_dict.get("summary")

    if summary_df is not None and not summary_df.empty:
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    per_sub_df = fusion_dict.get("per_subject")
    if per_sub_df is not None and not per_sub_df.empty:
        with st.expander("Per-Participant Fusion AUC Breakdown (18 Participants)"):
            st.dataframe(per_sub_df, use_container_width=True, hide_index=True)


def render_confound_eval(metrics_loader: ResearchMetricsLoader):
    """Render video feature ablation and confound analysis."""
    st.markdown("#### Video Confound & Feature Set Ablation")
    st.markdown(
        """
        To investigate whether the high video performance (96.5% balanced accuracy) reflects true distress
        or merely speech activity during Task 2 (Public Speaking), ablations across feature subsets were conducted:
        """
    )

    fusion_dict = metrics_loader.get_fusion_metrics()
    confound_df = fusion_dict.get("confounds")

    if confound_df is not None and not confound_df.empty:
        st.dataframe(confound_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="research-warning">
            <strong>Key Finding:</strong> Removing mouth opening/width features drops video performance significantly,
            proving that video-based classification is heavily driven by speech kinematics rather than pure affective distress.
        </div>
        """,
        unsafe_allow_html=True
    )


def render_cross_dataset_eval(metrics_loader: ResearchMetricsLoader):
    """Render cross-dataset generalization findings (WESAD ↔ CLAS)."""
    st.markdown("#### Cross-Dataset Generalization & Transfer Domain Shift")
    st.markdown(
        """
        Cross-dataset transfer was evaluated between WESAD (Empatica E4 wrist BVP) and CLAS (Shimmer3 finger PPG).
        Both datasets used 60-second windows at 64 Hz, but differing sensor physics and experimental protocols
        led to significant domain collapse:
        """
    )

    cross_data = pd.DataFrame([
        {
            "Experiment": "E2 (WESAD Train → CLAS Test)",
            "Train Data": "WESAD (15 subjects, 812 windows)",
            "Test Data": "CLAS (53 subjects, 2271 windows)",
            "Test Accuracy": "62.65%",
            "Test Macro F1": "0.4026",
            "Test ROC-AUC": "0.3670",
            "Failure Mode": "Sub-random AUC (< 0.5): domain shift between wrist BVP and finger PPG"
        },
        {
            "Experiment": "E3 (CLAS Train → WESAD Test)",
            "Train Data": "CLAS (53 subjects, 2271 windows)",
            "Test Data": "WESAD (15 subjects, 812 windows)",
            "Test Accuracy": "23.89%",
            "Test Macro F1": "0.2176",
            "Test ROC-AUC": "0.2226",
            "Failure Mode": "Systematic ranking inversion: Shimmer3 sensor PPG polarity was inverted relative to Empatica E4"
        }
    ])
    st.dataframe(cross_data, use_container_width=True, hide_index=True)

