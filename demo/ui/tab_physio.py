"""Tab 1: Physiological Stress Prediction (UBFC-Phys & WESAD)."""
import time
from typing import Dict, Any
import numpy as np
import pandas as pd
import streamlit as st

from demo.adapters.ubfc_adapter import UBFCPhysInferenceAdapter
from demo.adapters.wesad_adapter import WESADInferenceAdapter
from demo.adapters.clas_adapter import CLASInferenceAdapter
from demo.ui.styles import render_prediction_badge
from demo.utils.visualization import plot_wesad_window, plot_ubfc_signals


def render_tab_physio(
    ubfc_adapter: UBFCPhysInferenceAdapter,
    wesad_adapter: WESADInferenceAdapter,
    clas_adapter: CLASInferenceAdapter
):
    """Render Tab 1: Physiological Stress Prediction."""
    st.markdown("### Physiological Stress Recognition Replay")
    st.caption("Demonstrating binary stress classification on recorded benchmark datasets using validated models.")

    dataset_choice = st.radio(
        "Select Physiological Dataset",
        ["UBFC-Phys (Multimodal BVP + EDA)", "WESAD (Wrist BVP)", "CLAS (Audit & Limitations)"],
        horizontal=True
    )

    if dataset_choice == "CLAS (Audit & Limitations)":
        render_clas_audit_panel(clas_adapter)
        return

    if dataset_choice == "UBFC-Phys (Multimodal BVP + EDA)":
        render_ubfc_panel(ubfc_adapter)
    else:
        render_wesad_panel(wesad_adapter)


def render_ubfc_panel(adapter: UBFCPhysInferenceAdapter):
    """Render UBFC-Phys data replay and OOF temporal model evaluation."""
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.2, 1.2, 1.6])

    with col_ctrl1:
        participants = adapter.get_participants()
        selected_p = st.selectbox("Participant (56 total)", participants, index=0)

    with col_ctrl2:
        models = adapter.get_models_available()
        default_idx = models.index("LSTM") if "LSTM" in models else 0
        selected_model = st.selectbox("Temporal Model", models, index=default_idx)

    indices, bvp_sub, eda_sub, y_sub, meta_sub, oof_sub = adapter.get_participant_windows(
        selected_p, model_name=selected_model
    )

    n_windows = len(indices)
    if n_windows == 0:
        st.warning(f"No windows available for Participant {selected_p} under model {selected_model}.")
        return

    with col_ctrl3:
        win_pos = st.slider("Window Index", min_value=0, max_value=n_windows - 1, value=0)

    # Current window details
    current_meta = meta_sub.iloc[win_pos].to_dict()
    current_bvp = bvp_sub[win_pos]
    current_eda = eda_sub[win_pos]
    global_idx = int(indices[win_pos])

    prediction_result = adapter.predict(
        window_data={"bvp": current_bvp, "eda": current_eda},
        metadata={
            "window_index": global_idx,
            "model_name": selected_model,
            "participant": selected_p,
            "task": current_meta["task"],
            "start_s": current_meta["start_s"],
            "end_s": current_meta["end_s"],
        }
    )

    # Metrics Display Cards
    st.markdown("---")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.metric(
            label="Model Prediction",
            value="Stress-related" if prediction_result.predicted_class == 1 else "Non-stress",
            delta=f"Class {prediction_result.predicted_class}",
            delta_color="inverse" if prediction_result.predicted_class == 1 else "normal"
        )

    with m_col2:
        prob1 = float(prediction_result.probabilities[1])
        st.metric(
            label="Class-1 Distress Probability",
            value=f"{prob1:.1%}",
            delta="Uncalibrated OOF",
            delta_color="off"
        )

    with m_col3:
        gt_val = prediction_result.ground_truth
        gt_text = "Stress (T2/T3)" if gt_val == 1 else "Non-stress (T1)"
        match_str = "MATCH" if gt_val == prediction_result.predicted_class else "MISMATCH"
        st.metric(
            label="True Dataset Label",
            value=gt_text,
            delta=match_str,
            delta_color="normal" if match_str == "MATCH" else "inverse"
        )

    with m_col4:
        task_id = current_meta["task"]
        task_label = adapter.TASK_NAMES.get(task_id, f"Task {task_id}")
        st.metric(
            label="Protocol Phase & Window",
            value=f"Task {task_id}",
            delta=f"{current_meta['start_s']:.0f}s – {current_meta['end_s']:.0f}s"
        )

    # Signal Waveform Plot
    fig = plot_ubfc_signals(current_bvp, current_eda)
    st.pyplot(fig)

    # Context and Warnings
    st.markdown(
        f"""
        <div class="research-disclaimer">
            <strong>Participant:</strong> #{selected_p} &nbsp;|&nbsp;
            <strong>Protocol Task:</strong> {task_label} &nbsp;|&nbsp;
            <strong>Window Span:</strong> {current_meta['start_s']:.1f}s to {current_meta['end_s']:.1f}s (10.0 s @ 64 Hz BVP / 4 Hz EDA) &nbsp;|&nbsp;
            <strong>Evaluation:</strong> Strict Leave-One-Subject-Out (LOSO) cross-validation.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Participant trajectory table
    with st.expander(f"Participant #{selected_p} Prediction Trajectory Across All Tasks"):
        traj_df = pd.DataFrame({
            "Window": range(len(oof_sub)),
            "Task": meta_sub["task"],
            "Time Range": [f"{r.start_s:.0f}s - {r.end_s:.0f}s" for _, r in meta_sub.iterrows()],
            "True Label": ["Stress" if y == 1 else "Rest" for y in y_sub],
            "Predicted": ["Stress" if p == 1 else "Rest" for p in oof_sub["prediction"]],
            "Distress Probability": [f"{p:.3f}" for p in oof_sub["probability"]],
            "Match": ["✓" if y == p else "✗" for y, p in zip(y_sub, oof_sub["prediction"])],
        })
        st.dataframe(traj_df, use_container_width=True, hide_index=True)


def render_wesad_panel(adapter: WESADInferenceAdapter):
    """Render WESAD live CNN-TCN-LSTM checkpoint inference and data replay."""
    if not adapter.is_model_available():
        st.error(f"WESAD checkpoint not found at {adapter.model_path}")
        return

    manifest = adapter.get_manifest() or {}
    val_subjects = set(manifest.get("validation_subjects", []))

    col_ctrl1, col_ctrl2 = st.columns([1.5, 2.5])

    with col_ctrl1:
        subjects = adapter.get_subject_list()
        selected_sub = st.selectbox(
            "Select WESAD Participant (15 available)",
            subjects,
            index=0,
            help="Participants S10, S4, and S6 were used for validation checkpoint selection."
        )

    indices, windows, labels = adapter.get_subject_windows(selected_sub)
    n_windows = len(indices)

    with col_ctrl2:
        win_pos = st.slider("Window Index (60-second windows)", 0, n_windows - 1, 0)

    current_window = windows[win_pos:win_pos + 1]
    true_label = int(labels[win_pos])

    # Run actual model inference via adapter
    prediction_result = adapter.predict(
        current_window,
        metadata={"subject_id": selected_sub, "window_index": int(indices[win_pos]), "ground_truth": true_label}
    )

    # Metrics Display
    st.markdown("---")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.metric(
            label="Model Prediction",
            value="Stress-related" if prediction_result.predicted_class == 1 else "Non-stress",
            delta=f"Class {prediction_result.predicted_class}",
            delta_color="inverse" if prediction_result.predicted_class == 1 else "normal"
        )

    with m_col2:
        prob1 = float(prediction_result.probabilities[1])
        st.metric(
            label="Predicted Stress Probability",
            value=f"{prob1:.1%}",
            delta="Softmax Output",
            delta_color="off"
        )

    with m_col3:
        gt_text = "Stress (TSST)" if true_label == 1 else "Non-stress"
        is_match = (true_label == prediction_result.predicted_class)
        st.metric(
            label="Ground Truth Label",
            value=gt_text,
            delta="MATCH" if is_match else "MISMATCH",
            delta_color="normal" if is_match else "inverse"
        )

    with m_col4:
        is_val = selected_sub in val_subjects
        st.metric(
            label="Cohort Status",
            value="Validation Split" if is_val else "Training Split",
            delta="Checkpoint Selector" if is_val else "Model Training Fold",
            delta_color="off"
        )

    # Waveform Plot
    fig = plot_wesad_window(current_window, fs=64.0)
    st.pyplot(fig)

    st.markdown(
        f"""
        <div class="research-disclaimer">
            <strong>Architecture:</strong> CNN–TCN–LSTM deployment checkpoint (classifier.keras) &nbsp;|&nbsp;
            <strong>Input Contract:</strong> 3840 samples (60 seconds @ 64 Hz BVP) &nbsp;|&nbsp;
            <strong>Best Epoch:</strong> {manifest.get('best_epoch', 'N/A')} &nbsp;|&nbsp;
            <strong>Best Val Loss:</strong> {manifest.get('best_validation_loss', 0.0):.4f} &nbsp;|&nbsp;
            <strong>Caution:</strong> Validation participants ({', '.join(val_subjects)}) were used for early stopping checkpoint selection.
            Independent test evaluation was performed via corrected LOSO (reported on Tab 5).
        </div>
        """,
        unsafe_allow_html=True
    )


def render_clas_audit_panel(adapter: CLASInferenceAdapter):
    """Render transparent audit of CLAS dataset and explain why inference is disabled."""
    summary = adapter.get_audit_summary()

    st.markdown("#### CLAS Dataset & Model Inventory Audit")
    st.markdown(
        """
        <div class="research-warning">
            <strong>Inference Deliberately Disabled:</strong>
            The repository contains self-supervised Masked Autoencoder (MAE) encoder weights for CLAS
            (<code>models/pretrained_clas_mae_encoder.weights.h5</code>), but <strong>no trained classification head exists</strong>.
            In accordance with project integrity standards, fabricated predictions are strictly prohibited.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        **Audited Findings:**
        - **Dataset Present:** `{summary['has_dataset']}` ({summary['dataset_path']})
        - **Pretrained Encoder Weights:** `{summary['has_encoder_weights']}`
        - **Architecture:** `{summary['encoder_architecture']}`
        - **Labels Used During MAE Pretraining:** `{summary['labels_used_in_training']}` (Self-supervised masked reconstruction only)
        - **Trained Classification Head:** `None (No checkpoint found)`
        """
    )

    st.markdown("#### Cross-Dataset Transfer Failure Analysis (From `docs/experiment_log.csv`)")
    st.markdown(
        """
        Prior experiments attempting zero-shot or cross-dataset transfer between WESAD and CLAS failed significantly:
        """
    )

    cross_df = pd.DataFrame([
        {
            "Experiment": "E2 (WESAD → CLAS)",
            "Train Dataset": "WESAD (Empatica E4 BVP)",
            "Test Dataset": "CLAS (Shimmer3 PPG)",
            "Accuracy": "62.6%",
            "Macro F1": "0.4026",
            "ROC-AUC": "0.3670 (Sub-random)",
            "Outcome": "Severe domain collapse due to sensor and protocol mismatch"
        },
        {
            "Experiment": "E3 (CLAS → WESAD)",
            "Train Dataset": "CLAS (Shimmer3 PPG)",
            "Test Dataset": "WESAD (Empatica E4 BVP)",
            "Accuracy": "23.9%",
            "Macro F1": "0.2176",
            "ROC-AUC": "0.2226 (Inverted ranking)",
            "Outcome": "Systematic polarity/signal inversion between photoplethysmography hardware"
        }
    ])
    st.dataframe(cross_df, use_container_width=True, hide_index=True)

