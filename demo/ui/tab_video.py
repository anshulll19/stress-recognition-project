"""Tab 2: Video-Based Facial Distress Analysis."""
from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st

from demo.adapters.video_adapter import VideoInferenceAdapter
from demo.ui.styles import render_prediction_badge


def render_tab_video(video_adapter: VideoInferenceAdapter):
    """Render Tab 2: Video-Based Facial Distress Analysis."""
    st.markdown("### Video-Based Facial & Behavioral Analysis")
    st.caption("Offline video baseline and recorded facial feature window replay using Random Forest classifier.")

    if not video_adapter.is_model_available():
        st.error(f"Video baseline model or feature names missing from {video_adapter.model_path}")
        return

    mode = st.radio(
        "Demonstration Mode",
        ["Recorded Video Windows Replay (18 Participants)", "Live Webcam & MediaPipe Pipeline Diagnostics"],
        horizontal=True
    )

    if mode == "Recorded Video Windows Replay (18 Participants)":
        render_offline_replay(video_adapter)
    else:
        render_webcam_diagnostics(video_adapter)


def render_offline_replay(adapter: VideoInferenceAdapter):
    """Render replay of precomputed 23-feature video windows."""
    subjects = adapter.get_subjects()

    col_ctrl1, col_ctrl2 = st.columns([1.5, 2.5])
    with col_ctrl1:
        selected_sub = st.selectbox("Participant (18 available)", subjects, index=0)

    sub_windows = adapter.get_subject_windows(selected_sub)
    n_windows = len(sub_windows)

    with col_ctrl2:
        win_pos = st.slider("Video Window Index (10s windows, 5s stride)", 0, n_windows - 1, 0)

    current_row = sub_windows.iloc[win_pos]
    true_label = int(current_row.get("label", 0))
    detection_fraction = float(current_row.get("detection_fraction", 1.0))

    # Run model prediction on exact 23 features
    pred_res = adapter.predict(
        current_row,
        metadata={
            "subject_id": selected_sub,
            "session_id": current_row.get("session_id"),
            "window_idx": current_row.get("window_idx"),
            "ground_truth": true_label,
            "detection_fraction": detection_fraction,
        }
    )

    # Metric Cards
    st.markdown("---")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.metric(
            label="Model Prediction",
            value="Distress Proxy" if pred_res.predicted_class == 1 else "Rest Proxy",
            delta=f"Class {pred_res.predicted_class}",
            delta_color="inverse" if pred_res.predicted_class == 1 else "normal"
        )

    with m_col2:
        prob1 = float(pred_res.probabilities[1])
        st.metric(
            label="Distress Probability",
            value=f"{prob1:.1%}",
            delta="Tree Vote Ensemble",
            delta_color="off"
        )

    with m_col3:
        task_session = current_row.get("session_id", "Unknown")
        gt_text = "Stress Proxy" if true_label == 1 else "Rest Proxy"
        is_match = (true_label == pred_res.predicted_class)
        st.metric(
            label=f"Dataset Label ({task_session})",
            value=gt_text,
            delta="MATCH" if is_match else "MISMATCH",
            delta_color="normal" if is_match else "inverse"
        )

    with m_col4:
        st.metric(
            label="Face Detection Quality",
            value=f"{detection_fraction:.1%}",
            delta="Reliable" if detection_fraction >= 0.8 else "Degraded",
            delta_color="normal" if detection_fraction >= 0.8 else "inverse"
        )

    # 23-Feature Value Chart
    st.markdown("#### Active Window Feature Profile (23 Features)")
    feats = adapter.feature_names
    feat_values = [float(current_row[f]) for f in feats]

    feat_df = pd.DataFrame({"Feature": feats, "Value": feat_values})

    # Group into categories for clean display
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.caption("Motion & Eye Features")
        st.dataframe(feat_df.iloc[:8], use_container_width=True, hide_index=True)
    with col_f2:
        st.caption("Mouth, Brow & Head-Pose Proxies")
        st.dataframe(feat_df.iloc[8:], use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="research-warning">
            <strong>Critical Research Notice (Confound Audit):</strong><br>
            The video model achieves 96.7% balanced accuracy on UBFC-Phys, but feature importance analysis shows
            that over 70% of model weight rests on mouth opening/width and head motion differences between
            the speech task (T2) and silent rest (T1). <strong>Do not interpret this output as validated internal
            psychological stress or distress.</strong>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_webcam_diagnostics(adapter: VideoInferenceAdapter):
    """Render diagnostic pipeline verification for live webcam processing."""
    st.markdown("#### Live Webcam Processing Pipeline Diagnostics")
    st.markdown(
        """
        The live webcam pipeline uses Google MediaPipe Face Landmarker (`models/face_landmarker.task`)
        to compute 23 geometrical features across a rolling 10-second buffer.
        """
    )

    landmarker_exists = adapter.landmarker_path.exists()

    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.markdown(f"**MediaPipe Landmarker Task (`models/face_landmarker.task`):** {'`READY (3.7 MB)`' if landmarker_exists else '`MISSING`'}")
        st.markdown(f"**Trained Classifier (`video_distress_baseline.joblib`):** `READY (Random Forest, 300 Trees)`")
        st.markdown(f"**Expected Feature Dimension:** `23 Features`")
    with d_col2:
        st.markdown(f"**Window Contract:** `10.0 seconds`")
        st.markdown(f"**Minimum Valid Frames:** `20 frames`")
        st.markdown(f"**Minimum Face Detection Fraction:** `80.0%`")

    st.markdown("---")
    st.markdown("#### Camera Input Test")
    st.caption("Capture a snapshot from your local camera to test face landmark detection and geometry calculation:")

    camera_image = st.camera_input("Test Camera Frame")

    if camera_image is not None:
        import cv2
        import mediapipe as mp

        bytes_data = camera_image.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

        # Run MediaPipe detector
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(adapter.landmarker_path)),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1
        )

        with FaceLandmarker.create_from_options(options) as detector:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
            result = detector.detect(mp_image)

            if result.face_landmarks:
                st.success("Face Detected Successfully!")
                landmarks = result.face_landmarks[0]
                h, w = rgb_img.shape[:2]

                # Draw landmark mesh
                annotated = rgb_img.copy()
                for lm in landmarks[::5]:  # subsample for clean visualization
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(annotated, (x, y), 2, (0, 255, 0), -1)

                st.image(annotated, caption="Detected Face Landmarks", use_container_width=True)

                st.info(
                    "Face geometry detected. Live streaming inference requires accumulating a continuous "
                    "10-second rolling buffer (>=20 frames with >=80% face fraction) before a valid prediction "
                    "can be generated. For instant reproducible evaluation, use the Recorded Video Windows Replay tab."
                )
            else:
                st.warning("No face detected in the captured frame. Ensure adequate lighting and face alignment.")

