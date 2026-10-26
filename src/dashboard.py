
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "ubfc_pipeline" / "out" / "features_full" / "features.csv"

st.set_page_config(
    page_title="Multimodal Stress Research",
    page_icon="📊",
    layout="wide",
)

st.title("Multimodal Stress Research Platform")
st.caption("Physiological and facial-video research dashboard")

st.warning(
    "Experimental pilot data: one participant, one task (T1). "
    "This dashboard does not diagnose stress or provide a validated stress prediction."
)

if not FEATURES_PATH.is_file():
    st.error(f"Pilot feature table not found: {FEATURES_PATH}")
    st.stop()

df = pd.read_csv(FEATURES_PATH)

if df.empty:
    st.error("The pilot feature table contains no windows.")
    st.stop()

st.sidebar.header("Pilot data")
window_idx = st.sidebar.selectbox(
    "Select a 30-second window",
    options=list(range(len(df))),
    format_func=lambda i: (
        f"Window {i}: {df.iloc[i]['t0']:.0f}–{df.iloc[i]['t1']:.0f} s"
    ),
)

row = df.iloc[window_idx]

st.subheader(
    f"Window {window_idx} · {row['t0']:.0f}–{row['t1']:.0f} seconds"
)

# Main physiological measurements
st.markdown("### Physiological features")
c1, c2, c3, c4 = st.columns(4)

c1.metric("Mean heart rate", f"{row['p_hr_mean']:.1f} bpm")
c2.metric("SDNN", f"{row['p_sdnn']:.2f}")
c3.metric("RMSSD", f"{row['p_rmssd']:.2f}")
c4.metric("EDA standard deviation", f"{row['p_eda_std']:.5f}")

st.caption(
    "HRV and EDA values are algorithm-derived measurements; "
    "their validity has not been established by this single-participant pilot."
)

# Video features
st.markdown("### Facial and head-movement features")
v1, v2, v3, v4 = st.columns(4)

v1.metric("Brow-down mean", f"{row['v_brow_down_mean']:.4f}")
v2.metric("Eye-squint mean", f"{row['v_eye_squint_mean']:.4f}")
v3.metric("Mouth-press mean", f"{row['v_mouth_press_mean']:.4f}")
v4.metric("Head speed", f"{row['v_head_speed']:.4f}")

# Quality and consistency
st.markdown("### Signal quality indicators")
q1, q2, q3 = st.columns(3)

q1.metric("Face detection fraction", f"{row['q_face_frac']:.1%}")
q2.metric("Valid IBI fraction", f"{row['q_ibi_valid_frac']:.1%}")
q3.metric("BVP vs video HR difference", f"{row['q_hr_absdiff']:.1f} bpm")

st.caption(
    "These are pipeline-specific quality indicators, not a complete measure "
    "of signal quality or proof of synchronization."
)

# Optional trends across the pilot
st.markdown("### Measurements across pilot windows")
chart_cols = st.multiselect(
    "Choose measurements to plot",
    options=["p_hr_mean", "p_rmssd", "p_eda_std", "v_blink_rate", "v_head_speed"],
    default=["p_hr_mean"],
)

if chart_cols:
    chart_df = df[["t0"] + chart_cols].set_index("t0")
    st.line_chart(chart_df, height=300)

# Raw feature values
with st.expander("Inspect all features for this window"):
    feature_groups = {
        "Physiological": [c for c in df.columns if c.startswith("p_")],
        "Video": [c for c in df.columns if c.startswith("v_")],
        "Quality": [c for c in df.columns if c.startswith("q_")],
    }

    for group, columns in feature_groups.items():
        st.markdown(f"**{group}**")
        st.dataframe(
            pd.DataFrame({
                "Feature": columns,
                "Value": [row[c] for c in columns],
            }),
            hide_index=True,
            use_container_width=True,
        )

st.divider()
st.info(
    "Stress prediction: unavailable. The current pilot is not sufficient "
    "to establish a validated stress classifier or distress trend."
)

st.caption("Research prototype · Local development only")
