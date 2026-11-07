"""Streamlit panel that shows the physiology-first decision for one aligned window.

Used inside the Multimodal tab of app.py. All data is REPLAY (prerecorded).
"""
import streamlit as st

from src.aligned_replay import predictions_from_aligned_row
from src.decision_engine import decide
from src.prediction_schema import DecisionMode

TITLES = {
    DecisionMode.PHYSIOLOGY_PRIMARY: "Result determined by PHYSIOLOGY",
    DecisionMode.VIDEO_FALLBACK: "Result from VIDEO ONLY (fallback)",
    DecisionMode.INSUFFICIENT_DATA: "INSUFFICIENT DATA: no result",
}
LABELS = {"rest_task_proxy": "Non-stress (rest-task proxy)",
          "stress_task_proxy": "Stress-related (task proxy)"}


def _branch(title, pred, assessment):
    st.markdown(f"**{title}**")
    if pred is None or not pred.available:
        st.write("Not available")
    else:
        st.write(f"Prediction: {LABELS.get(pred.label, pred.label)}")
        for name, p in pred.probabilities.items():
            st.progress(float(p), text=f"{LABELS.get(name, name)}: {p:.0%}")
        if "quality_not_measured_in_replay" in pred.quality_flags:
            st.write("Signal quality: not measured in replay data (assumed nominal)")
        else:
            st.write(f"Signal quality: {pred.quality:.2f}")
        st.caption(f"Window {pred.window_ref} | Source: {pred.source.value.upper()}")
    if assessment is not None:
        if assessment.reliable:
            st.success("Reliable")
        else:
            st.warning("Not reliable: " + "; ".join(assessment.reasons))


def render_decision_panel(row, key_prefix: str = "decision") -> None:
    st.markdown("### Decision policy (physiology-first)")
    st.error("REPLAY: prerecorded UBFC-Phys out-of-fold predictions. Not live sensing.")
    c1, c2 = st.columns(2)
    sim_phys = c1.checkbox("Simulate physiology dropout", key=f"{key_prefix}_sim_phys")
    sim_vid = c2.checkbox("Simulate video face loss", key=f"{key_prefix}_sim_vid")

    phys, vid = predictions_from_aligned_row(
        row, simulate_physiology_dropout=sim_phys, simulate_video_loss=sim_vid)
    d = decide(phys, vid)

    st.subheader(TITLES[d.mode])
    if d.banner:
        st.warning(d.banner)
    m1, m2, m3 = st.columns([2, 1, 1])
    with m1:
        st.caption("Result")
        st.markdown(f"#### {LABELS.get(d.primary_label, 'No result')}")
    with m2:
        st.caption("Determined by")
        st.markdown(f"#### {d.determining_branch.value if d.determining_branch else 'none'}")
    with m3:
        st.caption("Data source")
        st.markdown(f"#### {d.source.value.upper()}")
    st.write(d.explanation)
    for w in d.warnings:
        st.warning(w)
    st.caption(f"Decision time (UTC): {d.timestamp}. Labels are task-derived proxies, "
               "not validated stress or distress labels.")
    left, right = st.columns(2)
    with left:
        _branch("Physiology (primary)", phys, d.physiology_assessment)
    with right:
        _branch("Video, upper-face (fallback, determining)" if d.is_fallback
                else "Video, upper-face (supporting)", vid, d.video_assessment)
    st.caption("The benchmark blend below is an offline fusion score. The decision above "
               "follows the physiology-first rule: video never overrides physiology.")
