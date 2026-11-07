"""Smoke test: the real app.py still renders all six tabs and shows the decision panel."""
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
pytestmark = pytest.mark.skipif(not APP.exists(), reason="app.py missing")


def run_app():
    at = st_testing.AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    return at


def test_app_runs_with_six_tabs_and_decision_panel():
    at = run_app()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 6
    assert any("REPLAY" in e.value and "Not live sensing" in e.value for e in at.error)
    assert any("Result determined by PHYSIOLOGY" in s.value for s in at.subheader)


def test_physiology_dropout_switches_to_fallback_banner():
    at = run_app()
    at.checkbox(key="fusion_decision_sim_phys").check().run()
    assert not at.exception
    assert any("VIDEO ONLY" in s.value for s in at.subheader)
    assert any("VIDEO-ONLY FALLBACK" in w.value for w in at.warning)


def test_both_dropouts_give_insufficient_data():
    at = run_app()
    at.checkbox(key="fusion_decision_sim_phys").check()
    at.checkbox(key="fusion_decision_sim_vid").check().run()
    assert not at.exception
    assert any("INSUFFICIENT DATA" in s.value for s in at.subheader)
