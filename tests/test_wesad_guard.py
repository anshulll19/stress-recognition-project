"""Choosing WESAD without its data/TensorFlow must show a message, not a traceback."""
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
pytestmark = pytest.mark.skipif(not APP.exists(), reason="app.py missing")

WESAD_LABEL = "WESAD (Wrist BVP · Empatica E4 · 15 Subjects · 60s Windows)"


def test_wesad_missing_resources_gives_message_not_exception():
    from demo.adapters.wesad_adapter import WESADInferenceAdapter
    from src.availability import wesad_unavailable_reason
    if wesad_unavailable_reason(WESADInferenceAdapter()) is None:
        pytest.skip("WESAD data and TensorFlow are present on this machine")
    at = st_testing.AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    at.radio(key="physio_cohort_radio").set_value(WESAD_LABEL).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("WESAD replay is unavailable" in w.value for w in at.warning)
