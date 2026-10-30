"""Checks that let the dashboard show a clear message instead of a traceback
when an optional dataset or library is missing on this machine."""
import importlib.util
from typing import Optional


def wesad_unavailable_reason(wesad_adapter) -> Optional[str]:
    """Return a short human-readable reason the WESAD replay cannot run, or None."""
    problems = []
    if not wesad_adapter.data_path.exists():
        problems.append(f"the dataset file is missing ({wesad_adapter.data_path.name})")
    if importlib.util.find_spec("tensorflow") is None and importlib.util.find_spec("keras") is None:
        problems.append("TensorFlow/Keras is not installed")
    if not problems:
        return None
    return "WESAD replay is unavailable on this machine: " + " and ".join(problems) + \
           ". UBFC-Phys replay and the decision policy are unaffected."
