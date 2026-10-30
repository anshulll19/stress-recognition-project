"""UI components package for multimodal stress recognition dashboard."""
from demo.ui.styles import DARK_THEME_CSS, render_prediction_badge, render_trend_badge
from demo.ui.tab_physio import render_tab_physio
from demo.ui.tab_video import render_tab_video
from demo.ui.tab_fusion import render_tab_fusion
from demo.ui.tab_trend import render_tab_trend
from demo.ui.tab_eval import render_tab_eval

__all__ = [
    "DARK_THEME_CSS",
    "render_prediction_badge",
    "render_trend_badge",
    "render_tab_physio",
    "render_tab_video",
    "render_tab_fusion",
    "render_tab_trend",
    "render_tab_eval",
]

