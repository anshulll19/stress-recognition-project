"""
demo/ui/modern_theme.py
=======================
Modern Healthcare Analytics Dark Theme & Visual Component Library.
Provides CSS and reusable HTML widgets for a professional, clinical-grade
multimodal stress monitoring dashboard.

Color Palette:
  - Base Background: #0B0F19 (Deep Charcoal Navy)
  - Card Surfaces:   #111827 (Elevation 1), #1E293B (Borders)
  - Primary Accent:  #00D2BA (Teal - Health/Baseline/Live)
  - Secondary Accent:#06B6D4 (Cyan - Signals/Waveforms)
  - Tertiary Accent: #8B5CF6 (Violet - Multimodal Fusion/AI)
  - Stress/Alert:    #F43F5E (Coral Red - Stress Detected)
  - Warning/Dropout: #F59E0B (Amber - Quality Dropout)
  - Text Primary:    #F9FAFB
  - Text Secondary:  #9CA3AF
  - Text Muted:      #6B7280
"""

from typing import Optional


HEALTHCARE_DARK_CSS = """
<style>
/* ============================================================
   GOOGLE FONTS & BASE STYLING
   ============================================================ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

html, body, [class*="st-emotion"], .stApp {
    background-color: #0B0F19 !important;
    color: #F3F4F6 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Hide Streamlit default branding & header decoration */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Container spacing */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1380px !important;
}

/* ============================================================
   SIDEBAR NAVIGATION
   ============================================================ */
section[data-testid="stSidebar"] {
    background-color: #0E1526 !important;
    border-right: 1px solid #1E293B !important;
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #F9FAFB !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] .stMarkdown p {
    color: #9CA3AF !important;
    font-size: 0.85rem !important;
}

/* ============================================================
   TABS NAVIGATION
   ============================================================ */
.stTabs [data-baseweb="tab-list"] {
    background-color: #111827;
    border-radius: 12px;
    padding: 6px;
    gap: 6px;
    border: 1px solid #1E293B;
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    border-radius: 8px !important;
    color: #9CA3AF !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    border: none !important;
    transition: all 0.2s ease !important;
}

.stTabs [aria-selected="true"] {
    background-color: #1E293B !important;
    color: #00D2BA !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
}

/* ============================================================
   ANALYTICS CARDS & SURFACES
   ============================================================ */
.health-card {
    background-color: #111827;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
}

.health-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #1E293B;
    padding-bottom: 12px;
    margin-bottom: 16px;
}

.health-card-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #F9FAFB;
    letter-spacing: -0.01em;
}

.health-card-subtitle {
    font-size: 0.78rem;
    color: #6B7280;
}

/* Top accent bars on cards */
.health-card-accent-teal {
    border-top: 3px solid #00D2BA !important;
}

.health-card-accent-cyan {
    border-top: 3px solid #06B6D4 !important;
}

.health-card-accent-violet {
    border-top: 3px solid #8B5CF6 !important;
}

.health-card-accent-rose {
    border-top: 3px solid #F43F5E !important;
}

/* ============================================================
   STATUS BADGES & INDICATORS
   ============================================================ */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.badge-live {
    background-color: rgba(0, 210, 186, 0.12);
    color: #00D2BA;
    border: 1px solid rgba(0, 210, 186, 0.3);
}

.badge-replay {
    background-color: rgba(6, 182, 212, 0.12);
    color: #06B6D4;
    border: 1px solid rgba(6, 182, 212, 0.3);
}

.badge-stress {
    background-color: rgba(244, 63, 94, 0.14);
    color: #F43F5E;
    border: 1px solid rgba(244, 63, 94, 0.35);
}

.badge-nonstress {
    background-color: rgba(16, 185, 129, 0.12);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge-warning {
    background-color: rgba(245, 158, 11, 0.12);
    color: #F59E0B;
    border: 1px solid rgba(245, 158, 11, 0.35);
}

.badge-muted {
    background-color: #1E293B;
    color: #9CA3AF;
    border: 1px solid #334155;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}

.status-dot-teal { background-color: #00D2BA; box-shadow: 0 0 6px rgba(0, 210, 186, 0.6); }
.status-dot-rose { background-color: #F43F5E; box-shadow: 0 0 6px rgba(244, 63, 94, 0.6); }
.status-dot-amber { background-color: #F59E0B; box-shadow: 0 0 6px rgba(245, 158, 11, 0.6); }
.status-dot-cyan { background-color: #06B6D4; box-shadow: 0 0 6px rgba(6, 182, 212, 0.6); }

/* ============================================================
   PROGRESS & PROBABILITY BARS
   ============================================================ */
.progress-container {
    background-color: #1E293B;
    border-radius: 6px;
    height: 8px;
    width: 100%;
    overflow: hidden;
    margin: 6px 0;
}

.progress-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.3s ease;
}

.progress-fill-teal { background: linear-gradient(90deg, #00D2BA, #06B6D4); }
.progress-fill-rose { background: linear-gradient(90deg, #F43F5E, #FB7185); }
.progress-fill-violet { background: linear-gradient(90deg, #8B5CF6, #A78BFA); }

/* ============================================================
   STREAMLIT METRIC OVERRIDES
   ============================================================ */
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    color: #F9FAFB !important;
}

[data-testid="stMetricLabel"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: #9CA3AF !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* ============================================================
   CALLOUTS & NOTICES
   ============================================================ */
.callout-info {
    background-color: rgba(6, 182, 212, 0.08);
    border-left: 3px solid #06B6D4;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 12px 0;
    font-size: 0.82rem;
    color: #CBD5E1;
}

.callout-warning {
    background-color: rgba(245, 158, 11, 0.08);
    border-left: 3px solid #F59E0B;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 12px 0;
    font-size: 0.82rem;
    color: #FDE68A;
}

.callout-alert {
    background-color: rgba(244, 63, 94, 0.08);
    border-left: 3px solid #F43F5E;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 12px 0;
    font-size: 0.82rem;
    color: #FECDD3;
}
</style>
"""


def render_header(title: str, subtitle: str, badge_text: str = "SYSTEM ACTIVE", badge_type: str = "live") -> str:
    """Render a modern healthcare analytics header with title, subtitle, and status badge."""
    dot_class = "status-dot-teal" if badge_type == "live" else ("status-dot-cyan" if badge_type == "replay" else "status-dot-amber")
    badge_class = f"badge-{badge_type}"
    return f"""
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:1.5rem; padding-bottom:1rem; border-bottom:1px solid #1E293B;">
        <div>
            <h2 style="margin:0; font-size:1.45rem; font-weight:700; color:#F9FAFB; letter-spacing:-0.02em;">{title}</h2>
            <p style="margin:4px 0 0 0; font-size:0.85rem; color:#9CA3AF;">{subtitle}</p>
        </div>
        <div class="badge {badge_class}">
            <span class="status-dot {dot_class}"></span>
            {badge_text}
        </div>
    </div>
    """


def render_metric_card(title: str, value: str, subtitle: Optional[str] = None, accent: str = "teal") -> str:
    """Render a rounded analytics metric card with an accent bar."""
    return f"""
    <div class="health-card health-card-accent-{accent}" style="padding:16px; margin-bottom:12px;">
        <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; color:#9CA3AF; font-weight:600;">{title}</div>
        <div style="font-size:1.6rem; font-weight:700; font-family:'JetBrains Mono',monospace; color:#F9FAFB; margin:6px 0;">{value}</div>
        {f'<div style="font-size:0.75rem; color:#6B7280;">{subtitle}</div>' if subtitle else ''}
    </div>
    """


def render_probability_bar(prob: float, label: str, accent: str = "teal") -> str:
    """Render a clean probability progress bar."""
    pct = max(0.0, min(100.0, prob * 100.0))
    fill_class = f"progress-fill-{accent}"
    return f"""
    <div style="margin: 8px 0;">
        <div style="display:flex; justify-content:space-between; font-size:0.78rem; font-family:'JetBrains Mono',monospace; margin-bottom:3px;">
            <span style="color:#9CA3AF;">{label}</span>
            <span style="color:#F9FAFB; font-weight:600;">{pct:.1f}%</span>
        </div>
        <div class="progress-container">
            <div class="progress-fill {fill_class}" style="width:{pct:.1f}%;"></div>
        </div>
    </div>
    """


def render_modality_weights(w_physio: Optional[float], w_video: Optional[float]) -> str:
    """Render dynamic multimodal weighting distribution bar."""
    if w_physio is None or w_video is None:
        return """
        <div style="margin: 10px 0;">
            <div style="font-size:0.75rem; font-family:'JetBrains Mono',monospace; color:#9CA3AF; margin-bottom:4px;">
                Arbitration Mode: <strong style="color:#8B5CF6;">Stacked Logistic Meta-Learner (Inner-LOSO Coefficients)</strong>
            </div>
        </div>
        """
    p_pct = int(round(w_physio * 100))
    v_pct = int(round(w_video * 100))
    return f"""
    <div style="margin: 10px 0;">
        <div style="display:flex; justify-content:space-between; font-size:0.75rem; font-family:'JetBrains Mono',monospace; color:#9CA3AF; margin-bottom:4px;">
            <span>Physiology Authority: <strong style="color:#00D2BA;">{p_pct}%</strong></span>
            <span>Video Supporting: <strong style="color:#8B5CF6;">{v_pct}%</strong></span>
        </div>
        <div style="display:flex; height:8px; border-radius:6px; overflow:hidden; background-color:#1E293B;">
            <div style="width:{p_pct}%; background:#00D2BA; transition: width 0.3s ease;"></div>
            <div style="width:{v_pct}%; background:#8B5CF6; transition: width 0.3s ease;"></div>
        </div>
    </div>
    """
