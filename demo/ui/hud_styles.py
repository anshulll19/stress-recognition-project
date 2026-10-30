"""
hud_styles.py
=============
High-quality HUD CSS for the SEC Stress Recognition Dashboard.
Military/research-grade dark interface with glowing indicators,
animated status rings, and live signal aesthetics.
"""

HUD_CSS = """
<style>
/* ============================================================
   GLOBAL RESET & FONTS
   ============================================================ */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {
    background: #080B10 !important;
    color: #C8D6E5 !important;
    font-family: 'Inter', sans-serif;
}

/* Scanline overlay for HUD feel */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0, 255, 136, 0.01) 2px,
        rgba(0, 255, 136, 0.01) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ============================================================
   SIDEBAR
   ============================================================ */
section[data-testid="stSidebar"] {
    background: #090D14 !important;
    border-right: 1px solid rgba(0, 255, 136, 0.15) !important;
}

section[data-testid="stSidebar"] .stMarkdown {
    color: #8899AA !important;
}

/* ============================================================
   TABS
   ============================================================ */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(0, 255, 136, 0.2) !important;
    gap: 0 !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #556677 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 12px 24px !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #00FF88 !important;
    background: rgba(0, 255, 136, 0.05) !important;
}

.stTabs [aria-selected="true"] {
    color: #00FF88 !important;
    border-bottom: 2px solid #00FF88 !important;
    background: rgba(0, 255, 136, 0.07) !important;
}

/* ============================================================
   HUD METRIC CARDS
   ============================================================ */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0D1520 0%, #111827 100%) !important;
    border: 1px solid rgba(0, 255, 136, 0.15) !important;
    border-top: 2px solid rgba(0, 255, 136, 0.4) !important;
    padding: 18px 20px !important;
    border-radius: 6px !important;
    box-shadow:
        0 0 20px rgba(0, 255, 136, 0.04),
        inset 0 1px 0 rgba(255, 255, 255, 0.04) !important;
    position: relative !important;
    overflow: hidden !important;
}

div[data-testid="stMetric"]::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 40px; height: 40px;
    background: radial-gradient(circle at top right, rgba(0, 255, 136, 0.08), transparent 70%);
    pointer-events: none;
}

div[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    color: #4A6A7A !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    font-weight: 700 !important;
    margin-bottom: 4px !important;
}

div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #E8F4F8 !important;
    letter-spacing: -0.02em !important;
}

div[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
}

/* ============================================================
   HUD PANEL CONTAINERS
   ============================================================ */
.hud-panel {
    background: linear-gradient(135deg, #0D1520 0%, #101822 100%);
    border: 1px solid rgba(0, 150, 100, 0.2);
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 14px;
    position: relative;
    overflow: hidden;
}

.hud-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0, 255, 136, 0.4), transparent);
}

.hud-panel-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    color: #00CC6A;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.hud-panel-title::before {
    content: '▶';
    font-size: 0.5rem;
    color: #00FF88;
    opacity: 0.8;
}

/* ============================================================
   STATUS BADGES
   ============================================================ */
.hud-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 14px;
    border-radius: 3px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.hud-badge-stress {
    background: rgba(255, 50, 50, 0.12);
    border: 1px solid rgba(255, 80, 80, 0.5);
    color: #FF6B6B;
    box-shadow: 0 0 12px rgba(255, 50, 50, 0.15);
}

.hud-badge-stress::before { content: '⬤ '; color: #FF4444; font-size: 0.5rem; }

.hud-badge-nonstress {
    background: rgba(0, 200, 100, 0.12);
    border: 1px solid rgba(0, 220, 120, 0.4);
    color: #00DC78;
    box-shadow: 0 0 12px rgba(0, 200, 100, 0.15);
}

.hud-badge-nonstress::before { content: '⬤ '; color: #00FF88; font-size: 0.5rem; }

.hud-badge-dropout {
    background: rgba(255, 165, 0, 0.12);
    border: 1px solid rgba(255, 165, 0, 0.4);
    color: #FFA500;
    animation: pulse-warn 1.5s ease-in-out infinite;
}

.hud-badge-dropout::before { content: '⚠ '; font-size: 0.6rem; }

.hud-badge-ok {
    background: rgba(0, 200, 255, 0.1);
    border: 1px solid rgba(0, 200, 255, 0.3);
    color: #00C8FF;
}

.hud-badge-ok::before { content: '✓ '; }

/* ============================================================
   TREND BADGES
   ============================================================ */
.hud-trend-rising {
    background: rgba(255, 50, 50, 0.1);
    border: 1px solid rgba(255, 80, 80, 0.4);
    color: #FF7070;
    padding: 5px 16px;
    border-radius: 3px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    display: inline-block;
    animation: pulse-stress 2s ease-in-out infinite;
}

.hud-trend-falling {
    background: rgba(0, 200, 100, 0.1);
    border: 1px solid rgba(0, 220, 120, 0.4);
    color: #00DC78;
    padding: 5px 16px;
    border-radius: 3px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    display: inline-block;
}

.hud-trend-stable {
    background: rgba(0, 180, 255, 0.1);
    border: 1px solid rgba(0, 180, 255, 0.35);
    color: #00B4FF;
    padding: 5px 16px;
    border-radius: 3px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    display: inline-block;
}

.hud-trend-nodata {
    background: rgba(80, 100, 120, 0.15);
    border: 1px solid rgba(80, 100, 120, 0.35);
    color: #607080;
    padding: 5px 16px;
    border-radius: 3px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    display: inline-block;
}

/* ============================================================
   PROBABILITY BAR
   ============================================================ */
.prob-bar-container {
    width: 100%;
    height: 8px;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 4px;
    overflow: hidden;
    margin: 4px 0;
    position: relative;
}

.prob-bar-fill-stress {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #FF4444, #FF8800);
    box-shadow: 0 0 6px rgba(255, 100, 0, 0.6);
    transition: width 0.3s ease;
}

.prob-bar-fill-ok {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #00CC66, #00FFAA);
    box-shadow: 0 0 6px rgba(0, 200, 100, 0.6);
    transition: width 0.3s ease;
}

/* ============================================================
   MODALITY WEIGHTS DISPLAY
   ============================================================ */
.modality-weight-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 6px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
}

.modality-label {
    color: #4A6A7A;
    width: 100px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.modality-bar {
    flex: 1;
    height: 4px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 2px;
    overflow: hidden;
}

.modality-bar-fill {
    height: 100%;
    border-radius: 2px;
    background: rgba(0, 200, 255, 0.7);
}

.modality-value {
    color: #8899AA;
    width: 40px;
    text-align: right;
}

/* ============================================================
   CALLOUT BOXES
   ============================================================ */
.hud-info {
    background: rgba(0, 100, 200, 0.08);
    border-left: 3px solid #0064C8;
    padding: 10px 16px;
    border-radius: 0 5px 5px 0;
    font-size: 0.78rem;
    color: #8AB4D0;
    margin: 12px 0;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.6;
}

.hud-warning {
    background: rgba(200, 100, 0, 0.08);
    border-left: 3px solid #C86400;
    padding: 10px 16px;
    border-radius: 0 5px 5px 0;
    font-size: 0.78rem;
    color: #D4904A;
    margin: 12px 0;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.6;
}

.hud-success {
    background: rgba(0, 180, 80, 0.08);
    border-left: 3px solid #00B450;
    padding: 10px 16px;
    border-radius: 0 5px 5px 0;
    font-size: 0.78rem;
    color: #44CC88;
    margin: 12px 0;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.6;
}

.hud-critical {
    background: rgba(200, 0, 0, 0.08);
    border-left: 3px solid #C80000;
    padding: 10px 16px;
    border-radius: 0 5px 5px 0;
    font-size: 0.78rem;
    color: #FF6666;
    margin: 12px 0;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.6;
}

/* ============================================================
   SYSTEM STATUS HEADER
   ============================================================ */
.system-header {
    background: linear-gradient(90deg, #080B10 0%, #0D1520 40%, #080B10 100%);
    border: 1px solid rgba(0, 255, 136, 0.12);
    border-radius: 6px;
    padding: 12px 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 16px;
    font-family: 'JetBrains Mono', monospace;
}

.sys-status-dot {
    width: 8px; height: 8px;
    background: #00FF88;
    border-radius: 50%;
    box-shadow: 0 0 6px #00FF88;
    animation: pulse-live 2s ease-in-out infinite;
    flex-shrink: 0;
}

.sys-status-text {
    font-size: 0.65rem;
    color: #00CC6A;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 700;
}

.sys-info-item {
    font-size: 0.65rem;
    color: #3A5A6A;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.sys-info-value {
    color: #6A8A9A;
}

/* ============================================================
   DIVIDER
   ============================================================ */
.hud-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0, 255, 136, 0.15), transparent);
    margin: 20px 0;
}

/* ============================================================
   DATA TABLE OVERRIDE
   ============================================================ */
.stDataFrame {
    border: 1px solid rgba(0, 150, 100, 0.2) !important;
    border-radius: 6px !important;
    overflow: hidden !important;
}

.stDataFrame thead th {
    background: #0D1520 !important;
    color: #00CC6A !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    border-bottom: 1px solid rgba(0, 255, 136, 0.2) !important;
}

/* ============================================================
   SELECT BOX / INPUT OVERRIDES
   ============================================================ */
.stSelectbox > div > div {
    background: #0D1520 !important;
    border: 1px solid rgba(0, 150, 100, 0.25) !important;
    color: #C8D6E5 !important;
    border-radius: 5px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}

.stSelectbox > div > div:focus-within {
    border-color: rgba(0, 255, 136, 0.5) !important;
    box-shadow: 0 0 0 1px rgba(0, 255, 136, 0.2) !important;
}

.stSlider > div > div > div {
    background: #0D1520 !important;
}

/* ============================================================
   RADIO BUTTONS
   ============================================================ */
.stRadio > div {
    gap: 12px !important;
}

.stRadio label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #7A9AAA !important;
}

/* ============================================================
   ANIMATIONS
   ============================================================ */
@keyframes pulse-live {
    0%, 100% { opacity: 1; box-shadow: 0 0 6px #00FF88; }
    50% { opacity: 0.4; box-shadow: 0 0 14px #00FF88; }
}

@keyframes pulse-stress {
    0%, 100% { box-shadow: none; }
    50% { box-shadow: 0 0 10px rgba(255, 80, 80, 0.3); }
}

@keyframes pulse-warn {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

/* ============================================================
   CHARTS — make matplotlib transparent
   ============================================================ */
.stPlotly, canvas {
    background: transparent !important;
}

/* ============================================================
   CUSTOM TITLE STYLE
   ============================================================ */
h1, h2, h3 {
    font-family: 'Inter', sans-serif !important;
    color: #D4E8F0 !important;
}

.hud-main-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    font-weight: 700;
    color: #00FF88;
    letter-spacing: 0.04em;
    line-height: 1.4;
}

.hud-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #3A5A6A;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ============================================================
   SCROLLBAR
   ============================================================ */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #080B10; }
::-webkit-scrollbar-thumb { background: rgba(0, 200, 100, 0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0, 255, 136, 0.5); }
</style>
"""


def prob_bar(prob: float, label: str = "", width_px: int = 0) -> str:
    """Render an HTML progress bar for a probability value."""
    pct = round(prob * 100, 1)
    css_class = "prob-bar-fill-stress" if prob >= 0.5 else "prob-bar-fill-ok"
    return f"""
    <div style="margin: 3px 0;">
        <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#4A6A7A;text-transform:uppercase;letter-spacing:0.06em;">{label}</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#8899AA;">{pct}%</span>
        </div>
        <div class="prob-bar-container">
            <div class="{css_class}" style="width:{pct}%;"></div>
        </div>
    </div>
    """


def stress_badge(prediction: int, short: bool = False) -> str:
    """Render a stress/non-stress HUD badge."""
    if prediction == 1:
        label = "STRESS" if short else "STRESS DETECTED"
        return f'<span class="hud-badge hud-badge-stress">{label}</span>'
    label = "NON-STRESS" if short else "NON-STRESS / REST"
    return f'<span class="hud-badge hud-badge-nonstress">{label}</span>'


def quality_badge(quality: str) -> str:
    """Render signal quality badge."""
    if quality == "DROPOUT":
        return '<span class="hud-badge hud-badge-dropout">SENSOR DROPOUT</span>'
    if quality == "NO_FACE":
        return '<span class="hud-badge hud-badge-dropout">NO FACE DETECTED</span>'
    if quality == "UNAVAILABLE":
        return '<span class="hud-badge" style="background:rgba(80,80,80,0.1);border:1px solid rgba(80,80,80,0.3);color:#556677;">UNAVAILABLE</span>'
    return '<span class="hud-badge hud-badge-ok">SIGNAL OK</span>'


def trend_badge(trend: str) -> str:
    """Render a distress trend HUD badge."""
    t = trend.lower()
    if t == "increasing":
        return '<span class="hud-trend-rising">▲ DISTRESS RISING</span>'
    elif t == "decreasing":
        return '<span class="hud-trend-falling">▼ DISTRESS FALLING</span>'
    elif t == "stable":
        return '<span class="hud-trend-stable">● STABLE STATE</span>'
    elif "insufficient" in t:
        return '<span class="hud-trend-nodata">⏳ COLD START</span>'
    return f'<span class="hud-trend-nodata">? {trend.upper()}</span>'


def modality_weight_display(physio_w: float, video_w: float) -> str:
    """Render modality weight bars."""
    pw = round(physio_w * 100)
    vw = round(video_w * 100)
    return f"""
    <div style="margin: 8px 0;">
        <div class="modality-weight-row">
            <span class="modality-label">Physiology</span>
            <div class="modality-bar">
                <div class="modality-bar-fill" style="width:{pw}%; background:rgba(0,200,255,0.7);"></div>
            </div>
            <span class="modality-value">{pw}%</span>
        </div>
        <div class="modality-weight-row">
            <span class="modality-label">Video</span>
            <div class="modality-bar">
                <div class="modality-bar-fill" style="width:{vw}%; background:rgba(180,100,255,0.7);"></div>
            </div>
            <span class="modality-value">{vw}%</span>
        </div>
    </div>
    """


def system_status_bar(n_subjects: int, n_windows: int, model_status: str = "LOADED") -> str:
    """Render the top system status HUD bar."""
    return f"""
    <div class="system-header">
        <div class="sys-status-dot"></div>
        <span class="sys-status-text">System Online</span>
        <span style="flex:1;"></span>
        <span class="sys-info-item">SUBJECTS: <span class="sys-info-value">{n_subjects}</span></span>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <span class="sys-info-item">WINDOWS: <span class="sys-info-value">{n_windows}</span></span>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <span class="sys-info-item">MODELS: <span class="sys-info-value">{model_status}</span></span>
    </div>
    """

