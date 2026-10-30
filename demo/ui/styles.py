"""CSS styling and HTML components for research-oriented dark dashboard."""

DARK_THEME_CSS = """
<style>
/* Main dashboard theme overrides */
.stApp {
    background-color: #0E1117;
    color: #FAFAFA;
}

/* Metric card styling */
div[data-testid="stMetric"] {
    background-color: #1A1D24;
    border: 1px solid #2D3748;
    padding: 16px 20px;
    border-radius: 8px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
}

div[data-testid="stMetricLabel"] {
    font-size: 0.85rem !important;
    color: #A0AAB2 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #FAFAFA !important;
}

/* Custom research cards */
.research-card {
    background-color: #1A1D24;
    border: 1px solid #2D3748;
    border-radius: 8px;
    padding: 18px 22px;
    margin-bottom: 16px;
}

.research-card-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #E2E8F0;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* Status pills */
.badge-stress {
    background-color: rgba(239, 68, 68, 0.2);
    color: #EF4444;
    border: 1px solid #EF4444;
    padding: 4px 12px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.85rem;
    display: inline-block;
}

.badge-nonstress {
    background-color: rgba(16, 185, 129, 0.2);
    color: #10B981;
    border: 1px solid #10B981;
    padding: 4px 12px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.85rem;
    display: inline-block;
}

.badge-neutral {
    background-color: rgba(100, 116, 139, 0.2);
    color: #94A3B8;
    border: 1px solid #64748B;
    padding: 4px 12px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.85rem;
    display: inline-block;
}

.badge-trend-inc {
    background-color: rgba(239, 68, 68, 0.25);
    color: #F87171;
    border: 1px solid #EF4444;
    padding: 4px 12px;
    border-radius: 12px;
    font-weight: 700;
    font-size: 0.85rem;
}

.badge-trend-dec {
    background-color: rgba(16, 185, 129, 0.25);
    color: #34D399;
    border: 1px solid #10B981;
    padding: 4px 12px;
    border-radius: 12px;
    font-weight: 700;
    font-size: 0.85rem;
}

.badge-trend-stable {
    background-color: rgba(59, 130, 246, 0.25);
    color: #60A5FA;
    border: 1px solid #3B82F6;
    padding: 4px 12px;
    border-radius: 12px;
    font-weight: 700;
    font-size: 0.85rem;
}

/* Callouts and Disclaimers */
.research-disclaimer {
    background-color: #171E2E;
    border-left: 4px solid #3B82F6;
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    font-size: 0.85rem;
    color: #CBD5E1;
    margin: 14px 0;
}

.research-warning {
    background-color: #271C17;
    border-left: 4px solid #F59E0B;
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    font-size: 0.85rem;
    color: #FDE68A;
    margin: 14px 0;
}
</style>
"""


def render_prediction_badge(predicted_class: int, label: str) -> str:
    """Render colored HTML badge for class prediction."""
    if predicted_class == 1:
        return f'<span class="badge-stress">STRESS: {label}</span>'
    return f'<span class="badge-nonstress">NON-STRESS: {label}</span>'


def render_trend_badge(trend_label: str) -> str:
    """Render colored HTML badge for temporal trend status."""
    tl = trend_label.lower()
    if tl == "increasing":
        return f'<span class="badge-trend-inc">▲ INCREASING DISTRESS</span>'
    elif tl == "decreasing":
        return f'<span class="badge-trend-dec">▼ DECREASING / RECOVERING</span>'
    elif tl == "stable":
        return f'<span class="badge-trend-stable">● STABLE STATE</span>'
    elif tl == "fluctuating":
        return f'<span class="badge-neutral">〰 FLUCTUATING</span>'
    elif tl == "insufficient data":
        return f'<span class="badge-neutral">⏳ INSUFFICIENT DATA (COLD START)</span>'
    return f'<span class="badge-neutral">? {trend_label.upper()}</span>'

