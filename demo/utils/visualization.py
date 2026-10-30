"""Visualization helpers for research-grade dark theme plotting."""
import os
os.environ["MPLCONFIGDIR"] = "/tmp"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Research Dark Theme Palette
DARK_BG = "#0E1117"
CARD_BG = "#1A1D24"
TEXT_COLOR = "#FAFAFA"
MUTED_TEXT = "#A0AAB2"
ACCENT_BLUE = "#3B82F6"
ACCENT_GREEN = "#10B981"
ACCENT_RED = "#EF4444"
ACCENT_AMBER = "#F59E0B"
ACCENT_PURPLE = "#8B5CF6"
GRID_COLOR = "#2D3748"


def apply_dark_theme(fig, ax):
    """Apply consistent research dark theme to matplotlib axes."""
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=MUTED_TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, linestyle="--", alpha=0.3, color=GRID_COLOR)


def plot_wesad_window(window_array: np.ndarray, fs: float = 64.0) -> plt.Figure:
    """Plot single-channel 60-second WESAD BVP waveform."""
    fig, ax = plt.subplots(figsize=(10, 2.8), dpi=100)
    apply_dark_theme(fig, ax)

    signal = window_array.squeeze()
    time_s = np.arange(len(signal)) / fs

    ax.plot(time_s, signal, color=ACCENT_BLUE, linewidth=1.0, label="BVP (Wrist PPG)")
    ax.set_title("WESAD 60-Second BVP Waveform (64 Hz)", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Time (seconds)", fontsize=9)
    ax.set_ylabel("Normalized Amplitude (z-score)", fontsize=9)
    ax.set_xlim(0, max(time_s))
    ax.legend(loc="upper right", facecolor=DARK_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    fig.tight_layout()
    return fig


def plot_ubfc_signals(bvp_array: np.ndarray, eda_array: np.ndarray) -> plt.Figure:
    """Plot synchronized 10-second BVP (64 Hz) and EDA (4 Hz) waveforms for UBFC-Phys."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 3.8), dpi=100, sharex=False)
    apply_dark_theme(fig, ax1)
    apply_dark_theme(fig, ax2)

    t_bvp = np.arange(bvp_array.size) / 64.0
    t_eda = np.arange(eda_array.size) / 4.0

    ax1.plot(t_bvp, bvp_array.squeeze(), color=ACCENT_BLUE, linewidth=1.1, label="BVP (64 Hz)")
    ax1.set_title("Synchronous 10-Second Physiological Windows (UBFC-Phys)", fontsize=10, fontweight="bold", pad=6)
    ax1.set_ylabel("BVP", fontsize=8)
    ax1.legend(loc="upper right", facecolor=DARK_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)

    ax2.plot(t_eda, eda_array.squeeze(), color=ACCENT_AMBER, linewidth=1.4, label="EDA (4 Hz)")
    ax2.set_xlabel("Window Time (seconds)", fontsize=8)
    ax2.set_ylabel("EDA (μS)", fontsize=8)
    ax2.legend(loc="upper right", facecolor=DARK_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)

    fig.tight_layout()
    return fig


def plot_multimodal_comparison(p_physio: float, p_video: float, p_fused: float) -> plt.Figure:
    """Plot bar comparison of unimodal vs multimodal distress probabilities."""
    fig, ax = plt.subplots(figsize=(6, 2.6), dpi=100)
    apply_dark_theme(fig, ax)

    labels = ["Physiology (LSTM)", "Video (Random Forest)", "Multimodal Blend (0.3)"]
    probs = [p_physio, p_video, p_fused]
    colors = [ACCENT_BLUE, ACCENT_PURPLE, ACCENT_GREEN]

    bars = ax.barh(labels, probs, color=colors, height=0.45, edgecolor=GRID_COLOR)
    ax.axvline(0.5, color=ACCENT_RED, linestyle=":", alpha=0.8, linewidth=1.2, label="Threshold (0.5)")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Predicted Distress Probability", fontsize=8)
    ax.set_title("Decision-Level Modality Comparison", fontsize=10, fontweight="bold")

    for bar, p in zip(bars, probs):
        ax.text(p + 0.02, bar.get_y() + bar.get_height() / 2, f"{p:.1%}",
                va="center", ha="left", color=TEXT_COLOR, fontsize=8, fontweight="bold")

    ax.legend(loc="lower right", facecolor=DARK_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    fig.tight_layout()
    return fig


def plot_trend_series(
    probabilities: np.ndarray,
    current_idx: int,
    trend_results: dict,
    K: int = 12,
    step_s: int = 5
) -> plt.Figure:
    """Plot timeline of probabilities and Theil-Sen regression slope for current window."""
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=100)
    apply_dark_theme(fig, ax)

    n = len(probabilities)
    time_s = np.arange(n) * step_s

    # Plot full probability history up to current window
    history_p = probabilities[:current_idx + 1]
    history_t = time_s[:current_idx + 1]
    ax.plot(history_t, history_p, color=MUTED_TEXT, alpha=0.6, linewidth=1.0, linestyle="--", label="Past History")
    ax.scatter(history_t, history_p, color=MUTED_TEXT, alpha=0.5, s=15)

    # Highlight active K-window (past-only)
    a = max(0, current_idx - K + 1)
    k_t = time_s[a:current_idx + 1]
    k_p = probabilities[a:current_idx + 1]

    ax.plot(k_t, k_p, color=ACCENT_BLUE, linewidth=1.8, label=f"Active K={K} Window")
    ax.scatter(k_t, k_p, color=ACCENT_BLUE, s=30)
    ax.scatter([time_s[current_idx]], [probabilities[current_idx]], color=ACCENT_RED, s=60, zorder=5, label="Current Window")

    # If trend is computed, display slope info
    slope_val = trend_results["slope"][current_idx]
    label_str = trend_results["displayed_label"][current_idx]
    is_suff = trend_results["is_sufficient"][current_idx]

    title_suffix = f" · Status: {label_str.upper()}"
    if is_suff and np.isfinite(slope_val):
        title_suffix += f" (Slope: {slope_val:+.2f} logit/60s)"
    ax.set_title(f"Causal Temporal Trend Analysis (Past {K} Windows){title_suffix}", fontsize=10, fontweight="bold")

    ax.axhline(0.5, color=GRID_COLOR, linestyle=":", alpha=0.6)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time from Task Start (seconds)", fontsize=8)
    ax.set_ylabel("Distress Probability", fontsize=8)
    ax.legend(loc="upper left", facecolor=DARK_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)

    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: list, title: str = "Confusion Matrix") -> plt.Figure:
    """Plot styled 2x2 confusion matrix."""
    fig, ax = plt.subplots(figsize=(4.5, 3.2), dpi=100)
    apply_dark_theme(fig, ax)

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_ylabel("True Label", fontsize=8)
    ax.set_xlabel("Predicted Label", fontsize=8)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else TEXT_COLOR,
                    fontsize=10, fontweight="bold")

    fig.tight_layout()
    return fig

