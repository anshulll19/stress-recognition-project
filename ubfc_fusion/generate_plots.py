import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.family': 'sans-serif'})

oof_path = Path("results/fusion_upperface/upperface_oof_predictions.csv")
summary_path = Path("results/fusion_upperface/upperface_fusion_summary.csv")

if not oof_path.exists():
    print(f"Error: {oof_path} not found. Please run this script from the project root.")
    exit(1)

oof_df = pd.read_csv(oof_path)
summary_df = pd.read_csv(summary_path)

print("[-] Generating Research Visualizations from Frozen LOSO Benchmark Data...")

# ==============================================================================
# 1. PROTOCOL TASK ACCURACY BREAKDOWN (T1 vs T2 vs T3)
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
tasks = [
    ('T1', 'T1: Silent Rest\n(Label: 0)', 0),
    ('T2', 'T2: Speech Stress\n(Label: 1)', 1),
    ('T3', 'T3: Silent Math\n(Label: 1)', 1)
]
labels = [t[1] for t in tasks]
x = np.arange(len(tasks))
width = 0.22

phys_vals, vid_vals, blend_vals = [], [], []
for task_id, _, true_label in tasks:
    sub = oof_df[oof_df['session_id'] == task_id]
    phys_vals.append(((sub['prob_phys_lstm'] >= 0.5).astype(int) == true_label).mean() * 100)
    vid_vals.append(((sub['prob_video_upperface'] >= 0.5).astype(int) == true_label).mean() * 100)
    blend_vals.append(((sub['prob_blend_0.3'] >= 0.5).astype(int) == true_label).mean() * 100)

rects1 = ax.bar(x - width, phys_vals, width, label='Physiology (LSTM)', color='#90a4ae', edgecolor='none')
rects2 = ax.bar(x, vid_vals, width, label='Upper-Face Video', color='#29b6f6', edgecolor='none')
rects3 = ax.bar(x + width, blend_vals, width, label='Blend 0.3 (Fused)', color='#26a69a', edgecolor='none')

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title('Classification Accuracy Across Protocol Conditions\n(Testing Non-Verbal Cognitive Load T3)', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(50, 105)
ax.legend(loc='lower left', frameon=True, fontsize=10)

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, fontweight='bold')

autolabel(rects1); autolabel(rects2); autolabel(rects3)
plt.tight_layout()
plt.savefig("plot_conditions_accuracy.png")
plt.close()
print("  [✓] Saved plot_conditions_accuracy.png")

# ==============================================================================
# 2. OVERALL BENCHMARK PERFORMANCE ACROSS 18 LOSO SUBJECTS
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
models_order = ['Physiology only (LSTM)', 'Upper-face Video only', 'Blend 0.2', 'Blend 0.3', 'Stacked LOSO']
display_names = ['Physiology (LSTM)', 'Upper-Face Video', 'Blend 0.2', 'Blend 0.3', 'Stacked LOSO']

bacc_vals = [summary_df[summary_df['Method'] == m].iloc[0]['Balanced Accuracy'] * 100 for m in models_order]
f1_vals = [summary_df[summary_df['Method'] == m].iloc[0]['Macro-F1'] * 100 for m in models_order]

x = np.arange(len(models_order))
width = 0.3

rects1 = ax.bar(x - width/2, bacc_vals, width, label='Balanced Accuracy', color='#2b5c8f', edgecolor='none')
rects2 = ax.bar(x + width/2, f1_vals, width, label='Macro-F1 Score', color='#d96b52', edgecolor='none')

ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title('Multimodal Model Performance Across 18 LOSO Subjects', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(display_names, fontsize=11)
ax.set_ylim(70, 102)
ax.legend(loc='lower right', frameon=True, fontsize=10)
autolabel(rects1); autolabel(rects2)
plt.tight_layout()
plt.savefig("plot_loso_performance.png")
plt.close()
print("  [✓] Saved plot_loso_performance.png")

# ==============================================================================
# 3. CONFUSION MATRIX: BLEND 0.3 MULTIMODAL FUSION
# ==============================================================================
fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=300)
from sklearn.metrics import confusion_matrix

y_true = oof_df['label']
y_pred_blend = (oof_df['prob_blend_0.3'] >= 0.5).astype(int)
cm_blend = confusion_matrix(y_true, y_pred_blend)
cm_norm = cm_blend.astype('float') / cm_blend.sum(axis=1)[:, np.newaxis]

annot = np.empty_like(cm_blend).astype(str)
for i in range(2):
    for j in range(2):
        annot[i, j] = f"{cm_blend[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

sns.heatmap(cm_blend, annot=annot, fmt='', cmap='Blues', cbar=True,
            xticklabels=['Predicted Rest (0)', 'Predicted Stress (1)'],
            yticklabels=['True Rest (0)', 'True Stress (1)'],
            annot_kws={'size': 13, 'weight': 'bold'}, ax=ax)

ax.set_title('Confusion Matrix: Blend 0.3 Multimodal Fusion\n(972 Synchronized LOSO Windows)', fontsize=12, fontweight='bold', pad=12)
ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig("plot_confusion_matrix_blend03.png")
plt.close()
print("  [✓] Saved plot_confusion_matrix_blend03.png")

# ==============================================================================
# 4. SIDE-BY-SIDE CONFUSION MATRIX COMPARISON (PHYSIOLOGY vs VIDEO vs FUSION)
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=300)
cms = [
    ("Physiology only (LSTM)", confusion_matrix(y_true, (oof_df['prob_phys_lstm'] >= 0.5).astype(int)), 'Purples'),
    ("Upper-Face Video only", confusion_matrix(y_true, (oof_df['prob_video_upperface'] >= 0.5).astype(int)), 'Blues'),
    ("Blend 0.3 (Multimodal Fusion)", cm_blend, 'Greens')
]

for idx, (title, cm, cmap) in enumerate(cms):
    ax = axes[idx]
    norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    annot = np.empty_like(cm).astype(str)
    for i in range(2):
        for j in range(2):
            annot[i, j] = f"{cm[i, j]}\n({norm[i, j]*100:.1f}%)"
            
    sns.heatmap(cm, annot=annot, fmt='', cmap=cmap, cbar=False,
                xticklabels=['Pred Rest', 'Pred Stress'],
                yticklabels=['True Rest', 'True Stress'] if idx == 0 else ['', ''],
                annot_kws={'size': 12, 'weight': 'bold'}, ax=ax)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    if idx == 0:
        ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')

plt.suptitle('Confusion Matrix Comparison Across Modalities (18 LOSO Subjects, N=972)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("plot_confusion_matrix_comparison.png")
plt.close()
print("  [✓] Saved plot_confusion_matrix_comparison.png")

print("\n[+] SUCCESS: All 4 presentation-ready plots generated directly from benchmark telemetry!")
