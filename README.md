# Cross-Dataset Robust Physiological Representation Learning for Multimodal Stress Recognition

Research project investigating whether PPG/BVP-based physiological representations
generalize across datasets, and whether that robustness improves multimodal
(physiological + facial video) stress recognition.

See `docs/Research_Proposal_Cross_Dataset_Multimodal_Stress.md` for full methodology
(research questions, hypotheses, experimental tracks) and `docs/Project_Workflow.md`
for current status, phase gates, and team assignments.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Repo structure

- `docs/` — research proposal, workflow tracker, progress updates
- `src/` — model architecture (`model.py`) and LOSO training harness (`loso_harness.py`)
- `preprocessing/` — per-dataset preprocessing scripts (WESAD, PPGE, CLAS)
- `data/` — raw datasets (gitignored — download locally, never commit)
- `notebooks/` — exploration / sanity checks

## Quick sanity check

```bash
cd src
python3 model.py          # builds the CNN-TCN-LSTM model, runs a dummy forward pass
python3 loso_harness.py   # runs the LOSO training loop on synthetic data
```

Both should complete without errors before any real dataset is plugged in —
this confirms the architecture and training loop work end-to-end.

## Current status

See `docs/Project_Workflow.md` Section 4 for the live status snapshot
(dataset access, phase gates, current assignments).
