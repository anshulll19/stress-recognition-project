# StressSense frontend

React + Vite replay dashboard for the stress-recognition project. It shows stored benchmark
results and does not run any model.

## Run it

    cd frontend
    npm install
    npm run dev        # development server
    npm run build      # production build
    npm run lint

## Where the data comes from

All numbers shown on the pages are exported from the repository's result files into
`frontend/src/content/`. The folder is called `content` (not `data`) because the root
`.gitignore` ignores any folder named `data/`.

Run these from the repository root to regenerate the exports:

    python scripts/export_trajectory.py   # Overview session chart (subject s3)
    python scripts/export_physio.py       # Physiological tab (participants 1-5)
    python scripts/export_fusion.py       # Multimodal and Facial tabs
    python scripts/export_confounds.py    # confound ablation table
    python scripts/export_loso.py         # per-subject LOSO tables
    python scripts/export_trend.py        # Trends tab, real participants
    python scripts/export_synthetic.py    # Trends tab, synthetic patterns

## Known gaps

- WESAD per-window replay is unavailable: `data/wesad/wesad_binary.npz` is not in the repository.
- `models/ubfc_deployment/loso_per_subject.csv` is not in the repository, so that table is omitted.
- CLAS inference is disabled: there is no trained classification head.
- The Trends caveat cites 7 of 17 failing synthetic criteria from `reports/PHASE1_PROJECT_STATUS.md`;
  only three synthetic patterns exist in the repository.
