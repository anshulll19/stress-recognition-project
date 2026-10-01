# WESAD Preprocessing — Phase 0 (Person 1)

This folder contains my part of the project: downloading WESAD, validating it, and building the preprocessing pipeline on the wrist BVP (heartbeat) signal. No model was trained here and no accuracy was measured — this is purely a data-engineering step, done before any machine learning happens.

## What's in this folder

| File | What it is |
|---|---|
| `wesad_preprocess.py` | The script. Loads each subject's raw WESAD `.pkl` file, validates the BVP channel, aligns the 700Hz labels to the 64Hz BVP signal by real time (not raw index), filters the signal, cuts it into windows, and normalizes each window. |
| `report.txt` / `report.json` | Output of running the script on all 15 subjects — which subjects loaded, final array shapes, and any data-quality issues found. |

(The raw WESAD `.pkl` files and the generated `wesad_processed/*.npz` output are **not** included in this repo — they're data files, not code, and are easy to regenerate by running the script against the public dataset.)

## How to run it

```bash
pip install numpy scipy

# test on one subject first
python wesad_preprocess.py --data_dir /path/to/WESAD --mode single --subject S2

# once that looks correct, run all subjects
python wesad_preprocess.py --data_dir /path/to/WESAD --mode all --label_rule majority
```

## What the pipeline does

1. **Loads one subject's `.pkl` file** and checks the wrist BVP channel is present, non-empty, and sampled at ~64Hz.
2. **Aligns labels to BVP by real time**, not raw index — WESAD labels are recorded at 700Hz (chest device) while BVP is 64Hz, so matching them by position drifts and gives wrong labels. This script converts both to actual timestamps and matches on time instead.
3. **Filters the signal**: Butterworth bandpass, 0.7–3.7 Hz (keeps the realistic human heart-rate range, removes noise).
4. **Windows the signal**: 60-second windows, 5-second overlap (3840 samples/window at 64Hz).
5. **Normalizes each window**: z-score per window.
6. **Assigns a label to each window** using a documented rule (`--label_rule majority` by default — a window gets whichever label covers more than 50% of it; `discard` is also available, which drops any window that isn't 100% one label).

## Results (all 15 subjects)

- All 15 expected subjects (S2–S11, S13–S17) loaded and processed successfully. **0 excluded.**
- No corrupted files, no missing BVP channels, no label-misalignment issues found.
- 1570 total windows produced across all subjects, shape `(n_windows, 3840)` per subject.
- Full details in `report.txt`.

## Important note for the rest of the team

Subjects are **only** ever excluded for genuine data-integrity problems (corrupted file, missing sensor, broken labels) — never because of how they might affect a future accuracy score. Doing the latter would be cherry-picking / invalid methodology. None of the 15 subjects had any such problem.

The `--label_rule` choice (majority vote, by default) needs to be applied identically when preprocessing PPGE and EmpathicSchool, so cross-dataset comparisons stay valid.
