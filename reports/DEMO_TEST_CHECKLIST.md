# Demo Test Checklist

Owner: Aaniya (integration). Branch: `aaniya`. All data shown is prerecorded UBFC-Phys replay data, never live sensing.

## 1. Setup (fresh machine)
1. Clone and switch to the branch:
   ```
   git clone https://github.com/anshulll19/stress-recognition-project.git
   cd stress-recognition-project
   git checkout aaniya
   ```
2. Install what the dashboard and tests need (Python 3.12+; tested on 3.12 and 3.14):
   ```
   pip install streamlit matplotlib scikit-learn scipy pandas numpy pytest
   ```
   `tensorflow` (listed in requirements.txt) is not needed for the dashboard or for `tests/`.
3. Saved video models were trained with scikit-learn 1.6.1. Newer versions load them with a version warning. If loading ever fails, the video branch is reported as unavailable instead of crashing.

## 2. Start the dashboard
```
python -m streamlit run app.py
```
Opens at http://localhost:8501. First load takes about 30 seconds. Stop with Ctrl+C.

## 3. Automated checks
```
python -m pytest tests -q
```
Expected: all tests pass, none skipped (about 42). Run from the repo root.
Do not run plain `pytest` at the root: `src/test_face_detection.py` needs OpenCV (`cv2`), and the root-level `test_adapters.py` / `test_physiology_inference.py` need the WESAD dataset and TensorFlow, which are not in the repo.

## 4. Decision-policy scenarios (Multimodal Prediction tab, "Decision policy" block)
| # | Physiology | Video | How to produce it | Expected heading | Expected extras |
|---|-----------|-------|-------------------|------------------|-----------------|
| 1 | reliable | reliable | default | Result determined by PHYSIOLOGY | Video shown as supporting; REPLAY label |
| 2 | reliable | disagrees | pick a window where the two differ | Result determined by PHYSIOLOGY | Result unchanged; "disagree" warning |
| 3 | reliable | lost | tick "Simulate video face loss" | Result determined by PHYSIOLOGY | "Video was not used" with reason |
| 4 | dropout | reliable | tick "Simulate physiology dropout" | Result from VIDEO ONLY (fallback) | Fallback banner; physiology "Not available" |
| 5 | dropout | lost | tick both | INSUFFICIENT DATA: no result | No label; reasons for both branches |

## 5. Display requirements
- [ ] Probabilities shown for each branch
- [ ] Quality shown per branch (physiology says "not measured in replay data (assumed nominal)")
- [ ] Decision timestamp (UTC) shown
- [ ] REPLAY label shown in the banner, the data-source field and each branch caption
- [ ] Nothing prerecorded is labelled live
- [ ] Labels are described as task-derived proxies, not validated stress labels

## 6. Existing dashboard preserved
- [ ] All six tabs render: Overview, Health Profile, Physiological Monitoring, Facial Analysis, Multimodal Prediction, Trends & Research
- [ ] The offline fusion benchmark table and trajectory chart in the Multimodal tab are unchanged
- [ ] `app.py` differs from `main` only by the import and call of `render_decision_panel` (check with `git diff main -- app.py`)

## 7. Trends
- [ ] Trends & Research tab shows "Insufficient Data" during cold start (fewer than 12 windows)
- [ ] The detector's gate: at least 75% of the last 12 windows valid (`TrendConfig.min_valid_frac`)
- [ ] `src/trend_gate.py` builds the valid-window mask from the decision engine; `tests/test_trend_gating.py` verifies a physiology dropout yields "insufficient data" and not a stale trend
- [ ] OPEN: pass that mask to `evaluate_series` inside `demo/ui/tab_trend.py` (needs Anshul's approval)

## 8. Known limitations (state these in the demo)
- Physiology quality is not measured in the replay data. The real flag (OK / SUSPECT / DROPOUT) in `src/physiology_inference.py` is not connected yet.
- In the offline benchmark the physiology-only model (balanced accuracy 0.745) is weaker than the upper-face video model (0.968). Physiology-first is a design rule, not a claim that it is more accurate.
- The video model was trained on all 18 participants; use the leave-one-participant-out numbers, not in-sample results.
- The existing blend (70/30) in the Multimodal tab is an offline benchmark, not the decision policy.

## 9. Sign-off
Tester: ______  Date: ______  Commit: ______  Result: PASS / FAIL
