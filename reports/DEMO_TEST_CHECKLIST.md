# Demo Test Checklist

Owner: Aaniya (integration). Status: DRAFT. Items marked TODO need the real repo.

## 1. Environment
- [ ] Fresh clone / clean virtual environment
- [ ] `pip install -r requirements.txt` succeeds (TODO: confirm file name)
- [ ] Python version recorded: ______
- [ ] Model files referenced in `models/deployment_candidate/manifest.json` and `models/video_model_metadata.json` exist

## 2. Automated checks
- [ ] `python -m pytest -q` from the repo root: 0 failures (skips are listed and explained)
- [ ] `python -m py_compile app.py`

## 3. Startup
- [ ] `streamlit run app.py` starts with no traceback (TODO: confirm exact command and port)
- [ ] All five existing tabs render, unchanged from the pre-integration baseline (compare against screenshots taken before editing)

## 4. Decision-policy scenarios (use mock/replay inputs)
| # | Physiology | Video | Expected mode | Expected label on screen |
|---|-----------|-------|---------------|--------------------------|
| 1 | reliable | reliable, agrees | physiology_primary | Physiology determined result; video shown as supporting |
| 2 | reliable | reliable, disagrees | physiology_primary | Result unchanged; disagreement warning shown |
| 3 | reliable | unreliable/missing | physiology_primary | Video shown as not used, with reason |
| 4 | unreliable | reliable | video_only_fallback | Visible "VIDEO-ONLY FALLBACK" banner |
| 5 | missing | reliable | video_only_fallback | Same banner |
| 6 | unreliable | unreliable | insufficient_data | No class label; reasons listed |

## 5. Display requirements
- [ ] Class probabilities shown for the determining branch
- [ ] Quality indicator shown per branch
- [ ] Timestamp shown on every result
- [ ] Replay/live label shown on every result
- [ ] Prerecorded (UBFC-Phys) data is labelled REPLAY everywhere; nothing prerecorded says "live"
- [ ] Mixed live + replay inputs produce a REPLAY label and a warning

## 6. Trends
- [ ] Trend view appears only when enough valid data exists (threshold from the trend module: TODO)
- [ ] Below the threshold, a clear "not enough valid data" message is shown instead of a chart

## 7. Failure handling
- [ ] Missing model file: dashboard shows a message, does not crash
- [ ] Malformed prediction output: treated as unreliable, not a crash
- [ ] Stale live prediction rejected (default limit 15 s)

## 8. Sign-off
Tester: ______  Date: ______  Commit hash: ______  Result: PASS / FAIL
