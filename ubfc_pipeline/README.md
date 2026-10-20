# UBFC-Phys multimodal pilot pipeline (offline, M2-friendly)

Run order (each stage writes small CSV/NPZ files under `out/`; every stage is resumable/re-runnable):

```
python s0_inventory.py     --root /path/UBFC-Phys --out out        # integrity, sampling-rate and duration checks
python s1_video_ts.py      --root /path/UBFC-Phys --model face_landmarker.task --subjects 1 --limit-frames 600   # smoke test
python s1_video_ts.py      --root /path/UBFC-Phys --model face_landmarker.task                                   # full run (hours)
python s2_windows.py       --root /path/UBFC-Phys --ts out/ts --out out   # 30 s / 10 s windows + rPPG-vs-BVP sync QC
python s3_label_audit.py                                                   # which labels are defensible?
python s4_loso_explain.py  --target stress_vs_rest --norm subject_all      # LOSO + modality Shapley + SHAP
python s4_loso_explain.py  --target stress_vs_rest --norm subject_all --drop v_mouth v_jaw v_rppg   # confound ablation
python s5_trend.py         --scores out/windows_scores_stress_vs_rest.csv  # trend/change-point on out-of-fold scores
```

Tested on synthetic data (stages 0, 2-5). Stage 1 (MediaPipe + video decoding) is only syntax-checked: run the
smoke test on one subject first and inspect the .npz (face_frac, rgb trace, blendshape names).

Disk: stage 1 output is ~100 KB per video; no frames are stored.
