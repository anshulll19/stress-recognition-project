"""Parity test: src/video_pipeline.py vs results/video/video_windows.csv.

Skipped unless VIDEO_PARITY_PATH points to vid_s16_T3.avi (not shipped, 5.84 GB).
Run:  VIDEO_PARITY_PATH="C:/Users/<you>/Downloads/vid_s16_T3.avi" python -m unittest test_video_parity -v
Requires the environment in requirements-video.txt (scikit-learn 1.9.1).
"""
import json
import os
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
VIDEO = os.environ.get("VIDEO_PARITY_PATH", "")
SECONDS = 60
TOL = 1e-3


@unittest.skipUnless(VIDEO and Path(VIDEO).exists(),
                     "set VIDEO_PARITY_PATH to vid_s16_T3.avi to run")
class TestVideoParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.video_pipeline import VideoPipeline
        cls.w23, cls.w19, _ = VideoPipeline().extract_features_from_video(
            VIDEO, max_duration_sec=SECONDS)
        cls.w23 = cls.w23.sort_values("window_idx").reset_index(drop=True)
        ref = pd.read_csv(ROOT / "results/video/video_windows.csv")
        ref = ref[(ref.subject_id == "s16") & (ref.session_id == "T3")]
        cls.ref = ref.sort_values("window_idx").reset_index(drop=True)
        cls.n = len(cls.w23)

    def test_window_count(self):
        self.assertEqual(self.n, SECONDS // 10)

    def test_features_match_reference(self):
        feats = [c for c in self.ref.columns if c.startswith("f_")]
        for c in feats:
            self.assertIn(c, self.w23.columns)
            a = self.w23[c].to_numpy(float)
            b = self.ref[c].to_numpy(float)[:self.n]
            self.assertLessEqual(np.nanmax(np.abs(a - b)), TOL, c)

    def test_upperface_probabilities_match(self):
        import joblib
        with open(ROOT / "models/video_upperface_features.json") as fh:
            names = json.load(fh)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = joblib.load(ROOT / "models/video_distress_upperface.joblib")
            pe = model.predict_proba(self.w23[names].to_numpy(float))[:, 1]
            pr = model.predict_proba(self.ref[names].to_numpy(float)[:self.n])[:, 1]
        self.assertLessEqual(float(np.max(np.abs(pe - pr))), 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
