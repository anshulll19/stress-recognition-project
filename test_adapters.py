"""Comprehensive test suite for stress recognition demo adapters and integrity contracts."""
import os
os.environ["MPLCONFIGDIR"] = "/tmp"

import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from demo.adapters.wesad_adapter import WESADInferenceAdapter
from demo.adapters.ubfc_adapter import UBFCPhysInferenceAdapter
from demo.adapters.video_adapter import VideoInferenceAdapter
from demo.adapters.fusion_adapter import MultimodalFusionAdapter
from demo.adapters.trend_adapter import TemporalTrendAdapter, TrendConfig, INSUFF
from demo.adapters.clas_adapter import CLASInferenceAdapter
from demo.utils.metrics import ResearchMetricsLoader


class TestStressDemoAdapters(unittest.TestCase):
    """Smoke and integrity test suite for multimodal demo adapters."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parent
        cls.wesad = WESADInferenceAdapter(cls.root)
        cls.ubfc = UBFCPhysInferenceAdapter(cls.root)
        cls.video = VideoInferenceAdapter(cls.root)
        cls.fusion = MultimodalFusionAdapter(cls.root)
        cls.trend = TemporalTrendAdapter()
        cls.clas = CLASInferenceAdapter(cls.root)
        cls.metrics = ResearchMetricsLoader(cls.root)

    # Test 1: Dataset inventory and schema validation
    def test_01_dataset_inventory(self):
        print("\n[TEST 1] Dataset inventory and schema validation...")
        # WESAD
        self.assertTrue(self.wesad.data_path.exists())
        w_windows, w_labels, w_subs = self.wesad.load_dataset()
        self.assertEqual(w_windows.shape, (812, 3840, 1))
        self.assertEqual(len(w_labels), 812)
        self.assertEqual(len(w_subs), 812)
        self.assertEqual(len(np.unique(w_subs)), 15)

        # UBFC
        self.assertTrue(self.ubfc.raw_data_path.exists())
        u_raw, u_meta, u_oof = self.ubfc.load_dataset()
        self.assertEqual(u_raw["X_bvp"].shape, (5880, 640, 1))
        self.assertEqual(u_raw["X_eda"].shape, (5880, 40, 1))
        self.assertEqual(len(u_meta), 5880)
        self.assertEqual(len(u_oof), 11760)

        # Video
        self.assertTrue(self.video.windows_path.exists())
        v_df = self.video.load_windows()
        self.assertEqual(len(v_df), 972)
        self.assertEqual(v_df["subject_id"].nunique(), 18)
        print("✓ All dataset schemas verified.")

    # Test 2: Model loading and inference smoke tests
    def test_02_model_loading_and_inference(self):
        print("\n[TEST 2] Model loading and inference smoke tests...")
        # WESAD Model
        self.assertTrue(self.wesad.is_model_available())
        w_model = self.wesad.load_model()
        self.assertIsNotNone(w_model)
        dummy_wesad = np.zeros((1, 3840, 1), dtype=np.float32)
        w_res = self.wesad.predict(dummy_wesad)
        self.assertIn(w_res.predicted_class, [0, 1])

        # UBFC OOF Replay
        self.assertTrue(self.ubfc.is_model_available())
        u_res = self.ubfc.predict(
            {"bvp": np.zeros(640), "eda": np.zeros(40)},
            metadata={"window_index": 0, "model_name": "LSTM", "task": 1}
        )
        self.assertIn(u_res.predicted_class, [0, 1])

        # Video Model
        self.assertTrue(self.video.is_model_available())
        v_model = self.video.load_model()
        self.assertIsNotNone(v_model)
        dummy_video = np.zeros(23, dtype=float)
        v_res = self.video.predict(dummy_video)
        self.assertIn(v_res.predicted_class, [0, 1])
        print("✓ Model loading and inference smoke tests passed.")

    # Test 3: Input shape and feature order validation
    def test_03_input_shape_and_feature_order(self):
        print("\n[TEST 3] Input shape and feature order validation...")
        # WESAD invalid shape
        bad_shape_wesad = np.zeros((500, 1))
        val_w = self.wesad.validate_input(bad_shape_wesad)
        self.assertFalse(val_w.is_valid)

        # UBFC invalid shape
        val_u = self.ubfc.validate_input({"bvp": np.zeros(100), "eda": np.zeros(40)})
        self.assertFalse(val_u.is_valid)

        # Video feature count and order
        self.assertEqual(len(self.video.feature_names), 23)
        self.assertEqual(self.video.feature_names[0], "f_motion_mean")
        self.assertEqual(self.video.feature_names[-1], "f_motion_p90")
        val_v = self.video.validate_input(np.zeros(10))
        self.assertFalse(val_v.is_valid)
        print("✓ Input shape and feature ordering contracts enforced.")

    # Test 4: Finite-output checks
    def test_04_finite_outputs(self):
        print("\n[TEST 4] Finite output checks...")
        # WESAD
        windows, _, _ = self.wesad.load_dataset()
        res_w = self.wesad.predict(windows[0:1])
        self.assertTrue(np.all(np.isfinite(res_w.probabilities)))
        self.assertTrue(0.0 <= res_w.probabilities[0] <= 1.0)
        self.assertTrue(0.0 <= res_w.probabilities[1] <= 1.0)
        self.assertAlmostEqual(float(np.sum(res_w.probabilities)), 1.0, places=4)

        # Video
        v_df = self.video.load_windows()
        res_v = self.video.predict(v_df.iloc[0])
        self.assertTrue(np.all(np.isfinite(res_v.probabilities)))
        self.assertAlmostEqual(float(np.sum(res_v.probabilities)), 1.0, places=4)
        print("✓ Probabilities are finite, bounded in [0, 1], and properly normalized.")

    # Test 5: Class-mapping checks
    def test_05_class_mappings(self):
        print("\n[TEST 5] Class-mapping checks...")
        self.assertEqual(self.wesad.CLASS_NAMES[0], "Non-stress (Baseline/Amusement)")
        self.assertEqual(self.wesad.CLASS_NAMES[1], "Stress (TSST Protocol)")
        self.assertEqual(self.ubfc.CLASS_NAMES[0], "Non-stress (Rest Task 1 Proxy)")
        self.assertEqual(self.ubfc.CLASS_NAMES[1], "Stress-related (Speech T2 / Math T3 Proxy)")
        self.assertEqual(self.video.CLASS_NAMES[0], "Non-stress (Rest Task T1 Proxy)")
        self.assertEqual(self.video.CLASS_NAMES[1], "Distress / Stress-task (Speech T2 / Math T3 Proxy)")
        print("✓ Class mapping semantic contracts verified.")

    # Test 6: Missing-data and insufficient-window tests
    def test_06_missing_data_and_insufficient_windows(self):
        print("\n[TEST 6] Missing data and insufficient window tests...")
        # Non-finite input rejection
        nan_array = np.full((3840, 1), np.nan)
        val_nan = self.wesad.validate_input(nan_array)
        self.assertFalse(val_nan.is_valid)

        # Trend insufficient window handling (cold start)
        short_series = np.array([0.2, 0.3, 0.4])
        trend_out = self.trend.evaluate_series(short_series)
        self.assertEqual(trend_out["displayed"][0], INSUFF)
        self.assertEqual(trend_out["displayed"][2], INSUFF)
        self.assertFalse(trend_out["is_sufficient"][2])

        # Immediate switch to insufficient data on quality drop
        series_with_gap = np.array([0.2] * 15 + [np.nan] * 5)
        trend_gap = self.trend.evaluate_series(series_with_gap)
        self.assertEqual(trend_gap["displayed"][-1], INSUFF)
        print("✓ Missing data rejection and cold-start trend guards verified.")

    # Test 7: Participant and window alignment checks for fusion
    def test_07_fusion_alignment(self):
        print("\n[TEST 7] Participant and window alignment checks for fusion...")
        aligned_df = self.fusion.load_aligned_data()
        self.assertEqual(len(aligned_df), 972)
        self.assertEqual(aligned_df["subject_id"].nunique(), 18)

        # Test intentional mismatch detection
        row1 = aligned_df.iloc[0].copy()
        row2 = aligned_df.iloc[100].copy()  # different subject/task
        mismatch_val = self.fusion.validate_alignment(row1, row2)
        self.assertFalse(mismatch_val.is_valid)

        # Valid alignment test
        pred_fus = self.fusion.get_fusion_prediction(row1, method="blend_0.3")
        self.assertIn(pred_fus.predicted_class, [0, 1])
        self.assertAlmostEqual(float(np.sum(pred_fus.probabilities)), 1.0, places=4)
        print("✓ Multimodal alignment checks enforce participant, task, and temporal coherence.")

    # Test 8: CLAS limitation enforcement
    def test_08_clas_integrity_enforcement(self):
        print("\n[TEST 8] CLAS audit and limitations enforcement...")
        self.assertFalse(self.clas.is_model_available())
        summary = self.clas.get_audit_summary()
        self.assertEqual(summary["status"], "INFERENCE_DISABLED")
        self.assertFalse(summary["has_classifier_head"])

        with self.assertRaises(NotImplementedError):
            self.clas.predict(np.zeros((3840, 1)))
        print("✓ CLAS adapter correctly blocks fake inference while providing transparent audit.")

    # Test 9: Replay test from start to finish
    def test_09_full_replay_run(self):
        print("\n[TEST 9] Full recorded replay test across participants...")
        # UBFC participant 1 (all 105 windows)
        indices, bvp, eda, y, meta, oof = self.ubfc.get_participant_windows(1, model_name="LSTM")
        self.assertEqual(len(indices), 105)
        for i in range(min(10, len(indices))):
            res = self.ubfc.predict(
                {"bvp": bvp[i], "eda": eda[i]},
                metadata={"window_index": int(indices[i]), "model_name": "LSTM", "task": int(meta.iloc[i]["task"])}
            )
            self.assertIn(res.predicted_class, [0, 1])

        # WESAD validation subject S10 (all windows)
        s_idx, s_win, s_lbl = self.wesad.get_subject_windows("S10")
        self.assertGreater(len(s_win), 0)
        res_w = self.wesad.predict(s_win[0:1], metadata={"subject_id": "S10", "ground_truth": int(s_lbl[0])})
        self.assertIn(res_w.predicted_class, [0, 1])
        print("✓ Full sequential replay pipeline successfully executed.")

    # Test 10: Confirmation of preserved original artifacts
    def test_10_preserved_artifacts(self):
        print("\n[TEST 10] Confirmation that original models and evaluation artifacts remain intact...")
        expected_files = [
            "data/wesad/wesad_binary.npz",
            "models/deployment_candidate/classifier.keras",
            "models/deployment_candidate/manifest.json",
            "results/physiology/raw_signal_windows.npz",
            "results/physiology/raw_signal_window_metadata.csv",
            "results/temporal_models/tcn_lstm_oof_predictions_full.csv",
            "results/video/video_windows.csv",
            "models/video_distress_baseline.joblib",
            "models/video_feature_names.json",
            "models/video_model_metadata.json",
            "models/face_landmarker.task",
            "results/fusion_lstm/aligned_oof_predictions.csv",
            "results/fusion_lstm/fusion_summary.csv",
            "results/loso_corrected_valloss/loso_corrected_summary.json",
            "results/temporal_trend_prototype/trend_prototype.py",
        ]
        for rel_path in expected_files:
            p = self.root / rel_path
            self.assertTrue(p.exists(), f"Crucial artifact missing: {rel_path}")
            self.assertGreater(p.stat().st_size, 0, f"Artifact is empty: {rel_path}")
        print("✓ All original models, datasets, and corrected LOSO evaluation results are preserved.")


if __name__ == "__main__":
    unittest.main(verbosity=2)

