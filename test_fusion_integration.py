import json
import joblib
import numpy as np

from src.physiology_inference import UBFCInferenceEngine
from src.multimodal_fusion import UpperFaceVideoEngine, PhysiologyFirstFusionEngine

print("\n1. Loading physiology models...")
phys = UBFCInferenceEngine()
print("   Physiology models loaded")

print("\n2. Loading facial-video model...")
video = UpperFaceVideoEngine()
print("   Video model loaded")

print("\n3. Checking model metadata...")
with open("models/video_upperface_features.json") as f:
    video_features = json.load(f)

print("   Video feature configuration:", video_features)

print("\n4. Checking fusion engine constructor...")
fusion = PhysiologyFirstFusionEngine(
    physiology_engine=phys,
    video_engine=video,
    physio_weight=0.30,
    video_weight=0.70,
)

print("   Fusion engine initialized")

print("\nSUCCESS: Both modalities and fusion engine initialized.")
