from pathlib import Path
import zipfile

ROOT = Path.cwd()

REQUIRED_FILES = [
    "models/ubfc_deployment/ubfc_phys_uncalibrated_rf.joblib",
    "models/ubfc_deployment/manifest.json",
    "models/ubfc_deployment/loso_summary.json",
    "src/physiology_inference.py",
    "src/multimodal_fusion.py",
    "src/run_multimodal_inference.py",
]

OPTIONAL_FILES = [
    "src/__init__.py",
]

OUTPUT = ROOT / "multimodal_inference_bundle.zip"

included = []
missing = []

with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for relative_path in REQUIRED_FILES + OPTIONAL_FILES:
        file_path = ROOT / relative_path

        if file_path.is_file():
            zf.write(file_path, arcname=relative_path)
            included.append(relative_path)
        elif relative_path in REQUIRED_FILES:
            missing.append(relative_path)

print("\nIncluded files:")
for f in included:
    print("[OK]", f)

if missing:
    print("\nMissing required files:")
    for f in missing:
        print("[MISSING]", f)

print("\nZIP:", OUTPUT)
print("Size:", round(OUTPUT.stat().st_size / 1024, 2), "KB")

if missing:
    print("WARNING: Bundle incomplete.")
else:
    print("SUCCESS: All required files included.")
