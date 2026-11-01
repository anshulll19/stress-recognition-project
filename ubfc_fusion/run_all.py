"""
StressSense One-Click Demo & Execution Runner
"""
import sys
import os
import subprocess

def banner(title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78 + "\n")

def main():
    banner("STRESSSENSE: MULTIMODAL STRESS RECOGNITION & DIAGNOSTICS")
    print("Select an option to execute:")
    print("  [1] Run Diagnostic & Explainability Engine (CASI Scores + Baseline Attribution + Remedies)")
    print("  [2] Run Controlled Behavioral Validation Audit (Rest vs Speech vs Head Motion vs Math)")
    print("  [3] Re-run Core 18-Subject LOSO Upper-Face Late Fusion Pipeline (Full Benchmark)")
    print("  [4] Run ALL three sequentially")
    print("  [0] Exit")
    
    choice = input("\nEnter choice [1-4, default=1]: ").strip() or "1"
    
    scripts = {
        "1": [("CLINICAL DIAGNOSTIC REPORT ENGINE", "ubfc_pipeline/stresssense_diagnostic_engine.py")],
        "2": [("CONTROLLED BEHAVIORAL VALIDATION AUDIT", "ubfc_pipeline/validate_controlled_conditions.py")],
        "3": [("18-SUBJECT LOSO LATE FUSION PIPELINE", "ubfc_pipeline/run_upperface_fusion.py")],
        "4": [
            ("CLINICAL DIAGNOSTIC REPORT ENGINE", "ubfc_pipeline/stresssense_diagnostic_engine.py"),
            ("CONTROLLED BEHAVIORAL VALIDATION AUDIT", "ubfc_pipeline/validate_controlled_conditions.py"),
            ("18-SUBJECT LOSO LATE FUSION PIPELINE", "ubfc_pipeline/run_upperface_fusion.py")
        ]
    }
    
    if choice not in scripts:
        print("Exiting.")
        return

    for title, script in scripts[choice]:
        banner(f"EXECUTING: {title}")
        ret = subprocess.run([sys.executable, script])
        if ret.returncode != 0:
            print(f"\n[!] Execution of {script} returned non-zero exit code: {ret.returncode}")

if __name__ == "__main__":
    main()
