from pathlib import Path
import pandas as pd

root = Path("data/ubfc_phys_subset/s1")
pilot = root / "pilot"

phys = pd.read_csv(pilot / "t1_physiological_windows.csv")
multi = pd.read_csv(pilot / "t1_multimodal_pilot.csv")
quality = pd.read_csv(root / "frames" / "visual_quality_features.csv")

# Aggregate quality indicators over the same overlapping windows.
rows = []
for _, window in phys.iterrows():
    start, end = window["start_s"], window["end_s"]
    selected = quality[
        (quality["timestamp_s"] >= start) &
        (quality["timestamp_s"] < end)
    ]

    rows.append({
        "start_s": start,
        "end_s": end,
        "quality_frames": len(selected),
        "mean_intensity_mean": selected["mean_intensity"].mean(),
        "intensity_std_mean": selected["intensity_std"].mean(),
        "laplacian_variance_mean": selected["laplacian_variance"].mean(),
    })

quality_windows = pd.DataFrame(rows)

# Merge using the window boundaries.
final = multi.merge(
    quality_windows,
    on=["start_s", "end_s"],
    how="left",
    validate="one_to_one",
)

final["label"] = "unlabelled"

output = pilot / "t1_multimodal_pilot_final.csv"
final.to_csv(output, index=False)

print("Final pilot dataset")
print("Rows:", len(final))
print("Columns:", len(final.columns))
print("Duplicate window boundaries:", final.duplicated(["start_s", "end_s"]).sum())
print("Missing values in quality features:")
print(final[[
    "mean_intensity_mean",
    "intensity_std_mean",
    "laplacian_variance_mean",
]].isna().sum().to_string())
print("Output:", output)
print("Labels remain unlabelled.")
