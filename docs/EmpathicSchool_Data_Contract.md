# EmpathicSchool Data Contract & Multimodal Integration Specification

## 1. Current Research & Dataset Status
- **Target Dataset**: EmpathicSchool (Multimodal: physiological signals + facial video recordings).
- **Access Status**: Access request submitted via Zenodo; currently **PENDING**.
- **No Physical Files Present**: The repository contains no raw or processed EmpathicSchool files at this stage.
- **Zero-Assumption Principle**: No assumptions are made regarding directory hierarchies, raw column naming, sampling frequencies, video codecs, participant directory naming, or synchronization schemes until physical files are downloaded and verified via `scripts/inspect_empathicschool.py`.

---

## 2. Processing Pipeline Lifecycle: Three-Tier Architecture

To preserve strict separation of concerns and dataset-agnostic modeling, the pipeline is divided into three distinct tiers:

```
[ Tier 1: Raw EmpathicSchool Data ]
       │ (Subject to inspection: unknown directory layout, raw files, CSV/MAT/MP4, unknown fs)
       ▼
[ Tier 2: Dataset-Specific Preprocessing (empathicschool_preprocess.py) ]
       │ - Butterworth bandpass (0.7 - 3.7 Hz)
       │ - Resampling to locked 64 Hz
       │ - Sliding window extraction (60s length, 5s stride / 55s step)
       │ - Per-window z-score normalization
       │ - Video feature extraction / temporal window alignment
       │ - Subject identifier retention and synchronization tagging
       ▼
[ Tier 3: Normalized Internal Research Format (.npz contract) ]
       │ (Dataset-agnostic: model and evaluation consume this directly)
       ▼
[ Downstream Controlled Experiments (E5 - E9) ]
```

---

## 3. Normalized Internal Research Contracts

### 3.1. Unimodal Physiology Contract (`empathicschool_binary.npz`)
Matching the established repository contract implemented in `src/dataset_loader.py`:
- `windows` (`X_physio`): `np.ndarray` of shape `(N, 3840, 1)`, dtype `float32`.
  - 3840 samples corresponds to exactly 60 seconds at the project-standard 64 Hz.
  - Per-window z-score normalized ($\mu=0, \sigma=1$).
- `labels` (`y`): `np.ndarray` of shape `(N,)`, dtype `int` with binary values in `{0, 1}`.
- `subject_ids`: `np.ndarray` of shape `(N,)`, dtype `object` or `string` identifying the participant for each window.

### 3.2. Multimodal Aligned Representation Contract (`empathicschool_multimodal.npz`)
For synchronized physiological and facial-video experiments:
- `physio_windows`: `np.ndarray` of shape `(N, 3840, 1)`, dtype `float32`.
- `video_features` (or `video_segments`): `np.ndarray` of shape `(N, T_video, D_video)` or `(N, D_video_pooled)`, dtype `float32`, aligned to the identical 60-second temporal window as `physio_windows`.
- `labels`: `np.ndarray` of shape `(N,)`, dtype `int` `{0, 1}`, matching the ground-truth state for that synchronized window.
- `subject_ids`: `np.ndarray` of shape `(N,)`, dtype `object`/`string`, identifying the subject.
- `session_ids` / `window_timestamps`: `np.ndarray` of shape `(N, 2)` or metadata array capturing `[start_time, end_time]` or session identifier for traceability and auditing.

---

## 4. Leakage Prevention Protocol (Mandatory)

1. **Strict Subject-Level Splitting**:
   - Splitting into Train, Validation, and Test must strictly be performed across **subjects** (using `GroupShuffleSplit` or `GroupKFold` grouped by `subject_ids`).
   - Under no circumstances may windows from the same subject appear across any partition:
     $$\text{Subjects}(\text{Train}) \cap \text{Subjects}(\text{Validation}) = \emptyset$$
     $$\text{Subjects}(\text{Train}) \cap \text{Subjects}(\text{Test}) = \emptyset$$
     $$\text{Subjects}(\text{Validation}) \cap \text{Subjects}(\text{Test}) = \emptyset$$
2. **Multi-Session Handling**:
   - If a subject participated in multiple sessions or trials, all sessions for that subject must be grouped together under the same subject ID before any train/validation/test split occurs.
3. **Multimodal Co-occurrence**:
   - For any given participant, their physiological windows and their video segments are paired. Both modalities must remain together in the same split. Cross-modality split bleeding (e.g. video in train, physiology in test for the same subject) is prohibited.
4. **Target Dataset Isolation**:
   - In cross-dataset transfer experiments (e.g. E4, E6), the target dataset (EmpathicSchool) is **NEVER** accessed during source training or source early-stopping validation.
   - Validation sets for early stopping must come exclusively from the source dataset(s) via subject-level partitioning.
5. **Class Weighting Isolation**:
   - Class imbalance weights must be computed strictly on the training partition:
     $$w_c = \frac{N_{\text{train}}}{2 \cdot N_{\text{train}, c}}$$
   - Target test class distributions or validation distributions must never influence training weights.

---

## 5. Controlled Multimodal Experimental Design (E5 – E9)

To rigorously test whether cross-dataset pretraining on physiological signals improves target multimodal recognition, experiments E5–E9 must be strictly controlled:

| Exp ID | Model Description | Modality | Encoder Initialization | Evaluation Partition |
| :--- | :--- | :--- | :--- | :--- |
| **E5** | EmpathicSchool Physio Baseline | Physiology only | Random / Scratch | EmpathicSchool Test Split |
| **E6** | Cross-Dataset Pretrained Multimodal | Physio + Video Fusion | Physio: Pretrained on WESAD+CLAS (E4) <br> Video: Identical feature head | EmpathicSchool Test Split |
| **E7** | EmpathicSchool Video Baseline | Video only | Feature extractor / scratch head | EmpathicSchool Test Split |
| **E8** | EmpathicSchool Target-Only Multimodal | Physio + Video Fusion | Physio: Target-only (random/scratch) <br> Video: Identical feature head | EmpathicSchool Test Split |
| **E9** | EmpathicSchool Joint Multimodal Study | Physio + Video Fusion | Systematic comparison across fusion methods | EmpathicSchool Test Split |

### Key Hypothesis & Fair Comparison
- **Direct Controlled Comparison**: **E8** (Target-only physio encoder + video) vs. **E6** (Cross-dataset-pretrained physio encoder + video).
- **Controlled Variables**:
  - Identical test subjects and train/val/test splits.
  - Identical video feature representations and video branch architecture.
  - Identical fusion head (e.g. Concatenate $\to$ Dense $\to$ Softmax).
  - Identical optimization parameters (Adam, learning rate, batch size, early stopping patience).
  - Identical evaluation metrics (Accuracy, Macro F1, Weighted F1, Class 0 F1, Class 1 F1, ROC-AUC).
- **Independent Variable**: Physiological encoder weights (Random target-initialized vs. Pretrained on WESAD+CLAS source datasets).
