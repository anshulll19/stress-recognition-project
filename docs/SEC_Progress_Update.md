# Project Progress Update
### Cross-Dataset Robust Physiological Representation Learning for Multimodal Stress Recognition

**Prepared for:** SEC Faculty Update
**Date:** [Insert Date]
**Team:** [Insert Team/Group Name]

---

## 1. Project Overview

Our project investigates whether physiological stress/affect recognition models — built on PPG/BVP (photoplethysmogram / blood volume pulse) signals — can generalize across datasets collected with different devices, protocols, and populations, and whether a more robust physiological representation improves multimodal stress recognition when combined with facial video.

This builds directly on prior published work from our lab (Alghoul et al., 2025, *"Enhancing Generalization in PPG-Based Emotion Recognition with a CNN-TCN-LSTM Model"*), which showed that a hybrid CNN-TCN-LSTM architecture improves generalization *across subjects within a single dataset*. Our work extends this to a question that, to our knowledge, has not been fully addressed for this signal type at this level of specificity: generalization *across datasets*, and whether that robustness transfers into a real multimodal setting.

---

## 2. Central Research Question

**RQ0:** Can cross-dataset learning improve the robustness of physiological representations for stress recognition, and does this robustness improve multimodal stress recognition when physiological signals are combined with facial video?

This umbrella question is broken into five specific, answerable sub-questions (RQ1–RQ5), organized into three connected experimental tracks:

| Track | Focus | Answers |
|---|---|---|
| **Track 1** | Within-dataset physiological baseline | RQ1 |
| **Track 2** | Cross-dataset generalization (train on one dataset, test on another) + pooled training | RQ2, RQ3 |
| **Track 3** | Multimodal stress recognition (physiological + facial video) | RQ4, RQ5 |

---

## 3. Work Completed So Far

### 3.1 Literature Review & Gap Validation
- Reviewed our lab's prior CNN-TCN-LSTM paper (Alghoul et al., 2025) in detail, including its exact architecture, preprocessing pipeline, and reported benchmark (AUC 0.66–0.69 via Leave-One-Subject-Out cross-validation on the PPGE dataset).
- Reviewed two additional relevant papers: a WESAD cross-modality transformer study, and a newly introduced multimodal stress database (Interspeech 2025).
- Conducted a broader literature search to stress-test our assumed research gap. This surfaced three additional papers doing related cross-dataset physiological stress work, which are currently under full-text review to precisely determine what has and hasn't been done, and to ensure our contribution is genuinely novel rather than overlapping with existing work.

### 3.2 Research Design
- Finalized five specific research questions (RQ1–RQ5) and four testable hypotheses (H1–H4), including hypotheses that explicitly allow for null/negative results rather than assuming improvement in advance.
- Defined a full experimental framework connecting all three tracks, with a clear comparison table of planned experiments (E1–E5).
- Produced a complete research design document, structured to be extended into a formal paper.

### 3.3 Dataset Identification & Access
| Dataset | Role | Status |
|---|---|---|
| **PPGE** | Physiological baseline + cross-dataset generalization | Access requested; pending |
| **WESAD** | Physiological baseline + cross-dataset generalization | Public; access secured, download in progress |
| **EmpathicSchool** | Multimodal target domain (synchronized physiological + facial video) | Public (open-access, Zenodo); access request submitted |

### 3.4 Team Organization
Work has been divided across three members, with each workstream currently unblocked and running in parallel:
- **Dataset & preprocessing:** downloading and validating WESAD, building the signal preprocessing pipeline (bandpass filtering, windowing, normalization).
- **Label harmonization:** defining a common stress/affect label across all three datasets' differing label schemes (categorical vs. continuous valence-arousal vs. questionnaire-derived).
- **Literature validation:** completing the full-text review of overlapping papers to finalize the precise scope of our contribution.

---

## 4. Plan for the Coming Days

| Task | Owner | Expected outcome |
|---|---|---|
| Complete WESAD single-subject validation and preprocessing pipeline | Dataset workstream | Confirmed, working pipeline ready to scale to all subjects |
| Finalize label harmonization scheme across all 3 datasets | Label workstream | Documented, justified label-mapping table |
| Complete literature-overlap review (3 papers) | Literature workstream | Confirmation (or refinement) of the exact research gap |
| Follow up on PPGE and EmpathicSchool access requests | Label workstream | Updated access status |
| **Gate decision:** confirm all of the above before starting model training | Team lead | Go/no-go into Track 1 (within-dataset baseline experiments) |

Once this phase is complete, the team moves into **Track 1** — reproducing and establishing within-dataset baselines on PPGE and WESAD using the CNN-TCN-LSTM architecture from our lab's prior work, which forms the foundation for the cross-dataset and multimodal experiments that follow.

---

## 5. Anticipated Timeline (high-level)

1. **Current phase:** Dataset access, preprocessing, label harmonization, and literature validation (in progress)
2. **Next:** Track 1 — within-dataset physiological baselines
3. **Then:** Track 2 — cross-dataset generalization and pooled training
4. **Then:** Track 3 — multimodal stress recognition (physiological + video)
5. **Final:** Analysis, write-up, and software/demo wrapper

---

## 6. Notes for Discussion

- We are being deliberate about validating that our research gap is genuinely novel before committing significant implementation time — several adjacent papers exist, and we want to be precise about how our contribution differs (raw-signal deep learning cross-dataset generalization for dimensional PPG affect recognition, extended into a multimodal setting) rather than overstate novelty.
- We are treating null or negative results (e.g., if cross-dataset training does *not* improve multimodal performance) as valid, reportable findings, not failures — this is reflected explicitly in our hypotheses.
