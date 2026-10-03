# Research Proposal / Experimental Design Document
### Cross-Dataset Robust Physiological Representation Learning for Multimodal Stress Recognition
*(working title — to be refined after methodology is finalized)*

**Status: DRAFT — Sections 7–10 intentionally incomplete pending Step 2 (dataset + label mapping)**

---

## Document build order (do not skip ahead)
- [x] Step 1: Research Questions
- [ ] Step 2: Dataset + label mapping ← **current step**
- [ ] Step 3: Exact experimental design
- [ ] Step 4: Final architecture
- [ ] Step 5: Evaluation / ablations
- [ ] Step 6: Research gap + literature support (final pass)
- [ ] Step 7: Final polished document

---

## 1. Research Title
Cross-Dataset Robust Physiological Representation Learning for Multimodal Stress Recognition *(working title)*

---

## 2. Research Background
- Stress recognition from physiological signals (PPG/BVP in particular) is attractive for real-world use: non-invasive, wearable-friendly, low-cost.
- Most published models are validated **within a single dataset**, typically via subject-wise (LOSO) cross-validation.
- Real deployments involve different subjects, sensors, protocols, and environments than training data — i.e., **dataset/domain shift** — and within-dataset validation does not measure robustness to this.
- Multimodal stress recognition (physiological + facial video) is a natural extension, since facial behavior can complement physiological signals, particularly when physiological signal quality is degraded.
- Open question: does making the *physiological* representation more robust (via cross-dataset training) actually help when that representation is later used inside a *multimodal* system on a new target domain?

---

## 3. Research Gap

> Existing stress and affect recognition studies frequently demonstrate strong performance under within-dataset evaluation. However, performance can deteriorate when models encounter differences in subjects, sensing devices, protocols, and recording environments. Furthermore, it remains important to investigate whether physiological representations that are more robust to dataset variation can provide benefits when physiological information is subsequently combined with facial video for multimodal stress recognition.

**Note:** this gap statement is provisional. See Section 3a (carried over from prior gap-validation work) — several papers already address cross-dataset physiological stress generalization, so the literature-support pass (Step 6) must confirm exactly where the unclaimed space is before this is finalized for a paper/proposal submission.

### 3a. Literature already reviewed (carry-over — needs full-text confirmation, not just abstracts)

| Paper | Overlap | Status |
|---|---|---|
| Alghoul et al. (2025) — CNN-TCN-LSTM on PPGE | Source of our physiological architecture; within-dataset LOSO only | Fully reviewed |
| Cross-Modality Investigation on WESAD Stress Classification | Cross-*modality* (not cross-dataset) transfer within WESAD; no fusion | Fully reviewed |
| A Naturally Elicited Multimodal Stress Database (Interspeech 2025) | New proprietary dataset — context only, not usable | Abstract reviewed |
| "Enhanced PPG-based stress detection: a multivariate cross-dataset..." (SIPD, WESAD, CLAS) | Direct cross-dataset PPG stress validation | **Needs full-text read** |
| "Stressor Type Matters!" (WESAD, SWELL-KW, ForDigitStress, VerBIO) | Pooled multi-dataset training | **Needs full-text read** |
| "A cross-domain framework..." (WESAD, SCIENTISST-MOVE, DREAMER) | Pretrain/fine-tune transfer across stress+emotion datasets | **Needs full-text read** |

---

## 4. Research Objective

### Primary Objective
To investigate whether cross-dataset training can produce robust physiological representations, and whether these representations improve multimodal stress recognition when combined with facial video.

### Secondary Objectives
1. Establish within-dataset PPG/BVP baselines.
2. Quantify cross-dataset performance degradation.
3. Investigate training on heterogeneous physiological datasets.
4. Evaluate physiological-only and video-only stress recognition.
5. Evaluate physiological–video fusion.
6. Compare target-domain physiological representations against cross-dataset-robust representations.
7. Analyze when and why multimodal fusion provides improvements.

---

## 5. Research Questions

**RQ0 (umbrella):** Can cross-dataset learning improve the robustness of physiological representations for stress recognition, and does this robustness improve multimodal stress recognition when physiological signals are combined with facial video?

- **RQ1 (within-dataset):** How effectively can a deep temporal model based on raw PPG/BVP signals recognize stress/affect within individual datasets?
- **RQ2 (cross-dataset generalization):** How much does PPG/BVP-based stress/affect recognition performance degrade under cross-dataset evaluation?
- **RQ3 (robust representation):** Can training on heterogeneous physiological datasets produce a more robust physiological representation than single-dataset training?
- **RQ4 (multimodal benefit):** Does combining physiological signals with facial video improve stress recognition compared with either modality alone?
- **RQ5 (cross-dataset → multimodal transfer):** Does a cross-dataset-robust physiological representation improve multimodal stress recognition in a separate target domain, compared with a target-domain-trained physiological representation?

---

## 6. Hypotheses

- **H1:** Within-dataset evaluation will outperform cross-dataset evaluation due to dataset/domain differences.
- **H2:** Training on heterogeneous physiological datasets will produce representations that generalize better than representations trained on a single dataset.
- **H3:** Multimodal physiological + facial-video models will outperform individual unimodal models under comparable evaluation conditions.
- **H4:** A cross-dataset-robust physiological representation will provide equal or better multimodal performance compared with a physiological encoder trained only on the target dataset.

*(H4 deliberately does not assume improvement — a null or negative result is still a valid, reportable finding.)*

---

## 7. Datasets *(partial — Step 2 in progress)*

| Dataset | Signals | Video | Subjects | Labels | Role | Access status |
|---|---|---|---|---|---|---|
| **PPGE** | PPG (fingertip, 100Hz) | No | 18 | Valence / Arousal (1–9 scale) | Physiological training / generalization | Requested — pending |
| **WESAD** | BVP (wrist, Empatica E4, ~64Hz) + chest ECG/EDA/EMG/RESP/TEMP/ACC | No | 15 | Stress / baseline / amusement / meditation | Physiological training / generalization | Requested — pending |
| **EmpathicSchool** | HR, EDA, TEMP, ACC (Empatica E4) — **confirm raw BVP availability, not just derived HR** | Yes (facial video + landmarks) | 20 (v1) or 30 (v2, ~40hrs) | Stress level, derived from NASA-TLX | Multimodal target domain | **Public / open access** (arXiv 2209.13542, Zenodo) — use v2 (30 subjects) if possible |
| **CLAS** | PPG (256Hz, also ECG/EDA/ACC available) | No | 62 | **Task correctness + stimulus/task tags** — NOT participant-level valence/arousal self-report (see caution below) | Additional cross-dataset generalization dataset (physiological only) | **Public** (IEEE DataPort / Mendeley, no access request) — download in progress |

> **Important caution on CLAS:** despite superficially resembling PPGE (both reference the arousal-valence circumplex model), CLAS's label is **task/stimulus-design-based** — each of its 5 tasks was designed to target a specific quadrant, and CLAS additionally records task *correctness* (interactive tasks) — not a per-participant subjective valence/arousal rating like PPGE's. These are different constructs and **must not be treated as the same supervised label** without an explicit, defensible mapping. CLAS is strong as a *large-N, PPG, timestamped, cross-dataset generalization dataset* (62 subjects, more than PPGE + WESAD combined) — its role should be scoped accordingly, not assumed equivalent to PPGE for direct label transfer until the 4 verification items below are resolved.
>
> **Physical verification still required (from actual downloaded files, not documentation alone):**
> 1. Actual PPG column name/location in the `Data/` files
> 2. Actual participant folder list (confirm N=62, any gaps/exclusions)
> 3. Exact fields in `Block_details` (timing/task metadata)
> 4. Actual contents of `Answers/*.csv` (confirm whether this is task-correctness only, or includes any subjective rating)

> **Important:** These datasets will not be treated as one synchronized multimodal dataset. PPGE and WESAD are used to investigate physiological representation learning and cross-dataset generalization; EmpathicSchool provides the synchronized physiological–video setting for multimodal evaluation.

### Open items for Step 2 completion
- [ ] Confirm EmpathicSchool provides raw BVP (not just processed HR) from the E4 device
- [ ] Confirm PPGE and WESAD access
- [ ] Define exact label harmonization across all three datasets (see below)

### Label harmonization — draft, needs finalizing
| Dataset | Native label | Harmonized target |
|---|---|---|
| WESAD | Categorical: baseline/stress/amusement/meditation | Binary: stress vs. non-stress (native) |
| PPGE | Continuous valence (1–9) / arousal (1–9) | Binary "stress-like": arousal ≥ midpoint AND valence < midpoint (buffer zone excluded) — **provisional, needs validation** |
| EmpathicSchool | NASA-TLX-derived stress level | **Needs definition** — likely binary or ordinal; confirm exact TLX-to-label conversion used by original authors |

This mapping determines whether RQ2/RQ3/RQ5 are legitimate supervised transfer, or require self-supervised/representation-level pretraining instead of label-level transfer. **This decision blocks Sections 8–10 below.**

### Additional decision — window-level label assignment
When a 60-second window straddles a label transition (e.g., part baseline, part stress), a rule is needed:
- **Option A — majority vote:** assign the window the label that covers >50% of its span.
- **Option B — discard:** drop windows that straddle a transition boundary entirely.

**Decision:** *(to be filled in by whoever owns preprocessing — document the choice and reasoning here once made, since it directly affects final sample counts and must be applied identically across PPGE, WESAD, and EmpathicSchool for the cross-dataset comparisons to be valid).*

---

## 8. Overall Research Framework

```
                  PPGE
                    │
                    ├──────────────┐
                    │              │
                    ▼              ▼
            Physiological      Cross-Dataset
               Baseline        Generalization
                    │              │
                    └──────┬───────┘
                           ▼
                 Robust Physiological
                    Representation
                           │
                           ▼
                  EmpathicSchool
                   Target Domain
                    /         \
                   /           \
                 BVP           Video
                  │              │
                  ▼              ▼
          Physiological       Visual
             Encoder          Encoder
                  │              │
                  └──────┬───────┘
                         ▼
                       Fusion
                         │
                         ▼
                 Stress Prediction
```

---

## 9. Experimental Tracks

### Track 1 — Physiological Baseline (answers RQ1)
```
PPGE  → CNN → TCN → LSTM → Prediction
WESAD → CNN → TCN → LSTM → Prediction
```
Evaluation: LOSO, AUC, F1, Accuracy.

### Track 2 — Cross-Dataset Generalization (answers RQ2, RQ3)
- **2A:** PPGE → Model → test on WESAD
- **2B:** WESAD → Model → test on PPGE
- Compare within-dataset vs. cross-dataset performance → quantify degradation.
- **Pooled variants (RQ3):**
  - Option A — Pooled LOSO: combine PPGE + WESAD subjects, LOSO across pool, report per-dataset.
  - Option B — Cross-dataset pretraining: pretrain encoder on PPGE+WESAD, carry into Track 3.

### Track 3 — Multimodal Stress Recognition (answers RQ4, RQ5)
Using EmpathicSchool:
- **T3-A — BVP only:** `BVP → Physiological Encoder → Stress`
- **T3-B — Video only:** `Video → Visual Encoder → Temporal Encoder → Stress`
- **T3-C — Normal multimodal:** `BVP + Video → Fusion → Stress` (physiological encoder trained only on EmpathicSchool)
- **T3-D — Cross-dataset-robust multimodal:** physiological encoder pretrained on PPGE+WESAD (from Track 2) → fine-tuned/applied on EmpathicSchool BVP → fused with Video → Stress

**T3-D vs. T3-C is the central comparison of the entire project (answers RQ5 / H4).**

---

## 10. Model Architecture *(placeholder — do not lock until Step 4)*

### Physiological branch
```
Raw PPG/BVP → CNN → TCN → LSTM → Physiological Representation
```
(Matches Alghoul et al. 2025 layer specs: CNN — 8 filters/kernel 64/stride 4, then 16 filters/kernel 32/stride 2; TCN — 8 filters/kernel 32/dilations [1,2,4,8]; LSTM — 12 units.)

### Video branch
**Not yet decided.** Options to evaluate:
- Raw video (heavier, needs strong compute + likely pretrained CNN backbone)
- Facial landmarks (lighter, EmpathicSchool already provides 68-point landmarks per prior work)
- Facial action units
- Pretrained visual embeddings (e.g., a face-recognition or expression-recognition backbone, frozen or fine-tuned)

Decision depends on compute budget and Track 3 team member's assessment — flag for Step 4.

---

## 11. Experimental Comparison Table

| Experiment | Physiological | Video | Cross-dataset training | Purpose |
|---|---|---|---|---|
| E1 | ✓ | ✗ | ✗ | Baseline (Track 1) |
| E2 | ✓ | ✗ | ✓ | Generalization (Track 2) |
| E3 | ✗ | ✓ | ✗ | Video baseline (Track 3) |
| E4 | ✓ | ✓ | ✗ | Standard multimodal (Track 3, T3-C) |
| E5 | ✓ | ✓ | ✓ | Robust multimodal (Track 3, T3-D) |

*(Ablations to be added after Step 3.)*

---

## 12. Evaluation Metrics

**Primary:** ROC-AUC
**Secondary:** F1-score, Accuracy, Precision, Recall

**For multimodal experiments, additionally consider:**
- Per-modality contribution analysis
- Performance difference (fusion vs. best unimodal)
- Calibration / confidence
- Temporal prediction consistency

---

## 13. Expected Contributions

1. A systematic evaluation of PPG/BVP cross-dataset generalization.
2. Analysis of physiological representation robustness under dataset shift.
3. A multimodal physiological + facial-video stress recognition framework.
4. An investigation of whether cross-dataset physiological robustness transfers to multimodal recognition.
5. Comparative analysis of physiological-only, video-only, and multimodal systems.

*(Deliberately framed as investigative contributions, not performance guarantees — outcomes are not assumed in advance.)*

---

## 14. Limitations

- Small number of subjects in PPGE (18) and WESAD (15).
- Differences in protocols and populations across all three datasets.
- Different physiological sensors (fingertip PPG vs. wrist BVP vs. Empatica E4 HR).
- Different label definitions (valence/arousal vs. categorical stress vs. TLX-derived stress) — label harmonization is a modeling assumption, not ground-truth equivalence.
- Restricted/pending access to PPGE and WESAD at time of writing.
- Potential domain-specific effects not fully separable from genuine generalization ability.

---

## 15. Team Structure

| Member | Responsibility |
|---|---|
| Person 1 | Dataset research + access + preprocessing + label mapping |
| Person 2 | Physiological model — PPG/BVP CNN-TCN-LSTM + Track 1/2 experiments |
| Person 3 | Video branch + multimodal fusion + Track 3 |
| Person 4 | Evaluation, literature validation, experiment tracking + explainability/software |

---

## 16. Immediate Next Steps

1. Complete Step 2: finalize dataset access (PPGE, WESAD), confirm EmpathicSchool raw BVP availability, finalize label harmonization scheme.
2. Complete literature-support pass (Step 6 items pulled forward): full-text read of the 3 overlapping cross-dataset physiological stress papers (Section 3a) to confirm exactly where the unclaimed space is.
3. Only after 1–2 are resolved: lock Sections 8–10 (framework, tracks, architecture) and proceed to Step 3 (exact experimental design).
