# Project Workflow
### Cross-Dataset Robust Physiological Representation Learning for Multimodal Stress Recognition

This document is the single end-to-end workflow view of the project — phases, gates, team ownership, and current status in one place. It sits alongside the full Research Proposal / Experimental Design Document (methodology, RQs, hypotheses) and the two are meant to be read together: this one answers "what happens when, and who owns it," that one answers "what exactly are we testing and why."

---

## 1. End-to-End Pipeline (target state)

```
   PPGE            WESAD
     │                │
     └───────┬────────┘
             ▼
     Physiological Baseline (Track 1)
             │
             ▼
   Cross-Dataset Generalization (Track 2)
   ┌─────────┴─────────┐
   ▼                   ▼
 Pooled LOSO      Cross-dataset pretraining
   │                   │
   └─────────┬─────────┘
             ▼
   Robust Physiological Representation
             │
             ▼
       EmpathicSchool (target domain)
        /              \
      BVP              Video
       │                 │
       ▼                 ▼
 Physiological       Visual
   Encoder           Encoder
       │                 │
       └────────┬────────┘
                ▼
            Fusion (Track 3)
                │
                ▼
        Stress Prediction
                │
                ▼
      Write-up + Software Wrapper
```

---

## 2. Phase-Gate Workflow

Each phase has an explicit **gate** — a condition that must be true before moving to the next phase. No phase starts early just because someone has spare time; if blocked, work shifts to a later-but-unblocked task within the same phase (see Section 4, current assignments).

| Phase | Name | Gate to proceed | Owner(s) |
|---|---|---|---|
| **-1** | Gap validation | Literature check confirms the RQ0–RQ5 story is still a genuine, unclaimed contribution (or is narrowed accordingly) | Person 3 |
| **0** | Dataset + label mapping (current) | All 3 datasets' access status resolved (or explicitly deprioritized) + label harmonization table complete | Person 1 (access/data), Person 2 (labels) |
| **1** | Track 1 — Physiological baseline | LOSO results on PPGE and WESAD reproduced/established, matching or explaining deviation from Alghoul et al.'s numbers | Person 1 → hands off to whoever owns modeling |
| **2** | Track 2 — Cross-dataset generalization | PPGE↔WESAD zero-shot + pooled LOSO results in hand; degradation quantified | Physiological/modeling owner |
| **3** | Track 3 — Multimodal (EmpathicSchool) | Video branch decided (Section 10 architecture); T3-A through T3-D all run | Video/fusion owner |
| **4** | Analysis & write-up | All 5 RQs have an answer (including null results) documented against H1–H4 | Team lead + evaluation owner |
| **5** | Software wrapper / demo | Final model(s) wrapped into an inference pipeline | Whoever isn't bottlenecked at that point |

**We are currently in Phase 0**, with Phase -1 running in parallel (not yet closed out).

---

## 3. Track ↔ RQ ↔ Phase Cross-Reference

So nobody loses track of why a given task exists:

| Track | Answers | Depends on |
|---|---|---|
| Track 1 | RQ1 (within-dataset performance) | Phase 0 complete (data + labels) |
| Track 2 | RQ2 (degradation), RQ3 (pooled robustness) | Track 1 complete |
| Track 3 | RQ4 (multimodal benefit), RQ5 (cross-dataset-robust transfer) | Track 2's pretrained encoder + EmpathicSchool access |

RQ5 (T3-D vs. T3-C) is the project's central result — it cannot be attempted until Track 2 produces a usable pretrained encoder AND EmpathicSchool access + label mapping is resolved.

---

## 4. Current Status Snapshot

| Item | Status |
|---|---|
| WESAD access | ✅ Available — in progress (Person 1) |
| PPGE access | ⏳ **Delayed** — no response yet from original authors or Raina's lab. Team proceeding with WESAD-only for Track 1 in the meantime. |
| **Backup dataset (CLAS)** | 🆕 Candidate — 62 subjects, PPG+ECG+EDA+ACC, arousal-valence framing (matches PPGE better than WESAD's categorical labels). Openly hosted on Mendeley Data, no access request needed. **Decision pending**: commit to CLAS as Track 2's second dataset if PPGE doesn't clear by [checkpoint date — lead to set]. |
| EmpathicSchool access | ⏳ Pending (Zenodo request submitted) |
| Label harmonization | 🔲 In progress (Person 2) |
| Literature validation (3 overlapping papers) | 🔲 In progress (Person 3) |
| Track 1 experiments | Not started — blocked on Phase 0 gate |
| Video branch architecture decision | Not started — deferred to Phase 3 |

---

## 5. Current Assignments (Phase 0 / Phase -1, active now)

| Person | Current task | Feeds into |
|---|---|---|
| **Person 1** | Download WESAD → validate single subject (BVP channel, sampling rate, label alignment) → build preprocessing pipeline (Butterworth 0.7–3.7Hz, 60s/5s-overlap windows, z-score). **Full task spec: `WESAD_Task_Explanation.md`** — includes the label/BVP sampling-rate mismatch trap (labels at 700Hz vs. BVP at 64Hz) and the no-cherry-picking rule (only exclude subjects for genuine data-integrity issues, never for accuracy reasons) | Track 1 baseline, once Phase 0 gate clears |
| **Person 2** | Finalize label mapping table across PPGE/WESAD/EmpathicSchool; own follow-up on PPGE + EmpathicSchool access requests | Phase 0 gate; Section 7 of research doc |
| **Person 3** | Full-text read of 3 overlapping cross-dataset papers; complete comparison checklist | Phase -1 gate; confirms/narrows RQ0–RQ5 |
| **Lead (you)** | Integrate all three inputs; decide when Phase -1 and Phase 0 gates are actually satisfied; unblock reassignment if PPGE/EmpathicSchool access stalls too long | Go/no-go into Phase 1 |

---

## 6. Reassignment Triggers (so the team doesn't stall silently)

- If PPGE access is still pending once Person 1 finishes the WESAD pipeline → Person 1 moves to help Track 1 modeling work using WESAD alone, doesn't wait idle.
- If EmpathicSchool access is still pending once Track 2 (Phase 2) is underway → reprioritize Track 3 architecture decisions (Section 10, video branch) using publicly available sample data/documentation, so no time is lost once access lands.
- If Person 3's literature check finds the gap needs narrowing → lead calls a checkpoint before Phase 0 closes, to avoid finalizing label mapping/experiments around a research question that's about to change.

---

## 7. What "Done" Looks Like for This Project

- All 5 RQs answered (including honest null/negative results — H4 explicitly allows for this).
- A working inference pipeline demonstrating the final multimodal model.
- A polished write-up (Steps 1–7 of the Research Proposal document) suitable for a professor/SEC presentation and as the foundation for a paper.
