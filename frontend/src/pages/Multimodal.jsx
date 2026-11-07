import { useMemo, useState } from "react";
import DecisionPanel from "../components/DecisionPanel";
import FusionTimeline from "../components/FusionTimeline";
import data from "../content/fusion/aligned.json";
import summary from "../content/fusion/summary.json";
import {
  FUSION_METHODS,
  METHOD_DESCRIPTIONS,
  columnFor,
  strategyFor,
} from "../content/fusionConfig";
import styles from "./Multimodal.module.css";

const SUBJECTS = Object.keys(data.subjects);
const TOTAL_WINDOWS = Object.values(data.subjects).reduce((n, rows) => n + rows.length, 0);
const pct = (v) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(1)}%`);
const num = (v) => (v === null || v === undefined ? "—" : Number(v).toFixed(3));

const SUMMARY_COLUMNS = [
  ["method", "Method"],
  ["n_subjects", "Subjects"],
  ["bacc_mean", "BAcc"],
  ["bacc_sd", "BAcc SD"],
  ["macro_f1_mean", "Macro-F1"],
  ["macro_f1_sd", "F1 SD"],
  ["auc_mean", "AUC"],
  ["auc_sd", "AUC SD"],
  ["auc_ci_lo", "AUC CI low"],
  ["auc_ci_hi", "AUC CI high"],
];

export default function Multimodal() {
  const [subject, setSubject] = useState(SUBJECTS[0]);
  const [method, setMethod] = useState(FUSION_METHODS[0]);
  const [win, setWin] = useState(0);

  const rows = useMemo(
    () =>
      data.subjects[subject].map((values) =>
        Object.fromEntries(data.columns.map((c, i) => [c, values[i]])),
      ),
    [subject],
  );

  const row = rows[Math.min(win, rows.length - 1)];
  const detection = row.detection_fraction ?? 1;
  const raw = row[columnFor(method)];
  const fused = raw === null || raw === undefined ? null : Math.min(1, Math.max(0, raw));
  const strategy = strategyFor(method, detection);
  const faceOk = detection >= 0.8;
  const truth = row.y_true;
  const match = fused !== null && truth === (fused >= 0.5 ? 1 : 0);

  return (
    <div className={styles.page}>
      <header className={styles.head}>
        <div>
          <h1 className={styles.title}>Multimodal Stress Prediction &amp; Late Fusion</h1>
          <p className={styles.sub}>
            Physiology-First decision policy with dynamic quality gating and video fallback
          </p>
        </div>
        <span className={styles.badge}>Fusion arbitrator · 70:30 validated ratio</span>
      </header>

      <section className={styles.controls}>
        <label className={styles.field}>
          <span className={styles.label}>Participant ({SUBJECTS.length} Aligned Cohort)</span>
          <select
            className={styles.select}
            value={subject}
            onChange={(e) => {
              setSubject(e.target.value);
              setWin(0);
            }}
          >
            {SUBJECTS.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>

        <label className={styles.field}>
          <span className={styles.label}>Fusion Decision Strategy</span>
          <select
            className={styles.select}
            value={method}
            onChange={(e) => setMethod(e.target.value)}
          >
            {FUSION_METHODS.map((m) => (
              <option key={m} value={m}>
                {METHOD_DESCRIPTIONS[m]}
              </option>
            ))}
          </select>
        </label>

        <label className={styles.field}>
          <span className={styles.label}>
            Synchronized Window <span className={styles.rangeVal}>{win}</span>
          </span>
          <input
            className={styles.range}
            type="range"
            min={0}
            max={rows.length - 1}
            value={win}
            onChange={(e) => setWin(Number(e.target.value))}
          />
        </label>
      </section>

      <div className={styles.weights}>
        {strategy.w === null ? (
          <p className={styles.weightHead}>
            Arbitration Mode:&nbsp;
            <strong>Stacked Logistic Meta-Learner (Inner-LOSO Coefficients)</strong>
          </p>
        ) : (
          <>
            <div className={styles.weightHead}>
              <span>
                Physiology Authority: <strong>{Math.round(strategy.w[0] * 100)}%</strong>
              </span>
              <span>
                Video Supporting: <strong>{Math.round(strategy.w[1] * 100)}%</strong>
              </span>
            </div>
            <div className={styles.weightBar}>
              <div className={styles.weightPhys} style={{ width: `${strategy.w[0] * 100}%` }} />
              <div className={styles.weightVid} style={{ width: `${strategy.w[1] * 100}%` }} />
            </div>
          </>
        )}
      </div>

      <section className={styles.metrics}>
        <div className={styles.card}>
          <p className={styles.metricLabel}>Physiology P(Stress)</p>
          <p className={styles.metricValue}>{pct(row.p_physio)}</p>
          <p className={styles.metricNote}>Primary Source</p>
        </div>
        <div className={styles.card}>
          <p className={styles.metricLabel}>Video P(Stress)</p>
          <p className={styles.metricValue}>{pct(row.p_video)}</p>
          <p className={styles.metricNote}>Secondary Supporting</p>
        </div>
        <div className={styles.card}>
          <p className={styles.metricLabel}>Fused Decision</p>
          <p className={styles.metricValue}>{pct(fused)}</p>
          <p
            className={styles.metricNote}
            data-tone={fused === null ? undefined : fused >= 0.5 ? "high" : "good"}
          >
            {fused === null ? "—" : fused >= 0.5 ? "STRESS" : "REST"}
          </p>
        </div>
        <div className={styles.card}>
          <p className={styles.metricLabel}>Ground Truth</p>
          <p className={styles.metricValue}>{truth === 1 ? "Stress" : "Rest"}</p>
          <p className={styles.metricNote} data-tone={match ? "good" : "high"}>
            {match ? "✓ MATCH" : "✗ MISMATCH"}
          </p>
        </div>
      </section>

      <div className={styles.callout}>
        <strong>Active Decision Explanation:</strong> {strategy.text}.<br />
        Quality Gates: BVP/EDA signal nominal (valid hardware recording) · Face landmark detection
        quality ={" "}
        <span className={faceOk ? styles.good : styles.warnText}>
          {faceOk ? "SATISFACTORY (>=80%)" : `DEGRADED (${Math.round(detection * 100)}%)`}
        </span>
        .<br />
        <em>
          Validation Notice: Probabilities represent synchronized out-of-fold benchmark predictions
          evaluated on UBFC-Phys. No artificial dropout fallback is simulated on valid benchmark
          data.
        </em>
      </div>

      <DecisionPanel row={row} />

      <FusionTimeline rows={rows} current={Math.min(win, rows.length - 1)} />

      <section className={styles.card}>
        <h2 className={styles.sectionTitle}>
          Late Fusion Benchmark Across {SUBJECTS.length} Participants ({TOTAL_WINDOWS} Windows)
        </h2>
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                {SUMMARY_COLUMNS.map(([key, label]) => (
                  <th key={key}>{label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {summary.map((r) => (
                <tr key={r.method}>
                  {SUMMARY_COLUMNS.map(([key]) => (
                    <td key={key}>
                      {key === "method" || key === "n_subjects" ? r[key] : num(r[key])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}