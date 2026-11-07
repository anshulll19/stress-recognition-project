import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import data from "../content/fusion/aligned.json";
import confounds from "../content/fusion/confounds.json";
import base from "./Multimodal.module.css";
import own from "./Facial.module.css";

const SUBJECTS = Object.keys(data.subjects);
const pct = (v) => `${(v * 100).toFixed(1)}%`;
const num = (v) => (v === null || v === undefined ? "—" : Number(v).toFixed(3));

const MODES = [
  {
    id: "upper",
    label: "Upper-Face (19 features — mouth landmarks removed ✓ Clean affect estimate)",
  },
  {
    id: "full",
    label: "Full-Face (23 features — includes mouth landmarks ⚠ Confounded by speech)",
  },
];

const CONFOUND_COLUMNS = [
  ["feature_set", "Feature set"],
  ["n_features", "Features"],
  ["bacc", "BAcc"],
  ["macro_f1", "Macro-F1"],
  ["auc", "AUC"],
  ["frac_pred1", "Fraction predicted stress"],
];

function ProbBar({ label, value, tone }) {
  return (
    <div className={own.bar}>
      <div className={own.barHead}>
        <span>{label}</span>
        <strong>{pct(value)}</strong>
      </div>
      <div className={own.track}>
        <div className={own.fill} data-tone={tone} style={{ width: `${value * 100}%` }} />
      </div>
    </div>
  );
}

export default function Facial() {
  const [subject, setSubject] = useState(SUBJECTS[0]);
  const [mode, setMode] = useState("upper");
  const [win, setWin] = useState(0);

  const rows = useMemo(
    () =>
      data.subjects[subject].map((values) =>
        Object.fromEntries(data.columns.map((c, i) => [c, values[i]])),
      ),
    [subject],
  );

  const row = rows[Math.min(win, rows.length - 1)];
  const upper = mode === "upper";
  const pFull = row.p_video;
  const pUpper = row.p_video_upperface ?? pFull;
  const pShown = upper ? pUpper : pFull;
  const truth = row.y_true;
  const predicted = pShown >= 0.5 ? 1 : 0;
  const match = predicted === truth;
  const detection = row.detection_fraction ?? 1;
  const shift = Math.abs(pFull - pUpper);

  // The Streamlit chart passed the full-face blend (p_blend_0.3) as "Multimodal Fused".
  // This chart uses the clean upper-face blend so it matches the Overview and the policy.
  const chart = [
    { name: "Physiology (LSTM)", v: row.p_physio, color: "#0e9f8e" },
    { name: "Video Full-Face (23)", v: pFull, color: "#c98206" },
    { name: "Video Upper-Face (19)", v: pUpper, color: "#6d57d8" },
    { name: "Multimodal Fused", v: row["p_blend_0.3_uf"], color: "#14141a" },
  ];

  return (
    <div className={base.page}>
      <header className={base.head}>
        <div>
          <h1 className={base.title}>Facial Behavioral Analysis &amp; Confound Isolation</h1>
          <p className={base.sub}>
            Computer vision facial action units with speech-confound isolation (19-feature
            upper-face)
          </p>
        </div>
        <span className={base.badge}>Supporting inference branch (30% weight)</span>
      </header>

      <fieldset className={own.radios}>
        <legend>Facial Feature Architecture</legend>
        {MODES.map((m) => (
          <label key={m.id} className={own.radio}>
            <input
              type="radio"
              name="featureMode"
              checked={mode === m.id}
              onChange={() => setMode(m.id)}
            />
            {m.label}
          </label>
        ))}
      </fieldset>

      <details className={own.details}>
        <summary>Video Input &amp; Pipeline Verification Status (Upload Unavailable)</summary>
        <div className={own.detailsBody}>
          <p className={own.pending}>
            Video decoding is not implemented in this dashboard build. The deployed Random Forest
            models need precomputed 19- or 23-dimensional feature vectors, and the dashboard does
            not decode raw video on the fly or fabricate predictions. Inference here runs strictly
            on verified UBFC-Phys precomputed benchmark windows (results/video/video_windows.csv).
          </p>
          <div className={own.statusBox}>
            <p className={own.statusTitle}>Hardware Ingestion Status</p>
            <div className={own.statusRow}>
              <span>Live Webcam Capture:</span>
              <span className={own.pill}>Not implemented / disabled</span>
            </div>
            <div className={own.statusRow}>
              <span>Video File Decoding:</span>
              <span className={own.pill}>Unavailable / offline only</span>
            </div>
            <p className={own.small}>
              Real-time webcam ingestion and file-based video decoding need dedicated background
              extraction workers. Active inference is strictly offline replay of verified UBFC-Phys
              benchmark feature recordings.
            </p>
          </div>
        </div>
      </details>

      <section className={base.controls} style={{ gridTemplateColumns: "1.2fr 2fr" }}>
        <label className={base.field}>
          <span className={base.label}>Participant ({SUBJECTS.length} Video Cohort)</span>
          <select
            className={base.select}
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
        <label className={base.field}>
          <span className={base.label}>
            Window Selection (10s Video Window) <span className={base.rangeVal}>{win}</span>
          </span>
          <input
            className={base.range}
            type="range"
            min={0}
            max={rows.length - 1}
            value={win}
            onChange={(e) => setWin(Number(e.target.value))}
          />
        </label>
      </section>

      <section className={base.metrics}>
        <div className={base.card}>
          <p className={base.metricLabel}>Feature Set</p>
          <p className={base.metricValue}>{upper ? "19 Features (Clean)" : "23 Features (Confounded)"}</p>
          <p className={base.metricNote} data-tone={upper ? "good" : "high"}>
            {upper ? "Mouth Removed" : "⚠ Speech Risk"}
          </p>
        </div>
        <div className={base.card}>
          <p className={base.metricLabel}>P(Stress)</p>
          <p className={base.metricValue}>{pct(pShown)}</p>
          <p className={base.metricNote} data-tone={predicted === 1 ? "high" : "good"}>
            {predicted === 1 ? "STRESS" : "REST"}
          </p>
        </div>
        <div className={base.card}>
          <p className={base.metricLabel}>Ground Truth</p>
          <p className={base.metricValue}>{truth === 1 ? "Stress" : "Non-stress"}</p>
          <p className={base.metricNote} data-tone={match ? "good" : "high"}>
            {match ? "✓ MATCH" : "✗ MISMATCH"}
          </p>
        </div>
        <div className={base.card}>
          <p className={base.metricLabel}>Face Detection Quality</p>
          <p className={base.metricValue}>{Math.round(detection * 100)}%</p>
          <p className={base.metricNote} data-tone={detection >= 0.8 ? "good" : "high"}>
            {detection >= 0.8 ? "Good Quality" : "Low Quality"}
          </p>
        </div>
      </section>

      <div className={own.twoCol}>
        <div className={own.side}>
          <section className={base.card}>
            <h2 className={own.sectionTitle}>Model Comparison</h2>
            <ProbBar
              label="Full-Face (23-Feat)"
              value={pFull}
              tone={pFull >= 0.5 ? "high" : "warm"}
            />
            <ProbBar
              label="Upper-Face (19-Feat)"
              value={pUpper}
              tone={pUpper >= 0.5 ? "high" : undefined}
            />
          </section>

          {shift > 0.05 ? (
            <p className={`${own.note} ${own.noteWarn}`}>
              <strong>Mouth Speech Confound:</strong>
              <br />
              Mouth articulation shifts probability by {pct(shift)}.
            </p>
          ) : (
            <p className={`${own.note} ${own.noteInfo}`}>
              Mouth speech confound minimal on this window.
            </p>
          )}
        </div>

        <section className={base.card}>
          <h2 className={own.sectionTitle}>Modality Decision Comparison</h2>
          <div className={own.chartBox}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={chart}
                layout="vertical"
                margin={{ top: 12, right: 48, bottom: 4, left: 8 }}
              >
                <CartesianGrid stroke="#e9e9f1" horizontal={false} />
                <XAxis
                  type="number"
                  domain={[0, 1]}
                  ticks={[0, 0.25, 0.5, 0.75, 1]}
                  tickFormatter={(v) => `${v * 100}%`}
                  stroke="#6b6b7b"
                />
                <YAxis type="category" dataKey="name" width={140} stroke="#6b6b7b" />
                <Tooltip formatter={(v) => [pct(v), "P(Stress)"]} />
                <ReferenceLine x={0.5} stroke="#6b6b7b" strokeDasharray="3 3" />
                <Bar dataKey="v" isAnimationActive={false}>
                  {chart.map((c) => (
                    <Cell key={c.name} fill={c.color} />
                  ))}
                  <LabelList dataKey="v" position="right" formatter={pct} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>

      <section className={base.card}>
        <h2 className={base.sectionTitle}>
          Confound Ablation Benchmark Across 972 Aligned Windows
        </h2>
        <div className={base.tableWrap}>
          <table className={base.table}>
            <thead>
              <tr>
                {CONFOUND_COLUMNS.map(([key, label]) => (
                  <th key={key}>{label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {confounds.map((r) => (
                <tr key={r.feature_set}>
                  {CONFOUND_COLUMNS.map(([key]) => (
                    <td key={key}>
                      {key === "feature_set" || key === "n_features" ? r[key] : num(r[key])}
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