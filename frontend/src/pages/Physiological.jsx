import { useEffect, useMemo, useState } from "react";
import SignalChart from "../components/SignalChart";
import OtherCohort from "../components/OtherCohort";
import index from "../content/physio/index.json";
import styles from "./Physiological.module.css";

const loaders = import.meta.glob("../content/physio/p*.json");

const TASKS = {
  1: "T1 — Rest (Baseline)",
  2: "T2 — Speech Preparation / Public Speaking",
  3: "T3 — Mental Arithmetic Stress",
};

const DATASETS = [
  { id: "ubfc", label: "UBFC-Phys (BVP + EDA · 56 Participants · 10s Windows)" },
  { id: "wesad", label: "WESAD (Wrist BVP · Empatica E4 · 15 Subjects · 60s Windows)" },
  { id: "clas", label: "CLAS Sensor Transfer Audit (Permanent Blocker)" },
];

const pct = (v) => `${(v * 100).toFixed(1)}%`;
const isFlat = (s) => s === null || s < 1e-7;
const fixed = (v) => (v === null ? "n/a" : v.toFixed(4));

function Bar({ label, value, tone }) {
  return (
    <div className={styles.bar}>
      <div className={styles.barHead}>
        <span>{label}</span>
        <strong>{pct(value)}</strong>
      </div>
      <div className={styles.track}>
        <div className={styles.fill} data-tone={tone} style={{ width: `${value * 100}%` }} />
      </div>
    </div>
  );
}

export default function Physiological() {
  const [dataset, setDataset] = useState("ubfc");
  const [participant, setParticipant] = useState(index.participants[0]);
  const [model, setModel] = useState(index.models.includes("LSTM") ? "LSTM" : index.models[0]);
  const [win, setWin] = useState(0);
  const [loaded, setLoaded] = useState({ participant: null, data: null });
  const data = loaded.participant === participant ? loaded.data : null;

  useEffect(() => {
    let alive = true;

    loaders[`../content/physio/p${participant}.json`]().then((m) => {
      if (alive) setLoaded({ participant, data: m.default });
    });
    return () => {
      alive = false;
    };
  }, [participant]);

  const total = data ? data.windows.length : 0;
  const w = data ? data.windows[win] : null;
  const bvpData = useMemo(() => (w ? w.bvp.map((v, i) => ({ t: i / 32, v })) : []), [w]);
  const edaData = useMemo(() => (w ? w.eda.map((v, i) => ({ t: i / 4, v })) : []), [w]);

  const pred = w ? w.pred[model] : null;
  const pStress = pred ? pred.p : null;
  const match = pred && w ? pred.c === w.label : null;
  const bvpFlat = w ? isFlat(w.bvpStd) : false;
  const edaFlat = w ? isFlat(w.edaStd) : false;
  const valid = !(bvpFlat || edaFlat);
  const truth = w ? (w.label === 1 ? "STRESS" : "REST") : "";

  return (
    <div className={styles.page}>
      <header className={styles.head}>
        <div>
          <h1 className={styles.title}>Physiological Signal Monitoring &amp; Inference</h1>
          <p className={styles.sub}>
            Raw cardiovascular and sympathetic autonomic nervous system signal intelligence
          </p>
        </div>
        <span className={styles.badge}>Primary inference branch (70% weight)</span>
      </header>

      <fieldset className={styles.cohort}>
        <legend>Physiological Dataset / Hardware Cohort</legend>
        {DATASETS.map((d) => (
          <label key={d.id} className={styles.radio}>
            <input
              type="radio"
              name="dataset"
              checked={dataset === d.id}
              onChange={() => setDataset(d.id)}
            />
            {d.label}
          </label>
        ))}
      </fieldset>

      {dataset !== "ubfc" ? (
        <OtherCohort dataset={dataset} />
      ) : (
        <>
          <section className={styles.controls}>
            <label className={styles.field}>
              <span className={styles.label}>
                Participant <span className={styles.hint}>(5 of 56 in this build)</span>
              </span>
              <select
                className={styles.select}
                value={participant}
                onChange={(e) => {
                  setParticipant(Number(e.target.value));
                  setWin(0);
                }}
              >
                {index.participants.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </label>

            <label className={styles.field}>
              <span className={styles.label}>Temporal Inference Model</span>
              <select
                className={styles.select}
                value={model}
                onChange={(e) => setModel(e.target.value)}
              >
                {index.models.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </label>

            <label className={styles.field}>
              <span className={styles.label}>
                Window Selection (10s stride 5s){" "}
                <span className={styles.rangeVal}>{win}</span>
              </span>
              <input
                className={styles.range}
                type="range"
                min={0}
                max={Math.max(total - 1, 0)}
                value={win}
                onChange={(e) => setWin(Number(e.target.value))}
                disabled={!w}
              />
            </label>
          </section>

          {!w ? (
            <p className={styles.notice}>Loading participant data…</p>
          ) : (
            <>
              <section className={styles.metrics}>
                <div className={styles.card}>
                  <p className={styles.metricLabel}>Predicted State</p>
                  <p
                    className={styles.metricValue}
                    data-tone={pred ? (pred.c === 1 ? "high" : "good") : undefined}
                  >
                    {pred ? (pred.c === 1 ? "STRESS" : "NON-STRESS") : "—"}
                  </p>
                  <p className={styles.metricNote}>{pred ? `Class ${pred.c}` : "No prediction"}</p>
                </div>

                <div className={styles.card}>
                  <p className={styles.metricLabel}>P(Stress)</p>
                  <p className={styles.metricValue}>{pStress === null ? "—" : pct(pStress)}</p>
                  <p className={styles.metricNote}>{TASKS[w.task] ?? "—"}</p>
                </div>

                <div className={styles.card}>
                  <p className={styles.metricLabel}>Ground Truth</p>
                  <p className={styles.metricValue}>{w.label === 1 ? "Stress" : "Rest"}</p>
                  <p
                    className={styles.metricNote}
                    data-tone={match === null ? undefined : match ? "good" : "high"}
                  >
                    {match === null ? "—" : match ? "✓ MATCH" : "✗ MISMATCH"}
                  </p>
                </div>

                <div className={styles.card}>
                  <p className={styles.metricLabel}>Protocol Interval</p>
                  <p className={styles.metricValue}>
                    T{w.task} · {w.start}–{w.end}s
                  </p>
                  <p className={styles.metricNote}>Window #{w.index}</p>
                </div>
              </section>

              <div className={styles.body}>
                <div className={styles.side}>
                  <section className={styles.card}>
                    <h2 className={styles.sectionTitle}>Probability Analysis</h2>
                    {pStress === null ? (
                      <p className={styles.qText}>No prediction for this window and model.</p>
                    ) : (
                      <>
                        <Bar
                          label="Stress Probability"
                          value={pStress}
                          tone={pStress >= 0.5 ? "high" : "teal"}
                        />
                        <Bar label="Rest Probability" value={1 - pStress} tone="teal" />
                      </>
                    )}
                  </section>

                  <section className={styles.card}>
                    <p className={styles.qTitle}>Signal Quality Gate &amp; Telemetry</p>
                    <p className={styles.qState} data-ok={valid}>
                      {valid
                        ? "VALID (Active Waveform Dynamics)"
                        : "DEGRADED (Zero-Variance Flatline)"}
                    </p>
                    <p className={styles.qText}>
                      Window: {w.nBvp} BVP samples (64 Hz) · {w.nEda} EDA samples (4 Hz)
                      <br />
                      BVP Std Dev: <span className={styles.mono}>{fixed(w.bvpStd)}</span> | EDA Std
                      Dev: <span className={styles.mono}>{fixed(w.edaStd)}</span>
                      <br />
                      Flatline check:{" "}
                      <span className={bvpFlat ? styles.fail : styles.pass}>
                        {bvpFlat ? "FAIL (BVP Flatline)" : "PASS"}
                      </span>{" "}
                      ·{" "}
                      <span className={edaFlat ? styles.fail : styles.pass}>
                        {edaFlat ? "FAIL (EDA Flatline)" : "PASS"}
                      </span>
                    </p>
                  </section>
                </div>

                <div className={styles.charts}>
                  <section className={styles.card}>
                    <p className={styles.chartTitle}>
                      BVP Channel · 64 Hz source (drawn at 32 Hz) · 10s Window · Ground Truth:{" "}
                      {truth}
                      {pStress === null ? "" : ` · P(Stress) = ${pct(pStress)}`}
                    </p>
                    <div className={styles.chartBox}>
                      <SignalChart data={bvpData} color="#0e9f8e" name="BVP" unit="V" />
                    </div>
                  </section>

                  <section className={styles.card}>
                    <p className={styles.chartTitle}>
                      Electrodermal Activity (EDA/GSR) · 4 Hz · Tonic/Phasic Skin Conductance
                    </p>
                    <div className={styles.chartBox}>
                      <SignalChart
                        data={edaData}
                        color="#6d57d8"
                        name="EDA"
                        unit="µS"
                        decimals={3}
                      />
                    </div>
                  </section>
                </div>
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}