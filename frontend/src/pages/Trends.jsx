import { useEffect, useState } from "react";
import BenchmarkTable from "../components/BenchmarkTable";
import TrendChart from "../components/TrendChart";
import SyntheticTrend from "./SyntheticTrend";
import index from "../content/trend/index.json";
import { TRENDS_COPY as C } from "../content/trends";
import { TREND_LABELS, TREND_RULES } from "../content/trendLabels";
import styles from "./Trends.module.css";

const loaders = import.meta.glob("../content/trend/p*.json");

const TASK_NAME = { 1: "Rest", 2: "Speech", 3: "Arithmetic" };
const STEP_S = TREND_RULES.stepSeconds;
const sign = (v) => (v < 0 ? "−" : "+") + Math.abs(v).toFixed(2);

function RealTrend() {
  const [participant, setParticipant] = useState(index.participants[0]);
  const [model, setModel] = useState(index.models[0]);
  const [win, setWin] = useState(15);
  const [loaded, setLoaded] = useState({ participant: null, data: null });
  const data = loaded.participant === participant ? loaded.data : null;

  useEffect(() => {
    let alive = true;
    loaders[`../content/trend/p${participant}.json`]().then((m) => {
      if (alive) setLoaded({ participant, data: m.default });
    });
    return () => {
      alive = false;
    };
  }, [participant]);

  const K = index.K;
  const series = data ? data.models[model] : null;
  const total = data ? data.task.length : 0;
  const w = Math.min(win, Math.max(total - 1, 0));

  let body = <p className={styles.notice} role="status">Loading participant data…</p>;

  if (data && !series) {
    body = <p className={styles.notice}>No predictions for this participant and model.</p>;
  } else if (data && series) {
    const shown = TREND_LABELS[series.displayed[w]];
    const raw = TREND_LABELS[series.raw[w]];
    const sufficient = series.sufficient[w];
    const slope = series.slope[w];
    const reason =
      w < K - 1
        ? `Cold start: ${w + 1} of ${K} windows so far`
        : `Fewer than ${TREND_RULES.minValidWindows} valid windows in the last ${K}`;
    const status = shown.label.toUpperCase();
    const summary = sufficient && slope != null
      ? `Causal temporal trend analysis (past ${K} windows) · Status: ${status} (slope ${sign(slope)} logit/60s)`
      : `Causal temporal trend analysis (past ${K} windows) · Status: ${status}`;

    body = (
      <>
        <div className={styles.metrics}>
          <div className={styles.card}>
            <p className={styles.metricLabel}>Displayed trend</p>
            <p className={styles.metricValue}>
              <span className={styles.dot} style={{ background: shown.color }} aria-hidden="true" />
              {shown.label}
            </p>
            <p className={styles.metricNote}>
              {shown.id === "stable" && `${C.stableNote} `}
              A new label needs two matching windows to appear.
              {raw.id !== shown.id && ` This window alone reads: ${raw.label}.`}
            </p>
          </div>

          <div className={styles.card}>
            <p className={styles.metricLabel}>Theil-Sen slope</p>
            <p className={styles.metricValue}>
              {sufficient && slope != null ? `${sign(slope)} logit/60s` : "N/A"}
            </p>
            <p className={styles.metricNote}>
              {sufficient && slope != null
                ? `90% CI: [${sign(series.lo[w])}, ${sign(series.hi[w])}]`
                : reason}
            </p>
          </div>

          <div className={styles.card}>
            <p className={styles.metricLabel}>Data sufficiency</p>
            <p className={styles.metricValue}>{sufficient ? "Sufficient" : "Insufficient data"}</p>
            <p className={styles.metricNote}>
              {series.nValid[w]} of {K} windows valid (needs {TREND_RULES.minValidWindows})
            </p>
          </div>

          <div className={styles.card}>
            <p className={styles.metricLabel}>Protocol phase</p>
            <p className={styles.metricValue}>
              T{data.task[w]} · {TASK_NAME[data.task[w]]}
            </p>
            <p className={styles.metricNote}>Window {w} · {w * STEP_S}s from recording start</p>
          </div>
        </div>

        {sufficient && data.spansTasks[w] && (
          <p className={styles.crossNote} role="note">
            The last {K} windows cross a task change, so this slope mixes protocol phases.
          </p>
        )}

        <TrendChart probs={series.p} task={data.task} win={w} K={K} summary={summary} />

        <details className={styles.details}>
          <summary>Detailed trend diagnostic trajectory (all windows)</summary>
          <div className={styles.detailsBody}>
            <div className={styles.scroll} role="region" aria-label="Trend diagnostics" tabIndex={0}>
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th scope="col">Window</th>
                    <th scope="col">Task</th>
                    <th scope="col">Probability</th>
                    <th scope="col">Data sufficient</th>
                    <th scope="col">Slope (logit/60s)</th>
                    <th scope="col">Raw classification</th>
                    <th scope="col">Displayed label</th>
                  </tr>
                </thead>
                <tbody>
                  {data.task.map((t, i) => (
                    <tr key={i} className={i === w ? styles.current : undefined} aria-current={i === w ? "true" : undefined}>
                      <td>{i}</td>
                      <td>T{t}</td>
                      <td>{series.p[i] == null ? "—" : series.p[i].toFixed(3)}</td>
                      <td>{series.sufficient[i] ? "Yes" : "No"}</td>
                      <td>{series.slope[i] == null ? "N/A" : sign(series.slope[i])}</td>
                      <td>{TREND_LABELS[series.raw[i]].label}</td>
                      <td>{TREND_LABELS[series.displayed[i]].label}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </details>
      </>
    );
  }

  return (
    <>
      <section className={styles.controls}>
        <label className={styles.field}>
          <span className={styles.label}>Participant</span>
          <select
            className={styles.select}
            value={participant}
            onChange={(e) => setParticipant(Number(e.target.value))}
          >
            {index.participants.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </label>

        <label className={styles.field}>
          <span className={styles.label}>Model</span>
          <select className={styles.select} value={model} onChange={(e) => setModel(e.target.value)}>
            {index.models.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </label>

        <label className={styles.field}>
          <span className={styles.label}>Time step (window index): {w}</span>
          <input
            className={styles.range}
            type="range"
            min={0}
            max={Math.max(total - 1, 0)}
            value={w}
            onChange={(e) => setWin(Number(e.target.value))}
            disabled={!data}
            aria-valuetext={`Window ${w}, ${w * STEP_S} seconds`}
          />
        </label>
      </section>
      {body}
    </>
  );
}

export default function Trends() {
  const [source, setSource] = useState("real");

  return (
    <div className={styles.page}>
      <header className={styles.head}>
        <div>
          <h1 className={styles.title}>{C.title}</h1>
          <p className={styles.sub}>{C.subtitle}</p>
        </div>
        <span className={styles.badge}>{C.badge}</span>
      </header>

      <section className={styles.section} aria-labelledby="trend-title">
        <div>
          <h2 id="trend-title" className={styles.sectionTitle}>{C.trendTitle}</h2>
          <p className={styles.sub}>{C.trendSubtitle}</p>
        </div>

        <div className={styles.caveat} role="note">
          <strong>{C.caveatTitle}:</strong> {C.caveat}
        </div>

        <fieldset className={styles.source}>
          <legend>{C.sourceLabel}</legend>
          {C.sources.map((s) => (
            <label key={s.id} className={styles.radio}>
              <input
                className={styles.accent}
                type="radio"
                name="trend-source"
                checked={source === s.id}
                onChange={() => setSource(s.id)}
              />
              {s.label}
            </label>
          ))}
        </fieldset>

        {source === "real" ? (
          <RealTrend />
        ) : (
          <SyntheticTrend />
        )}

        <details className={styles.details}>
          <summary>How each trend label is decided</summary>
          <div className={styles.detailsBody}>
            <ul className={styles.rules}>
              {TREND_LABELS.map((l) => (
                <li key={l.id}>
                  <strong>{l.label}:</strong> {l.rule}
                </li>
              ))}
            </ul>
          </div>
        </details>
      </section>

      <section className={styles.section} aria-labelledby="bench-title">
        <div>
          <h2 id="bench-title" className={styles.sectionTitle}>{C.benchmarkTitle}</h2>
          <p className={styles.sub}>{C.benchmarkSubtitle}</p>
        </div>
        <BenchmarkTable />
      </section>
    </div>
  );
}