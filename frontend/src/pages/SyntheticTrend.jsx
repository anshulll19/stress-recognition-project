import { useState } from "react";
import TrendChart from "../components/TrendChart";
import synthetic from "../content/synthetic.json";
import { TREND_LABELS, TREND_RULES } from "../content/trendLabels";
import styles from "./Trends.module.css";

const STEP_S = TREND_RULES.stepSeconds;
const sign = (v) => (v < 0 ? "−" : "+") + Math.abs(v).toFixed(2);
const LABEL_BY_ID = Object.fromEntries(TREND_LABELS.map((l) => [l.id, l]));

export default function SyntheticTrend() {
  const [id, setId] = useState(synthetic.patterns[0].id);
  const [win, setWin] = useState(20);

  const K = synthetic.K;
  const pattern = synthetic.patterns.find((p) => p.id === id);
  const total = pattern.p.length;
  const w = Math.min(win, total - 1);

  const shown = TREND_LABELS[pattern.displayed[w]];
  const planted = LABEL_BY_ID[pattern.planted];
  const sufficient = pattern.sufficient[w];
  const slope = pattern.slope[w];

  let verdict = `Cold start: ${w + 1} of ${K} windows so far, so no trend is shown yet.`;
  if (sufficient) {
    verdict =
      shown.id === planted.id
        ? "The detector agrees with the planted pattern."
        : `The detector shows ${shown.label}, which differs from the planted pattern.`;
  }

  const summary = sufficient
    ? `Synthetic case · Status: ${shown.label.toUpperCase()} (slope ${sign(slope)} logit/60s)`
    : `Synthetic case · Status: ${shown.label.toUpperCase()}`;

  return (
    <>
      <section className={styles.controls}>
        <label className={styles.field}>
          <span className={styles.label}>Planted pattern</span>
          <select
            className={styles.select}
            value={id}
            onChange={(e) => setId(e.target.value)}
          >
            {synthetic.patterns.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </select>
        </label>

        <label className={styles.field}>
          <span className={styles.label}>Window index along pattern: {w}</span>
          <input
            className={styles.range}
            type="range"
            min={0}
            max={total - 1}
            value={w}
            onChange={(e) => setWin(Number(e.target.value))}
            aria-valuetext={`Window ${w}, ${w * STEP_S} seconds`}
          />
        </label>
      </section>

      <p className={styles.notice}>
        These three patterns are cases the detector handles. The caveat above
        lists patterns it failed on (gradual ramp, fluctuating signal, abrupt
        step). Those are not in this selector.
      </p>

      <div className={styles.metrics}>
        <div className={styles.card}>
          <p className={styles.metricLabel}>Planted pattern</p>
          <p className={styles.metricValue}>
            <span
              className={styles.dot}
              style={{ background: planted.color }}
              aria-hidden="true"
            />
            {planted.label}
          </p>
          <p className={styles.metricNote}>
            What the generator put into the series
          </p>
        </div>

        <div className={styles.card}>
          <p className={styles.metricLabel}>Displayed trend</p>
          <p className={styles.metricValue}>
            <span
              className={styles.dot}
              style={{ background: shown.color }}
              aria-hidden="true"
            />
            {shown.label}
          </p>
          <p className={styles.metricNote}>{verdict}</p>
        </div>

        <div className={styles.card}>
          <p className={styles.metricLabel}>Theil-Sen slope</p>
          <p className={styles.metricValue}>
            {sufficient && slope != null ? `${sign(slope)} logit/60s` : "N/A"}
          </p>
          <p className={styles.metricNote}>
            {sufficient && slope != null
              ? `90% CI: [${sign(pattern.lo[w])}, ${sign(pattern.hi[w])}]${pattern.hi[w] - pattern.lo[w] < 0.01 ? ". Narrow because this synthetic series has no noise." : ""}`
              : `Cold start: ${w + 1} of ${K} windows so far`}
          </p>
        </div>

        <div className={styles.card}>
          <p className={styles.metricLabel}>Data sufficiency</p>
          <p className={styles.metricValue}>
            {sufficient ? "Sufficient" : "Insufficient data"}
          </p>
          <p className={styles.metricNote}>
            {pattern.nValid[w]} of {K} windows valid (needs{" "}
            {TREND_RULES.minValidWindows})
          </p>
        </div>
      </div>

      <TrendChart
        probs={pattern.p}
        win={w}
        K={K}
        summary={summary}
        xLabel="Series time (seconds, 5 s per window)"
      />
    </>
  );
}
