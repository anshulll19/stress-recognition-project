import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import styles from "./TrendChart.module.css";

const STEP_S = 5; // seconds between windows
const COLORS = {
  history: "#9a9aab",
  active: "#6d57d8",
  current: "#d63b3b",
  grid: "#e4e4ee",
  text: "#6b6b7b",
};
const TASK_FILL = { 1: "#efecfa", 2: "#fde3dc", 3: "#d9f3ef" };
const TASK_NAME = { 1: "T1 · Rest", 2: "T2 · Speech", 3: "T3 · Arithmetic" };

function Tip({ active, payload, task }) {
  if (!active || !payload?.length) return null;
  const { t, p } = payload[0].payload;
  const i = t / STEP_S;
  return (
    <div className={styles.tip}>
      Window {i} · {t}s
      <br />
      P(distress) {p == null ? "—" : `${(p * 100).toFixed(1)}%`}
      <br />
      {TASK_NAME[task[i]] ?? "—"}
    </div>
  );
}

// probs: probability per window (null when missing); task: task id per window;
// win: current window index; K: rolling window length; summary: text for screen readers and the caption.
export default function TrendChart({ probs, task, win, K, summary }) {
  const activeStart = Math.max(0, win - K + 1);
  const maxT = (probs.length - 1) * STEP_S;
  const ticks = Array.from(
    { length: Math.floor(maxT / 60) + 1 },
    (_, i) => i * 60,
  );

  // Causal: only windows up to the current one are drawn. The x axis stays fixed so it doesn't jump.
  const rows = useMemo(
    () =>
      probs.slice(0, win + 1).map((p, i) => ({
        t: i * STEP_S,
        p,
        history: i <= activeStart ? p : null,
        active: i >= activeStart ? p : null,
      })),
    [probs, win, activeStart],
  );

  // Shaded bands for each task segment seen so far.
  const bands = useMemo(() => {
    const out = [];
    let s = 0;
    for (let i = 1; i <= win + 1; i++) {
      if (i === win + 1 || task[i] !== task[s]) {
        out.push({
          task: task[s],
          x1: s * STEP_S - STEP_S / 2,
          x2: (i - 1) * STEP_S + STEP_S / 2,
        });
        s = i;
      }
    }
    return out;
  }, [task, win]);

  const current = probs[win];

  return (
    <figure className={styles.figure}>
      <figcaption className={styles.caption}>{summary}</figcaption>
      <div className={styles.plot}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={rows}
            margin={{ top: 8, right: 16, bottom: 20, left: 4 }}
          >
            <CartesianGrid stroke={COLORS.grid} strokeDasharray="3 4" />
            {bands.map((b, i) => (
              <ReferenceArea
                key={i}
                x1={Math.max(0, b.x1)}
                x2={b.x2}
                fill={TASK_FILL[b.task]}
                fillOpacity={0.6}
                ifOverflow="hidden"
              />
            ))}
            <ReferenceLine y={0.5} stroke={COLORS.text} strokeDasharray="2 4" />
            <XAxis
              type="number"
              dataKey="t"
              domain={[0, maxT]}
              ticks={ticks}
              tick={{ fill: COLORS.text, fontSize: 12 }}
              label={{
                value: "Time from recording start (seconds)",
                position: "insideBottom",
                offset: -12,
                fill: COLORS.text,
                fontSize: 12,
              }}
            />
            <YAxis
              domain={[0, 1]}
              ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
              tickFormatter={(v) => v.toFixed(1)}
              tick={{ fill: COLORS.text, fontSize: 12 }}
              label={{
                value: "Distress probability",
                angle: -90,
                position: "insideLeft",
                offset: 12,
                fill: COLORS.text,
                fontSize: 12,
              }}
            />
            <Tooltip content={<Tip task={task} />} />
            <Line
              dataKey="history"
              stroke={COLORS.history}
              strokeWidth={2}
              strokeDasharray="5 4"
              dot={{ r: 3, fill: COLORS.history }}
              isAnimationActive={false}
            />
            <Line
              dataKey="active"
              stroke={COLORS.active}
              strokeWidth={2.5}
              dot={{ r: 3, fill: COLORS.active }}
              isAnimationActive={false}
            />
            {current != null && (
              <ReferenceDot
                x={win * STEP_S}
                y={current}
                r={6}
                fill={COLORS.current}
                stroke="#fff"
                strokeWidth={2}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <ul className={styles.legend} aria-label="Chart legend">
        <li>
          <span
            className={`${styles.swatchLine} ${styles.dashed}`}
            style={{ borderColor: COLORS.history }}
          />
          Past history
        </li>
        <li>
          <span
            className={styles.swatchLine}
            style={{ borderColor: COLORS.active }}
          />
          Active {K}-window slope range
        </li>
        <li>
          <span
            className={styles.swatchDot}
            style={{ background: COLORS.current }}
          />
          Current window
        </li>
        {[1, 2, 3].map((id) => (
          <li key={id}>
            <span
              className={styles.swatchBand}
              style={{ background: TASK_FILL[id] }}
            />
            {TASK_NAME[id]}
          </li>
        ))}
      </ul>
    </figure>
  );
}
