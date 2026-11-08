import { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import styles from "./TrajectoryChart.module.css";

const TASKS = [
  { id: 1, name: "Rest (T1)", color: "#1fae5b" },
  { id: 2, name: "Speech (T2)", color: "#d63b3b" },
  { id: 3, name: "Math (T3)", color: "#c98206" },
];
const WINDOW_SECONDS = 10;
const pct = (v) => `${(v * 100).toFixed(1)}%`;

// The CSV restarts the clock for every task, so tasks are laid end to end.
function build(rows) {
  const points = [];
  const bands = [];
  let offset = 0;
  for (const task of TASKS) {
    const part = rows
      .map((r, idx) => ({ r, idx }))
      .filter(({ r }) => r.task === task.id)
      .sort((a, b) => a.r.window_start_sec - b.r.window_start_sec);
    if (!part.length) continue;
    const base = part[0].r.window_start_sec;
    for (const { r, idx } of part) {
      points.push({
        idx,
        t: offset + r.window_start_sec - base,
        physio: r.p_physio,
        video: r.p_video_upperface,
        fused: r["p_blend_0.3_uf"],
      });
    }
    const length = part[part.length - 1].r.window_start_sec - base + WINDOW_SECONDS;
    bands.push({ ...task, start: offset, end: offset + length });
    offset += length;
  }
  return { points, bands, end: offset };
}

export default function FusionTimeline({ rows, current }) {
  const { points, bands, end } = useMemo(() => build(rows), [rows]);
  const currentT = points.find((p) => p.idx === current)?.t;

  return (
    <div className={styles.wrap}>
      <div className={styles.chart}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={points} margin={{ top: 8, right: 12, bottom: 4, left: -8 }}>
            <CartesianGrid stroke="#e9e9f1" vertical={false} />
            {bands.map((b) => (
              <ReferenceArea
                key={b.id}
                x1={b.start}
                x2={b.end}
                fill={b.color}
                fillOpacity={0.12}
                stroke="none"
              />
            ))}
            <ReferenceLine y={0.5} stroke="#6b6b7b" strokeDasharray="3 3" />
            {currentT !== undefined && <ReferenceLine x={currentT} stroke="#14141a" />}
            <XAxis
              dataKey="t"
              type="number"
              domain={[0, end]}
              tickFormatter={(v) => `${v}s`}
              stroke="#6b6b7b"
            />
            <YAxis
              domain={[0, 1]}
              ticks={[0, 0.25, 0.5, 0.75, 1]}
              tickFormatter={(v) => `${v * 100}%`}
              stroke="#6b6b7b"
            />
            <Tooltip formatter={(v, name) => [pct(v), name]} labelFormatter={(t) => `${t}s`} />
            <Legend />
            <Line
              dataKey="physio"
              name="Physiology"
              stroke="#0e9f8e"
              strokeWidth={1.6}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              dataKey="video"
              name="Video (Upper-Face)"
              stroke="#c98206"
              strokeWidth={1.4}
              strokeDasharray="5 4"
              dot={false}
              isAnimationActive={false}
            />
            <Line
              dataKey="fused"
              name="Multimodal Fused"
              stroke="#6d57d8"
              strokeWidth={2.6}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <ul className={styles.tasks}>
        {bands.map((b) => (
          <li key={b.id}>
            <span className={styles.dot} style={{ background: b.color }} />
            {b.name}
          </li>
        ))}
      </ul>
    </div>
  );
}