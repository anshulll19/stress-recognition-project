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
import data from "../content/trajectory.json";
import styles from "./TrajectoryChart.module.css";

const TASK_COLORS = { 1: "#1fae5b", 2: "#d63b3b", 3: "#c98206" };
const pct = (v) => `${(v * 100).toFixed(1)}%`;

export default function TrajectoryChart() {
  const { points, tasks } = data;
  const end = points[points.length - 1].t + 10;

  return (
    <div className={styles.wrap}>
      <div className={styles.chart}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={points} margin={{ top: 8, right: 12, bottom: 4, left: -8 }}>
            <CartesianGrid stroke="#e9e9f1" vertical={false} />
            {tasks.map((t) => (
              <ReferenceArea
                key={t.id}
                x1={t.start}
                x2={t.end}
                fill={TASK_COLORS[t.id]}
                fillOpacity={0.12}
              />
            ))}
            <ReferenceLine y={0.5} stroke="#6b6b7b" strokeDasharray="3 3" />
            <XAxis
              dataKey="t"
              type="number"
              domain={[points[0].t, end]}
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
              name="Video (Full)"
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
        {tasks.map((t) => (
          <li key={t.id}>
            <span className={styles.dot} style={{ background: TASK_COLORS[t.id] }} />
            {t.name}
          </li>
        ))}
      </ul>
    </div>
  );
}