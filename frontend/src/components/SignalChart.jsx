import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function SignalChart({ data, color, name, unit, decimals = 2 }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: -4 }}>
        <CartesianGrid stroke="#e9e9f1" />
        <XAxis
          dataKey="t"
          type="number"
          domain={[0, 10]}
          ticks={[0, 2, 4, 6, 8, 10]}
          tickFormatter={(v) => `${v}s`}
          stroke="#6b6b7b"
        />
        <YAxis
          domain={["auto", "auto"]}
          tickFormatter={(v) => Number(v).toFixed(1)}
          width={44}
          stroke="#6b6b7b"
        />
        <Tooltip
          formatter={(v, n) => [`${Number(v).toFixed(decimals)} ${unit}`, n]}
          labelFormatter={(t) => `${Number(t).toFixed(2)}s`}
        />
        <Line
          dataKey="v"
          name={name}
          stroke={color}
          strokeWidth={1.6}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}