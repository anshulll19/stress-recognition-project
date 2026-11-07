import styles from "./BenchmarkTable.module.css";

const show = (v) => {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(3);
  return String(v);
};

export default function DataTable({ label, data }) {
  return (
    <div className={styles.scroll} role="region" aria-label={label} tabIndex={0}>
      <table className={styles.table}>
        <caption className={styles.srOnly}>{label}</caption>
        <thead>
          <tr>
            {data.columns.map((c) => (
              <th key={c} scope="col">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.rows.map((row, i) => (
            <tr key={i}>
              {row.map((v, j) => (
                <td key={j} className={typeof v === "number" ? styles.num : undefined}>
                  {show(v)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
