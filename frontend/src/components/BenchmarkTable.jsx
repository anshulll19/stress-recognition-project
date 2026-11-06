import { AlertTriangle, CheckCircle2, XCircle } from "lucide-react";
import { BENCHMARKS, BENCHMARK_FOOTNOTE, STATUS } from "../content/benchmarks";
import styles from "./BenchmarkTable.module.css";

const ICONS = { good: CheckCircle2, warn: AlertTriangle, high: XCircle };

const pct = (v) => (v == null ? "—" : `${v.toFixed(1)}%`);
const auroc = (v) => (v == null ? "—" : v.toFixed(3));

export default function BenchmarkTable() {
  return (
    <div>
      <div className={styles.scroll} role="region" aria-label="Benchmark results" tabIndex={0}>
        <table className={styles.table}>
          <caption className={styles.srOnly}>
            Accuracy, balanced accuracy and AUROC for each model and validation protocol
          </caption>
          <thead>
            <tr>
              <th scope="col">Model / Pipeline</th>
              <th scope="col">Protocol</th>
              <th scope="col">Accuracy</th>
              <th scope="col">Balanced Acc</th>
              <th scope="col">AUROC</th>
              <th scope="col">Validation Status</th>
            </tr>
          </thead>
          <tbody>
            {BENCHMARKS.map((row) => {
              const status = STATUS[row.status];
              const Icon = ICONS[status.tone];
              const star = row.devSplit ? " *" : "";
              return (
                <tr key={row.model}>
                  <th scope="row" className={`${styles.model}`}>
                    {row.model}
                    {row.note && <span className={styles.note}>{row.note}</span>}
                  </th>
                  <td>{row.protocol}</td>
                  <td className={styles.num}>{pct(row.accuracy)}{row.accuracy != null && star}</td>
                  <td className={styles.num}>{pct(row.balancedAcc)}{row.balancedAcc != null && star}</td>
                  <td className={styles.num}>{auroc(row.auroc)}{star}</td>
                  <td>
                    <span className={`${styles.pill} ${styles[status.tone]}`}>
                      <Icon size={14} aria-hidden="true" />
                      {status.label}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className={styles.footnote}>{BENCHMARK_FOOTNOTE}</p>
    </div>
  );
}