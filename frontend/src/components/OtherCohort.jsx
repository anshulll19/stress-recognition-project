import { CLAS, CROSS_DATASET, WESAD } from "../content/cohortInfo";
import styles from "./OtherCohort.module.css";

function Rows({ rows }) {
  return (
    <dl className={styles.rows}>
      {rows.map(([key, value]) => (
        <div key={key} className={styles.row}>
          <dt className={styles.key}>{key}</dt>
          <dd className={styles.value}>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

export default function OtherCohort({ dataset }) {
  if (dataset === "wesad") {
    return (
      <div className={styles.stack}>
        <p className={styles.warn}>
          WESAD replay is not available in this dashboard build. This dashboard replays stored
          results and does not run models, and the WESAD window file (data/wesad/wesad_binary.npz)
          is not in this repository, so there are no per-window predictions or waveforms to show.
          UBFC-Phys replay and the decision policy are unaffected.
        </p>
        <section className={styles.card}>
          <h2 className={styles.title}>Deployment checkpoint (from manifest.json)</h2>
          <Rows rows={WESAD.rows} />
          <p className={styles.note}>{WESAD.note}</p>
        </section>
      </div>
    );
  }

  return (
    <div className={styles.stack}>
      <p className={styles.warn}>{CLAS.warning}</p>
      <section className={styles.card}>
        <h2 className={styles.title}>CLAS Dataset &amp; Model Inventory Audit</h2>
        <Rows rows={CLAS.findings} />
      </section>
      <section className={styles.card}>
        <h2 className={styles.title}>Cross-Dataset Transfer Failure Analysis</h2>
        <p className={styles.note}>
          Experiments attempting zero-shot transfer between WESAD and CLAS failed significantly
          (from docs/experiment_log.csv).
        </p>
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Experiment</th>
                <th>Train dataset</th>
                <th>Test dataset</th>
                <th>Accuracy</th>
                <th>Macro F1</th>
                <th>ROC-AUC</th>
                <th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {CROSS_DATASET.map((r) => (
                <tr key={r.experiment}>
                  <td>{r.experiment}</td>
                  <td>{r.train}</td>
                  <td>{r.test}</td>
                  <td>{r.accuracy}</td>
                  <td>{r.f1}</td>
                  <td>{r.auc}</td>
                  <td>{r.outcome}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
