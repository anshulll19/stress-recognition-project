import { useEffect, useState } from "react";
import { OVERVIEW as O } from "../content/overview";
import styles from "./Overview.module.css";

function Rich({ parts }) {
  return parts.map((p, i) => (p.b ? <strong key={i}>{p.t}</strong> : <span key={i}>{p.t}</span>));
}

function useClock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  const p = (n) => String(n).padStart(2, "0");
  return `${now.getFullYear()}-${p(now.getMonth() + 1)}-${p(now.getDate())} ${p(now.getHours())}:${p(now.getMinutes())}:${p(now.getSeconds())}`;
}

function CardHead({ title, subtitle, badge }) {
  return (
    <div className={styles.cardHead}>
      <div>
        <h2 className={styles.cardTitle}>{title}</h2>
        <p className={styles.cardSub}>{subtitle}</p>
      </div>
      {badge && (
        <span className={styles.pill} data-tone="info">
          {badge}
        </span>
      )}
    </div>
  );
}

export default function Overview() {
  const clock = useClock();

  return (
    <div className={styles.overview}>
      <div className={styles.intro}>
        <h1>{O.title}</h1>
        <p>{O.subtitle}</p>
      </div>

      <section className={styles.engines}>
        {O.engines.map((e) => (
          <div key={e.label} className={styles.card}>
            <p className={styles.engineLabel}>{e.label}</p>
            <p className={styles.engineValue}>{e.value}</p>
            <p className={styles.engineNote}>{e.note}</p>
          </div>
        ))}
      </section>

      <div className={styles.grid}>
        <div className={styles.col}>
          <section className={styles.card}>
            <CardHead title={O.status.title} subtitle={O.status.subtitle} badge={O.status.badge} />
            <p className={styles.body}>
              <Rich parts={O.status.intro} />
            </p>
            <ul className={styles.list}>
              {O.status.bullets.map((b) => (
                <li key={b.lead}>
                  <strong>{b.lead}</strong> {b.text}
                </li>
              ))}
            </ul>
          </section>

          <section className={styles.card}>
            <CardHead
              title={O.trajectory.title}
              subtitle={O.trajectory.subtitle}
              badge={O.trajectory.badge}
            />
            <div className={styles.chart}>
              Trajectory chart goes here once we wire in the session data.
            </div>
          </section>
        </div>

        <div className={styles.col}>
          <section className={styles.card}>
            <CardHead title={O.telemetry.title} subtitle={O.telemetry.subtitle} />
            <dl className={styles.rows}>
              <div className={styles.row}>
                <dt className={styles.rowLabel}>System Clock:</dt>
                <dd className={styles.rowValue}>{clock}</dd>
              </div>
              {O.telemetry.rows.map((r) => (
                <div key={r.label} className={styles.row} data-tone={r.tone}>
                  <dt className={styles.rowLabel}>{r.label}</dt>
                  <dd className={styles.rowValue}>{r.value}</dd>
                </div>
              ))}
            </dl>
          </section>

          <section className={`${styles.card} ${styles.alert}`}>
            <h2 className={styles.cardTitle}>{O.integrity.title}</h2>
            <p className={styles.body} style={{ marginTop: "0.75rem" }}>
              <Rich parts={O.integrity.parts} />
            </p>
          </section>

          <section className={styles.card}>
            <h2 className={styles.cardTitle} style={{ marginBottom: "1rem" }}>
              Model Registry
            </h2>
            <dl className={styles.rows}>
              {O.registry.map((m) => (
                <div key={m.name} className={styles.row}>
                  <dt>{m.name}</dt>
                  <dd style={{ margin: 0 }}>
                    <span className={styles.pill} data-tone={m.tone}>
                      {m.status}
                    </span>
                  </dd>
                </div>
              ))}
            </dl>
          </section>

          <aside className={styles.callout}>
            <strong>{O.architecture.title}</strong> {O.architecture.text}
          </aside>
        </div>
      </div>

      <p className={styles.notice}>
        <strong>Research Notice:</strong> {O.notice}
      </p>
    </div>
  );
}