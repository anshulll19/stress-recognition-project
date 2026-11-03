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

function Orb() {
  return (
    <svg viewBox="0 0 400 400" className={styles.orb} aria-hidden="true">
      <defs>
        <radialGradient id="orbGlow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#a78bfa" stopOpacity="0.55" />
          <stop offset="60%" stopColor="#d8d2ec" stopOpacity="0.35" />
          <stop offset="100%" stopColor="#efecfa" stopOpacity="0" />
        </radialGradient>
      </defs>
      <circle cx="200" cy="200" r="190" fill="url(#orbGlow)" />
      <circle cx="200" cy="200" r="150" fill="none" stroke="#b8aed8" strokeOpacity="0.5" />
      <circle cx="200" cy="200" r="110" fill="none" stroke="#b8aed8" strokeOpacity="0.6" />
      <circle cx="200" cy="200" r="70" fill="#9e94c4" fillOpacity="0.35" />
      <path
        d="M40 200 H130 L150 160 L180 250 L205 130 L230 220 L250 200 H360"
        fill="none"
        stroke="#6d57d8"
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function Overview() {
  const clock = useClock();

  return (
    <div className={styles.overview}>
      <div className={styles.layout}>
        <div className={styles.main}>
          <section className={styles.hero}>
            <h1 className={styles.heroTitle}>
              Multimodal<span>&amp;</span>
              <br />
              Stress Monitoring
            </h1>
            <p className={styles.heroSub}>{O.subtitle}</p>
            <Orb />
            {O.highlights.map((h, i) => (
              <div key={h.label} className={`${styles.chip} ${styles["chip" + i]}`}>
                <span className={styles.chipValue}>{h.value}</span>
                <span className={styles.chipLabel}>{h.label}</span>
              </div>
            ))}
          </section>

          <section className={styles.engines}>
            {O.engines.map((e) => (
              <div key={e.label} className={styles.card}>
                <p className={styles.engineLabel}>{e.label}</p>
                <p className={styles.engineValue}>{e.value}</p>
                <p className={styles.engineNote}>{e.note}</p>
              </div>
            ))}
          </section>

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
        </div>

        <aside className={styles.side}>
          <section className={styles.panel}>
            <div className={styles.panelHead}>
              <div>
                <h2 className={styles.panelTitle}>{O.trajectory.title}</h2>
                <p className={styles.panelSub}>{O.trajectory.subtitle}</p>
              </div>
              <span className={styles.pill} data-tone="glass">
                {O.trajectory.badge}
              </span>
            </div>
            <div className={styles.plot}>Trajectory chart goes here once the session data is wired in.</div>
          </section>

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

          <section className={`${styles.card} ${styles.alert}`}>
            <h2 className={styles.cardTitle}>{O.integrity.title}</h2>
            <p className={styles.body} style={{ marginTop: "0.75rem" }}>
              <Rich parts={O.integrity.parts} />
            </p>
          </section>

          <aside className={styles.callout}>
            <strong>{O.architecture.title}</strong> {O.architecture.text}
          </aside>
        </aside>
      </div>

      <p className={styles.notice}>
        <strong>Research Notice:</strong> {O.notice}
      </p>
    </div>
  );
}