import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { TABS } from "../content/tabs";
import styles from "./Layout.module.css";

// What the data on screen comes from. Pages change it through the Outlet context.
const SOURCES = {
  replay: "Benchmark replay active",
  recording: "Uploaded recording · physiology API",
};

export default function Layout() {
  const [source, setSource] = useState("replay");

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div className={styles.brand}>
          <svg viewBox="0 0 24 24" aria-hidden="true" className={styles.logo}>
            <path
              d="M12 2v20M3.3 7l17.4 10M3.3 17L20.7 7"
              stroke="currentColor"
              strokeWidth="2.4"
              strokeLinecap="round"
              fill="none"
            />
          </svg>
          <div className={styles.brandText}>
            <span>StressSense</span>
            <small className={styles.tagline}>Multimodal &amp; Stress Monitoring</small>
          </div>
        </div>

        <nav className={styles.nav} aria-label="Dashboard sections">
          {TABS.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) => `${styles.tab} ${isActive ? styles.active : ""}`}
            >
              <Icon size={16} aria-hidden="true" />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        {/* Always visible, and it must match the data on screen: replay of stored
            benchmark results, or a recording analysed by the physiology API. */}
        <div className={styles.status} data-source={source} role="status">
          <span className={styles.statusDot} aria-hidden="true" />
          {SOURCES[source]}
        </div>
      </header>

      <main>
        <Outlet context={{ setSource }} />
      </main>
    </div>
  );
}