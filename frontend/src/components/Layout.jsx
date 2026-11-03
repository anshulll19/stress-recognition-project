import { NavLink, Outlet } from "react-router-dom";
import { TABS } from "../data/tabs";
import styles from "./Layout.module.css";

export default function Layout() {
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
          <span>VitalPulse</span>
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

        {/* Always visible: this build shows benchmark replay data, not live sensing. */}
        <div className={styles.status}>
          <span className={styles.statusDot} aria-hidden="true" />
          Benchmark replay active
        </div>
      </header>

      <main>
        <Outlet />
      </main>
    </div>
  );
}