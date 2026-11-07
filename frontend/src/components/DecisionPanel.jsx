import { useMemo, useState } from "react";
import {
  LABELS,
  decide,
  physiologyBranch,
  videoBranch,
  windowRef,
} from "../content/decisionPolicy";
import styles from "./DecisionPanel.module.css";

const TITLES = {
  physiology_primary: "Result determined by PHYSIOLOGY",
  video_only_fallback: "Result from VIDEO ONLY (fallback)",
  insufficient_data: "INSUFFICIENT DATA: no result",
};

const nice = (label) => LABELS[label] ?? label;
const whole = (v) => `${Math.round(v * 100)}%`;

function Branch({ title, pred, assessment }) {
  return (
    <section className={styles.branch}>
      <h4 className={styles.branchTitle}>{title}</h4>
      {!pred.available ? (
        <p>Not available</p>
      ) : (
        <>
          <p>Prediction: {nice(pred.label)}</p>
          {Object.entries(pred.probabilities).map(([key, value]) => (
            <div key={key} className={styles.bar}>
              <div className={styles.barHead}>
                <span>{nice(key)}</span>
                <strong>{whole(value)}</strong>
              </div>
              <div className={styles.track}>
                <div className={styles.fill} style={{ width: `${value * 100}%` }} />
              </div>
            </div>
          ))}
          <p className={styles.note}>
            {pred.flags.includes("quality_not_measured_in_replay")
              ? "Signal quality: not measured in replay data (assumed nominal)"
              : `Signal quality: ${pred.quality.toFixed(2)}`}
          </p>
          <p className={styles.note}>Window {pred.ref} | Source: REPLAY</p>
        </>
      )}
      <p className={assessment.reliable ? styles.ok : styles.bad}>
        {assessment.reliable ? "Reliable" : `Not reliable: ${assessment.reasons.join("; ")}`}
      </p>
    </section>
  );
}

export default function DecisionPanel({ row }) {
  const [simPhys, setSimPhys] = useState(false);
  const [simVid, setSimVid] = useState(false);

  const { phys, vid, d, stamp } = useMemo(() => {
    const ref = windowRef(row);
    const p = physiologyBranch(row, ref, simPhys);
    const v = videoBranch(row, ref, simVid);
    return { phys: p, vid: v, d: decide(p, v), stamp: new Date().toISOString() };
  }, [row, simPhys, simVid]);

  return (
    <section className={styles.panel}>
      <h3 className={styles.heading}>Decision policy (physiology-first)</h3>
      <p className={styles.replay}>
        REPLAY: prerecorded UBFC-Phys out-of-fold predictions. Not live sensing.
      </p>

      <div className={styles.sims}>
        <label className={styles.check}>
          <input type="checkbox" checked={simPhys} onChange={(e) => setSimPhys(e.target.checked)} />
          Simulate physiology dropout
        </label>
        <label className={styles.check}>
          <input type="checkbox" checked={simVid} onChange={(e) => setSimVid(e.target.checked)} />
          Simulate video face loss
        </label>
      </div>

      <h4 className={styles.verdict}>{TITLES[d.mode]}</h4>
      {d.banner && <p className={styles.banner}>{d.banner}</p>}

      <div className={styles.facts}>
        <div>
          <p className={styles.factLabel}>Result</p>
          <p className={styles.factValue}>{d.label ? nice(d.label) : "No result"}</p>
        </div>
        <div>
          <p className={styles.factLabel}>Determined by</p>
          <p className={styles.factValue}>{d.determiningBranch ?? "none"}</p>
        </div>
        <div>
          <p className={styles.factLabel}>Data source</p>
          <p className={styles.factValue}>{d.source.toUpperCase()}</p>
        </div>
      </div>

      <p className={styles.explain}>{d.explanation}</p>
      {d.warnings.map((w) => (
        <p key={w} className={styles.warn}>
          {w}
        </p>
      ))}
      <p className={styles.caption}>
        Decision time (UTC): {stamp}. Labels are task-derived proxies, not validated stress or
        distress labels.
      </p>

      <div className={styles.branches}>
        <Branch title="Physiology (primary)" pred={phys} assessment={d.physiologyAssessment} />
        <Branch
          title={
            d.mode === "video_only_fallback"
              ? "Video, upper-face (fallback, determining)"
              : "Video, upper-face (supporting)"
          }
          pred={vid}
          assessment={d.videoAssessment}
        />
      </div>

      <p className={styles.caption}>
        The benchmark blend above is an offline fusion score. The decision here follows the
        physiology-first rule: video never overrides physiology.
      </p>
    </section>
  );
}