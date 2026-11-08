import { useEffect, useState } from "react";
import { useOutletContext } from "react-router-dom";
import {
  ApiError,
  apiConfigured,
  calibrateBaseline,
  createSession,
  getSummary,
  runTask,
  uploadBaseline,
} from "../api/physiology";
import styles from "./RecordingAnalysis.module.css";

const TASKS = [
  { id: "T2", name: "Speech task" },
  { id: "T3", name: "Mental arithmetic task" },
];

const CLASS_LABEL = { stress: "Stress", "non-stress": "Non-stress", unreliable: "Unreliable" };
const CLASS_TONE = { stress: "high", "non-stress": "good", unreliable: "warn" };

const prob = (v) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
const percent = (v) => (v == null ? "—" : `${Number(v).toFixed(1)}%`);
const words = (s) => (s ? String(s).replace(/_/g, " ").toLowerCase() : "—");

function Stat({ label, value, note, tone }) {
  return (
    <div className={styles.stat}>
      <p className={styles.statLabel}>{label}</p>
      <p className={styles.statValue} data-tone={tone}>{value}</p>
      {note && <p className={styles.hint}>{note}</p>}
    </div>
  );
}

function FilePair({ disabled, onBvp, onEda }) {
  return (
    <div className={styles.fields}>
      <label className={styles.field}>
        BVP file (64 Hz)
        <input
          className={styles.input}
          type="file"
          accept=".csv,.txt"
          disabled={disabled}
          onChange={(e) => onBvp(e.target.files?.[0] ?? null)}
        />
      </label>
      <label className={styles.field}>
        EDA file (4 Hz)
        <input
          className={styles.input}
          type="file"
          accept=".csv,.txt"
          disabled={disabled}
          onChange={(e) => onEda(e.target.files?.[0] ?? null)}
        />
      </label>
    </div>
  );
}

function WindowTable({ windows }) {
  return (
    <div className={styles.scroll} role="region" aria-label="Per-window results" tabIndex={0}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th scope="col">Window</th>
            <th scope="col">Result</th>
            <th scope="col">P(stress)</th>
            <th scope="col">Signal quality</th>
          </tr>
        </thead>
        <tbody>
          {windows.map((w) => (
            <tr key={w.window_index}>
              <td>{w.start_s}–{w.end_s}s</td>
              <td data-tone={CLASS_TONE[w.class_name]}>
                {CLASS_LABEL[w.class_name] ?? w.class_name}
              </td>
              <td>{prob(w.probability)}</td>
              <td>{w.is_reliable ? "OK" : w.quality_reason || w.quality_flag}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function RecordingAnalysis() {
  const [round, setRound] = useState(0); // changing it clears the file inputs
  const [participant, setParticipant] = useState("");
  const [session, setSession] = useState(null);
  const [files, setFiles] = useState({});
  const [calibration, setCalibration] = useState(null);
  const [results, setResults] = useState({});
  const [summary, setSummary] = useState(null);
  const [busy, setBusy] = useState("");
   const [error, setError] = useState(null);
  const { setSource } = useOutletContext();

  // While this view is open the header must say the data is an uploaded recording.
  // It goes back to "replay" when the view closes, including when you leave the tab.
  useEffect(() => {
    if (!apiConfigured) return undefined;
    setSource("recording");
    return () => setSource("replay");
  }, [setSource]);

  if (!apiConfigured) {
    return (
      <p className={styles.error} role="status">
        The physiology API address is not set. Copy <code>frontend/.env.example</code> to{" "}
        <code>frontend/.env.local</code>, start the backend, and restart <code>npm run dev</code>.
      </p>
    );
  }

  const setFile = (slot, kind, file) =>
    setFiles((prev) => ({ ...prev, [slot]: { ...prev[slot], [kind]: file } }));
  const ready = (slot) => Boolean(files[slot]?.bvp && files[slot]?.eda);

  async function step(name, fn) {
    setBusy(name);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(e instanceof ApiError ? e : new ApiError(e?.message ?? "Something went wrong."));
    } finally {
      setBusy("");
    }
  }

  function reset() {
    setRound((r) => r + 1);
    setSession(null);
    setFiles({});
    setCalibration(null);
    setResults({});
    setSummary(null);
    setError(null);
  }

  const start = () =>
    step("session", async () => {
      setSession(await createSession(participant.trim()));
    });

  const calibrate = () =>
    step("baseline", async () => {
      await uploadBaseline(session.session_id, files.baseline.bvp, files.baseline.eda, session.participant_id);
      setCalibration(await calibrateBaseline(session.session_id));
    });

  const analyse = (id) =>
    step(id, async () => {
      const r = await runTask(session.session_id, id, files[id].bvp, files[id].eda, session.participant_id);
      setResults((prev) => ({ ...prev, [id]: r }));
      setSummary(null);
    });

  const compare = () =>
    step("summary", async () => {
      setSummary(await getSummary(session.session_id));
    });

  const working = busy !== "";
  const anyTask = TASKS.some((t) => results[t.id]);
  const profile = summary?.reactivity_profile;

  return (
    <div className={styles.root} key={round}>
      <p className={styles.banner}>
        Uploaded recording, analysed by the physiology API. This uses a rest-calibrated model on
        30-second windows, which is different from the replay views. Research output, not a
        clinical diagnostic instrument.
      </p>

      {error && (
        <div className={styles.error} role="alert">
          <span>{error.message}</span>
          {error.kind === "session_expired" && (
            <div className={styles.actions}>
              <button type="button" className={styles.ghost} onClick={reset}>
                Start a new session
              </button>
            </div>
          )}
        </div>
      )}

      <section className={styles.step} aria-labelledby="rec-1">
        <div className={styles.stepHead}>
          <h2 id="rec-1" className={styles.stepTitle}>1. Start a session</h2>
          {session && <span className={styles.done}>Session started for {session.participant_id}</span>}
        </div>
        <p className={styles.hint}>
          Sessions are kept in the backend&apos;s memory and are lost when the server restarts.
        </p>
        <div className={styles.fields}>
          <label className={styles.field}>
            Participant ID
            <input
              className={styles.input}
              value={participant}
              maxLength={64}
              disabled={Boolean(session) || working}
              onChange={(e) => setParticipant(e.target.value)}
              placeholder="e.g. P01"
            />
          </label>
        </div>
        <div className={styles.actions}>
          <button
            type="button"
            className={styles.btn}
            onClick={start}
            disabled={Boolean(session) || working || participant.trim() === ""}
          >
            {busy === "session" ? "Starting…" : "Start session"}
          </button>
          {session && (
            <button type="button" className={styles.ghost} onClick={reset} disabled={working}>
              Start over
            </button>
          )}
        </div>
      </section>

      <section className={styles.step} aria-labelledby="rec-2">
        <div className={styles.stepHead}>
          <h2 id="rec-2" className={styles.stepTitle}>2. Rest baseline (T1)</h2>
          {calibration && <span className={styles.done}>Calibrated</span>}
        </div>
        <p className={styles.hint}>
          At least 180 seconds of continuous rest. BVP at 64 Hz and EDA at 4 Hz, as two files.
        </p>
        <FilePair
          disabled={!session || working}
          onBvp={(f) => setFile("baseline", "bvp", f)}
          onEda={(f) => setFile("baseline", "eda", f)}
        />
        <div className={styles.actions}>
          <button
            type="button"
            className={styles.btn}
            onClick={calibrate}
            disabled={!session || working || !ready("baseline")}
          >
            {busy === "baseline" ? "Uploading…" : "Upload and calibrate"}
          </button>
        </div>
        {calibration && (
          <div className={styles.stats}>
            <Stat
              label="Reliable baseline windows"
              value={`${calibration.n_reliable_windows} of ${calibration.n_total_windows}`}
              note={`${calibration.n_unreliable_windows} unreliable windows were left out`}
            />
            <Stat
              label="Mean baseline P(stress)"
              value={prob(calibration.mean_baseline_probability)}
              note="These windows were used to calibrate the model, so a low value here is expected and does not measure accuracy."
            />
          </div>
        )}
      </section>

      {TASKS.map((t, i) => {
        const r = results[t.id];
        return (
          <section key={t.id} className={styles.step} aria-labelledby={`rec-${t.id}`}>
            <div className={styles.stepHead}>
              <h2 id={`rec-${t.id}`} className={styles.stepTitle}>
                {i + 3}. {t.name} ({t.id})
              </h2>
              {r && <span className={styles.done}>Analysed</span>}
            </div>
            <p className={styles.hint}>At least 40 seconds. Needs a calibrated baseline first.</p>
            <FilePair
              disabled={!calibration || working}
              onBvp={(f) => setFile(t.id, "bvp", f)}
              onEda={(f) => setFile(t.id, "eda", f)}
            />
            <div className={styles.actions}>
              <button
                type="button"
                className={styles.btn}
                onClick={() => analyse(t.id)}
                disabled={!calibration || working || !ready(t.id)}
              >
                {busy === t.id ? "Analysing…" : "Analyse"}
              </button>
            </div>
            {r && (
              <>
                <div className={styles.stats}>
                  <Stat
                    label="Reliable windows"
                    value={`${r.summary.reliable_windows} of ${r.summary.total_windows}`}
                    note="Unreliable windows are never counted as non-stress."
                  />
                  <Stat
                    label="Stress windows"
                    value={percent(r.summary.stress_percentage)}
                    note="Share of the reliable windows"
                    tone={r.summary.dominant_state === "stress" ? "high" : undefined}
                  />
                  <Stat label="Mean P(stress)" value={prob(r.summary.mean_stress_probability)} />
                  <Stat label="Dominant state" value={words(r.summary.dominant_state)} />
                </div>
                <WindowTable windows={r.windows} />
              </>
            )}
          </section>
        );
      })}

      <section className={styles.step} aria-labelledby="rec-sum">
        <div className={styles.stepHead}>
          <h2 id="rec-sum" className={styles.stepTitle}>5. Compare tasks</h2>
        </div>
        <div className={styles.actions}>
          <button type="button" className={styles.btn} onClick={compare} disabled={!anyTask || working}>
            {busy === "summary" ? "Loading…" : "Show session summary"}
          </button>
        </div>
        {profile && (
          <>
            <div className={styles.stats}>
              <Stat label="Baseline (T1)" value={prob(profile.baseline_t1_prob)} />
              <Stat label="Speech (T2)" value={prob(profile.speech_t2_prob)} note={`Change vs baseline: ${profile.delta_t2_vs_baseline ?? "—"}`} />
              <Stat label="Arithmetic (T3)" value={prob(profile.arithmetic_t3_prob)} note={`Change vs baseline: ${profile.delta_t3_vs_baseline ?? "—"}`} />
              <Stat label="Reactivity" value={words(profile.reactivity_class)} />
            </div>
            {summary.interpretation && <p className={styles.hint}>{summary.interpretation}</p>}
          </>
        )}
      </section>
    </div>
  );
}