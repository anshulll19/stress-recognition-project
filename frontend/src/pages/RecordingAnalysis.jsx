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

// The v5 model reports "rest" and "distress". The older labels are kept so other model versions still render.
const CLASS_LABEL = {
  distress: "Distress",
  rest: "Rest",
  inconclusive: "Inconclusive",
  stress: "Stress",
  "non-stress": "Non-stress",
  unreliable: "Unreliable",
};
const CLASS_TONE = {
  distress: "high",
  rest: "good",
  inconclusive: "warn",
  stress: "high",
  "non-stress": "good",
  unreliable: "warn",
};
const isDistress = (state) => state === "distress" || state === "stress";

const score = (v) => (v == null ? "—" : Number(v).toFixed(2));
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

function Warnings({ items }) {
  if (!items || items.length === 0) return null;
  return (
    <ul className={styles.hint}>
      {items.map((w, i) => (
        <li key={i}>{typeof w === "string" ? w : JSON.stringify(w)}</li>
      ))}
    </ul>
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
    <>
      <div className={styles.scroll} role="region" aria-label="Per-window results" tabIndex={0}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th scope="col">Window</th>
              <th scope="col">Result</th>
              <th scope="col">Decision score</th>
            </tr>
          </thead>
          <tbody>
            {windows.map((w) => (
              <tr key={w.window_index}>
                <td>{w.start_s}–{w.end_s}s</td>
                <td data-tone={CLASS_TONE[w.class_name]}>
                  {CLASS_LABEL[w.class_name] ?? w.class_name}
                </td>
                <td>{score(w.decision_score)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className={styles.hint}>
        The decision score is the raw output of the model, not a probability. Scores above 0 lean
        towards distress and scores below 0 towards rest.
      </p>
    </>
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
  const calibrated = Boolean(calibration) && !calibration.abstained;
  const anyTask = TASKS.some((t) => results[t.id] && !results[t.id].abstained);
  const taskSummaries = summary?.tasks ? Object.values(summary.tasks) : [];

  return (
    <div className={styles.root} key={round}>
      <p className={styles.banner}>
        Uploaded recording, analysed by the physiology API. This uses the v5 model, which reads the
        EDA signal only: the BVP file is accepted but not used for the result. It is calibrated to
        the rest baseline and works on 30-second windows, which is different from the replay
        views. Research output, not a clinical diagnostic instrument.
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
          {calibrated && <span className={styles.done}>Calibrated</span>}
        </div>
        <p className={styles.hint}>
          At least 180 seconds of continuous rest. BVP at 64 Hz and EDA at 4 Hz, as two files. The
          model calibrates on the first 85 seconds of the EDA signal.
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
        {calibration?.abstained && (
          <div className={styles.error} role="status">
            <span>
              The baseline could not be used ({words(calibration.status)}). Check that the EDA
              signal is clean and long enough, then upload the baseline again.
            </span>
            <Warnings items={calibration.warnings} />
          </div>
        )}
        {calibrated && (
          <>
            <div className={styles.stats}>
              <Stat
                label="Calibration windows"
                value={calibration.calibration_windows ?? "—"}
                note="30-second EDA windows from the rest baseline"
              />
              <Stat
                label="Calibration span"
                value={
                  calibration.calibration_duration_s == null
                    ? "—"
                    : `${Number(calibration.calibration_duration_s).toFixed(0)} s`
                }
                note="These windows set the rest reference, so they do not measure accuracy."
              />
            </div>
            <Warnings items={calibration.warnings} />
          </>
        )}
      </section>

      {TASKS.map((t, i) => {
        const r = results[t.id];
        const abstained = Boolean(r?.abstained);
        return (
          <section key={t.id} className={styles.step} aria-labelledby={`rec-${t.id}`}>
            <div className={styles.stepHead}>
              <h2 id={`rec-${t.id}`} className={styles.stepTitle}>
                {i + 3}. {t.name} ({t.id})
              </h2>
              {r && !abstained && <span className={styles.done}>Analysed</span>}
            </div>
            <p className={styles.hint}>At least 40 seconds. Needs a calibrated baseline first.</p>
            <FilePair
              disabled={!calibrated || working}
              onBvp={(f) => setFile(t.id, "bvp", f)}
              onEda={(f) => setFile(t.id, "eda", f)}
            />
            <div className={styles.actions}>
              <button
                type="button"
                className={styles.btn}
                onClick={() => analyse(t.id)}
                disabled={!calibrated || working || !ready(t.id)}
              >
                {busy === t.id ? "Analysing…" : "Analyse"}
              </button>
            </div>
            {r && abstained && (
              <div className={styles.error} role="status">
                <span>
                  No result for this recording ({words(r.status)}). The signal did not pass the
                  quality checks, so the model did not make a prediction.
                </span>
                <Warnings items={r.warnings} />
              </div>
            )}
            {r && !abstained && (
              <>
                <div className={styles.stats}>
                  <Stat
                    label="Windows analysed"
                    value={r.summary.total_windows}
                    note="Signal quality is checked for the whole recording, not per window."
                  />
                  <Stat
                    label="Distress windows"
                    value={percent(r.summary.stress_percentage)}
                    note="Share of the analysed windows"
                    tone={isDistress(r.summary.dominant_state) ? "high" : undefined}
                  />
                  <Stat label="Mean decision score" value={score(r.summary.mean_decision_score)} />
                  <Stat label="Dominant state" value={words(r.summary.dominant_state)} />
                </div>
                <Warnings items={r.warnings} />
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
        {summary && (
          <>
            {taskSummaries.length > 0 && (
              <div className={styles.stats}>
                {taskSummaries.map((s) => (
                  <Stat
                    key={s.task_id}
                    label={`${s.task_name} (${s.task_id})`}
                    value={percent(s.stress_percentage)}
                    note={`Distress windows · mean decision score ${score(s.mean_decision_score)}`}
                    tone={isDistress(s.dominant_state) ? "high" : undefined}
                  />
                ))}
              </div>
            )}
            {summary.interpretation && <p className={styles.hint}>{summary.interpretation}</p>}
          </>
        )}
      </section>
    </div>
  );
}
