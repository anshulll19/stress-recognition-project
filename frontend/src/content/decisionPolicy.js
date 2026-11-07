export const LABELS = {
  rest_task_proxy: "Non-stress (rest-task proxy)",
  stress_task_proxy: "Stress-related (task proxy)",
};

const MIN_QUALITY = { physiology: 0.5, video: 0.5 };
const BLOCKING_FLAGS = ["signal_lost", "sensor_disconnected", "face_not_detected"];
const LOW_DETECTION_FRACTION = 0.8;
const PROB_SUM_TOLERANCE = 0.02;

const whole = (v) => `${Math.round(v * 100)}%`;
const nice = (label) => LABELS[label] ?? label;

// Assumes src/adapters._probs(p) returns {rest_task_proxy: 1 - p, stress_task_proxy: p}.
const probsFrom = (p) => ({ rest_task_proxy: 1 - p, stress_task_proxy: p });

function topLabel(probabilities) {
  let best = null;
  for (const [key, value] of Object.entries(probabilities)) {
    if (best === null || value > probabilities[best]) best = key;
  }
  return best;
}

export const windowRef = (row) =>
  `${row.subject_id}/T${row.task}/${row.window_start_sec}-${row.window_end_sec}s`;

export function physiologyBranch(row, ref, simulateDropout) {
  if (simulateDropout) {
    return { branch: "physiology", available: false, flags: ["signal_lost"], ref };
  }
  const probabilities = probsFrom(row.p_physio);
  return {
    branch: "physiology",
    available: true,
    probabilities,
    label: topLabel(probabilities),
    // The aligned file has no per-window physiology quality, so nominal quality is assumed.
    quality: 1,
    flags: ["quality_not_measured_in_replay"],
    ref,
  };
}

export function videoBranch(row, ref, simulateLoss) {
  if (simulateLoss) {
    return { branch: "video", available: false, flags: ["face_not_detected"], ref };
  }
  const fraction = row.detection_fraction ?? 1;
  const flags = [];
  if (fraction <= 0) flags.push("face_not_detected");
  else if (fraction < LOW_DETECTION_FRACTION) flags.push("low_face_detection");
  const probabilities = probsFrom(row.p_video_upperface);
  return {
    branch: "video",
    available: true,
    probabilities,
    label: topLabel(probabilities),
    quality: fraction,
    flags,
    ref,
  };
}

function schemaProblems(pred) {
  const out = [];
  const entries = Object.entries(pred.probabilities ?? {});
  if (!entries.length) {
    out.push("available but no probabilities supplied");
  } else {
    for (const [key, value] of entries) {
      if (!Number.isFinite(value) || value < 0 || value > 1) {
        out.push(`probability for '${key}' is not a finite value in [0, 1]`);
      }
    }
    const finite = entries.map((e) => e[1]).filter(Number.isFinite);
    const sum = finite.reduce((a, b) => a + b, 0);
    if (finite.length && Math.abs(sum - 1) > PROB_SUM_TOLERANCE) {
      out.push(`probabilities sum to ${sum.toFixed(3)}, expected 1.0`);
    }
  }
  if (!Number.isFinite(pred.quality) || pred.quality < 0 || pred.quality > 1) {
    out.push("quality must be a finite value in [0, 1]");
  }
  return out;
}

function assess(pred, branch) {
  if (!pred.available) return { reliable: false, reasons: ["branch unavailable"] };
  const problems = schemaProblems(pred);
  if (problems.length) {
    return { reliable: false, reasons: problems.map((p) => `invalid output: ${p}`) };
  }
  const reasons = [];
  if (pred.quality < MIN_QUALITY[branch]) {
    reasons.push(
      `signal quality ${pred.quality.toFixed(2)} below threshold ${MIN_QUALITY[branch].toFixed(2)}`,
    );
  }
  const hit = pred.flags.filter((f) => BLOCKING_FLAGS.includes(f));
  if (hit.length) reasons.push(`blocking quality flags: ${hit.join(", ")}`);
  return { reliable: reasons.length === 0, reasons };
}

export function decide(phys, vid) {
  const pa = assess(phys, "physiology");
  const va = assess(vid, "video");
  const base = { source: "replay", physiologyAssessment: pa, videoAssessment: va, warnings: [] };

  if (pa.reliable) {
    const top = phys.probabilities[phys.label];
    let text = `Physiology determined the result (${nice(phys.label)}, ${whole(top)}).`;
    const warnings = [];
    if (va.reliable) {
      const vTop = vid.probabilities[vid.label];
      if (vid.label === phys.label) {
        text += ` Video agrees (${nice(vid.label)}, ${whole(vTop)}) as supporting evidence.`;
      } else {
        text += ` Video suggests '${nice(vid.label)}' (${whole(vTop)}); it is supporting evidence only and does not change the result.`;
        warnings.push("Physiology and video disagree.");
      }
    } else {
      text += ` Video was not used: ${va.reasons.join("; ")}.`;
    }
    return {
      ...base,
      mode: "physiology_primary",
      determiningBranch: "physiology",
      label: phys.label,
      explanation: text,
      warnings,
    };
  }

  if (va.reliable) {
    const vTop = vid.probabilities[vid.label];
    return {
      ...base,
      mode: "video_only_fallback",
      determiningBranch: "video",
      label: vid.label,
      explanation: `Physiology is not usable (${pa.reasons.join("; ")}). Result comes from the video branch only (${nice(vid.label)}, ${whole(vTop)}).`,
      banner:
        "VIDEO-ONLY FALLBACK: physiology unavailable or unreliable, so the result comes from video alone.",
    };
  }

  return {
    ...base,
    mode: "insufficient_data",
    determiningBranch: null,
    label: null,
    explanation: `Insufficient data. Physiology: ${pa.reasons.join("; ")}. Video: ${va.reasons.join("; ")}.`,
  };
}