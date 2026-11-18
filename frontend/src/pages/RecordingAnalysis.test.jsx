import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Outlet, Route, Routes } from "react-router-dom";

// The API module is replaced by mocks, so no backend is needed.
vi.mock("../api/physiology", () => {
  class ApiError extends Error {
    constructor(message, { status = null, kind = "http" } = {}) {
      super(message);
      this.status = status;
      this.kind = kind;
    }
  }
  return {
    ApiError,
    apiConfigured: true,
    createSession: vi.fn(),
    uploadBaseline: vi.fn(),
    calibrateBaseline: vi.fn(),
    runTask: vi.fn(),
    getSummary: vi.fn(),
  };
});

import * as api from "../api/physiology";
import RecordingAnalysis from "./RecordingAnalysis";

const csv = (name) => new File(["1\n2\n3\n"], name, { type: "text/csv" });

const CALIBRATION = {
  abstained: false,
  calibration_windows: 12,
  calibration_duration_s: 85,
  warnings: [],
};

const T2_RESULT = {
  abstained: false,
  warnings: ["BVP is accepted for upload compatibility but is not used by V5 inference."],
  summary: {
    total_windows: 2,
    stress_percentage: 100,
    mean_decision_score: 1.5,
    dominant_state: "distress",
  },
  windows: [
    { window_index: 0, start_s: 0, end_s: 30, class_name: "distress", decision_score: 1.4 },
    { window_index: 1, start_s: 5, end_s: 35, class_name: "distress", decision_score: 1.6 },
  ],
};

const SUMMARY = {
  tasks: {
    T2: {
      task_id: "T2",
      task_name: "Speech task",
      stress_percentage: 100,
      mean_decision_score: 1.5,
      dominant_state: "distress",
    },
  },
  interpretation: "V5 uses raw SVM decision scores, not calibrated probabilities.",
};

function renderPage() {
  return render(
    <MemoryRouter>
      <Routes>
        <Route element={<Outlet context={{ setSource: vi.fn() }} />}>
          <Route index element={<RecordingAnalysis />} />
        </Route>
      </Routes>
    </MemoryRouter>,
  );
}

// File inputs appear in this order: baseline, T2, T3.
const bvpInputs = () => screen.getAllByLabelText(/BVP file/);
const edaInputs = () => screen.getAllByLabelText(/EDA file/);
const button = (name) => screen.getAllByRole("button", { name })[0];

async function startSession(user) {
  await user.type(screen.getByLabelText(/Participant ID/), "p01");
  await user.click(button("Start session"));
  await screen.findByText(/Session started for p01/);
}

async function calibrate(user) {
  await user.upload(bvpInputs()[0], csv("bvp_T1.csv"));
  await user.upload(edaInputs()[0], csv("eda_T1.csv"));
  await user.click(button("Upload and calibrate"));
  await screen.findByText("Calibrated");
}

async function analyseT2AndSummary(user) {
  await user.upload(bvpInputs()[1], csv("bvp_T2.csv"));
  await user.upload(edaInputs()[1], csv("eda_T2.csv"));
  await user.click(screen.getAllByRole("button", { name: "Analyse" })[0]);
  await screen.findByText("Analysed");
  await user.click(button("Show session summary"));
  await screen.findByText(/not calibrated probabilities/);
}

beforeEach(() => {
  vi.resetAllMocks();
  api.createSession.mockResolvedValue({ session_id: "s1", participant_id: "p01" });
  api.uploadBaseline.mockResolvedValue({});
  api.calibrateBaseline.mockResolvedValue(CALIBRATION);
  api.runTask.mockResolvedValue(T2_RESULT);
  api.getSummary.mockResolvedValue(SUMMARY);
});

afterEach(cleanup);

describe("recording analysis workflow (mocked API)", () => {
  it("runs session, baseline calibration, a task and the summary", async () => {
    const user = userEvent.setup();
    renderPage();
    await startSession(user);
    await calibrate(user);

    expect(api.createSession).toHaveBeenCalledWith("p01");
    expect(api.uploadBaseline).toHaveBeenCalledTimes(1);
    expect(screen.getByText("12")).toBeTruthy();
    expect(screen.getByText("85 s")).toBeTruthy();

    await analyseT2AndSummary(user);
    expect(api.runTask).toHaveBeenCalledWith("s1", "T2", expect.any(File), expect.any(File), "p01");
    expect(screen.getByText("Distress windows")).toBeTruthy();
    expect(screen.getByText("Speech task (T2)")).toBeTruthy();
  });

  it("keeps task analysis locked until the baseline is calibrated", async () => {
    const user = userEvent.setup();
    renderPage();
    await startSession(user);
    for (const b of screen.getAllByRole("button", { name: "Analyse" })) {
      expect(b.disabled).toBe(true);
    }
  });

  it("clears calibration, task results and summary when new baseline files are selected", async () => {
    const user = userEvent.setup();
    renderPage();
    await startSession(user);
    await calibrate(user);
    await analyseT2AndSummary(user);

    await user.upload(bvpInputs()[0], csv("bvp_T1_new.csv"));

    expect(screen.queryByText("Calibrated")).toBeNull();
    expect(screen.queryByText("Analysed")).toBeNull();
    expect(screen.queryByText(/not calibrated probabilities/)).toBeNull();
    expect(screen.queryByText("Distress windows")).toBeNull();
    for (const b of screen.getAllByRole("button", { name: "Analyse" })) {
      expect(b.disabled).toBe(true);
    }
    expect(button("Show session summary").disabled).toBe(true);
  });

  it("clears the previous calibration and results as soon as a replacement upload starts", async () => {
    const user = userEvent.setup();
    renderPage();
    await startSession(user);
    await calibrate(user);
    await analyseT2AndSummary(user);

    // Same files, upload again; the upload fails, so no new calibration arrives.
    api.uploadBaseline.mockRejectedValueOnce(new api.ApiError("Upload failed."));
    await user.click(button("Upload and calibrate"));
    await screen.findByText("Upload failed.");

    expect(screen.queryByText("Calibrated")).toBeNull();
    expect(screen.queryByText("Analysed")).toBeNull();
    expect(screen.queryByText(/not calibrated probabilities/)).toBeNull();
    for (const b of screen.getAllByRole("button", { name: "Analyse" })) {
      expect(b.disabled).toBe(true);
    }
  });

  it("does not unlock the tasks when the baseline is rejected", async () => {
    api.calibrateBaseline.mockResolvedValue({
      abstained: true,
      status: "insufficient_signal",
      warnings: ["Baseline too short."],
    });
    const user = userEvent.setup();
    renderPage();
    await startSession(user);
    await user.upload(bvpInputs()[0], csv("bvp_T1.csv"));
    await user.upload(edaInputs()[0], csv("eda_T1.csv"));
    await user.click(button("Upload and calibrate"));

    await screen.findByText(/The baseline could not be used/);
    expect(screen.queryByText("Calibrated")).toBeNull();
    for (const b of screen.getAllByRole("button", { name: "Analyse" })) {
      expect(b.disabled).toBe(true);
    }
  });

  it("shows no result when a task recording is abstained", async () => {
    api.runTask.mockResolvedValue({
      abstained: true,
      status: "unreliable_signal",
      warnings: ["Flat EDA signal."],
    });
    const user = userEvent.setup();
    renderPage();
    await startSession(user);
    await calibrate(user);
    await user.upload(bvpInputs()[1], csv("bvp_T2.csv"));
    await user.upload(edaInputs()[1], csv("eda_T2.csv"));
    await user.click(screen.getAllByRole("button", { name: "Analyse" })[0]);

    await screen.findByText(/No result for this recording/);
    await waitFor(() => expect(screen.queryByText("Analysed")).toBeNull());
    expect(button("Show session summary").disabled).toBe(true);
  });
});
