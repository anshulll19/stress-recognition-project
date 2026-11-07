import { Suspense, lazy } from "react";
import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import { TABS } from "./content/tabs";

const PAGES = {
  "/": lazy(() => import("./pages/Overview")),
  "/physiological": lazy(() => import("./pages/Physiological")),
  "/facial": lazy(() => import("./pages/Facial")),
  "/multimodal": lazy(() => import("./pages/Multimodal")),
  "/trends": lazy(() => import("./pages/Trends")),
};

// Fallback for any tab that has no page yet.
function Placeholder({ title }) {
  return <h1>{title}</h1>;
}

export default function App() {
  return (
    <Suspense fallback={<p>Loading…</p>}>
      <Routes>
        <Route element={<Layout />}>
          {TABS.map(({ to, label }) => {
            const Page = PAGES[to];
            return (
              <Route
                key={to}
                path={to}
                element={Page ? <Page /> : <Placeholder title={label} />}
              />
            );
          })}
        </Route>
      </Routes>
    </Suspense>
  );
}
