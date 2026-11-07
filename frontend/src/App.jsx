import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import { TABS } from "./content/tabs";
import Overview from "./pages/Overview";
import Multimodal from "./pages/Multimodal";
import Physiological from "./pages/Physiological";
import Trends from "./pages/Trends";

const PAGES = {
  "/": <Overview />,
  "/physiological": <Physiological />,
  "/multimodal": <Multimodal />,
  "/trends": <Trends />,
};

// Placeholder until each tab page is built.
function Placeholder({ title }) {
  return <h1>{title}</h1>;
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        {TABS.map(({ to, label }) => (
          <Route key={to} path={to} element={PAGES[to] ?? <Placeholder title={label} />} />
        ))}
      </Route>
    </Routes>
  );
}