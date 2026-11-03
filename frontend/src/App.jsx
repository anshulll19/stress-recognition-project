import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import { TABS } from "./content/tabs";

// Placeholder until each tab page is built.
function Placeholder({ title }) {
  return <h1>{title}</h1>;
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        {TABS.map(({ to, label }) => (
          <Route key={to} path={to} element={<Placeholder title={label} />} />
        ))}
      </Route>
    </Routes>
  );
}