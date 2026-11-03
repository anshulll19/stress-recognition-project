import { Globe, HeartPulse, ScanFace, Layers, TrendingUp } from "lucide-react";

// One place to rename, reorder or add tabs.
export const TABS = [
  { to: "/", label: "Overview", icon: Globe, end: true },

  { to: "/physiological", label: "Physiological Monitoring", icon: HeartPulse },
  { to: "/facial", label: "Facial Analysis", icon: ScanFace },
  { to: "/multimodal", label: "Multimodal Prediction", icon: Layers },
  { to: "/trends", label: "Trends", icon: TrendingUp }, // TODO: use the full name of your last tab
];