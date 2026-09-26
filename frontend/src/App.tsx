import { useEffect, useState } from "react";
import { api } from "./api";
import { MockBanner } from "./components/MockBanner";
import ResearchDashboard from "./tabs/ResearchDashboard";
import SnapToShop from "./tabs/SnapToShop";
import TrustDemo from "./tabs/TrustDemo";
import type { HealthResponse } from "./types";

type Tab = "demo" | "research" | "shop";

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("demo");

  useEffect(() => {
    api.health().then(setHealth).catch((e) => setHealthError(String(e)));
  }, []);

  const tabs: { id: Tab; label: string; show: boolean }[] = [
    { id: "demo", label: "Trust Demo", show: true },
    { id: "research", label: "Research Dashboard", show: true },
    { id: "shop", label: "Snap to Shop", show: !!health?.shop_enabled },
  ];

  return (
    <div className="app">
      <MockBanner show={!!health?.any_mock} />
      <header>
        <h1>ML Reliability Lab</h1>
        <nav>
          {tabs.filter((t) => t.show).map((t) => (
            <button key={t.id} className={tab === t.id ? "active" : ""} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          ))}
        </nav>
      </header>
      {healthError && <p className="warn">Backend unreachable — using fallback JSON where available.</p>}
      <main>
        {tab === "demo" && <TrustDemo />}
        {tab === "research" && <ResearchDashboard />}
        {tab === "shop" && health?.shop_enabled && <SnapToShop />}
      </main>
    </div>
  );
}
