// Tab 2 — Research Dashboard. Owner: D. ALL values come from evaluation.json; no hardcoded numbers.
// TODO(D):
//  - fold selector (default: evaluation.headline_fold)
//  - headline chart: accuracy vs raw confidence vs temp-scaled vs p_correct by severity (by_severity)
//  - reliability diagram with method toggle
//  - risk-coverage curves with AURC in legend
//  - metrics table per method with CIs; per-family LOFO table; CIFAR-10.1 panel; broken-promise callout
export default function ResearchDashboard() {
  return (
    <section>
      <h2>Research Dashboard</h2>
      <p className="todo">TODO(D): charts from /api/evaluation (Recharts).</p>
    </section>
  );
}
