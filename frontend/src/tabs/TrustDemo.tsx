// Tab 1 — Trust Demo. Owner: D.
// TODO(D):
//  - base image picker (api.demoSamples), corruption dropdown, severity slider 0-5
//  - image panel (<img className="pixelated">), ResNet prediction + raw confidence + top-k
//  - baseline ladder: raw -> temp-scaled clean -> temp-scaled corrupted -> p_correct
//  - <DecisionBadge />, 2-3 SHAP reasons
export default function TrustDemo() {
  return (
    <section>
      <h2>Trust Demo</h2>
      <p className="todo">TODO(D): picker, severity slider, prediction panel, baseline ladder, decision badge, reasons.</p>
    </section>
  );
}
