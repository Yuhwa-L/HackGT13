import type { Decision } from "../types";

export function DecisionBadge({ decision }: { decision: Decision }) {
  return <span className={`badge badge-${decision}`}>{decision.toUpperCase()}</span>;
}
