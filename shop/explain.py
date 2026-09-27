"""One plain-English sentence on why the trust layer decided what it did. Built from the decision, the candidates
and the SHAP reason groups (stability / familiarity / confidence), plus the photo-quality issue when known.
Deterministic, no LLM, so it always matches what the trust layer actually used."""
from shop.config import a_product

SIGNAL_PHRASE = {
    "stability": "its answer changes when the photo is nudged slightly",
    "familiarity": "the photo looks unlike the product photos it learned from",
    "confidence": "the model itself is unsure",
}


def _pct(x):
    return "99%+" if x >= 0.995 else f"{round(100 * x)}%"  # never display certainty


def explain(item):
    decision, p = item["decision"], item["p_correct"]
    raw = item.get("raw_confidence", p)
    why = [SIGNAL_PHRASE[r["signal"]] for r in item.get("reasons") or [] if r.get("signal") in SIGNAL_PHRASE]
    issue = (item.get("quality") or {}).get("label")
    if issue and decision != "trust":
        why = [f"the photo looks {issue}"] + why
    overconfident = raw - p >= 0.25
    if decision == "trust":
        return (f"Safe to act on: the answer holds up under small flips and shifts and the photo looks like ones the "
                f"model knows (p_correct {_pct(p)}).")
    if decision == "caution":
        cands = item.get("candidates") or []
        if len(cands) >= 2:
            tail = f": {why[0]}" if why else ""
            return (f"It could be {a_product(cands[0]['class'])} or {a_product(cands[1]['class'])}{tail}. p_correct {_pct(p)}"
                    + (f" (the model alone claims {_pct(raw)})." if overconfident else "."))
        return (f"Probably {a_product(item['pred_class'])} (p_correct {_pct(p)}), but just below the bar for acting on it "
                "without asking you" + (f": {why[0]}." if why else "."))
    if not why:
        return (f"Too risky to act on: the combined signals put its chance of being right at only {_pct(p)}"
                + (f", even though the model alone claims {_pct(raw)}." if overconfident else "."))
    return (f"Too risky to act on: {' and '.join(why[:2])}. p_correct {_pct(p)}"
            + (f", even though the model alone claims {_pct(raw)}." if overconfident else "."))
