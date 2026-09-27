"""Shopping assistant: an LLM writes the words and the personalized ranking; code decides what is allowed.

LLM path: the official openai client, Structured Outputs (strict JSON schema), reasoning_effort="none", 5 s timeout.
Configured from env vars or a repo-root .env file: LLM_API_KEY, LLM_MODEL (default gpt-6-luna),
LLM_BASE_URL (default https://api.openai.com/v1). DEMO_OFFLINE=1 forces the offline path.
Offline path (also used on any LLM failure): deterministic ranking + templates, same response shape, llm_used=false.
Server-side enforcement always runs after the LLM: forbidden actions are replaced, unknown or out-of-class product
IDs are dropped, and cart/checkout affordances are stripped when the gate forbids them.
"""
import json
import os
import re

from shop.checkout_policy import checkout_requirements
from shop.config import CHECKOUT_POLICY, ROOT
from shop.explain import explain
from shop.gate import ALLOWED, DEFAULT_ACTION, allowed_actions, cart_allowed, effective_decision, shoppable_classes

ACTIONS = sorted(set().union(*ALLOWED.values()) | {"compare"})
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["reply", "action", "ranked_products", "comparison"],
    "properties": {
        "reply": {"type": "string"},
        "action": {"type": "string", "enum": ACTIONS},
        "ranked_products": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["product_id", "why_for_you"],
            "properties": {"product_id": {"type": "string"}, "why_for_you": {"type": "string"}}}},
        "comparison": {"type": "string"},
    },
}
MAX_PRODUCTS = 4
BUY_INTENT = re.compile(r"\b(buy|purchase|order|check ?out|add (it )?to (my )?cart|pay)\b", re.I)
GENERIC_TIP = "Fill the frame with just the item, on a plain background, in even light."


def _env():
    env = dict(os.environ)
    dotenv = ROOT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return env


def llm_client():
    """(client, model) or None when offline / not configured."""
    env = _env()
    if env.get("DEMO_OFFLINE") == "1" or not env.get("LLM_API_KEY"):
        return None
    from openai import OpenAI
    client = OpenAI(api_key=env["LLM_API_KEY"], base_url=env.get("LLM_BASE_URL") or "https://api.openai.com/v1",
                    timeout=float(env.get("LLM_TIMEOUT", 5)), max_retries=0)
    return client, env.get("LLM_MODEL") or "gpt-6-luna"


# ---- offline (deterministic) ----

def rank_offline(products, profile, catalog_by_id):
    """Budget fit + style-tag overlap + brand match with purchase history; ties by price. Adds why_for_you."""
    owned = [catalog_by_id[h["product_id"]] for h in profile["purchase_history"] if h["product_id"] in catalog_by_id]
    brands = {p["brand"] for p in owned}
    out = []
    for p in products:
        fits = p["price"] <= profile["budget_per_item"]
        style = sorted(set(p["style_tags"]) & set(profile["style_tags"]))
        brand = p["brand"] in brands
        reasons = ([f"within your ${profile['budget_per_item']:.0f} budget"] if fits else
                   [f"${p['price'] - profile['budget_per_item']:.0f} over your usual budget"])
        if style:
            reasons.append(f"matches your {' & '.join(style)} style")
        if brand:
            reasons.append(f"you've bought {p['brand']} before")
        score = (2.0 if fits else 0.0) + len(style) + (1.0 if brand else 0.0)
        out.append((-score, p["price"], {**p, "why_for_you": (lambda t: t[:1].upper() + t[1:])("; ".join(reasons)) + "."}))
    return [p for *_, p in sorted(out, key=lambda t: (t[0], t[1]))]


def gate_note(eff, p_correct, user_message):
    """Code-generated note when the shopper asks to buy but the gate forbids it. The LLM can't suppress it."""
    if eff in ("trust", "trust_confirmed") or not BUY_INTENT.search(user_message or ""):
        return None
    why = ("the photo can't be identified reliably" if eff == "reject"
           else "the model isn't sure which item this is; pick one of the options first")
    return f"Checkout blocked by the trust gate ({eff.upper()}, p_correct {p_correct:.2f}): {why}. The assistant can't override this."


def offline_text(eff, item, profile, ranked, user_message=""):
    name, cands = profile["name"], item["candidates"]
    refused = BUY_INTENT.search(user_message or "") and eff in ("reject", "caution")
    if eff == "reject":
        q = item.get("quality") or {}
        issue = f"The photo looks {q['label']}. " if q.get("label") else ""
        lead = (f"I can't buy this for you, {name}: I can't identify the item reliably, so checkout is locked. "
                if refused else f"Sorry {name}, I can't identify this photo reliably. ")
        return (f"{lead}{issue}Could you retake it? {q.get('tip') or GENERIC_TIP}", "")
    if eff == "caution":
        names = [c["class"] for c in cands]
        picks = {c: next((p for p in ranked if p["class"] == c), None) for c in names}
        comp = " ".join(f"If it's a {c}, the {p['name']} (${p['price']:.2f}) suits you: {p['why_for_you'][:1].lower() + p['why_for_you'][1:]}"
                        for c, p in picks.items() if p)
        lead = "I can't check out until we know which item this is. " if refused else ""
        if len(names) == 1:
            return (lead + f"This is probably a {names[0]}, but I'm not sure enough to go ahead. Is that right?", comp)
        return (lead + f"I'm not sure whether this is a {' or a '.join(names)}. Which one did you mean?", comp)
    top = ranked[0] if ranked else None
    what = item["pred_class"] if eff == "trust" else item["confirmed_class"]
    msg = f"This looks like a {what}." + (f" For you, {name}, I'd start with the {top['name']} (${top['price']:.2f})."
                                          if top else "")
    return (msg, "")


# ---- LLM ----

def _prompt(eff, item, profile, products, user_message):
    rules = {
        "trust": "The identification is verified. Recommend the best products for this shopper; you may suggest adding to cart.",
        "trust_confirmed": "The shopper confirmed the item. Recommend products of that class; you may suggest adding to cart.",
        "caution": "The identification is uncertain. Do NOT recommend buying. Compare the candidate classes in terms of this "
                   "shopper's needs in 'comparison' (one or two sentences) and ask which one they meant.",
        "reject": "The photo cannot be identified reliably. Ask for a better photo; if photo_issue is given, name it "
                  "and use retake_tip. Return no products and an empty comparison.",
    }[eff]
    return [
        {"role": "system", "content": (
            "You are a concise shopping assistant inside a checkout app. A separate trust layer has already decided "
            f"what you may do. Decision: {eff}. Allowed actions: {', '.join(allowed_actions(eff))}. {rules} "
            "Only use product_ids from the provided list; write one short why_for_you per product using the shopper's "
            "budget, style tags and past purchases. Prices are approximate list prices; don't claim they are current. "
            "Never state or imply more certainty than p_correct. "
            "Always respond directly to the shopper's latest message first. If they ask for something not in the allowed "
            "actions (for example buying or checking out when that isn't allowed), say plainly that you can't and why, "
            "in one short sentence, then do what is allowed. Keep 'reply' under 60 words.")},
        {"role": "user", "content": json.dumps({
            "p_correct": round(item["p_correct"], 3), "predicted_class": item["pred_class"],
            "photo_issue": (item.get("quality") or {}).get("label"),
            "retake_tip": (item.get("quality") or {}).get("tip"),
            "candidates": item["candidates"], "confirmed_class": item.get("confirmed_class"),
            "shopper": {k: profile[k] for k in ("name", "budget_per_item", "style_tags")},
            "past_purchases": profile.get("history_items", []),
            "products": [{k: p[k] for k in ("product_id", "class", "name", "brand", "price", "style_tags")} for p in products],
            "shopper_message": user_message or ""})},
    ]


def call_llm(llm, eff, item, profile, products, user_message):
    client, model = llm
    resp = client.chat.completions.create(
        model=model, messages=_prompt(eff, item, profile, products, user_message), reasoning_effort="none",
        response_format={"type": "json_schema", "json_schema": {"name": "shop_reply", "strict": True, "schema": SCHEMA}})
    return json.loads(resp.choices[0].message.content)


def enforce(out, eff, products):
    """Server-side post-processing. Applied to every LLM answer; the LLM never decides what is allowed."""
    allowed = set(allowed_actions(eff))
    action = out.get("action") if out.get("action") in allowed else DEFAULT_ACTION[eff]
    ok = {p["product_id"]: p for p in products}
    ranked, seen = [], set()
    for r in out.get("ranked_products") or []:
        pid = r.get("product_id") if isinstance(r, dict) else None
        if pid in ok and pid not in seen:
            seen.add(pid)
            ranked.append({**ok[pid], "why_for_you": str(r.get("why_for_you", ""))[:200]})
    if eff == "reject":
        ranked = []
    return {"reply": str(out.get("reply", ""))[:600], "action": action, "ranked": ranked,
            "comparison": str(out.get("comparison", ""))[:600] if eff == "caution" else ""}


# ---- entry point ----

def assist(item, profile, catalog, user_message="", confirmed_class=None, llm="auto", policy_cfg=CHECKOUT_POLICY):
    """item: a shop_cache entry. Returns the /api/shop/assist response."""
    by_id = {p["product_id"]: p for p in catalog}
    eff = effective_decision(item["decision"], confirmed_class, item["candidates"])
    item = {**item, "confirmed_class": confirmed_class if eff == "trust_confirmed" else None}
    classes = shoppable_classes(eff, item["pred_class"], item["candidates"], confirmed_class)
    products = [p for p in catalog if p["class"] in classes]
    ranked_off = rank_offline(products, profile, by_id)
    profile = {**profile, "history_items": [by_id[h["product_id"]]["name"] for h in profile["purchase_history"]
                                            if h["product_id"] in by_id]}
    llm = llm_client() if llm == "auto" else llm
    llm_used = False
    if llm is not None:
        try:
            out = enforce(call_llm(llm, eff, item, profile, products, user_message), eff, products)
            llm_used = True
        except Exception as e:  # timeout, bad JSON, API error: fall back, never fail the demo
            print(f"shop assistant: LLM unavailable ({type(e).__name__}: {str(e)[:120]}); offline fallback")
    if not llm_used:
        reply, comp = offline_text(eff, item, profile, ranked_off, user_message)
        out = {"reply": reply, "action": DEFAULT_ACTION[eff], "ranked": [] if eff == "reject" else ranked_off,
               "comparison": comp}
    ranked = out["ranked"] or ([] if eff == "reject" else ranked_off)  # LLM returned none: keep the offline order
    if eff == "caution":  # show each candidate class, best pick first
        ranked = sorted(ranked, key=lambda p: [c["class"] for c in item["candidates"]].index(p["class"]))
        ranked = [p for i, p in enumerate(ranked) if sum(q["class"] == p["class"] for q in ranked[:i]) < 2]
    ranked = ranked[:MAX_PRODUCTS]
    policy = checkout_requirements(eff, item["p_correct"], 0.0, policy_cfg)
    return {
        "decision": item["decision"], "effective_decision": eff, "p_correct": item["p_correct"],
        "allowed_actions": allowed_actions(eff), "action": out["action"], "assistant_message": out["reply"],
        "candidates": item["candidates"], "comparison": out["comparison"],
        "products": [{**{k: p[k] for k in ("product_id", "name", "brand", "price", "class", "style_tags", "why_for_you")},
                      "description": p.get("description", ""), "search_url": p.get("search_url")} for p in ranked],
        "cart_allowed": cart_allowed(eff),
        "checkout_policy": {k: policy[k] for k in ("tier", "max_one_tap_total", "explanation")},
        "gate_note": gate_note(eff, item["p_correct"], user_message),
        "explanation": explain(item),
        "llm_used": llm_used,
    }
