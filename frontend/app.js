/* Owner: D. Demo + research page. Reads data/demo_cache.json and data/evaluation.json, served by python -m backend.main.
   Charts are hand-drawn inline SVG so the page needs no libraries and no internet. */
"use strict";

const $ = id => document.getElementById(id);
const pct = (v, d = 1) => v == null ? "n/a" : (100 * v).toFixed(d) + "%";
const pctUI = v => v >= 0.995 ? ">99%" : v < 0.005 ? "<1%" : pct(v, 0);   // the demo never claims certainty
const f3 = v => v == null ? "n/a" : v.toFixed(3);
const FAMILIES = ["noise", "blur", "weather", "digital"];
const famName = f => ({ noise: "Noise", blur: "Blur", weather: "Weather", digital: "Digital", clean: "Clean" })[f] || f;
const CORR_NAME = { gaussian_noise: "Gaussian noise", impulse_noise: "Impulse noise", defocus_blur: "Defocus blur", motion_blur: "Motion blur",
  fog: "Fog", brightness: "Brightness", contrast: "Contrast", jpeg_compression: "JPEG compression" };
const corrName = c => CORR_NAME[c] || c.replace(/_/g, " ");
const M = {   // research-tab names in plain words; small .tech labels carry the technical names
  raw_confidence: { label: "Model's own confidence", color: "--raw" },
  temp_clean: { label: "Rescaled, tuned on clean photos", color: "--tclean" },
  temp_corrupted: { label: "Rescaled, tuned on damaged photos", color: "--tcorr" },
  tta_only: { label: "Stability check only", color: "--tta" },
  trust_layer: { label: "Trust layer's estimate", color: "--trust" },
};
const DECISION = {
  trust: { label: "TRUST", what: "Act on it automatically.", color: "--good", chip: "good", icon: "✓" },
  caution: { label: "CAUTION", what: "Use it, but flag it for review.", color: "--warn", chip: "warn", icon: "!" },
  reject: { label: "REJECT", what: "Don't act on it: send it to a person or a stronger model.", color: "--crit", chip: "crit", icon: "✕" },
};
const state = { cache: null, ev: null, img: null, corr: null, sev: 0, apiToken: 0 };

// ---------------- small DOM helpers ----------------
function h(tag, attrs = {}, kids = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") e.textContent = v; else if (k === "style") e.style.cssText = v; else e.setAttribute(k, v);
  }
  for (const k of [].concat(kids)) if (k != null) e.append(k);
  return e;
}
function S(tag, attrs, parent) {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function key(color, kind = "") { const k = h("span", { class: "key " + kind }); k.style.background = `var(${color})`; return k; }
function legend(host, items) { host.textContent = ""; for (const it of items) host.append(h("span", {}, [key(it.color, it.kind), it.label])); }
function chip(decision) { const d = DECISION[decision]; return h("span", { class: "chip " + d.chip, text: `${d.icon} ${d.label}` }); }
function sigChip(c, higherIsBetter = true) {
  if (!c) return h("span", { class: "chip", text: "n/a" });
  const better = higherIsBetter ? c[0] > 0 : c[1] < 0, worse = higherIsBetter ? c[1] < 0 : c[0] > 0;
  if (better) return h("span", { class: "chip good", text: "▲ better" });
  if (worse) return h("span", { class: "chip crit", text: "▼ worse" });
  return h("span", { class: "chip", text: "● n.s.", title: "no clear difference: the 95% interval includes 0" });
}
function niceMax(v) {
  for (const s of [0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5]) { const top = Math.ceil(v / s) * s; if (top / s <= 6) return [top, s]; }
  return [1, 0.2];
}
const range = (a, b, s) => { const out = []; for (let v = a; v <= b + 1e-9; v += s) out.push(+v.toFixed(6)); return out; };
function table(rows, head) {
  const t = h("table");
  t.append(h("thead", {}, h("tr", {}, head.map(c => h("th", { text: c })))));
  t.append(h("tbody", {}, rows.map(r => h("tr", {}, r.map(c => h("td", {}, c))))));
  return t;
}

// ---------------- line chart: crosshair tooltip, keyboard arrows ----------------
function lineChart(host, o) {
  host.textContent = "";
  const W = Math.max(260, host.clientWidth), H = o.height || 280;
  const endLab = !!o.endLabels && W >= 520;
  const m = { l: 44, r: endLab ? 124 : 14, t: 10, b: o.xLabel ? 46 : 28 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const sx = v => m.l + (v - o.x.domain[0]) / (o.x.domain[1] - o.x.domain[0]) * iw;
  const sy = v => m.t + ih - (v - o.y.domain[0]) / (o.y.domain[1] - o.y.domain[0]) * ih;
  const svg = S("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.aria, tabindex: 0 }, host);
  for (const t of o.y.ticks) {
    S("line", { x1: m.l, x2: m.l + iw, y1: sy(t), y2: sy(t), class: "grid" }, svg);
    S("text", { x: m.l - 8, y: sy(t), class: "tick", "text-anchor": "end", "dominant-baseline": "middle" }, svg).textContent = o.y.fmt(t);
  }
  for (const t of o.x.ticks) S("text", { x: sx(t), y: m.t + ih + 18, class: "tick", "text-anchor": "middle" }, svg).textContent = o.x.fmt(t);
  S("line", { x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih, class: "axis" }, svg);
  if (o.xLabel) S("text", { x: m.l + iw / 2, y: H - 8, class: "axis-label", "text-anchor": "middle" }, svg).textContent = o.xLabel;
  for (const r of o.refs || []) S("line", { x1: sx(r[0][0]), y1: sy(r[0][1]), x2: sx(r[1][0]), y2: sy(r[1][1]), class: "ref" }, svg);
  for (const s of o.series) {
    const d = s.pts.map((p, i) => (i ? "L" : "M") + sx(p[0]).toFixed(1) + "," + sy(p[1]).toFixed(1)).join("");
    S("path", { d, class: "line" }, svg).style.stroke = `var(${s.color})`;
    if (s.markers) for (const p of s.pts) S("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 4, class: "mark" }, svg).style.fill = `var(${s.color})`;
  }
  if (endLab) for (const s of o.series.filter(s => s.endLabel)) {
    const p = s.pts[s.pts.length - 1];
    S("text", { x: sx(p[0]) + 10, y: sy(p[1]), class: "end-label", "dominant-baseline": "middle" }, svg).textContent = `${s.short} ${(o.valueFmt || o.y.fmt)(p[1])}`;
  }
  const xs = o.snapXs, cross = S("line", { y1: m.t, y2: m.t + ih, class: "cross", visibility: "hidden" }, svg);
  const marks = o.series.map(s => { const c = S("circle", { r: 4.5, class: "mark", visibility: "hidden" }, svg); c.style.fill = `var(${s.color})`; return c; });
  const tip = h("div", { class: "tip" }); tip.hidden = true; host.append(tip);
  let idx = 0;
  const nearest = (s, x) => s.pts.reduce((b, p) => Math.abs(p[0] - x) < Math.abs(b[0] - x) ? p : b, s.pts[0]);
  function show(i) {
    idx = i; const X = sx(xs[i]);
    cross.setAttribute("x1", X); cross.setAttribute("x2", X); cross.setAttribute("visibility", "visible");
    o.series.forEach((s, j) => { const p = nearest(s, xs[i]); marks[j].setAttribute("cx", sx(p[0])); marks[j].setAttribute("cy", sy(p[1])); marks[j].setAttribute("visibility", "visible"); });
    const t = o.tip(i); tip.textContent = "";
    tip.append(h("div", { class: "t", text: t.title }));
    for (const r of t.rows) tip.append(h("div", { class: "r" }, [key(r.color, r.kind || ""), h("strong", { text: r.value }), h("span", { text: r.label })]));
    tip.hidden = false;
    const tw = tip.offsetWidth; tip.style.left = (X + 14 + tw > W ? X - 14 - tw : X + 14) + "px";
  }
  function hide() { cross.setAttribute("visibility", "hidden"); marks.forEach(c => c.setAttribute("visibility", "hidden")); tip.hidden = true; }
  svg.addEventListener("pointermove", e => {
    const x = e.clientX - svg.getBoundingClientRect().left;
    let best = 0; xs.forEach((v, i) => { if (Math.abs(sx(v) - x) < Math.abs(sx(xs[best]) - x)) best = i; }); show(best);
  });
  svg.addEventListener("pointerleave", hide);
  svg.addEventListener("focus", () => show(idx));
  svg.addEventListener("blur", hide);
  svg.addEventListener("keydown", e => {
    if (e.key === "ArrowRight") { show(Math.min(xs.length - 1, idx + 1)); e.preventDefault(); }
    if (e.key === "ArrowLeft") { show(Math.max(0, idx - 1)); e.preventDefault(); }
  });
}

// ================= demo tab =================
const versionOf = (img, corr, sev) => sev === 0 ? img.versions.clean["0"] : img.versions[corr][String(sev)];
const current = () => versionOf(state.img, state.corr, state.sev);
const thresholdsFor = fold => (state.cache.meta.thresholds || {})[fold] || {};

function setupDemo() {
  const c = state.cache, sel = $("corr");
  for (const fam of FAMILIES) {
    const types = Object.entries(c.corruptions).filter(([, f]) => f === fam).map(([t]) => t);
    if (types.length) sel.append(h("optgroup", { label: famName(fam) }, types.map(t => h("option", { value: t, text: corrName(t) }))));
  }
  sel.addEventListener("change", () => select(null, sel.value, state.sev === 0 ? 3 : state.sev));
  $("sev").addEventListener("input", e => select(null, null, +e.target.value));
  $("api").addEventListener("toggle", renderApi);

  const groups = $("gallery-groups");
  for (const [title, note, list] of [
    ["Story photos", "picked because they show all three decisions", c.images.filter(im => im.curated !== false)],
    ["Random test photos", "not picked: drawn at random from the test split, mistakes included", c.images.filter(im => im.curated === false)]]) {
    if (!list.length) continue;
    const g = h("div", { class: "gallery" });
    for (const im of list) {
      const b = h("button", { type: "button", "data-id": im.image_id, "aria-pressed": "false", title: `${im.true_class} (${im.image_id})` },
        [h("img", { src: im.versions.clean["0"].image, alt: "" }), im.true_class]);
      b.addEventListener("click", () => select(im.image_id, im.story ? im.story.corruption : null, null));
      g.append(b);
    }
    groups.append(h("h3", {}, [title, " ", h("span", { class: "sub", text: `(${note})` })]), g);
  }

  const st = c.story, host = $("beats");
  if (st) {
    const img = c.images.find(i => i.image_id === st.image_id);
    $("story-sub").textContent = `One ${img.true_class} photo from the test split, which the trust layer never trained on, with ${corrName(st.corruption).toLowerCase()} added step by step. The trust layer judging it never saw any ${famName(c.corruptions[st.corruption]).toLowerCase()} corruption either. The photo was picked to show all three decisions; Research results covers every test photo.`;
    st.beats.forEach((beat, i) => {
      const v = versionOf(img, st.corruption, beat.severity);
      const b = h("button", { class: "beat", type: "button", "aria-pressed": "false", "data-beat": String(i) }, [
        h("img", { src: v.image, alt: "" }), h("b", { text: `${i + 1} · ${beat.label}` }),
        h("span", {}, [chip(v.trust_layer.decision), " ", h("span", { class: "sub", text: beat.severity ? `severity ${beat.severity}` : "clean" })])]);
      b.addEventListener("click", () => select(st.image_id, st.corruption, beat.severity));
      host.append(b);
    });
    select(st.image_id, st.corruption, 0);
  } else {
    host.append(h("p", { class: "sub", text: "No story image in this cache: pick any photo below." }));
    select(c.images[0].image_id, Object.keys(c.corruptions)[0], 0);
  }
}

function select(imageId, corr, sev) {
  if (imageId) state.img = state.cache.images.find(i => i.image_id === imageId) || state.img;
  if (corr) state.corr = corr;
  if (sev != null) state.sev = sev;
  renderDemo();
}

function renderDemo() {
  const img = state.img, v = current(), c = state.cache;
  const where = state.sev === 0 ? "Clean test photo" : `${corrName(state.corr)}, severity ${state.sev} of 5`;
  $("big").src = v.image;
  $("big").alt = `${where}; the true class is ${img.true_class}`;
  $("caption").textContent = `${where} · true class: ${img.true_class}`;
  $("corr").value = state.corr;
  $("sev").value = String(state.sev);
  $("sev-out").textContent = state.sev === 0 ? "clean" : `${state.sev} of 5`;

  const strip = $("strip"); strip.textContent = "";
  for (let s = 0; s <= 5; s++) {
    const sv = versionOf(img, state.corr, s), d = DECISION[sv.trust_layer.decision];
    const dot = h("span", { class: "dot" }); dot.style.background = `var(${d.color})`;
    const b = h("button", { type: "button", "aria-pressed": String(s === state.sev), title: `${s ? "severity " + s : "clean"}: ${d.label}, p_correct ${pctUI(sv.trust_layer.p_correct)}` },
      [h("img", { src: sv.image, alt: `${s ? "severity " + s : "clean"}, ${d.label}` }), h("span", {}, [dot, ` ${s ? s : "clean"}`])]);
    b.addEventListener("click", () => select(null, null, s));
    strip.append(b);
  }

  $("pred").textContent = v.target_model.prediction;
  const t = $("truth"); t.textContent = "";
  t.append(`truth: ${v.truth.true_class} · `, h("span", { class: v.truth.correct ? "right" : "wrong", text: v.truth.correct ? "✓ right" : "✕ wrong" }));
  renderLadder(v);

  const d = DECISION[v.trust_layer.decision], th = thresholdsFor(v.context.fold), dec = $("decision");
  dec.className = `decision d-${v.trust_layer.decision}`; dec.textContent = "";
  const unseen = v.context.corruption === "clean" ? "This photo was never used to train the trust layer."
    : `Judged by a trust layer that never saw any ${famName(v.context.family).toLowerCase()} corruption.`;
  dec.append(h("span", { class: "badge", text: d.label }), h("span", { class: "what", text: d.what }),
    h("span", { class: "why", text: `p_correct ${pctUI(v.trust_layer.p_correct)}. REJECT below ${pct(th.tau_reject)}, TRUST at ${th.tau_trust == null ? "(unreachable)" : pct(th.tau_trust)} or above, both chosen on calibration images for 5% and 1% error. ${unseen}` }));

  const rs = $("reasons"); rs.textContent = "";
  const reasons = v.trust_layer.reasons || [];
  if (!reasons.length) rs.append(h("li", { class: "quiet", text: v.trust_layer.decision === "trust"
    ? "No warning signs: the prediction is stable under small flips and shifts, and the image looks familiar."
    : "No single signal stands out; together they put p_correct below the TRUST cutoff." }));
  for (const r of reasons) rs.append(h("li", {}, [r.text + " ", h("small", { text: `(${r.signal}, +${r.shap.toFixed(2)} risk)` })]));

  for (const b of $("gallery-groups").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.id === img.image_id));
  const st = c.story;
  for (const b of $("beats").querySelectorAll(".beat")) {
    const beat = st.beats[+b.dataset.beat];
    b.setAttribute("aria-pressed", String(img.image_id === st.image_id && beat.severity === state.sev && (state.sev === 0 || state.corr === st.corruption)));
  }
  renderApi();
}

function renderLadder(v) {
  const th = thresholdsFor(v.context.fold), host = $("ladder"); host.textContent = "";
  const rows = [["Raw confidence", v.target_model.raw_confidence, "--raw"], ["Temp-scaled, clean val", v.baselines.temp_scaled_clean, "--tclean"],
    ["Temp-scaled, corrupted val", v.baselines.temp_scaled_corrupted, "--tcorr"], ["Trust layer p_correct", v.trust_layer.p_correct, "--trust", true]];
  for (const [label, val, color, final] of rows) {
    const track = h("div", { class: "track" });
    if (final) {
      const tr = th.tau_reject ?? 0, tt = th.tau_trust ?? 1;
      for (const [lo, hi, c] of [[0, tr, "--crit"], [tr, tt, "--warn"], [tt, 1, "--good"]]) {
        if (hi <= lo) continue;
        const z = h("div", { class: "zone" }); z.style.left = `${lo * 100}%`; z.style.width = `${(hi - lo) * 100}%`; z.style.background = `var(${c})`;
        track.append(z);
      }
    }
    const fill = h("div", { class: "fill" }); fill.style.width = `${val * 100}%`; fill.style.background = `var(${color})`;
    track.append(fill);
    if (final) for (const tau of [th.tau_reject, th.tau_trust]) if (tau != null) {
      const tk = h("div", { class: "tau" }); tk.style.left = `calc(${tau * 100}% - 1px)`; track.append(tk);
    }
    host.append(h("div", { class: "rung" + (final ? " final" : "") }, [h("span", { text: label }), track, h("span", { class: "v", text: pctUI(val) })]));
    if (final) {
      const lab = h("div", { class: "zone-labels" }), tr = th.tau_reject ?? 0, tt = th.tau_trust ?? 1;
      for (const [text, lo, hi] of [["REJECT", 0, tr], ["CAUTION", tr, tt], ["TRUST", tt, 1]]) if (hi - lo > 0.06) {
        const s = h("span", { text }); s.style.left = `${(lo + hi) / 2 * 100}%`; lab.append(s);
      }
      host.append(h("div", { class: "rung" }, [h("span"), lab, h("span")]));
    }
  }
}

async function renderApi() {
  if (!$("api").open || !state.img) return;
  const token = ++state.apiToken;
  const req = { sample_id: state.img.image_id, corruption: state.sev === 0 ? "clean" : state.corr, severity: state.sev };
  $("api-req").textContent = "POST /predict\n" + JSON.stringify(req, null, 2);
  let text;
  try {
    const r = await fetch("predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(req) });
    if (!r.ok) throw new Error(r.status);
    text = JSON.stringify(await r.json(), null, 2);
  } catch {
    const { image, ...rest } = current();
    text = "// served from data/demo_cache.json; run python -m backend.main for the live endpoint\n" + JSON.stringify(rest, null, 2);
  }
  if (token === state.apiToken) $("api-res").textContent = text;
}

// ================= research tab =================
const fold = () => state.ev.folds[state.ev.headline_fold];

function renderKpis() {
  const f = fold(), mt = f.heldout_test.methods, b = f.bootstrap_heldout || {}, bp = f.broken_promise_heldout, br = b.trust_minus_raw_confidence || {};
  const host = $("kpis"); host.textContent = "";
  const tile = (takeaway, value, small, explain, tech, chip) => host.append(h("div", { class: "kpi" }, [h("div", { class: "label", text: takeaway }),
    h("div", { class: "value" }, [value, small ? h("small", { text: small }) : null]), h("div", { class: "delta" }, [explain, chip]),
    h("div", { class: "tech", text: tech })]));
  const rawClean = bp.raw_conf_rule_set_on_clean_cal, trustRule = bp.trust_layer_reject_rule, target = pct(trustRule.target_error, 0);
  tile(`Keeps its “at most ${target} wrong” promise on blur it never saw`, pct(trustRule.error), "wrong",
    `The model's own confidence, tuned the usual way, lets ${pct(rawClean.error)} wrong answers through.`, `selective risk at a ${target} target`,
    rawClean.error > rawClean.target_error ? h("span", { class: "chip crit", text: "✕ promise broken" }) : null);
  tile("Better at telling right answers from wrong ones", f3(mt.trust_layer.auroc), `model alone ${f3(mt.raw_confidence.auroc)}`,
    "1 is perfect; 0.5 is guessing.", "AUROC", sigChip(br.auroc_diff_ci, true));
  tile("Its stated confidence is much closer to reality", f3(mt.trust_layer.ece), `model alone ${f3(mt.raw_confidence.ece)}`,
    "The average gap between stated confidence and real accuracy; 0 is perfect.", "ECE");
  tile("Fewer mistakes among the answers it trusts most", pct(mt.trust_layer.err_at_20), `model alone ${pct(mt.raw_confidence.err_at_20)}`,
    "Share wrong among the 20% of answers it trusts most.", "selective risk at 20% coverage");
}

function renderHeadline() {
  const rows = fold().headline_chart || [], host = $("headline-chart");
  if (!rows.length) { host.textContent = "No headline_chart in evaluation.json."; return; }
  const keys = [["accuracy", "Actually right", "Right", "--acc", true], ["raw_confidence", M.raw_confidence.label, "Model", "--raw", false],
    ["temp_clean", M.temp_clean.label, "", "--tclean", false], ["temp_corrupted", M.temp_corrupted.label, "", "--tcorr", false],
    ["p_correct", M.trust_layer.label, "Trust layer", "--trust", false]];
  const series = keys.map(([k, label, short, color, markers]) => ({ key: k, label, short, color, markers,
    endLabel: ["accuracy", "raw_confidence", "p_correct"].includes(k), pts: rows.map(r => [r.severity, r[k]]) }));
  const lo = Math.min(...series.flatMap(s => s.pts.map(p => p[1]))), y0 = Math.max(0, Math.floor(lo * 10 - 0.5) / 10), last = rows[rows.length - 1];
  const fam = famName(state.ev.headline_fold);
  $("headline-desc").textContent = `At the strongest ${fam.toLowerCase()}, the model is right ${pct(last.accuracy)} of the time but claims ${pct(last.raw_confidence)}, while the trust layer estimates ${pct(last.p_correct)}.`;
  legend($("headline-legend"), series.map(s => ({ label: s.label, color: s.color, kind: s.markers ? "dot" : "" })));
  lineChart(host, {
    height: 300, aria: `How often the model was right, and four confidence estimates, by ${fam.toLowerCase()} level`, endLabels: true, valueFmt: v => pct(v),
    x: { domain: [-0.25, last.severity + 0.25], ticks: rows.map(r => r.severity), fmt: v => v === 0 ? "clean" : `${v}` },
    y: { domain: [y0, 1], ticks: range(y0, 1, 0.1), fmt: v => Math.round(v * 100) + "%" },
    xLabel: `${fam} level (never seen in training)`, series, snapXs: rows.map(r => r.severity),
    tip: i => ({ title: `${rows[i].severity === 0 ? "Clean photos" : `${fam} level ${rows[i].severity}`} · ${rows[i].n.toLocaleString()} photos`,
      rows: series.map(s => ({ color: s.color, kind: s.markers ? "dot" : "", value: pct(s.pts[i][1]), label: s.label })) }),
  });
  $("headline-table").textContent = "";
  $("headline-table").append(table(rows.map(r => [r.severity === 0 ? "clean" : `${r.severity}`, r.n.toLocaleString(), ...keys.map(k => pct(r[k[0]]))]),
    [`${fam} level`, "Photos", ...keys.map(k => k[1])]));
}

function renderDrift() {
  const fc = state.cache.family_curves || {}, host = $("drift-chart"), tbl = $("drift-table");
  const defs = [["accuracy", "Actually right", "--acc", true], ["p_correct", M.trust_layer.label, "--trust", false], ["reject", "Share rejected", "--crit", false]];
  legend($("drift-legend"), defs.map(([, label, color, mk]) => ({ label, color, kind: mk ? "dot" : "" })));
  host.textContent = ""; tbl.textContent = "";
  $("drift-desc").textContent = "Each panel is a kind of damage the trust layer never saw in training: as the damage grows, accuracy falls and the share it rejects rises.";
  const draws = [];   // panels first, charts after: a lone panel in an auto-fit grid measures full width
  for (const fam of FAMILIES.filter(f => fc[f])) {
    const pts = fc[fam], first = pts[0], last = pts[pts.length - 1];
    const chart = h("div", { class: "chart" });
    host.append(h("div", { class: "panel" }, [h("h3", { text: famName(fam) }),
      h("div", { class: "sub", text: `Clean → level ${last.severity}: rejected ${pct(first.reject, 0)} → ${pct(last.reject, 0)}, right ${pct(first.accuracy, 0)} → ${pct(last.accuracy, 0)}` }), chart]));
    const series = defs.map(([k, label, color, markers]) => ({ key: k, label, color, markers, pts: pts.map(p => [p.severity, p[k]]) }));
    draws.push(() => lineChart(chart, {
      height: 220, aria: `${famName(fam)}: how often the model was right, the trust layer's estimate and the share rejected, by level`,
      x: { domain: [-0.25, 5.25], ticks: [0, 1, 2, 3, 4, 5], fmt: v => v === 0 ? "clean" : `${v}` },
      y: { domain: [0, 1], ticks: [0, .25, .5, .75, 1], fmt: v => Math.round(v * 100) + "%" }, series, snapXs: pts.map(p => p.severity),
      tip: i => ({ title: `${famName(fam)} · ${pts[i].severity === 0 ? "clean" : "level " + pts[i].severity} · ${pts[i].n.toLocaleString()} photos`,
        rows: series.map(s => ({ color: s.color, kind: s.markers ? "dot" : "", value: pct(s.pts[i][1]), label: s.label })) }),
    }));
    tbl.append(h("h3", { text: famName(fam), style: "margin:10px 0 4px" }), table(pts.map(p => [p.severity === 0 ? "clean" : `${p.severity}`, p.n.toLocaleString(),
      pct(p.accuracy), pct(p.raw_confidence), pct(p.p_correct), pct(p.trust), pct(p.caution), pct(p.reject)]),
      ["Level", "Photos", "Actually right", "Model's own confidence", M.trust_layer.label, "TRUST", "CAUTION", "REJECT"]));
  }
  draws.forEach(d => d());
}

function renderPromise() {
  const bp = fold().broken_promise_heldout, fam = famName(state.ev.headline_fold).toLowerCase();
  const rules = [["raw_conf_rule_set_on_clean_cal", "Model's own confidence", "tuned the usual way, on clean photos", "--raw"],
    ["raw_conf_rule_set_on_seen_cal", "Model's own confidence", "tuned on damaged photos", "--raw"],
    ["trust_layer_reject_rule", "Trust layer", "tuned on damaged photos", "--trust"]].filter(r => bp[r[0]]);
  const [top] = niceMax(Math.max(...rules.map(r => Math.max(bp[r[0]].error || 0, bp[r[0]].target_error))) / 0.58);
  const host = $("promise"); host.textContent = "";
  for (const [k, name, sub, color] of rules) {
    const r = bp[k], track = h("div", { class: "ptrack" }), w = (r.error || 0) / top * 100;
    const fill = h("div", { class: "fill" }); fill.style.width = `${w}%`; fill.style.background = `var(${color})`;
    const tgt = h("div", { class: "target", title: `promised ${pct(r.target_error, 0)}` }); tgt.style.left = `calc(${r.target_error / top * 100}% - 1px)`;
    const val = h("div", { class: "val", text: r.error == null ? "approves nothing" : `${pct(r.error)} wrong · ${pct(r.coverage, 0)} approved` });
    val.style.left = `calc(${Math.max(w, r.target_error / top * 100)}% + 8px)`;
    track.append(fill, tgt, val);
    host.append(h("div", { class: "promise-row" }, [h("div", { class: "name" }, [name, h("small", { text: sub })]), track]));
  }
  const rs = bp.raw_conf_rule_set_on_seen_cal, tr = bp.trust_layer_reject_rule, t = pct(tr.target_error, 0);
  const kept = r => r.error != null && r.error <= r.target_error;
  $("promise-scale").textContent = `Bar = share of auto-approved answers that were wrong on ${fam} it never saw (scale 0–${pct(top, 0)}).`;
  $("promise-desc").textContent = kept(rs) && kept(tr)
    ? `Both rules tuned on damaged photos stay under the ${t} line, but the trust layer auto-approves more photos (${pct(tr.coverage, 0)} vs ${pct(rs.coverage, 0)}).`
    : `Each bar must end left of the black line to keep the “at most ${t} wrong” promise.`;
}

function renderReliability() {
  const f = fold(), rel = f.reliability_heldout || {}, mt = f.heldout_test.methods;
  const host = $("rel-chart"), tbl = $("rel-table"), draws = []; host.textContent = ""; tbl.textContent = "";
  for (const k of ["raw_confidence", "temp_clean", "temp_corrupted", "trust_layer"].filter(k => rel[k])) {
    const bins = rel[k], chart = h("div", { class: "chart" });
    host.append(h("div", { class: "panel" }, [h("h3", { text: M[k].label }), h("div", { class: "sub", text: `Average gap ${f3(mt[k].ece)} (ECE)` }), chart]));
    draws.push(() => lineChart(chart, {
      height: 230, aria: `${M[k].label}: stated confidence vs. how often it was right`, x: { domain: [0, 1], ticks: [0, .5, 1], fmt: v => Math.round(v * 100) + "%" },
      y: { domain: [0, 1], ticks: [0, .25, .5, .75, 1], fmt: v => Math.round(v * 100) + "%" }, xLabel: "stated confidence",
      refs: [[[0, 0], [1, 1]]], series: [{ key: k, color: M[k].color, markers: true, pts: bins.map(b => [b.conf, b.acc]) }], snapXs: bins.map(b => b.conf),
      tip: i => ({ title: `Stated ${pct(bins[i].lo, 0)}–${pct(bins[i].hi, 0)} · ${bins[i].n.toLocaleString()} photos`,
        rows: [{ color: M[k].color, kind: "dot", value: pct(bins[i].acc), label: "actually right" }, { color: "--axis", value: pct(bins[i].conf), label: "stated" }] }),
    }));
    tbl.append(h("div", { class: "panel" }, [h("h3", { text: M[k].label }), h("div", { class: "scroll" },
      table(bins.map(b => [`${pct(b.lo, 0)}–${pct(b.hi, 0)}`, b.n.toLocaleString(), pct(b.conf), pct(b.acc)]), ["Stated confidence", "Photos", "Average stated", "Actually right"]))]));
  }
  draws.forEach(d => d());
}

function renderResearch() {
  if ($("research").hidden) return;
  renderKpis(); renderHeadline(); renderDrift(); renderPromise(); renderReliability();
}

// ================= page wiring =================
function showTab(name) {
  for (const t of ["demo", "research"]) {
    $(t).hidden = t !== name;
    $(`tab-${t}`).setAttribute("aria-selected", String(t === name));
  }
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
  renderResearch();
}

function renderProvenance() {
  const m = state.ev.meta || {}, b = m.base_images || {}, cm = state.cache.meta || {};
  $("provenance").textContent = [`Results from ${m.n_rows ? m.n_rows.toLocaleString() + " predictions" : "evaluation.json"}`,
    b.test != null && `${(b.train + b.val + b.cal + b.test).toLocaleString()} CIFAR-10 test photos × 41 versions (${b.test} of them held out for testing)`,
    m.cifar10_1_images && `${m.cifar10_1_images.toLocaleString()} CIFAR-10.1 photos`,
    m.created && `trust layer trained ${m.created.slice(0, 16).replace("T", " ")} UTC`,
    cm.created && `demo cache built ${cm.created.slice(0, 16).replace("T", " ")} UTC`].filter(Boolean).join(" · ");
}

document.querySelectorAll("[data-toggle]").forEach(b => b.addEventListener("click", () => {
  const id = b.dataset.toggle, chart = $(`${id}-chart`), tbl = $(`${id}-table`), leg = $(`${id}-legend`);
  const showTable = tbl.hidden;
  tbl.hidden = !showTable; chart.hidden = showTable; if (leg) leg.hidden = showTable;
  b.textContent = showTable ? "Show chart" : "Show table";
  if (!showTable) renderResearch();
}));
$("tab-demo").addEventListener("click", () => showTab("demo"));
$("tab-research").addEventListener("click", () => showTab("research"));
window.addEventListener("hashchange", () => state.cache && showTab(location.hash === "#research" ? "research" : "demo"));

async function boot() {
  try {
    const get = async p => { const r = await fetch(p); if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`); return r.json(); };
    [state.cache, state.ev] = await Promise.all([get("data/demo_cache.json"), get("data/evaluation.json")]);
  } catch (e) {
    const el = $("load-error");
    el.textContent = `Could not load the demo data (${e.message}). The demo needs data/demo_cache.json and data/evaluation.json, which come with the repository. If they are missing, create them with REQUIREMENTS.md, section 3 "Rebuild everything from scratch", then start the demo again with python -m backend.main.`;
    el.hidden = false; $("demo").hidden = true;
    return;
  }
  setupDemo();
  renderProvenance();
  showTab(location.hash === "#research" ? "research" : "demo");
  let raf = 0, lastW = 0;
  new ResizeObserver(es => {
    const w = Math.round(es[0].contentRect.width); if (w === lastW) return; lastW = w;
    cancelAnimationFrame(raf); raf = requestAnimationFrame(renderResearch);
  }).observe(document.querySelector("main"));
}
boot();
