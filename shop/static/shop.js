// Snap-to-Shop tab (Visa wrapper). Installs itself only if /api/shop/samples answers; otherwise does nothing,
// so the core demo is unchanged when shop/ is absent. Uses the page's CSS tokens. No payment data anywhere.
(() => {
  const api = async (path, body) => {
    const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
    const j = await r.json().catch(() => ({}));
    return { ok: r.ok, status: r.status, j };
  };
  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "text") n.textContent = v; else if (k === "class") n.className = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v); else if (v != null && v !== false) n.setAttribute(k, v);
    }
    for (const c of kids.flat()) if (c != null) n.append(c);
    return n;
  };
  const money = x => `$${Number(x).toFixed(2)}`;
  const pct = x => `${Math.round(100 * x)}%`;
  const nice = s => s === "clean" ? "clean" : s.replace(/_/g, " ");
  const TIER = { one_tap: "One-tap", confirm: "Confirm once", confirm_twice: "Confirm twice", blocked_until_confirmed: "Locked", blocked: "Locked" };
  const DEC = { trust: "TRUST", trust_confirmed: "TRUST (you confirmed)", caution: "CAUTION", reject: "REJECT" };

  const CSS = `
  #shop .shop-grid { display: grid; gap: 18px; grid-template-columns: minmax(0, 330px) minmax(0, 1fr); align-items: start; }
  @media (max-width: 820px) { #shop .shop-grid { grid-template-columns: minmax(0, 1fr); } }
  #shop .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  #shop select, #shop input[type=text] { font: inherit; color: var(--ink); background: var(--page); border: 1px solid var(--border); border-radius: 7px; padding: 7px 9px; }
  #shop input[type=text] { flex: 1 1 200px; min-width: 0; }
  #shop .thumbs { display: grid; grid-template-columns: repeat(auto-fill, minmax(52px, 1fr)); gap: 6px; }
  #shop .thumbs button { padding: 0; border: 2px solid transparent; border-radius: 6px; background: none; cursor: pointer; aspect-ratio: 1; overflow: hidden; }
  #shop .thumbs button[aria-pressed=true] { border-color: var(--trust); }
  #shop .thumbs img { width: 100%; height: 100%; object-fit: cover; display: block; }
  #shop .photo { width: 100%; aspect-ratio: 1; border-radius: 10px; object-fit: cover; background: var(--sunken); }
  #shop .seg { display: inline-flex; background: var(--sunken); border-radius: 8px; padding: 2px; gap: 2px; }
  #shop .seg button { border: 0; background: none; border-radius: 6px; padding: 5px 10px; cursor: pointer; font-size: 13px; }
  #shop .seg button[aria-pressed=true] { background: var(--surface); box-shadow: 0 1px 2px var(--shadow); font-weight: 600; }
  #shop .badge { display: inline-block; font-weight: 700; font-size: 13px; padding: 4px 10px; border-radius: 999px; }
  #shop .b-trust, #shop .b-trust_confirmed { background: color-mix(in srgb, var(--good) 18%, transparent); color: var(--good-ink); }
  #shop .b-caution { background: color-mix(in srgb, var(--warn) 22%, transparent); color: var(--warn-ink); }
  #shop .b-reject { background: color-mix(in srgb, var(--crit) 18%, transparent); color: var(--crit-ink); }
  #shop .why-line { font-size: 14.5px; line-height: 1.45; margin: 0; }
  #shop .coach { border-color: color-mix(in srgb, var(--warn) 45%, transparent); }
  #shop .kv { display: grid; grid-template-columns: auto 1fr; gap: 3px 12px; font-size: 13.5px; }
  #shop .kv span:nth-child(odd) { color: var(--ink-2); }
  #shop .chat { display: grid; gap: 8px; max-height: 260px; overflow-y: auto; padding: 2px; }
  #shop .msg { padding: 9px 12px; border-radius: 10px; font-size: 14px; max-width: 92%; }
  #shop .msg.a { background: var(--sunken); justify-self: start; }
  #shop .msg.g { justify-self: stretch; max-width: none; font-size: 13px; font-weight: 600; color: var(--crit-ink);
    background: color-mix(in srgb, var(--crit) 10%, transparent); border: 1px solid color-mix(in srgb, var(--crit) 35%, transparent); }
  #shop .msg.g::before { content: "🔒 "; }
  #shop .msg.u { background: color-mix(in srgb, var(--trust) 16%, transparent); justify-self: end; }
  #shop .products { display: grid; gap: 10px; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); }
  #shop .prod { border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; display: grid; gap: 5px; background: var(--page); }
  #shop .prod .price { font-weight: 700; font-variant-numeric: tabular-nums; }
  #shop .prod .why { font-size: 12.5px; color: var(--ink-2); }
  #shop .cmp { border: 1px dashed var(--warn); border-radius: 10px; padding: 10px 12px; display: grid; gap: 8px; }
  #shop .tier { font-weight: 700; font-size: 15px; }
  #shop .btn.primary { background: var(--trust); color: #fff; border-color: transparent; }
  #shop .btn[disabled] { opacity: .5; cursor: not-allowed; }
  #shop .small { font-size: 12.5px; color: var(--muted); }
  #shop .offline { font-size: 12px; padding: 2px 8px; border-radius: 999px; background: var(--sunken); color: var(--ink-2); }
  `;

  const S = { samples: null, meta: null, me: null, mode: "me", editing: false, base: null, photo: null, confirmed: null,
              cart: {}, confirmations: 0, assist: null, chat: [], busy: false, uploads: [], uploadStatus: "loading",
              uploading: false, uploadError: null };
  const RISK_TEXT = { relaxed: "Relaxed: one-tap up to $300", normal: "Normal: one-tap up to $150",
                      strict: "Strict: confirm every purchase" };
  const store = { get: () => { try { return JSON.parse(localStorage.getItem("shop.profile")); } catch (e) { return null; } },
                  set: v => { try { localStorage.setItem("shop.profile", JSON.stringify(v)); } catch (e) { /* private mode */ } } };
  // The shopper's own profile is the default; the fictional examples are a proof of concept.
  const who = () => S.mode === "me" ? { profile: S.me } : { profile_id: S.mode };
  const whoName = () => S.mode === "me" ? (S.me.name || "You") : S.meta.examples.find(p => p.profile_id === S.mode).name;
  let root;

  function versionsOf(base) { return [...S.uploads, ...S.samples.photos].find(p => p.base_image_id === base).versions; }

  // ---- live upload: downscale in the browser (max 800 px), then the server runs the model + trust layer ----
  async function toDataURL(file) {
    try {
      const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
      const k = Math.min(1, 800 / Math.max(bmp.width, bmp.height));
      const c = document.createElement("canvas");
      c.width = Math.round(bmp.width * k); c.height = Math.round(bmp.height * k);
      c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
      return c.toDataURL("image/jpeg", 0.9);
    } catch (e) {  // browsers that can't decode it (e.g. HEIC outside Safari): let the server try
      return await new Promise((ok, bad) => { const r = new FileReader(); r.onload = () => ok(r.result); r.onerror = bad; r.readAsDataURL(file); });
    }
  }
  async function upload(file) {
    if (!file) return;
    S.uploading = true; S.uploadError = null; render();
    const r = await api("/api/shop/upload", { image: await toDataURL(file) }).catch(e => ({ ok: false, j: { error: String(e) } }));
    S.uploading = false;
    if (!r.ok) { S.uploadError = r.j.error || `Upload failed (HTTP ${r.status})`; return render(); }
    const it = r.j;
    S.uploads = [{ base_image_id: it.base_image_id, true_class: null, versions: [it] }, ...S.uploads.filter(u => u.base_image_id !== it.base_image_id)].slice(0, 6);
    S.chat.push({ who: "u", text: "(uploaded a photo)" });
    select(it.base_image_id, it.photo_id, { keepChat: true });
  }
  function uploadButton(label = "Upload or snap a photo") {
    const input = el("input", { type: "file", accept: "image/*", hidden: "", onchange: e => upload(e.target.files[0]) });
    const ready = S.uploadStatus === "ready";
    return el("span", {}, input, el("button", { class: "btn primary", type: "button", disabled: !ready || S.uploading ? "" : null,
      title: S.uploadStatus === "error" ? S.uploadErrorDetail : null,
      text: S.uploading ? "Analyzing…" : ready ? label : S.uploadStatus === "loading" ? "Model loading…" : "Upload unavailable",
      onclick: () => input.click() }));
  }
  function item() { return versionsOf(S.base).find(v => v.photo_id === S.photo); }

  async function runAssist(userMessage) {
    S.busy = true; render();
    const r = await api("/api/shop/assist", { photo_id: S.photo, ...who(), user_message: userMessage || "", confirmed_class: S.confirmed });
    S.busy = false;
    if (!r.ok) { S.chat.push({ who: "a", text: `Error: ${r.j.error || r.status}` }); return render(); }
    S.assist = r.j;
    if (r.j.gate_note) S.chat.push({ who: "g", text: r.j.gate_note });
    S.chat.push({ who: "a", text: r.j.assistant_message });
    if (!r.j.cart_allowed) S.cart = {};
    render(); quote();
  }

  async function select(base, photo, opts = {}) {
    S.base = base; S.photo = photo; S.confirmed = null; S.cart = {}; S.confirmations = 0; S.order = null; S.quote = null;
    S.chat = opts.keepChat ? S.chat : [];
    await runAssist(opts.message || "");
    if (opts.qty && S.assist && S.assist.cart_allowed && S.assist.products.length) {
      S.cart = { [S.assist.products[0].product_id]: opts.qty }; render(); quote();
    }
  }

  async function quote() {
    const cart = Object.entries(S.cart).filter(([, q]) => q > 0).map(([product_id, qty]) => ({ product_id, qty }));
    if (!cart.length) { S.quote = null; return render(); }
    const r = await api("/api/shop/quote", { photo_id: S.photo, ...who(), confirmed_class: S.confirmed, cart });
    S.quote = r.ok ? r.j : { tier: "blocked", explanation: r.j.error || "Unavailable", allowed: false, total: 0 };
    render();
  }

  async function checkout() {
    const cart = Object.entries(S.cart).filter(([, q]) => q > 0).map(([product_id, qty]) => ({ product_id, qty }));
    const r = await api("/api/shop/checkout", { photo_id: S.photo, ...who(), confirmed_class: S.confirmed, cart, confirmations: S.confirmations });
    S.order = r.ok ? r.j : { error: r.j.explanation || r.j.error, need: r.j.confirmations_required };
    render();
  }

  function onBuy() {
    const need = S.quote ? S.quote.confirmations_required || 0 : 0;
    if (S.confirmations < need) { S.confirmations += 1; render(); if (S.confirmations < need) return; return; }
    checkout();
  }

  function setQty(pid, d) {
    S.cart[pid] = Math.max(0, Math.min(20, (S.cart[pid] || 0) + d)); S.confirmations = 0; S.order = null; quote();
  }

  function renderPicker() {
    const photos = [...S.uploads, ...S.samples.photos];
    const vers = versionsOf(S.base);
    const cur = item();
    const isUpload = cur.corruption === "upload";
    const corrs = [...new Set(vers.map(v => v.corruption))];
    const sevs = vers.filter(v => v.corruption === (cur.corruption === "clean" ? corrs.find(c => c !== "clean") : cur.corruption) || v.corruption === "clean");
    const pick = (c, s) => { const v = vers.find(x => x.corruption === c && x.severity === s) || vers.find(x => x.corruption === "clean"); select(S.base, v.photo_id); };
    const corrSel = el("select", { "aria-label": "Corruption", onchange: e => pick(e.target.value, e.target.value === "clean" ? 0 : Math.max(cur.severity, 1)) },
      corrs.map(c => el("option", { value: c, selected: c === cur.corruption ? "" : null, text: nice(c) })));
    const sevSeg = el("div", { class: "seg", role: "group", "aria-label": "Severity" },
      [...new Set(sevs.map(v => v.severity))].sort().map(s => el("button", { type: "button", "aria-pressed": String(s === cur.severity),
        text: s === 0 ? "clean" : `sev ${s}`, onclick: () => pick(s === 0 ? "clean" : (cur.corruption === "clean" ? corrs.find(c => c !== "clean") : cur.corruption), s) })));
    return el("section", { class: "card" },
      el("h2", { text: "1 · Snap a photo" }),
      el("div", { class: "row" }, uploadButton(), el("span", { class: "small", text: "or pick a sample:" })),
      S.uploadError ? el("p", { class: "err", text: S.uploadError }) : null,
      el("div", { class: "thumbs" }, photos.map(p => { const v0 = p.versions.find(v => ["clean", "upload"].includes(v.corruption));
        return el("button", { type: "button", "aria-pressed": String(p.base_image_id === S.base), title: p.true_class || "your photo",
          onclick: () => select(p.base_image_id, v0.photo_id) },
          el("img", { src: v0.image, alt: p.true_class || "your photo" })); })),
      el("img", { class: "photo", src: cur.image, alt: isUpload ? "your photo" : `${cur.true_class}, ${nice(cur.corruption)}`, style: "image-rendering:auto" }),
      isUpload ? el("p", { class: "small", text: "Your photo, scored live by the model and the trust layer (30 product types; anything else should come back CAUTION or REJECT)." })
        : el("div", { class: "row" }, corrSel, sevSeg),
      isUpload ? null : el("p", { class: "small", text: "Corrupted versions simulate a bad phone photo: noise, blur, fog, low contrast, compression." }));
  }

  function renderTrust() {
    const a = S.assist, cur = item();
    if (!a) return el("section", { class: "card" }, el("p", { class: "small", text: "Loading…" }));
    const reasons = (cur.reasons || []).map(r => el("li", { text: r.text }));
    return el("section", { class: "card" },
      el("div", { class: "row" }, el("h2", { text: "2 · Trust check" }), el("span", { class: `badge b-${a.effective_decision}`, text: DEC[a.effective_decision] })),
      a.explanation ? el("p", { class: "why-line", text: a.explanation }) : null,
      el("div", { class: "kv" },
        el("span", { text: "Model says" }), el("span", { text: `${cur.pred_class} (raw confidence ${pct(cur.raw_confidence)})` }),
        el("span", { text: "Trust layer" }), el("span", { text: `p_correct ${pct(a.p_correct)}` }),
        el("span", { text: "Assistant may" }), el("span", { text: a.allowed_actions.map(nice).join(", ") })),
      reasons.length ? el("ul", { class: "small", style: "margin:0;padding-left:18px" }, reasons) : null);
  }

  function clearerVersion(cur) {
    // Same product, gentler corruption: the lowest-severity version that the trust layer doesn't REJECT.
    const vers = versionsOf(S.base).filter(v => v.corruption === cur.corruption && v.severity < cur.severity && v.decision !== "reject");
    if (cur.corruption === "upload") return null;
    const clean = versionsOf(S.base).find(v => v.corruption === "clean");
    return vers.sort((a, b) => b.severity - a.severity)[0] || (clean && clean.photo_id !== cur.photo_id ? clean : null);
  }

  function renderCoach() {
    const a = S.assist, cur = item();
    if (!a || !["reject", "caution"].includes(a.effective_decision)) return null;
    const q = cur.quality || {};
    if (a.effective_decision === "caution" && !q.issue) return null;
    const better = clearerVersion(cur);
    return el("section", { class: "card coach" },
      el("h2", { text: "Retake coach" }),
      el("p", { text: q.issue ? `This photo looks ${q.label}. ${q.tip}` : q.tip || "Try a clearer photo of just the item." }),
      el("p", { class: "small", text: "Detected from the pixels (sharpness, noise, brightness, contrast vs clean product photos)." }),
      cur.corruption === "upload" ? el("div", { class: "row" }, uploadButton("Upload a new photo")) : null,
      better ? el("div", { class: "row" }, el("button", { class: "btn", type: "button", text: "Try a clearer shot",
        onclick: () => { S.chat.push({ who: "u", text: "Here's a clearer photo." }); select(S.base, better.photo_id, { keepChat: true }); } }),
        el("span", { class: "small", text: better.corruption === "clean" ? "(the original photo)" : `(same item, ${nice(better.corruption)} severity ${better.severity})` })) : null);
  }

  function renderChat() {
    const a = S.assist;
    const input = el("input", { type: "text", placeholder: "Ask the assistant…", "aria-label": "Message", disabled: S.busy ? "" : null });
    const send = () => { const t = input.value.trim(); if (!t) return; S.chat.push({ who: "u", text: t }); runAssist(t); };
    input.addEventListener("keydown", e => { if (e.key === "Enter") send(); });
    const chat = el("div", { class: "chat", "aria-live": "polite" }, S.chat.map(m => el("div", { class: `msg ${m.who}`, text: m.text })),
      S.busy ? el("div", { class: "msg a small", text: "Thinking…" }) : null);
    const cmp = a && a.effective_decision === "caution" ? el("div", { class: "cmp" },
      a.comparison ? el("p", { text: a.comparison }) : null,
      el("div", { class: "row" }, el("span", { class: "small", text: a.candidates.length > 1 ? "Which is it?" : "Is this right?" }),
        a.candidates.map(c => el("button", { class: "btn", type: "button", text: a.candidates.length > 1 ? `${c.class} (${pct(c.prob)})` : `Yes, it's a ${c.class}`,
          onclick: () => { S.confirmed = c.class; S.chat.push({ who: "u", text: `It's a ${c.class}.` }); runAssist(`It's a ${c.class}.`); } })))) : null;
    return el("section", { class: "card" },
      el("div", { class: "row" }, el("h2", { text: "3 · Assistant" }), a && !a.llm_used ? el("span", { class: "offline", text: "Offline mode" }) : null),
      chat, cmp, el("div", { class: "row" }, input, el("button", { class: "btn", type: "button", text: "Send", onclick: send, disabled: S.busy ? "" : null })));
  }

  function renderProducts() {
    const a = S.assist;
    if (!a || !a.products.length) return null;
    return el("section", { class: "card" }, el("h2", { text: a.cart_allowed ? "4 · Picked for you" : "4 · Options (cart locked until confirmed)" }),
      el("div", { class: "products" }, a.products.map(p => el("div", { class: "prod" },
        el("strong", { text: p.name }), el("span", { class: "price", text: money(p.price) }),
        el("span", { class: "why", text: p.why_for_you }),
        el("span", { class: "small" }, `${p.description ? p.description + " · " : ""}approx. price · `,
          p.search_url ? el("a", { href: p.search_url, target: "_blank", rel: "noopener noreferrer", text: "Search ↗" }) : null),
        a.cart_allowed ? el("div", { class: "row" },
          el("button", { class: "btn", type: "button", "aria-label": `Remove one ${p.name}`, text: "−", onclick: () => setQty(p.product_id, -1) }),
          el("span", { text: String(S.cart[p.product_id] || 0) }),
          el("button", { class: "btn", type: "button", "aria-label": `Add one ${p.name}`, text: "+", onclick: () => setQty(p.product_id, 1) })) : null))));
  }

  function renderCheckout() {
    const a = S.assist;
    if (!a) return null;
    const q = S.quote, pol = q || a.checkout_policy;
    const locked = !a.cart_allowed;
    const need = q ? q.confirmations_required || 0 : 0;
    const label = locked ? "Checkout locked" : !q ? "Add an item to the cart" :
      need === 0 ? `Buy now · ${money(q.total)}` : S.confirmations < need ? `Confirm purchase${need > 1 ? ` (${S.confirmations + 1} of ${need})` : ""} · ${money(q.total)}` : `Place order · ${money(q.total)}`;
    const order = S.order ? (S.order.error ? el("p", { class: "err", text: S.order.error })
      : el("div", { class: "notice" }, el("strong", { text: `Order ${S.order.order_id} placed · ${money(S.order.total)}` }), el("br"),
          el("span", { class: "small", text: S.order.note }))) : null;
    return el("section", { class: "card" },
      el("div", { class: "row" }, el("h2", { text: "5 · Checkout" }), el("span", { class: "tier", text: TIER[pol.tier] || pol.tier })),
      el("p", { class: "sub", text: pol.explanation }),
      a.effective_decision === "trust" ? el("p", { class: "small", text: `One-tap limit ${money(a.checkout_policy.max_one_tap_total)} (demo setting).` }) : null,
      el("div", { class: "row" }, el("button", { class: "btn primary", type: "button", text: label, onclick: onBuy,
        disabled: locked || !q || !q.allowed || (S.order && !S.order.error) ? "" : null })),
      order);
  }

  function renderProfileEditor() {
    const d = S.draft;
    const chip = t => el("button", { type: "button", class: "btn", "aria-pressed": String(d.style_tags.includes(t)),
      style: d.style_tags.includes(t) ? "background:var(--trust);color:#fff;border-color:transparent" : null, text: t,
      onclick: () => { d.style_tags = d.style_tags.includes(t) ? d.style_tags.filter(x => x !== t) : [...d.style_tags, t].slice(0, 4); render(); } });
    const byClass = {};
    for (const p of S.meta.catalog) (byClass[p.class] = byClass[p.class] || []).push(p);
    const owned = el("select", { multiple: "", size: "6", "aria-label": "Things you already own", style: "width:100%",
      onchange: e => { d.owned = [...e.target.selectedOptions].map(o => o.value).slice(0, 12); } },
      Object.entries(byClass).map(([c, ps]) => el("optgroup", { label: c }, ps.map(p =>
        el("option", { value: p.product_id, selected: d.owned.includes(p.product_id) ? "" : null, text: p.name })))));
    const save = () => {
      const budget = Number(d.budget_per_item);
      if (!(budget >= 1 && budget <= 10000)) { S.formError = "Budget per item must be between $1 and $10,000."; return render(); }
      S.formError = null; S.me = { ...d, budget_per_item: budget }; store.set(S.me); S.mode = "me"; S.editing = false;
      select(S.base, S.photo);
    };
    return el("section", { class: "card", "aria-labelledby": "shop-prof-h" },
      el("h2", { id: "shop-prof-h", text: "Your shopping profile" }),
      el("p", { class: "sub", text: "The assistant personalizes picks with this, and your careful-checkout setting sets how much " +
        "confirmation checkout asks for. Saved in this browser only." }),
      el("div", { class: "row" },
        el("label", { class: "small", for: "shop-name", text: "Name" }),
        el("input", { id: "shop-name", type: "text", value: d.name, maxlength: "40", placeholder: "Your name",
          oninput: e => { d.name = e.target.value; } }),
        el("label", { class: "small", for: "shop-budget", text: "Budget per item ($)" }),
        el("input", { id: "shop-budget", type: "number", min: "1", max: "10000", step: "1", value: String(d.budget_per_item),
          style: "width:7em", oninput: e => { d.budget_per_item = e.target.value; } })),
      el("div", { class: "row" }, el("span", { class: "small", text: "Your style (up to 4):" }), S.meta.style_tags.map(chip)),
      el("div", { class: "row" }, el("span", { class: "small", text: "Careful checkout:" }),
        el("div", { class: "seg", role: "group", "aria-label": "Careful checkout" }, Object.keys(S.meta.risk_levels).map(r =>
          el("button", { type: "button", "aria-pressed": String(d.risk === r), text: r[0].toUpperCase() + r.slice(1),
            title: RISK_TEXT[r], onclick: () => { d.risk = r; render(); } }))),
        el("span", { class: "small", text: RISK_TEXT[d.risk] })),
      el("div", {}, el("span", { class: "small", text: "Things you already own (optional; Cmd/Ctrl-click for several):" }), owned),
      S.formError ? el("p", { class: "err", text: S.formError }) : null,
      el("div", { class: "row" }, el("button", { class: "btn primary", type: "button", text: "Save profile", onclick: save }),
        S.me ? el("button", { class: "btn", type: "button", text: "Cancel", onclick: () => { S.editing = false; render(); } }) : null));
  }

  function renderShopperBar() {
    const sel = el("select", { id: "shop-profile", "aria-label": "Shopper",
      onchange: e => { S.mode = e.target.value; select(S.base, S.photo); } },
      el("option", { value: "me", selected: S.mode === "me" ? "" : null, text: `You${S.me && S.me.name ? ` (${S.me.name})` : ""}` }),
      el("optgroup", { label: "Examples (fictional)" }, S.meta.examples.map(p => el("option", { value: p.profile_id,
        selected: p.profile_id === S.mode ? "" : null, text: `${p.name} · ${money(p.budget_per_item)}/item · ${p.style_tags.join(", ")}` }))));
    const cur = S.mode === "me" ? S.me : S.meta.examples.find(p => p.profile_id === S.mode);
    return el("div", { class: "row" }, el("label", { class: "small", for: "shop-profile", text: "Shopper" }), sel,
      S.mode === "me" && S.me ? el("button", { class: "btn", type: "button", text: "Edit profile",
        onclick: () => { S.draft = { ...S.me }; S.editing = true; render(); } }) : null,
      cur ? el("span", { class: "small", text: `${money(cur.budget_per_item)}/item · ${cur.style_tags.join(", ") || "no style set"} · ${(cur.risk || "normal")} checkout` }) : null);
  }

  function renderStory() {
    const st = S.samples.story || {};
    const steps = [["1 · TRUST: one-tap", st.trust, 1], ["2 · TRUST: big order", st.trust, "big"], ["3 · CAUTION: confirm", st.caution, 1], ["4 · REJECT: retake", st.reject, 1]];
    return el("div", { class: "row" }, el("span", { class: "small", text: "Demo story:" }),
      steps.filter(s => s[1]).map(([t, v, q]) => el("button", { class: "btn", type: "button", text: t,
        onclick: async () => {
          await select(v.base_image_id, v.photo_id);
          if (S.assist && S.assist.cart_allowed && S.assist.products.length) {
            const p = S.assist.products[0];
            const qty = q === "big" ? Math.max(2, Math.ceil((S.assist.checkout_policy.max_one_tap_total + 1) / p.price)) : 1;
            S.cart = { [p.product_id]: Math.min(20, qty) }; quote();
          }
        } })));
  }

  function render() {
    if (!root || !S.samples) return;
    root.textContent = "";
    if (S.editing || (S.mode === "me" && !S.me)) {  // first visit: set up your own profile before shopping
      if (!S.draft) S.draft = { name: "", budget_per_item: 100, style_tags: [], risk: "normal", owned: [] };
      root.append(renderProfileEditor(),
        el("p", { class: "small", text: "Just looking? Pick an example shopper instead:" }),
        el("div", { class: "row" }, S.meta.examples.map(p => el("button", { class: "btn", type: "button", text: `${p.name} (example)`,
          onclick: () => { S.mode = p.profile_id; S.editing = false; select(S.base, S.photo); } }))));
      return;
    }
    root.append(
      el("section", { class: "card" },
        el("div", { class: "card-head" },
          el("div", {}, el("h2", { text: "Snap to Shop: a shopping agent that knows when it might be wrong" }),
            el("p", { class: "sub", text: "Upload a product photo; the same trust layer decides what the AI assistant may do. " +
              "TRUST allows one-tap checkout, CAUTION asks which item you meant, REJECT asks for a better photo. " +
              "Rules are enforced in code, not by the AI. Products are real models with approximate prices (not affiliated); " +
              "example shoppers are fictional; checkout is a mock and nothing is charged." })),
          renderShopperBar()),
        renderStory()),
      el("div", { class: "shop-grid" },
        el("div", { style: "display:grid;gap:18px" }, renderPicker(), renderTrust(), renderCoach()),
        el("div", { style: "display:grid;gap:18px" }, renderChat(), renderProducts(), renderCheckout())));
  }

  function install() {
    document.head.append(el("style", { text: CSS }));
    const nav = document.querySelector("nav.tabs"), main = document.querySelector("main");
    const btn = el("button", { role: "tab", id: "tab-shop", "aria-controls": "shop", "aria-selected": "false", text: "Snap to Shop" });
    root = el("section", { id: "shop", role: "tabpanel", "aria-labelledby": "tab-shop", hidden: "" });
    nav.append(btn); main.append(root);
    const others = ["demo", "research"];
    btn.addEventListener("click", () => {
      for (const t of others) { document.getElementById(t).hidden = true; document.getElementById(`tab-${t}`).setAttribute("aria-selected", "false"); }
      root.hidden = false; btn.setAttribute("aria-selected", "true"); render();
    });
    for (const t of others) document.getElementById(`tab-${t}`).addEventListener("click", () => { root.hidden = true; btn.setAttribute("aria-selected", "false"); });
  }

  (async () => {
    const [s, p] = await Promise.all([api("/api/shop/samples"), api("/api/shop/profiles")]).catch(() => [{}, {}]);
    if (!s.ok || !p.ok) return;   // shop not installed: leave the page untouched
    S.samples = s.j; S.meta = p.j; S.me = store.get();
    install();
    (async function poll() {   // the live model loads in the background after the server starts
      const r = await api("/api/shop/status").catch(() => ({ ok: false, j: {} }));
      S.uploadStatus = r.ok ? r.j.upload : "error"; S.uploadErrorDetail = r.j.upload_error;
      render();
      if (S.uploadStatus === "loading") setTimeout(poll, 1500);
    })();
    const first = S.samples.story && S.samples.story.trust ? S.samples.story.trust : S.samples.photos[0].versions[0];
    S.base = first.base_image_id; S.photo = first.photo_id;
    if (S.me) select(first.base_image_id, first.photo_id); else render();
  })();
})();
