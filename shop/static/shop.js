// Shopping-assistant tab (the trust layer in agentic commerce). Installs itself only if /api/shop/samples answers; otherwise does nothing,
// so the core demo is unchanged when shop/ is absent. Uses the page's CSS tokens. No payment data anywhere.
(() => {
  const WANT_SHOP = location.hash === "#shop";   // read now: the core page rewrites the hash while it loads
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
  const pctErr = x => 100 * x < 9.95 ? `${(100 * x).toFixed(1)}%` : pct(x);   // small error rates need the decimal (0.8%, not 1%)
  const a_ = cls => { const n = ({ sunglasses: "pair of sunglasses", jean: "pair of jeans", binoculars: "pair of binoculars",
    "cellular telephone": "phone", Loafer: "loafer" })[cls] || cls; return (/^[aeiou]/i.test(n) ? "an " : "a ") + n; };
  const nice = s => s === "clean" ? "clean" : s.replace(/_/g, " ");
  const TIER = { one_tap: "One-tap", confirm: "Confirm once", confirm_twice: "Confirm twice", blocked_until_confirmed: "Locked", blocked: "Locked" };
  const DEC = { trust: "TRUST", trust_confirmed: "TRUST (you confirmed)", caution: "CAUTION", reject: "REJECT" };

  const CSS = `
  #shop { display: grid; gap: 16px; }
  body.shop-active #provenance { display: none; }   /* the CIFAR results line belongs to the core tabs */
  #shop .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  #shop select, #shop input[type=text], #shop input[type=number] { font: inherit; color: var(--ink); background: var(--page); border: 1px solid var(--border); border-radius: 8px; padding: 7px 9px; }
  #shop input[type=text] { flex: 1 1 200px; min-width: 0; }
  #shop .card { border-radius: 14px; }
  #shop .head { display: flex; flex-wrap: wrap; gap: 10px 24px; align-items: center; justify-content: space-between; }
  #shop .head h2 { font-size: 18px; }
  #shop .chips { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
  #shop .chip { border: 1px solid var(--border); background: var(--page); border-radius: 999px; padding: 5px 11px; font-size: 13px; cursor: pointer; }
  #shop .chip:hover { border-color: var(--axis); }
  #shop .chip .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: 1px; }
  #shop .strip { display: flex; gap: 8px; overflow-x: auto; padding: 2px 2px 6px; scrollbar-width: thin; }
  #shop .strip button { flex: 0 0 64px; height: 64px; padding: 0; border: 2px solid transparent; border-radius: 10px; background: none; cursor: pointer; overflow: hidden; }
  #shop .strip button[aria-pressed=true] { border-color: var(--trust); }
  #shop .strip img { width: 100%; height: 100%; object-fit: cover; display: block; }
  #shop .main { display: grid; gap: 16px; align-items: start;
    grid-template-columns: minmax(0, 300px) minmax(0, 1fr) minmax(0, 1.1fr); grid-template-areas: "photo trust chat"; }
  @media (max-width: 1100px) { #shop .main { grid-template-columns: minmax(0, 280px) minmax(0, 1fr); grid-template-areas: "photo trust" "chat chat"; } }
  @media (max-width: 700px) { #shop .main { grid-template-columns: minmax(0, 1fr); grid-template-areas: "photo" "trust" "chat"; } }
  @media (max-width: 700px) { #shop .photo { max-height: 240px; } }
  #shop .a-photo { grid-area: photo; } #shop .a-trust { grid-area: trust; } #shop .a-chat { grid-area: chat; }
  #shop .shop-bottom { display: grid; gap: 16px; align-items: start; grid-template-columns: minmax(0, 2fr) minmax(0, 1fr); }
  @media (max-width: 900px) { #shop .shop-bottom { grid-template-columns: minmax(0, 1fr); } }
  #shop .photo-wrap { position: relative; }
  #shop .photo { width: 100%; aspect-ratio: 1; border-radius: 12px; object-fit: cover; background: var(--sunken); display: block; }
  #shop .photo-wrap .badge { position: absolute; top: 10px; left: 10px; box-shadow: 0 1px 4px var(--shadow); }
  #shop .seg { display: inline-flex; background: var(--sunken); border-radius: 8px; padding: 2px; gap: 2px; }
  #shop .seg button { border: 0; background: none; border-radius: 6px; padding: 5px 10px; cursor: pointer; font-size: 13px; }
  #shop .seg button[aria-pressed=true] { background: var(--surface); box-shadow: 0 1px 2px var(--shadow); font-weight: 600; }
  #shop .badge { display: inline-block; font-weight: 700; font-size: 13px; padding: 4px 10px; border-radius: 999px; letter-spacing: .02em; }
  #shop .b-trust, #shop .b-trust_confirmed { background: color-mix(in srgb, var(--good) 22%, var(--surface)); color: var(--good-ink); }
  #shop .b-caution { background: color-mix(in srgb, var(--warn) 30%, var(--surface)); color: var(--warn-ink); }
  #shop .b-reject { background: color-mix(in srgb, var(--crit) 22%, var(--surface)); color: var(--crit-ink); }
  #shop .verdict { border-left: 5px solid var(--dc); }
  #shop .d-trust, #shop .d-trust_confirmed { --dc: var(--good); } #shop .d-caution { --dc: var(--warn); } #shop .d-reject { --dc: var(--crit); }
  #shop .verdict-title { font-size: 20px; font-weight: 700; color: var(--ink); }
  #shop .why-line { font-size: 14.5px; line-height: 1.45; margin: 0; }
  #shop .meter { display: grid; grid-template-columns: 11.5em 1fr 3.2em; gap: 8px; align-items: center; font-size: 13px; }
  #shop .meter .track { height: 9px; background: var(--sunken); border-radius: 999px; overflow: hidden; }
  #shop .meter .fill { height: 100%; border-radius: 999px; }
  #shop .meter .v { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
  #shop .coach { background: color-mix(in srgb, var(--warn) 12%, var(--surface)); border: 1px solid color-mix(in srgb, var(--warn) 45%, transparent); border-radius: 10px; padding: 10px 12px; display: grid; gap: 6px; }
  #shop .coach strong { font-size: 14px; }
  #shop .pills { display: flex; flex-wrap: wrap; gap: 5px; }
  #shop .pill { font-size: 12px; padding: 2px 8px; border-radius: 999px; background: var(--sunken); color: var(--ink-2); }
  #shop .chat { display: grid; gap: 8px; max-height: 300px; min-height: 120px; overflow-y: auto; padding: 2px; align-content: start; }
  #shop .msg { padding: 9px 12px; border-radius: 12px; font-size: 14px; max-width: 92%; line-height: 1.4; }
  #shop .msg.a { background: var(--sunken); justify-self: start; border-bottom-left-radius: 4px; }
  #shop .msg.u { background: color-mix(in srgb, var(--trust) 16%, transparent); justify-self: end; border-bottom-right-radius: 4px; }
  #shop .msg.g { justify-self: stretch; max-width: none; font-size: 13px; font-weight: 600; color: var(--crit-ink);
    background: color-mix(in srgb, var(--crit) 10%, transparent); border: 1px solid color-mix(in srgb, var(--crit) 35%, transparent); }
  #shop .msg.g::before { content: "🔒 "; }
  #shop .msg.n { justify-self: stretch; max-width: none; font-size: 12.5px; color: var(--ink-2); background: var(--page); border: 1px dashed var(--axis); }
  #shop .msg.n::before { content: "ⓘ "; }
  #shop .products { display: grid; gap: 10px; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); }
  #shop .prod { border: 1px solid var(--border); border-radius: 12px; padding: 12px; display: grid; gap: 6px; background: var(--page); align-content: start; }
  #shop .prod .top { display: flex; justify-content: space-between; gap: 8px; align-items: baseline; }
  #shop .prod .price { font-weight: 700; font-variant-numeric: tabular-nums; white-space: nowrap; }
  #shop .prod .why { font-size: 13px; color: var(--ink); }
  #shop .qty { display: inline-flex; align-items: center; gap: 8px; }
  #shop .qty .btn { padding: 5px 10px; }
  #shop .cmp { border: 1px dashed var(--warn); border-radius: 10px; padding: 10px 12px; display: grid; gap: 8px; background: color-mix(in srgb, var(--warn) 7%, var(--surface)); }
  #shop .tier { font-weight: 700; font-size: 15px; }
  #shop .btn.primary { background: var(--trust); color: #fff; border-color: transparent; }
  #shop .btn.big { width: 100%; padding: 12px; font-size: 15px; }
  #shop .btn[disabled] { opacity: .5; cursor: not-allowed; }
  #shop .btn.primary.locked { background: var(--sunken); color: var(--ink-2); border: 1px solid var(--border); opacity: 1; }
  #shop select { width: auto; max-width: 100%; }
  #shop .head > .row { justify-content: flex-end; }
  #shop .small { font-size: 12.5px; color: var(--muted); }
  #shop .offline { font-size: 12px; padding: 2px 8px; border-radius: 999px; background: var(--sunken); color: var(--ink-2); cursor: help; }
  #shop .receipt { display: grid; gap: 8px; border: 1px solid color-mix(in srgb, var(--good) 45%, transparent);
    background: color-mix(in srgb, var(--good) 8%, var(--surface)); border-radius: 10px; padding: 10px 12px; }
  #shop .receipt summary { cursor: pointer; font-size: 13.5px; font-weight: 600; margin-bottom: 6px; }
  #shop .kv { display: grid; grid-template-columns: auto 1fr; gap: 3px 12px; font-size: 13px; }
  #shop .kv span:nth-child(odd) { color: var(--ink-2); }
  #shop .ok-line { color: var(--good-ink); font-weight: 600; font-size: 13.5px; margin: 0; }
  #shop .eval { gap: 18px; }
  #shop .eval-toggle > summary { cursor: pointer; list-style: none; display: flex; gap: 10px; align-items: flex-start; }
  #shop .eval-toggle > summary::-webkit-details-marker { display: none; }
  #shop .eval-toggle > summary::before { content: "▸"; font-size: 15px; line-height: 1.5; color: var(--ink-2); transition: transform .15s; }
  #shop .eval-toggle[open] > summary::before { transform: rotate(90deg); }
  #shop .eval-title { display: grid; gap: 2px; }
  #shop .eval-body { display: grid; gap: 18px; margin-top: 16px; }
  #shop .eval-block { display: grid; gap: 10px; grid-template-columns: minmax(0, 1fr); }   /* wide tables scroll in .tscroll, not the page */
  #shop .eval h3 { font-size: 15px; }
  #shop .tiles { display: grid; gap: 10px; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); }
  #shop .tile { border: 1px solid var(--border); border-radius: 12px; padding: 10px 12px; background: var(--page); display: grid; gap: 2px; border-top: 4px solid var(--axis); }
  #shop .tile.good { border-top-color: var(--good); } #shop .tile.warn { border-top-color: var(--warn); } #shop .tile.crit { border-top-color: var(--crit); }
  #shop .tile .big { font-size: 24px; font-weight: 700; font-variant-numeric: tabular-nums; }
  #shop .tile .lbl { font-size: 13px; }
  #shop .compare { display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
  #shop .compare > div { border-radius: 10px; padding: 10px 12px; background: var(--sunken); font-size: 13.5px; }
  #shop .compare > div:last-child { background: color-mix(in srgb, var(--good) 10%, var(--surface)); }
  #shop .compare p { margin-top: 4px; }
  #shop .tscroll { overflow-x: auto; }
  #shop .eval table td, #shop .eval table th { text-align: left; }
  #shop .mini { display: grid; grid-template-columns: 90px 3em; gap: 6px; align-items: center; }
  #shop .mini .track { height: 7px; background: var(--sunken); border-radius: 999px; overflow: hidden; }
  #shop .mini .fill { height: 100%; }
  #shop .notes { margin: 0; padding-left: 18px; display: grid; gap: 4px; font-size: 13.5px; color: var(--ink-2); }
  #shop .eval summary { cursor: pointer; font-weight: 600; font-size: 13.5px; }
  #shop .takeaway { margin: 0; font-weight: 600; font-size: 14.5px; }
  #shop .fine { font-size: 12px; color: var(--muted); text-align: center; }
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
    if (file) return uploadDataURL(await toDataURL(file));
  }
  async function uploadDataURL(dataURL) {
    S.uploading = true; S.uploadError = null; render();
    const r = await api("/api/shop/upload", { image: dataURL }).catch(e => ({ ok: false, j: { error: String(e) } }));
    S.uploading = false;
    if (!r.ok) { S.uploadError = r.j.error || `Upload failed (HTTP ${r.status})`; return render(); }
    const it = r.j;
    S.uploads = [{ base_image_id: it.base_image_id, true_class: null, versions: [it] }, ...S.uploads.filter(u => u.base_image_id !== it.base_image_id)].slice(0, 6);
    S.chat.push({ who: "u", text: "(uploaded a photo)" });
    select(it.base_image_id, it.photo_id, { keepChat: true });
  }
  // ---- webcam: live preview, capture one frame, same upload path (camera access works on localhost) ----
  const hasCamera = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  async function openCamera() {
    S.cameraError = null;
    try {
      S.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 } }, audio: false });
    } catch (e) {
      S.cameraError = e.name === "NotAllowedError" ? "Camera permission was denied; allow it in the browser's address bar."
        : `Couldn't open a camera (${e.name || e}).`;
    }
    render();
  }
  function closeCamera() {
    if (S.stream) S.stream.getTracks().forEach(t => t.stop());
    S.stream = null; render();
  }
  function capture(video) {
    const c = document.createElement("canvas"), k = Math.min(1, 800 / Math.max(video.videoWidth, video.videoHeight));
    c.width = Math.round(video.videoWidth * k); c.height = Math.round(video.videoHeight * k);
    c.getContext("2d").drawImage(video, 0, 0, c.width, c.height);
    const url = c.toDataURL("image/jpeg", 0.9);
    closeCamera();
    uploadDataURL(url);
  }
  function renderCamera() {
    if (!S.stream) return S.cameraError ? el("p", { class: "err", text: S.cameraError }) : null;
    const video = el("video", { autoplay: "", playsinline: "", muted: "", class: "photo", style: "object-fit:cover" });
    video.srcObject = S.stream;
    return el("div", { class: "camera" }, video,
      el("div", { class: "row" }, el("button", { class: "btn primary", type: "button", text: "Capture", onclick: () => capture(video) }),
        el("button", { class: "btn", type: "button", text: "Cancel", onclick: closeCamera }),
        el("span", { class: "small", text: "Fill the frame with one item." })));
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
    if (!r.j.llm_used && userMessage && !S.offlineNoted) { S.chat.push({ who: "n", text: OFFLINE_NOTE }); S.offlineNoted = true; }
    if (r.j.llm_used) S.offlineNoted = false;
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
    S.verify = null;
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

  const DEC_TITLE = { trust: "Safe to buy", trust_confirmed: "You confirmed it", caution: "Not sure: check with you", reject: "Can't identify: retake" };
  const DEC_COLOR = { trust: "var(--good)", trust_confirmed: "var(--good)", caution: "var(--warn)", reject: "var(--crit)" };

  function renderSource() {
    const photos = [...S.uploads, ...S.samples.photos];
    return el("section", { class: "card", "aria-label": "Choose a photo" },
      el("div", { class: "row" }, el("h2", { text: "Snap a product" }), uploadButton("Upload a photo"),
        hasCamera && S.uploadStatus === "ready" ? el("button", { class: "btn", type: "button", text: S.stream ? "Camera on" : "Use camera",
          disabled: S.stream || S.uploading ? "" : null, onclick: openCamera }) : null,
        el("span", { class: "small", text: "…or tap a sample photo:" })),
      renderCamera(),
      S.uploadError ? el("p", { class: "err", text: S.uploadError }) : null,
      el("div", { class: "strip" }, photos.map(p => { const v0 = p.versions.find(v => ["clean", "upload"].includes(v.corruption));
        return el("button", { type: "button", "aria-pressed": String(p.base_image_id === S.base), title: p.true_class || "your photo",
          onclick: () => select(p.base_image_id, v0.photo_id) },
          el("img", { src: v0.image, alt: p.true_class || "your photo" })); })));
  }

  function renderPhoto() {
    const vers = versionsOf(S.base), cur = item(), a = S.assist;
    const isUpload = cur.corruption === "upload";
    const corrs = [...new Set(vers.map(v => v.corruption))];
    const active = cur.corruption === "clean" ? corrs.find(c => c !== "clean") : cur.corruption;
    const pick = (c, s) => { const v = vers.find(x => x.corruption === c && x.severity === s) || vers.find(x => x.corruption === "clean"); select(S.base, v.photo_id); };
    const corrSel = el("select", { "aria-label": "Simulate a bad photo", onchange: e => pick(e.target.value, e.target.value === "clean" ? 0 : Math.max(cur.severity, 1)) },
      corrs.map(c => el("option", { value: c, selected: c === cur.corruption ? "" : null, text: c === "clean" ? "original photo" : nice(c) })));
    const sevs = [...new Set(vers.filter(v => v.corruption === active || v.corruption === "clean").map(v => v.severity))].sort();
    const sevSeg = el("div", { class: "seg", role: "group", "aria-label": "Severity" }, sevs.map(sv => el("button", { type: "button",
      "aria-pressed": String(sv === cur.severity), text: sv === 0 ? "clean" : `${sv}`, title: sv === 0 ? "original" : `severity ${sv}`,
      onclick: () => pick(sv === 0 ? "clean" : active, sv) })));
    return el("section", { class: "card a-photo" },
      el("div", { class: "photo-wrap" }, el("img", { class: "photo", src: cur.image, alt: isUpload ? "your photo" : `${cur.true_class}, ${nice(cur.corruption)}` }),
        a ? el("span", { class: `badge b-${a.effective_decision}`, text: DEC[a.effective_decision] }) : null),
      isUpload ? el("p", { class: "small", text: "Your photo, scored live. The model knows 30 product types; anything else should come back CAUTION or REJECT." })
        : el("div", { class: "row" }, el("span", { class: "small", text: "Simulate:" }), corrSel, cur.corruption === "clean" ? null : sevSeg));
  }

  function meter(label, value, color) {
    return el("div", { class: "meter" }, el("span", { class: "small", text: label }),
      el("div", { class: "track" }, el("div", { class: "fill", style: `width:${Math.round(100 * value)}%;background:${color}` })),
      el("span", { class: "v", text: pct(value) }));
  }

  function clearerVersion(cur) {
    // Same product, gentler corruption: the lowest-severity version that the trust layer doesn't REJECT.
    if (cur.corruption === "upload") return null;
    const vers = versionsOf(S.base).filter(v => v.corruption === cur.corruption && v.severity < cur.severity && v.decision !== "reject");
    const clean = versionsOf(S.base).find(v => v.corruption === "clean");
    return vers.sort((a, b) => b.severity - a.severity)[0] || (clean && clean.photo_id !== cur.photo_id ? clean : null);
  }

  function renderCoach(a, cur) {
    if (!["reject", "caution"].includes(a.effective_decision)) return null;
    const q = cur.quality || {};
    if (a.effective_decision === "caution" && !q.issue) return null;
    const better = clearerVersion(cur);
    return el("div", { class: "coach", role: "note" },
      el("strong", { text: q.issue ? `📷 Retake tip: this photo looks ${q.label}` : "📷 Retake tip" }),
      el("span", { text: q.tip || "Try a clearer photo of just the item." }),
      el("div", { class: "row" },
        cur.corruption === "upload" ? uploadButton("Upload a new photo") : null,
        better ? el("button", { class: "btn", type: "button", text: "Try a clearer shot",
          onclick: () => { S.chat.push({ who: "u", text: "Here's a clearer photo." }); select(S.base, better.photo_id, { keepChat: true }); } }) : null,
        better ? el("span", { class: "small", text: better.corruption === "clean" ? "(the original photo)" : `(same item, severity ${better.severity})` }) : null));
  }

  function renderTrust() {
    const a = S.assist, cur = item();
    if (!a) return el("section", { class: "card a-trust" }, el("p", { class: "small", text: "Checking…" }));
    const eff = a.effective_decision;
    return el("section", { class: `card a-trust verdict d-${eff}`, "aria-label": "Trust check" },
      el("div", { class: "row" }, el("span", { class: "verdict-title", text: DEC_TITLE[eff] }), el("span", { class: `badge b-${eff}`, text: DEC[eff] })),
      a.explanation ? el("p", { class: "why-line", text: a.explanation }) : null,
      renderCoach(a, cur),
      el("div", { style: "display:grid;gap:6px" },
        el("p", { class: "small", text: `The model says “${cur.pred_class}”. How sure should we be?` }),
        meter("Model's own confidence", cur.raw_confidence, "var(--raw)"),
        meter("Trust layer (p_correct)", a.p_correct, DEC_COLOR[eff])),
      el("div", { class: "pills" }, el("span", { class: "small", text: "Assistant may:" }), a.allowed_actions.map(x => el("span", { class: "pill", text: nice(x) }))));
  }

  const OFFLINE_NOTE = "The AI assistant is offline right now (no API key, no network, or it took too long), so you're seeing " +
    "pre-written replies. They follow the same rules, but they won't change with what you type: asking again gives the same answer.";

  function renderChat() {
    const a = S.assist;
    const input = el("input", { type: "text", placeholder: "Ask the assistant (e.g. “anything cheaper?”)", "aria-label": "Message", disabled: S.busy ? "" : null });
    const send = () => { const t = input.value.trim(); if (!t) return; S.chat.push({ who: "u", text: t }); runAssist(t); };
    input.addEventListener("keydown", e => { if (e.key === "Enter") send(); });
    const chat = el("div", { class: "chat", "aria-live": "polite" }, S.chat.map(m => el("div", { class: `msg ${m.who}`, text: m.text })),
      S.busy ? el("div", { class: "msg a small", text: "Thinking…" }) : null);
    const cmp = a && a.effective_decision === "caution" ? el("div", { class: "cmp" },
      a.comparison ? el("p", { text: a.comparison }) : null,
      el("div", { class: "row" }, el("span", { class: "small", text: a.candidates.length > 1 ? "Which is it?" : "Is this right?" }),
        a.candidates.map(c => el("button", { class: "btn", type: "button", text: a.candidates.length > 1 ? `${c.class} (${pct(c.prob)})` : `Yes, it's a ${c.class}`,
          onclick: () => { S.confirmed = c.class; S.chat.push({ who: "u", text: `It's a ${c.class}.` }); runAssist(`It's a ${c.class}.`); } })))) : null;
    return el("section", { class: "card a-chat" },
      el("div", { class: "row" }, el("h2", { text: "Chat" }),
        a && !a.llm_used ? el("span", { class: "offline", title: OFFLINE_NOTE, text: "Offline: fixed replies" }) : null),
      chat, cmp, el("div", { class: "row" }, input, el("button", { class: "btn", type: "button", text: "Send", onclick: send, disabled: S.busy ? "" : null })));
  }

  function renderProducts() {
    const a = S.assist;
    if (!a || !a.products.length) return el("section", { class: "card" }, el("h2", { text: "Picks for you" }),
      el("p", { class: "small", text: a && a.effective_decision === "reject" ? "No products until the photo can be identified." : "Loading…" }));
    return el("section", { class: "card" }, el("h2", { text: a.cart_allowed ? "Picks for you" : "Options (the cart unlocks once you confirm the item)" }),
      el("div", { class: "products" }, a.products.map(p => el("div", { class: "prod" },
        el("div", { class: "top" }, el("strong", { text: p.name }), el("span", { class: "price", text: money(p.price) })),
        el("span", { class: "why", text: p.why_for_you }),
        el("span", { class: "small" }, `${p.description ? p.description + " · " : ""}approx. price · `,
          p.search_url ? el("a", { href: p.search_url, target: "_blank", rel: "noopener noreferrer", text: "Search ↗" }) : null),
        a.cart_allowed ? el("div", { class: "qty" },
          el("button", { class: "btn", type: "button", "aria-label": `Remove one ${p.name}`, text: "−", onclick: () => setQty(p.product_id, -1) }),
          el("span", { text: String(S.cart[p.product_id] || 0) }),
          el("button", { class: "btn", type: "button", "aria-label": `Add one ${p.name}`, text: "+", onclick: () => setQty(p.product_id, 1) })) : null))));
  }

  function renderCheckout() {
    const a = S.assist;
    if (!a) return el("section", { class: "card" });
    const q = S.quote, pol = q || a.checkout_policy;
    const locked = !a.cart_allowed;
    const need = q ? q.confirmations_required || 0 : 0;
    const label = locked ? "🔒 Checkout locked" : !q ? "Add an item to the cart" :
      need === 0 ? `Buy now · ${money(q.total)}` : S.confirmations < need ? `Confirm purchase${need > 1 ? ` (${S.confirmations + 1} of ${need})` : ""} · ${money(q.total)}` : `Place order · ${money(q.total)}`;
    const order = S.order ? (S.order.error ? el("p", { class: "err", text: S.order.error }) : renderReceipt(S.order)) : null;
    return el("section", { class: `card verdict d-${a.effective_decision}` },
      el("div", { class: "row" }, el("h2", { text: "Checkout" }), el("span", { class: "tier", text: TIER[pol.tier] || pol.tier })),
      el("p", { class: "sub", text: pol.explanation }),
      a.effective_decision === "trust" && a.checkout_policy.max_one_tap_total > 0
        ? el("p", { class: "small", text: `One-tap limit ${money(a.checkout_policy.max_one_tap_total)} (your careful-checkout setting).` }) : null,
      el("button", { class: `btn primary big${locked ? " locked" : ""}`, type: "button", text: label, onclick: onBuy,
        disabled: locked || !q || !q.allowed || (S.order && !S.order.error) ? "" : null }),
      order);
  }

  // ---- trust-trail receipt: the evidence each agent purchase was made on, fingerprinted and verifiable ----
  async function verifyReceipt(receipt, tamper) {
    const r = tamper ? JSON.parse(JSON.stringify(receipt)) : receipt;
    if (tamper) r.evidence.p_correct = Math.min(1, r.evidence.p_correct + 0.2);   // pretend someone inflated the score
    const res = await api("/api/shop/verify", { receipt: r });
    S.verify = { tamper, valid: res.ok && res.j.valid };
    render();
  }
  function renderReceipt(o) {
    const e = o.evidence || {}, th = e.thresholds || {};
    const row = (k, v) => [el("span", { text: k }), el("span", { text: v })];
    const decided = e.shopper_confirmed ? `${e.decision.toUpperCase()} → you confirmed “${e.shopper_confirmed}”` : e.effective_decision.toUpperCase();
    const v = S.verify;
    return el("div", { class: "receipt" },
      el("strong", { text: `✓ Order ${o.order_id} placed · ${money(o.total)}` }),
      el("span", { class: "small", text: `Paid with a ${o.payment.method} (${o.payment.token}), bound to this order only. ${o.payment.note}` }),
      el("details", {}, el("summary", { text: "Trust trail: why this purchase was allowed" }),
        el("div", { class: "kv" },
          row("Model said", `${e.model_said} (its own confidence ${pct(e.raw_confidence)})`),
          row("Trust layer", `p_correct ${pct(e.p_correct)}: TRUST at ≥ ${pct(th.trust_at_or_above)}, REJECT below ${pct(th.reject_below)}`),
          row("Decision", decided),
          row("Checkout", `${TIER[e.checkout_tier] || e.checkout_tier}, ${e.confirmations_given} of ${e.confirmations_required} confirmations · ${e.careful_checkout} setting`),
          row("Items", e.items.map(i => `${i.qty} × ${i.name}`).join(", ")),
          row("Time (UTC)", e.time_utc),
          row("Fingerprint", `${o.fingerprint.slice(0, 16)}…`))),
      el("div", { class: "row" },
        el("button", { class: "btn", type: "button", text: "Verify receipt", onclick: () => verifyReceipt(o, false) }),
        el("button", { class: "btn", type: "button", text: "Tamper test", title: "Inflate p_correct in a copy and ask the server to verify it",
          onclick: () => verifyReceipt(o, true) })),
      v ? el("p", { class: v.valid ? "ok-line" : "err", text: v.valid ? "✓ Verified: this is exactly the evidence the server signed."
        : v.tamper ? "✗ Rejected: the evidence was altered (p_correct inflated), so the fingerprint no longer matches."
        : "✗ Not verified." }) : null);
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

  // ---- evaluation card: the committed results (held-out test photos + our own real phone photos) ----
  function renderEvaluation() {
    const E = S.evaluation;
    if (!E || (!E.test && !E.real_photos)) return null;
    const tile = (big, label, sub, tone) => el("div", { class: `tile ${tone || ""}` },
      el("div", { class: "big", text: big }), el("div", { class: "lbl", text: label }), sub ? el("div", { class: "small", text: sub }) : null);
    const bar = (v, color) => el("div", { class: "mini" }, el("div", { class: "track" }, el("div", { class: "fill", style: `width:${Math.round(100 * v)}%;background:${color}` })),
      el("span", { text: pct(v) }));
    const decBadge = d => el("span", { class: `badge b-${d}`, text: d.toUpperCase() });
    const verdict = r => !r.in_catalog ? (r.decision === "trust" ? "✗ trusted an unknown item" : "✓ not trusted")
      : r.decision === "trust" ? (r.correct ? "✓ right buy" : "✗ wrong buy")
      : r.decision === "caution" ? (r.correct ? "asked first (it was right)" : "✓ asked first (it was wrong)")
      : (r.correct ? "held back (it was right)" : "✓ stopped a mistake");
    const parts = [];

    const R = E.real_photos;
    if (R) {
      const sm = R.summary, rows = R.photos;
      const inc = rows.filter(r => r.in_catalog);
      const rejectedRight = inc.filter(r => r.correct && r.decision === "reject").length;
      const rawKey = Object.keys(sm.one_tap_purchases).find(k => k.startsWith("raw_confidence"));
      const rawCut = rawKey ? rawKey.split("_").pop() : "0.8";
      const naiveWrong = rows.filter(r => r.raw_confidence >= +rawCut && !r.correct);
      const gateN = sm.one_tap_purchases.trust_gate, gateWrong = sm.wrong_one_tap_purchases.trust_gate;
      const unknownTrusted = +String(sm.outside_catalog_trusted).split("/")[0];
      parts.push(el("div", { class: "eval-block" },
        el("h3", { text: `1 · Real phone photos we took (${sm.photos} photos, ${sm.outside_catalog} of them items outside the catalog)` }),
        el("p", { class: "takeaway", text: (gateWrong === 0 ? `All ${gateN} one-tap purchases the gate allowed were the right item`
          : `${gateWrong} of the ${gateN} one-tap purchases the gate allowed were the wrong item`) + (unknownTrusted === 0
          ? `, and it trusted none of the ${sm.outside_catalog} items the store doesn't sell.` : `, and it trusted ${sm.outside_catalog_trusted} items the store doesn't sell.`) }),
        el("p", { class: "sub", text: "Scored like an upload and never used in training, they include busy backgrounds and washed-out, dark and blurry shots." }),
        el("div", { class: "tiles" },
          tile(sm.trust_correct, "one-tap purchases were the right item", "the TRUST tier", "good"),
          tile(sm.outside_catalog_trusted, "unknown items trusted", "things the store doesn't sell", "good"),
          tile(sm.caution_true_item_offered, "times the right item was offered", "when it asked “which one?”", "warn"),
          tile(`${rejectedRight}`, "correct answers it still rejected", "the price of caution", "")),
        el("div", { class: "compare" },
          el("div", {}, el("strong", { text: "An agent that trusts its own confidence" }),
            el("p", { text: `Buys whenever it's at least ${pct(+rawCut)} sure: ${sm.one_tap_purchases[rawKey]} one-tap purchases, ` +
              `${sm.wrong_one_tap_purchases[rawKey]} of them wrong` + (naiveWrong.length ? ` (${naiveWrong.map(r =>
                `${r.condition.replace(/_/g, " ")} ${r.true_class} bought as ${a_(r.pred_class)} at ${pct(r.raw_confidence)}`).join("; ")}).` : ".") })),
          el("div", {}, el("strong", { text: "With the trust layer's gate" }),
            el("p", { text: `${sm.one_tap_purchases.trust_gate} one-tap purchases, ${sm.wrong_one_tap_purchases.trust_gate} wrong. ` +
              "Everything else asked the shopper first or asked for a better photo." }))),
        el("details", { open: S.photosOpen ? "" : null, ontoggle: e => { S.photosOpen = e.target.open; } },
          el("summary", { text: `All ${rows.length} photos` }),
          el("div", { class: "tscroll" }, el("table", { class: "compact" },
            el("thead", {}, el("tr", {}, ["Photo", "Really is", "Model said", "Own confidence", "Trust layer's estimate", "Decision", ""].map(h => el("th", { text: h })))),
            el("tbody", {}, rows.map(r => el("tr", {},
              el("td", { text: r.file.replace(/\.[^.]+$/, "").replace(/_/g, " ") }),
              el("td", { text: r.true_class || "(not in catalog)" }), el("td", { text: r.pred_class }),
              el("td", { text: pct(r.raw_confidence) }), el("td", { text: pct(r.p_correct) }), el("td", {}, decBadge(r.decision)),
              el("td", { text: verdict(r) })))))))));
    }

    const T = E.test, th = E.thresholds || {};
    if (T) {
      const dec = T.decisions_test || {};
      parts.push(el("div", { class: "eval-block" },
        el("h3", { text: `2 · Test photos the trust layer never saw (${T.photos.test} photos × 25 versions = ${T.test_rows.toLocaleString()} images)` }),
        dec.trust ? el("p", { class: "takeaway", text: `The one-tap TRUST tier covered ${pct(dec.trust.share)} of images, and ${pctErr(dec.trust.error)} ` +
          `of those were wrong (target: at most ${pct((th.target_error || {}).trust)}).` }) : null,
        el("p", { class: "sub", text: `Each photo appears clean and with 8 kinds of damage, none used to build the trust layer; ` +
          `TRUST needs an estimate of at least ${pct(th.tau_trust)}, and REJECT is below ${pct(th.tau_reject)}.` }),
        el("div", { class: "tiles" }, ["trust", "caution", "reject"].map(d => dec[d] ? tile(pct(dec[d].share), `of images got ${d.toUpperCase()}`,
          `${pctErr(dec[d].error)} of those were wrong`, d === "trust" ? "good" : d === "caution" ? "warn" : "crit") : null),
          tile(pct(T.test_accuracy), "model accuracy overall", "30 product types", "")),
        el("div", { class: "tscroll" }, el("table", { class: "compact" },
          el("thead", {}, el("tr", {}, ["Damage", "Images", "Actually right", "Model's own confidence", "Trust layer's estimate"].map(h => el("th", { text: h })))),
          el("tbody", {}, (T.by_severity_test || []).map(r => el("tr", {},
            el("td", { text: r.severity === 0 ? "none (clean)" : `severity ${r.severity}` }), el("td", { text: r.n.toLocaleString() }),
            el("td", {}, bar(r.accuracy, "var(--ink-2)")), el("td", {}, bar(r.raw_confidence, "var(--raw)")), el("td", {}, bar(r.p_correct, "var(--trust)"))))))),
        el("p", { class: "small", text: `Telling right from wrong answers (1 is perfect): model alone ${T.auroc.raw_confidence.toFixed(3)}, ` +
          `trust layer ${T.auroc.trust_layer.toFixed(3)}; gap between stated confidence and reality (0 is perfect): ` +
          `${T.ece.raw_confidence.toFixed(3)} vs ${T.ece.p_correct.toFixed(3)}.` }),
        el("p", { class: "tech", text: "AUROC · ECE · ImageNetV2 photos" })));
    }

    parts.push(el("div", { class: "eval-block" }, el("h3", { text: "3 · What this does and doesn't show" }),
      el("ul", { class: "notes" },
        el("li", { text: "The gate is the win here: one-tap purchases are held to a 1% error target, and it held on real photos." }),
        el("li", { text: "This vision model's own confidence is already close to reality on these photos, so the trust layer's estimate " +
          "doesn't beat it there. The overconfident model is the core project's CIFAR one (see Research results)." }),
        el("li", { text: "It's deliberately cautious: some correct answers get held back, and the shopper is asked instead." }),
        el("li", { text: "Demo scale: 95 test photos and 26 real ones. The numbers come from data/shop/evaluation.json and real_photo_eval.json." }))));

    const teaser = R ? `On our own phone photos: ${R.summary.trust_correct} one-tap purchases right, ` +
      `${R.summary.outside_catalog_trusted} unknown items trusted. Click to see the evidence.` : "Click to see the evidence.";
    return el("section", { class: "card eval", "aria-labelledby": "shop-eval-h" },
      el("details", { class: "eval-toggle", open: S.evalOpen ? "" : null, ontoggle: e => { S.evalOpen = e.target.open; } },
        el("summary", {}, el("span", { class: "eval-title" }, el("h2", { id: "shop-eval-h", text: "One-tap purchases stayed within the gate's 1% error target, on test photos and on real ones." }),
          el("span", { class: "small", text: teaser }))),
        el("div", { class: "eval-body" }, ...parts)));
  }

  function renderStory() {
    const st = S.samples.story || {};
    const steps = [["TRUST: one-tap", st.trust, 1, "var(--good)"], ["TRUST: big order", st.trust, "big", "var(--good)"],
                   ["CAUTION: confirm", st.caution, 1, "var(--warn)"], ["REJECT: retake", st.reject, 1, "var(--crit)"]];
    return el("div", { class: "chips" }, el("span", { class: "small", text: "Demo story:" }),
      steps.filter(x => x[1]).map(([t, v, q, color], n) => el("button", { class: "chip", type: "button",
        onclick: async () => {
          await select(v.base_image_id, v.photo_id);
          if (S.assist && S.assist.cart_allowed && S.assist.products.length) {
            const p = S.assist.products[0];
            const qty = q === "big" ? Math.max(2, Math.ceil((S.assist.checkout_policy.max_one_tap_total + 1) / p.price)) : 1;
            S.cart = { [p.product_id]: Math.min(20, qty) }; quote();
          }
        } }, el("span", { class: "dot", style: `background:${color}` }), `${n + 1} · ${t}`)));
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
        el("div", { class: "head" },
          el("div", {}, el("h2", { text: "Shopping assistant" }),
            el("p", { class: "sub", text: "The same trust layer, applied to shopping: before the AI recommends or buys anything, the layer decides what it's allowed to do." })),
          renderShopperBar()),
        renderStory()),
      renderSource(),
      el("div", { class: "main" }, renderPhoto(), renderTrust(), renderChat()),
      el("div", { class: "shop-bottom" }, renderProducts(), renderCheckout()),
      renderEvaluation(),
      el("p", { class: "fine", text: "Rules are enforced in code, not by the AI. Products are real models at approximate prices " +
        "(not affiliated); example shoppers are fictional; checkout is a mock and nothing is charged." }));
  }

  function install() {
    document.head.append(el("style", { text: CSS }));
    const nav = document.querySelector("nav.tabs"), main = document.querySelector("main");
    const btn = el("button", { role: "tab", id: "tab-shop", "aria-controls": "shop", "aria-selected": "false", text: "Shopping assistant" });
    root = el("section", { id: "shop", role: "tabpanel", "aria-labelledby": "tab-shop", hidden: "" });
    nav.append(btn); main.append(root);
    const others = ["demo", "research"];
    btn.addEventListener("click", () => {
      for (const t of others) { document.getElementById(t).hidden = true; document.getElementById(`tab-${t}`).setAttribute("aria-selected", "false"); }
      root.hidden = false; btn.setAttribute("aria-selected", "true"); render();
      document.body.classList.add("shop-active");   // hides the core page's CIFAR results line (see CSS)
    });
    if (WANT_SHOP) {  // deep link /#shop: wait for the core page to finish its own tab setup, then open this tab
      const t0 = Date.now();
      (function go() {
        if (location.hash === "#shop" && Date.now() - t0 < 4000) return setTimeout(go, 100);
        btn.click(); history.replaceState(null, "", "#shop");
      })();
    }
    for (const t of others) document.getElementById(`tab-${t}`).addEventListener("click", () => {
      root.hidden = true; btn.setAttribute("aria-selected", "false");
      document.body.classList.remove("shop-active");
      if (S.stream) closeCamera();   // don't leave the camera light on in another tab
    });
  }

  (async () => {
    const [s, p] = await Promise.all([api("/api/shop/samples"), api("/api/shop/profiles")]).catch(() => [{}, {}]);
    if (!s.ok || !p.ok) return;   // shop not installed: leave the page untouched
    S.samples = s.j; S.meta = p.j; S.me = store.get();
    api("/api/shop/evaluation").then(r => { if (r.ok) { S.evaluation = r.j; render(); } }).catch(() => {});
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
