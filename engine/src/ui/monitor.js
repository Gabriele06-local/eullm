// EuLLM Monitor — the requests the engine is answering and has answered, and
// the machine it runs on. Reads /api/monitor every second and /api/ps every
// few, only while the page is open; everything shown lives in the engine's
// memory and in this tab's, and nothing is stored.

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const els = {
    btn: $("monitor-btn"),
    view: $("monitor"),
    live: $("mon-live"),
    models: $("mon-models"),
    tiles: $("mon-tiles"),
    gpus: $("mon-gpus"),
    requests: $("mon-requests"),
    messages: $("messages"),
    form: $("chat-form"),
  };
  if (!els.btn || !els.view) return;

  // The key app.js keeps for this tab (see its API key comment).
  const auth = () => {
    let key = "";
    try { key = sessionStorage.getItem("eullm.apiKey") || ""; } catch {}
    return key ? { headers: { "X-Api-Key": key } } : {};
  };

  const POINTS = 120;
  // What the tab itself remembers between polls: the engine's speed at each
  // second, which no single response carries.
  const decodeSeries = [];
  let timer = null;
  let psTimer = null;
  let running = [];

  // ── Formatting ────────────────────────────────────────────────────────
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const gib = (b) => (b == null ? "–" : (b / 2 ** 30).toFixed(b >= 10 * 2 ** 30 ? 1 : 2));
  const rateBytes = (bps) => {
    if (bps == null) return "–";
    if (bps >= 1e9) return `${(bps / 1e9).toFixed(2)} GB/s`;
    if (bps >= 1e6) return `${(bps / 1e6).toFixed(1)} MB/s`;
    if (bps >= 1e3) return `${(bps / 1e3).toFixed(0)} kB/s`;
    return `${bps} B/s`;
  };
  const num = (v, d = 1) => (v == null ? "–" : Number(v).toFixed(d));
  const ms = (v) => (v == null ? "–" : v >= 10000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`);
  const clock = (t) => new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const pct = (used, total) => (used != null && total ? Math.min(100, (used / total) * 100) : null);

  // ── Sparkline: one series, one axis from 0, the latest value labelled ──
  function spark(points, max) {
    const w = 160, h = 36;
    const vals = points.map((p) => p.v);
    const top = Math.max(max ?? 0, ...vals.filter((v) => v != null), 1);
    if (points.length < 2) return `<svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true"></svg>`;
    const step = w / (POINTS - 1);
    const x0 = w - (points.length - 1) * step;
    const xy = points.map((p, i) => [x0 + i * step, h - 2 - ((p.v ?? 0) / top) * (h - 4)]);
    const line = xy.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
    const area = `${x0.toFixed(1)},${h} ${line} ${w},${h}`;
    const data = esc(JSON.stringify(points.map((p) => [p.t, p.v])));
    return `<svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" data-points="${data}" data-top="${top}">
      <polygon class="spark-area" points="${area}"/>
      <polyline class="spark-line" points="${line}"/>
      <line class="spark-cross" x1="0" x2="0" y1="0" y2="${h}" hidden/>
    </svg>`;
  }

  // Hover on a sparkline: the second and the value under the pointer, in
  // the tile's own caption.
  els.view.addEventListener("pointermove", (e) => {
    const svg = e.target.closest?.("svg.spark");
    els.view.querySelectorAll(".spark-cross").forEach((c) => { if (!svg || !svg.contains(c)) c.setAttribute("hidden", ""); });
    if (!svg || !svg.dataset.points) return;
    const pts = JSON.parse(svg.dataset.points);
    const r = svg.getBoundingClientRect();
    const w = 160, step = w / (POINTS - 1);
    const x0 = w - (pts.length - 1) * step;
    const x = ((e.clientX - r.left) / r.width) * w;
    const i = Math.max(0, Math.min(pts.length - 1, Math.round((x - x0) / step)));
    const cross = svg.querySelector(".spark-cross");
    const cx = x0 + i * step;
    cross.setAttribute("x1", cx); cross.setAttribute("x2", cx); cross.removeAttribute("hidden");
    const tile = svg.closest(".mon-tile");
    const cap = tile?.querySelector(".mon-hover");
    if (cap) cap.textContent = `${clock(pts[i][0])} · ${tile.dataset.fmt === "bytes" ? rateBytes(pts[i][1]) : num(pts[i][1], tile.dataset.fmt === "int" ? 0 : 1)}${tile.dataset.unit ? " " + tile.dataset.unit : ""}`;
  });
  els.view.addEventListener("pointerleave", () => {
    els.view.querySelectorAll(".spark-cross").forEach((c) => c.setAttribute("hidden", ""));
    els.view.querySelectorAll(".mon-hover").forEach((c) => (c.textContent = ""));
  });

  function tile({ label, value, unit = "", sub = "", series = null, max = null, bar = null, fmt = "" }) {
    const meter = bar == null ? "" : `<div class="mon-bar" role="meter" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${bar.toFixed(0)}"><span style="width:${bar.toFixed(1)}%"></span></div>`;
    return `<div class="mon-tile" data-unit="${esc(unit)}" data-fmt="${fmt}">
      <div class="mon-label">${esc(label)}</div>
      <div class="mon-value">${esc(value)}<span class="mon-unit">${esc(unit)}</span></div>
      ${meter}
      ${series ? spark(series, max) : ""}
      <div class="mon-sub">${esc(sub)}</div>
      <div class="mon-hover" aria-live="off"></div>
    </div>`;
  }

  // ── Render ───────────────────────────────────────────────────────────
  function renderModels(active) {
    if (!running.length) {
      els.models.innerHTML = `<div class="mon-model idle"><span class="mon-dot"></span><strong>No model loaded</strong><span class="mon-dim">it loads on the first request</span></div>`;
      return;
    }
    els.models.innerHTML = running.map((m) => {
      const mine = active.filter((r) => r.model === m.name);
      const state = mine.some((r) => r.phase === "generating") ? "generating" : mine.length ? "reading" : "idle";
      const e = m.eullm || {};
      const facts = [
        e.slot && e.slot !== "generation" ? e.slot : null,
        m.context_length ? `${m.context_length.toLocaleString()} ctx per request` : null,
        e.batch_size > 1 ? `${e.batch_size} slots` : null,
        m.size_vram ? `${gib(m.size_vram)} GiB in VRAM` : "on CPU",
        !m.size_vram ? null : e.gpu_layers === -1 ? "all layers on GPU" : e.gpu_layers > 0 ? `${e.gpu_layers} layers on GPU` : null,
        mine.length ? `${mine.length} running` : null,
      ].filter(Boolean);
      return `<div class="mon-model ${state}"><span class="mon-dot" title="${state}"></span><strong>${esc(m.name)}</strong><span class="mon-state">${state}</span><span class="mon-dim">${esc(facts.join(" · "))}</span></div>`;
    }).join("");
  }

  function renderTiles(active, history, system) {
    const now = system[system.length - 1] || {};
    const generating = active.filter((r) => r.phase === "generating");
    const decodeNow = generating.reduce((a, r) => a + (r.decode_tps || 0), 0);
    decodeSeries.push({ t: Date.now(), v: generating.length ? decodeNow : 0 });
    while (decodeSeries.length > POINTS) decodeSeries.shift();
    const last = history.find((r) => r.phase === "done");
    const series = (f) => system.map((s) => ({ t: s.t_ms, v: f(s) }));
    const tiles = [
      tile({
        label: "Writing", value: generating.length ? num(decodeNow) : last ? num(last.decode_tps) : "–", unit: "tok/s",
        sub: generating.length ? `${generating.length} answer${generating.length > 1 ? "s" : ""} being written` : last ? "last answer" : "no answer yet",
        series: decodeSeries, max: 0,
      }),
      tile({
        label: "Reading the prompt", value: last ? num(last.prefill_tps, 0) : "–", unit: "tok/s",
        sub: last ? `${(last.prompt_tokens ?? 0).toLocaleString()} token${last.prompt_tokens === 1 ? "" : "s"}, first token in ${ms(last.first_token_ms)}` : "no answer yet", fmt: "int",
      }),
      tile({
        label: "CPU", value: num(now.cpu_pct, 0), unit: "%", sub: now.cores ? `${now.cores} cores` : "",
        series: series((s) => s.cpu_pct), max: 100, fmt: "int",
      }),
      tile({
        label: "Memory", value: gib(now.ram_used), unit: `/ ${gib(now.ram_total)} GiB`,
        sub: now.process_ram ? `engine ${gib(now.process_ram)} GiB` : "", bar: pct(now.ram_used, now.ram_total),
      }),
      tile({
        label: "Disk reads", value: rateBytes(now.disk_read_bps), sub: "the engine: model loading, mmap'd weights",
        series: series((s) => s.disk_read_bps), max: 0, fmt: "bytes",
      }),
    ];
    els.tiles.innerHTML = tiles.join("");
  }

  function renderGpus(system) {
    const now = system[system.length - 1];
    if (!now || !now.gpus?.length) {
      els.gpus.innerHTML = system.length ? `<div class="mon-empty">No GPU found: the engine runs on the CPU.</div>` : "";
      return;
    }
    els.gpus.innerHTML = now.gpus.map((g, i) => {
      const series = (f) => system.map((s) => ({ t: s.t_ms, v: s.gpus?.[i] ? f(s.gpus[i]) : null }));
      const tiles = [];
      if (g.util_pct != null) tiles.push(tile({ label: "Load", value: num(g.util_pct, 0), unit: "%", series: series((x) => x.util_pct), max: 100, fmt: "int" }));
      if (g.vram_total != null) tiles.push(tile({ label: "VRAM", value: gib(g.vram_used), unit: `/ ${gib(g.vram_total)} GiB`, bar: pct(g.vram_used, g.vram_total), sub: `${num(pct(g.vram_used, g.vram_total), 0)}% used` }));
      if (g.temp_c != null) tiles.push(tile({ label: "Temperature", value: num(g.temp_c, 0), unit: "°C", series: series((x) => x.temp_c), max: 100, fmt: "int" }));
      if (g.power_w != null) tiles.push(tile({ label: "Power", value: num(g.power_w, 0), unit: "W", sub: g.power_limit_w ? `limit ${num(g.power_limit_w, 0)} W` : "", series: series((x) => x.power_w), max: g.power_limit_w || 0, fmt: "int" }));
      if (g.pcie_rx_bps != null) tiles.push(tile({ label: "PCIe to GPU", value: rateBytes(g.pcie_rx_bps), sub: `from GPU ${rateBytes(g.pcie_tx_bps)}`, series: series((x) => x.pcie_rx_bps), max: 0, fmt: "bytes" }));
      const source = { nvml: "NVIDIA driver", amdgpu: "amdgpu driver", ggml: "llama.cpp device" }[g.source] || g.source;
      return `<div class="mon-gpu"><div class="mon-gpu-head"><strong>${esc(g.name)}</strong><span class="mon-dim">GPU ${i} · ${esc(source)}</span></div><div class="mon-tiles">${tiles.join("")}</div></div>`;
    }).join("");
  }

  function renderRequests(active, history) {
    const rows = [...active.slice().reverse(), ...history];
    if (!rows.length) {
      els.requests.innerHTML = `<tr><td colspan="12" class="mon-empty">No request since the engine started.</td></tr>`;
      return;
    }
    els.requests.innerHTML = rows.map((r) => {
      const drafts = r.drafts ? `${Math.round((r.drafts[1] / Math.max(1, r.drafts[0])) * 100)}% of ${r.drafts[0]}` : "–";
      const total = r.phase === "reading" || r.phase === "generating" ? "…" : ms(r.duration_ms);
      return `<tr class="phase-${r.phase}">
        <td>${clock(r.started_ms)}</td><td class="mon-model-cell">${esc(r.model)}</td><td>${esc(r.endpoint)}</td>
        <td><span class="mon-phase">${esc(r.phase)}</span></td>
        <td class="num">${r.prompt_tokens != null ? r.prompt_tokens.toLocaleString() : "–"}</td>
        <td class="num">${r.output_tokens.toLocaleString()}</td>
        <td class="num">${num(r.prefill_tps, 0)}</td><td class="num">${num(r.decode_tps)}</td>
        <td class="num">${ms(r.first_token_ms)}</td><td class="num">${total}</td>
        <td class="num">${drafts}</td><td class="mon-ended" title="${esc(r.ended)}">${esc(r.ended ?? "")}</td>
      </tr>`;
    }).join("");
  }

  // ── Polling, only while the page is open and the tab visible ──────────
  async function poll() {
    try {
      const r = await fetch("/api/monitor", auth());
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = await r.json();
      renderModels(d.active);
      renderTiles(d.active, d.history, d.system);
      renderGpus(d.system);
      renderRequests(d.active, d.history);
      els.live.classList.remove("off");
      els.live.textContent = "live";
    } catch (e) {
      els.live.classList.add("off");
      els.live.textContent = `offline (${e.message})`;
    }
  }

  async function pollModels() {
    try {
      const r = await fetch("/api/ps", auth());
      if (r.ok) running = (await r.json()).models || [];
    } catch {}
  }

  function start() {
    if (timer) return;
    pollModels().then(poll);
    timer = setInterval(poll, 1000);
    psTimer = setInterval(pollModels, 3000);
  }

  function stop() {
    clearInterval(timer); clearInterval(psTimer);
    timer = psTimer = null;
  }

  function show(on) {
    els.view.hidden = !on;
    els.messages.hidden = on;
    els.form.hidden = on;
    els.btn.setAttribute("aria-pressed", String(on));
    els.btn.classList.toggle("active", on);
    if (on) start(); else stop();
  }

  els.btn.addEventListener("click", () => show(els.view.hidden));
  document.addEventListener("visibilitychange", () => {
    if (els.view.hidden) return;
    if (document.hidden) stop(); else start();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !els.view.hidden && !document.querySelector("dialog[open]")) show(false);
  });
})();
