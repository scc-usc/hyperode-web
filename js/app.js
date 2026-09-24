/* AXIOM — frontend inference app: ODE + random params -> trajectory quantiles. */
(function () {
  "use strict";
  const MODELS = window.AXIOM_MODELS;
  const $ = (id) => document.getElementById(id);

  // `scalable: true` = a compartmental system that can be expanded into k coupled
  // sub-populations (meta-population) via the Scale-k control (ring mobility on every state).
  // `signed: true` = states can go negative -> use a Diverse (signed) model, not the paper trio.
  const PRESETS = {
    // ---- epidemic / compartmental (scalable) ----
    SIR: { tMax: 8, steps: 40, unc: 0.3, scalable: true,
      ode: "dS/dt = -beta*S*I\ndI/dt = beta*S*I - gamma*I\ndR/dt = gamma*I",
      params: { beta: 1.2, gamma: 0.3 }, x0: { S: 0.99, I: 0.01, R: 0.0 } },
    SEIR: { tMax: 8, steps: 40, unc: 0.3, scalable: true,
      ode: "dS/dt = -beta*S*I\ndE/dt = beta*S*I - sigma*E\ndI/dt = sigma*E - gamma*I\ndR/dt = gamma*I",
      params: { beta: 1.5, sigma: 0.5, gamma: 0.3 }, x0: { S: 0.98, E: 0.0, I: 0.02, R: 0.0 } },
    SIS: { tMax: 8, steps: 40, unc: 0.3, scalable: true,
      ode: "dS/dt = -beta*S*I + gamma*I\ndI/dt = beta*S*I - gamma*I",
      params: { beta: 1.4, gamma: 0.4 }, x0: { S: 0.9, I: 0.1 } },
    SIRS: { tMax: 12, steps: 50, unc: 0.3, scalable: true,
      ode: "dS/dt = -beta*S*I + omega*R\ndI/dt = beta*S*I - gamma*I\ndR/dt = gamma*I - omega*R",
      params: { beta: 1.3, gamma: 0.3, omega: 0.1 }, x0: { S: 0.99, I: 0.01, R: 0.0 } },
    SEIRS: { tMax: 16, steps: 60, unc: 0.3, scalable: true,
      ode: "dS/dt = -beta*S*I + omega*R\ndE/dt = beta*S*I - sigma*E\ndI/dt = sigma*E - gamma*I\ndR/dt = gamma*I - omega*R",
      params: { beta: 1.6, sigma: 0.5, gamma: 0.3, omega: 0.1 }, x0: { S: 0.98, E: 0.0, I: 0.02, R: 0.0 } },
    SIRD: { tMax: 10, steps: 50, unc: 0.3, scalable: true,
      ode: "dS/dt = -beta*S*I\ndI/dt = beta*S*I - gamma*I - mu*I\ndR/dt = gamma*I\ndD/dt = mu*I",
      params: { beta: 1.4, gamma: 0.3, mu: 0.05 }, x0: { S: 0.99, I: 0.01, R: 0.0, D: 0.0 } },
    SI: { tMax: 8, steps: 40, unc: 0.3, scalable: true,
      ode: "dS/dt = -beta*S*I\ndI/dt = beta*S*I",
      params: { beta: 1.0 }, x0: { S: 0.99, I: 0.01 } },
    // ---- non-mass-action incidence (auto-swapped nonlinearity; best with a Diverse model) ----
    "SEIR · sublinear (S·I^0.5)": { tMax: 8, steps: 40, unc: 0.3, scalable: true,
      ode: "dS/dt = -beta*S*I^0.5\ndE/dt = beta*S*I^0.5 - sigma*E\ndI/dt = sigma*E - gamma*I\ndR/dt = gamma*I",
      params: { beta: 1.6, sigma: 0.5, gamma: 0.3 }, x0: { S: 0.98, E: 0.0, I: 0.02, R: 0.0 } },
    "SEIR · saturating (S·I/(1+8·I))": { tMax: 8, steps: 40, unc: 0.3, scalable: true,
      ode: "dS/dt = -beta*S*I/(1+8*I)\ndE/dt = beta*S*I/(1+8*I) - sigma*E\ndI/dt = sigma*E - gamma*I\ndR/dt = gamma*I",
      params: { beta: 2.5, sigma: 0.5, gamma: 0.3 }, x0: { S: 0.98, E: 0.0, I: 0.02, R: 0.0 } },
    "SIR · Hill (S·I^3/(K^3+I^3))": { tMax: 10, steps: 50, unc: 0.3,
      ode: "dS/dt = -beta*S*I^3/(0.04+I^3)\ndI/dt = beta*S*I^3/(0.04+I^3) - gamma*I\ndR/dt = gamma*I",
      params: { beta: 2.0, gamma: 0.3 }, x0: { S: 0.99, I: 0.05, R: 0.0 } },
    // ---- ecological / population ----
    "Lotka–Volterra": { tMax: 12, steps: 60, unc: 0.25,
      ode: "dX/dt = a*X - b*X*Y\ndY/dt = d*X*Y - c*Y",
      params: { a: 0.6, b: 0.5, c: 0.4, d: 0.5 }, x0: { X: 0.5, Y: 0.3 } },
    "Competitive Lotka–Volterra": { tMax: 20, steps: 60, unc: 0.2,
      ode: "dX/dt = r1*X*(1 - X - a12*Y)\ndY/dt = r2*Y*(1 - Y - a21*X)",
      params: { r1: 0.7, r2: 0.6, a12: 0.6, a21: 0.7 }, x0: { X: 0.3, Y: 0.3 } },
    "Rosenzweig–MacArthur (Holling-II)": { tMax: 30, steps: 80, unc: 0.2,
      ode: "dX/dt = r*X*(1 - X) - a*X*Y/(1 + h*X)\ndY/dt = e*a*X*Y/(1 + h*X) - m*Y",
      params: { r: 0.8, a: 1.2, h: 1.5, e: 0.6, m: 0.3 }, x0: { X: 0.4, Y: 0.2 } },
    "Logistic growth": { tMax: 12, steps: 40, unc: 0.3,
      ode: "dN/dt = r*N*(1 - N)",
      params: { r: 0.8 }, x0: { N: 0.05 } },
    "Gompertz growth": { tMax: 12, steps: 40, unc: 0.3,
      ode: "dN/dt = r*N*log(1/N)",
      params: { r: 0.6 }, x0: { N: 0.05 } },
    // ---- chemical / enzyme ----
    "Michaelis–Menten": { tMax: 12, steps: 50, unc: 0.25,
      ode: "dS/dt = -V*S/(Km + S)\ndP/dt = V*S/(Km + S)",
      params: { V: 0.5, Km: 0.3 }, x0: { S: 1.0, P: 0.0 } },
    "Chemostat (Monod)": { tMax: 30, steps: 80, unc: 0.2,
      ode: "dS/dt = D*(S0 - S) - mu*S/(Ks + S)*X\ndX/dt = mu*S/(Ks + S)*X - D*X",
      params: { D: 0.2, S0: 1.0, mu: 0.6, Ks: 0.2 }, x0: { S: 0.5, X: 0.1 } },
    "Brusselator": { tMax: 20, steps: 80, unc: 0.15,
      ode: "dX/dt = a - (b+1)*X + X^2*Y\ndY/dt = b*X - X^2*Y",
      params: { a: 1.0, b: 2.2 }, x0: { X: 1.0, Y: 1.0 } },
    // ---- signed / oscillatory (states go negative -> use a Diverse model) ----
    "Damped pendulum": { tMax: 16, steps: 80, unc: 0.15, signed: true,
      ode: "dTheta/dt = W\ndW/dt = -sin(Theta) - c*W",
      params: { c: 0.3 }, x0: { Theta: 1.0, W: 0.0 } },
    "Van der Pol": { tMax: 20, steps: 100, unc: 0.15, signed: true,
      ode: "dX/dt = Y\ndY/dt = mu*(1 - X^2)*Y - X",
      params: { mu: 1.0 }, x0: { X: 1.0, Y: 0.0 } },
    "FitzHugh–Nagumo": { tMax: 40, steps: 100, unc: 0.15, signed: true,
      ode: "dV/dt = V - V^3/3 - W + I0\ndW/dt = eps*(V + a - b*W)",
      params: { I0: 0.5, eps: 0.08, a: 0.7, b: 0.8 }, x0: { V: -1.0, W: 1.0 } },
    "Gene toggle (Hill)": { tMax: 20, steps: 60, unc: 0.2,
      ode: "dX/dt = 1/(1 + Y^3) - X\ndY/dt = 1/(1 + X^3) - Y",
      params: {}, x0: { X: 0.2, Y: 0.8 } },
  };

  // ---- meta-population scale-out: expand a base compartmental preset into k groups ----
  // Each state X -> X1..Xk; within-group dynamics are replicated and every state diffuses to its
  // ring neighbours (mobility m): + m*(X_{g-1} + X_{g+1} - 2*X_g). k=1 reproduces the base.
  function expandScale(base, k, mob) {
    const { states, rhs } = parseODE(base.ode);
    // rename longest state names first so e.g. "SI" isn't corrupted by renaming "S"
    const order = states.slice().sort((a, b) => b.length - a.length);
    const lines = [];
    for (let g = 1; g <= k; g++) {
      const gm = g === 1 ? k : g - 1, gp = g === k ? 1 : g + 1;   // ring neighbours
      states.forEach((s, i) => {
        let r = rhs[i];
        for (const st of order) r = r.replace(new RegExp(`\\b${st}\\b`, "g"), `${st}\u0000${g}`);
        r = r.replace(/\u0000/g, "");                             // finalize X<g> tokens
        if (k > 1) r += ` + m*(${s}${gm} + ${s}${gp} - 2*${s}${g})`;
        lines.push(`d${s}${g}/dt = ${r}`);
      });
    }
    const params = Object.assign({}, base.params);
    if (k > 1) params.m = mob;
    const x0 = {};
    for (let g = 1; g <= k; g++) states.forEach((s) => { x0[`${s}${g}`] = base.x0[s]; });
    return { ode: lines.join("\n"), params, x0, tMax: base.tMax, steps: base.steps, unc: base.unc };
  }

  // A ready-made prompt: paste into any LLM, describe your system in words, get back a spec that
  // the Import box below loads in one click. Keeps users from hand-writing calculus.
  const LLM_PROMPT =
`Convert the dynamical system I describe into the JSON spec below for the AXIOM forward-UQ ODE tool.
Output ONLY the JSON (no prose, no code fences).

RULES
- "ode": one equation per line as "dX/dt = <expression>", lines joined with \\n. The X's on the
  left are the STATE variables; every other symbol is a PARAMETER.
- Systems are AUTONOMOUS: do NOT use a time variable t.
- Operators + - * / ^ and parentheses. Functions: sin, cos, tan, tanh, exp, log, sqrt, abs.
  Powers may be integer (X^2 self-interaction) or fractional (I^0.5). States may appear in
  nonlinear functions or denominators (e.g. saturating S*I/(1+8*I), Hill S*I^3/(K^3+I^3)).
- Keep state magnitudes ~order 1. For compartment fractions, initial conditions should sum to ~1.
- Set "signed": true ONLY if a state can go negative (mechanical/oscillatory systems, e.g. a
  pendulum angle or a neuron voltage); otherwise false.
- Each parameter: central "value", relative "std" (0 = fixed, ~0.3 = typical uncertainty),
  and "dist" in {"lognormal","normal","uniform"}. "tMax" = horizon, "steps" = output points (~40).

OUTPUT EXACTLY THIS SHAPE (example is SIR):
{
  "ode": "dS/dt = -beta*S*I\\ndI/dt = beta*S*I - gamma*I\\ndR/dt = gamma*I",
  "params": { "beta": {"value": 1.2, "std": 0.3, "dist": "lognormal"},
              "gamma": {"value": 0.3, "std": 0.3, "dist": "lognormal"} },
  "x0": { "S": 0.99, "I": 0.01, "R": 0.0 },
  "tMax": 8, "steps": 40, "signed": false
}

MY SYSTEM: `;

  // in-memory param/x0 values, preserved across re-parse by name
  let paramVals = {}, paramFam = {}, paramStd = {}, x0Vals = {};
  let defaultStd = 0.3;   // seed rel-std for newly detected params (set per preset)
  let LAST = null;   // last surrogate run (for MC overlay without recompute)

  function parseODE(text) {
    const states = [], rhs = [];
    for (const raw of text.split("\n")) {
      const line = raw.trim();
      if (!line || line.startsWith("#")) continue;
      const m = line.match(/^d\s*([A-Za-z_]\w*)\s*\/\s*dt\s*=\s*(.+)$/i);
      if (!m) throw new Error(`can't parse line: "${line}"  (use  dX/dt = ...)`);
      states.push(m[1]); rhs.push(m[2]);
    }
    if (!states.length) throw new Error("no equations found");
    return { states, rhs };
  }

  // symbols referenced in RHS that aren't states (and aren't function names) = parameters.
  // Uses the AST parser so custom non-polynomial terms (sin/exp/sqrt, non-integer powers,
  // state-dependent denominators) are handled -- their function names are NOT params.
  const FN_NAMES = new Set(["sin", "cos", "tan", "tanh", "exp", "log", "sqrt", "abs"]);
  function detectParams(states, rhs) {
    const names = new Set();
    rhs.forEach((r, i) => {
      let ast;
      try { ast = AXIOMEncode.parseAST(AXIOMEncode.tokenize(r)); }
      catch (e) { throw new Error(`d${states[i]}/dt: ${e.message}`); }
      const walk = (a) => {
        if (!a) return;
        if (a.t === "sym") { if (!states.includes(a.name) && !FN_NAMES.has(a.name)) names.add(a.name); }
        else if (a.t === "neg") walk(a.x);
        else if (a.t === "pow") walk(a.base);
        else if (a.t === "fn") walk(a.arg);      // function NAME is not a parameter
        else if (a.t === "bin") { walk(a.l); walk(a.r); }
      };
      walk(ast);
    });
    return [...names].sort();
  }

  function syncTables() {
    let parsed;
    try { parsed = parseODE($("ode").value); }
    catch (e) { showError(e.message); return null; }
    hideError();
    const { states, rhs } = parsed;
    let params;
    try { params = detectParams(states, rhs); }
    catch (e) { showError(e.message); return null; }

    // params table
    const pt = $("paramTable"); pt.innerHTML = "";
    if (params.length) {
      const head = document.createElement("div"); head.className = "prow phead";
      head.innerHTML = `<label></label><span>value</span><span>±std</span><span>dist</span>`;
      pt.appendChild(head);
    }
    params.forEach((nm) => {
      if (paramVals[nm] === undefined) paramVals[nm] = 1.0;
      if (paramFam[nm] === undefined) paramFam[nm] = "lognormal";
      if (paramStd[nm] === undefined) paramStd[nm] = defaultStd;
      const row = document.createElement("div"); row.className = "prow";
      row.innerHTML =
        `<label>${nm}</label>` +
        `<input type="number" step="0.05" value="${paramVals[nm]}" data-p="${nm}" class="pval">` +
        `<input type="number" step="0.05" min="0" value="${paramStd[nm]}" data-p="${nm}" class="pstd" title="relative std (0 = fixed). Set fractions like efficacy to 0 or keep them in [0,1].">` +
        `<select data-p="${nm}" class="pfam">` +
          ["lognormal", "normal", "uniform"].map((f) =>
            `<option ${paramFam[nm] === f ? "selected" : ""}>${f}</option>`).join("") +
        `</select>`;
      pt.appendChild(row);
    });
    // x0 table
    const xt = $("x0Table"); xt.innerHTML = "";
    states.forEach((s) => {
      if (x0Vals[s] === undefined) x0Vals[s] = 0.0;
      const row = document.createElement("div"); row.className = "prow";
      row.innerHTML = `<label>${s}(0)</label>` +
        `<input type="number" step="0.01" value="${x0Vals[s]}" data-x="${s}" class="xval">`;
      xt.appendChild(row);
    });
    pt.querySelectorAll(".pval").forEach((el) =>
      el.addEventListener("change", () => { paramVals[el.dataset.p] = parseFloat(el.value); }));
    pt.querySelectorAll(".pstd").forEach((el) =>
      el.addEventListener("change", () => { paramStd[el.dataset.p] = Math.max(0, parseFloat(el.value) || 0); }));
    pt.querySelectorAll(".pfam").forEach((el) =>
      el.addEventListener("change", () => { paramFam[el.dataset.p] = el.value; }));
    xt.querySelectorAll(".xval").forEach((el) =>
      el.addEventListener("change", () => { x0Vals[el.dataset.x] = parseFloat(el.value); }));
    return { states, rhs, params };
  }

  function currentK() { const k = parseInt($("scaleK").value, 10); return k > 1 ? k : 1; }

  function loadPreset(name) {
    const base = PRESETS[name];
    if (!base) return;
    $("scaleK").disabled = !base.scalable;                 // k only applies to compartmental presets
    // signed presets (oscillators/mechanics) need a signed-capable model, else output is clipped/wrong
    if (base.signed) {
      const cur = MODELS.models[$("modelSize").value];
      if (!cur || !cur.signed) $("modelSize").value = "diverse";
    }
    const k = base.scalable ? currentK() : 1;
    const p = (base.scalable && k > 1) ? expandScale(base, k, 0.1) : base;
    $("ode").value = p.ode;
    $("tMax").value = p.tMax; $("steps").value = p.steps; defaultStd = p.unc ?? 0.3;
    paramVals = Object.assign({}, p.params); paramFam = {}; paramStd = {}; x0Vals = Object.assign({}, p.x0);
    syncTables();
    saveState();
  }

  // Load a spec the user pasted: either the JSON produced by the LLM prompt (populates equations,
  // parameters+uncertainty, ICs, horizon at once) or raw "dX/dt = ..." lines (just the equations).
  function importSpec() {
    const raw = $("importJson").value.trim();
    if (!raw) { showError("Paste the JSON (or equations) the LLM produced first."); return; }
    let spec = null;
    try { spec = JSON.parse(raw); } catch (e) { /* not JSON -> treat as raw ODE text below */ }
    try {
      if (spec && typeof spec === "object") {
        if (typeof spec.ode !== "string") throw new Error("JSON has no \"ode\" string");
        $("ode").value = spec.ode;
        if (spec.tMax != null) $("tMax").value = spec.tMax;
        if (spec.steps != null) $("steps").value = spec.steps;
        paramVals = {}; paramStd = {}; paramFam = {};
        for (const nm in (spec.params || {})) {
          const p = spec.params[nm];
          paramVals[nm] = (typeof p === "number") ? p : (p.value != null ? p.value : 1.0);
          paramStd[nm] = (p && p.std != null) ? p.std : defaultStd;
          paramFam[nm] = (p && p.dist) ? p.dist : "lognormal";
        }
        x0Vals = Object.assign({}, spec.x0 || {});
        if (spec.signed) {                                 // signed system -> needs a signed model
          const cur = MODELS.models[$("modelSize").value];
          if (!cur || !cur.signed) $("modelSize").value = "diverse";
        }
      } else {
        if (!/d\s*[A-Za-z_]\w*\s*\/\s*dt/i.test(raw)) throw new Error("not JSON and no dX/dt equations found");
        $("ode").value = raw;                              // raw equations: keep existing params/ICs by name
      }
      hideError();
      if (!syncTables()) return;                           // shows a parse error if the ODE is malformed
      saveState();
      // Loading a spec only POPULATES the controls -- it does not infer. The user reviews the
      // parsed equations / parameters / ICs and presses Run inference themselves.
      markStale("Spec loaded — review the parameters, then press Run inference.");
      flashBtn("importBtn", "✓ loaded — press Run inference");
    } catch (e) { showError("Import failed: " + e.message); }
  }

  function showError(msg) { const e = $("err"); e.textContent = "⚠ " + msg; e.style.display = "block"; }
  function hideError() { $("err").style.display = "none"; }

  // Results no longer match the controls (a spec was loaded but not run yet): dim the plots and
  // say so, so a stale band is never mistaken for the newly loaded system. Cleared by run().
  function markStale(msg) {
    const r = document.querySelector(".results");
    if (r) r.classList.add("stale");
    const n = $("staleNote");
    if (n) { n.textContent = msg; n.style.display = "block"; }
  }
  function clearStale() {
    const r = document.querySelector(".results");
    if (r) r.classList.remove("stale");
    const n = $("staleNote");
    if (n) n.style.display = "none";
  }

  // transient confirmation on a button, then restore its label
  function flashBtn(id, text, ms) {
    const b = $(id); if (!b) return;
    if (b._flash) { clearTimeout(b._flash); } else { b._label = b.textContent; }
    b.textContent = text;
    b._flash = setTimeout(() => { b.textContent = b._label; b._flash = null; }, ms || 2200);
  }

  // ---- draggable sidebar width ----------------------------------------------
  const SIDE_KEY = "axiom.sideW.v1", SIDE_MIN = 260, SIDE_DEFAULT = 340;
  function sideMax() { return Math.max(SIDE_MIN + 60, Math.min(760, window.innerWidth - 380)); }
  function setSideWidth(px, save) {
    const w = Math.round(Math.max(SIDE_MIN, Math.min(sideMax(), px)));
    document.documentElement.style.setProperty("--side-w", w + "px");
    if (save) { try { localStorage.setItem(SIDE_KEY, String(w)); } catch (e) {} }
    return w;
  }
  function initResizer() {
    const g = $("gutter"); if (!g) return;
    try { const s = parseFloat(localStorage.getItem(SIDE_KEY)); if (isFinite(s)) setSideWidth(s, false); } catch (e) {}
    let startX = 0, startW = 0, curW = SIDE_DEFAULT, dragging = false, raf = 0;
    const redraw = () => { raf = 0; if (LAST) renderPlots(LAST.mc ? LAST.mc.pred : null); };
    const onMove = (e) => {
      if (!dragging) return;
      curW = setSideWidth(startW + (e.clientX - startX), false);   // save on release, not every move
      if (!raf) raf = requestAnimationFrame(redraw);
    };
    const onUp = () => {
      if (!dragging) return;
      dragging = false;
      try { localStorage.setItem(SIDE_KEY, String(curW)); } catch (e) {}
      g.classList.remove("drag"); document.body.classList.remove("resizing");
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      redraw();
    };
    g.addEventListener("pointerdown", (e) => {
      dragging = true; startX = e.clientX;
      startW = parseFloat(getComputedStyle(document.querySelector(".controls")).width) || SIDE_DEFAULT;
      g.classList.add("drag"); document.body.classList.add("resizing");
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
      e.preventDefault();
    });
    g.addEventListener("dblclick", () => { setSideWidth(SIDE_DEFAULT, true); redraw(); });
  }

  // ---- settings persistence (localStorage) ----------------------------------
  const LS_KEY = "axiom.settings.v1";
  let saveTimer = null;

  function snapshotState() {
    return {
      modelSize: $("modelSize").value, preset: $("preset").value, scaleK: $("scaleK").value,
      ode: $("ode").value, tMax: $("tMax").value, steps: $("steps").value,
      K: $("K").value, seed: $("seed").value, Kmc: $("Kmc").value,
      defaultStd, paramVals, paramFam, paramStd, x0Vals,
    };
  }
  function saveState() {
    try { localStorage.setItem(LS_KEY, JSON.stringify(snapshotState())); } catch (e) { /* full/blocked */ }
  }
  function saveStateSoon() {   // debounce so typing doesn't thrash localStorage
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(saveState, 300);
  }
  function loadState() {
    try { return JSON.parse(localStorage.getItem(LS_KEY) || "null"); }
    catch (e) { return null; }
  }
  function applyState(s) {
    if (s.modelSize && [...$("modelSize").options].some((o) => o.value === s.modelSize))
      $("modelSize").value = s.modelSize;
    if (s.preset) $("preset").value = s.preset;
    if (s.scaleK != null) $("scaleK").value = s.scaleK;
    if (typeof s.ode === "string") $("ode").value = s.ode;
    if (s.tMax != null) $("tMax").value = s.tMax;
    if (s.steps != null) $("steps").value = s.steps;
    if (s.K != null) $("K").value = s.K;
    if (s.seed != null) $("seed").value = s.seed;
    if (s.Kmc != null) $("Kmc").value = s.Kmc;
    defaultStd = s.defaultStd != null ? s.defaultStd : 0.3;
    paramVals = s.paramVals || {}; paramFam = s.paramFam || {};
    paramStd = s.paramStd || {}; x0Vals = s.x0Vals || {};
    syncTables();
  }

  function run() {
    const info = syncTables();
    if (!info) return;
    clearStale();
    const { states, rhs } = info;
    const size = $("modelSize").value;
    const cfg = MODELS.models[size];

    const params = {};
    for (const nm in paramVals)
      params[nm] = { value: paramVals[nm], relStd: (paramStd[nm] ?? defaultStd), family: paramFam[nm] || "lognormal" };
    const x0 = states.map((s) => x0Vals[s] || 0.0);

    let input;
    try {
      input = AXIOMEncode.encode({
        states, rhs, params, x0,
        tMax: parseFloat($("tMax").value), steps: parseInt($("steps").value, 10),
        inTaus: MODELS.in_taus, K: parseInt($("K").value, 10) || 2000,
        seed: parseInt($("seed").value, 10) || 12345,
      });
    } catch (e) { showError(e.message); return; }
    hideError();

    const t0 = performance.now();
    const model = AXIOMModel.build(cfg);
    const pred = model.run(input);
    const ms = performance.now() - t0;

    // display cleanup: sort quantiles ascending (no crossing). Clamp to >=0 ONLY for the
    // non-signed models, whose states are non-negative; the diverse/signed models are trained to
    // emit negative values (oscillators, mechanics) so their output must NOT be clipped.
    const clampNonNeg = !cfg.signed;
    for (let t = 0; t < pred.length; t++)
      for (let i = 0; i < input.n; i++) {
        const q = pred[t][i];
        if (clampNonNeg) for (let k = 0; k < q.length; k++) if (q[k] < 0) q[k] = 0;
        q.sort((a, b) => a - b);
      }

    // MC persistence: the Monte-Carlo overlay is a property of the SYSTEM+params (the true ODE),
    // NOT of the surrogate. So carry it across re-runs (e.g. switching model size) as long as the
    // ODE / params / x0 / horizon are unchanged; it clears only when the system truly changes.
    const sig = mcSignature();
    const carryMc = (LAST && LAST.mc && LAST.mcSig === sig) ? LAST.mc : null;
    const tGrid = Array.from({ length: input.steps }, (_, i) => i * input.dt);
    LAST = { states, rhs: info.rhs, tGrid, pred, taus: model.taus, input, cfg, size, ms,
             params, x0, tMax: parseFloat($("tMax").value), steps: input.steps,
             seed: parseInt($("seed").value, 10) || 12345, mcSig: sig, mc: carryMc };
    renderPlots(carryMc ? carryMc.pred : null);
    setStats(carryMc || null);
    window._lastRun = true;
  }

  // signature of everything the true-ODE Monte-Carlo depends on (NOT the surrogate model)
  function mcSignature() {
    return JSON.stringify({
      ode: $("ode").value, tMax: $("tMax").value, steps: $("steps").value,
      params: paramVals, std: paramStd, fam: paramFam, x0: x0Vals,
    });
  }

  // Monte-Carlo: integrate the TRUE ODE over K parameter draws, overlay its quantiles.
  function compareMC() {
    if (!LAST) { run(); if (!LAST) return; }
    const btn = $("mc"); btn.disabled = true; const label = btn.textContent;
    btn.textContent = "running Monte-Carlo…";
    // let the button repaint before the (blocking) MC loop
    setTimeout(() => {
      try {
        const mc = AXIOMMC.run({
          states: LAST.states, rhs: LAST.rhs, params: LAST.params, x0: LAST.x0,
          tMax: LAST.tMax, steps: LAST.steps, taus: LAST.taus,
          Kmc: parseInt($("Kmc").value, 10) || 400, seed: (LAST.seed ^ 0x9e3779b9) >>> 0,
          substeps: 8,
        });
        LAST.mc = mc;
        renderPlots(mc.pred);
        setStats(mc);
      } catch (e) {
        showError("MC failed: " + e.message);
      } finally {
        // always restore the button, even if MC or stats threw — else it hangs on "running…"
        btn.disabled = false; btn.textContent = label;
      }
    }, 30);
  }

  // mean & max abs difference between surrogate and MC MEDIAN trajectories.
  // Compare only over the real states shared by both: the surrogate pads an extra
  // "forcing node" row for constant/forcing terms that MC never produces.
  function compareMetric(pred, mcPred, mid) {
    let sum = 0, mx = 0, n = 0;
    const T = Math.min(pred.length, mcPred.length);
    for (let t = 0; t < T; t++) {
      const ns = Math.min(pred[t].length, mcPred[t].length);
      for (let i = 0; i < ns; i++) {
        const d = Math.abs(pred[t][i][mid] - mcPred[t][i][mid]);
        if (!isFinite(d)) continue;
        sum += d; mx = Math.max(mx, d); n++;
      }
    }
    return { mae: n ? sum / n : NaN, max: mx };
  }

  function setStats(mc) {
    const L = LAST;
    const name = L.size.charAt(0).toUpperCase() + L.size.slice(1);
    let s = `<b>${name}</b> model · ${L.input.n} states · ${L.steps} steps · <b>${L.ms.toFixed(1)} ms</b>`;
    if (mc) {
      const mid = L.taus.reduce((b, t, i) => Math.abs(t - 0.5) < Math.abs(L.taus[b] - 0.5) ? i : b, 0);
      const m = compareMetric(L.pred, mc.pred, mid);
      let sp = "";
      if (mc.ms > L.ms * 1.2) sp = ` (${(mc.ms / L.ms).toFixed(1)}× faster)`;
      s += ` · <span class="mc">vs ${mc.Kmc.toLocaleString()} reference simulations <b>${mc.ms.toFixed(0)} ms</b>${sp} · ` +
        `median error <b>${m.mae.toFixed(4)}</b> (max ${m.max.toFixed(3)})</span>`;
    }
    $("stats").innerHTML = s;
  }

  function renderPlots(mcPred) {
    const L = LAST, grid = $("plots"); grid.innerHTML = "";
    L.states.forEach((s, i) => {
      const cell = document.createElement("div"); cell.className = "plotcell";
      const cv = document.createElement("canvas"); cell.appendChild(cv);
      grid.appendChild(cell);
      requestAnimationFrame(() =>
        AXIOMPlot.drawState(cv, L.tGrid, L.pred, i, s, L.taus,
          AXIOMPlot.PAL[i % AXIOMPlot.PAL.length], mcPred ? mcPred : null, null));
    });
  }

  function populateModelInfo() {
    const sel = $("modelSize");
    [["small", "Small"], ["medium", "Medium"], ["large", "Large"],
     ["diverse", "Diverse (broad data)"]].forEach(([k, lbl]) => {
      if (!MODELS.models[k]) return;
      const o = document.createElement("option");
      o.value = k;
      o.textContent = lbl;
      sel.appendChild(o);
    });
    sel.value = "medium";
  }

  function init() {
    if (!MODELS) { showError("models.js failed to load (run build/export.py)"); return; }
    populateModelInfo();
    const ps = $("preset");
    Object.keys(PRESETS).forEach((n) => { const o = document.createElement("option"); o.textContent = n; ps.appendChild(o); });
    ps.addEventListener("change", () => loadPreset(ps.value));
    $("scaleK").addEventListener("change", () => loadPreset($("preset").value));  // re-expand at new k
    $("ode").addEventListener("input", () => { hideError(); });
    $("sync").addEventListener("click", syncTables);
    $("run").addEventListener("click", run);
    $("mc").addEventListener("click", compareMC);
    // LLM-assisted input: show the prompt, copy it, and import the spec the LLM returns
    if ($("llmPrompt")) $("llmPrompt").value = LLM_PROMPT;
    if ($("copyPrompt")) $("copyPrompt").addEventListener("click", () => {
      const btn = $("copyPrompt"), old = btn.textContent;
      const done = () => { btn.textContent = "✓ copied"; setTimeout(() => (btn.textContent = old), 1200); };
      try { navigator.clipboard.writeText(LLM_PROMPT).then(done, () => { $("llmPrompt").select(); }); }
      catch (e) { $("llmPrompt").select(); }
    });
    if ($("importBtn")) $("importBtn").addEventListener("click", importSpec);
    initResizer();

    // restore saved settings if present, else start from the default preset
    const saved = loadState();
    if (saved) { try { applyState(saved); } catch (e) { ps.value = "SEIR"; loadPreset("SEIR"); } }
    else { ps.value = "SEIR"; loadPreset("SEIR"); }

    // persist on any edit to a control (ode text, numbers, param table, selects)
    const controls = document.querySelector(".controls");
    controls.addEventListener("input", saveStateSoon);
    controls.addEventListener("change", saveStateSoon);

    run();
    if (location.hash.indexOf("mc") >= 0) compareMC();   // demo/test hook
    window.addEventListener("resize", () => { if (LAST) renderPlots(LAST.mc ? LAST.mc.pred : null); });
  }

  document.addEventListener("DOMContentLoaded", init);
})();
