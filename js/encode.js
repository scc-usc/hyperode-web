/* AXIOM — ODE text -> hypergraph + coefficient-quantile features (browser).
 *
 * Mirrors meta_pop_I:
 *   stochastic.hypergraph_structure  (symbolic RHS -> CSR hyperedges + coeff exprs)
 *   uq.sample_param_samples + uq.coeff_features  (mode="quantiles")
 *
 * Each additive monomial `c * (params) * (states)` on the RHS of dX_i/dt becomes a
 * hyperedge: sources = the DISTINCT state nodes in the term (matching sympy
 * free_symbols; state powers are dropped, i.e. states substituted to 1), target = i,
 * coefficient = c * prod(param^power). Coefficient samples over the parameter
 * distribution give he_scale (median) and he_feat (quantiles at IN_TAUS).
 */
(function (root, factory) {
  const m = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = m;
  else root.AXIOMEncode = m;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ----------------------------------------------------------------- tokenizer
  function tokenize(s) {
    const toks = [];
    const re = /\s*([A-Za-z_][A-Za-z0-9_]*|\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|[()+\-*/^])/g;
    let last = 0, m;
    while ((m = re.exec(s)) !== null) {
      if (m.index !== last && s.slice(last, m.index).trim() !== "")
        throw new Error(`unexpected '${s.slice(last, m.index).trim()}' in "${s}"`);
      toks.push(m[1]);
      last = re.lastIndex;
    }
    if (s.slice(last).trim() !== "") throw new Error(`unexpected trailing '${s.slice(last).trim()}'`);
    return toks;
  }

  // ------------------------------------------------- polynomial representation
  // A monomial is {c:Number, s:Map<name,power>, d:Array<Poly>}; a polynomial is an
  // array of monomials. `d` holds DENOMINATOR polynomials (rational factors), so a
  // term like `beta/(1+gamma)` is representable.
  //
  // Mirrors the Python encoder (stochastic.hypergraph_structure), where the coefficient
  // is an arbitrary sympy expression in the PARAMETERS and only the STATE structure has
  // to be a monomial. The parser can't tell states from params (it doesn't know the
  // state list), so it accepts any denominator here and `buildStructure` rejects the
  // ones that reference a state.
  function symKey(s) {
    return [...s.entries()].filter(([, p]) => p !== 0).sort((a, b) => a[0] < b[0] ? -1 : 1)
      .map(([n, p]) => `${n}^${p}`).join("*");
  }
  function polyKey(poly) {
    return poly.map((m) => `${m.c}:${symKey(m.s)}${denKey(m.d)}`).sort().join("+");
  }
  function denKey(d) {
    return d && d.length ? "/(" + d.map(polyKey).sort().join(")(") + ")" : "";
  }
  function monKey(mon) { return symKey(mon.s) + denKey(mon.d); }
  const dOf = (m) => (m.d && m.d.length ? m.d : []);
  const hasDen = (poly) => poly.some((m) => dOf(m).length > 0);

  function polyAdd(a, b, sign = 1) {
    const map = new Map();
    const push = (mon, sg) => {
      const k = monKey(mon);
      if (map.has(k)) map.get(k).c += sg * mon.c;
      else map.set(k, { c: sg * mon.c, s: new Map(mon.s), d: dOf(mon).slice() });
    };
    a.forEach((mn) => push(mn, 1));
    b.forEach((mn) => push(mn, sign));
    return [...map.values()].filter((mn) => Math.abs(mn.c) > 1e-14 || mn.s.size > 0 ? true : mn.c !== 0);
  }

  function polyMul(a, b) {
    const out = [];
    for (const ma of a) for (const mb of b) {
      const s = new Map(ma.s);
      for (const [n, p] of mb.s) s.set(n, (s.get(n) || 0) + p);
      out.push({ c: ma.c * mb.c, s, d: dOf(ma).concat(dOf(mb)) });
    }
    return polyAdd(out, []);                 // collect like terms
  }

  function polyPowInt(a, n) {
    if (n === 0) return [{ c: 1, s: new Map(), d: [] }];
    if (n < 0) {
      if (a.length === 1 && dOf(a[0]).length === 0) {      // single monomial: invert exactly
        const inv = [{ c: 1 / a[0].c, s: new Map([...a[0].s].map(([k, p]) => [k, -p])), d: [] }];
        return polyPowInt(inv, -n);
      }
      if (hasDen(a)) throw new Error("cannot invert a nested rational expression");
      // a SUM to a negative power -> carry |n| denominator factors
      return [{ c: 1, s: new Map(), d: Array.from({ length: -n }, () => a) }];
    }
    let r = [{ c: 1, s: new Map(), d: [] }];
    for (let i = 0; i < n; i++) r = polyMul(r, a);
    return r;
  }

  function polyDiv(a, b) {
    if (b.length === 1 && dOf(b[0]).length === 0) {        // single monomial: exact inverse
      const bm = b[0];
      if (bm.c === 0) throw new Error("division by zero");
      const inv = { c: 1 / bm.c, s: new Map([...bm.s].map(([k, p]) => [k, -p])), d: [] };
      return polyMul(a, [inv]);
    }
    if (hasDen(b)) throw new Error("division by a nested rational expression is unsupported");
    // division by a SUM: keep it as a denominator factor (legal iff parameter-only,
    // which buildStructure verifies once the state list is known).
    return a.map((m) => ({ c: m.c, s: new Map(m.s), d: dOf(m).concat([b]) }));
  }

  // ---------------------------------------------------------- recursive parser
  function parse(tokens) {
    let i = 0;
    const peek = () => tokens[i];
    const eat = (t) => { if (tokens[i] !== t) throw new Error(`expected '${t}' got '${tokens[i] ?? "EOF"}'`); i++; };
    const isNum = (t) => t !== undefined && /^(\d|\.)/.test(t);
    const isId = (t) => t !== undefined && /^[A-Za-z_]/.test(t);

    function parseExpr() {                    // term (('+'|'-') term)*
      let node = parseTerm();
      while (peek() === "+" || peek() === "-") {
        const op = peek(); i++;
        node = polyAdd(node, parseTerm(), op === "-" ? -1 : 1);
      }
      return node;
    }
    function parseTerm() {                    // factor (('*'|'/') factor)*
      let node = parseFactor();
      while (peek() === "*" || peek() === "/") {
        const op = peek(); i++;
        const rhs = parseFactor();
        node = op === "*" ? polyMul(node, rhs) : polyDiv(node, rhs);
      }
      return node;
    }
    function parseFactor() {                  // unary +/- then power
      if (peek() === "+") { i++; return parseFactor(); }
      if (peek() === "-") { i++; return polyMul([{ c: -1, s: new Map(), d: [] }], parseFactor()); }
      let base = parseBase();
      if (peek() === "^") {
        i++;
        // exponent: signed integer literal (optionally parenthesized)
        let neg = false;
        if (peek() === "+") i++; else if (peek() === "-") { neg = true; i++; }
        let par = false;
        if (peek() === "(") { par = true; i++; if (peek() === "-") { neg = true; i++; } }
        if (!isNum(peek())) throw new Error("exponent must be an integer");
        const n = Number(peek()); i++;
        if (par) eat(")");
        if (!Number.isInteger(n)) throw new Error("non-integer exponent unsupported");
        base = polyPowInt(base, neg ? -n : n);
      }
      return base;
    }
    function parseBase() {
      const t = peek();
      if (t === "(") { i++; const e = parseExpr(); eat(")"); return e; }
      if (isNum(t)) { i++; return [{ c: Number(t), s: new Map(), d: [] }]; }
      if (isId(t)) { i++; return [{ c: 1, s: new Map([[t, 1]]), d: [] }]; }
      throw new Error(`unexpected '${t ?? "EOF"}'`);
    }

    const poly = parseExpr();
    if (i < tokens.length) throw new Error(`unexpected '${tokens[i]}'`);
    return poly;
  }

  const parseRHS = (str) => parse(tokenize(str));

  // ------------------------------------------------ build hypergraph structure
  // states: ordered list of state names. rhsPolys: parsed poly per equation.
  // Returns CSR arrays + per-edge coefficient descriptor {c, params:Map}.
  const FRC_VALUE = 1.0;    // clamped value of the forcing node (matches build/export.py)

  function buildStructure(states, rhsPolys) {
    const stateIdx = new Map(states.map((s, i) => [s, i]));
    let he_src = [], he_ptr = [0]; const he_tgt = [], edgeCoef = [];
    rhsPolys.forEach((poly, ti) => {
      for (const mon of poly) {
        if (Math.abs(mon.c) < 1e-15) continue;
        const srcs = [];
        const params = new Map();
        for (const [name, pow] of mon.s) {
          if (stateIdx.has(name)) {
            // The RHS must be POLYNOMIAL IN THE STATES: a state may not sit in a
            // denominator. (The parameters are unrestricted -- see `den` below.)
            if (pow < 0)
              throw new Error(`d${states[ti]}/dt: state '${name}' appears in a denominator ` +
                `(power ${pow}) -- the RHS must be polynomial in the states`);
            // Expand a state power into REPEATED sources so the multiplicative term is exact:
            // N^2 -> [N,N] (prod = N*N), X^2*Y -> [X,X,Y]. (Previously the power was dropped, which
            // silently turned N^2 into N -- wrong for self-interaction, e.g. logistic/Brusselator.)
            for (let r = 0; r < pow; r++) srcs.push(stateIdx.get(name));
          } else if (pow !== 0) params.set(name, pow);   // param powers -> coefficient
        }
        // rational parameter factors: allowed, but only if state-free
        const den = dOf(mon);
        for (const dp of den)
          for (const dm of dp)
            for (const [nm] of dm.s)
              if (stateIdx.has(nm))
                throw new Error(`d${states[ti]}/dt: state '${nm}' appears inside a denominator ` +
                  `sum -- the RHS must be polynomial in the states (parameters may divide freely)`);
        srcs.sort((a, b) => a - b);          // distinct states in state order
        for (const si of srcs) he_src.push(si);
        he_ptr.push(he_src.length);
        he_tgt.push(ti);
        edgeCoef.push({ c: mon.c, params, den });
      }
    });
    // Route every CONSTANT (empty-source) hyperedge through a single clamped forcing node Frc1
    // (matches how the forcing-aware model was trained; must mirror build/export.py:forcify).
    let n = states.length, hasForce = false;
    const H = he_tgt.length;
    for (let h = 0; h < H; h++) if (he_ptr[h + 1] === he_ptr[h]) { hasForce = true; break; }
    if (hasForce) {
      const frc = n, ns = [], np = [0];
      for (let h = 0; h < H; h++) {
        if (he_ptr[h + 1] === he_ptr[h]) ns.push(frc);
        else for (let i = he_ptr[h]; i < he_ptr[h + 1]; i++) ns.push(he_src[i]);
        np.push(ns.length);
      }
      he_src = ns; he_ptr = np; n += 1;
    }
    const forceMask = new Array(n).fill(0);
    if (hasForce) forceMask[n - 1] = 1;
    return { n, he_src, he_ptr, he_tgt, edgeCoef, forceMask, hasForce };
  }

  // ----------------------------------------------------- coefficient sampling
  function randn(rng) {                       // Box-Muller
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // Deterministic PRNG (mulberry32) so a seed reproduces the bands.
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function sampleParam(spec, rng) {           // one draw for a parameter
    const c = Math.max(spec.value, 1e-6), rs = spec.relStd;
    switch (spec.family) {
      case "uniform": {
        const w = rs * c, lo = Math.max(c - w, 0);
        return lo + rng() * (Math.max(c + w, lo + 1e-6) - lo);
      }
      case "normal":
        return Math.max(c + randn(rng) * rs * c, 1e-6);
      default:                                // lognormal
        return c * Math.exp(randn(rng) * rs);
    }
  }

  // np.quantile (linear interpolation, method="linear"/type 7) on a sorted array.
  function quantileSorted(sorted, p) {
    const n = sorted.length;
    if (n === 0) return NaN;                  // no valid draws (all diverged)
    if (n === 1) return sorted[0];
    const idx = p * (n - 1), lo = Math.floor(idx), frac = idx - lo;
    if (lo + 1 >= n) return sorted[n - 1];
    return sorted[lo] * (1 - frac) + sorted[lo + 1] * frac;
  }

  // Numeric value of a parameter-only polynomial for one parameter draw
  // (the JS stand-in for sympy's lambdify of a coefficient expression).
  function evalPoly(poly, draw) {
    let sum = 0;
    for (const m of poly) {
      let v = m.c;
      for (const [nm, pow] of m.s) v *= Math.pow(draw[nm], pow);
      for (const dp of dOf(m)) v /= evalPoly(dp, draw);
      sum += v;
    }
    return sum;
  }

  const evalCoef = (ec, draw) => {
    let v = ec.c;
    for (const [name, pow] of ec.params) v *= Math.pow(draw[name], pow);
    for (const dp of (ec.den || [])) v /= evalPoly(dp, draw);
    return v;
  };

  // Produce he_scale (median coeff) + he_feat (coeff quantiles at inTaus).
  function coeffFeatures(edgeCoef, paramSpecs, inTaus, K, seed) {
    const rng = mulberry32(seed >>> 0);
    const H = edgeCoef.length;
    const samples = Array.from({ length: H }, () => new Float64Array(K));
    const names = Object.keys(paramSpecs);
    for (let k = 0; k < K; k++) {
      const draw = {};
      for (const nm of names) draw[nm] = sampleParam(paramSpecs[nm], rng);
      for (let h = 0; h < H; h++) samples[h][k] = evalCoef(edgeCoef[h], draw);
    }
    const he_scale = new Array(H), he_feat = new Array(H);
    for (let h = 0; h < H; h++) {
      const col = Array.from(samples[h]).sort((a, b) => a - b);
      he_scale[h] = quantileSorted(col, 0.5);
      he_feat[h] = inTaus.map((t) => quantileSorted(col, t));
    }
    return { he_scale, he_feat };
  }

  // ------------------------------------------------ CUSTOM (non-polynomial) terms
  // The mass-action product prod_h = PROD_s x_s is the hyperedge's driving nonlinearity, fed to
  // psi and computed from the raw states each substep. A term whose STATE part is NOT a plain
  // product of states -- sin/cos/exp/log/sqrt/tanh of a state, a non-integer state power, a state
  // in a denominator -- gets that state-nonlinearity g(sources) used IN PLACE of prod at inference
  // (js/model.js:input.prodOverride), no retraining. The multiplicative PARAMETER factor stays the
  // (uncertain) edge coefficient; only the state-dependent part becomes g. Polynomial terms are
  // untouched (routed through the validated path above). RULE: g must stay NON-NEGATIVE on [0,1]
  // states (the trained psi only saw non-negative prod) -- true for sqrt/power/ratio/Hill/exp(-.)/
  // saturating; a sign-changing g (sin/cos) must be written non-negatively (see README).
  const EPS = 1e-8;
  const FUNCS = {
    sin: Math.sin, cos: Math.cos, tanh: Math.tanh, abs: Math.abs, exp: Math.exp,
    log: (x) => Math.log(Math.max(x, EPS)), sqrt: (x) => Math.sqrt(Math.max(x, 0)),
  };

  // recursive-descent AST parser (superset of the poly parser: functions + real exponents)
  function parseAST(toks) {
    let i = 0;
    const peek = () => toks[i], eat = (t) => { if (toks[i] !== t) throw new Error(`expected '${t}'`); i++; };
    const isNum = (t) => t !== undefined && /^(\d|\.)/.test(t);
    const isId = (t) => t !== undefined && /^[A-Za-z_]/.test(t);
    function expr() { let n = term(); while (peek() === "+" || peek() === "-") { const op = toks[i++]; n = { t: "bin", op, l: n, r: term() }; } return n; }
    function term() { let n = fac(); while (peek() === "*" || peek() === "/") { const op = toks[i++]; n = { t: "bin", op, l: n, r: fac() }; } return n; }
    function fac() { if (peek() === "+") { i++; return fac(); } if (peek() === "-") { i++; return { t: "neg", x: fac() }; } return pw(); }
    function pw() {
      let b = base();
      if (peek() === "^") {
        i++; let sg = 1; if (peek() === "+") i++; else if (peek() === "-") { sg = -1; i++; }
        let par = false; if (peek() === "(") { par = true; i++; if (peek() === "-") { sg = -1; i++; } }
        if (!isNum(peek())) throw new Error("exponent must be a number"); const e = sg * Number(toks[i++]);
        if (par) eat(")"); b = { t: "pow", base: b, exp: e };
      }
      return b;
    }
    function base() {
      const t = peek();
      if (t === "(") { i++; const e = expr(); eat(")"); return e; }
      if (isNum(t)) { i++; return { t: "num", v: Number(t) }; }
      if (isId(t)) { i++; if (peek() === "(") { i++; const a = expr(); eat(")"); return { t: "fn", name: t, arg: a }; } return { t: "sym", name: t }; }
      throw new Error(`unexpected '${t ?? "EOF"}'`);
    }
    const e = expr(); if (i < toks.length) throw new Error(`unexpected '${toks[i]}'`); return e;
  }

  function evalAST(a, env) {
    switch (a.t) {
      case "num": return a.v;
      case "sym": { const v = env[a.name]; if (v === undefined) throw new Error(`no value for '${a.name}'`); return v; }
      case "neg": return -evalAST(a.x, env);
      case "bin": { const l = evalAST(a.l, env), r = evalAST(a.r, env); return a.op === "+" ? l + r : a.op === "-" ? l - r : a.op === "*" ? l * r : l / r; }
      case "pow": return Math.pow(evalAST(a.base, env), a.exp);
      case "fn": { const f = FUNCS[a.name]; if (!f) throw new Error(`unknown function '${a.name}'`); return f(evalAST(a.arg, env)); }
    }
    throw new Error("bad AST");
  }
  function freeSyms(a, out = new Set()) {
    if (a.t === "sym") out.add(a.name);
    else if (a.t === "neg") freeSyms(a.x, out);
    else if (a.t === "pow") freeSyms(a.base, out);
    else if (a.t === "fn") freeSyms(a.arg, out);
    else if (a.t === "bin") { freeSyms(a.l, out); freeSyms(a.r, out); }
    return out;
  }

  // split a token stream into top-level additive terms (respecting parens), with signs
  function splitTermsTokens(toks) {
    const out = []; let depth = 0, cur = [], sign = 1;
    const valEnd = (t) => t !== undefined && (t === ")" || /^[A-Za-z0-9_.]/.test(t));
    for (let i = 0; i < toks.length; i++) {
      const t = toks[i];
      if (t === "(") depth++; else if (t === ")") depth--;
      if (depth === 0 && (t === "+" || t === "-") && valEnd(toks[i - 1])) { out.push({ sign, toks: cur }); cur = []; sign = t === "-" ? -1 : 1; }
      else if (depth === 0 && (t === "+" || t === "-") && !valEnd(toks[i - 1])) { if (t === "-") sign = -sign; }
      else cur.push(t);
    }
    if (cur.length) out.push({ sign, toks: cur });
    return out;
  }

  // flatten a term AST into multiplicative factors {ast, denom}, pulling out the sign
  function factorize(a, denom, acc) {
    if (a.t === "bin" && a.op === "*") { factorize(a.l, denom, acc); factorize(a.r, denom, acc); }
    else if (a.t === "bin" && a.op === "/") { factorize(a.l, denom, acc); factorize(a.r, !denom, acc); }
    else if (a.t === "neg") { acc.sign *= -1; factorize(a.x, denom, acc); }
    else acc.factors.push({ ast: a, denom });
    return acc;
  }

  // does a parsed POLY put a state in a denominator / negative power? (then it is custom)
  function polyHasStateDenom(poly, stateSet) {
    for (const m of poly) {
      for (const [nm, p] of m.s) if (stateSet.has(nm) && p < 0) return true;
      for (const dp of dOf(m)) for (const dm of dp) for (const [nm] of dm.s) if (stateSet.has(nm)) return true;
    }
    return false;
  }

  // Build one custom edge from a term's tokens: coefficient (param-only factors, uncertain) +
  // g (state-containing factors, evaluated on X at inference with params at their central value).
  function buildCustomEdge(term, ti, states, stateSet, stateIdx, params) {
    const ast = parseAST(term.toks);
    const acc = factorize(ast, false, { sign: term.sign, factors: [] });
    const central = {}; for (const nm in params) central[nm] = params[nm].value;
    const coefF = [], gF = [], gStates = new Set(); const refP = new Set();
    for (const f of acc.factors) {
      const fs = freeSyms(f.ast);
      let hasState = false;
      for (const s of fs) { if (stateSet.has(s)) { hasState = true; gStates.add(s); } else refP.add(s); }
      (hasState ? gF : coefF).push(f);
    }
    const sources = [...gStates].map((s) => stateIdx.get(s)).sort((a, b) => a - b);
    const gNames = [...gStates];
    const coefFn = (draw) => { let v = acc.sign; for (const f of coefF) { const x = evalAST(f.ast, draw); v = f.denom ? v / x : v * x; } return v; };
    const gFn = (X, q) => {
      const env = Object.assign({}, central);
      // floor state values at EPS, matching the mass-action product's log-space clamp
      // (model.js default prod uses Math.max(x, EPS)); keeps x^p / log / division well-defined.
      for (const nm of gNames) env[nm] = Math.max(X[stateIdx.get(nm)][q], EPS);
      let v = 1; for (const f of gF) { const x = evalAST(f.ast, env); v = f.denom ? v / x : v * x; }
      return v;
    };
    return { tgt: ti, sources, coefFn, gFn, refP: [...refP] };
  }

  // Full pipeline: parsed ODE + params + timegrid -> model input.
  // opts: {states, rhs:[strings], params:{name:{value,relStd,family}}, x0:[..],
  //        tMax, steps, inTaus, K, seed}
  // Polynomial ODEs use the validated path; if any term is a non-polynomial function of the
  // states, that hyperedge's product is auto-replaced by the custom g at inference.
  function encode(opts) {
    const states = opts.states, stateSet = new Set(states);
    const stateIdx = new Map(states.map((s, i) => [s, i]));
    const polyPart = states.map(() => []);        // polynomial terms per equation
    const customEdges = [];                        // one per non-polynomial term
    opts.rhs.forEach((rstr, ti) => {
      let toks; try { toks = tokenize(rstr); } catch (e) { throw new Error(`equation ${ti + 1} (d${states[ti]}/dt): ${e.message}`); }
      for (const term of splitTermsTokens(toks)) {
        let poly = null; try { poly = parse(term.toks); } catch (e) { poly = null; }
        if (poly && !polyHasStateDenom(poly, stateSet)) {
          polyPart[ti] = polyAdd(polyPart[ti], poly, term.sign);   // polynomial-in-states -> default prod
        } else {
          try { customEdges.push(buildCustomEdge(term, ti, states, stateSet, stateIdx, opts.params)); }
          catch (e) { throw new Error(`equation ${ti + 1} (d${states[ti]}/dt): ${e.message}`); }
        }
      }
    });

    if (customEdges.length === 0) return encodePoly(opts);   // untouched validated path

    // --- combine polynomial edges (validated builder) with the custom edges ---
    const struct = buildStructure(states, polyPart);          // n, he_src/ptr/tgt, edgeCoef, forceMask
    let { n, he_src, he_ptr, he_tgt, edgeCoef, forceMask } = struct;
    const coefFns = edgeCoef.map((ec) => (draw) => evalCoef(ec, draw));  // poly edge coefficients
    const prodFns = {};                                       // combined-edge-index -> g
    for (const ce of customEdges) {
      const eIdx = he_tgt.length;
      for (const s of ce.sources) he_src.push(s);
      he_ptr.push(he_src.length);
      he_tgt.push(ce.tgt);
      coefFns.push(ce.coefFn);
      prodFns[eIdx] = ce.gFn;
    }
    if (forceMask.length < n) { while (forceMask.length < n) forceMask.push(0); }

    // validate parameters have values (poly + custom)
    const referenced = new Set();
    edgeCoef.forEach((ec) => {
      ec.params.forEach((_, nm) => referenced.add(nm));
      (ec.den || []).forEach((dp) => dp.forEach((dm) => dm.s.forEach((_p, nm) => referenced.add(nm))));
    });
    customEdges.forEach((ce) => ce.refP.forEach((nm) => referenced.add(nm)));
    for (const nm of referenced) if (!opts.params[nm]) throw new Error(`parameter '${nm}' has no value`);

    const { he_scale, he_feat } = coeffFeaturesFns(coefFns, opts.params, opts.inTaus, opts.K || 2000, opts.seed || 12345);
    const steps = opts.steps, dt = opts.tMax / (steps - 1);
    const x0 = opts.x0.slice();
    while (x0.length < n) x0.push(FRC_VALUE);

    // prodOverride: replace prod for the custom edges with their g(sources); others stay default.
    const entries = Object.entries(prodFns).map(([e, fn]) => [Number(e), fn]);
    const prodOverride = (prod, X, idx, input, Q) => {
      for (const [e, fn] of entries) { const p = prod[e]; for (let q = 0; q < Q; q++) p[q] = fn(X, q); }
      return prod;
    };
    return {
      n, he_src, he_ptr, he_tgt, he_scale, he_feat, x0, dt, steps, force_mask: forceMask,
      prodOverride, customEdges: entries.map(([e]) => e), referenced: [...referenced],
    };
  }

  // coefficient features from a list of coefFn(draw) closures (poly + custom), mirroring
  // coeffFeatures but source-agnostic (each edge's coefficient is an arbitrary param function).
  function coeffFeaturesFns(coefFns, paramSpecs, inTaus, K, seed) {
    const rng = mulberry32(seed >>> 0), H = coefFns.length;
    const samples = Array.from({ length: H }, () => new Float64Array(K));
    const names = Object.keys(paramSpecs);
    for (let k = 0; k < K; k++) {
      const draw = {}; for (const nm of names) draw[nm] = sampleParam(paramSpecs[nm], rng);
      for (let h = 0; h < H; h++) samples[h][k] = coefFns[h](draw);
    }
    const he_scale = new Array(H), he_feat = new Array(H);
    for (let h = 0; h < H; h++) {
      const col = Array.from(samples[h]).sort((a, b) => a - b);
      he_scale[h] = quantileSorted(col, 0.5);
      he_feat[h] = inTaus.map((t) => quantileSorted(col, t));
    }
    return { he_scale, he_feat };
  }

  // the original (polynomial-only) pipeline, unchanged
  function encodePoly(opts) {
    const rhsPolys = opts.rhs.map((r, i) => {
      try { return parseRHS(r); }
      catch (e) { throw new Error(`equation ${i + 1} (d${opts.states[i]}/dt): ${e.message}`); }
    });
    const struct = buildStructure(opts.states, rhsPolys);
    const referenced = new Set();
    struct.edgeCoef.forEach((ec) => {
      ec.params.forEach((_, nm) => referenced.add(nm));
      (ec.den || []).forEach((dp) => dp.forEach((dm) => dm.s.forEach((_p, nm) => referenced.add(nm))));
    });
    for (const nm of referenced)
      if (!opts.params[nm]) throw new Error(`parameter '${nm}' has no value`);
    const { he_scale, he_feat } = coeffFeatures(
      struct.edgeCoef, opts.params, opts.inTaus, opts.K || 2000, opts.seed || 12345);
    const steps = opts.steps, dt = opts.tMax / (steps - 1);
    const x0 = opts.x0.slice();
    while (x0.length < struct.n) x0.push(FRC_VALUE);
    return {
      n: struct.n, he_src: struct.he_src, he_ptr: struct.he_ptr, he_tgt: struct.he_tgt,
      he_scale, he_feat, x0, dt, steps, force_mask: struct.forceMask,
      edgeCoef: struct.edgeCoef, referenced: [...referenced],
    };
  }

  return { tokenize, parseRHS, buildStructure, coeffFeatures, encode, encodePoly,
           parseAST, evalAST, splitTermsTokens, factorize, coeffFeaturesFns,
           evalCoef, evalPoly, quantileSorted, mulberry32, sampleParam, randn,
           _polyAdd: polyAdd, _polyMul: polyMul };
});
