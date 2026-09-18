/* AXIOM — Monte-Carlo reference: integrate the TRUE ODE for K parameter draws and
 * take empirical quantiles. Unlike the surrogate this uses the exact polynomial RHS
 * (real state powers, no hypergraph collapse), so it is an honest ground truth.
 */
(function (root, factory) {
  const m = factory(root.AXIOMEncode || (typeof require !== "undefined" && require("./encode.js")));
  if (typeof module !== "undefined" && module.exports) module.exports = m;
  else root.AXIOMMC = m;
})(typeof self !== "undefined" ? self : this, function (EN) {
  "use strict";

  // Parse each RHS to an AST. Using the AST evaluator (not just monomials) means the true ODE
  // integrates EXACTLY the expression as written -- including custom nonlinear functions
  // (sin/cos/exp/log/sqrt, non-integer powers, state-dependent denominators) -- so the MC ground
  // truth stays valid for the same custom ODEs the surrogate now handles. For polynomial RHS this
  // is numerically identical to the old monomial sum.
  function buildTrueODE(states, rhs) {
    const stateNames = states.slice();
    return { asts: rhs.map((r) => EN.parseAST(EN.tokenize(r))), stateNames };
  }

  function deriv(ode, x, pv) {
    const env = Object.assign({}, pv);
    const nm = ode.stateNames;
    for (let i = 0; i < nm.length; i++) env[nm[i]] = x[i];
    const dx = new Array(x.length);
    for (let i = 0; i < ode.asts.length; i++) dx[i] = EN.evalAST(ode.asts[i], env);
    return dx;
  }

  // RK4 with `substeps` per output step -> [steps][N]
  function integrate(eqs, x0, dt, steps, substeps, pv) {
    const N = x0.length, h = dt / substeps;
    let x = x0.slice();
    const traj = [x.slice()];
    const axpy = (a, D) => { const y = new Array(N); for (let i = 0; i < N; i++) y[i] = x[i] + a * D[i]; return y; };
    for (let s = 0; s < steps - 1; s++) {
      for (let ss = 0; ss < substeps; ss++) {
        const k1 = deriv(eqs, x, pv);
        const k2 = deriv(eqs, axpy(0.5 * h, k1), pv);
        const k3 = deriv(eqs, axpy(0.5 * h, k2), pv);
        const k4 = deriv(eqs, axpy(h, k3), pv);
        const nx = new Array(N);
        for (let i = 0; i < N; i++) nx[i] = x[i] + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
        x = nx;
      }
      traj.push(x.slice());
    }
    return traj;
  }

  // opts: {states, rhs, params:{name:{value,relStd,family}}, x0, tMax, steps,
  //        taus, Kmc, seed, substeps}
  function run(opts) {
    const eqs = buildTrueODE(opts.states, opts.rhs);
    const N = opts.states.length, steps = opts.steps, Q = opts.taus.length;
    const dt = opts.tMax / (steps - 1), substeps = opts.substeps || 8;
    const names = Object.keys(opts.params);
    const rng = EN.mulberry32((opts.seed || 999) >>> 0);
    const Kmc = opts.Kmc || 400;

    // per (t, state) sample buffers
    const buf = Array.from({ length: steps }, () =>
      Array.from({ length: N }, () => new Float64Array(Kmc)));

    const t0 = (typeof performance !== "undefined" ? performance.now() : Date.now());
    let ok = 0;
    for (let k = 0; k < Kmc; k++) {
      const pv = {};
      for (const nm of names) pv[nm] = EN.sampleParam(opts.params[nm], rng);
      let traj;
      try { traj = integrate(eqs, opts.x0, dt, steps, substeps, pv); }
      catch (e) { continue; }
      let finite = true;
      for (let t = 0; t < steps && finite; t++) for (let i = 0; i < N; i++)
        if (!isFinite(traj[t][i])) { finite = false; break; }
      if (!finite) continue;
      for (let t = 0; t < steps; t++) for (let i = 0; i < N; i++) buf[t][i][ok] = traj[t][i];
      ok++;
    }
    const ms = (typeof performance !== "undefined" ? performance.now() : Date.now()) - t0;

    // empirical quantiles
    const pred = Array.from({ length: steps }, () => Array.from({ length: N }, () => new Array(Q)));
    for (let t = 0; t < steps; t++) for (let i = 0; i < N; i++) {
      const col = Array.prototype.slice.call(buf[t][i], 0, ok).sort((a, b) => a - b);
      for (let q = 0; q < Q; q++) pred[t][i][q] = EN.quantileSorted(col, opts.taus[q]);
    }
    return { pred, ms, Kmc: ok };
  }

  // Single deterministic solve at the MEDIAN parameters (each param at its central value) ->
  // [steps][N]. This is solve(median theta); contrast with the MC/surrogate median-of-solves,
  // which differ for nonlinear ODEs (Jensen gap). opts: {states, rhs, params, x0, tMax, steps, substeps}
  function solveMedian(opts) {
    const eqs = buildTrueODE(opts.states, opts.rhs);
    const steps = opts.steps, dt = opts.tMax / (steps - 1), substeps = opts.substeps || 8;
    const pv = {};
    for (const nm in opts.params) pv[nm] = opts.params[nm].value;   // median = central value
    const t0 = (typeof performance !== "undefined" ? performance.now() : Date.now());
    const traj = integrate(eqs, opts.x0, dt, steps, substeps, pv);  // [steps][N]
    return { traj, ms: (typeof performance !== "undefined" ? performance.now() : Date.now()) - t0 };
  }

  return { run, buildTrueODE, solveMedian };
});
