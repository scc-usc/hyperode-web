/* AXIOM generalized product — custom hyperedge nonlinearities (browser).
 *
 * The surrogate's per-hyperedge feature prod_h = PROD_{s in S_h} x_s is the term's driving
 * nonlinearity. It is computed from the raw states each substep, so it can be SWAPPED for any
 * g_h(x_{S_h}) at inference with no retraining: sin, S*I/N, Hill, Ricker, saturating, x^p, ...
 *
 * TWO RULES (or the swap goes out-of-distribution and the rollout flatlines / diverges):
 *   (1) every state must be shifted+normalized into [0,1] (non-negative, bounded above);
 *   (2) every swapped value must be NON-NEGATIVE -- the mass-action product is always >=0, so
 *       the trained psi only ever saw non-negative prod. Push any sign/offset into the edge
 *       coefficient (he_scale) plus a constant forcing node. Example: a restoring force
 *       cos(pi*s) is sign-changing; write cos(pi*s) = 2*c(s) - 1 with c(s)=(cos(pi*s)+1)/2 in
 *       [0,1] as the swapped feature, and move the "-1" into a constant edge.
 *
 * Usage:
 *   input.prodOverride = AXIOMProd.compose(
 *       AXIOMProd.power(1, 0, 2, 0.5),        // edge 1: prod = x[0] * x[2]^0.5   (S * I^0.5)
 *       AXIOMProd.saturating(4, 3, 2, 8),     // edge 4: prod = x[3]*x[2]/(1+8*x[2])
 *   );
 *   const traj = model.run(input);            // uses the swapped nonlinearities
 *
 * Each builder returns an edge-override (prod, X, idx, input, Q) that MUTATES prod[edge] in
 * place; compose() chains them and leaves all other edges at their default mass-action product.
 */
(function (root, factory) {
  const m = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = m;
  else root.AXIOMProd = m;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  const EPS = 1e-8;
  const nn = (v) => (v > 0 ? v : 0);                 // clamp to non-negative (rule 2)

  // Generic: set prod[edge][q] = fn(X, q)   (fn MUST be non-negative; it is clamped anyway).
  function edge(edgeIndex, fn) {
    return function (prod, X, idx, input, Q) {
      const p = prod[edgeIndex];
      for (let q = 0; q < Q; q++) p[q] = nn(fn(X, q));
    };
  }

  // prod = x[mulNode] * x[powNode]^p     (sub/super-linear incidence S*I^p)
  function power(edgeIndex, mulNode, powNode, p) {
    return edge(edgeIndex, (X, q) =>
      Math.max(X[mulNode][q], EPS) * Math.pow(Math.max(X[powNode][q], EPS), p));
  }

  // prod = x[a] * x[b] / (1 + alpha * x[b])     (saturating / Holling-II incidence)
  function saturating(edgeIndex, a, b, alpha) {
    return edge(edgeIndex, (X, q) => X[a][q] * X[b][q] / (1 + alpha * X[b][q]));
  }

  // prod = x[a] * x[b] * exp(-alpha * x[b])     (Ricker / overcompensating)
  function ricker(edgeIndex, a, b, alpha) {
    return edge(edgeIndex, (X, q) => X[a][q] * X[b][q] * Math.exp(-alpha * X[b][q]));
  }

  // prod = x[a] * x[b]^n / (K^n + x[b]^n)       (Hill / sigmoidal switch)
  function hill(edgeIndex, a, b, K, n) {
    const Kn = Math.pow(K, n);
    return edge(edgeIndex, (X, q) => {
      const bn = Math.pow(Math.max(X[b][q], EPS), n);
      return X[a][q] * bn / (Kn + bn);
    });
  }

  // prod = x[a] * x[b] / sum(x[denomNodes])     (frequency-dependent transmission S*I/N)
  function freqdep(edgeIndex, a, b, denomNodes) {
    return edge(edgeIndex, (X, q) => {
      let N = 0; for (const d of denomNodes) N += X[d][q];
      return X[a][q] * X[b][q] / Math.max(N, EPS);
    });
  }

  // prod = (cos(a * x[node] + b) + 1) / 2  in [0,1]   (non-negative encoding of a cosine
  // restoring force; pair with a constant forcing edge carrying the -1/2, see README).
  function cosShifted(edgeIndex, node, a, b) {
    return edge(edgeIndex, (X, q) => (Math.cos(a * X[node][q] + b) + 1) / 2);
  }

  // Chain several edge-overrides into one input.prodOverride; untouched edges keep the default.
  function compose(...fns) {
    return function (prod, X, idx, input, Q) {
      for (const f of fns) f(prod, X, idx, input, Q);
      return prod;
    };
  }

  return { edge, power, saturating, ricker, hill, freqdep, cosShifted, compose, EPS };
});
