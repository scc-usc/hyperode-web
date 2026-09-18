/* AXIOM forward-UQ surrogate — browser runtime.
 *
 * Faithful re-implementation of uq.py:UQNeuralODE.forward for layers>=1,
 * mass_action=True (the exported trio is layers=2, Euler, substeps=2).
 *
 * A model is a quantile neural-ODE: the per-node state is a vector of Q quantiles,
 * and dX/dt is a learned hypergraph message-passing field integrated with a
 * fixed-step solver. Input = hypergraph (CSR) + per-edge coefficient median
 * (he_scale) and coefficient-quantile features (he_feat).
 */
(function (root, factory) {
  const m = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = m;
  else root.AXIOMModel = m;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  const EPS = 1e-8;

  const relu = (v) => { for (let i = 0; i < v.length; i++) if (v[i] < 0) v[i] = 0; return v; };

  // linear layer: W is [out][in] (row-major, as PyTorch Linear.weight), b is [out]
  function linear(W, b, x) {
    const out = new Array(W.length);
    for (let i = 0; i < W.length; i++) {
      const row = W[i];
      let s = b[i];
      for (let j = 0; j < row.length; j++) s += row[j] * x[j];
      out[i] = s;
    }
    return out;
  }

  // two-layer MLP: Linear -> ReLU -> Linear, with optional trailing ReLU (last_act)
  function mlp2(P, x, lastAct) {
    const h = relu(linear(P.w0, P.b0, x));
    const o = linear(P.w2, P.b2, h);
    return lastAct ? relu(o) : o;
  }

  // Build a callable model from one entry of models.json.
  function build(cfg) {
    const W = cfg.weights;
    const L = cfg.layers, Q = cfg.taus.length;
    const phiH = cfg.phi_hidden, efd = cfg.edge_feat_dim;
    const width = cfg.psi_hidden;            // mp_width defaults to psi_hidden
    // DIVERSE/SIGNED line: a per-edge nonlinearity-TYPE embedding replaces the source-count
    // feature, and the driving product is SIGNED (states may be < 0). Default (paper models):
    // use_types=false -> count feature + non-negative product (byte-identical to before).
    const useTypes = !!cfg.use_types, signed = !!cfg.signed;
    const ted = useTypes ? cfg.type_embed_dim : 0;
    const typeEmbed = useTypes ? cfg.weights["type_embed.weight"] : null;   // [n_types][ted]
    // GAT-style HYPEREDGE attention field: NO source pooling; each hyperedge message is
    // psi([prod, type, edge_feat]) and the messages incident on a node are combined by a learned
    // per-hyperedge weight -- "hgate" = sigmoid gate then SUM (additive, mass-action-preserving),
    // "hgat" = segment-softmax over the node's hyperedges. Default "hypergraph" = the sum-pool field.
    const fieldType = cfg.field_type || "hypergraph";
    const isHgate = fieldType === "hgat" || fieldType === "hgate";
    const hgateSoftmax = fieldType === "hgat";
    if (isHgate && (cfg.n_heads || 1) > 1) throw new Error("hgate JS port: n_heads>1 unsupported");
    const get = (k) => W[k];
    const layers = [];
    for (let l = 0; l < L; l++) {
      const pfx = L === 1 ? "" : `s.${l}`;   // placeholder, set per-family below
      const phiKey = L === 1 ? "phi" : `phis.${l}`;
      const psiKey = L === 1 ? "psi" : `psis.${l}`;
      const outDim = L === 1 ? Q : (l < L - 1 ? width : Q);
      const layer = {
        phi: { w0: get(`${phiKey}.0.weight`), b0: get(`${phiKey}.0.bias`),
               w2: get(`${phiKey}.2.weight`), b2: get(`${phiKey}.2.bias`) },
        psi: { w0: get(`${psiKey}.0.weight`), b0: get(`${psiKey}.0.bias`),
               w2: get(`${psiKey}.2.weight`), b2: get(`${psiKey}.2.bias`) },
        outDim,
        self: (L > 1) ? { w: get(`selfs.${l}.weight`), b: get(`selfs.${l}.bias`) } : null,
      };
      layers.push(layer);
    }

    // hgate/hgat layers: per-layer message MLP (hgat_psi), attention-score MLP (hgat_att),
    // and self transform (hgat_self). Only built for the attention field types.
    const hgateLayers = [];
    if (isHgate) {
      for (let l = 0; l < L; l++) {
        hgateLayers.push({
          psi: { w0: get(`hgat_psi.${l}.0.weight`), b0: get(`hgat_psi.${l}.0.bias`),
                 w2: get(`hgat_psi.${l}.2.weight`), b2: get(`hgat_psi.${l}.2.bias`) },
          att: { w0: get(`hgat_att.${l}.0.weight`), b0: get(`hgat_att.${l}.0.bias`),
                 w2: get(`hgat_att.${l}.2.weight`), b2: get(`hgat_att.${l}.2.bias`) },
          self: { w: get(`hgat_self.${l}.weight`), b: get(`hgat_self.${l}.bias`) },
          outDim: l < L - 1 ? width : Q,
        });
      }
    }

    // Precompute per-batch index tensors (src->edge map, counts).
    function makeIdx(input) {
      const { he_ptr, he_src, he_tgt } = input;
      const H = he_tgt.length, nnz = he_src.length;
      const srcHe = new Int32Array(nnz);
      const counts = new Array(H);
      for (let h = 0; h < H; h++) {
        counts[h] = he_ptr[h + 1] - he_ptr[h];
        for (let i = he_ptr[h]; i < he_ptr[h + 1]; i++) srcHe[i] = h;
      }
      return { H, nnz, srcHe, counts };
    }

    // The learned derivative field dX = f(X). X is N x Q.
    function field(X, input, idx) {
      const { he_src, he_tgt, he_scale, he_feat } = input;
      const N = input.n, H = idx.H;
      if (H === 0) return X.map(() => new Array(Q).fill(0));

      // prod[h][q] = product over sources of clamp(X[src][q], EPS); 1 if no sources.
      // This is the hyperedge's mass-action DRIVING NONLINEARITY. It can be swapped for an
      // arbitrary g_h(x_{S_h}) at inference via input.prodOverride (sin/1-over-N/Hill/x^p/...)
      // with NO retraining -- see js/prod.js. Rule: the swapped value must stay NON-NEGATIVE
      // (the trained psi only ever saw non-negative prod), so push any sign/offset into he_scale
      // and a constant forcing node. Default (no override) is byte-identical to the Python port.
      const prod = new Array(H);
      for (let h = 0; h < H; h++) prod[h] = new Array(Q).fill(1);
      for (let i = 0; i < idx.nnz; i++) {
        const node = he_src[i], h = idx.srcHe[i], xr = X[node];
        const p = prod[h];
        for (let q = 0; q < Q; q++) {
          const v = xr[q];
          // signed model: |v| with EPS floor, sign preserved (matches uq _seg_signed_prod);
          // classic model: clamp to non-negative (matches the mass-action log-space product).
          p[q] *= signed ? (v < 0 ? -Math.max(-v, EPS) : Math.max(v, EPS)) : Math.max(v, EPS);
        }
      }
      if (input.prodOverride) input.prodOverride(prod, X, idx, input, Q);

      if (isHgate) {
        // e_h = [prod (Q) | type embedding (ted) | edge_feat (efd)] -- NO pool, NO source count.
        const eDim = Q + (useTypes ? ted : 0) + efd;
        const E = new Array(H);
        for (let h = 0; h < H; h++) {
          const e = new Array(eDim); let o = 0;
          const pr = prod[h]; for (let q = 0; q < Q; q++) e[o++] = pr[q];
          if (useTypes) {
            const row = typeEmbed[input.he_type ? input.he_type[h] : 0];
            for (let k = 0; k < ted; k++) e[o++] = row[k];
          }
          const ef = he_feat[h]; for (let k = 0; k < efd; k++) e[o++] = ef[k];
          E[h] = e;
        }
        let hE = X;                          // node embedding
        for (let l = 0; l < L; l++) {
          const lay = hgateLayers[l], outDim = lay.outDim;
          const msg = new Array(H), score = new Array(H);
          for (let h = 0; h < H; h++) {
            const m = mlp2(lay.psi, E[h], false);           // [outDim]
            const sc = he_scale[h]; for (let k = 0; k < outDim; k++) m[k] *= sc;
            msg[h] = m;
            const attIn = hE[he_tgt[h]].concat(E[h]);        // [inDim + eDim]
            score[h] = mlp2(lay.att, attIn, false)[0];
          }
          // per-hyperedge weight: additive sigmoid gate, or segment-softmax over a node's hyperedges
          const w = new Array(H);
          if (hgateSoftmax) {
            let mx = -Infinity;
            for (let h = 0; h < H; h++) if (score[h] > mx) mx = score[h];
            const ex = new Array(H), den = new Array(N).fill(0);
            for (let h = 0; h < H; h++) { ex[h] = Math.exp(score[h] - mx); den[he_tgt[h]] += ex[h]; }
            for (let h = 0; h < H; h++) w[h] = ex[h] / (den[he_tgt[h]] || EPS);
          } else {
            for (let h = 0; h < H; h++) w[h] = 1 / (1 + Math.exp(-score[h]));
          }
          const newH = new Array(N);
          for (let node = 0; node < N; node++) newH[node] = linear(lay.self.w, lay.self.b, hE[node]);
          for (let h = 0; h < H; h++) {
            const t = he_tgt[h], m = msg[h], ww = w[h], dst = newH[t];
            for (let k = 0; k < outDim; k++) dst[k] += ww * m[k];
          }
          if (l < L - 1) for (let node = 0; node < N; node++) relu(newH[node]);
          hE = newH;
        }
        return hE;                           // N x Q = dX
      }

      let hEmb = X;                          // node embedding; layer 0 = raw state
      for (let l = 0; l < L; l++) {
        const lay = layers[l], outDim = lay.outDim;
        // pool[h] = sum over sources of phi(hEmb[src])
        const pool = new Array(H);
        for (let h = 0; h < H; h++) pool[h] = new Array(phiH).fill(0);
        for (let i = 0; i < idx.nnz; i++) {
          const node = he_src[i], h = idx.srcHe[i];
          const g = mlp2(lay.phi, hEmb[node], true);
          const pv = pool[h];
          for (let k = 0; k < phiH; k++) pv[k] += g[k];
        }
        // message per edge: scale * psi([pool, prod, count, efeat])
        const msg = new Array(H);
        for (let h = 0; h < H; h++) {
          const inp = new Array(phiH + Q + (useTypes ? ted : 1) + efd);
          let o = 0;
          const pv = pool[h]; for (let k = 0; k < phiH; k++) inp[o++] = pv[k];
          const pr = prod[h]; for (let k = 0; k < Q; k++) inp[o++] = pr[k];
          if (useTypes) {                        // learned nonlinearity-type embedding (drops count)
            const row = typeEmbed[(input.he_type ? input.he_type[h] : 0)];
            for (let k = 0; k < ted; k++) inp[o++] = row[k];
          } else {
            inp[o++] = idx.counts[h];
          }
          const ef = he_feat[h]; for (let k = 0; k < efd; k++) inp[o++] = ef[k];
          const mm = mlp2(lay.psi, inp, false);
          const sc = he_scale[h];
          for (let k = 0; k < outDim; k++) mm[k] *= sc;
          msg[h] = mm;
        }
        // scatter to targets + self transform
        const newH = new Array(N);
        for (let node = 0; node < N; node++) {
          newH[node] = lay.self ? linear(lay.self.w, lay.self.b, hEmb[node])
                                : new Array(outDim).fill(0);
        }
        for (let h = 0; h < H; h++) {
          const t = he_tgt[h], mm = msg[h], dst = newH[t];
          for (let k = 0; k < outDim; k++) dst[k] += mm[k];
        }
        if (l < L - 1) for (let node = 0; node < N; node++) relu(newH[node]);
        hEmb = newH;
      }
      return hEmb;                           // N x Q = dX
    }

    // Integrate dX/dt = field(X) from X0 (broadcast x0 over Q) -> [steps][N][Q].
    function run(input) {
      const N = input.n, steps = input.steps, dt = input.dt;
      const substeps = cfg.substeps, solver = cfg.solver;
      const idx = makeIdx(input);
      const clone = (X) => X.map((r) => r.slice());
      // forcing / constant-input nodes (force_mask=1) are held at their initial value throughout
      // the rollout -- they have no incoming edges, so their field would otherwise drift.
      const forceIdx = [];
      const fm = input.force_mask;
      if (fm) for (let i = 0; i < N; i++) if (fm[i]) forceIdx.push(i);
      const cap = cfg.state_cap;                 // signed models: bound |x| each substep (else Inf/NaN)
      const clampInPlace = (X) => {
        if (cap) for (let i = 0; i < N; i++) {
          const xr = X[i];
          for (let q = 0; q < Q; q++) { if (xr[q] > cap) xr[q] = cap; else if (xr[q] < -cap) xr[q] = -cap; }
        }
        for (const i of forceIdx) { const v = input.x0[i], xr = X[i]; for (let q = 0; q < Q; q++) xr[q] = v; }
        return X;
      };
      const cf = (X) => field(clampInPlace(X), input, idx);   // field always sees clamped forcing nodes
      let x = new Array(N);
      for (let i = 0; i < N; i++) x[i] = new Array(Q).fill(input.x0[i]);
      const xs = [clone(x)];
      const h = dt / Math.max(1, substeps);
      const axpy = (X, a, D) => {            // X + a*D  (new array)
        const Y = new Array(N);
        for (let i = 0; i < N; i++) {
          const yr = new Array(Q), xr = X[i], dr = D[i];
          for (let q = 0; q < Q; q++) yr[q] = xr[q] + a * dr[q];
          Y[i] = yr;
        }
        return Y;
      };
      for (let s = 0; s < steps - 1; s++) {
        for (let ss = 0; ss < substeps; ss++) {
          if (solver === "rk4") {
            const k1 = cf(x);
            const k2 = cf(axpy(x, 0.5 * h, k1));
            const k3 = cf(axpy(x, 0.5 * h, k2));
            const k4 = cf(axpy(x, h, k3));
            const Y = new Array(N);
            for (let i = 0; i < N; i++) {
              const yr = new Array(Q), xr = x[i];
              for (let q = 0; q < Q; q++)
                yr[q] = xr[q] + (h / 6) * (k1[i][q] + 2 * k2[i][q] + 2 * k3[i][q] + k4[i][q]);
              Y[i] = yr;
            }
            x = clampInPlace(Y);
          } else {                           // explicit Euler
            x = clampInPlace(axpy(x, h, cf(x)));
          }
        }
        xs.push(clone(x));
      }
      return xs;                             // [steps][N][Q]
    }

    return { run, cfg, Q, taus: cfg.taus };
  }

  return { build, EPS };
});
