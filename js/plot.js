/* AXIOM — minimal dependency-free quantile-band plotting on canvas. */
(function (root, factory) {
  const m = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = m;
  else root.AXIOMPlot = m;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const PAL = ["#4f9dff", "#ff7a59", "#40c9a2", "#c77dff", "#ffd166", "#f45b69",
               "#7bdff2", "#b8f2b8"];

  function niceCeil(x) {
    if (x <= 0) return 1;
    const e = Math.pow(10, Math.floor(Math.log10(x)));
    const f = x / e;
    const steps = [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10];
    return (steps.find((s) => f <= s + 1e-9) || 10) * e;
  }

  // pred: [steps][N][Q]; draw one state (index si) into canvas. mcPred (optional,
  // same shape) is overlaid as dashed Monte-Carlo reference lines.
  function drawState(canvas, tGrid, pred, si, name, taus, color, mcPred, single) {
    const dpr = window.devicePixelRatio || 1;
    const cw = canvas.clientWidth, ch = canvas.clientHeight;
    canvas.width = cw * dpr; canvas.height = ch * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);

    const Q = taus.length, steps = pred.length;
    const nearest = (p) => taus.reduce((b, t, i) => Math.abs(t - p) < Math.abs(taus[b] - p) ? i : b, 0);
    const mid = nearest(0.5), iLo = nearest(0.25), iHi = nearest(0.75);   // MC shows median + 50% interval

    let vmin = 0, vmax = 0;
    for (let t = 0; t < steps; t++) {
      for (let q = 0; q < Q; q++) { const v = pred[t][si][q]; if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
      if (mcPred) for (const qi of [iLo, iHi, mid]) { const v = mcPred[t][si][qi]; if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
      if (single) { const v = single[t][si]; if (isFinite(v)) { if (v < vmin) vmin = v; if (v > vmax) vmax = v; } }
    }
    vmax = niceCeil(vmax || 1);
    vmin = vmin < 0 ? -niceCeil(-vmin) : 0;    // include 0; extend below only if states go negative (signed models)
    const span = (vmax - vmin) || 1;
    const tmax = tGrid[tGrid.length - 1] || 1;

    const m = { l: 46, r: 12, t: 22, b: 28 };
    const W = cw - m.l - m.r, H = ch - m.t - m.b;
    const X = (tt) => m.l + (tt / tmax) * W;
    const Y = (v) => m.t + H - ((v - vmin) / span) * H;

    // grid + axes
    ctx.strokeStyle = "rgba(255,255,255,0.07)"; ctx.fillStyle = "#8aa0b6";
    ctx.font = "10px ui-monospace, monospace"; ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {
      const v = vmin + (span * g) / 4, y = Y(v);
      ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(cw - m.r, y); ctx.stroke();
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      ctx.fillText(v.toPrecision(2), m.l - 5, y);
    }
    if (vmin < 0) {                            // emphasize the zero baseline for signed states
      const y0 = Y(0); ctx.save(); ctx.strokeStyle = "rgba(255,255,255,0.18)";
      ctx.beginPath(); ctx.moveTo(m.l, y0); ctx.lineTo(cw - m.r, y0); ctx.stroke(); ctx.restore();
    }
    for (let g = 0; g <= 4; g++) {
      const tt = (tmax * g) / 4, x = X(tt);
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      ctx.fillText(tt.toPrecision(2), x, ch - m.b + 6);
    }

    const hexA = (hex, a) => {
      const n = parseInt(hex.slice(1), 16);
      return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`;
    };
    // symmetric band fills from outermost to innermost quantile pairs
    const bandPairs = [];
    for (let q = 0; q < Math.floor(Q / 2); q++) bandPairs.push([q, Q - 1 - q]);
    bandPairs.forEach(([lo, hi], k) => {
      ctx.beginPath();
      for (let t = 0; t < steps; t++) { const x = X(tGrid[t]); (t ? ctx.lineTo : ctx.moveTo).call(ctx, x, Y(pred[t][si][hi])); }
      for (let t = steps - 1; t >= 0; t--) { const x = X(tGrid[t]); ctx.lineTo(x, Y(pred[t][si][lo])); }
      ctx.closePath();
      ctx.fillStyle = hexA(color, 0.12 + 0.12 * k);
      ctx.fill();
    });
    // median line
    ctx.beginPath();
    for (let t = 0; t < steps; t++) { const x = X(tGrid[t]); (t ? ctx.lineTo : ctx.moveTo).call(ctx, x, Y(pred[t][si][mid])); }
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();

    // Monte-Carlo overlay: dashed median + 50% interval (q25, q75)
    if (mcPred) {
      ctx.save(); ctx.strokeStyle = "rgba(230,238,247,0.9)";
      const line = (qi, w, dash) => {
        ctx.setLineDash(dash); ctx.lineWidth = w; ctx.beginPath();
        for (let t = 0; t < steps; t++) { const x = X(tGrid[t]); (t ? ctx.lineTo : ctx.moveTo).call(ctx, x, Y(mcPred[t][si][qi])); }
        ctx.stroke();
      };
      line(iLo, 1, [3, 3]);
      line(iHi, 1, [3, 3]);
      line(mid, 1.75, [5, 3]);
      ctx.restore();
    }

    // single solve at median theta: dotted amber line (contrast the median-of-solves)
    if (single) {
      ctx.save(); ctx.strokeStyle = "#ffd166"; ctx.setLineDash([2, 3]); ctx.lineWidth = 1.75;
      ctx.beginPath();
      for (let t = 0; t < steps; t++) { const x = X(tGrid[t]); (t ? ctx.lineTo : ctx.moveTo).call(ctx, x, Y(single[t][si])); }
      ctx.stroke(); ctx.restore();
    }

    // title
    ctx.fillStyle = "#e6eef7"; ctx.font = "600 12px ui-sans-serif, system-ui";
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
    ctx.fillText(name, m.l, 14);
  }

  return { drawState, PAL };
});
