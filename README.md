# HyperODE

**[▶ Live demo](https://scc-usc.github.io/hyperode-web/) · [📄 Paper](https://arxiv.org/abs/2608.00852)**

Give it an ODE with uncertain parameters; it returns the predicted **trajectory
quantiles** (5 / 25 / 50 / 75 / 95%) for every state — in a **single forward pass**,
with no step-by-step ODE solving. Everything runs locally in your browser; your
equations never leave your device.

The engine is a pretrained **quantile neural-ODE surrogate over a typed hypergraph**
representation of the system. Type an ODE (or generate one from a plain-English
description with the built-in LLM prompt), pick a model, and read off the uncertainty
bands.

## Run locally

It's a static site — no build step, no server needed. Any static file server works:

```bash
# from this folder
python -m http.server 8000     # then open http://localhost:8000
```

Opening `index.html` directly over `file://` also works in most browsers.

## Features

- **23 built-in presets** — SIR/SEIR/SIS variants, Lotka–Volterra, Rosenzweig–MacArthur,
  Brusselator, Van der Pol, FitzHugh–Nagumo, Michaelis–Menten, Hill/saturating incidence, …
- **Scale *k*** — expand a compartmental preset into *k* ring-coupled sub-populations
  (meta-population).
- **LLM prompt input** — copy the built-in prompt into any LLM, describe your system in
  words, paste the returned spec back, and load it.
- **Compare vs Monte-Carlo** — integrate the *true* ODE (exact RHS, RK4) over parameter
  draws and overlay its empirical quantiles as an honest ground-truth check.
- **Non-polynomial terms** — `sin`, `exp`, fractional powers, saturating/Hill/Ricker
  incidence, and denominators are parsed and driven zero-shot (no retraining).

## Models

| model | width | layers | notes |
|-------|------:|------:|-------|
| Small | 16 | 2 | smallest |
| Medium | 32 | 3 | default |
| Large | 128 | 3 | largest capacity |
| Medium: long horizon | 32 | 2 | trained on horizons t = 8–100 (state clamped to ±20); use for t > 8. 90% bands cover ~0.85–0.93 out to t = 96, vs 0.20 for the original h32; slightly narrower bands for t ≤ 8 |
| Diverse (broad data) | 32 | 3 | typed + signed; handles non-mass-action nonlinearities and negative/unbounded states |

## Scope & caveats

- Best accuracy is on epidemic-like and mass-action systems near the training
  distribution; exotic ODEs extrapolate.
- Quantiles are sorted for display, and clamped ≥ 0 for the non-signed models (whose
  states are non-negative). The Diverse/signed model is allowed to emit negative values.

## Source & training

This repository hosts the built demo. The model training, data generation, and export
pipeline live in the main project repository, **[scc-usc/hyperODE](https://github.com/scc-usc/hyperODE)**.
