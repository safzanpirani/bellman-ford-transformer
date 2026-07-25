# Circuit tracer — visualisations

An interactive trace of the hand-constructed Bellman-Ford transformer.
React + Vite + Tailwind v4, no charting library — every plot is hand-drawn SVG.

## Running

```bash
# 1. export the traces from the real model (writes viz/public/trace.json, ~2 MB)
cd .. && .venv/bin/python -m experiments.export_traces --out viz/public/trace.json

# 2. run the app
cd viz && npm install && npm run dev
```

`npm run build` produces a static bundle in `dist/`.

## Where the data comes from

Nothing on the page is illustrative or recomputed in the browser.
`experiments/export_traces.py` runs real forward passes and dumps:

| Export | Used by |
|---|---|
| residual stream after **every sublayer of every round**, per graph | graph canvas, residual heatmap, convergence plot |
| attention patterns for all 9 routing heads, per round | attention matrices |
| pre-softmax logits for the edge-reads-source head | one-hot evidence panel |
| every `W_Q/W_K/W_V/W_O`, `W_1/b_1/W_2` in the block | weight inspector |
| reference synchronous Bellman-Ford state per round | "matches reference" badge |
| networkx ground truth + negative-cycle flags | truth row, FR7 panel |
| soft-vs-exact error sweep over 40 graphs | FR8 chart |

The browser only *reads* `DIST` out of the exported stream — it never runs the
algorithm itself, so a disagreement between the graph view and the reference
badge would be a real defect, not a rendering artefact.

## The shared playhead

A single `(round, sublayer)` transport drives every panel at once. Round 0 is the
encoder output; rounds 1..7 each expose their 14 sublayers. Because `DIST` is
only written by the last sublayer (the delta write-back), the reference badge
compares against the *previous* round's state until the write-back executes —
that is expected, not drift.

## Structure

```
src/
  types.ts                  shape of trace.json
  lib/theme.ts              colour ramps, register grouping, per-layer prose
  lib/useTrace.ts           data loading, register lookups, the transport hook
  components/
    Chrome.tsx              panel / stat / tag primitives
    Transport.tsx           the sticky playhead bar
    GraphCanvas.tsx         graph with live distances + winning-edge highlight
    ResidualStream.tsx      tokens × d_model heatmap, register-banded
    AttentionView.tsx       attention patterns + the logit-gap evidence
    CircuitDiagram.tsx      sublayer chain with per-layer read/write wiring
    MinGadget.tsx           draggable min(a,b) = a - ReLU(a - b) explainer
    WeightInspector.tsx     every matrix in the (weight-tied) model
    Charts.tsx              convergence + soft-vs-exact plots
    RegisterRibbon.tsx      the d_model map, annotated
```

## Notes

- Tailwind v4 is wired through **PostCSS** (`@tailwindcss/postcss`), not
  `@tailwindcss/vite` — the Vite plugin does not transform CSS under Vite 8 and
  silently emits unprocessed `@tailwind utilities`.
- Fonts (IBM Plex Mono / Instrument Serif) load from Google Fonts; the page
  degrades to system mono/serif offline.
- Respects `prefers-reduced-motion`.
