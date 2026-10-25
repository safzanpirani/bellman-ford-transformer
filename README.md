# Hand-Coded Bellman-Ford Transformer

A transformer whose weights are constructed by hand — no training anywhere —
such that, within the bounded configuration described below, a forward pass
executes the Bellman-Ford single-source shortest-path algorithm exactly on
integer-weight graphs. One transformer block is one relaxation round; the block
is weight-tied and applied `N_MAX − 1` times.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/            # incl. the 1,000-graph property suite
.venv/bin/python -m bf_transformer.verify    # FR-V: executable induction proof
.venv/bin/python -m experiments.soft_vs_exact  # FR8: exact vs soft-min plot
```

## Interactive circuit tracer

A React/Vite app renders the model's actual execution — residual stream per
sublayer, attention patterns, every weight matrix, convergence vs. the
reference, and the soft-vs-exact study. All of it is exported from real forward
passes; nothing is mocked.

```bash
.venv/bin/python -m experiments.export_traces --out viz/public/trace.json
cd viz && npm install && npm run dev
```

See [`viz/README.md`](viz/README.md) for what each panel reads.

## Using the model

```python
from bf_transformer import BFTransformer, Graph, encode_graph, decode_dist

g = Graph(n=4, edges=((0, 1, 3.0), (1, 2, 2.0), (0, 2, 9.0)))
model = BFTransformer()
dist = decode_dist(model(encode_graph(g, source=0)), g.n)
# {0: 0.0, 1: 3.0, 2: 5.0, 3: inf}
```

## Paper

`docs/paper.md` is the prose source of truth; `paper/main.tex` typesets it.
The body compiles in two typographic variants, selected by `\paperstyle`:

```bash
make -C paper           # Times, navy headings, running head (default)
make -C paper lmodern   # Latin Modern, plain article, no colour
make -C paper both
```

Each variant is a pair of files — `style-<variant>.tex` (packages, geometry,
spacing, colours) and `titleblock-<variant>.tex` — and `main.tex` inputs them
by name, so the two builds share one body with no forked copy. Both style
files must define every command the body uses, including the colours the TikZ
architecture figure references; the Latin Modern one redefines them as grays.
Builds run under separate `-jobname`s so their `.aux`/`.bbl` never collide.
Output lands in `output/pdf/`. The Latin Modern variant runs about a page
longer, since Computer Modern sets wider than Times at the same nominal size.

## Prior art and scope

Constructing transformers that execute programs is established by
[RASP (arXiv:2106.06981)](https://arxiv.org/abs/2106.06981) and
[Tracr (arXiv:2301.05062)](https://arxiv.org/abs/2301.05062).
[Simulation of Graph Algorithms with Looped Transformers
(arXiv:2402.01107)](https://arxiv.org/abs/2402.01107) constructs looped
transformers for graph algorithms and discusses their finite-precision limits.
This repository does not claim novelty for hand-compiling Bellman-Ford into a
transformer. Its differentiators are narrower: explicit bit-exactness within a
declared implementation regime, an executable round-by-round verifier, and a
tracer exposing the complete residual state and routing.

### Four exactness notions that must not be conflated

1. **Real-arithmetic softmax.** At finite logits every attention weight is
   strictly positive. A softmax-weighted average therefore never implements an
   exact hard selection among unequal values.
2. **Approximate sharpness.** To keep selection error at most
   \(\varepsilon\) as the candidate count grows, the required score spread is
   \(\Theta(\log(n/\varepsilon))\). This size-dispersion result is prior art:
   [*Softmax is not Enough (for Sharp Size Generalisation)*
   (arXiv:2410.01104)](https://arxiv.org/abs/2410.01104).
3. **This float32 construction.** Its non-target exponentials underflow to
   exactly zero at a fixed score gap. Once they are zero, increasing \(n\)
   does not require a larger gap. This is an implementation property, not a
   theorem: it depends on dtype, the softmax kernel, subnormal handling, and
   flush-to-zero behaviour.
4. **ReLU reduction.** The identity
   `min(a, b) = a − ReLU(a − b)` is exact with constant-sized coefficients.
   Reducing many candidates costs sequential or tree depth rather than growing
   softmax score precision.

## How the construction works

### Input encoding

One token per node, one token per directed edge, plus a **sink token**
(token 0) with an all-zero payload. Distances are initialised to `0` at the
source and to a bounded infinity `INF_B = N_MAX·W_MAX + 1` elsewhere —
provably larger than any real path length, comfortably inside float32's
exact-integer range.

### The register map

The `d_model` axis is partitioned into named, disjoint slices
(`registers.py`), e.g. `NODE_ID` (one-hot identity), `DIST` (current
distance), `EDGE_FROM` / `EDGE_TO` / `EDGE_W`, `SLOT` (which in-edge slot of
its destination this edge occupies), and scratch registers that every round
must leave zeroed. All weight construction addresses the stream through
`reg.slice(name)` — raw indices are banned by a lint test, so changing
`N_MAX` requires zero changes outside `config.py`/`registers.py`.

### One round = one block

1. **Edge-reads-source head.** Each edge token attends (exactly one-hot, see
   below) to the node matching its `EDGE_FROM` and copies that node's `DIST`
   into scratch. An MLP then computes `CAND = dist[u] + w`, with an INF
   guard: if `dist[u]` is at `INF_B`, the candidate is pushed above `INF_B`
   and clamped back to exactly `INF_B`, so unreachable sources can never leak
   finite-looking candidates (this is what makes negative weights safe).
2. **Gather heads.** `D_MAX` heads; head *k* lets node *v* read `CAND` from
   its in-edge in slot *k* (slots are assigned per destination at encode
   time, so each head sees at most one match). If a node has no edge in slot
   *k*, the head falls back to attending to the node itself and returns
   `dist[v]` — a no-op under min.
3. **The min gadget.** Exact minimum via the ReLU identity
   `min(a, b) = a − ReLU(a − b)`, folded sequentially over the `D_MAX`
   gathered candidates: `m ← dist[v]`, then `m ← m − ReLU(m − c_k)`. Exact
   for integer-valued floats; ≤ 1 ulp for arbitrary floats.
4. **Delta write-back.** The residual stream is additive, so the block never
   overwrites: it adds `−ReLU(dist[v] − m)` into `DIST` and adds the negation
   of every scratch register (an exact ReLU-pair identity,
   `x = ReLU(x) − ReLU(−x)`), leaving the stream ready for the next tied
   application of the same block. A converged state is a bit-exact fixed
   point — tested.

### Why attention is bit-exact in the tested float32 implementation

Every attention head is designed so the intended target's logit exceeds all
competitors' by at least `C_HARD/4 = 2500`. After softmax's internal
max-subtraction, every off-target weight is `exp(−gap)` with `gap ≥ 2500`,
which **underflows float32 to exactly 0.0** — so attention patterns are
exactly one-hot and value copies are float-exact, not merely close. Tokens
with nothing to read (e.g. node tokens during the edge-reads-source head)
fall back to the sink token and copy a clean zero. This is legitimate
because every match in the construction is exact-or-nothing. It is not a
claim about real-arithmetic softmax or arbitrary hardware.

### Why there is no LayerNorm

LayerNorm rescales the residual stream per token, which would destroy the
literal arithmetic the registers carry: `DIST` holds an actual distance, and
the min gadget and one-hot routing rely on absolute magnitudes (logit gaps,
the `INF_B` threshold). Normalisation would also couple unrelated registers
through the shared mean/variance. Since no weights are trained, none of
LayerNorm's optimisation benefits apply — so it is simply omitted, along
with dropout and positional embeddings (all position information lives in
the registers).

### Negative-cycle detection (FR7)

After the `N_MAX − 1` standard rounds, the model snapshots `DIST`, runs one
extra round, and writes `ReLU(prev − dist)` into a flag register per node.
Any hot flag ⇔ some distance decreased in round N ⇔ a negative cycle is
reachable from the source — matching networkx's `NetworkXUnbounded`
behaviour (tested on random negative-weight graphs).

### Soft-min variant (FR8, v2)

A drop-in alternative reducer replaces the gather heads + ReLU tournament
with a single attention head whose logits include `−C_SOFT · value`: within
node *v*'s candidate set the softmax computes weights `∝ exp(−C_SOFT·val)` —
a soft minimum (routing into the candidate set is still exact). The
experiment `experiments/soft_vs_exact.py` sweeps `C_SOFT` and round count;
the soft reducer plateaus at a temperature-dependent error floor (~2.5 at
`C_SOFT=0.5`, ~1e-4 at `C_SOFT=10`) while the exact reducer is at zero error
every round.

## Guarantees & limits

- **Tested exactness:** integer-weight graphs decode bit-exactly equal to
  `networkx.single_source_bellman_ford_path_length`; float weights within
  `1e-4`. Verified by a 1,000-graph property suite plus hypothesis.
- **FR-V:** `verify.py` checks the model state against a reference
  synchronous Bellman-Ford *after every round* — the induction proof, made
  executable.
- **Zero trained parameters:** every tensor is produced by a construction
  function with `requires_grad=False` (tested).
- **Limits:** graphs with `n ≤ N_MAX` (default 8), `|w| ≤ W_MAX` (10), and
  in-degree `≤ D_MAX` (8, counting self-loops and parallel edges). The
  encoder rejects anything larger loudly; nothing is silently truncated.
- **Empirical break points** (`python -m experiments.stress_bounds`, Config
  guards bypassed, 30 random non-negative integer graphs, n ≤ 8, float32 CPU;
  raw output in `experiments/stress_bounds.txt`):
  - *Logit scale:* exact at `c_hard/4 ≥ 32` (`c_hard ≥ 128`); at 16 only 33%
    of graphs are bit-exact (err ~5e-5), at 4 err 0.24, at 2 err 55. The
    `c_hard/4 > 200` guard is ~6× conservative.
  - *Weight magnitude:* exact up to `w_max = 2e6` (`INF_B = 1.6e7 < 2²⁴`);
    at `w_max = 4e6` (`INF_B = 3.2e7 > 2²⁴`) only 40% exact, error 2 — the
    float32 integer-mantissa limit. The `INF_B + W_MAX < 1e6` guard is ~16×
    conservative; the break is consistent with `INF_B + W_MAX ≤ 2²⁴`.
