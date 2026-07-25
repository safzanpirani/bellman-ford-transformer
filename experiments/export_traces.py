"""Export everything the visualisation app renders, straight from the model.

No mock data: residual-stream snapshots, attention patterns, weight matrices,
reference Bellman-Ford states and the soft-vs-exact sweep are all read out of
the real hand-constructed transformer.

    python -m experiments.export_traces --out viz/public/trace.json
"""
import argparse
import json
import math
import random

import networkx as nx
import torch

from bf_transformer import BFTransformer, Graph, encode_graph, DEFAULT_CONFIG
from bf_transformer.blocks.attention import HandAttention
from bf_transformer.blocks.minred import HandMLP
from bf_transformer.registers import RegisterMap
from bf_transformer.verify import reference_states, random_graph

ROUND_TO = 4


def rnd(x: float) -> float:
    v = round(float(x), ROUND_TO)
    return 0.0 if v == 0 else v


def mat(t: torch.Tensor) -> list:
    return [[rnd(v) for v in row] for row in t.tolist()]


# --- curated demo graphs -------------------------------------------------
def demo_graphs(cfg):
    rng = random.Random(11)
    g_rand, s_rand = random_graph(rng, cfg, n=7)
    return [
        {
            "id": "chain",
            "name": "Detour chain",
            "blurb": "The shortest path to node 4 is the long way round — "
                     "distances must propagate one hop per round.",
            "graph": Graph(n=5, edges=((0, 1, 2.0), (1, 2, 1.0), (2, 3, 1.0),
                                       (3, 4, 1.0), (0, 4, 9.0))),
            "source": 0,
        },
        {
            "id": "diamond",
            "name": "Diamond",
            "blurb": "Two competing routes into node 3; the min gadget has to "
                     "pick the cheaper one in a single round.",
            "graph": Graph(n=4, edges=((0, 1, 1.0), (0, 2, 5.0), (1, 3, 7.0),
                                       (2, 3, 1.0), (1, 2, 2.0))),
            "source": 0,
        },
        {
            "id": "unreachable",
            "name": "Islands",
            "blurb": "Nodes 3 and 4 are unreachable — their DIST pins to INF_B "
                     "and the clamp keeps it from drifting.",
            "graph": Graph(n=5, edges=((0, 1, 4.0), (1, 2, 2.0), (2, 0, 1.0),
                                       (3, 4, 1.0), (4, 3, 1.0))),
            "source": 0,
        },
        {
            "id": "negcycle",
            "name": "Negative cycle",
            "blurb": "A reachable negative cycle: distances keep falling, and "
                     "the extra detection round lights the flag register.",
            "graph": Graph(n=4, edges=((0, 1, 3.0), (1, 2, -2.0), (2, 1, -1.0),
                                       (1, 3, 2.0))),
            "source": 0,
        },
        {
            "id": "random7",
            "name": "Random (n=7)",
            "blurb": "A seeded random graph with self-loops and parallel edges "
                     "— the kind the 1,000-graph property suite hammers.",
            "graph": g_rand,
            "source": s_rand,
        },
    ]


def nx_truth(g: Graph, s: int):
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(g.n))
    for (u, v, w) in g.edges:
        G.add_edge(u, v, weight=w)
    try:
        L = nx.single_source_bellman_ford_path_length(G, s)
        return [None if i not in L else rnd(L[i]) for i in range(g.n)], False
    except nx.NetworkXUnbounded:
        return [None] * g.n, True


def layout(n: int) -> list[dict]:
    """Circular layout; the app may re-run force layout but this is a sane base."""
    return [{"x": rnd(math.cos(2 * math.pi * i / n - math.pi / 2)),
             "y": rnd(math.sin(2 * math.pi * i / n - math.pi / 2))} for i in range(n)]


def export_registers(reg: RegisterMap):
    groups = {
        "ONES": "struct", "SINK": "struct", "IS_NODE": "struct", "IS_EDGE": "struct",
        "NODE_ID": "node", "DIST": "node",
        "EDGE_FROM": "edge", "EDGE_TO": "edge", "EDGE_W": "edge", "SLOT": "edge",
        "CAND": "scratch", "SCRATCH_SRC_DIST": "scratch", "SCRATCH_GATHER": "scratch",
        "SCRATCH_MIN": "scratch", "SCRATCH_SPARE": "scratch",
        "SCRATCH_PREV_DIST": "flag", "FLAG_DEC": "flag",
    }
    out = []
    for name in reg.names():
        s = reg.slice(name)
        out.append({"name": name, "start": s.start, "stop": s.stop,
                    "size": s.stop - s.start, "group": groups.get(name, "scratch")})
    return out


def export_tokens(g: Graph, source: int, cfg, reg):
    """Human-readable label per *real* token row (padding rows are dropped)."""
    toks = [{"row": 0, "kind": "sink", "label": "SINK"}]
    for i in range(g.n):
        toks.append({"row": 1 + i, "kind": "node", "label": f"n{i}",
                     "node": i, "isSource": i == source})
    for j, (u, v, w) in enumerate(g.edges):
        toks.append({"row": 1 + cfg.n_max + j, "kind": "edge",
                     "label": f"{u}→{v}", "u": u, "v": v, "w": rnd(w),
                     "edgeIndex": j})
    return toks


def export_layer_weights(model, reg):
    """Weight matrices of one round's sublayers (the block is weight-tied, so
    this is the entire model)."""
    out = []
    for name, layer in zip(model.block.layer_names, model.block.layers):
        if isinstance(layer, HandAttention):
            heads = []
            for h in range(len(layer.wq)):
                heads.append({
                    "wq": mat(layer.wq[h]), "wk": mat(layer.wk[h]),
                    "wv": mat(layer.wv[h].T), "wo": mat(layer.wo[h]),
                })
            out.append({"name": name, "kind": "attention", "heads": heads,
                        "dk": layer.wq[0].shape[1]})
        elif isinstance(layer, HandMLP):
            out.append({"name": name, "kind": "mlp",
                        "w1": mat(layer.w1), "b1": [rnd(v) for v in layer.b1.tolist()],
                        "w2": mat(layer.w2), "hidden": layer.w1.shape[1]})
    return out


def export_case(case, cfg, reg, model):
    g, s = case["graph"], case["source"]
    rows = [t["row"] for t in export_tokens(g, s, cfg, reg)]
    stream = encode_graph(g, s, cfg, reg)

    ref = reference_states(g, s, cfg.rounds)
    truth, unbounded = nx_truth(g, s)

    rounds = []
    x = stream
    for r in range(cfg.rounds):
        steps = [{"name": "input", "state": mat(x[rows]), "attn": None}]
        for name, layer, after in model.block.trace(x):
            entry = {"name": name, "state": mat(after[rows]), "attn": None}
            if isinstance(layer, HandAttention):
                entry["attn"] = [mat(layer.attention_pattern(x, head=h)[rows][:, rows])
                                 for h in range(len(layer.wq))]
            steps.append(entry)
            x = after
        rounds.append({
            "round": r + 1,
            "steps": steps,
            "reference": [None if math.isinf(v) else rnd(v) for v in ref[r + 1]],
        })

    # FR7: snapshot -> extra round -> flag
    xf = model.snapshot(x)
    xf = model.block(xf)
    xf = model.flag(xf)
    flags = [rnd(xf[1 + i, reg.idx("FLAG_DEC")].item()) for i in range(g.n)]

    return {
        "id": case["id"], "name": case["name"], "blurb": case["blurb"],
        "n": g.n, "source": s,
        "edges": [{"u": u, "v": v, "w": rnd(w)} for (u, v, w) in g.edges],
        "layout": layout(g.n),
        "tokens": export_tokens(g, s, cfg, reg),
        "initial": mat(stream[rows]),
        "rounds": rounds,
        "referenceInit": [None if math.isinf(v) else rnd(v) for v in ref[0]],
        "truth": truth,
        "negativeCycle": unbounded,
        "flags": flags,
        "flagged": max(flags) > 1e-3,
    }


def export_soft_sweep(cfg):
    """FR8: max error per round for the exact reducer and several C_SOFT."""
    rng = random.Random(0)
    graphs = [random_graph(rng, cfg) for _ in range(40)]
    reg = RegisterMap(cfg)
    streams0 = torch.stack([encode_graph(g, s, cfg, reg) for g, s in graphs])
    truth = [reference_states(g, s, cfg.rounds)[-1] for g, s in graphs]
    max_rounds = 2 * cfg.rounds

    def sweep(model):
        st = streams0.clone()
        errs = []
        for _ in range(max_rounds):
            st = model.apply_round(st)
            worst = 0.0
            for row, (g, s), tr in zip(st, graphs, truth):
                d = row[1:1 + g.n, reg.idx("DIST")]
                for i in range(g.n):
                    want = cfg.inf_b if math.isinf(tr[i]) else min(tr[i], cfg.inf_b)
                    worst = max(worst, abs(d[i].item() - want))
            errs.append(round(worst, 8))
        return errs

    series = [{"label": "exact (ReLU tournament)", "cSoft": None,
               "errors": sweep(BFTransformer(cfg))}]
    for c in [0.5, 1.0, 2.0, 5.0, 10.0]:
        series.append({"label": f"soft C_SOFT={c}", "cSoft": c,
                       "errors": sweep(BFTransformer(cfg, reducer="soft", c_soft=c))})
    return {"maxRounds": max_rounds, "series": series, "graphs": len(graphs)}


def export_attention_logits(cfg, reg, model):
    """Raw (pre-softmax) logit spread for the routing heads — the evidence that
    softmax is exactly one-hot rather than approximately so."""
    g = Graph(n=4, edges=((0, 1, 3.0), (2, 3, 5.0), (1, 2, 2.0)))
    stream = encode_graph(g, 0, cfg, reg)
    rows = [t["row"] for t in export_tokens(g, 0, cfg, reg)]
    head = model.block.layers[0]
    q = stream @ head.wq[0]
    k = stream @ head.wk[0]
    logits = (q @ k.T)[rows][:, rows]
    return {
        "labels": [t["label"] for t in export_tokens(g, 0, cfg, reg)],
        "logits": mat(logits),
        "cMatch": cfg.c_match,
        "underflowThreshold": 104.0,  # exp(-x) == 0 in float32 below this
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="viz/public/trace.json")
    args = ap.parse_args()

    cfg = DEFAULT_CONFIG
    reg = RegisterMap(cfg)
    model = BFTransformer(cfg)

    n_params = sum(p.numel() for p in model.parameters())
    payload = {
        "config": {
            "nMax": cfg.n_max, "wMax": cfg.w_max, "dMax": cfg.d_max,
            "infB": cfg.inf_b, "cHard": cfg.c_hard, "cMatch": cfg.c_match,
            "rounds": cfg.rounds, "seqLen": cfg.seq_len, "eMax": cfg.e_max,
            "dModel": reg.d_model, "kPush": cfg.k_push,
        },
        "stats": {
            "params": n_params,
            "trainedParams": 0,
            "sublayers": len(model.block.layers),
            "layerNames": list(model.block.layer_names),
        },
        "registers": export_registers(reg),
        "weights": export_layer_weights(model, reg),
        "logitEvidence": export_attention_logits(cfg, reg, model),
        "cases": [export_case(c, cfg, reg, model) for c in demo_graphs(cfg)],
        "softSweep": export_soft_sweep(cfg),
    }

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    size = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out} ({size:.1f} MB), {len(payload['cases'])} cases, "
          f"{n_params} parameter tensors' worth of weights")


if __name__ == "__main__":
    main()
