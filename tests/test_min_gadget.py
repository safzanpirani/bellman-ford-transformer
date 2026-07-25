"""M2: the ReLU min gadget — exact on pairs and on D_MAX-way folds."""
import random

import torch

from bf_transformer import DEFAULT_CONFIG
from bf_transformer.registers import RegisterMap
from bf_transformer.blocks.minred import build_min_fold_layer, build_copy_layer
from bf_transformer.blocks.round import build_min_init, build_min_step

cfg = DEFAULT_CONFIG
reg = RegisterMap(cfg)
A = reg.idx_in("SCRATCH_SPARE", 0)
B = reg.idx_in("SCRATCH_SPARE", 1)


def stream_with(**kv):
    x = torch.zeros(3, reg.d_model)
    for idx, val in kv.items():
        x[1, int(idx)] = val
    return x


def test_pairwise_min_exact():
    layer = build_min_fold_layer(reg.d_model, A, B)
    rng = random.Random(0)
    # integer-valued inputs (incl. equal values and INF): bit-exact
    int_cases = [(0.0, 0.0), (5.0, 5.0), (cfg.inf_b, cfg.inf_b), (cfg.inf_b, 3.0),
                 (3.0, cfg.inf_b), (-4.0, 2.0), (2.0, -4.0), (0.0, cfg.inf_b)]
    int_cases += [(float(rng.randint(-20, int(cfg.inf_b))),
                   float(rng.randint(-20, int(cfg.inf_b)))) for _ in range(500)]
    # arbitrary floats: min(a,b) = a - ReLU(a-b) carries <= 1 ulp from the
    # subtraction; the PRD float tolerance is 1e-4, we hold it to 1e-5 here
    float_cases = [(rng.uniform(-20, cfg.inf_b), rng.uniform(-20, cfg.inf_b))
                   for _ in range(500)]
    for cases, tol in ((int_cases, 0.0), (float_cases, 1e-5)):
        for a, b in cases:
            x = torch.zeros(1, reg.d_model)
            x[0, A], x[0, B] = a, b
            a32, b32 = x[0, A].item(), x[0, B].item()  # float32-representable
            y = layer(x)
            assert abs(y[0, A].item() - min(a32, b32)) <= tol, (a, b)
            assert y[0, B].item() == b32  # source operand untouched


def test_copy_layer_exact_both_signs():
    layer = build_copy_layer(reg.d_model, A, B)
    for v in (0.0, 7.5, -7.5, cfg.inf_b, -cfg.inf_b):
        x = torch.zeros(1, reg.d_model)
        x[0, A] = v
        y = layer(x)
        assert y[0, B].item() == v
        assert y[0, A].item() == v


def test_tournament_fold_up_to_d_max():
    """min over DIST plus up to D_MAX gathered values, exactly."""
    init = build_min_init(reg)
    steps = [build_min_step(reg, k) for k in range(cfg.d_max)]
    rng = random.Random(1)
    for _ in range(200):
        vals = [rng.choice([cfg.inf_b, float(rng.randint(0, int(cfg.inf_b)))])
                for _ in range(cfg.d_max)]
        dist = rng.choice([cfg.inf_b, float(rng.randint(0, int(cfg.inf_b)))])
        x = torch.zeros(1, reg.d_model)
        x[0, reg.idx("DIST")] = dist
        for k, v in enumerate(vals):
            x[0, reg.idx_in("SCRATCH_GATHER", k)] = v
        y = init(x)
        for s in steps:
            y = s(y)
        assert y[0, reg.idx("SCRATCH_MIN")].item() == min([dist] + vals)
