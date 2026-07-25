"""FR8 sanity: the soft reducer is drop-in swappable and converges to the
true distances as C_SOFT grows."""
import math
import random

from bf_transformer import BFTransformer, encode_graph, decode_dist, DEFAULT_CONFIG
from bf_transformer.verify import random_graph
from tests.test_full_model import nx_distances

cfg = DEFAULT_CONFIG


def test_soft_reducer_high_temperature_close_to_exact():
    model = BFTransformer(cfg, reducer="soft", c_soft=10.0)
    rng = random.Random(3)
    for _ in range(20):
        g, s = random_graph(rng, cfg)
        want = nx_distances(g, s)
        # soft-min needs extra rounds to settle; give it 2R
        got = decode_dist(model(encode_graph(g, s), rounds=2 * cfg.rounds), g.n)
        for i in range(g.n):
            if math.isinf(want[i]):
                assert got[i] >= cfg.inf_b - 1.0 or math.isinf(got[i])
            else:
                assert abs(got[i] - want[i]) < 1e-2, (g, s, i, want[i], got[i])


def test_soft_temperature_monotonically_improves():
    rng = random.Random(4)
    cases = [random_graph(rng, cfg) for _ in range(10)]

    def worst(c_soft):
        model = BFTransformer(cfg, reducer="soft", c_soft=c_soft)
        e = 0.0
        for g, s in cases:
            want = nx_distances(g, s)
            got = decode_dist(model(encode_graph(g, s), rounds=2 * cfg.rounds), g.n)
            for i in range(g.n):
                if not math.isinf(want[i]):
                    e = max(e, abs(got[i] - want[i]))
        return e

    assert worst(10.0) <= worst(2.0) <= worst(0.5)
