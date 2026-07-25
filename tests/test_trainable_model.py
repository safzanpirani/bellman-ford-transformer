"""G1: the trainable tied model contains the exact constructed circuit."""

import math

import torch
from torch import nn

from bf_transformer import BFTransformer, Config, Graph, encode_graph
from bf_transformer.verify import reference_states
from train import TRAIN_CONFIG, TrainableBFTransformer


def _reference_distance_tensor(
    g: Graph,
    source: int,
    cfg: Config,
    rounds: int,
) -> torch.Tensor:
    values = reference_states(g, source, rounds)[-1]
    return torch.tensor(
        [cfg.inf_b if math.isinf(value) else value for value in values],
        dtype=torch.float32,
    )


def test_constructed_weights_are_exactly_representable():
    cfg = Config(n_max=16)
    constructed = BFTransformer(cfg)
    trainable = TrainableBFTransformer(cfg=cfg, tied=True)
    trainable.load_constructed_weights(constructed)

    constructed_state = constructed.block.state_dict()
    trainable_state = trainable.block.state_dict()
    assert constructed_state.keys() == trainable_state.keys()
    assert {
        name: value.shape for name, value in constructed_state.items()
    } == {
        name: value.shape for name, value in trainable_state.items()
    }
    assert all(parameter.requires_grad for parameter in trainable.parameters())

    cases = [
        (Graph(n=2, edges=()), 0),
        (
            Graph(
                n=9,
                edges=tuple((node, node + 1, 1.0) for node in range(8)),
            ),
            0,
        ),
        (
            Graph(
                n=16,
                edges=(
                    (0, 1, 5.0),
                    (0, 1, 2.0),
                    (1, 7, 3.0),
                    (7, 15, 4.0),
                    (3, 3, 0.0),
                ),
            ),
            0,
        ),
    ]
    for graph, source in cases:
        stream = encode_graph(graph, source, cfg, constructed.reg)
        expected = constructed(stream.clone())
        actual = trainable(stream.clone())
        assert torch.equal(actual, expected)

        node_distances = actual[1 : 1 + graph.n, trainable.reg.idx("DIST")]
        target = _reference_distance_tensor(graph, source, cfg, cfg.rounds)
        assert torch.nn.functional.mse_loss(node_distances, target).item() == 0.0


def test_tied_and_untied_variants_have_identical_block_shapes_and_no_norm():
    cfg = TRAIN_CONFIG
    assert cfg.n_max == 16
    tied = TrainableBFTransformer(cfg=cfg, tied=True)
    untied = TrainableBFTransformer(cfg=cfg, tied=False)

    tied_shapes = {
        name: parameter.shape
        for name, parameter in tied.block.named_parameters()
    }
    assert len(untied.blocks) == cfg.rounds
    for block in untied.blocks:
        assert {
            name: parameter.shape
            for name, parameter in block.named_parameters()
        } == tied_shapes

    for model in (tied, untied):
        assert not any(
            isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.Dropout))
            for module in model.modules()
        )
