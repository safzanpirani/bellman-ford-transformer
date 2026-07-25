"""Registered pilot data uses the constructed encoder and exact round targets."""

import torch

from bf_transformer import Graph
from train import TRAIN_CONFIG
from train.data import collate_examples, make_example


def test_example_targets_and_hops_match_a_known_chain():
    graph = Graph(
        n=4,
        edges=((0, 1, 2.0), (1, 2, 3.0), (0, 2, 9.0)),
    )
    example = make_example(graph, source=0)
    batch = collate_examples([example])

    assert batch.streams.shape == (
        1,
        TRAIN_CONFIG.seq_len,
        example.stream.shape[-1],
    )
    assert batch.targets.shape == (1, TRAIN_CONFIG.rounds + 1, 4)
    assert torch.equal(
        batch.targets[0, :, 3],
        torch.full((TRAIN_CONFIG.rounds + 1,), TRAIN_CONFIG.inf_b),
    )
    assert batch.targets[0, 0].tolist() == [
        0.0,
        TRAIN_CONFIG.inf_b,
        TRAIN_CONFIG.inf_b,
        TRAIN_CONFIG.inf_b,
    ]
    assert batch.targets[0, 1].tolist() == [
        0.0,
        2.0,
        9.0,
        TRAIN_CONFIG.inf_b,
    ]
    assert batch.targets[0, 2].tolist() == [
        0.0,
        2.0,
        5.0,
        TRAIN_CONFIG.inf_b,
    ]
    assert example.hops == (0, 1, 2, -1)
