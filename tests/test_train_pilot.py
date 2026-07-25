"""Loss and G2 metrics follow the frozen pre-registration."""

import torch

from bf_transformer import Graph
from train import TrainableBFTransformer
from train.data import collate_examples, make_example
from train.pilot import behaviour_metrics, supervised_loss


def test_constructed_solution_has_zero_final_and_round_loss():
    examples = [
        make_example(
            Graph(n=3, edges=((0, 1, 2.0), (1, 2, 3.0))),
            source=0,
        ),
        make_example(Graph(n=2, edges=()), source=1),
    ]
    batch = collate_examples(examples)
    model = TrainableBFTransformer(
        tied=True,
        random_initialisation=False,
    )
    states = model.forward_with_states(batch.streams)

    assert supervised_loss(states, batch, supervision="final").item() == 0.0
    assert supervised_loss(states, batch, supervision="round").item() == 0.0


def test_exact_match_uses_registered_half_unit_threshold():
    target = torch.tensor([[0.0, 2.0, 161.0], [0.0, 3.0, 4.0]])
    prediction = torch.tensor([[0.49, 2.49, 161.0], [0.0, 3.5, 4.0]])
    mask = torch.ones_like(target, dtype=torch.bool)

    metrics = behaviour_metrics(prediction, target, mask)

    assert metrics["node_exact"] == 5 / 6
    assert metrics["graph_exact"] == 0.5
