"""Fixed graph corpus for the registered recurrence pilot."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from bf_transformer import Config, Graph, encode_graph
from bf_transformer.verify import random_graph, reference_states

from .config import TRAIN_CONFIG


DATA_SEED = 20260726
SPLIT_SIZES = {
    "train": 4096,
    "validation": 512,
    "id_test": 1024,
    "ood_test": 1024,
    "ood_probe_fit": 1024,
    "ood_probe_test": 1024,
}


@dataclass(frozen=True)
class Example:
    graph: Graph
    source: int
    stream: torch.Tensor
    targets: torch.Tensor
    hops: tuple[int, ...]


@dataclass(frozen=True)
class Batch:
    streams: torch.Tensor
    targets: torch.Tensor
    node_mask: torch.Tensor
    node_counts: torch.Tensor
    hops: torch.Tensor

    def to(self, device: torch.device | str) -> "Batch":
        return Batch(
            streams=self.streams.to(device),
            targets=self.targets.to(device),
            node_mask=self.node_mask.to(device),
            node_counts=self.node_counts.to(device),
            hops=self.hops.to(device),
        )


def _shortest_path_hops(graph: Graph, source: int) -> tuple[int, ...]:
    """Minimum hop count among minimum-weight paths; -1 means unreachable."""
    distances = [math.inf] * graph.n
    hops = [math.inf] * graph.n
    distances[source] = 0.0
    hops[source] = 0
    for _ in range(graph.n - 1):
        next_distances = list(distances)
        next_hops = list(hops)
        for u, v, weight in graph.edges:
            if math.isinf(distances[u]):
                continue
            candidate = distances[u] + weight
            candidate_hops = hops[u] + 1
            if (
                candidate < next_distances[v]
                or (
                    candidate == next_distances[v]
                    and candidate_hops < next_hops[v]
                )
            ):
                next_distances[v] = candidate
                next_hops[v] = candidate_hops
        distances, hops = next_distances, next_hops
    return tuple(-1 if math.isinf(value) else int(value) for value in hops)


def make_example(
    graph: Graph,
    source: int,
    cfg: Config = TRAIN_CONFIG,
) -> Example:
    states = reference_states(graph, source, cfg.rounds)
    bounded = [
        [cfg.inf_b if math.isinf(value) else value for value in state]
        for state in states
    ]
    return Example(
        graph=graph,
        source=source,
        stream=encode_graph(graph, source, cfg),
        targets=torch.tensor(bounded, dtype=torch.float32),
        hops=_shortest_path_hops(graph, source),
    )


def collate_examples(examples: list[Example]) -> Batch:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    max_nodes = max(example.graph.n for example in examples)
    batch_size = len(examples)
    targets = torch.full(
        (batch_size, TRAIN_CONFIG.rounds + 1, max_nodes),
        TRAIN_CONFIG.inf_b,
        dtype=torch.float32,
    )
    mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    hops = torch.full((batch_size, max_nodes), -1, dtype=torch.int64)
    for row, example in enumerate(examples):
        n = example.graph.n
        targets[row, :, :n] = example.targets
        mask[row, :n] = True
        hops[row, :n] = torch.tensor(example.hops, dtype=torch.int64)
    return Batch(
        streams=torch.stack([example.stream for example in examples]),
        targets=targets,
        node_mask=mask,
        node_counts=torch.tensor(
            [example.graph.n for example in examples],
            dtype=torch.int64,
        ),
        hops=hops,
    )


def _graph_key(graph: Graph, source: int) -> tuple:
    return graph.n, source, tuple(sorted(graph.edges))


def _generate_split(
    rng: random.Random,
    *,
    count: int,
    node_schedule: list[int],
    accepted_offset: int,
    seen: set[tuple],
) -> list[Example]:
    dense_cfg = TRAIN_CONFIG
    sparse_cfg = Config(
        n_max=TRAIN_CONFIG.n_max,
        d_max=2,
        w_max=TRAIN_CONFIG.w_max,
        c_hard=TRAIN_CONFIG.c_hard,
    )
    examples: list[Example] = []
    while len(examples) < count:
        accepted_index = accepted_offset + len(examples)
        generator_cfg = dense_cfg if accepted_index % 2 == 0 else sparse_cfg
        n = node_schedule[len(examples)]
        graph, source = random_graph(
            rng,
            cfg=generator_cfg,
            n=n,
            integer_weights=True,
            allow_negative=False,
        )
        key = _graph_key(graph, source)
        if key in seen:
            continue
        seen.add(key)
        examples.append(make_example(graph, source))
    return examples


def generate_registered_corpus() -> dict[str, list[Example]]:
    rng = random.Random(DATA_SEED)
    seen: set[tuple] = set()
    corpus: dict[str, list[Example]] = {}
    accepted = 0
    for split in ("train", "validation", "id_test"):
        count = SPLIT_SIZES[split]
        schedule = [rng.randint(2, 8) for _ in range(count)]
        corpus[split] = _generate_split(
            rng,
            count=count,
            node_schedule=schedule,
            accepted_offset=accepted,
            seen=seen,
        )
        accepted += count
    ood_schedule = [
        n
        for n in range(9, 17)
        for _ in range(128)
    ]
    for split in ("ood_test", "ood_probe_fit", "ood_probe_test"):
        corpus[split] = _generate_split(
            rng,
            count=SPLIT_SIZES[split],
            node_schedule=ood_schedule,
            accepted_offset=accepted,
            seen=seen,
        )
        accepted += SPLIT_SIZES[split]
    return corpus


def _serialise_example(example: Example) -> dict:
    return {
        "n": example.graph.n,
        "edges": [list(edge) for edge in example.graph.edges],
        "source": example.source,
        "hops": list(example.hops),
    }


def save_corpus(
    corpus: dict[str, list[Example]],
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        split: [_serialise_example(example) for example in examples]
        for split, examples in corpus.items()
    }
    corpus_bytes = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(corpus_bytes).hexdigest()
    (output_dir / "corpus.json").write_bytes(corpus_bytes)
    manifest = {
        "data_seed": DATA_SEED,
        "sha256": digest,
        "split_sizes": {
            split: len(examples) for split, examples in corpus.items()
        },
        "node_ranges": {
            split: [
                min(example.graph.n for example in examples),
                max(example.graph.n for example in examples),
            ]
            for split, examples in corpus.items()
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def load_corpus(input_dir: Path) -> dict[str, list[Example]]:
    corpus_bytes = (input_dir / "corpus.json").read_bytes()
    manifest = json.loads((input_dir / "manifest.json").read_text())
    actual_digest = hashlib.sha256(corpus_bytes).hexdigest()
    if actual_digest != manifest["sha256"]:
        raise ValueError(
            f"corpus digest mismatch: {actual_digest} != {manifest['sha256']}"
        )
    payload = json.loads(corpus_bytes)
    corpus = {}
    for split, records in payload.items():
        corpus[split] = [
            make_example(
                Graph(
                    n=record["n"],
                    edges=tuple(
                        (int(u), int(v), float(weight))
                        for u, v, weight in record["edges"]
                    ),
                ),
                int(record["source"]),
            )
            for record in records
        ]
    return corpus
