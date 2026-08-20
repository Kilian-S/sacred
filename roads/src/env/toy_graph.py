"""Deterministic toy graph for the first SACRED adversarial testbed."""

from __future__ import annotations

from typing import Any

from src.env.graph_env import GraphEnv


TOY_NODES: dict[str, dict[str, Any]] = {
    "depot": {"x": 0.0, "y": 0.0, "demand": 0.0, "has_depot": True},
    "a": {"x": 2.0, "y": 1.2, "demand": 1.0, "has_depot": False},
    "b": {"x": 4.2, "y": 1.1, "demand": 2.0, "has_depot": False},
    "c": {"x": 6.4, "y": 0.4, "demand": 1.0, "has_depot": False},
    "d": {"x": 2.0, "y": -1.5, "demand": 1.0, "has_depot": False},
    "e": {"x": 4.4, "y": -1.3, "demand": 2.0, "has_depot": False},
    "f": {"x": 6.8, "y": -1.1, "demand": 1.0, "has_depot": False},
    "hub": {"x": 3.2, "y": 0.0, "demand": 0.0, "has_depot": False},
    "bridge": {"x": 5.3, "y": -0.2, "demand": 0.0, "has_depot": False},
}

TOY_EDGES: list[tuple[str, str, dict[str, float]]] = [
    ("depot", "a", {"distance": 2.4}),
    ("depot", "d", {"distance": 2.5}),
    ("depot", "hub", {"distance": 3.2}),
    ("a", "hub", {"distance": 1.7}),
    ("d", "hub", {"distance": 1.9}),
    ("a", "b", {"distance": 2.2}),
    ("d", "e", {"distance": 2.4}),
    ("hub", "bridge", {"distance": 2.1}),
    ("b", "bridge", {"distance": 1.5}),
    ("e", "bridge", {"distance": 1.2}),
    ("b", "c", {"distance": 2.4}),
    ("e", "f", {"distance": 2.4}),
    ("bridge", "c", {"distance": 1.3}),
    ("bridge", "f", {"distance": 1.6}),
    ("c", "f", {"distance": 2.2}),
]


def make_toy_graph_env(
    *,
    num_trucks: int = 2,
    truck_speed: float = 0.35,
    truck_capacity: float = 1.0,
    max_time: int = 240,
) -> GraphEnv:
    """Create the fixed first-stage SDVRP environment."""

    return GraphEnv(
        nodes=TOY_NODES,
        edges=TOY_EDGES,
        num_trucks=num_trucks,
        truck_speed=truck_speed,
        truck_capacity=truck_capacity,
        max_time=max_time,
    )

