"""
game_model.py
=============
Modèle de jeu attaquant / défenseur pour un réseau informatique.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, List, Sequence

import numpy as np


@dataclass(frozen=True)
class Node:
    name: str
    attack_value: float
    defense_cost: float
    vulnerability: float = 1.0
    attack_cost: float = 0.0
    criticality: float = 1.0

    def __post_init__(self):
        if not 0.0 <= self.vulnerability <= 1.0:
            raise ValueError(f"Node '{self.name}': vulnerability must be in [0, 1].")
        if self.criticality < 0:
            raise ValueError(f"Node '{self.name}': criticality must be >= 0.")
        if self.attack_value < 0:
            raise ValueError(f"Node '{self.name}': attack_value must be >= 0.")


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    weight: float = 1.0

    def __post_init__(self):
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"Edge '{self.source}->{self.target}': weight must be in [0, 1].")


@dataclass(frozen=True)
class DefenseAction:
    name: str
    protected_nodes: tuple
    cost: float

    def protects(self, node_name: str) -> bool:
        return node_name in self.protected_nodes


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    description: str
    game: "Game"


class Game:
    def __init__(
        self,
        nodes: Sequence[Node],
        edges: Sequence[Edge] | None = None,
        defense_actions: Sequence[DefenseAction] | None = None,
        defense_success_rate: float = 0.8,
        lateral_movement_factor: float = 0.2,
        allow_pair_defense: bool = True,
        defender_budget: int = 2,
        attacker_budget: int = 1,
    ):
        if not nodes:
            raise ValueError("Le jeu nécessite au moins un nœud.")
        if not 0.0 <= defense_success_rate <= 1.0:
            raise ValueError("defense_success_rate doit être dans [0, 1].")
        if not 0.0 <= lateral_movement_factor <= 1.0:
            raise ValueError("lateral_movement_factor doit être dans [0, 1].")
        if not 1 <= defender_budget <= 4:
            raise ValueError("defender_budget doit être entre 1 et 4.")
        if not 1 <= attacker_budget <= 4:
            raise ValueError("attacker_budget doit être entre 1 et 4.")

        self.nodes: List[Node] = list(nodes)
        self.edges: List[Edge] = list(edges or [])
        self.defense_success_rate = float(defense_success_rate)
        self.lateral_movement_factor = float(lateral_movement_factor)
        self.allow_pair_defense = allow_pair_defense
        self.defender_budget = defender_budget
        self.attacker_budget = attacker_budget

        self.defense_actions = (
            list(defense_actions) if defense_actions is not None
            else self._build_default_actions()
        )
        self._validate()

    def _validate(self):
        node_names = {n.name for n in self.nodes}
        for edge in self.edges:
            if edge.source not in node_names:
                raise ValueError(f"Arête : nœud source inconnu '{edge.source}'.")
            if edge.target not in node_names:
                raise ValueError(f"Arête : nœud cible inconnu '{edge.target}'.")
        for action in self.defense_actions:
            for protected in action.protected_nodes:
                if protected not in node_names:
                    raise ValueError(f"Action '{action.name}' : nœud inconnu '{protected}'.")

    def _build_default_actions(self) -> list:
        actions = [DefenseAction(name="No defense", protected_nodes=(), cost=0.0)]
        n_nodes = len(self.nodes)
        max_k = min(self.defender_budget, n_nodes)

        for k in range(1, max_k + 1):
            for combo in combinations(self.nodes, k):
                node_names = tuple(n.name for n in combo)
                if k == 1:
                    cost = combo[0].defense_cost
                else:
                    synergy = 0.9 ** (k - 1)
                    cost = synergy * sum(n.defense_cost for n in combo)

                if len(node_names) > 3:
                    name = f"Protect {node_names[0]}+...+{node_names[-1]} ({k})"
                else:
                    name = f"Protect {' + '.join(node_names)}"

                actions.append(DefenseAction(name=name, protected_nodes=node_names, cost=float(cost)))

        if len(actions) > 200:
            actions = actions[:200]
        return actions

    def get_attack_labels(self) -> list:
        if self.attacker_budget == 1:
            return [f"Attack {node.name}" for node in self.nodes]
        else:
            labels = []
            max_k = min(self.attacker_budget, len(self.nodes))
            for k in range(1, max_k + 1):
                for combo in combinations(self.nodes, k):
                    names = '+'.join(n.name for n in combo)
                    if len(names) > 30:
                        names = names[:27] + "..."
                    labels.append(f"Attack {names}")
            return labels

    def get_defense_labels(self) -> list:
        return [action.name for action in self.defense_actions]

    def get_node(self, name: str) -> Node:
        for node in self.nodes:
            if node.name == name:
                return node
        raise KeyError(f"Nœud inconnu : '{name}'.")

    def get_neighbors(self, name: str) -> list:
        neighbors = []
        for edge in self.edges:
            if edge.source == name:
                neighbors.append((self.get_node(edge.target), edge.weight))
            elif edge.target == name:
                neighbors.append((self.get_node(edge.source), edge.weight))
        return neighbors

    def compute_attack_impact(self, node_name: str) -> float:
        node = self.get_node(node_name)
        direct = node.attack_value * node.vulnerability * node.criticality
        lateral = sum(
            neighbor.attack_value * neighbor.criticality * self.lateral_movement_factor * weight
            for neighbor, weight in self.get_neighbors(node_name)
        )
        return direct + lateral

    def get_payoff_matrix(self) -> np.ndarray:
        n_actions = len(self.defense_actions)
        if self.attacker_budget == 1:
            n_attacks = len(self.nodes)
            matrix = np.zeros((n_actions, n_attacks), dtype=float)
            for row_idx, action in enumerate(self.defense_actions):
                for col_idx, target in enumerate(self.nodes):
                    impact = self.compute_attack_impact(target.name)
                    residual = (1.0 - self.defense_success_rate if action.protects(target.name) else 1.0)
                    matrix[row_idx, col_idx] = -impact * residual - action.cost
            return matrix
        else:
            attack_combos = []
            max_k = min(self.attacker_budget, len(self.nodes))
            for k in range(1, max_k + 1):
                attack_combos.extend(combinations(self.nodes, k))
            n_attacks = len(attack_combos)
            matrix = np.zeros((n_actions, n_attacks), dtype=float)
            for row_idx, action in enumerate(self.defense_actions):
                for col_idx, targets in enumerate(attack_combos):
                    total_impact = 0.0
                    for target in targets:
                        impact = self.compute_attack_impact(target.name)
                        residual = (1.0 - self.defense_success_rate if action.protects(target.name) else 1.0)
                        total_impact += impact * residual
                    matrix[row_idx, col_idx] = -total_impact - action.cost
            return matrix

    def get_attacker_payoff_matrix(self) -> np.ndarray:
        n_actions = len(self.defense_actions)
        if self.attacker_budget == 1:
            n_attacks = len(self.nodes)
            matrix = np.zeros((n_actions, n_attacks), dtype=float)
            for row_idx, action in enumerate(self.defense_actions):
                for col_idx, target in enumerate(self.nodes):
                    impact = self.compute_attack_impact(target.name)
                    residual = (1.0 - self.defense_success_rate if action.protects(target.name) else 1.0)
                    matrix[row_idx, col_idx] = impact * residual - target.attack_cost
            return matrix
        else:
            attack_combos = []
            max_k = min(self.attacker_budget, len(self.nodes))
            for k in range(1, max_k + 1):
                attack_combos.extend(combinations(self.nodes, k))
            n_attacks = len(attack_combos)
            matrix = np.zeros((n_actions, n_attacks), dtype=float)
            for row_idx, action in enumerate(self.defense_actions):
                for col_idx, targets in enumerate(attack_combos):
                    total_gain = 0.0
                    total_cost = 0.0
                    for target in targets:
                        impact = self.compute_attack_impact(target.name)
                        residual = (1.0 - self.defense_success_rate if action.protects(target.name) else 1.0)
                        total_gain += impact * residual
                        total_cost += target.attack_cost
                    matrix[row_idx, col_idx] = total_gain - total_cost
            return matrix

    def describe(self) -> str:
        lines = [
            f"Nodes: {len(self.nodes)}",
            f"Edges: {len(self.edges)}",
            f"Defense actions: {len(self.defense_actions)}",
            f"Defense budget: {self.defender_budget}",
            f"Attack budget: {self.attacker_budget}",
            f"Defense success rate: {self.defense_success_rate:.2f}",
            f"Lateral movement factor: {self.lateral_movement_factor:.2f}",
        ]
        for node in self.nodes:
            lines.append(
                f"- {node.name}: value={node.attack_value:.1f}, "
                f"defense_cost={node.defense_cost:.1f}, "
                f"vulnerability={node.vulnerability:.2f}, "
                f"attack_cost={node.attack_cost:.1f}, "
                f"criticality={node.criticality:.2f}"
            )
        return "\n".join(lines)

    @classmethod
    def sample_game(cls) -> "Game":
        nodes = [
            Node("Database", 14.0, 3.0, vulnerability=0.95, attack_cost=2.0, criticality=1.4),
            Node("WebApp",   8.0,  2.0, vulnerability=0.90, attack_cost=1.0, criticality=1.1),
            Node("Mail",     5.0,  1.5, vulnerability=0.80, attack_cost=1.0, criticality=0.9),
            Node("Backup",   6.0,  1.2, vulnerability=0.60, attack_cost=1.5, criticality=1.0),
        ]
        edges = [
            Edge("WebApp", "Database", 1.0),
            Edge("Mail",   "Database", 0.7),
            Edge("Backup", "Database", 0.5),
            Edge("WebApp", "Mail",     0.8),
        ]
        return cls(nodes=nodes, edges=edges, defense_success_rate=0.82,
                   lateral_movement_factor=0.28, defender_budget=2, attacker_budget=1)

    @classmethod
    def from_values(cls, names, attack_values, defense_costs,
                    vulnerabilities=None, attack_costs=None, criticalities=None,
                    defender_budget=2, attacker_budget=1) -> "Game":
        names_list = list(names)
        n = len(names_list)
        vuln  = list(vulnerabilities) if vulnerabilities is not None else [1.0] * n
        acost = list(attack_costs)    if attack_costs    is not None else [0.0] * n
        crit  = list(criticalities)   if criticalities   is not None else [1.0] * n
        nodes = [
            Node(name=name, attack_value=float(av), defense_cost=float(dc),
                 vulnerability=float(v), attack_cost=float(ac), criticality=float(cr))
            for name, av, dc, v, ac, cr in zip(
                names_list, list(attack_values), list(defense_costs), vuln, acost, crit, strict=True)
        ]
        return cls(nodes=nodes, defender_budget=defender_budget, attacker_budget=attacker_budget)
