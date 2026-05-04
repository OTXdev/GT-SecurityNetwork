"""
game_model.py
=============
Modèle de jeu attaquant / défenseur pour un réseau informatique.

Concepts clés
-------------
- Node       : un nœud du réseau (serveur, base de données, etc.)
- Edge       : un lien entre deux nœuds (vecteur de propagation latérale)
- DefenseAction : une action de défense (protège 0, 1 ou 2 nœuds)
- Game       : le jeu complet — construit la matrice de payoff

Formule du payoff défenseur pour la cellule (action d, attaque nœud t)
-----------------------------------------------------------------------
    impact(t)         = valeur(t) × vulnérabilité(t) × criticité(t)
                      + Σ_{voisin v} valeur(v) × criticité(v)
                                   × lateral_factor × poids_arête(t,v)

    résidu(t, d)      = impact(t) × (1 - defense_success)  si t protégé
                      = impact(t)                           sinon

    payoff_def(d, t)  = −résidu(t, d) − coût(d)
    payoff_att(d, t)  = résidu(t, d)  − coût_attaque(t)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, List, Sequence

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Structures de données
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Node:
    """Un nœud du réseau.

    Attributs
    ---------
    name          : identifiant unique
    attack_value  : valeur de l'actif (gain brut si compromis)
    defense_cost  : coût pour protéger ce nœud
    vulnerability : probabilité que l'attaque réussisse sans défense  [0, 1]
    attack_cost   : ressources dépensées par l'attaquant pour cibler ce nœud
    criticality   : multiplicateur d'importance (1.0 = normal)
    """
    name: str
    attack_value: float
    defense_cost: float
    vulnerability: float = 1.0
    attack_cost: float = 0.0
    criticality: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.vulnerability <= 1.0:
            raise ValueError(f"Node '{self.name}': vulnerability must be in [0, 1].")
        if self.criticality < 0:
            raise ValueError(f"Node '{self.name}': criticality must be >= 0.")
        if self.attack_value < 0:
            raise ValueError(f"Node '{self.name}': attack_value must be >= 0.")


@dataclass(frozen=True)
class Edge:
    """Lien directionnel ou non entre deux nœuds.

    Représente la capacité de propagation latérale d'une attaque.
    Le poids (weight) module l'impact latéral transmis.
    """
    source: str
    target: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"Edge '{self.source}->{self.target}': weight must be in [0, 1].")


@dataclass(frozen=True)
class DefenseAction:
    """Une action de défense disponible pour le défenseur.

    Exemples : 'No defense', 'Protect WebApp', 'Protect DB + Mail'
    """
    name: str
    protected_nodes: tuple[str, ...]
    cost: float

    def protects(self, node_name: str) -> bool:
        return node_name in self.protected_nodes


@dataclass(frozen=True)
class Scenario:
    """Un scénario de démonstration prêt à l'emploi."""
    key: str
    title: str
    description: str
    game: "Game"


# ─────────────────────────────────────────────────────────────────────────────
# Classe principale
# ─────────────────────────────────────────────────────────────────────────────

class Game:
    """Jeu matriciel de sécurité réseau (attaquant vs défenseur).

    La matrice de payoff a pour dimensions :
        lignes  → actions du défenseur
        colonnes → nœuds ciblés par l'attaquant

    Les valeurs représentent l'utilité du DÉFENSEUR (négatif = perte).
    """

    def __init__(
        self,
        nodes: Sequence[Node],
        edges: Sequence[Edge] | None = None,
        defense_actions: Sequence[DefenseAction] | None = None,
        defense_success_rate: float = 0.8,
        lateral_movement_factor: float = 0.2,
        allow_pair_defense: bool = True,
    ) -> None:
        """
        Paramètres
        ----------
        nodes                 : liste des nœuds du réseau
        edges                 : liens entre nœuds (propagation latérale)
        defense_actions       : actions disponibles (auto-générées si None)
        defense_success_rate  : taux de réduction de l'impact quand un nœud est protégé
        lateral_movement_factor : fraction de l'impact transmise aux voisins
        allow_pair_defense    : si True, génère des actions de protection par paires
        """
        if not nodes:
            raise ValueError("Le jeu nécessite au moins un nœud.")
        if not 0.0 <= defense_success_rate <= 1.0:
            raise ValueError("defense_success_rate doit être dans [0, 1].")
        if not 0.0 <= lateral_movement_factor <= 1.0:
            raise ValueError("lateral_movement_factor doit être dans [0, 1].")

        self.nodes: List[Node] = list(nodes)
        self.edges: List[Edge] = list(edges or [])
        self.defense_success_rate = float(defense_success_rate)
        self.lateral_movement_factor = float(lateral_movement_factor)
        self.allow_pair_defense = allow_pair_defense

        self.defense_actions = (
            list(defense_actions)
            if defense_actions is not None
            else self._build_default_actions()
        )
        self._validate()

    # ── Validation ────────────────────────────────────────────────────────

    def _validate(self) -> None:
        """Vérifie la cohérence interne du jeu."""
        node_names = {n.name for n in self.nodes}

        # Vérification des arêtes
        for edge in self.edges:
            if edge.source not in node_names:
                raise ValueError(f"Arête : nœud source inconnu '{edge.source}'.")
            if edge.target not in node_names:
                raise ValueError(f"Arête : nœud cible inconnu '{edge.target}'.")

        # Vérification des actions de défense
        for action in self.defense_actions:
            for protected in action.protected_nodes:
                if protected not in node_names:
                    raise ValueError(
                        f"Action '{action.name}' : nœud inconnu '{protected}'."
                    )

    # ── Construction des actions par défaut ───────────────────────────────

    def _build_default_actions(self) -> list[DefenseAction]:
        """Génère automatiquement les actions de défense :
        - Aucune défense
        - Protection individuelle de chaque nœud
        - Protection par paires (avec remise de 10 % sur le coût)
        """
        actions: list[DefenseAction] = [
            DefenseAction(name="No defense", protected_nodes=(), cost=0.0)
        ]

        # Protection individuelle
        for node in self.nodes:
            actions.append(
                DefenseAction(
                    name=f"Protect {node.name}",
                    protected_nodes=(node.name,),
                    cost=float(node.defense_cost),
                )
            )

        # Protection par paires (synérgie de coût)
        if self.allow_pair_defense and len(self.nodes) > 1:
            for left, right in combinations(self.nodes, 2):
                synergy_cost = 0.9 * (left.defense_cost + right.defense_cost)
                actions.append(
                    DefenseAction(
                        name=f"Protect {left.name} + {right.name}",
                        protected_nodes=(left.name, right.name),
                        cost=float(synergy_cost),
                    )
                )

        return actions

    # ── Accesseurs utilitaires ─────────────────────────────────────────────

    def get_attack_labels(self) -> list[str]:
        """Retourne les étiquettes des colonnes (attaques possibles)."""
        return [f"Attack {node.name}" for node in self.nodes]

    def get_defense_labels(self) -> list[str]:
        """Retourne les étiquettes des lignes (actions de défense)."""
        return [action.name for action in self.defense_actions]

    def get_node(self, name: str) -> Node:
        """Retourne le nœud par son nom (KeyError si absent)."""
        for node in self.nodes:
            if node.name == name:
                return node
        raise KeyError(f"Nœud inconnu : '{name}'.")

    def get_neighbors(self, name: str) -> list[tuple[Node, float]]:
        """Retourne les voisins d'un nœud et le poids de l'arête (non orienté)."""
        neighbors: list[tuple[Node, float]] = []
        for edge in self.edges:
            if edge.source == name:
                neighbors.append((self.get_node(edge.target), edge.weight))
            elif edge.target == name:
                neighbors.append((self.get_node(edge.source), edge.weight))
        return neighbors

    # ── Calcul d'impact ───────────────────────────────────────────────────

    def compute_attack_impact(self, node_name: str) -> float:
        """Calcule l'impact total d'une attaque sur un nœud.

        impact = impact_direct + propagation_latérale

        impact_direct       = valeur × vulnérabilité × criticité
        propagation_latérale = Σ_voisins valeur_v × criticité_v
                                        × lateral_factor × poids_arête
        """
        node = self.get_node(node_name)

        direct = node.attack_value * node.vulnerability * node.criticality

        lateral = sum(
            neighbor.attack_value
            * neighbor.criticality
            * self.lateral_movement_factor
            * weight
            for neighbor, weight in self.get_neighbors(node_name)
        )

        return direct + lateral

    # ── Matrices de payoff ────────────────────────────────────────────────

    def get_payoff_matrix(self) -> np.ndarray:
        """Matrice de payoff du DÉFENSEUR.

        Shape : (nb_actions_défense, nb_nœuds)
        Valeurs négatives : le défenseur subit toujours une perte.

        payoff_def(d, t) = −impact(t) × résidu(t,d) − coût(d)
        avec résidu(t,d) = 1 − defense_success si t protégé, sinon 1
        """
        matrix = np.zeros((len(self.defense_actions), len(self.nodes)), dtype=float)

        for row_idx, action in enumerate(self.defense_actions):
            for col_idx, target in enumerate(self.nodes):
                impact = self.compute_attack_impact(target.name)
                residual = (
                    1.0 - self.defense_success_rate
                    if action.protects(target.name)
                    else 1.0
                )
                matrix[row_idx, col_idx] = -impact * residual - action.cost

        return matrix

    def get_attacker_payoff_matrix(self) -> np.ndarray:
        """Matrice de payoff de l'ATTAQUANT.

        Shape : (nb_actions_défense, nb_nœuds)

        payoff_att(d, t) = impact(t) × résidu(t,d) − coût_attaque(t)

        Note : ce jeu n'est pas à somme nulle car l'attaquant supporte
        un coût d'attaque indépendant du défenseur.
        """
        matrix = np.zeros((len(self.defense_actions), len(self.nodes)), dtype=float)

        for row_idx, action in enumerate(self.defense_actions):
            for col_idx, target in enumerate(self.nodes):
                impact = self.compute_attack_impact(target.name)
                residual = (
                    1.0 - self.defense_success_rate
                    if action.protects(target.name)
                    else 1.0
                )
                matrix[row_idx, col_idx] = impact * residual - target.attack_cost

        return matrix

    # ── Description textuelle ─────────────────────────────────────────────

    def describe(self) -> str:
        """Résumé lisible des paramètres du jeu."""
        lines = [
            f"Nodes: {len(self.nodes)}",
            f"Edges: {len(self.edges)}",
            f"Defense actions: {len(self.defense_actions)}",
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

    # ── Scénarios de démonstration ────────────────────────────────────────

    @classmethod
    def get_scenarios(cls) -> list[Scenario]:
        """Retourne les 3 scénarios prédéfinis."""
        return [
            Scenario(
                key="critical-db",
                title="Base de données critique",
                description=(
                    "Réseau où la base de données concentre la valeur "
                    "et la latéralité. Tester si le défenseur la protège en priorité."
                ),
                game=cls(
                    nodes=[
                        Node("Database", 14.0, 3.0, vulnerability=0.95, attack_cost=2.0, criticality=1.4),
                        Node("WebApp",   8.0,  2.0, vulnerability=0.90, attack_cost=1.0, criticality=1.1),
                        Node("Mail",     5.0,  1.5, vulnerability=0.80, attack_cost=1.0, criticality=0.9),
                        Node("Backup",   6.0,  1.2, vulnerability=0.60, attack_cost=1.5, criticality=1.0),
                    ],
                    edges=[
                        Edge("WebApp", "Database", 1.0),
                        Edge("Mail",   "Database", 0.7),
                        Edge("Backup", "Database", 0.5),
                        Edge("WebApp", "Mail",     0.8),
                    ],
                    defense_success_rate=0.82,
                    lateral_movement_factor=0.28,
                ),
            ),
            Scenario(
                key="balanced-network",
                title="Réseau équilibré",
                description=(
                    "Services de valeur comparable. "
                    "Utile pour observer la divergence Nash / Stackelberg."
                ),
                game=cls(
                    nodes=[
                        Node("Gateway", 8.0, 2.0, vulnerability=0.75, attack_cost=1.2, criticality=1.00),
                        Node("ERP",     9.0, 2.5, vulnerability=0.70, attack_cost=1.5, criticality=1.10),
                        Node("CRM",     8.0, 2.1, vulnerability=0.72, attack_cost=1.3, criticality=1.00),
                        Node("HR",      7.5, 1.8, vulnerability=0.68, attack_cost=1.1, criticality=0.95),
                    ],
                    edges=[
                        Edge("Gateway", "ERP", 0.8),
                        Edge("Gateway", "CRM", 0.7),
                        Edge("ERP",     "HR",  0.6),
                        Edge("CRM",     "HR",  0.6),
                    ],
                    defense_success_rate=0.78,
                    lateral_movement_factor=0.18,
                ),
            ),
            Scenario(
                key="limited-attacker",
                title="Attaquant limité",
                description=(
                    "Les cibles très rentables exigent un effort d'attaque élevé. "
                    "Observer comment le coût dissuade l'attaquant."
                ),
                game=cls(
                    nodes=[
                        Node("Finance",   13.0, 3.2, vulnerability=0.85, attack_cost=4.5, criticality=1.25),
                        Node("Portal",    7.0,  1.8, vulnerability=0.88, attack_cost=1.0, criticality=1.00),
                        Node("Analytics", 10.0, 2.4, vulnerability=0.74, attack_cost=3.2, criticality=1.15),
                        Node("Storage",   9.0,  2.0, vulnerability=0.64, attack_cost=2.4, criticality=1.05),
                    ],
                    edges=[
                        Edge("Portal",  "Finance",   0.9),
                        Edge("Portal",  "Analytics", 0.8),
                        Edge("Storage", "Analytics", 0.6),
                    ],
                    defense_success_rate=0.80,
                    lateral_movement_factor=0.22,
                ),
            ),
        ]

    @classmethod
    def sample_game(cls) -> "Game":
        """Retourne le premier scénario (raccourci pratique)."""
        return cls.get_scenarios()[0].game

    @classmethod
    def from_values(
        cls,
        names: Iterable[str],
        attack_values: Iterable[float],
        defense_costs: Iterable[float],
        vulnerabilities: Iterable[float] | None = None,
        attack_costs: Iterable[float] | None = None,
        criticalities: Iterable[float] | None = None,
    ) -> "Game":
        """Crée un jeu simple à partir de listes de valeurs (sans arêtes).

        Pratique pour les tests ou les scénarios manuels.
        """
        names_list = list(names)
        n = len(names_list)
        vuln  = list(vulnerabilities) if vulnerabilities is not None else [1.0] * n
        acost = list(attack_costs)    if attack_costs    is not None else [0.0] * n
        crit  = list(criticalities)   if criticalities   is not None else [1.0] * n

        nodes = [
            Node(
                name=name,
                attack_value=float(av),
                defense_cost=float(dc),
                vulnerability=float(v),
                attack_cost=float(ac),
                criticality=float(cr),
            )
            for name, av, dc, v, ac, cr in zip(
                names_list,
                list(attack_values),
                list(defense_costs),
                vuln, acost, crit,
                strict=True,
            )
        ]
        return cls(nodes=nodes)
