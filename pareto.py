"""
pareto.py
=========
Calcul de la frontière de Pareto pour le jeu de sécurité réseau.

Principe
--------
Une allocation (payoff_def, payoff_att) est Pareto-optimale si
aucune autre allocation ne domine simultanément les deux joueurs.
(payoff_def' >= payoff_def ET payoff_att' >= payoff_att, avec au moins une stricte)

On calcule la frontière de Pareto sur l'ensemble des cellules
de la matrice de payoff (M_def, M_att).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ParetoPoint:
    defense_action: str
    attack_strategy: str
    defender_payoff: float
    attacker_payoff: float
    action_idx: int
    attack_idx: int


@dataclass
class ParetoResult:
    frontier: list[ParetoPoint]
    all_points: list[ParetoPoint]
    nash_point: ParetoPoint | None
    stackelberg_point: ParetoPoint | None


def compute_pareto(
    M_def: np.ndarray,
    M_att: np.ndarray,
    defense_labels: list[str],
    attack_labels: list[str],
    nash_def_payoff: float | None = None,
    nash_att_payoff: float | None = None,
    stack_def_payoff: float | None = None,
    stack_att_payoff: float | None = None,
) -> ParetoResult:
    """
    Calcule la frontière de Pareto à partir des matrices de payoff.

    Paramètres
    ----------
    M_def           : matrice de payoff du défenseur (n_actions x n_attacks)
    M_att           : matrice de payoff de l'attaquant (n_actions x n_attacks)
    defense_labels  : noms des actions de défense
    attack_labels   : noms des stratégies d'attaque
    nash_def_payoff : payoff Nash du défenseur (pour repère)
    nash_att_payoff : payoff Nash de l'attaquant (pour repère)
    stack_def_payoff: payoff Stackelberg du défenseur (pour repère)
    stack_att_payoff: payoff Stackelberg de l'attaquant (pour repère)

    Retourne
    --------
    ParetoResult avec la frontière et les points de référence.
    """
    n_rows, n_cols = M_def.shape

    # Construire tous les points (une cellule = un point)
    all_points: list[ParetoPoint] = []
    for i in range(n_rows):
        for j in range(n_cols):
            all_points.append(ParetoPoint(
                defense_action=defense_labels[i],
                attack_strategy=attack_labels[j],
                defender_payoff=float(M_def[i, j]),
                attacker_payoff=float(M_att[i, j]),
                action_idx=i,
                attack_idx=j,
            ))

    # Identifier les points Pareto-optimaux
    frontier: list[ParetoPoint] = []
    for p in all_points:
        dominated = False
        for q in all_points:
            if q is p:
                continue
            # q domine p si q est meilleur ou égal pour les deux joueurs
            # et strictement meilleur pour au moins un
            if (q.defender_payoff >= p.defender_payoff and
                q.attacker_payoff >= p.attacker_payoff and
                (q.defender_payoff > p.defender_payoff or
                 q.attacker_payoff > p.attacker_payoff)):
                dominated = True
                break
        if not dominated:
            frontier.append(p)

    # Trier la frontière par payoff défenseur croissant
    frontier.sort(key=lambda p: p.defender_payoff)

    # Point Nash le plus proche (dans l'espace payoff)
    nash_point = None
    if nash_def_payoff is not None and nash_att_payoff is not None:
        best_dist = float("inf")
        for p in all_points:
            dist = (p.defender_payoff - nash_def_payoff) ** 2 + \
                   (p.attacker_payoff - nash_att_payoff) ** 2
            if dist < best_dist:
                best_dist = dist
                nash_point = p

    # Point Stackelberg le plus proche
    stackelberg_point = None
    if stack_def_payoff is not None and stack_att_payoff is not None:
        best_dist = float("inf")
        for p in all_points:
            dist = (p.defender_payoff - stack_def_payoff) ** 2 + \
                   (p.attacker_payoff - stack_att_payoff) ** 2
            if dist < best_dist:
                best_dist = dist
                stackelberg_point = p

    return ParetoResult(
        frontier=frontier,
        all_points=all_points,
        nash_point=nash_point,
        stackelberg_point=stackelberg_point,
    )


def pareto_efficiency_score(frontier: list[ParetoPoint], nash_point: ParetoPoint | None) -> float:
    """
    Calcule un score d'efficacité Pareto pour le Nash Equilibrium.

    Score = distance du Nash à la frontière de Pareto normalisée.
    Score proche de 0 = Nash est quasi-Pareto-optimal.
    Score élevé = Nash est loin de l'efficacité sociale.
    """
    if nash_point is None or not frontier:
        return 0.0

    min_dist = min(
        ((p.defender_payoff - nash_point.defender_payoff) ** 2 +
         (p.attacker_payoff - nash_point.attacker_payoff) ** 2) ** 0.5
        for p in frontier
    )
    return round(min_dist, 4)
