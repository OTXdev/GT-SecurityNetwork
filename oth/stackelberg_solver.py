"""
stackelberg_solver.py
=====================
Solveur du jeu de Stackelberg pour la sécurité réseau.

Principe
--------
Le jeu de Stackelberg est un jeu SÉQUENTIEL à deux joueurs :

    1. Le DÉFENSEUR (leader) annonce et s'engage dans une stratégie mixte p.
    2. L'ATTAQUANT (follower) observe p et choisit la MEILLEURE RÉPONSE PURE.
    3. Le défenseur anticipe cette réaction et choisit p de façon optimale.

Pourquoi le défenseur fait mieux qu'en Nash ?
    En Nash, les deux joueurs décident simultanément.
    En Stackelberg, le défenseur engage sa stratégie EN PREMIER,
    ce qui lui permet d'orienter le comportement de l'attaquant.
    → Le leader a toujours un payoff ≥ valeur du jeu Nash.

Formulation LP (pour chaque meilleure réponse candidate j*)
------------------------------------------------------------
Pour un j* fixé (colonne = nœud ciblé par l'attaquant) :

    Maximise   Σ_i  p_i × M_def[i, j*]          (payoff défenseur)
    sous :
        Σ_j  p_i × M_att[i, j] ≤ Σ_i p_i × M_att[i, j*]   ∀ j ≠ j*
            (j* est bien la meilleure réponse de l'attaquant)
        Σ_i p_i = 1
        0 ≤ p_i ≤ 1

On résout un LP par colonne j* et on garde le meilleur résultat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass
class StackelbergResult:
    """Résultat du solveur de Stackelberg.

    Attributs
    ---------
    defender_strategy      : stratégie mixte optimale du défenseur  (shape: n_rows)
    attacker_best_response : indice de la colonne (nœud attaqué en réponse)
    defender_payoff        : payoff garanti au défenseur
    attacker_payoff        : payoff de l'attaquant correspondant
    """
    defender_strategy: np.ndarray
    attacker_best_response: int
    defender_payoff: float
    attacker_payoff: float


def compute_stackelberg(
    payoff_matrix: np.ndarray,
    attacker_payoff_matrix: np.ndarray | None = None,
    tolerance: float = 1e-9,
) -> StackelbergResult:
    """Calcule l'engagement optimal du défenseur (leader de Stackelberg).

    Paramètres
    ----------
    payoff_matrix          : matrice de payoff du défenseur  (n_rows × n_cols)
                             lignes = actions défense, colonnes = nœuds attaqués
    attacker_payoff_matrix : matrice de payoff de l'attaquant (même shape)
                             Si None, on suppose un jeu à somme nulle :
                             payoff_att = −payoff_def
    tolerance              : marge pour comparer deux payoffs défenseur

    Retourne
    --------
    StackelbergResult avec la meilleure stratégie trouvée.

    Raises
    ------
    ValueError  si les matrices sont mal formées
    RuntimeError si aucune solution réalisable n'existe
    """
    M_def = np.asarray(payoff_matrix, dtype=float)
    if M_def.ndim != 2 or M_def.size == 0:
        raise ValueError("payoff_matrix doit être un tableau 2D non vide.")

    if attacker_payoff_matrix is None:
        # Hypothèse zéro-somme
        M_att = -M_def
    else:
        M_att = np.asarray(attacker_payoff_matrix, dtype=float)
        if M_att.shape != M_def.shape:
            raise ValueError(
                "attacker_payoff_matrix doit avoir la même shape que payoff_matrix."
            )

    n_rows, n_cols = M_def.shape
    best: StackelbergResult | None = None

    # Contrainte de probabilité : Σ p_i = 1
    A_eq = np.ones((1, n_rows))
    b_eq = np.array([1.0])
    bounds = [(0.0, 1.0)] * n_rows

    for j_star in range(n_cols):
        # ── Objectif : maximiser Σ_i p_i × M_def[i, j*] ──────────────────
        # linprog minimise, donc on passe l'opposé
        c = -M_def[:, j_star]

        # ── Contraintes d'incitation ──────────────────────────────────────
        # Pour tout j ≠ j*, l'attaquant ne doit pas préférer j à j* :
        #   Σ_i p_i × M_att[i, j] ≤ Σ_i p_i × M_att[i, j*]
        #   ↔  Σ_i p_i × (M_att[i, j] − M_att[i, j*]) ≤ 0
        A_ub_rows = []
        for j in range(n_cols):
            if j != j_star:
                A_ub_rows.append(M_att[:, j] - M_att[:, j_star])

        if A_ub_rows:
            A_ub = np.array(A_ub_rows)
            b_ub = np.zeros(len(A_ub_rows))
        else:
            # Un seul nœud — pas de contrainte d'incitation
            A_ub = np.zeros((1, n_rows))
            b_ub = np.array([0.0])

        lp = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if not lp.success:
            # Cette meilleure réponse n'est pas réalisable, on passe à la suivante
            continue

        # Normaliser la stratégie (éviter les erreurs numériques)
        strategy = np.clip(lp.x, 0.0, 1.0)
        strategy /= strategy.sum()

        def_payoff = float(strategy @ M_def[:, j_star])
        att_payoff = float(strategy @ M_att[:, j_star])

        # Conserver la meilleure solution pour le défenseur
        if best is None or def_payoff > best.defender_payoff + tolerance:
            best = StackelbergResult(
                defender_strategy=strategy,
                attacker_best_response=j_star,
                defender_payoff=def_payoff,
                attacker_payoff=att_payoff,
            )

    if best is None:
        raise RuntimeError(
            "Aucune solution de Stackelberg réalisable trouvée. "
            "Vérifiez les paramètres du jeu."
        )

    return best
