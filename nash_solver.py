"""
nash_solver.py
==============
Solveur de l'Équilibre de Nash en stratégies mixtes (minimax / maximin).

Formulation LP — Théorème Minimax de von Neumann (jeu à somme nulle)
---------------------------------------------------------------------
La valeur du jeu V satisfait :

    V = max_p  min_j  (p · M[:, j])   [défenseur — maximin]
      = min_q  max_i  (M[i, :] · q)   [attaquant — minimax]

LP Défenseur (maximin) — variables : (p_0,...,p_{n-1}, V_s)
    Minimise  -V_s
    s.t.  -Ms[:, j] · p + V_s ≤ 0   ∀j    (V_s ≤ colonnes)
          Σ p_i = 1
          p_i ≥ 0,  V_s ≥ 0

LP Attaquant (minimax) — variables : (q_0,...,q_{m-1}, V_s)
    Minimise  V_s
    s.t.  Ms[i, :] · q - V_s ≤ 0    ∀i    (V_s ≥ lignes)
          Σ q_j = 1
          q_j ≥ 0,  V_s ≥ 0

On décale Ms = M + shift (shift > 0) pour garantir V_s > 0.
La valeur réelle est V = V_s - shift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


# ─────────────────────────────────────────────────────────────────────────────
# Résultat
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NashResult:
    """Résultat du solveur de Nash.

    Attributs
    ---------
    defender_strategy : stratégie mixte optimale du défenseur  (shape: n_rows,)
    attacker_strategy : stratégie mixte optimale de l'attaquant (shape: n_cols,)
    game_value        : valeur du jeu — utilité garantie au défenseur à l'équilibre
    """
    defender_strategy: np.ndarray
    attacker_strategy: np.ndarray
    game_value: float


# ─────────────────────────────────────────────────────────────────────────────
# Solveur principal
# ─────────────────────────────────────────────────────────────────────────────

def compute_nash(payoff_matrix: np.ndarray) -> NashResult:
    """Calcule l'équilibre de Nash en stratégies mixtes via Programmation Linéaire.

    Paramètres
    ----------
    payoff_matrix : np.ndarray  shape (n_rows, n_cols)
        Matrice de payoff du DÉFENSEUR.
        Lignes    = actions de défense disponibles
        Colonnes  = nœuds pouvant être ciblés par l'attaquant
        On suppose un jeu à somme nulle : payoff_att = -payoff_def.

    Retourne
    --------
    NashResult
        defender_strategy  : probabilités sur les lignes (actions de défense)
        attacker_strategy  : probabilités sur les colonnes (nœuds ciblés)
        game_value         : valeur du jeu (utilité garantie au défenseur)

    Exceptions
    ----------
    ValueError   si la matrice est mal formée (non-2D ou vide)
    RuntimeError si le solveur LP échoue
    """
    M = np.asarray(payoff_matrix, dtype=float)
    if M.ndim != 2 or M.size == 0:
        raise ValueError("payoff_matrix doit être un tableau 2D non vide.")

    n_rows, n_cols = M.shape

    # ── Décalage pour rendre tous les payoffs strictement positifs ────────────
    # Garantit que V_s > 0 dans les deux LP.
    shift = float(-M.min() + 1.0)
    Ms = M + shift  # Ms[i, j] > 0  ∀i, j

    # ══════════════════════════════════════════════════════════════════════════
    # LP du défenseur (maximin)
    # Variables : x = [p_0, ..., p_{n_rows-1}, V_s]  longueur n_rows + 1
    # ══════════════════════════════════════════════════════════════════════════

    # Objectif : minimiser -V_s
    c_def = np.zeros(n_rows + 1)
    c_def[-1] = -1.0

    # Contraintes inégalité : -Ms[:, j] · p + V_s ≤ 0   ∀j
    A_ub_def = np.zeros((n_cols, n_rows + 1))
    A_ub_def[:, :n_rows] = -Ms.T   # ligne j = -Ms[:, j]ᵀ
    A_ub_def[:, n_rows]  =  1.0    # +V_s
    b_ub_def = np.zeros(n_cols)

    # Contrainte égalité : Σ p_i = 1
    A_eq_def = np.zeros((1, n_rows + 1))
    A_eq_def[0, :n_rows] = 1.0
    b_eq_def = np.array([1.0])

    bounds_def = [(0.0, None)] * n_rows + [(0.0, None)]

    lp_def = linprog(
        c=c_def,
        A_ub=A_ub_def, b_ub=b_ub_def,
        A_eq=A_eq_def, b_eq=b_eq_def,
        bounds=bounds_def,
        method="highs",
    )
    if not lp_def.success:
        raise RuntimeError(
            f"Nash LP (défenseur) a échoué : {lp_def.message}"
        )

    # Extraire et normaliser la stratégie
    p = np.clip(lp_def.x[:n_rows], 0.0, None)
    total_p = p.sum()
    p /= total_p if total_p > 1e-12 else 1.0
    V = float(lp_def.x[n_rows]) - shift   # décalage inverse

    # ══════════════════════════════════════════════════════════════════════════
    # LP de l'attaquant (minimax)
    # Variables : y = [q_0, ..., q_{n_cols-1}, V_s]  longueur n_cols + 1
    # ══════════════════════════════════════════════════════════════════════════

    # Objectif : minimiser V_s
    c_att = np.zeros(n_cols + 1)
    c_att[-1] = 1.0

    # Contraintes inégalité : Ms[i, :] · q - V_s ≤ 0   ∀i
    A_ub_att = np.zeros((n_rows, n_cols + 1))
    A_ub_att[:, :n_cols] =  Ms       # ligne i = Ms[i, :]
    A_ub_att[:, n_cols]  = -1.0      # -V_s
    b_ub_att = np.zeros(n_rows)

    # Contrainte égalité : Σ q_j = 1
    A_eq_att = np.zeros((1, n_cols + 1))
    A_eq_att[0, :n_cols] = 1.0
    b_eq_att = np.array([1.0])

    bounds_att = [(0.0, None)] * n_cols + [(0.0, None)]

    lp_att = linprog(
        c=c_att,
        A_ub=A_ub_att, b_ub=b_ub_att,
        A_eq=A_eq_att, b_eq=b_eq_att,
        bounds=bounds_att,
        method="highs",
    )
    if not lp_att.success:
        raise RuntimeError(
            f"Nash LP (attaquant) a échoué : {lp_att.message}"
        )

    # Extraire et normaliser la stratégie
    q = np.clip(lp_att.x[:n_cols], 0.0, None)
    total_q = q.sum()
    q /= total_q if total_q > 1e-12 else 1.0

    return NashResult(
        defender_strategy=p,
        attacker_strategy=q,
        game_value=V,
    )
