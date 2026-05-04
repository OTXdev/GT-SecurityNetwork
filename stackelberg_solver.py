"""
stackelberg_solver.py
=====================
Solveur du jeu de Stackelberg (défenseur leader).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog


@dataclass
class StackelbergResult:
    defender_strategy: np.ndarray
    attacker_best_response: int
    defender_payoff: float
    attacker_payoff: float


def compute_stackelberg(
    payoff_matrix: np.ndarray,
    attacker_payoff_matrix: np.ndarray | None = None,
    tolerance: float = 1e-9,
) -> StackelbergResult:
    M_def = np.asarray(payoff_matrix, dtype=float)
    if M_def.ndim != 2 or M_def.size == 0:
        raise ValueError("payoff_matrix doit être un tableau 2D non vide.")

    if attacker_payoff_matrix is None:
        M_att = -M_def
    else:
        M_att = np.asarray(attacker_payoff_matrix, dtype=float)
        if M_att.shape != M_def.shape:
            raise ValueError("attacker_payoff_matrix doit avoir la même shape que payoff_matrix.")

    n_rows, n_cols = M_def.shape
    best: StackelbergResult | None = None

    A_eq = np.ones((1, n_rows))
    b_eq = np.array([1.0])
    bounds = [(0.0, 1.0)] * n_rows

    for j_star in range(n_cols):
        c = -M_def[:, j_star]

        A_ub_rows = []
        for j in range(n_cols):
            if j != j_star:
                A_ub_rows.append(M_att[:, j] - M_att[:, j_star])

        if A_ub_rows:
            A_ub = np.array(A_ub_rows)
            b_ub = np.zeros(len(A_ub_rows))
        else:
            A_ub = np.zeros((1, n_rows))
            b_ub = np.array([0.0])

        lp = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method="highs")

        if not lp.success:
            continue

        strategy = np.clip(lp.x, 0.0, 1.0)
        strategy /= strategy.sum()

        def_payoff = float(strategy @ M_def[:, j_star])
        att_payoff = float(strategy @ M_att[:, j_star])

        if best is None or def_payoff > best.defender_payoff + tolerance:
            best = StackelbergResult(
                defender_strategy=strategy,
                attacker_best_response=j_star,
                defender_payoff=def_payoff,
                attacker_payoff=att_payoff,
            )

    if best is None:
        raise RuntimeError("Aucune solution de Stackelberg réalisable trouvée.")

    return best
