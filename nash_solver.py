"""
nash_solver.py
==============
Solveur Nash Equilibrium en stratégies mixtes (jeu non à somme nulle).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog


@dataclass
class NashResult:
    defender_strategy: np.ndarray
    attacker_strategy: np.ndarray
    defender_payoff: float
    attacker_payoff: float
    defender_labels: list
    attacker_labels: list
    converged: bool
    message: str

    def defender_dict(self) -> dict:
        return {
            label: round(float(prob), 4)
            for label, prob in zip(self.defender_labels, self.defender_strategy)
            if prob > 0.001
        }

    def attacker_dict(self) -> dict:
        return {
            label: round(float(prob), 4)
            for label, prob in zip(self.attacker_labels, self.attacker_strategy)
            if prob > 0.001
        }

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "  NASH EQUILIBRIUM",
            "=" * 55,
            f"  Payoff défenseur : {self.defender_payoff:+.4f}",
            f"  Payoff attaquant : {self.attacker_payoff:+.4f}",
            "",
            "  Stratégie DÉFENSEUR :",
        ]
        for label, prob in self.defender_dict().items():
            lines.append(f"    {label[:35]:<35s} {prob:.4f}")
        lines += ["", "  Stratégie ATTAQUANT :"]
        for label, prob in self.attacker_dict().items():
            lines.append(f"    {label[:35]:<35s} {prob:.4f}")
        lines += [f"\n  [{self.message}]", "=" * 55]
        return "\n".join(lines)


class NashSolver:
    def __init__(self, game) -> None:
        self.game = game
        self.M_def = game.get_payoff_matrix()
        self.M_att = game.get_attacker_payoff_matrix()
        self.defender_labels = [a.name for a in game.defense_actions]
        self.attacker_labels = game.get_attack_labels()
        self.n_actions = len(self.defender_labels)
        self.n_attacks = len(self.attacker_labels)

        if self.M_def.shape != (self.n_actions, self.n_attacks):
            raise ValueError(
                f"Matrice de payoff dimension incorrecte: "
                f"attendue ({self.n_actions}, {self.n_attacks}), "
                f"obtenue {self.M_def.shape}"
            )

    def _solve_defender(self) -> tuple:
        n = self.n_actions
        m = self.n_attacks

        c = np.zeros(n + 1)
        c[-1] = -1.0

        A_ub = np.zeros((m, n + 1))
        for j in range(m):
            A_ub[j, :n] = -self.M_def[:, j]
            A_ub[j, n]  =  1.0
        b_ub = np.zeros(m)

        A_eq = np.zeros((1, n + 1))
        A_eq[0, :n] = 1.0
        b_eq = np.array([1.0])

        bounds = [(0.0, 1.0)] * n + [(None, None)]

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method="highs")

        if result.success:
            strategy = np.clip(result.x[:n], 0.0, 1.0)
            total = strategy.sum()
            if total > 1e-10:
                strategy /= total
            value = float(-result.fun)
            return strategy, value, True
        else:
            strategy = np.ones(n) / n
            value = float(strategy @ self.M_def).mean()
            return strategy, value, False

    def _solve_attacker(self) -> tuple:
        n = self.n_actions
        m = self.n_attacks

        c = np.zeros(m + 1)
        c[-1] = 1.0

        A_ub = np.zeros((n, m + 1))
        for i in range(n):
            A_ub[i, :m] =  self.M_att[i, :]
            A_ub[i, m]  = -1.0
        b_ub = np.zeros(n)

        A_eq = np.zeros((1, m + 1))
        A_eq[0, :m] = 1.0
        b_eq = np.array([1.0])

        bounds = [(0.0, 1.0)] * m + [(None, None)]

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method="highs")

        if result.success:
            strategy = np.clip(result.x[:m], 0.0, 1.0)
            total = strategy.sum()
            if total > 1e-10:
                strategy /= total
            value = float(result.fun)
            return strategy, value, True
        else:
            strategy = np.ones(m) / m
            value = float(self.M_att).mean()
            return strategy, value, False

    def solve(self) -> NashResult:
        def_strategy, def_payoff, def_ok = self._solve_defender()
        att_strategy, att_payoff, att_ok = self._solve_attacker()
        converged = def_ok and att_ok

        if converged:
            message = "Nash Equilibrium trouvé (LP HiGHS)"
        elif def_ok:
            message = "LP attaquant non convergé — stratégie uniforme utilisée"
        elif att_ok:
            message = "LP défenseur non convergé — stratégie uniforme utilisée"
        else:
            message = "Les deux LP n'ont pas convergé — stratégies uniformes"

        return NashResult(
            defender_strategy=def_strategy,
            attacker_strategy=att_strategy,
            defender_payoff=def_payoff,
            attacker_payoff=att_payoff,
            defender_labels=self.defender_labels,
            attacker_labels=self.attacker_labels,
            converged=converged,
            message=message,
        )

    def price_of_anarchy(self, optimal_payoff: float) -> float:
        result = self.solve()
        nash_payoff = result.defender_payoff
        if abs(nash_payoff) < 1e-10:
            return 1.0
        poa = abs(nash_payoff) / abs(optimal_payoff) if abs(optimal_payoff) > 1e-10 else 1.0
        return round(poa, 4)

    def optimal_centralized(self) -> tuple:
        worst_case = self.M_def.min(axis=1)
        best_action_idx = int(np.argmax(worst_case))
        best_payoff = float(worst_case[best_action_idx])
        one_hot = np.zeros(self.n_actions)
        one_hot[best_action_idx] = 1.0
        return best_payoff, one_hot
