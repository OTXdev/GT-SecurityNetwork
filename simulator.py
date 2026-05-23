"""
simulator.py
============
Simulation de convergence vers l'équilibre de Nash.
"""

import numpy as np
import pandas as pd


class GameSimulator:
    """Simule des tours de jeu basés sur les stratégies Nash et trace la convergence."""

    def __init__(self, game, nash_result):
        self.game = game
        self.nash_result = nash_result
        self.history = []

    def play_turn(self) -> dict:
        defender_action = np.random.choice(
            len(self.nash_result.defender_strategy),
            p=self.nash_result.defender_strategy
        )
        attacker_action = np.random.choice(
            len(self.nash_result.attacker_strategy),
            p=self.nash_result.attacker_strategy
        )

        defender_label = self.nash_result.defender_labels[defender_action]
        attacker_label = self.nash_result.attacker_labels[attacker_action]

        M_def = self.game.get_payoff_matrix()
        payoff = M_def[defender_action, attacker_action]

        turn = {
            'tour': len(self.history) + 1,
            'defenseur': defender_label,
            'attaquant': attacker_label,
            'payoff_defenseur': payoff,
            'action_def_idx': defender_action,
            'action_att_idx': attacker_action,
        }
        self.history.append(turn)
        return turn

    def play_multiple_turns(self, n_turns: int) -> list:
        for _ in range(n_turns):
            self.play_turn()
        return self.history

    def get_convergence_data(self) -> dict:
        """
        Calcule les données de convergence tour par tour.

        Retourne un dict avec :
        - tours          : numéros de tour
        - payoff_cumul   : moyenne cumulative du payoff défenseur
        - nash_value     : valeur théorique Nash (ligne horizontale)
        - def_freq_over_time  : fréquence cumulée de chaque action défense
        - att_freq_over_time  : fréquence cumulée de chaque stratégie attaque
        """
        if not self.history:
            return {}

        tours = [h['tour'] for h in self.history]
        payoffs = [h['payoff_defenseur'] for h in self.history]
        payoff_cumul = np.cumsum(payoffs) / np.arange(1, len(payoffs) + 1)

        M_def = self.game.get_payoff_matrix()
        nash_value = float(
            self.nash_result.defender_strategy @ M_def @ self.nash_result.attacker_strategy
        )

        # Fréquences cumulées défenseur
        n_def = len(self.nash_result.defender_labels)
        def_counts = np.zeros((len(self.history), n_def))
        for t, h in enumerate(self.history):
            if t > 0:
                def_counts[t] = def_counts[t - 1].copy()
            def_counts[t, h['action_def_idx']] += 1

        def_freq_over_time = def_counts / np.arange(1, len(self.history) + 1).reshape(-1, 1)

        # Fréquences cumulées attaquant
        n_att = len(self.nash_result.attacker_labels)
        att_counts = np.zeros((len(self.history), n_att))
        for t, h in enumerate(self.history):
            if t > 0:
                att_counts[t] = att_counts[t - 1].copy()
            att_counts[t, h['action_att_idx']] += 1

        att_freq_over_time = att_counts / np.arange(1, len(self.history) + 1).reshape(-1, 1)

        return {
            'tours': tours,
            'payoff_cumul': payoff_cumul.tolist(),
            'nash_value': nash_value,
            'def_freq_over_time': def_freq_over_time,
            'att_freq_over_time': att_freq_over_time,
        }

    def get_frequencies(self) -> tuple:
        df = pd.DataFrame(self.history)
        def_freq = df['defenseur'].value_counts(normalize=True)
        att_freq = df['attaquant'].value_counts(normalize=True)
        return def_freq, att_freq

    def convergence_check(self) -> dict | str:
        if len(self.history) < 10:
            return "Pas assez de tours pour vérifier la convergence."

        def_freq, att_freq = self.get_frequencies()

        def_theo = {label: prob for label, prob in
                    zip(self.nash_result.defender_labels, self.nash_result.defender_strategy)}
        att_theo = {label: prob for label, prob in
                    zip(self.nash_result.attacker_labels, self.nash_result.attacker_strategy)}

        def_error = sum(abs(def_freq.get(k, 0) - v) for k, v in def_theo.items())
        att_error = sum(abs(att_freq.get(k, 0) - v) for k, v in att_theo.items())

        return {
            'defender_error': def_error,
            'attacker_error': att_error,
            'converged': def_error < 0.1 and att_error < 0.1,
        }

    def reset(self):
        self.history = []
