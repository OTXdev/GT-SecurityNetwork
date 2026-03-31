"""
tests/test_visualization.py
============================
Tests unitaires pour visualization.py uniquement.
"""

import unittest
import numpy as np
import matplotlib
matplotlib.use("Agg")  # pas d'écran nécessaire
import matplotlib.pyplot as plt

from game_model import Game, Node, Edge
from visualization import (
    draw_network,
    plot_payoff_heatmap,
    plot_strategy_bars,
    plot_solver_comparison,
    plot_attack_impact,
)


def is_figure(obj) -> bool:
    return isinstance(obj, plt.Figure)


class TestDrawNetwork(unittest.TestCase):

    def setUp(self):
        self.game = Game.sample_game()

    def test_returns_figure(self):
        """draw_network retourne une Figure matplotlib."""
        self.assertTrue(is_figure(draw_network(self.game)))

    def test_with_protected_and_attacked(self):
        """Fonctionne avec nœuds protégés et nœud attaqué."""
        names = [n.name for n in self.game.nodes]
        fig = draw_network(self.game, protected_nodes=[names[0]], attacked_node=names[1])
        self.assertTrue(is_figure(fig))

    def test_protected_and_attacked_same_node(self):
        """Un nœud peut être protégé ET attaqué en même temps (couleur orange)."""
        name = self.game.nodes[0].name
        fig = draw_network(self.game, protected_nodes=[name], attacked_node=name)
        self.assertTrue(is_figure(fig))

    def test_no_edges_fallback(self):
        """Sans arêtes, draw_network génère une chaîne par défaut."""
        game = Game(nodes=[Node("X", 5.0, 1.0), Node("Y", 4.0, 1.0)], edges=[])
        self.assertTrue(is_figure(draw_network(game)))

    def test_single_node(self):
        """Fonctionne avec un seul nœud."""
        game = Game(nodes=[Node("Alone", 5.0, 1.0)])
        self.assertTrue(is_figure(draw_network(game)))

    def test_no_protected_no_attacked(self):
        """Fonctionne sans aucun argument optionnel."""
        self.assertTrue(is_figure(draw_network(self.game, None, None)))


class TestPlotPayoffHeatmap(unittest.TestCase):

    def setUp(self):
        self.game = Game.sample_game()
        self.M    = self.game.get_payoff_matrix()

    def test_returns_figure(self):
        fig = plot_payoff_heatmap(
            self.M, self.game.get_defense_labels(), self.game.get_attack_labels()
        )
        self.assertTrue(is_figure(fig))

    def test_custom_title(self):
        """Le titre personnalisé est accepté sans erreur."""
        fig = plot_payoff_heatmap(
            self.M, self.game.get_defense_labels(), self.game.get_attack_labels(),
            title="Test titre"
        )
        self.assertTrue(is_figure(fig))

    def test_attacker_matrix(self):
        """Fonctionne aussi avec la matrice attaquant (valeurs positives)."""
        M_att = self.game.get_attacker_payoff_matrix()
        fig = plot_payoff_heatmap(
            M_att, self.game.get_defense_labels(), self.game.get_attack_labels(),
            title="Payoff attaquant"
        )
        self.assertTrue(is_figure(fig))

    def test_2x2_matrix(self):
        """Fonctionne avec une petite matrice 2x2."""
        M = np.array([[-3.0, -7.0], [-1.0, -9.0]])
        fig = plot_payoff_heatmap(M, ["D1", "D2"], ["A1", "A2"])
        self.assertTrue(is_figure(fig))


class TestPlotStrategyBars(unittest.TestCase):

    def setUp(self):
        self.game = Game.sample_game()
        self.labels = self.game.get_defense_labels()
        n = len(self.labels)
        self.uniform = np.ones(n) / n

    def test_returns_figure(self):
        fig = plot_strategy_bars(self.labels, self.uniform, "Test")
        self.assertTrue(is_figure(fig))

    def test_sparse_strategy(self):
        """Une seule action à probabilité 1.0."""
        sparse = np.zeros(len(self.labels))
        sparse[0] = 1.0
        fig = plot_strategy_bars(self.labels, sparse, "Sparse")
        self.assertTrue(is_figure(fig))

    def test_all_zeros_except_one(self):
        """Toutes les probas sous le seuil sauf une."""
        values = np.full(len(self.labels), 0.001)
        values[2] = 1.0 - values.sum() + values[2]
        fig = plot_strategy_bars(self.labels, values, "Quasi-sparse")
        self.assertTrue(is_figure(fig))

    def test_custom_color(self):
        """La couleur personnalisée est acceptée."""
        fig = plot_strategy_bars(self.labels, self.uniform, "Couleur", color="#e63946")
        self.assertTrue(is_figure(fig))

    def test_two_actions(self):
        """Fonctionne avec seulement 2 actions."""
        fig = plot_strategy_bars(["A", "B"], np.array([0.4, 0.6]), "Deux actions")
        self.assertTrue(is_figure(fig))


class TestPlotSolverComparison(unittest.TestCase):

    def test_returns_figure(self):
        fig = plot_solver_comparison(-8.15, -7.93)
        self.assertTrue(is_figure(fig))

    def test_equal_values(self):
        """Nash = Stackelberg — gain nul, pas d'erreur."""
        fig = plot_solver_comparison(-5.0, -5.0)
        self.assertTrue(is_figure(fig))

    def test_positive_values(self):
        """Valeurs positives (cas limite) ne lèvent pas d'erreur."""
        fig = plot_solver_comparison(2.0, 3.5)
        self.assertTrue(is_figure(fig))

    def test_large_gap(self):
        """Grand écart entre Nash et Stackelberg."""
        fig = plot_solver_comparison(-20.0, -5.0)
        self.assertTrue(is_figure(fig))


class TestPlotAttackImpact(unittest.TestCase):

    def test_returns_figure(self):
        fig = plot_attack_impact(Game.sample_game())
        self.assertTrue(is_figure(fig))

    def test_all_scenarios(self):
        """Fonctionne sur les 3 scénarios."""
        for scenario in Game.get_scenarios():
            fig = plot_attack_impact(scenario.game)
            self.assertTrue(is_figure(fig), f"Echec sur {scenario.key}")

    def test_high_value_node_has_higher_impact(self):
        """Sans arêtes ni latéralité, le nœud avec la plus haute valeur
        a l'impact le plus élevé."""
        game = Game(
            nodes=[
                Node("Low",  attack_value=2.0,  defense_cost=1.0, vulnerability=1.0),
                Node("High", attack_value=10.0, defense_cost=1.0, vulnerability=1.0),
            ],
            edges=[],
            lateral_movement_factor=0.0,
        )
        self.assertGreater(
            game.compute_attack_impact("High"),
            game.compute_attack_impact("Low"),
        )

    def test_single_node_game(self):
        """Fonctionne avec un seul nœud."""
        game = Game(nodes=[Node("Only", 5.0, 1.0)])
        fig = plot_attack_impact(game)
        self.assertTrue(is_figure(fig))


if __name__ == "__main__":
    unittest.main(verbosity=2)
