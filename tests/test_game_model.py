"""
tests/test_game_model.py
========================
Tests unitaires pour game_model.py uniquement.
"""

import unittest
import numpy as np
from game_model import Game, Node, Edge, DefenseAction


class TestNode(unittest.TestCase):

    def test_valid_node(self):
        """Un nœud valide se crée sans erreur."""
        node = Node("Server", attack_value=10.0, defense_cost=2.0,
                    vulnerability=0.8, attack_cost=1.0, criticality=1.2)
        self.assertEqual(node.name, "Server")
        self.assertEqual(node.attack_value, 10.0)

    def test_negative_attack_value_raises(self):
        """attack_value négative doit lever ValueError."""
        with self.assertRaises(ValueError):
            Node("Bad", attack_value=-1.0, defense_cost=1.0)

    def test_vulnerability_too_high_raises(self):
        """vulnerability > 1 doit lever ValueError."""
        with self.assertRaises(ValueError):
            Node("Bad", attack_value=5.0, defense_cost=1.0, vulnerability=1.5)

    def test_vulnerability_negative_raises(self):
        """vulnerability < 0 doit lever ValueError."""
        with self.assertRaises(ValueError):
            Node("Bad", attack_value=5.0, defense_cost=1.0, vulnerability=-0.1)

    def test_negative_criticality_raises(self):
        """criticality négative doit lever ValueError."""
        with self.assertRaises(ValueError):
            Node("Bad", attack_value=5.0, defense_cost=1.0, criticality=-1.0)

    def test_default_values(self):
        """Les valeurs par défaut sont correctes."""
        node = Node("Default", attack_value=5.0, defense_cost=1.0)
        self.assertEqual(node.vulnerability, 1.0)
        self.assertEqual(node.attack_cost, 0.0)
        self.assertEqual(node.criticality, 1.0)

    def test_boundary_vulnerability_zero(self):
        """vulnerability = 0 est valide."""
        node = Node("Zero", attack_value=5.0, defense_cost=1.0, vulnerability=0.0)
        self.assertEqual(node.vulnerability, 0.0)

    def test_boundary_vulnerability_one(self):
        """vulnerability = 1 est valide."""
        node = Node("Full", attack_value=5.0, defense_cost=1.0, vulnerability=1.0)
        self.assertEqual(node.vulnerability, 1.0)


class TestEdge(unittest.TestCase):

    def test_valid_edge(self):
        """Une arête valide se crée sans erreur."""
        edge = Edge("A", "B", weight=0.7)
        self.assertEqual(edge.source, "A")
        self.assertEqual(edge.target, "B")
        self.assertEqual(edge.weight, 0.7)

    def test_weight_too_high_raises(self):
        """weight > 1 doit lever ValueError."""
        with self.assertRaises(ValueError):
            Edge("A", "B", weight=1.5)

    def test_weight_negative_raises(self):
        """weight < 0 doit lever ValueError."""
        with self.assertRaises(ValueError):
            Edge("A", "B", weight=-0.1)

    def test_default_weight(self):
        """Le poids par défaut est 1.0."""
        self.assertEqual(Edge("A", "B").weight, 1.0)

    def test_boundary_weight_zero(self):
        """weight = 0 est valide."""
        self.assertEqual(Edge("A", "B", weight=0.0).weight, 0.0)


class TestGameConstruction(unittest.TestCase):

    def setUp(self):
        self.nodes = [
            Node("A", attack_value=10.0, defense_cost=2.0, vulnerability=0.9),
            Node("B", attack_value=6.0,  defense_cost=1.5, vulnerability=0.7),
        ]

    def test_empty_nodes_raises(self):
        with self.assertRaises(ValueError):
            Game(nodes=[])

    def test_defense_success_rate_too_high_raises(self):
        with self.assertRaises(ValueError):
            Game(nodes=self.nodes, defense_success_rate=1.5)

    def test_defense_success_rate_negative_raises(self):
        with self.assertRaises(ValueError):
            Game(nodes=self.nodes, defense_success_rate=-0.1)

    def test_lateral_factor_too_high_raises(self):
        with self.assertRaises(ValueError):
            Game(nodes=self.nodes, lateral_movement_factor=2.0)

    def test_unknown_node_in_edge_raises(self):
        with self.assertRaises(ValueError):
            Game(nodes=self.nodes, edges=[Edge("A", "UNKNOWN")])

    def test_unknown_node_in_defense_action_raises(self):
        bad = DefenseAction("Bad", protected_nodes=("GHOST",), cost=1.0)
        with self.assertRaises(ValueError):
            Game(nodes=self.nodes, defense_actions=[bad])

    def test_default_actions_no_defense(self):
        self.assertIn("No defense", Game(nodes=self.nodes).get_defense_labels())

    def test_default_actions_single_protections(self):
        labels = Game(nodes=self.nodes).get_defense_labels()
        self.assertIn("Protect A", labels)
        self.assertIn("Protect B", labels)

    def test_default_actions_pairs(self):
        self.assertIn("Protect A + B", Game(nodes=self.nodes).get_defense_labels())

    def test_no_pair_defense_flag(self):
        game = Game(nodes=self.nodes, allow_pair_defense=False)
        self.assertNotIn("Protect A + B", game.get_defense_labels())

    def test_get_node_valid(self):
        self.assertEqual(Game(nodes=self.nodes).get_node("A").name, "A")

    def test_get_node_unknown_raises(self):
        with self.assertRaises(KeyError):
            Game(nodes=self.nodes).get_node("UNKNOWN")


class TestPayoffMatrix(unittest.TestCase):

    def setUp(self):
        self.game = Game(
            nodes=[
                Node("X", attack_value=8.0, defense_cost=2.0,
                     vulnerability=1.0, attack_cost=0.0, criticality=1.0),
                Node("Y", attack_value=5.0, defense_cost=1.0,
                     vulnerability=1.0, attack_cost=0.0, criticality=1.0),
            ],
            edges=[],
            defense_success_rate=1.0,
            lateral_movement_factor=0.0,
        )

    def test_defender_matrix_shape(self):
        M = self.game.get_payoff_matrix()
        self.assertEqual(M.shape, (len(self.game.defense_actions), len(self.game.nodes)))

    def test_attacker_matrix_same_shape(self):
        self.assertEqual(
            self.game.get_payoff_matrix().shape,
            self.game.get_attacker_payoff_matrix().shape,
        )

    def test_defender_values_non_positive(self):
        self.assertTrue(np.all(self.game.get_payoff_matrix() <= 1e-9))

    def test_no_defense_is_worst(self):
        M = self.game.get_payoff_matrix()
        no_def = self.game.get_defense_labels().index("No defense")
        for col in range(M.shape[1]):
            self.assertLessEqual(M[no_def, col], M[:, col].max() + 1e-9)

    def test_perfect_defense_reduces_loss(self):
        M  = self.game.get_payoff_matrix()
        dl = self.game.get_defense_labels()
        al = self.game.get_attack_labels()
        self.assertGreater(
            M[dl.index("Protect X"), al.index("Attack X")],
            M[dl.index("No defense"), al.index("Attack X")],
        )

    def test_attacker_gains_without_defense(self):
        M_att  = self.game.get_attacker_payoff_matrix()
        no_def = self.game.get_defense_labels().index("No defense")
        self.assertTrue(np.all(M_att[no_def, :] >= 0))

    def test_no_nan_or_inf(self):
        M = self.game.get_payoff_matrix()
        self.assertFalse(np.any(np.isnan(M)))
        self.assertFalse(np.any(np.isinf(M)))


class TestLateralMovement(unittest.TestCase):

    def _game(self, edges, lateral):
        return Game(
            nodes=[
                Node("A", attack_value=10.0, defense_cost=2.0, vulnerability=1.0),
                Node("B", attack_value=5.0,  defense_cost=1.0, vulnerability=1.0),
            ],
            edges=edges, lateral_movement_factor=lateral,
        )

    def test_no_edges_no_lateral(self):
        self.assertAlmostEqual(
            self._game([], 0.5).compute_attack_impact("A"), 10.0, places=6)

    def test_edge_increases_impact(self):
        self.assertGreater(
            self._game([Edge("A", "B", weight=1.0)], 0.5).compute_attack_impact("A"),
            self._game([], 0.5).compute_attack_impact("A"),
        )

    def test_zero_lateral_factor_disables_propagation(self):
        self.assertAlmostEqual(
            self._game([Edge("A", "B", weight=1.0)], 0.0).compute_attack_impact("A"),
            10.0, places=6,
        )

    def test_impact_proportional_to_edge_weight(self):
        self.assertGreater(
            self._game([Edge("A", "B", weight=0.8)], 0.5).compute_attack_impact("A"),
            self._game([Edge("A", "B", weight=0.2)], 0.5).compute_attack_impact("A"),
        )


class TestScenarios(unittest.TestCase):

    def test_three_scenarios(self):
        self.assertEqual(len(Game.get_scenarios()), 3)

    def test_scenario_keys_unique(self):
        keys = [s.key for s in Game.get_scenarios()]
        self.assertEqual(len(keys), len(set(keys)))

    def test_all_scenarios_valid_matrix(self):
        for s in Game.get_scenarios():
            M = s.game.get_payoff_matrix()
            self.assertFalse(np.any(np.isnan(M)), f"NaN dans {s.key}")
            self.assertFalse(np.any(np.isinf(M)), f"Inf dans {s.key}")

    def test_sample_game_returns_game(self):
        self.assertIsInstance(Game.sample_game(), Game)

    def test_from_values_node_count(self):
        game = Game.from_values(["S1","S2","S3"], [10.0,8.0,6.0], [2.0,1.5,1.0])
        self.assertEqual(len(game.nodes), 3)

    def test_from_values_matrix_columns(self):
        game = Game.from_values(["S1","S2"], [10.0,8.0], [2.0,1.5])
        self.assertEqual(game.get_payoff_matrix().shape[1], 2)

    def test_describe_contains_node_names(self):
        game = Game.sample_game()
        desc = game.describe()
        for node in game.nodes:
            self.assertIn(node.name, desc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
