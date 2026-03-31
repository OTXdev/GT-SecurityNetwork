"""
tests/test_stackelberg_solver.py
=================================
Tests unitaires pour stackelberg_solver.py uniquement.
"""

import unittest
import numpy as np
from stackelberg_solver import compute_stackelberg, StackelbergResult


class TestStackelbergInputValidation(unittest.TestCase):

    def test_1d_matrix_raises(self):
        """Un tableau 1D doit lever ValueError."""
        with self.assertRaises(ValueError):
            compute_stackelberg(np.array([1.0, 2.0, 3.0]))

    def test_empty_matrix_raises(self):
        """Une matrice vide doit lever ValueError."""
        with self.assertRaises(ValueError):
            compute_stackelberg(np.empty((0, 0)))

    def test_mismatched_attacker_matrix_raises(self):
        """attacker_payoff_matrix de shape différente doit lever ValueError."""
        M = np.array([[-5.0, -8.0], [-3.0, -9.0]])
        with self.assertRaises(ValueError):
            compute_stackelberg(M, np.array([[-5.0, -8.0, -1.0]]))


class TestStackelbergStrategy(unittest.TestCase):

    def setUp(self):
        """Jeu 2x2 simple avec solution connue."""
        # Défenseur perd plus si nœud 1 est attaqué sans défense
        self.M_def = np.array([
            [-2.0, -5.0],   # No defense
            [-1.0, -8.0],   # Protect node 0
            [-4.0, -1.0],   # Protect node 1
        ])
        self.M_att = -self.M_def  # jeu à somme nulle

    def test_returns_stackelberg_result(self):
        """Le résultat est bien un StackelbergResult."""
        result = compute_stackelberg(self.M_def, self.M_att)
        self.assertIsInstance(result, StackelbergResult)

    def test_strategy_sums_to_one(self):
        """La somme des probabilités du défenseur vaut 1."""
        result = compute_stackelberg(self.M_def, self.M_att)
        self.assertAlmostEqual(result.defender_strategy.sum(), 1.0, places=7)

    def test_strategy_non_negative(self):
        """Aucune probabilité n'est négative."""
        result = compute_stackelberg(self.M_def, self.M_att)
        self.assertTrue(np.all(result.defender_strategy >= -1e-9))

    def test_strategy_shape(self):
        """La stratégie a autant d'éléments que de lignes dans la matrice."""
        result = compute_stackelberg(self.M_def, self.M_att)
        self.assertEqual(len(result.defender_strategy), self.M_def.shape[0])

    def test_best_response_valid_index(self):
        """La meilleure réponse est un indice valide de colonne."""
        result = compute_stackelberg(self.M_def, self.M_att)
        self.assertGreaterEqual(result.attacker_best_response, 0)
        self.assertLess(result.attacker_best_response, self.M_def.shape[1])

    def test_defender_payoff_consistent(self):
        """defender_payoff == stratégie @ colonne de la meilleure réponse."""
        result = compute_stackelberg(self.M_def, self.M_att)
        j = result.attacker_best_response
        expected = result.defender_strategy @ self.M_def[:, j]
        self.assertAlmostEqual(result.defender_payoff, expected, places=7)

    def test_attacker_payoff_consistent(self):
        """attacker_payoff == stratégie @ colonne attaquant de la BR."""
        result = compute_stackelberg(self.M_def, self.M_att)
        j = result.attacker_best_response
        expected = result.defender_strategy @ self.M_att[:, j]
        self.assertAlmostEqual(result.attacker_payoff, expected, places=7)


class TestStackelbergZeroSum(unittest.TestCase):

    def setUp(self):
        """Jeu à somme nulle — attacker_payoff_matrix omis."""
        self.M_def = np.array([
            [-3.0, -7.0, -5.0],
            [-1.0, -9.0, -6.0],
            [-5.0, -2.0, -4.0],
            [-6.0, -4.0, -1.0],
        ])

    def test_zero_sum_assumption(self):
        """Sans attacker_payoff_matrix, le jeu est supposé à somme nulle."""
        result = compute_stackelberg(self.M_def)
        self.assertAlmostEqual(
            result.defender_payoff + result.attacker_payoff, 0.0, places=6
        )

    def test_strategy_sums_to_one(self):
        result = compute_stackelberg(self.M_def)
        self.assertAlmostEqual(result.defender_strategy.sum(), 1.0, places=7)

    def test_non_negative_probabilities(self):
        result = compute_stackelberg(self.M_def)
        self.assertTrue(np.all(result.defender_strategy >= -1e-9))


class TestStackelbergVsNash(unittest.TestCase):
    """Compare Stackelberg vs Nash sans dépendre de nash_solver.

    On vérifie que le payoff Stackelberg est au moins aussi bon que
    le meilleur payoff garanti par n'importe quelle stratégie PURE du défenseur.
    (Nash >= meilleure stratégie pure, et Stackelberg >= Nash)
    """

    def _best_pure_defense_payoff(self, M_def: np.ndarray) -> float:
        """Meilleur payoff garanti par une stratégie PURE du défenseur.

        Pour chaque ligne i, le pire cas = min sur les colonnes.
        On prend le max de ces pires cas.
        C'est une borne inférieure de la valeur Nash.
        """
        return float(max(M_def[i, :].min() for i in range(M_def.shape[0])))

    def test_stackelberg_geq_best_pure_defense(self):
        """Stackelberg >= meilleure stratégie pure du défenseur (sur 3 scénarios)."""
        from game_model import Game

        for scenario in Game.get_scenarios():
            M_def = scenario.game.get_payoff_matrix()
            M_att = scenario.game.get_attacker_payoff_matrix()
            stack = compute_stackelberg(M_def, M_att)
            bound = self._best_pure_defense_payoff(M_def)
            self.assertGreaterEqual(
                stack.defender_payoff,
                bound - 1e-6,
                msg=f"Stackelberg < meilleure défense pure dans '{scenario.key}'",
            )


class TestStackelbergSingleNode(unittest.TestCase):

    def test_single_column_game(self):
        """Un jeu avec un seul nœud (une seule colonne) fonctionne."""
        M_def = np.array([[-5.0], [-2.0], [-3.0]])
        result = compute_stackelberg(M_def)
        self.assertAlmostEqual(result.defender_strategy.sum(), 1.0, places=7)
        self.assertEqual(result.attacker_best_response, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
