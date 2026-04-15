"""
tests/test_nash_solver.py
=========================
Tests unitaires pour nash_solver.py.

Vérifie :
  - Cas 2×2 avec solutions analytiques connues
  - Propriétés générales (Σp=1, Σq=1, non-négativité, bornes de valeur)
  - Propriété d'indifférence à l'équilibre
  - Stackelberg ≥ Nash sur les 3 scénarios
  - Gestion des entrées invalides
"""

import numpy as np
import pytest

from nash_solver import compute_nash, NashResult
from game_model import Game


# ─────────────────────────────────────────────────────────────────────────────
# Cas analytiques 2×2
# ─────────────────────────────────────────────────────────────────────────────

class TestNashKnownSolutions:
    """Solutions analytiques calculées à la main pour validation."""

    def test_diagonal_2x2(self):
        """
        M = [[4, 0],
             [0, 6]]

        Équilibrage des lignes :
          4p = 6(1-p) → 10p = 6 → p* = (0.6, 0.4)
          V  = 4 × 0.6 = 2.4

        Par symétrie : q* = (0.6, 0.4)
        """
        M = np.array([[4.0, 0.0], [0.0, 6.0]])
        r = compute_nash(M)

        np.testing.assert_allclose(r.defender_strategy, [0.6, 0.4], atol=1e-4)
        np.testing.assert_allclose(r.attacker_strategy, [0.6, 0.4], atol=1e-4)
        np.testing.assert_allclose(r.game_value, 2.4, atol=1e-4)

    def test_mixed_sign_matrix(self):
        """
        M = [[3, -1],
             [-1, 2]]

        V = (ad - bc) / (a + d - b - c)
          = (3×2 - (-1)×(-1)) / (3 + 2 - (-1) - (-1))
          = (6 - 1) / 7 = 5/7 ≈ 0.7143

        p* = (d-c) / (a+d-b-c) = (2-(-1)) / 7 = 3/7
        q* = (d-b) / (a+d-b-c) = (2-(-1)) / 7 = 3/7
        """
        M = np.array([[3.0, -1.0], [-1.0, 2.0]])
        r = compute_nash(M)

        np.testing.assert_allclose(r.game_value,          5.0 / 7.0, atol=1e-4)
        np.testing.assert_allclose(r.defender_strategy, [3/7, 4/7], atol=1e-4)
        np.testing.assert_allclose(r.attacker_strategy, [3/7, 4/7], atol=1e-4)

    def test_pure_strategy_dominant_row(self):
        """
        M = [[10, 10],
             [ 0,  0]]
        La ligne 0 domine strictement → le défenseur la joue avec probabilité 1.
        V = 10.
        """
        M = np.array([[10.0, 10.0], [0.0, 0.0]])
        r = compute_nash(M)

        assert r.defender_strategy[0] > 0.99
        np.testing.assert_allclose(r.game_value, 10.0, atol=1e-3)

    def test_pure_strategy_dominant_col(self):
        """
        M = [[5, 1],
             [5, 1]]
        L'attaquant préfère toujours la colonne 1 (min pour lui = max pour défenseur).
        V = 1.
        """
        M = np.array([[5.0, 1.0], [5.0, 1.0]])
        r = compute_nash(M)

        np.testing.assert_allclose(r.game_value, 1.0, atol=1e-3)
        assert r.attacker_strategy[1] > 0.99   # attaque toujours col 1

    def test_uniform_matrix(self):
        """Matrice constante → toute stratégie est équilibre, V = constante."""
        v = 3.0
        M = np.full((3, 3), v)
        r = compute_nash(M)

        np.testing.assert_allclose(r.game_value, v, atol=1e-4)
        np.testing.assert_allclose(r.defender_strategy.sum(), 1.0, atol=1e-8)
        np.testing.assert_allclose(r.attacker_strategy.sum(), 1.0, atol=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Propriétés générales
# ─────────────────────────────────────────────────────────────────────────────

MATRICES = [
    np.array([[4.0, 0.0], [0.0, 6.0]]),
    np.array([[3.0, -1.0], [-1.0, 2.0]]),
    np.array([[5.0, 2.0, 1.0], [1.0, 4.0, 3.0], [2.0, 1.0, 5.0]]),
    np.array([[0.0, -8.0, 4.0], [6.0, 3.0, -1.0]]),
]


class TestNashProperties:
    """Propriétés qui doivent tenir pour tout jeu à somme nulle."""

    @pytest.fixture(params=MATRICES)
    def matrix(self, request):
        return request.param

    def test_probabilities_sum_to_one(self, matrix):
        r = compute_nash(matrix)
        assert abs(r.defender_strategy.sum() - 1.0) < 1e-6, "Σ p_i ≠ 1"
        assert abs(r.attacker_strategy.sum() - 1.0) < 1e-6, "Σ q_j ≠ 1"

    def test_all_probabilities_nonneg(self, matrix):
        r = compute_nash(matrix)
        assert np.all(r.defender_strategy >= -1e-7), "p_i < 0 détecté"
        assert np.all(r.attacker_strategy >= -1e-7), "q_j < 0 détecté"

    def test_shapes_match_matrix(self, matrix):
        r = compute_nash(matrix)
        assert r.defender_strategy.shape == (matrix.shape[0],)
        assert r.attacker_strategy.shape == (matrix.shape[1],)

    def test_game_value_within_matrix_bounds(self, matrix):
        r = compute_nash(matrix)
        assert r.game_value >= matrix.min() - 1e-4, "V < min(M)"
        assert r.game_value <= matrix.max() + 1e-4, "V > max(M)"

    def test_indifference_property(self, matrix):
        """
        Propriété d'indifférence : pour toute attaque j dans le support de q*,
        le payoff espéré du défenseur est exactement V.

        p · M[:, j] ≈ V   pour tout j tel que q_j > 1e-3
        """
        r = compute_nash(matrix)
        for j, prob in enumerate(r.attacker_strategy):
            if prob > 1e-3:
                payoff_col = float(r.defender_strategy @ matrix[:, j])
                np.testing.assert_allclose(
                    payoff_col, r.game_value, atol=1e-3,
                    err_msg=(
                        f"Indifférence violée col {j}: "
                        f"p·M[:,{j}]={payoff_col:.4f} ≠ V={r.game_value:.4f}"
                    ),
                )

    def test_return_type(self, matrix):
        assert isinstance(compute_nash(matrix), NashResult)


# ─────────────────────────────────────────────────────────────────────────────
# Scénarios prédéfinis (tests de non-régression)
# ─────────────────────────────────────────────────────────────────────────────

class TestNashScenarios:
    """Vérifie que les 3 scénarios du projet passent sans erreur."""

    @pytest.fixture(params=[s.key for s in Game.get_scenarios()])
    def scenario(self, request):
        return next(s for s in Game.get_scenarios() if s.key == request.param)

    def test_no_crash(self, scenario):
        M = scenario.game.get_payoff_matrix()
        r = compute_nash(M)
        assert r is not None

    def test_valid_probabilities(self, scenario):
        M = scenario.game.get_payoff_matrix()
        r = compute_nash(M)
        assert abs(r.defender_strategy.sum() - 1.0) < 1e-5
        assert abs(r.attacker_strategy.sum() - 1.0) < 1e-5
        assert np.all(r.defender_strategy >= -1e-7)
        assert np.all(r.attacker_strategy >= -1e-7)

    def test_stackelberg_payoff_geq_nash_value(self, scenario):
        """
        Propriété fondamentale : le défenseur leader (Stackelberg)
        réalise toujours un payoff ≥ valeur Nash.
        """
        from stackelberg_solver import compute_stackelberg

        game   = scenario.game
        M_def  = game.get_payoff_matrix()
        M_att  = game.get_attacker_payoff_matrix()
        nash   = compute_nash(M_def)
        stack  = compute_stackelberg(M_def, M_att)

        assert stack.defender_payoff >= nash.game_value - 1e-3, (
            f"[{scenario.key}] Stackelberg ({stack.defender_payoff:.4f}) "
            f"< Nash ({nash.game_value:.4f}) — propriété du leader violée !"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Gestion des entrées invalides
# ─────────────────────────────────────────────────────────────────────────────

class TestNashErrors:

    def test_empty_array_raises(self):
        with pytest.raises(ValueError, match="2D non vide"):
            compute_nash(np.array([]))

    def test_1d_array_raises(self):
        with pytest.raises(ValueError):
            compute_nash(np.array([1.0, 2.0, 3.0]))

    def test_3d_array_raises(self):
        with pytest.raises(ValueError):
            compute_nash(np.zeros((2, 2, 2)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
