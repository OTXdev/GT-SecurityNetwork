"""
main.py
=======
Script d'intégration CLI — GT-SecurityNetwork.

Lance tous les scénarios à travers Nash et Stackelberg,
affiche un rapport structuré et vérifie la propriété
Stackelberg ≥ Nash pour chaque scénario.

Usage :
    python main.py
"""

from __future__ import annotations

import sys
import textwrap

import numpy as np

from game_model import Game
from nash_solver import compute_nash
from stackelberg_solver import compute_stackelberg


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _bar(width: int = 70) -> str:
    return "─" * width

def _section(title: str) -> str:
    bar = _bar()
    return f"\n{bar}\n  {title}\n{bar}"

def _fmt_strategy(labels: list[str], probs: np.ndarray, threshold: float = 0.005) -> str:
    """Formate une stratégie mixte en lignes lisibles."""
    lines = []
    for label, prob in sorted(zip(labels, probs), key=lambda x: -x[1]):
        if prob >= threshold:
            bar_len = int(prob * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            lines.append(f"    {label:<35} {bar} {prob:.3f}")
    return "\n".join(lines) if lines else "    (aucune action active)"


# ─── Run principal ─────────────────────────────────────────────────────────────

def run() -> int:
    """
    Lance l'intégration complète. Retourne 0 si succès, 1 si une erreur surgit.
    """
    print("\n" + "═" * 70)
    print("  🛡️  GT-SecurityNetwork — Théorie des Jeux · Sécurité Réseau")
    print("═" * 70)

    all_ok = True
    scenarios = Game.get_scenarios()

    for scenario in scenarios:
        print(_section(f"📋 {scenario.title}"))
        print(f"\n  {textwrap.fill(scenario.description, width=66, subsequent_indent='  ')}\n")

        game  = scenario.game
        M_def = game.get_payoff_matrix()
        M_att = game.get_attacker_payoff_matrix()

        def_labels = game.get_defense_labels()
        att_labels  = game.get_attack_labels()

        print(f"  Matrice : {M_def.shape[0]} actions défense × {M_def.shape[1]} nœuds")
        print(f"  Taux succès défense     : {game.defense_success_rate:.0%}")
        print(f"  Propagation latérale    : {game.lateral_movement_factor:.0%}")

        # ── Nash ─────────────────────────────────────────────────────────────
        print(f"\n  {'─'*66}")
        print("  🎯 Équilibre de Nash (stratégies mixtes — minimax LP)")
        try:
            nash = compute_nash(M_def)
            print(f"\n  Valeur du jeu  V = {nash.game_value:+.4f}")
            print(f"\n  Stratégie du DÉFENSEUR :")
            print(_fmt_strategy(def_labels, nash.defender_strategy))
            print(f"\n  Stratégie de l'ATTAQUANT :")
            print(_fmt_strategy(att_labels, nash.attacker_strategy))
        except Exception as exc:
            print(f"  ❌ Erreur Nash : {exc}")
            all_ok = False
            nash = None

        # ── Stackelberg ───────────────────────────────────────────────────────
        print(f"\n  {'─'*66}")
        print("  👑 Jeu de Stackelberg (défenseur leader — sequential)")
        try:
            stack = compute_stackelberg(M_def, M_att)
            best_node = game.nodes[stack.attacker_best_response].name
            print(f"\n  Payoff défenseur  = {stack.defender_payoff:+.4f}")
            print(f"  Payoff attaquant  = {stack.attacker_payoff:+.4f}")
            print(f"  Best response att = {best_node}")
            print(f"\n  Engagement du DÉFENSEUR :")
            print(_fmt_strategy(def_labels, stack.defender_strategy))
        except Exception as exc:
            print(f"  ❌ Erreur Stackelberg : {exc}")
            all_ok = False
            stack = None

        # ── Comparaison ───────────────────────────────────────────────────────
        if nash and stack:
            print(f"\n  {'─'*66}")
            print("  ⚖️  Comparaison Nash vs Stackelberg")
            gain = stack.defender_payoff - nash.game_value
            gain_pct = (gain / abs(nash.game_value) * 100) if nash.game_value != 0 else float("inf")
            verdict = "✅ Propriété leader vérifiée" if gain >= -1e-4 else "❌ Propriété leader VIOLÉE"
            print(f"\n  Nash       : {nash.game_value:+.4f}")
            print(f"  Stackelberg: {stack.defender_payoff:+.4f}")
            print(f"  Gain       : {gain:+.4f}  ({gain_pct:+.1f}%)")
            print(f"  {verdict}")

    print("\n" + "═" * 70)
    status = "✅ Intégration réussie" if all_ok else "⚠️  Des erreurs ont été détectées"
    print(f"  {status}")
    print("═" * 70 + "\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(run())
