"""
visualization.py
================
Fonctions de visualisation pour le simulateur de sécurité réseau.

Graphiques disponibles
----------------------
draw_network          : topologie du réseau (nœuds colorés par état)
plot_payoff_heatmap   : matrice de payoff en heatmap annotée
plot_strategy_bars    : probabilités d'une stratégie mixte en barres
plot_solver_comparison: comparaison Nash vs Stackelberg
plot_attack_impact    : impact brut de chaque nœud (barre horizontale)

Chaque fonction retourne un objet Figure matplotlib prêt à être
intégré dans tkinter (FigureCanvasTkAgg) ou sauvegardé.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

from game_model import Game

# ── Palette de couleurs cohérente ────────────────────────────────────────────
COLOR_NEUTRAL   = "#8d99ae"   # nœud non impliqué
COLOR_PROTECTED = "#2a9d8f"   # nœud protégé (vert-bleu)
COLOR_ATTACKED  = "#e63946"   # nœud attaqué (rouge)
COLOR_BOTH      = "#f4a261"   # nœud protégé ET attaqué (orange)
COLOR_BAR       = "#457b9d"   # barres génériques
COLOR_NASH      = "#6c757d"   # barre Nash
COLOR_STACK     = "#2a9d8f"   # barre Stackelberg
COLOR_EDGE      = "#adb5bd"   # arêtes du réseau


# ─────────────────────────────────────────────────────────────────────────────
# 1. Topologie réseau
# ─────────────────────────────────────────────────────────────────────────────

def draw_network(
    game: Game,
    protected_nodes: list[str] | None = None,
    attacked_node: str | None = None,
) -> plt.Figure:
    """Affiche la topologie du réseau avec les nœuds colorés par état.

    Paramètres
    ----------
    game            : le jeu (nœuds + arêtes)
    protected_nodes : liste des nœuds protégés par l'action de défense active
    attacked_node   : nœud ciblé par l'attaquant (None = pas de simulation)

    Légende des couleurs
    --------------------
    Gris    → nœud neutre
    Vert    → protégé
    Rouge   → attaqué
    Orange  → protégé mais attaqué quand même
    """
    protected = set(protected_nodes or [])

    # ── Construction du graphe NetworkX ──────────────────────────────────
    graph = nx.Graph()
    for node in game.nodes:
        graph.add_node(node.name, value=node.attack_value)
    for edge in game.edges:
        graph.add_edge(edge.source, edge.target, weight=edge.weight)

    # Si pas d'arêtes, chaîne linéaire par défaut (pour l'affichage)
    if graph.number_of_edges() == 0:
        for i in range(len(game.nodes) - 1):
            graph.add_edge(game.nodes[i].name, game.nodes[i + 1].name)

    # ── Couleurs et tailles des nœuds ─────────────────────────────────────
    node_colors = []
    node_sizes  = []
    for node_name in graph.nodes:
        is_protected = node_name in protected
        is_attacked  = node_name == attacked_node
        if is_protected and is_attacked:
            node_colors.append(COLOR_BOTH)
        elif is_attacked:
            node_colors.append(COLOR_ATTACKED)
        elif is_protected:
            node_colors.append(COLOR_PROTECTED)
        else:
            node_colors.append(COLOR_NEUTRAL)

        # Taille proportionnelle à la valeur du nœud
        node = game.get_node(node_name)
        node_sizes.append(400 + node.attack_value * 50)

    # ── Épaisseur des arêtes proportionnelle au poids ─────────────────────
    edge_weights = [
        graph[u][v].get("weight", 0.5) * 3
        for u, v in graph.edges
    ]

    # ── Dessin ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    pos = nx.spring_layout(graph, seed=42)

    nx.draw_networkx(
        graph,
        pos=pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color=COLOR_EDGE,
        width=edge_weights,
        with_labels=True,
        font_size=9,
        font_weight="bold",
    )

    # Légende
    legend_items = [
        mpatches.Patch(color=COLOR_NEUTRAL,   label="Neutre"),
        mpatches.Patch(color=COLOR_PROTECTED, label="Protégé"),
        mpatches.Patch(color=COLOR_ATTACKED,  label="Attaqué"),
        mpatches.Patch(color=COLOR_BOTH,      label="Protégé + attaqué"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=7, framealpha=0.8)
    ax.set_title("Topologie du réseau", fontsize=11, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Heatmap de la matrice de payoff
# ─────────────────────────────────────────────────────────────────────────────

def plot_payoff_heatmap(
    payoff_matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str = "Matrice de payoff",
) -> plt.Figure:
    """Affiche la matrice de payoff sous forme de heatmap annotée.

    Les cellules sont colorées du bleu (meilleur pour le défenseur)
    au rouge (pire pour le défenseur).
    Les valeurs sont affichées dans chaque cellule.
    """
    matrix = np.asarray(payoff_matrix, dtype=float)
    n_rows, n_cols = matrix.shape

    # Hauteur dynamique selon le nombre de lignes
    fig_height = max(4, n_rows * 0.45 + 1.5)
    fig, ax = plt.subplots(figsize=(7, fig_height))

    heatmap = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")

    # Axes
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)

    # Annotations dans chaque cellule
    vmin, vmax = matrix.min(), matrix.max()
    mid = (vmin + vmax) / 2
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            # Texte noir sur fond clair, blanc sur fond foncé
            text_color = "black" if val > mid else "white"
            ax.text(
                j, i, f"{val:.1f}",
                ha="center", va="center",
                color=text_color, fontsize=7.5
            )

    fig.colorbar(heatmap, ax=ax, shrink=0.8, label="Utilité défenseur")
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Barres d'une stratégie mixte
# ─────────────────────────────────────────────────────────────────────────────

def plot_strategy_bars(
    labels: list[str],
    values: np.ndarray,
    title: str,
    color: str = COLOR_BAR,
) -> plt.Figure:
    """Affiche les probabilités d'une stratégie mixte sous forme de barres.

    Seules les actions avec probabilité > 0.01 sont affichées pour la lisibilité.
    Les autres sont regroupées sous 'Autres (< 1%)'.
    """
    values = np.asarray(values, dtype=float)

    # Filtrer les actions à probabilité non nulle
    nonzero_mask = values > 0.01
    filtered_labels = [l for l, m in zip(labels, nonzero_mask) if m]
    filtered_values = values[nonzero_mask]

    residual = 1.0 - filtered_values.sum()
    if residual > 0.005:
        filtered_labels.append("Autres (< 1%)")
        filtered_values = np.append(filtered_values, residual)

    n = len(filtered_labels)
    fig_width = max(5, n * 0.9 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 4))

    bars = ax.bar(range(n), filtered_values, color=color, edgecolor="white", linewidth=0.8)

    # Annotation au-dessus de chaque barre
    for idx, (bar, val) in enumerate(zip(bars, filtered_values)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=8
        )

    ax.set_xticks(range(n))
    ax.set_xticklabels(filtered_labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Probabilité", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.5)

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Comparaison Nash vs Stackelberg
# ─────────────────────────────────────────────────────────────────────────────

def plot_solver_comparison(
    nash_value: float,
    stackelberg_value: float,
) -> plt.Figure:
    """Compare les payoffs défenseur obtenus par Nash et Stackelberg.

    Un payoff plus élevé (moins négatif) est meilleur pour le défenseur.
    L'écart représente le 'gain du leadership' de Stackelberg.
    """
    labels = ["Nash\n(simultané)", "Stackelberg\n(séquentiel)"]
    values = [nash_value, stackelberg_value]
    colors = [COLOR_NASH, COLOR_STACK]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)

    # Annotation des valeurs
    for bar, val in zip(bars, values):
        offset = 0.15 if val >= 0 else -0.35
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + offset,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    # Flèche indiquant le gain de leadership
    gain = stackelberg_value - nash_value
    if abs(gain) > 0.01:
        ax.annotate(
            f"Gain : +{gain:.2f}",
            xy=(1, stackelberg_value),
            xytext=(0, stackelberg_value + abs(gain) * 0.6 + 0.5),
            arrowprops=dict(arrowstyle="->", color="black"),
            ha="center", fontsize=9, color="black",
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Payoff du défenseur", fontsize=9)
    ax.set_title("Nash vs Stackelberg\n(plus haut = meilleur pour le défenseur)", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Impact brut de chaque nœud (bonus)
# ─────────────────────────────────────────────────────────────────────────────

def plot_attack_impact(game: Game) -> plt.Figure:
    """Affiche l'impact total d'une attaque sur chaque nœud (direct + latéral).

    Utile pour visualiser quels nœuds sont les plus dangereux à compromettre,
    indépendamment des stratégies.
    """
    node_names = [node.name for node in game.nodes]
    impacts    = [game.compute_attack_impact(name) for name in node_names]

    # Trier par impact décroissant
    sorted_pairs = sorted(zip(impacts, node_names), reverse=True)
    sorted_impacts, sorted_names = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors_bar = [COLOR_ATTACKED if i == 0 else COLOR_BAR for i in range(len(sorted_names))]
    bars = ax.barh(sorted_names, sorted_impacts, color=colors_bar, edgecolor="white")

    for bar, val in zip(bars, sorted_impacts):
        ax.text(
            val + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center", fontsize=8
        )

    ax.set_xlabel("Impact total (direct + latéral)", fontsize=9)
    ax.set_title("Impact d'attaque par nœud", fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig
