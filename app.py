"""
app.py
======
GT-SecurityNetwork — Théorie des Jeux & Sécurité Réseau
Interface Streamlit professionnelle.

Onglets :
  1. Réseau           — topologie, métriques, simulateur d'attaque
  2. Matrice          — heatmap des payoffs
  3. Nash             — équilibre Nash stratégies mixtes
  4. Stackelberg      — jeu séquentiel leader/follower
  5. Pareto           — frontière de Pareto & efficacité sociale
  6. Comparaison      — Nash vs Stackelberg vs Optimal centralisé
  7. Simulation       — convergence vers l'équilibre
  8. État de l'art    — revue de la littérature
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import streamlit as st

from game_model import Game, Node, Edge
from nash_solver import NashSolver, NashResult
from stackelberg_solver import compute_stackelberg, StackelbergResult
from pareto import compute_pareto, pareto_efficiency_score
from simulator import GameSimulator

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="GT-SecurityNetwork",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS — PROFESSIONAL LIGHT/DARK THEME
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ============================================
   DARK THEME - GT-SecurityNetwork
   ============================================ */

:root {
    --bg-deep: #0a0c12;
    --bg-surface: #11131a;
    --bg-surface-hover: #1a1d26;
    --border-subtle: #232630;
    --border-medium: #2d313e;
    --text-primary: #e8edf2;
    --text-secondary: #94a3b8;
    --text-muted: #5b6b8c;
    
    --accent-blue: #3b82f6;
    --accent-blue-glow: #2563eb;
    --accent-red: #ef4444;
    --accent-green: #10b981;
    --accent-amber: #f59e0b;
    --accent-purple: #8b5cf6;
}

/* Base */
html, body, .stApp {
    font-family: 'DM Sans', sans-serif !important;
    background: var(--bg-deep) !important;
}

[data-testid="stAppViewContainer"] { 
    background: var(--bg-deep) !important;
}

[data-testid="stHeader"] { 
    background: var(--bg-deep) !important;
    border-bottom: 1px solid var(--border-subtle) !important;
}

section[data-testid="stSidebar"] > div {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

#MainMenu, footer { display: none !important; }

/* Couleurs texte globales */
* {
    color: var(--text-primary) !important;
}

/* Exceptions pour les badges */
.kpi-value.blue { color: #60a5fa !important; }
.kpi-value.green { color: #34d399 !important; }
.kpi-value.amber { color: #fbbf24 !important; }
.kpi-value.red { color: #f87171 !important; }
.badge { color: inherit !important; }
.b-blue { color: #60a5fa !important; background: #1e3a5f !important; }
.b-green { color: #34d399 !important; background: #14532d !important; }
.b-amber { color: #fbbf24 !important; background: #78350f !important; }
.b-red { color: #f87171 !important; background: #7f1d1d !important; }
.b-purple { color: #c084fc !important; background: #3b0764 !important; }

/* Sidebar brand */
.sidebar-brand {
    padding: 1.5rem 0 1.2rem;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 1.2rem;
    text-align: center;
}
.sidebar-brand h1 {
    font-size: 1.1rem;
    font-weight: 700;
    margin: 0.4rem 0 0.1rem;
}
.sidebar-brand p {
    font-size: 0.72rem;
    margin: 0;
    text-transform: uppercase;
    opacity: 0.6;
}

/* Page header */
.page-header {
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 2rem;
}
.page-title {
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0;
}
.page-sub {
    font-size: 0.88rem;
    margin: 0.3rem 0 0;
    opacity: 0.6;
}

/* KPI Row */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.kpi-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1.25rem 1.2rem;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, #3b82f6);
    border-radius: 12px 12px 0 0;
}
.kpi-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-bottom: 0.5rem;
    opacity: 0.6;
}
.kpi-value {
    font-size: 1.9rem;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
    line-height: 1;
}

/* Section title */
.sec-title {
    font-size: 1rem;
    font-weight: 600;
    margin: 1.8rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.sec-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border-subtle);
}

/* Cards */
.card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin: 0.75rem 0;
}
.card.blue { border-left: 3px solid var(--accent-blue); }
.card.green { border-left: 3px solid var(--accent-green); }
.card.amber { border-left: 3px solid var(--accent-amber); }
.card.red { border-left: 3px solid var(--accent-red); }
.card.purple { border-left: 3px solid var(--accent-purple); }

/* Badges */
.badge {
    display: inline-block;
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 0.15rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    padding: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.45rem 0.9rem !important;
}
.stTabs [aria-selected="true"] {
    background: #1e293b !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* Buttons */
.stButton > button {
    background: var(--accent-blue) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
}
.stButton > button:hover {
    background: var(--accent-blue-glow) !important;
    transform: translateY(-1px) !important;
}

/* Inputs */
.stTextInput input, .stNumberInput input {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
}
.stSelectbox > div > div {
    background: var(--bg-surface) !important;
    border-color: var(--border-subtle) !important;
}

/* Dataframe - DARK MODE */
.stDataFrame {
    border-radius: 10px !important;
    border: 1px solid var(--border-subtle);
    background: var(--bg-surface) !important;
}
.stDataFrame th {
    background: var(--bg-surface-hover) !important;
    font-weight: 600 !important;
}
.stDataFrame td {
    background: var(--bg-surface) !important;
}
.stDataFrame tbody tr:nth-of-type(even) {
    background: var(--bg-surface-hover) !important;
}

/* Expander */
.stExpander {
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    background: var(--bg-surface) !important;
}

/* SOTA sections */
.sota-section {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}
.sota-section h3 {
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--accent-blue);
    display: inline-block;
}
.sota-section p, .sota-section li {
    font-size: 0.88rem;
    line-height: 1.7;
    color: var(--text-secondary) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border-medium); border-radius: 3px; }

hr { border-color: var(--border-subtle) !important; }

/* Sliders */
.stSlider label { color: var(--text-primary) !important; }
.stSlider [role="slider"] { background: var(--accent-blue) !important; }

/* Metrics */
[data-testid="stMetricLabel"] { color: var(--text-secondary) !important; }
[data-testid="stMetricValue"] { color: var(--text-primary) !important; }
[data-testid="stMetricDelta"] { color: var(--accent-green) !important; }

/* Radio, Checkbox, Selectbox */
.stRadio label, .stCheckbox label, .stSelectbox label {
    color: var(--text-primary) !important;
}

/* Footer */
footer p { color: var(--text-muted) !important; }
</style>
""", unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ══════════════════════════════════════════════════════════════════════════════

_PL = dict(
    paper_bgcolor="#11131a",
    plot_bgcolor="#0d0f14",
    font=dict(family="DM Sans, sans-serif", color="#e2e8f0", size=12),
    hoverlabel=dict(bgcolor="#1e293b", bordercolor="#334155", font_color="#f1f5f9"),
)
_GRID = dict(gridcolor="#1e293b", zerolinecolor="#334155", linecolor="#334155")
C_DEF   = "#3b82f6"
C_ATT   = "#ef4444"
C_STACK = "#10b981"
C_NASH  = "#8b5cf6"
C_PARETO = "#f59e0b"

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def _init():
    defaults = {
        "network_nodes": {
            'Serveur Principal': {'value': 100, 'critical': True,  'type': 'server'},
            'Base de Données':   {'value': 90,  'critical': True,  'type': 'database'},
            'DNS':               {'value': 80,  'critical': True,  'type': 'dns'},
            'Firewall':          {'value': 95,  'critical': True,  'type': 'firewall'},
            'Routeur':           {'value': 70,  'critical': False, 'type': 'router'},
            'Switch':            {'value': 60,  'critical': False, 'type': 'switch'},
            'Poste Admin':       {'value': 85,  'critical': True,  'type': 'workstation'},
            'Serveur Backup':    {'value': 75,  'critical': False, 'type': 'server'},
        },
        "defender_budget": 2,
        "attacker_budget": 1,
        "nash_result": None,
        "stack_result": None,
        "game_model": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ══════════════════════════════════════════════════════════════════════════════
# GAME BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_game() -> Game:
    nodes = []
    for node_name, info in st.session_state.network_nodes.items():
        base_value   = float(info['value'])
        is_critical  = info.get('critical', False)
        defense_cost = base_value * 0.15
        attack_cost  = base_value * 0.20
        vulnerability = 0.95 if is_critical else 0.70
        criticality   = 1.5  if is_critical else 1.0
        nodes.append(Node(
            name=node_name,
            attack_value=base_value,
            defense_cost=defense_cost,
            vulnerability=vulnerability,
            attack_cost=attack_cost,
            criticality=criticality,
        ))

    node_names = [n.name for n in nodes]
    edges = []
    conn_map = [
        ('Serveur Principal', 'Base de Données', 0.9),
        ('Serveur Principal', 'DNS',             0.8),
        ('Serveur Principal', 'Firewall',        0.7),
        ('Firewall',          'Routeur',          1.0),
        ('Firewall',          'Poste Admin',      0.9),
        ('Routeur',           'Switch',           0.8),
        ('Base de Données',   'Serveur Backup',   0.6),
        ('DNS',               'Routeur',          0.5),
    ]
    for src, tgt, w in conn_map:
        if src in node_names and tgt in node_names:
            edges.append(Edge(src, tgt, w))

    if not edges:
        for i in range(len(node_names) - 1):
            edges.append(Edge(node_names[i], node_names[i + 1], 0.5))

    return Game(
        nodes=nodes,
        edges=edges,
        defense_success_rate=0.85,
        lateral_movement_factor=0.15,
        allow_pair_defense=True,
        defender_budget=st.session_state.defender_budget,
        attacker_budget=st.session_state.attacker_budget,
    )


@st.cache_data(show_spinner=False)
def _solve_nash(game_hash: str, _game: Game):
    solver = NashSolver(_game)
    return solver.solve(), solver

@st.cache_data(show_spinner=False)
def _solve_stack(game_hash: str, m_def: np.ndarray, m_att: np.ndarray):
    return compute_stackelberg(m_def, m_att)


def get_game_hash(game: Game) -> str:
    return f"{len(game.nodes)}_{len(game.edges)}_{game.defender_budget}_{game.attacker_budget}_{game.defense_success_rate}"


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
<div class="sidebar-brand">
    <div style="font-size:2rem">🛡</div>
    <h1>GT-SecurityNetwork</h1>
    <p>Game Theory · Network Security</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("#### Ajouter un nœud")
    c1, c2 = st.columns(2)
    with c1:
        new_name = st.text_input("Nom", placeholder="Ex: Web", label_visibility="collapsed")
    with c2:
        new_val = st.number_input("Valeur", 1, 100, 50, label_visibility="collapsed")
    new_crit = st.checkbox("Nœud critique", key="new_crit")
    if st.button("Ajouter le nœud", use_container_width=True):
        if new_name and new_name not in st.session_state.network_nodes:
            st.session_state.network_nodes[new_name] = {
                "value": new_val, "critical": new_crit, "type": "custom"
            }
            st.rerun()
        else:
            st.error("Nom vide ou déjà existant.")

    st.markdown("---")
    st.markdown("#### Budgets stratégiques")
    st.session_state.attacker_budget = st.slider(
        "Budget Attaquant (nœuds max)", 1, 4,
        st.session_state.attacker_budget,
        help="Nombre de nœuds que l'attaquant peut cibler simultanément"
    )
    st.session_state.defender_budget = st.slider(
        "Budget Défenseur (nœuds max)", 1, 4,
        st.session_state.defender_budget,
        help="Nombre de nœuds que le défenseur peut protéger simultanément"
    )
    if st.session_state.defender_budget > 3 or st.session_state.attacker_budget > 3:
        st.warning("Budgets élevés peuvent ralentir les calculs.")

    st.markdown("---")
    st.markdown("#### Gérer les nœuds")
    if st.session_state.network_nodes:
        node_sel = st.selectbox("Nœud", list(st.session_state.network_nodes.keys()))
        c3, c4 = st.columns(2)
        with c3:
            if st.button("Supprimer", use_container_width=True):
                del st.session_state.network_nodes[node_sel]
                st.rerun()
        with c4:
            new_v = st.number_input(
                "Valeur", 1, 100,
                st.session_state.network_nodes[node_sel]["value"],
                key="edit_v", label_visibility="collapsed"
            )
            if st.button("Modifier", use_container_width=True):
                st.session_state.network_nodes[node_sel]["value"] = new_v
                st.rerun()

    if st.button("Réinitialiser le réseau", use_container_width=True):
        st.session_state.network_nodes = {
            'Serveur Principal': {'value': 100, 'critical': True,  'type': 'server'},
            'Base de Données':   {'value': 90,  'critical': True,  'type': 'database'},
            'DNS':               {'value': 80,  'critical': True,  'type': 'dns'},
            'Firewall':          {'value': 95,  'critical': True,  'type': 'firewall'},
            'Routeur':           {'value': 70,  'critical': False, 'type': 'router'},
            'Switch':            {'value': 60,  'critical': False, 'type': 'switch'},
            'Poste Admin':       {'value': 85,  'critical': True,  'type': 'workstation'},
            'Serveur Backup':    {'value': 75,  'critical': False, 'type': 'server'},
        }
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<p style="color:#8792a2;font-size:.72rem;text-align:center">'
        'Projet Théorie des Jeux ·2025-2026</p>',
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# BUILD GAME & SOLVE
# ══════════════════════════════════════════════════════════════════════════════

game = build_game()
game_hash = get_game_hash(game)

with st.spinner("Calcul des équilibres..."):
    nash_result, nash_solver = _solve_nash(game_hash, game)
    M_def = game.get_payoff_matrix()
    M_att = game.get_attacker_payoff_matrix()
    stack_result = _solve_stack(game_hash, M_def, M_att)
    optimal_payoff, _ = nash_solver.optimal_centralized()
    poa = nash_solver.price_of_anarchy(optimal_payoff)

def_labels = game.get_defense_labels()
att_labels  = game.get_attack_labels()

attack_strategies = game.get_attack_labels()
best_resp_label = (attack_strategies[stack_result.attacker_best_response]
                   if stack_result.attacker_best_response < len(attack_strategies)
                   else f"Stratégie {stack_result.attacker_best_response}")

leadership_gain = stack_result.defender_payoff - nash_result.defender_payoff

# ══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="page-header">
    <div class="page-title">GT-SecurityNetwork</div>
    <div class="page-sub">Optimisation de la Sécurité Réseau par la Théorie des Jeux</div>
</div>
""", unsafe_allow_html=True)

# KPI row
total_value = sum(info['value'] for info in st.session_state.network_nodes.values())
gain_pct = (leadership_gain / abs(nash_result.defender_payoff) * 100) if nash_result.defender_payoff != 0 else 0

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card" style="--accent:#3b82f6">
    <div class="kpi-label">Nœuds réseau</div>
    <div class="kpi-value blue">{len(game.nodes)}</div>
  </div>
  <div class="kpi-card" style="--accent:#8b5cf6">
    <div class="kpi-label">Valeur du jeu (Nash)</div>
    <div class="kpi-value">{nash_result.defender_payoff:.2f}</div>
  </div>
  <div class="kpi-card" style="--accent:#10b981">
    <div class="kpi-label">Gain leadership</div>
    <div class="kpi-value green">+{leadership_gain:.2f}</div>
  </div>
  <div class="kpi-card" style="--accent:#f59e0b">
    <div class="kpi-label">Prix de l'Anarchie</div>
    <div class="kpi-value amber">{poa:.3f}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tabs = st.tabs([
    "Réseau",
    "Matrice",
    "Nash",
    "Stackelberg",
    "Pareto",
    "Comparaison",
    "Simulation",
    "État de l'art",
])

tab_net, tab_mat, tab_nash, tab_stack, tab_pareto, tab_comp, tab_sim, tab_sota = tabs

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — RÉSEAU
# ────────────────────────────────────────────────────────────────────────────
with tab_net:
    st.markdown('<div class="sec-title">Topologie du réseau</div>', unsafe_allow_html=True)

    col_g, col_s = st.columns([3, 2])

    with col_g:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        G = nx.Graph()
        for node, info in st.session_state.network_nodes.items():
            G.add_node(node, value=info['value'], critical=info['critical'])

        node_names_list = list(st.session_state.network_nodes.keys())
        conn_map_vis = [
            ('Serveur Principal', 'Base de Données'),
            ('Serveur Principal', 'DNS'),
            ('Serveur Principal', 'Firewall'),
            ('Firewall',          'Routeur'),
            ('Firewall',          'Poste Admin'),
            ('Routeur',           'Switch'),
            ('Base de Données',   'Serveur Backup'),
            ('DNS',               'Routeur'),
        ]
        for src, tgt in conn_map_vis:
            if src in node_names_list and tgt in node_names_list:
                G.add_edge(src, tgt)

        if G.number_of_edges() == 0:
            for i in range(len(node_names_list) - 1):
                G.add_edge(node_names_list[i], node_names_list[i + 1])

        node_colors = ['#ef4444' if st.session_state.network_nodes[n]['critical']
                       else '#3b82f6' for n in G.nodes()]

        fig_net, ax_net = plt.subplots(figsize=(10, 7), facecolor='white')
        pos = nx.spring_layout(G, k=2.5, seed=42, iterations=100)
        node_sizes = [600 + st.session_state.network_nodes[n]['value'] * 20 for n in G.nodes()]
        nx.draw(G, pos, node_color=node_colors, node_size=node_sizes,
                font_size=8, font_weight='bold', edge_color='#cbd5e1',
                width=2, ax=ax_net, alpha=0.92, with_labels=False)
        labels_vis = {n: f"{n}\n{st.session_state.network_nodes[n]['value']}" for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels_vis, font_size=8, font_weight='bold', ax=ax_net)
        legend_items = [
            mpatches.Patch(color='#ef4444', label='Nœud critique'),
            mpatches.Patch(color='#3b82f6', label='Nœud standard'),
        ]
        ax_net.legend(handles=legend_items, loc='upper left', fontsize=8, framealpha=0.9)
        ax_net.set_title("Topologie du réseau", fontsize=13, fontweight='bold', pad=15)
        ax_net.axis('off')
        plt.tight_layout()
        st.pyplot(fig_net, use_container_width=True)
        plt.close()

    with col_s:
        st.markdown('<div class="sec-title">Nœuds</div>', unsafe_allow_html=True)
        nodes_df = pd.DataFrame([
            {
                'Nœud': node,
                'Valeur': info['value'],
                'Critique': 'Oui' if info['critical'] else 'Non',
                'Impact': f"{game.compute_attack_impact(node):.2f}",
            }
            for node, info in st.session_state.network_nodes.items()
        ]).sort_values('Valeur', ascending=False)
        st.dataframe(nodes_df, use_container_width=True, hide_index=True, height=350)

    st.markdown('<div class="sec-title">Impact d\'attaque par nœud</div>', unsafe_allow_html=True)
    node_names_sorted = sorted(
        game.nodes, key=lambda n: game.compute_attack_impact(n.name), reverse=True
    )
    impacts = [game.compute_attack_impact(n.name) for n in node_names_sorted]
    colors_impact = [C_ATT if i == 0 else C_DEF for i in range(len(node_names_sorted))]

    fig_imp = go.Figure(go.Bar(
        x=impacts,
        y=[n.name for n in node_names_sorted],
        orientation='h',
        marker=dict(color=colors_impact, opacity=0.82),
        text=[f"{v:.2f}" for v in impacts],
        textposition='outside',
        hovertemplate="%{y}<br>Impact : <b>%{x:.2f}</b><extra></extra>",
    ))
    fig_imp.update_layout(
        **_PL, height=300, margin=dict(l=20, r=60, t=30, b=20),
        xaxis=dict(**_GRID, title="Impact total"),
        yaxis=dict(**_GRID),
    )
    st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})

# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — MATRICE
# ────────────────────────────────────────────────────────────────────────────
with tab_mat:
    st.markdown('<div class="sec-title">Matrice de payoff</div>', unsafe_allow_html=True)

    mode = st.radio("Afficher", ["Payoff Défenseur", "Payoff Attaquant"], horizontal=True)
    mat  = M_def if mode == "Payoff Défenseur" else M_att

    def _shorten(lbl: str) -> str:
        return lbl.replace("No defense", "Aucune").replace("Protect ", "Def: ").replace("Attack ", "Att: ")

    r_short = [_shorten(l) for l in def_labels]
    c_short = [_shorten(l) for l in att_labels]
    text2d  = [[f"{mat[i,j]:.2f}" for j in range(mat.shape[1])] for i in range(mat.shape[0])]

    fig_heat = go.Figure(go.Heatmap(
        z=mat, x=c_short, y=r_short,
        text=text2d, texttemplate="<b>%{text}</b>",
        colorscale=[[0, "#fef2f2"], [0.4, "#fde68a"], [0.7, "#bbf7d0"], [1, "#059669"]],
        showscale=True,
        hovertemplate="Défense : %{y}<br>Attaque : %{x}<br>Payoff : <b>%{z:.3f}</b><extra></extra>",
        colorbar=dict(title=dict(text="Utilité", font=dict(size=11)),
                      tickfont=dict(size=10), len=0.85),
    ))
    fig_heat.update_layout(
        **_PL,
        height=max(380, mat.shape[0] * 38 + 120),
        margin=dict(l=30, r=30, t=50, b=50),
        xaxis=dict(**_GRID, tickfont=dict(size=9)),
        yaxis=dict(**_GRID, tickfont=dict(size=9), autorange="reversed"),
        title=dict(text=f"Matrice de payoff — {mode}",
                   font=dict(size=14, color="#90bde6")),
    )
    st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

    with st.expander("Données brutes (CSV)"):
    # Convertir en array numpy puis en DataFrame simple
        df_simple = pd.DataFrame(mat)
        df_simple.index = [f"S{i}" for i in range(mat.shape[0])]
        df_simple.columns = [f"A{j}" for j in range(mat.shape[1])]
    
        st.dataframe(df_simple, height=300)
    
    # Pour le téléchargement, garder les vrais noms
        df_full = pd.DataFrame(mat, index=def_labels, columns=att_labels)
        st.download_button("Télécharger CSV (noms complets)", df_full.to_csv().encode(), "payoff_matrix.csv", "text/csv")
        st.caption("📊 Matrice affichée avec indices simplifiés (S0,S1... = stratégies défense, A0,A1... = stratégies attaque)")
# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — NASH
# ────────────────────────────────────────────────────────────────────────────
with tab_nash:
    st.markdown('<div class="sec-title">Equilibre de Nash — Stratégies mixtes</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="card blue">
<b>Principe</b> — Dans un équilibre de Nash en stratégies mixtes, aucun joueur ne peut
améliorer son espérance de gain en déviant unilatéralement.
Les deux joueurs choisissent <b>simultanément</b> et aléatoirement selon des distributions de probabilité.
Calculé par <b>Programmation Linéaire (minimax — von Neumann 1928)</b>.
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Payoff garanti (défenseur)", f"{nash_result.defender_payoff:.4f}")
    c2.metric("Payoff garanti (attaquant)",  f"{nash_result.attacker_payoff:.4f}")
    c3.metric("Convergence LP", "Oui" if nash_result.converged else "Non")

    def _bar_fig(labels, values, title, color, highlight):
        mask = np.array(values) > 0.005
        lf   = [l for l, m in zip(labels, mask) if m]
        vf   = np.array([v for v, m in zip(values, mask) if m])
        if len(vf) == 0:
            lf, vf = labels[:3], np.array(values[:3])
        cols = [color] * len(lf)
        if len(vf):
            cols[int(np.argmax(vf))] = highlight

        fig = go.Figure(go.Bar(
            x=lf, y=vf,
            marker=dict(color=cols, opacity=0.82),
            text=[f"{v:.1%}" for v in vf],
            textposition="outside",
            hovertemplate="%{x}<br>Probabilité : <b>%{y:.2%}</b><extra></extra>",
        ))
        fig.update_layout(
            **_PL, height=320, margin=dict(l=40, r=20, t=50, b=80),
            title=dict(text=title, font=dict(size=13, color="#90bde6")),
            xaxis=dict(**_GRID, tickfont=dict(size=9), tickangle=-30),
            yaxis=dict(**_GRID, range=[0, min(1.25, vf.max() + 0.2)],
                       tickformat=".0%", title="Probabilité"),
            bargap=0.35,
        )
        return fig

    cd, ca = st.columns(2)
    with cd:
        st.plotly_chart(
            _bar_fig(def_labels, nash_result.defender_strategy,
                     "Stratégie mixte — Défenseur",
                     "rgba(59,130,246,0.7)", C_DEF),
            use_container_width=True, config={"displayModeBar": False}
        )
    with ca:
        st.plotly_chart(
            _bar_fig(att_labels, nash_result.attacker_strategy,
                     "Stratégie mixte — Attaquant",
                     "rgba(239,68,68,0.7)", C_ATT),
            use_container_width=True, config={"displayModeBar": False}
        )

    top_def = def_labels[int(np.argmax(nash_result.defender_strategy))]
    top_att = att_labels[int(np.argmax(nash_result.attacker_strategy))]
    st.markdown(f"""
<div class="card purple">
<b>Lecture du résultat</b><br>
Le défenseur joue prioritairement <span class="badge b-blue">{top_def}</span>
à <b>{nash_result.defender_strategy[int(np.argmax(nash_result.defender_strategy))]:.1%}</b>.<br>
L'attaquant cible prioritairement <span class="badge b-red">{top_att.replace("Attack ","")}</span>
à <b>{nash_result.attacker_strategy[int(np.argmax(nash_result.attacker_strategy))]:.1%}</b>.<br>
<b>Propriété d'indifférence :</b> toute action dans le support donne exactement
le même espérance de payoff V = {nash_result.defender_payoff:.3f}.
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 4 — STACKELBERG
# ────────────────────────────────────────────────────────────────────────────
with tab_stack:
    st.markdown('<div class="sec-title">Jeu de Stackelberg — Défenseur Leader</div>', unsafe_allow_html=True)

    st.markdown(f"""
<div class="card green">
<b>Principe</b> — Le défenseur (leader) s'engage publiquement dans une stratégie mixte
avant que l'attaquant (follower) choisisse. L'attaquant observe cet engagement
et choisit la <b>meilleure réponse pure</b>.
Le défenseur anticipe cette réaction et optimise son engagement en conséquence.<br><br>
Best response de l'attaquant : <span class="badge b-red">{best_resp_label.replace('Attack ','')}</span>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Payoff défenseur (Stackelberg)", f"{stack_result.defender_payoff:.4f}",
              delta=f"+{leadership_gain:.4f} vs Nash")
    c2.metric("Payoff attaquant (Stackelberg)", f"{stack_result.attacker_payoff:.4f}")
    c3.metric("Best response attaquant", best_resp_label.replace("Attack ", ""))

    top_stack = def_labels[int(np.argmax(stack_result.defender_strategy))]
    fig_stack = go.Figure(go.Bar(
        x=[l for l, v in zip(def_labels, stack_result.defender_strategy) if v > 0.005],
        y=[v for v in stack_result.defender_strategy if v > 0.005],
        marker=dict(color="rgba(16,185,129,0.75)"),
        text=[f"{v:.1%}" for v in stack_result.defender_strategy if v > 0.005],
        textposition="outside",
        hovertemplate="%{x}<br>Probabilité : <b>%{y:.2%}</b><extra></extra>",
    ))
    fig_stack.update_layout(
        **_PL, height=320, margin=dict(l=40, r=20, t=50, b=80),
        title=dict(text="Engagement optimal du défenseur (Stackelberg)",
                   font=dict(size=13, color="#90bde6")),
        xaxis=dict(**_GRID, tickfont=dict(size=9), tickangle=-30),
        yaxis=dict(**_GRID, tickformat=".0%", title="Probabilité"),
        bargap=0.35,
    )
    st.plotly_chart(fig_stack, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""
<div class="card green">
L'engagement dominant est <span class="badge b-green">{top_stack}</span>
à <b>{stack_result.defender_strategy[int(np.argmax(stack_result.defender_strategy))]:.1%}</b>.
En observant cet engagement, l'attaquant choisit de façon déterministe
<span class="badge b-red">{best_resp_label.replace('Attack ','')}</span>.
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 5 — PARETO
# ────────────────────────────────────────────────────────────────────────────
with tab_pareto:
    st.markdown('<div class="sec-title">Frontière de Pareto & Efficacité Sociale</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="card amber">
<b>Principe</b> — Une allocation (payoff_défenseur, payoff_attaquant) est <b>Pareto-optimale</b>
si aucune autre allocation ne permet d'améliorer le payoff d'un joueur sans détériorer celui de l'autre.
La frontière de Pareto représente l'ensemble de ces allocations efficaces.
Un équilibre de Nash éloigné de cette frontière révèle un <b>coût de l'égoïsme</b>.
</div>
""", unsafe_allow_html=True)

    pareto_result = compute_pareto(
        M_def, M_att, def_labels, att_labels,
        nash_def_payoff=nash_result.defender_payoff,
        nash_att_payoff=nash_result.attacker_payoff,
        stack_def_payoff=stack_result.defender_payoff,
        stack_att_payoff=stack_result.attacker_payoff,
    )

    # Scatter plot : tous les points + frontière de Pareto
    all_x = [p.defender_payoff for p in pareto_result.all_points]
    all_y = [p.attacker_payoff for p in pareto_result.all_points]
    all_text = [f"Def: {p.defense_action}<br>Att: {p.attack_strategy}" for p in pareto_result.all_points]

    front_x = [p.defender_payoff for p in pareto_result.frontier]
    front_y = [p.attacker_payoff for p in pareto_result.frontier]
    front_text = [f"Def: {p.defense_action}<br>Att: {p.attack_strategy}" for p in pareto_result.frontier]

    fig_pareto = go.Figure()

    # Tous les points
    fig_pareto.add_trace(go.Scatter(
        x=all_x, y=all_y,
        mode='markers',
        marker=dict(color='#cbd5e1', size=6, opacity=0.6),
        name='Allocations possibles',
        hovertext=all_text,
        hoverinfo='text',
    ))

    # Frontière de Pareto
    fig_pareto.add_trace(go.Scatter(
        x=front_x, y=front_y,
        mode='markers+lines',
        marker=dict(color=C_PARETO, size=9, symbol='diamond'),
        line=dict(color=C_PARETO, width=2, dash='dot'),
        name='Frontière de Pareto',
        hovertext=front_text,
        hoverinfo='text',
    ))

    # Point Nash
    fig_pareto.add_trace(go.Scatter(
        x=[nash_result.defender_payoff],
        y=[nash_result.attacker_payoff],
        mode='markers+text',
        marker=dict(color=C_NASH, size=14, symbol='star'),
        text=["Nash"],
        textposition="top right",
        textfont=dict(size=11, color=C_NASH),
        name='Nash Equilibrium',
    ))

    # Point Stackelberg
    fig_pareto.add_trace(go.Scatter(
        x=[stack_result.defender_payoff],
        y=[stack_result.attacker_payoff],
        mode='markers+text',
        marker=dict(color=C_STACK, size=14, symbol='star'),
        text=["Stackelberg"],
        textposition="top right",
        textfont=dict(size=11, color=C_STACK),
        name='Stackelberg',
    ))

    # Point optimal centralisé
    fig_pareto.add_trace(go.Scatter(
        x=[optimal_payoff],
        y=[0],
        mode='markers+text',
        marker=dict(color='#90bde6', size=12, symbol='x'),
        text=["Optimal centralisé"],
        textposition="top right",
        textfont=dict(size=10, color='#90bde6'),
        name='Optimal centralisé',
    ))

    fig_pareto.update_layout(
        **_PL,
        height=500,
        margin=dict(l=60, r=30, t=60, b=60),
        title=dict(text="Frontière de Pareto — Espace des payoffs",
                   font=dict(size=14, color="#e2e8f0")),
        xaxis=dict(**_GRID, title="Payoff défenseur"),
        yaxis=dict(**_GRID, title="Payoff attaquant"),
    )
    st.plotly_chart(fig_pareto, use_container_width=True, config={"displayModeBar": False})

    # Métriques Pareto
    eff_score = pareto_efficiency_score(pareto_result.frontier, pareto_result.nash_point)
    cp1, cp2, cp3 = st.columns(3)
    cp1.metric("Points Pareto-optimaux", len(pareto_result.frontier))
    cp2.metric("Distance Nash → frontière", f"{eff_score:.4f}")
    cp3.metric("Allocations totales", len(pareto_result.all_points))

    st.markdown(f"""
<div class="card amber">
<b>Interprétation</b><br>
La frontière de Pareto contient <b>{len(pareto_result.frontier)} allocations Pareto-optimales</b>
sur {len(pareto_result.all_points)} possibles.<br>
La distance du Nash à la frontière est <b>{eff_score:.4f}</b> — 
{"Nash est quasi-Pareto-optimal (efficacité sociale élevée)." if eff_score < 1.0 else "Nash est éloigné de la frontière (coût de l'égoïsme significatif)."}
</div>
""", unsafe_allow_html=True)

    with st.expander("Points de la frontière de Pareto"):
        df_front = pd.DataFrame([
            {
                "Action de défense": p.defense_action,
                "Stratégie d'attaque": p.attack_strategy,
                "Payoff défenseur": f"{p.defender_payoff:.4f}",
                "Payoff attaquant": f"{p.attacker_payoff:.4f}",
            }
            for p in pareto_result.frontier
        ])
        st.dataframe(df_front, use_container_width=True, hide_index=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 6 — COMPARAISON
# ────────────────────────────────────────────────────────────────────────────
with tab_comp:
    st.markdown('<div class="sec-title">Comparaison Nash / Stackelberg / Optimal centralisé</div>', unsafe_allow_html=True)

    c_n, c_s, c_o = st.columns(3)
    with c_n:
        st.markdown(f"""
<div class="card purple" style="text-align:center">
<div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;color:#8b5cf6;font-weight:700">Nash</div>
<div style="font-size:2rem;font-weight:700;color:#90bde6;font-family:'DM Mono',monospace;margin:0.5rem 0">
{nash_result.defender_payoff:.3f}</div>
<div style="font-size:0.78rem;color:#8792a2">Jeu simultané — garanti</div>
</div>
""", unsafe_allow_html=True)
    with c_s:
        st.markdown(f"""
<div class="card green" style="text-align:center">
<div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;color:#10b981;font-weight:700">Stackelberg</div>
<div style="font-size:2rem;font-weight:700;color:#90bde6;font-family:'DM Mono',monospace;margin:0.5rem 0">
{stack_result.defender_payoff:.3f}</div>
<div style="font-size:0.78rem;color:#8792a2">Leader séquentiel</div>
</div>
""", unsafe_allow_html=True)
    with c_o:
        st.markdown(f"""
<div class="card blue" style="text-align:center">
<div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;color:#3b82f6;font-weight:700">Optimal centralisé</div>
<div style="font-size:2rem;font-weight:700;color:#90bde6;font-family:'DM Mono',monospace;margin:0.5rem 0">
{optimal_payoff:.3f}</div>
<div style="font-size:0.78rem;color:#8792a2">Minimax pur — planner central</div>
</div>
""", unsafe_allow_html=True)

    fig_comp = go.Figure()
    for name, val, color in [
        ("Nash<br>(simultané)",        nash_result.defender_payoff, C_NASH),
        ("Stackelberg<br>(leader)",    stack_result.defender_payoff, C_STACK),
        ("Optimal<br>centralisé",      optimal_payoff, C_DEF),
    ]:
        fig_comp.add_trace(go.Bar(
            name=name.replace("<br>", " "),
            x=[name], y=[val],
            marker=dict(color=color, opacity=0.82),
            width=0.35,
            text=[f"<b>{val:.3f}</b>"],
            textposition="outside",
            textfont=dict(size=13, color="#90bde6"),
        ))

    fig_comp.update_layout(
    **_PL, height=420,
    margin=dict(l=50, r=30, t=60, b=50),
    title=dict(text="Payoff défenseur — Comparaison des solutions",
               font=dict(size=14, color="#90bde6")),
    yaxis=dict(**_GRID, title="Payoff défenseur"),
    barmode="group", bargap=0.28,
    legend=dict(bgcolor="#1a1d26", bordercolor="#334155", font=dict(color="#e2e8f0")),
    )
    st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})

    gain_pct_str = f"+{gain_pct:.1f}%" if gain_pct >= 0 else f"{gain_pct:.1f}%"
    poa_loss = (poa - 1) * 100 if poa > 1 else 0

    st.markdown(f"""
<div class="card blue">
<b>Synthèse comparative</b><br><br>
<b>Nash (simultané) :</b> le défenseur randomise ses actions pour garantir {nash_result.defender_payoff:.3f},
rendant l'attaquant indifférent entre ses stratégies.<br><br>
<b>Stackelberg (leader) :</b> en s'engageant publiquement, le défenseur obtient {stack_result.defender_payoff:.3f}
— soit {gain_pct_str} de mieux qu'en Nash. C'est l'avantage du <i>first-mover</i>.<br><br>
<b>Optimal centralisé :</b> sans contrainte stratégique, un planificateur central atteindrait {optimal_payoff:.3f}.
Le prix de l'anarchie est {poa:.3f} — le comportement égoïste coûte {poa_loss:.1f}% d'efficacité.
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 7 — SIMULATION DE CONVERGENCE
# ────────────────────────────────────────────────────────────────────────────
with tab_sim:
    st.markdown('<div class="sec-title">Simulation de convergence vers l\'équilibre</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="card blue">
<b>Principe (3.2)</b> — On simule des tours de jeu répétés où chaque joueur tire
son action selon sa stratégie mixte Nash. Le payoff moyen cumulé doit converger
vers la valeur théorique de l'équilibre. Cette simulation vérifie empiriquement
le théorème minimax.
</div>
""", unsafe_allow_html=True)

    cs1, cs2 = st.columns([1, 2])
    with cs1:
        n_turns = st.number_input("Nombre de tours", 10, 2000, 300, step=50)
        run_sim = st.button("Lancer la simulation", use_container_width=True)

    if run_sim:
        simulator = GameSimulator(game, nash_result)
        with st.spinner(f"Simulation de {n_turns} tours..."):
            simulator.play_multiple_turns(n_turns)
            conv_data = simulator.get_convergence_data()

        st.success(f"Simulation terminée — {n_turns} tours joués.")

        tours = conv_data['tours']
        payoff_cumul = conv_data['payoff_cumul']
        nash_value   = conv_data['nash_value']

        # ── Courbe de convergence du payoff
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=tours, y=payoff_cumul,
            mode='lines',
            name='Payoff moyen cumulé',
            line=dict(color=C_DEF, width=2),
        ))
        fig_conv.add_hline(
            y=nash_value,
            line_dash="dash",
            line_color=C_NASH,
            annotation_text=f"Valeur Nash théorique ({nash_value:.3f})",
            annotation_position="bottom right",
            annotation_font=dict(size=11, color=C_NASH),
        )
        fig_conv.update_layout(
            **_PL, height=380,
            margin=dict(l=60, r=30, t=50, b=50),
            title=dict(text="Convergence du payoff moyen vers la valeur Nash",
                       font=dict(size=14, color="#90bde6")),
            xaxis=dict(**_GRID, title="Nombre de tours"),
            yaxis=dict(**_GRID, title="Payoff moyen cumulé"),
            legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#e4e7ef"),
        )
        st.plotly_chart(fig_conv, use_container_width=True, config={"displayModeBar": False})

        # ── Convergence des fréquences défenseur
        def_freq_data = conv_data['def_freq_over_time']
        fig_def_conv = go.Figure()
        for idx, label in enumerate(nash_result.defender_labels):
            theo = nash_result.defender_strategy[idx]
            if theo > 0.01:
                fig_def_conv.add_trace(go.Scatter(
                    x=tours, y=def_freq_data[:, idx].tolist(),
                    mode='lines', name=label[:25],
                    line=dict(width=1.5, dash='solid'),
                ))
                fig_def_conv.add_hline(
                    y=theo, line_dash="dot", line_width=1, opacity=0.5,
                    annotation_text=f"{theo:.2f}", annotation_font_size=9,
                )

        fig_def_conv.update_layout(
            **_PL, height=340,
            margin=dict(l=60, r=30, t=50, b=50),
            title=dict(text="Convergence des fréquences défenseur vers les probabilités Nash",
                       font=dict(size=13, color="#90bde6")),
            xaxis=dict(**_GRID, title="Tours"),
            yaxis=dict(**_GRID, title="Fréquence cumulée", range=[-0.05, 1.05]),
        )
        st.plotly_chart(fig_def_conv, use_container_width=True, config={"displayModeBar": False})

        # ── Vérification de convergence
        conv_check = simulator.convergence_check()
        if isinstance(conv_check, dict):
            st.markdown(f"""
<div class="card {'green' if conv_check['converged'] else 'amber'}">
<b>Vérification de convergence</b><br>
Erreur fréquences défenseur : <b>{conv_check['defender_error']:.4f}</b><br>
Erreur fréquences attaquant : <b>{conv_check['attacker_error']:.4f}</b><br>
{"Les fréquences convergent vers l'équilibre de Nash (erreur < 0.1)." if conv_check['converged']
 else "Convergence partielle — augmenter le nombre de tours pour améliorer la précision."}
</div>
""", unsafe_allow_html=True)

        with st.expander("Historique des tours"):
            st.dataframe(pd.DataFrame(simulator.history), use_container_width=True, height=250)

    else:
        st.info("Configurez le nombre de tours puis cliquez sur 'Lancer la simulation'.")
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center;padding:1.5rem 0;color:#8792a2;font-size:0.78rem'>
GT-SecurityNetwork · Théorie des Jeux & Optimisation Réseau ·2025-2026
</div>
""", unsafe_allow_html=True)
""