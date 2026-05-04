"""
app.py
======
Interface Streamlit premium — GT-SecurityNetwork
Théorie des Jeux · Sécurité Réseau

Intègre :
  game_model        → Game, Node, Edge
  nash_solver       → compute_nash
  stackelberg_solver→ compute_stackelberg
"""

from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from game_model import Game, Node, Edge
from nash_solver import compute_nash, NashResult
from stackelberg_solver import compute_stackelberg, StackelbergResult

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="GT-SecurityNetwork · Game Theory",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS — DARK CYBERSECURITY THEME
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ── Global reset ── */
html, body, .stApp {
    background: linear-gradient(145deg, #060b14 0%, #091525 60%, #060d1f 100%) !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stAppViewContainer"] { background: transparent !important; }
[data-testid="stHeader"]           { background: transparent !important; }
section[data-testid="stSidebar"] > div { background: rgba(6,11,20,0.97) !important; }
#MainMenu, footer, header          { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.03); }
::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.3); border-radius: 3px; }

/* ── Sidebar border ── */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(0,212,255,0.12) !important;
}

/* ── Hero header ── */
.hero-wrap {
    text-align: center;
    padding: 2rem 0 0.5rem;
}
.hero-title {
    font-size: clamp(2rem, 4vw, 3rem);
    font-weight: 800;
    background: linear-gradient(135deg, #00d4ff 0%, #00ff88 45%, #7c3aed 100%);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.025em;
    line-height: 1.15;
    animation: hero-shift 6s ease-in-out infinite;
}
@keyframes hero-shift {
    0%,100% { background-position: 0% 50%; }
    50%      { background-position: 100% 50%; }
}
.hero-sub {
    font-size: 1rem;
    color: #475569;
    margin-top: 0.4rem;
    letter-spacing: 0.06em;
    font-weight: 400;
}
.hero-scenario {
    display: inline-block;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 30px;
    padding: 0.25rem 1rem;
    color: #00d4ff;
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 0.6rem;
    letter-spacing: 0.04em;
}

/* ── Metric grid ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0 0.5rem;
}
.m-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 1.25rem 1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
}
.m-card:hover {
    transform: translateY(-3px);
    border-color: rgba(0,212,255,0.35);
    box-shadow: 0 10px 36px rgba(0,212,255,0.12);
}
.m-card::after {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--accent, linear-gradient(90deg,#00d4ff,#00ff88));
    opacity: 0; transition: opacity .25s;
}
.m-card:hover::after { opacity: 1; }
.m-icon  { font-size: 1.6rem; margin-bottom: .35rem; }
.m-label { font-size: .7rem; color: #475569; text-transform: uppercase; letter-spacing: .1em; font-weight: 600; }
.m-value {
    font-size: 2.1rem; font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    color: #00d4ff;
    text-shadow: 0 0 22px rgba(0,212,255,.45);
    line-height: 1.1;
}

/* ── Section title ── */
.sec-title {
    font-size: 1.25rem; font-weight: 700; color: #f1f5f9;
    margin: 1.8rem 0 .8rem;
    display: flex; align-items: center; gap: .5rem;
}
.sec-title::after {
    content: '';
    flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(0,212,255,.35), transparent);
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.028);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.15rem 1.25rem;
    margin: .75rem 0;
}
.card.blue   { border-color: rgba(0,212,255,.25); }
.card.green  { border-color: rgba(0,255,136,.25); }
.card.purple { border-color: rgba(124,58,237,.25); }
.card.amber  { border-color: rgba(245,158,11,.25); }
.card.red    { border-color: rgba(239,68,68,.25);  }

/* ── Badges ── */
.badge {
    display: inline-block; border-radius: 20px;
    padding: .2rem .7rem; font-size: .73rem; font-weight: 600;
    letter-spacing: .04em; margin: .2rem .1rem;
}
.b-blue   { background: rgba(0,212,255,.15); color:#00d4ff; border:1px solid rgba(0,212,255,.3); }
.b-green  { background: rgba(0,255,136,.15); color:#00ff88; border:1px solid rgba(0,255,136,.3); }
.b-purple { background: rgba(124,58,237,.15); color:#a78bfa; border:1px solid rgba(124,58,237,.3); }
.b-amber  { background: rgba(245,158,11,.15); color:#fbbf24; border:1px solid rgba(245,158,11,.3); }
.b-red    { background: rgba(239,68,68,.15); color:#f87171; border:1px solid rgba(239,68,68,.3); }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 14px !important; padding: 5px !important; gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #475569 !important;
    border-radius: 10px !important;
    font-weight: 600 !important; font-size: .88rem !important;
    padding: .5rem 1rem !important;
    transition: all .2s !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,212,255,0.13) !important;
    color: #00d4ff !important;
    box-shadow: 0 0 14px rgba(0,212,255,0.18) !important;
}
.stTabs [data-baseweb="tab-border"]   { display: none !important; }
.stTabs [data-baseweb="tab-highlight"]{ display: none !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff 0%, #0099bb 100%) !important;
    color: #060b14 !important; font-weight: 700 !important;
    border: none !important; border-radius: 10px !important;
    padding: .55rem 1.8rem !important;
    letter-spacing: .02em !important;
    transition: all .3s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(0,212,255,.38) !important;
}

/* ── Inputs ── */
.stTextInput input, .stNumberInput input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,.1) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
.stSelectbox > div > div {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(255,255,255,.1) !important;
    color: #e2e8f0 !important;
}
/* ── Slider ── */
.stSlider .stSlider > div { color: #00d4ff; }

/* ── Sidebar logo ── */
.sb-logo {
    text-align: center; padding: 1.2rem 0 1rem;
    border-bottom: 1px solid rgba(0,212,255,.12);
    margin-bottom: 1.2rem;
}
.sb-logo h2 {
    font-size: 1.2rem; font-weight: 800; margin: .3rem 0 0;
    background: linear-gradient(135deg,#00d4ff,#00ff88);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sb-logo p { color: #475569; font-size: .75rem; margin: .1rem 0 0; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 12px !important; overflow: hidden; }

/* ── Expander ── */
.stExpander { border: 1px solid rgba(255,255,255,0.07) !important; border-radius: 12px !important; }
.stExpander summary { color: #94a3b8 !important; }

/* ── Info / warning / success ── */
.stAlert { border-radius: 10px !important; }

/* ── Divider ── */
hr { border-color: rgba(0,212,255,.15) !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY THEME CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

_PL = dict(                                 # base plotly layout
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.025)",
    font=dict(family="Inter, sans-serif", color="#94a3b8", size=12),
    hoverlabel=dict(
        bgcolor="rgba(6,11,20,0.97)",
        bordercolor="rgba(0,212,255,0.4)",
        font_color="#e2e8f0",
    ),
)
_GRID = dict(
    gridcolor="rgba(255,255,255,0.06)",
    zerolinecolor="rgba(255,255,255,0.1)",
    linecolor="rgba(255,255,255,0.06)",
)

C_DEF   = "#00d4ff"   # défenseur — bleu électrique
C_ATT   = "#ef4444"   # attaquant — rouge
C_STACK = "#00ff88"   # Stackelberg — vert néon
C_NASH  = "#a78bfa"   # Nash — violet
C_BOTH  = "#f59e0b"   # protégé + attaqué — ambre
C_MUTED = "#475569"


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════

def _init():
    defaults = {
        "scenario_key":   "critical-db",
        "defense_rate":   0.80,
        "lateral_factor": 0.20,
        "custom_nodes": [
            {"name": "WebApp",   "value": 8.0,  "dcost": 2.0,  "vuln": 0.85, "acost": 1.0, "crit": 1.0},
            {"name": "Database", "value": 14.0, "dcost": 3.0,  "vuln": 0.90, "acost": 2.0, "crit": 1.3},
            {"name": "Mail",     "value": 5.0,  "dcost": 1.5,  "vuln": 0.70, "acost": 0.8, "crit": 0.9},
        ],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _network_fig(game: Game, protected: list[str] | None = None,
                 attacked: str | None = None) -> go.Figure:
    G = nx.Graph()
    for n in game.nodes:
        G.add_node(n.name)
    for e in game.edges:
        G.add_edge(e.source, e.target, weight=e.weight)
    if G.number_of_edges() == 0:
        for i in range(len(game.nodes) - 1):
            G.add_edge(game.nodes[i].name, game.nodes[i + 1].name, weight=0.4)

    pos = nx.spring_layout(G, seed=42, k=2.5, iterations=120)
    p_set = set(protected or [])
    traces: list[go.BaseTraceType] = []

    # Edge traces
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        w = data.get("weight", 0.4)
        traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None], mode="lines",
            line=dict(width=w * 4, color=f"rgba(100,116,139,{0.15 + w * 0.4})"),
            hoverinfo="none", showlegend=False,
        ))

    # Node trace
    nxl, nyl, nc, ns, nt, nh = [], [], [], [], [], []
    for nm in G.nodes:
        x, y = pos[nm]
        nxl.append(x); nyl.append(y)
        node = game.get_node(nm)
        ip, ia = nm in p_set, nm == attacked
        color = (C_BOTH if ip and ia else
                 C_ATT  if ia else
                 C_STACK if ip else C_DEF)
        nc.append(color)
        ns.append(22 + node.attack_value * 3.8)
        nt.append(nm)
        nh.append(
            f"<b>{nm}</b><br>"
            f"💰 Valeur : <b>{node.attack_value:.1f}</b><br>"
            f"🛡️ Coût déf. : {node.defense_cost:.1f}<br>"
            f"⚠️ Vulnérabilité : {node.vulnerability:.0%}<br>"
            f"📌 Criticité : {node.criticality:.2f}<br>"
            f"💥 Impact total : {game.compute_attack_impact(nm):.2f}"
        )

    traces.append(go.Scatter(
        x=nxl, y=nyl, mode="markers+text",
        marker=dict(
            color=nc, size=ns,
            line=dict(width=2, color="rgba(255,255,255,0.18)"),
            opacity=0.92,
        ),
        text=nt, textposition="top center",
        textfont=dict(size=12, color="#e2e8f0", family="Inter"),
        hovertext=nh, hoverinfo="text", showlegend=False,
    ))

    # Legend annotations
    legend_items = [
        ("🔵", C_DEF,   "Neutre"),
        ("🟢", C_STACK, "Protégé"),
        ("🔴", C_ATT,   "Attaqué"),
        ("🟡", C_BOTH,  "Protégé + Attaqué"),
    ]
    annotations = []
    for i, (icon, color, label) in enumerate(legend_items):
        annotations.append(dict(
            x=0.01, y=1 - i * 0.065, xref="paper", yref="paper",
            text=f'<span style="color:{color}">●</span>  {label}',
            showarrow=False, font=dict(size=11, color="#94a3b8"),
            xanchor="left",
        ))

    _layout = dict(_PL)
    fig = go.Figure(data=traces)
    fig.update_layout(
        **_PL,
        margin=dict(l=50, r=30, t=55, b=50),
        annotations=annotations,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=430,
        title=dict(text="🕸️ Topologie du réseau", font=dict(size=15, color="#e2e8f0")),
        hovermode="closest",
    )
    return fig


def _heatmap_fig(matrix: np.ndarray, row_labels: list[str],
                 col_labels: list[str], title: str) -> go.Figure:

    def _shorten(lbl: str) -> str:
        return lbl.replace("No defense", "⚫ Aucune").replace("Protect ", "🛡️ ").replace("Attack ", "⚔️ ")

    r_short = [_shorten(l) for l in row_labels]
    c_short = [_shorten(l) for l in col_labels]
    text2d = [[f"{matrix[i, j]:.2f}" for j in range(matrix.shape[1])]
              for i in range(matrix.shape[0])]

    fig = go.Figure(go.Heatmap(
        z=matrix, x=c_short, y=r_short,
        text=text2d, texttemplate="<b>%{text}</b>",
        colorscale=[
            [0.0, "#7f1d1d"], [0.25, "#dc2626"],
            [0.45, "#f59e0b"], [0.65, "#22c55e"],
            [1.0, "#00ff88"],
        ],
        hovertemplate=(
            "Défense : %{y}<br>Attaque : %{x}<br>"
            "Payoff : <b>%{z:.3f}</b><extra></extra>"
        ),
        showscale=True,
        colorbar=dict(
            title=dict(text="Utilité défenseur", font=dict(color="#94a3b8", size=11)),
            tickfont=dict(color="#94a3b8"),
            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.08)",
            len=0.85,
        ),
    ))
    _layout = dict(_PL)
    fig.update_layout(
        **_PL,
        margin=dict(l=50, r=30, t=55, b=50),
        title=dict(text=title, font=dict(size=15, color="#e2e8f0")),
        height=max(380, matrix.shape[0] * 40 + 120),
        xaxis=dict(**_GRID, tickfont=dict(size=10)),
        yaxis=dict(**_GRID, tickfont=dict(size=10), autorange="reversed"),
    )
    return fig


def _strategy_bars(labels: list[str], values: np.ndarray, title: str,
                   color: str, highlight_color: str | None = None,
                   threshold: float = 0.005) -> go.Figure:

    def _s(lbl: str) -> str:
        return (lbl.replace("No defense", "⚫ Aucune")
                   .replace("Protect ", "🛡️ ")
                   .replace("Attack ", "⚔️ "))

    idxs = [i for i, v in enumerate(values) if v > threshold] or list(range(len(values)))
    lf   = [_s(labels[i]) for i in idxs]
    vf   = np.array([values[i] for i in idxs])
    colors = [color] * len(lf)
    if highlight_color is not None and len(vf):
        colors[int(np.argmax(vf))] = highlight_color

    fig = go.Figure(go.Bar(
        x=lf, y=vf,
        marker=dict(color=colors, opacity=0.82, line=dict(width=0)),
        text=[f"{v:.1%}" for v in vf],
        textposition="outside",
        textfont=dict(size=11, color="#e2e8f0"),
        hovertemplate="%{x}<br>Probabilité : <b>%{y:.2%}</b><extra></extra>",
    ))
    _layout = dict(_PL)
    fig.update_layout(
        **_PL,
        margin=dict(l=50, r=30, t=55, b=50),
        title=dict(text=title, font=dict(size=14, color="#e2e8f0")),
        height=340,
        xaxis=dict(**_GRID, tickfont=dict(size=10)),
        yaxis=dict(**_GRID, range=[0, min(1.18, vf.max() + 0.18)],
                   tickformat=".0%", title="Probabilité"),
        bargap=0.35,
    )
    return fig


def _comparison_fig(nash_val: float, stack_val: float) -> go.Figure:
    gain = stack_val - nash_val
    fig  = go.Figure()
    for name, val, color, pattern in [
        ("Nash<br><sub style='font-size:10px'>(simultané)</sub>",       nash_val,  C_NASH,  "/"),
        ("Stackelberg<br><sub style='font-size:10px'>(leader)</sub>",   stack_val, C_STACK, ""),
    ]:
        fig.add_trace(go.Bar(
            name=name.split("<br>")[0], x=[name], y=[val],
            marker=dict(color=color, opacity=0.82),
            width=0.38,
            text=[f"<b>{val:.3f}</b>"], textposition="outside",
            textfont=dict(size=14, color="#e2e8f0", family="JetBrains Mono"),
        ))

    if abs(gain) > 0.05:
        y_annot = max(nash_val, stack_val) + abs(max(nash_val, stack_val)) * 0.22 + 1.5
        fig.add_annotation(
            x=1, y=y_annot,
            text=f"<b>🏆 Gain leadership<br>+{gain:.3f}</b>",
            showarrow=True, arrowhead=2, arrowcolor=C_STACK,
            font=dict(color=C_STACK, size=11),
            bgcolor="rgba(0,255,136,0.08)", bordercolor=C_STACK,
            borderwidth=1, borderpad=8, xref="x", yref="y",
        )

    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.18)")
    _layout = dict(_PL)
    fig.update_layout(
        **_PL,
        margin=dict(l=50, r=30, t=55, b=50),
        title=dict(text="⚖️ Nash vs Stackelberg — Payoff du défenseur",
                   font=dict(size=15, color="#e2e8f0")),
        height=440,
        yaxis=dict(**_GRID, title="Payoff défenseur"),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.08)",
                    font=dict(color="#94a3b8")),
        barmode="group", bargap=0.28,
    )
    return fig


def _impact_fig(game: Game) -> go.Figure:
    names   = [n.name for n in game.nodes]
    impacts = [game.compute_attack_impact(n) for n in names]
    pairs   = sorted(zip(impacts, names), reverse=True)
    simp, snam = zip(*pairs)

    bar_col = [C_ATT if i == 0 else f"rgba(0,212,255,{0.35 + 0.1 * (len(snam) - i) / len(snam)})"
               for i in range(len(snam))]

    fig = go.Figure(go.Bar(
        x=list(reversed(list(simp))),
        y=list(reversed(list(snam))),
        orientation="h",
        marker=dict(color=list(reversed(bar_col)), opacity=0.85),
        text=[f"<b>{v:.2f}</b>" for v in reversed(list(simp))],
        textposition="outside",
        textfont=dict(size=11, color="#e2e8f0"),
        hovertemplate="%{y}<br>Impact : <b>%{x:.2f}</b><extra></extra>",
    ))
    fig.update_layout(
        **_PL,
        margin=dict(l=30, r=60, t=55, b=40),
        title=dict(text="💥 Impact d'attaque par nœud (direct + latéral)",
                   font=dict(size=14, color="#e2e8f0")),
        height=280 + len(names) * 22,
        xaxis=dict(**_GRID, title="Impact total"),
        yaxis=dict(**_GRID),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# CACHED SOLVERS  (cache_data hashes numpy array content)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _nash(m_def: np.ndarray) -> NashResult:
    return compute_nash(m_def)


@st.cache_data(show_spinner=False)
def _stackelberg(m_def: np.ndarray, m_att: np.ndarray) -> StackelbergResult:
    return compute_stackelberg(m_def, m_att)


# ══════════════════════════════════════════════════════════════════════════════
# GAME BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _predefined_game(key: str, dr: float, lf: float) -> Game:
    scen = next(s for s in Game.get_scenarios() if s.key == key)
    return Game(
        nodes=scen.game.nodes,
        edges=scen.game.edges,
        defense_success_rate=dr,
        lateral_movement_factor=lf,
    )


def _custom_game(nodes_json: str, dr: float, lf: float) -> Game:
    nd = json.loads(nodes_json)
    nodes = [
        Node(
            name=n["name"],
            attack_value=float(n["value"]),
            defense_cost=float(n["dcost"]),
            vulnerability=float(n["vuln"]),
            attack_cost=float(n.get("acost", 0.0)),
            criticality=float(n.get("crit", 1.0)),
        )
        for n in nd
    ]
    return Game(nodes=nodes, defense_success_rate=dr, lateral_movement_factor=lf)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div style="font-size:2.2rem">🛡️</div>
        <h2>GT-SecurityNet</h2>
        <p>Game Theory · Network Security</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Scenario selector ────────────────────────────────────────────────────
    st.markdown("#### 🎮 Scénario")
    SCENARIO_OPTS = {
        "critical-db":       "🗄️  Base de données critique",
        "balanced-network":  "⚖️  Réseau équilibré",
        "limited-attacker":  "🔒 Attaquant limité",
        "custom":            "🔧 Jeu personnalisé",
    }
    selected = st.selectbox(
        "Choisir un scénario",
        options=list(SCENARIO_OPTS.keys()),
        format_func=lambda k: SCENARIO_OPTS[k],
        index=list(SCENARIO_OPTS.keys()).index(st.session_state.scenario_key),
    )
    st.session_state.scenario_key = selected

    if selected != "custom":
        scen_meta = {s.key: s for s in Game.get_scenarios()}
        if selected in scen_meta:
            st.info(f"ℹ️ {scen_meta[selected].description}", icon=None)

    st.markdown("---")

    # ── Game parameters ──────────────────────────────────────────────────────
    with st.expander("⚙️ Paramètres globaux", expanded=True):
        dr = st.slider(
            "🛡️ Taux de succès défense",
            0.50, 1.00, st.session_state.defense_rate, 0.01,
            help="Fraction de l'impact neutralisée quand le nœud est protégé",
        )
        lf = st.slider(
            "🌊 Propagation latérale",
            0.00, 0.50, st.session_state.lateral_factor, 0.01,
            help="Fraction de l'impact transmise aux nœuds voisins via les arêtes",
        )
        st.session_state.defense_rate   = dr
        st.session_state.lateral_factor = lf

    # ── Custom node editor ───────────────────────────────────────────────────
    if selected == "custom":
        st.markdown("---")
        st.markdown("#### 🔧 Éditeur de nœuds")

        updated_nodes: list[dict] = []
        for i, nd in enumerate(st.session_state.custom_nodes):
            with st.expander(f"**Nœud {i + 1} : {nd['name']}**", expanded=(i == 0)):
                c1, c2 = st.columns(2)
                name  = c1.text_input("Nom", nd["name"],  key=f"cn_{i}")
                value = c2.number_input("Valeur attaque", 1.0, 50.0, float(nd["value"]), 0.5, key=f"cv_{i}")
                c3, c4 = st.columns(2)
                dcost = c3.number_input("Coût défense", 0.1, 15.0, float(nd["dcost"]), 0.1, key=f"cd_{i}")
                vuln  = c4.slider("Vulnérabilité", 0.1, 1.0, float(nd["vuln"]), 0.05, key=f"vu_{i}")
                updated_nodes.append({"name": name, "value": value, "dcost": dcost,
                                      "vuln": vuln, "acost": nd.get("acost", 0.0),
                                      "crit": nd.get("crit", 1.0)})

        ca, cr = st.columns(2)
        if ca.button("➕ Ajouter", width="content") and len(updated_nodes) < 6:
            n = len(updated_nodes) + 1
            updated_nodes.append({"name": f"Node{n}", "value": 6.0, "dcost": 1.5,
                                   "vuln": 0.8, "acost": 0.8, "crit": 1.0})
        if cr.button("➖ Supprimer", width="content") and len(updated_nodes) > 2:
            updated_nodes.pop()
        st.session_state.custom_nodes = updated_nodes

    st.markdown("---")
    st.markdown(
        '<p style="color:#334155;font-size:.72rem;text-align:center">'
        "Projet GT · Sécurité Réseau · 2024</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# BUILD GAME + SOLVE
# ══════════════════════════════════════════════════════════════════════════════

game: Game | None = None
game_error: str | None = None

try:
    if st.session_state.scenario_key == "custom":
        nodes_json = json.dumps(st.session_state.custom_nodes)
        game = _custom_game(nodes_json, st.session_state.defense_rate, st.session_state.lateral_factor)
    else:
        game = _predefined_game(
            st.session_state.scenario_key,
            st.session_state.defense_rate,
            st.session_state.lateral_factor,
        )
except Exception as e:
    game_error = f"Impossible de construire le jeu : {e}"

if game_error or game is None:
    st.error(f"❌ {game_error}")
    st.stop()

try:
    with st.spinner("🧮 Calcul des équilibres en cours…"):
        M_def = game.get_payoff_matrix()
        M_att = game.get_attacker_payoff_matrix()
        nash  = _nash(M_def)
        stack = _stackelberg(M_def, M_att)
except Exception as e:
    st.error(f"❌ Erreur solveur : {e}")
    st.stop()

def_labels = game.get_defense_labels()
att_labels  = game.get_attack_labels()
best_node   = game.nodes[stack.attacker_best_response]


# ══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════

scenario_title = (
    {s.key: s.title for s in Game.get_scenarios()}.get(st.session_state.scenario_key)
    or "Jeu personnalisé"
)

st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-title">🛡️ GT-SecurityNetwork</div>
    <div class="hero-sub">Théorie des Jeux · Optimisation de la Sécurité Réseau</div>
    <div class="hero-scenario">{SCENARIO_OPTS[st.session_state.scenario_key]}</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TOP METRICS ROW
# ══════════════════════════════════════════════════════════════════════════════

leadership_gain = stack.defender_payoff - nash.game_value
gain_pct        = (leadership_gain / abs(nash.game_value) * 100) if nash.game_value != 0 else 0.0

st.markdown(f"""
<div class="metric-grid">
  <div class="m-card" style="--accent:linear-gradient(90deg,#00d4ff,#0099bb)">
    <div class="m-icon">🏢</div>
    <div class="m-label">Nœuds réseau</div>
    <div class="m-value">{len(game.nodes)}</div>
  </div>
  <div class="m-card" style="--accent:linear-gradient(90deg,#7c3aed,#a78bfa)">
    <div class="m-icon">🛡️</div>
    <div class="m-label">Actions défense</div>
    <div class="m-value">{len(game.defense_actions)}</div>
  </div>
  <div class="m-card" style="--accent:linear-gradient(90deg,#a78bfa,#7c3aed)">
    <div class="m-icon">🎯</div>
    <div class="m-label">Valeur jeu (Nash)</div>
    <div class="m-value" style="color:#a78bfa;text-shadow:0 0 20px rgba(124,58,237,.5)">{nash.game_value:.2f}</div>
  </div>
  <div class="m-card" style="--accent:linear-gradient(90deg,#00ff88,#00cc6a)">
    <div class="m-icon">🏆</div>
    <div class="m-label">Gain leadership</div>
    <div class="m-value" style="color:#00ff88;text-shadow:0 0 20px rgba(0,255,136,.5)">+{leadership_gain:.2f}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌐  Réseau",
    "📊  Matrice",
    "🎯  Nash",
    "👑  Stackelberg",
    "⚖️  Comparaison",
])


# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — RÉSEAU
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="sec-title">🌐 Topologie du réseau</div>', unsafe_allow_html=True)

    col_net, col_sim = st.columns([5, 3])

    with col_sim:
        st.markdown("""
        <div class="card blue">
            <b>🎮 Simulateur d'attaque</b>
            <p style="color:#64748b;font-size:.88rem;margin:.4rem 0 0">
            Choisissez un nœud attaqué et une action de défense pour voir
            l'impact et le payoff en temps réel.
            </p>
        </div>
        """, unsafe_allow_html=True)

        node_names   = [n.name for n in game.nodes]
        node_sim     = st.selectbox("⚔️ Nœud attaqué", ["(aucun)"] + node_names, key="sim_node")
        def_sim      = st.selectbox("🛡️ Action de défense", def_labels, key="sim_def")

        attacked_sim = None if node_sim == "(aucun)" else node_sim
        sel_action   = next((a for a in game.defense_actions if a.name == def_sim), None)
        protected_sim = list(sel_action.protected_nodes) if sel_action else []

        if attacked_sim and sel_action:
            impact       = game.compute_attack_impact(attacked_sim)
            a_idx        = game.defense_actions.index(sel_action)
            n_idx        = node_names.index(attacked_sim)
            payoff_cell  = M_def[a_idx, n_idx]
            ip           = sel_action.protects(attacked_sim)
            residual     = (1.0 - game.defense_success_rate) if ip else 1.0
            damage       = impact * residual

            defended_badge = (
                f'<span class="badge b-green">🛡️ -{(1-residual):.0%} dégâts</span>'
                if ip else
                '<span class="badge b-red">⚠️ Non protégé</span>'
            )
            st.markdown(f"""
            <div class="card {'green' if ip else 'red'}">
                <b>📊 Résultat de la simulation</b><br><br>
                <span class="badge b-red">⚔️ {attacked_sim}</span>
                {defended_badge}<br><br>
                Impact brut&nbsp;: <b>{impact:.2f}</b><br>
                Dommage réel&nbsp;: <b>{damage:.2f}</b><br>
                Coût défense&nbsp;: <b>{sel_action.cost:.2f}</b><br>
                <b>Payoff défenseur&nbsp;: {payoff_cell:.2f}</b>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card">
                <p style="color:#475569;font-size:.88rem;margin:0">
                👆 Sélectionnez un nœud et une action pour lancer la simulation.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with col_net:
        fig_net = _network_fig(game, protected=protected_sim, attacked=attacked_sim)
        st.plotly_chart(fig_net, width="stretch", config={"displayModeBar": False})

    # Impact chart + node table
    st.markdown('<div class="sec-title">💥 Impact d\'attaque par nœud</div>', unsafe_allow_html=True)
    col_imp, col_tab = st.columns([3, 2])

    with col_imp:
        st.plotly_chart(_impact_fig(game), width="stretch", config={"displayModeBar": False})

    with col_tab:
        node_df = pd.DataFrame([
            {
                "🖥️ Nœud":         n.name,
                "💰 Valeur":        n.attack_value,
                "🛡️ Coût déf.":    n.defense_cost,
                "⚠️ Vuln.":         f"{n.vulnerability:.0%}",
                "📌 Criticité":     n.criticality,
                "💥 Impact":        f"{game.compute_attack_impact(n.name):.2f}",
            }
            for n in sorted(game.nodes, key=lambda x: -game.compute_attack_impact(x.name))
        ])
        st.dataframe(node_df, width="stretch", hide_index=True, height=300)


# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — MATRICE DE PAYOFF
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="sec-title">📊 Matrice de Payoff</div>', unsafe_allow_html=True)

    col_info, col_toggle = st.columns([4, 1])
    with col_toggle:
        mode = st.radio("Vue", ["Défenseur", "Attaquant"], horizontal=False, key="matrix_mode")
    with col_info:
        st.markdown("""
        <div class="card blue">
            La <b>matrice de payoff</b> représente l'utilité pour chaque combinaison
            <em>(action de défense, nœud ciblé)</em>.<br>
            🟩 <b>Vert</b> → bon pour le défenseur &nbsp;|&nbsp;
            🟥 <b>Rouge</b> → perte élevée pour le défenseur.
        </div>
        """, unsafe_allow_html=True)

    if mode == "Défenseur":
        mat   = M_def
        title = "📊 Payoff du Défenseur  (lignes = défense · colonnes = attaque)"
    else:
        mat   = M_att
        title = "📊 Payoff de l'Attaquant  (lignes = défense · colonnes = attaque)"

    st.plotly_chart(
        _heatmap_fig(mat, def_labels, att_labels, title),
        width="stretch", config={"displayModeBar": False},
    )

    with st.expander("📋 Données brutes & export CSV"):
        df_mat = pd.DataFrame(mat, index=def_labels, columns=att_labels)
        st.dataframe(
            df_mat.style.format("{:.3f}").background_gradient(cmap="RdYlGn"),
            height=320,
        )
        st.download_button(
            "⬇️ Télécharger CSV",
            df_mat.to_csv().encode(),
            "payoff_matrix.csv",
            "text/csv",
        )


# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — NASH EQUILIBRIUM
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="sec-title">🎯 Équilibre de Nash — Stratégies Mixtes</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="card purple">
        <b>🎯 Principe de l'équilibre de Nash</b><br>
        Dans un équilibre de Nash en <b>stratégies mixtes</b>, aucun joueur ne peut
        augmenter son espérance de gain en changeant <em>unilatéralement</em> sa stratégie.
        Les deux joueurs décident <b>simultanément</b> et de façon aléatoire.
        Ce Nash est calculé par <b>Programmation Linéaire (minimax — von Neumann 1928)</b>.
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    nash_def_exp = float(nash.defender_strategy @ M_def @ nash.attacker_strategy)
    nash_att_exp = float(nash.defender_strategy @ M_att @ nash.attacker_strategy)
    c1, c2, c3 = st.columns(3)
    c1.metric("🎯 Valeur du jeu V",     f"{nash.game_value:.4f}",
              help="Payoff garanti au défenseur à l'équilibre (minimax = maximin)")
    c2.metric("🛡️ Espérance défenseur", f"{nash_def_exp:.4f}",
              help="Payoff attendu du défenseur avec les stratégies Nash")
    c3.metric("⚔️ Espérance attaquant", f"{nash_att_exp:.4f}",
              help="Payoff attendu de l'attaquant avec les stratégies Nash")

    # Charts
    col_d, col_a = st.columns(2)
    with col_d:
        st.plotly_chart(
            _strategy_bars(def_labels, nash.defender_strategy,
                           "🛡️ Stratégie mixte — Défenseur",
                           "rgba(0,212,255,0.65)", C_DEF),
            width="stretch", config={"displayModeBar": False},
        )
    with col_a:
        st.plotly_chart(
            _strategy_bars(att_labels, nash.attacker_strategy,
                           "⚔️ Stratégie mixte — Attaquant",
                           "rgba(239,68,68,0.65)", C_ATT),
            width="stretch", config={"displayModeBar": False},
        )

    # Interpretation
    top_def_i = int(np.argmax(nash.defender_strategy))
    top_att_j = int(np.argmax(nash.attacker_strategy))
    st.markdown(f"""
    <div class="card blue">
        <b>🔍 Interprétation</b><br><br>
        Le défenseur joue prioritairement
        <span class="badge b-blue">{def_labels[top_def_i]}</span>
        à <b>{nash.defender_strategy[top_def_i]:.1%}</b>.<br>
        L'attaquant cible prioritairement
        <span class="badge b-red">{att_labels[top_att_j].replace("Attack ", "")}</span>
        à <b>{nash.attacker_strategy[top_att_j]:.1%}</b>.<br><br>
        <b>Propriété d'indifférence :</b> À l'équilibre, toute action dans le support
        (probabilité > 0) donne exactement le même espérance de payoff V = {nash.game_value:.3f}.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📋 Probabilités détaillées"):
        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown("**Défenseur**")
            df_d = pd.DataFrame({
                "Action": def_labels,
                "Probabilité": [f"{v:.4f}" for v in nash.defender_strategy],
            }).sort_values("Probabilité", ascending=False)
            st.dataframe(df_d, width="stretch", hide_index=True)
        with tc2:
            st.markdown("**Attaquant**")
            df_a = pd.DataFrame({
                "Nœud cible": att_labels,
                "Probabilité": [f"{v:.4f}" for v in nash.attacker_strategy],
            }).sort_values("Probabilité", ascending=False)
            st.dataframe(df_a, width="stretch", hide_index=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 4 — STACKELBERG
# ────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="sec-title">👑 Jeu de Stackelberg — Défenseur Leader</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card green">
        <b>👑 Principe du jeu de Stackelberg</b><br>
        Le défenseur (<b>leader</b>) annonce et s'engage dans sa stratégie mixte
        <em>avant</em> que l'attaquant ne choisisse.
        L'attaquant (<b>follower</b>) observe cet engagement et choisit la
        <b>meilleure réponse pure</b> qui maximise son propre gain.
        Le défenseur anticipe cette réaction et optimise son engagement en conséquence.<br><br>
        🎯 Best response de l'attaquant :
        <span class="badge b-red">⚔️ {best_node.name}</span>
        (impact total : <b>{game.compute_attack_impact(best_node.name):.2f}</b>)
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    gain4 = stack.defender_payoff - nash.game_value
    c1, c2, c3 = st.columns(3)
    c1.metric("👑 Payoff défenseur",  f"{stack.defender_payoff:.4f}",
              delta=f"+{gain4:.4f} vs Nash")
    c2.metric("⚔️ Payoff attaquant",  f"{stack.attacker_payoff:.4f}")
    c3.metric("🎯 Best response",      best_node.name)

    # Strategy chart + explanation
    col_sc, col_se = st.columns([5, 3])
    with col_sc:
        st.plotly_chart(
            _strategy_bars(def_labels, stack.defender_strategy,
                           "🛡️ Engagement optimal du Défenseur (Stackelberg)",
                           "rgba(0,255,136,0.65)", C_STACK),
            width="stretch", config={"displayModeBar": False},
        )
    with col_se:
        top_stack_i = int(np.argmax(stack.defender_strategy))
        st.markdown(f"""
        <div class="card green">
            <b>🔍 Lecture du résultat</b><br><br>
            L'engagement dominant du défenseur est<br>
            <span class="badge b-green">{def_labels[top_stack_i]}</span>
            à <b>{stack.defender_strategy[top_stack_i]:.1%}</b>.<br><br>
            En voyant cet engagement, l'attaquant choisit de façon <b>déterministe</b>
            le nœud <span class="badge b-red">⚔️ {best_node.name}</span>
            car c'est sa meilleure réponse pure.
        </div>
        """, unsafe_allow_html=True)

        df_stk = pd.DataFrame({
            "Action défense": def_labels,
            "Probabilité": stack.defender_strategy,
        })
        df_stk = df_stk[df_stk["Probabilité"] > 0.001].sort_values(
            "Probabilité", ascending=False)
        df_stk["Probabilité"] = df_stk["Probabilité"].map("{:.4f}".format)
        st.dataframe(df_stk, width="stretch", hide_index=True)

    # Attacked node details
    st.markdown('<div class="sec-title">⚔️ Nœud ciblé par l\'attaquant</div>', unsafe_allow_html=True)
    ca1, ca2, ca3, ca4, ca5 = st.columns(5)
    ca1.metric("🏷️ Nœud",          best_node.name)
    ca2.metric("💰 Valeur",         best_node.attack_value)
    ca3.metric("⚠️ Vulnérabilité",   f"{best_node.vulnerability:.0%}")
    ca4.metric("📌 Criticité",       best_node.criticality)
    ca5.metric("💥 Impact total",    f"{game.compute_attack_impact(best_node.name):.2f}")


# ────────────────────────────────────────────────────────────────────────────
# TAB 5 — COMPARISON
# ────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="sec-title">⚖️ Nash vs Stackelberg — Analyse Comparative</div>',
                unsafe_allow_html=True)

    # Summary cards
    c_n, c_s, c_g = st.columns(3)
    with c_n:
        st.markdown(f"""
        <div class="card purple" style="text-align:center">
            <div style="font-size:1.8rem">🎯</div>
            <b>Nash — Simultané</b><br>
            <div style="font-size:2.2rem;font-weight:800;color:#a78bfa;
                        font-family:'JetBrains Mono',monospace;margin:.3rem 0">
                {nash.game_value:.3f}
            </div>
            <div style="color:#475569;font-size:.82rem">Valeur garantie du jeu</div>
        </div>
        """, unsafe_allow_html=True)
    with c_s:
        st.markdown(f"""
        <div class="card green" style="text-align:center">
            <div style="font-size:1.8rem">👑</div>
            <b>Stackelberg — Leader</b><br>
            <div style="font-size:2.2rem;font-weight:800;color:#00ff88;
                        font-family:'JetBrains Mono',monospace;margin:.3rem 0">
                {stack.defender_payoff:.3f}
            </div>
            <div style="color:#475569;font-size:.82rem">Payoff en tant que leader</div>
        </div>
        """, unsafe_allow_html=True)
    with c_g:
        st.markdown(f"""
        <div class="card amber" style="text-align:center">
            <div style="font-size:1.8rem">🏆</div>
            <b>Gain du Leadership</b><br>
            <div style="font-size:2.2rem;font-weight:800;color:#fbbf24;
                        font-family:'JetBrains Mono',monospace;margin:.3rem 0">
                +{leadership_gain:.3f}
            </div>
            <div style="color:#475569;font-size:.82rem">+{gain_pct:.1f}% de performance</div>
        </div>
        """, unsafe_allow_html=True)

    # Comparison bar chart
    st.plotly_chart(_comparison_fig(nash.game_value, stack.defender_payoff),
                    width="stretch", config={"displayModeBar": False})

    # Side-by-side strategy comparison
    st.markdown('<div class="sec-title">🛡️ Stratégies du Défenseur — Côte à Côte</div>',
                unsafe_allow_html=True)
    cc1, cc2 = st.columns(2)
    with cc1:
        st.plotly_chart(
            _strategy_bars(def_labels, nash.defender_strategy,
                           "Nash — Stratégie mixte",
                           "rgba(124,58,237,0.65)", C_NASH),
            width="stretch", config={"displayModeBar": False},
        )
    with cc2:
        st.plotly_chart(
            _strategy_bars(def_labels, stack.defender_strategy,
                           "Stackelberg — Engagement optimal",
                           "rgba(0,255,136,0.65)", C_STACK),
            width="stretch", config={"displayModeBar": False},
        )

    # Narrative
    top_def_nash  = def_labels[int(np.argmax(nash.defender_strategy))]
    top_def_stack = def_labels[int(np.argmax(stack.defender_strategy))]
    top_att_nash  = att_labels[int(np.argmax(nash.attacker_strategy))].replace("Attack ", "")

    st.markdown(f"""
    <div class="card blue">
        <b>📖 Ce que cet outil apporte vs une défense naïve</b><br><br>

        <b>Défense naïve</b> : protéger toujours le même nœud
        (ex. toujours <em>{att_labels[0].replace("Attack ","")}</em>) sans stratégie mixte.
        L'attaquant peut alors s'adapter et exploiter les nœuds non protégés
        pour un gain maximal.<br><br>

        <b>Nash (stratégies mixtes)</b> : le défenseur randomise ses actions selon
        <span class="badge b-purple">{top_def_nash} ({nash.defender_strategy[int(np.argmax(nash.defender_strategy))]:.1%})</span>
        L'attaquant répond en ciblant
        <span class="badge b-red">⚔️ {top_att_nash} ({np.max(nash.attacker_strategy):.1%})</span>.
        Valeur garantie : <b>{nash.game_value:.3f}</b>.<br><br>

        <b>Stackelberg (leader)</b> : en s'engageant publiquement dans
        <span class="badge b-green">{top_def_stack} ({stack.defender_strategy[int(np.argmax(stack.defender_strategy))]:.1%})</span>,
        le défenseur provoque une best response pure de l'attaquant vers
        <span class="badge b-red">⚔️ {best_node.name}</span>.
        Payoff réalisé : <b>{stack.defender_payoff:.3f}</b>
        — soit <b>+{leadership_gain:.3f}</b> de mieux qu'en Nash.<br><br>

        <b>🔑 Conclusion :</b> la théorie des jeux permet de dépasser
        une défense intuitive ou fixe, et de quantifier précisément l'avantage
        stratégique d'agir en premier (<em>first-mover advantage</em>).
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📚 Rappel des concepts théoriques"):
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("""
            #### 🎯 Nash (von Neumann, 1928)
            - Jeu simultané à somme nulle
            - Chaque joueur **randomise** ses actions
            - Aucun n'a intérêt à dévier unilatéralement
            - Calculé par LP minimax / maximin
            - Valeur unique garantie par le **Théorème Minimax**
            - Formule LP : max V s.t. M^T·p ≥ V·1, Σp=1, p≥0
            """)
        with col_t2:
            st.markdown("""
            #### 👑 Stackelberg (von Stackelberg, 1934)
            - Jeu séquentiel leader-follower
            - Défenseur s'engage **publiquement** en premier
            - Attaquant choisit la **best response pure**
            - Défenseur anticipe et optimise sous cette contrainte
            - Propriété fondamentale : **Payoff Stackelberg ≥ Nash**
            - Calculé par LP pour chaque best response candidate j*
            """)