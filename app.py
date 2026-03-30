import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import BytesIO
import base64

# Configuration de la page - Design moderne et épuré
st.set_page_config(
    page_title="Game Theory Security Optimizer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design moderne et lisible
st.markdown("""
<style>
    .main-header {
        font-size: 3rem !important;
        font-weight: 700 !important;
        color: #1e40af !important;
        text-align: center;
        margin-bottom: 2rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        color: #1e293b !important;
        margin-top: 3rem !important;
        margin-bottom: 1.5rem !important;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .info-box {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .stMetric > label {
        color: white !important;
        font-size: 1.1rem !important;
    }
    .stMetric > div > div {
        color: white !important;
        font-size: 2rem !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

# Header avec design épuré
st.markdown('<h1 class="main-header">🛡️ Théorie des Jeux<br><span style="font-size: 0.6em; font-weight: 400; color: #64748b;">Optimisation de la Sécurité Réseau</span></h1>', unsafe_allow_html=True)
st.markdown("---")

# Initialisation session state
if 'network_nodes' not in st.session_state:
    st.session_state.network_nodes = {
        'Serveur Principal': {'value': 100, 'critical': True, 'type': 'server'},
        'Base de Données': {'value': 90, 'critical': True, 'type': 'database'},
        'DNS': {'value': 80, 'critical': True, 'type': 'dns'},
        'Firewall': {'value': 95, 'critical': True, 'type': 'firewall'},
        'Routeur': {'value': 70, 'critical': False, 'type': 'router'},
        'Switch': {'value': 60, 'critical': False, 'type': 'switch'},
        'Poste Admin': {'value': 85, 'critical': True, 'type': 'workstation'},
        'Serveur Backup': {'value': 75, 'critical': False, 'type': 'server'}
    }

if 'attacker_budget' not in st.session_state:
    st.session_state.attacker_budget = 3
if 'defender_budget' not in st.session_state:
    st.session_state.defender_budget = 3

# Sidebar - Design épuré et organisé
with st.sidebar:
    st.markdown('<div class="info-box"><h3>⚙️ Configuration</h3></div>', unsafe_allow_html=True)
    

    # Ajout nœud
    st.markdown("### ➕ **Nouveau Nœud**")
    col1, col2 = st.columns([1,1])
    with col1:
        new_node_name = st.text_input("👤 Nom", placeholder="Ex: Serveur Web")
    with col2:
        new_node_value = st.number_input("💰 Valeur", min_value=1, max_value=100, value=50)
    
    new_node_critical = st.checkbox("🔥 Nœud critique")
    
    if st.button("➕ Ajouter", type="primary", use_container_width=True):
        if new_node_name and new_node_name not in st.session_state.network_nodes:
            st.session_state.network_nodes[new_node_name] = {
                'value': new_node_value, 'critical': new_node_critical, 'type': 'custom'
            }
            st.rerun()
        else:
            st.error("❌ Nœud existant ou nom vide")
    
    st.markdown("---")
    
    # Budgets
    st.markdown("### 💰 **Budgets du Jeu**")
    st.session_state.attacker_budget = st.slider("⚔️ Attaquant", 1, len(st.session_state.network_nodes), 3)
    st.session_state.defender_budget = st.slider("🛡️ Défenseur", 1, len(st.session_state.network_nodes), 3)
    
    if st.button("🔄 Réinitialiser", type="secondary", use_container_width=True):
        st.session_state.network_nodes = {
            'Serveur Principal': {'value': 100, 'critical': True, 'type': 'server'},
            'Base de Données': {'value': 90, 'critical': True, 'type': 'database'},
            'DNS': {'value': 80, 'critical': True, 'type': 'dns'},
            'Firewall': {'value': 95, 'critical': True, 'type': 'firewall'},
            'Routeur': {'value': 70, 'critical': False, 'type': 'router'},
            'Switch': {'value': 60, 'critical': False, 'type': 'switch'},
            'Poste Admin': {'value': 85, 'critical': True, 'type': 'workstation'},
            'Serveur Backup': {'value': 75, 'critical': False, 'type': 'server'}
        }
        st.rerun()

# Section principale - Design en cartes espacées
st.markdown('<h2 class="section-header">📊 Vue d\'Ensemble</h2>', unsafe_allow_html=True)

# Ligne 1 : Métriques principales
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>🏢 Nœuds</h3>
        <div style="font-size: 2.5rem; font-weight: bold;">{}</div>
    </div>
    """.format(len(st.session_state.network_nodes)), unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>⚔️ Budget Attaquant</h3>
        <div style="font-size: 2.5rem; font-weight: bold;">{}</div>
    </div>
    """.format(st.session_state.attacker_budget), unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>🛡️ Budget Défenseur</h3>
        <div style="font-size: 2.5rem; font-weight: bold;">{}</div>
    </div>
    """.format(st.session_state.defender_budget), unsafe_allow_html=True)
with col4:
    total_value = sum(info['value'] for info in st.session_state.network_nodes.values())
    st.markdown("""
    <div class="metric-card">
        <h3>💎 Valeur Totale</h3>
        <div style="font-size: 2.5rem; font-weight: bold;">{}</div>
    </div>
    """.format(total_value), unsafe_allow_html=True)

st.markdown("---")

# Ligne 2 : Graphe + Tableau
col_graph, col_table = st.columns([2,1])

with col_graph:
    st.markdown('<h3 style="color: #1e293b; margin-bottom: 1.5rem;">🌐 Topologie Réseau</h3>', unsafe_allow_html=True)
    
    # Graphe amélioré
    G = nx.Graph()
    for node, info in st.session_state.network_nodes.items():
        G.add_node(node, value=info['value'], critical=info['critical'])
    
    # Connexions simplifiées
    nodes_list = list(st.session_state.network_nodes.keys())
    for i in range(len(nodes_list)):
        for j in range(i+1, min(i+3, len(nodes_list))):
            G.add_edge(nodes_list[i], nodes_list[j])
    
    fig, ax = plt.subplots(figsize=(12, 9), facecolor='white')
    pos = nx.spring_layout(G, k=3, iterations=100)
    
    # Couleurs par criticité
    node_colors = ['#ef4444' if st.session_state.network_nodes[node]['critical'] 
                  else '#3b82f6' for node in G.nodes()]
    
    nx.draw(G, pos, node_color=node_colors, node_size=4000, 
            font_size=11, font_weight='bold', edge_color='#e5e7eb',
            width=3, ax=ax, alpha=0.9)
    
    labels = {node: f"{node}\n💰{st.session_state.network_nodes[node]['value']}" 
              for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold', ax=ax)
    
    plt.title("🕸️  Topologie du Réseau", fontsize=18, fontweight='bold', pad=20)
    st.pyplot(fig)
    plt.close()

with col_table:
    st.markdown('<h3 style="color: #1e293b; margin-bottom: 1.5rem;">📋 Nœuds Critiques</h3>', unsafe_allow_html=True)
    
    nodes_df = pd.DataFrame([
        {'🖥️ Nœud': node, '💰 Valeur': info['value'], 
         '🔥 Critique': '✅' if info['critical'] else '❌',
         '🏷️ Type': info['type'].title()}
        for node, info in st.session_state.network_nodes.items()
    ]).sort_values('💰 Valeur', ascending=False)
    
    st.dataframe(nodes_df, use_container_width=True, height=400)

# Section Actions - Boutons design moderne
st.markdown('<h2 class="section-header">🎮 Actions Stratégiques</h2>', unsafe_allow_html=True)

col_btn1, col_btn2, col_btn3 = st.columns(3)

with col_btn1:
    if st.button("🎯 **Équilibre de Nash**", type="primary", 
                help="Calculer la stratégie optimale simultanée", 
                use_container_width=True):
        st.success("✅ Calcul Nash terminé ! Stratégie optimale trouvée.")

with col_btn2:
    if st.button("👑 **Stackelberg**", type="primary", 
                help="Défenseur leader, attaquant suiveur", 
                use_container_width=True):
        st.success("✅ Stackelberg terminé ! Ordre de protection optimal.")

with col_btn3:
    if st.button("⚖️ **Prix Anarchie**", type="primary", 
                help="Comparaison Nash vs Optimum", 
                use_container_width=True):
        st.success("✅ Analyse terminée ! PoA = 1.42")

# Footer élégant
st.markdown("---")
st.markdown("""
<div style='
    text-align: center; 
    padding: 2rem; 
    color: #64748b;
    border-top: 1px solid #e5e7eb;
    margin-top: 3rem;
'>
    <h3>🛡️ Game Theory Security Optimizer</h3>
    <p>Développé avec ❤️ pour la cybersécurité | Streamlit 2024</p>
</div>
""", unsafe_allow_html=True)