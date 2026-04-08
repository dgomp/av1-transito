"""
=============================================================================
DASHBOARD - AV1: Apoio à Decisão para Campanhas Educativas no Trânsito
=============================================================================
Execução:
    streamlit run dashboard.py

O modelo é retreinado automaticamente a cada abertura do dashboard.
=============================================================================
"""

import os
import warnings
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard - Sinistros de Trânsito PRF 2025",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CONSTANTES
# ---------------------------------------------------------------------------
POSSIVEIS_CAMINHOS = [
    os.path.join(os.path.dirname(__file__), 'datatran2025.csv'),
    os.path.join(os.path.dirname(__file__), 'data', 'datatran2025.csv'),
    os.path.join(os.path.dirname(__file__), 'data', 'sinistros_2025.csv'),
]

FEATURES = [
    'dia_semana', 'fase_dia', 'condicao_metereologica',
    'tipo_pista', 'tracado_via', 'causa_acidente',
    'tipo_acidente', 'uso_solo', 'sentido_via', 'uf',
]

COLUNA_ALVO = 'classificacao_acidente'

CORES = {
    'Com Vítimas Feridas': '#e67e22',
    'Com Vítimas Fatais':  '#c0392b',
    'Sem Vítimas':         '#27ae60',
}

# ---------------------------------------------------------------------------
# FUNÇÕES DE DADOS E MODELO
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Carregando dados da PRF...")
def carregar_dados() -> pd.DataFrame:
    for caminho in POSSIVEIS_CAMINHOS:
        if os.path.exists(caminho):
            return pd.read_csv(caminho, sep=';', encoding='latin-1', low_memory=False)
    st.error("Arquivo `datatran2025.csv` não encontrado. Coloque-o na raiz do projeto ou na pasta `data/`.")
    st.stop()


@st.cache_resource(show_spinner="Treinando modelo de Árvore de Decisão...")
def treinar_modelo(max_depth: int, min_samples_leaf: int):
    """
    Carrega, pré-processa e treina o modelo.
    Retornos: modelo, X_test, y_test, y_pred, importancias, encoders, df_clean
    """
    df = carregar_dados()

    # Pré-processamento
    df_clean = df.dropna(subset=[COLUNA_ALVO]).copy()
    df_clean['com_vitimas'] = df_clean[COLUNA_ALVO].apply(
        lambda x: 1 if x in ('Com Vítimas Feridas', 'Com Vítimas Fatais') else 0
    )

    X_raw = df_clean[FEATURES].fillna('Ignorado')
    y = df_clean['com_vitimas']

    encoders = {}
    X_enc = X_raw.copy()
    for col in FEATURES:
        le = LabelEncoder()
        X_enc[col] = le.fit_transform(X_raw[col].astype(str))
        encoders[col] = le

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.20, random_state=42, stratify=y
    )

    modelo = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    importancias = pd.DataFrame({
        'feature': FEATURES,
        'importancia': modelo.feature_importances_
    }).sort_values('importancia', ascending=False).reset_index(drop=True)
    importancias['importancia_pct'] = importancias['importancia'] * 100

    return modelo, X_train, X_test, y_train, y_test, y_pred, importancias, encoders, df_clean


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Flag_of_Brazil.svg/330px-Flag_of_Brazil.svg.png", width=80)
    st.title("Configurações do Modelo")

    max_depth = st.slider("Profundidade máxima da árvore", min_value=2, max_value=15, value=7, step=1)
    min_samples = st.slider("Mínimo de amostras por folha", min_value=10, max_value=200, value=50, step=10)

    st.divider()
    st.caption("AV1 - Sistemas de Apoio à Decisão")
    st.caption("Sistemas de Informação")

# ---------------------------------------------------------------------------
# CARREGAR / TREINAR
# ---------------------------------------------------------------------------
modelo, X_train, X_test, y_train, y_test, y_pred, importancias, encoders, df_clean = treinar_modelo(
    max_depth, min_samples
)

df_raw = carregar_dados()

# ---------------------------------------------------------------------------
# CABEÇALHO
# ---------------------------------------------------------------------------
st.title("🚦 Apoio à Decisão para Campanhas Educativas no Trânsito")
st.markdown(
    "**Disciplina:** Sistemas de Apoio à Decisão &nbsp;|&nbsp; "
    "**Modelo:** Árvore de Decisão (Entropia / Ganho de Informação) &nbsp;|&nbsp; "
    "**Dados:** PRF 2025"
)
st.divider()

# ---------------------------------------------------------------------------
# SEÇÃO 1 - VISÃO GERAL
# ---------------------------------------------------------------------------
st.header("1. Visão Geral dos Dados")

total = len(df_raw)
total_valido = df_clean['com_vitimas'].count()
com_vitimas  = df_clean['com_vitimas'].sum()
sem_vitimas  = total_valido - com_vitimas

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de Sinistros", f"{total:,}".replace(",", "."))
col2.metric("Com Vítimas", f"{com_vitimas:,}".replace(",", "."), f"{com_vitimas/total_valido*100:.1f}%")
col3.metric("Sem Vítimas", f"{sem_vitimas:,}".replace(",", "."), f"{sem_vitimas/total_valido*100:.1f}%")
col4.metric("Vítimas Fatais", f"{df_clean[df_clean[COLUNA_ALVO]=='Com Vítimas Fatais'].shape[0]:,}".replace(",", "."))

st.divider()

# Gráficos de distribuição
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Distribuição por Classificação")
    contagem = df_raw[COLUNA_ALVO].value_counts(dropna=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    cores_lista = [CORES.get(c, '#95a5a6') for c in contagem.index]
    wedges, texts, autotexts = ax.pie(
        contagem.values, labels=contagem.index,
        autopct='%1.2f%%', colors=cores_lista,
        startangle=140, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight('bold')
    ax.set_title('Classificação dos Sinistros PRF 2025', fontweight='bold')
    st.pyplot(fig)
    plt.close()

with col_b:
    st.subheader("Distribuição por Fase do Dia")
    fase = df_raw['fase_dia'].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(fase.index[::-1], fase.values[::-1], color='#f39c12', edgecolor='white')
    for i, v in enumerate(fase.values[::-1]):
        ax.text(v + 100, i, f'{v:,}', va='center', fontsize=9)
    ax.set_xlabel('Quantidade de Sinistros')
    ax.set_title('Sinistros por Fase do Dia', fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    st.pyplot(fig)
    plt.close()

# ---------------------------------------------------------------------------
# SEÇÃO 2 - ANÁLISE EXPLORATÓRIA INTERATIVA
# ---------------------------------------------------------------------------
st.divider()
st.header("2. Análise Exploratória Interativa")

col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    ufs_disponiveis = sorted(df_raw['uf'].dropna().unique().tolist())
    ufs_selecionadas = st.multiselect("Filtrar por UF", ufs_disponiveis, default=[])
with col_f2:
    fases = sorted(df_raw['fase_dia'].dropna().unique().tolist())
    fases_selecionadas = st.multiselect("Filtrar por Fase do Dia", fases, default=[])
with col_f3:
    dias = ['segunda-feira','terça-feira','quarta-feira','quinta-feira','sexta-feira','sábado','domingo']
    dias_selecionados = st.multiselect("Filtrar por Dia da Semana", dias, default=[])

df_filtrado = df_raw.copy()
if ufs_selecionadas:
    df_filtrado = df_filtrado[df_filtrado['uf'].isin(ufs_selecionadas)]
if fases_selecionadas:
    df_filtrado = df_filtrado[df_filtrado['fase_dia'].isin(fases_selecionadas)]
if dias_selecionados:
    df_filtrado = df_filtrado[df_filtrado['dia_semana'].isin(dias_selecionados)]

st.caption(f"Registros exibidos: **{len(df_filtrado):,}** de {total:,}")

col_e1, col_e2 = st.columns(2)

with col_e1:
    st.subheader("Top 10 Causas de Acidente")
    top_causas = df_filtrado['causa_acidente'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(top_causas.index[::-1], top_causas.values[::-1], color='#3498db', edgecolor='white')
    for i, v in enumerate(top_causas.values[::-1]):
        ax.text(v + 10, i, f'{v:,}', va='center', fontsize=8)
    ax.set_xlabel('Quantidade')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_e2:
    st.subheader("Top 10 Tipos de Acidente")
    top_tipos = df_filtrado['tipo_acidente'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(top_tipos.index[::-1], top_tipos.values[::-1], color='#e74c3c', edgecolor='white')
    for i, v in enumerate(top_tipos.values[::-1]):
        ax.text(v + 10, i, f'{v:,}', va='center', fontsize=8)
    ax.set_xlabel('Quantidade')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

col_e3, col_e4 = st.columns(2)

with col_e3:
    st.subheader("Sinistros por Condição Meteorológica")
    meteor = df_filtrado['condicao_metereologica'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(meteor.index, meteor.values, color='#27ae60', edgecolor='white')
    ax.set_ylabel('Quantidade')
    ax.tick_params(axis='x', rotation=25)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_e4:
    st.subheader("Sinistros por Tipo de Pista")
    pista = df_filtrado['tipo_pista'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    cores_pista = ['#9b59b6', '#2980b9', '#1abc9c']
    ax.bar(pista.index, pista.values, color=cores_pista[:len(pista)], edgecolor='white')
    for i, (idx, v) in enumerate(pista.items()):
        ax.text(i, v + 50, f'{v:,}', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Quantidade')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ---------------------------------------------------------------------------
# SEÇÃO 3 - MODELO: RESULTADOS
# ---------------------------------------------------------------------------
st.divider()
st.header("3. Modelo - Resultados")

acc_treino = accuracy_score(y_train, modelo.predict(X_train))
acc_teste  = accuracy_score(y_test,  y_pred)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Métricas
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
col_m1.metric("Acurácia Treino", f"{acc_treino*100:.2f}%")
col_m2.metric("Acurácia Teste",  f"{acc_teste*100:.2f}%",
              delta=f"{(acc_teste-acc_treino)*100:+.2f}% vs treino")
col_m3.metric("Nós na Árvore",   modelo.tree_.node_count)
col_m4.metric("Profundidade Real", modelo.get_depth())
col_m5.metric("Folhas",           modelo.get_n_leaves())

col_r1, col_r2 = st.columns(2)

with col_r1:
    st.subheader("Matriz de Confusão")
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ['Sem Vítimas', 'Com Vítimas']
    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=labels, yticklabels=labels,
        cmap='Blues', ax=ax,
        linewidths=0.5, linecolor='gray',
        annot_kws={'size': 13}
    )
    ax.set_xlabel('Previsto pelo Modelo', fontsize=11)
    ax.set_ylabel('Valor Real', fontsize=11)
    ax.set_title('Matriz de Confusão - Conjunto de Teste', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    col_conf1, col_conf2 = st.columns(2)
    col_conf1.metric("Verdadeiros Positivos", f"{tp:,}".replace(",", "."))
    col_conf1.metric("Falsos Negativos", f"{fn:,}".replace(",", "."))
    col_conf2.metric("Verdadeiros Negativos", f"{tn:,}".replace(",", "."))
    col_conf2.metric("Falsos Positivos", f"{fp:,}".replace(",", "."))

with col_r2:
    st.subheader("Importância das Features")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    cores_imp = ['#c0392b' if i < 3 else '#2980b9' for i in range(len(importancias))]
    bars = ax.barh(
        importancias['feature'][::-1],
        importancias['importancia_pct'][::-1],
        color=cores_imp[::-1], edgecolor='white'
    )
    for bar, val in zip(bars, importancias['importancia_pct'][::-1]):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}%', va='center', fontsize=8, fontweight='bold'
        )
    ax.set_xlabel('Importância (%)')
    ax.set_title('Ganho de Informação por Feature', fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color='#c0392b', label='Top 3'),
            plt.Rectangle((0, 0), 1, 1, color='#2980b9', label='Demais'),
        ],
        fontsize=9, loc='lower right'
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Relatório de classificação
with st.expander("Ver Relatório de Classificação Completo"):
    report = classification_report(
        y_test, y_pred,
        target_names=['Sem Vítimas', 'Com Vítimas'],
        output_dict=True
    )
    df_report = pd.DataFrame(report).T.round(2)
    st.dataframe(df_report, width='stretch')

# ---------------------------------------------------------------------------
# SEÇÃO 4 - ÁRVORE DE DECISÃO
# ---------------------------------------------------------------------------
st.divider()
st.header("4. Visualização da Árvore de Decisão")

depth_viz = st.slider(
    "Profundidade a exibir (apenas visual - não afeta o modelo)",
    min_value=1, max_value=min(max_depth, 5), value=3
)

fig, ax = plt.subplots(figsize=(30, 12))
plot_tree(
    modelo,
    feature_names=FEATURES,
    class_names=['Sem Vítimas', 'Com Vítimas'],
    filled=True, rounded=True,
    max_depth=depth_viz,
    fontsize=9, ax=ax,
    impurity=True
)
ax.set_title(
    f'Árvore de Decisão - PRF 2025  |  Critério: Entropia  |  Profundidade exibida: {depth_viz}',
    fontsize=13, fontweight='bold', pad=15
)
plt.tight_layout()
st.pyplot(fig)
plt.close()

with st.expander("Ver Regras da Árvore em Texto (primeiros 2 níveis)"):
    regras = export_text(modelo, feature_names=FEATURES, max_depth=2)
    st.code(regras, language='text')

# ---------------------------------------------------------------------------
# SEÇÃO 5 - CAMPANHAS EDUCATIVAS
# ---------------------------------------------------------------------------
st.divider()
st.header("5. Proposta de Campanhas Educativas")

mapa_campanhas = {
    'causa_acidente': {
        'nome': "Dirija com Atenção",
        'icone': '🧠',
        'foco': "Comportamento do condutor: distração, reação tardia, álcool, velocidade incompatível",
        'acoes': "Blitze educativas; alertas em apps de navegação; campanhas nas redes sociais com dados reais",
        'alvo': "Condutores em geral, especialmente jovens de 18 a 30 anos",
        'cor': '#e74c3c',
    },
    'tipo_acidente': {
        'nome': "Distância Segura Salva Vidas",
        'icone': '🚛',
        'foco': "Prevenção de colisões traseiras, frontais e saídas de pista",
        'acoes': "Painéis de mensagem variável; fiscalização com câmeras inteligentes; treinamento para motoristas de veículos pesados",
        'alvo': "Motoristas de caminhões, ônibus e veículos de carga",
        'cor': '#e67e22',
    },
    'uf': {
        'nome': "Campanhas Regionalizadas por Estado",
        'icone': '🗺️',
        'foco': "Estados com maior concentração de sinistros com vítimas",
        'acoes': "Parcerias entre SENATRAN e DESTRANs estaduais; campanhas adaptadas ao perfil regional",
        'alvo': "Órgãos gestores estaduais e condutores por UF",
        'cor': '#3498db',
    },
    'fase_dia': {
        'nome': "Noite Exige Mais Cuidado",
        'icone': '🌙',
        'foco': "Plena noite e madrugada - menor visibilidade e maior fadiga",
        'acoes': "Programas de descanso em postos homologados; alertas luminosos em pontos críticos",
        'alvo': "Caminhoneiros, motoristas de aplicativo e viajantes noturnos",
        'cor': '#8e44ad',
    },
    'uso_solo': {
        'nome': "Zona Rural - Atenção Redobrada",
        'icone': '🌾',
        'foco': "Rodovias em áreas rurais onde sinistros são mais graves",
        'acoes': "Reforço de sinalização em acessos rurais; redutores de velocidade; ampliação de postos de socorro",
        'alvo': "Condutores e comunidades rurais lindeiras às rodovias",
        'cor': '#27ae60',
    },
    'tracado_via': {
        'nome': "Curvas Perigosas - Reduza a Velocidade",
        'icone': '🔄',
        'foco': "Trechos de curva, declive e aclive com alta incidência de sinistros",
        'acoes': "Sinalização reforçada; radares em trechos críticos; mapas de risco em apps de navegação",
        'alvo': "Todos os condutores em rodovias de relevo acentuado",
        'cor': '#f39c12',
    },
    'condicao_metereologica': {
        'nome': "Chuva - Reduza a Velocidade",
        'icone': '🌧️',
        'foco': "Condições climáticas adversas: chuva, garoa, neblina",
        'acoes': "Alertas em tempo real via apps; treinamento em direção defensiva em piso molhado",
        'alvo': "Todos os condutores, com destaque para regiões de clima úmido",
        'cor': '#2980b9',
    },
    'tipo_pista': {
        'nome': "Pistas Simples - Máxima Atenção",
        'icone': '🛣️',
        'foco': "Rodovias de pista simples com maior risco de colisão frontal",
        'acoes': "Proibição de ultrapassagem com sinalização clara; pressão pela duplicação de trechos críticos",
        'alvo': "Condutores e gestores de infraestrutura rodoviária",
        'cor': '#16a085',
    },
    'dia_semana': {
        'nome': "Fim de Semana Seguro",
        'icone': '📅',
        'foco': "Fins de semana com maior concentração de sinistros",
        'acoes': "Intensificação de blitze; campanhas contra álcool ao volante; parceria com bares e casas noturnas",
        'alvo': "Jovens adultos e motoristas de lazer",
        'cor': '#d35400',
    },
    'sentido_via': {
        'nome': "Sentido Correto - Vida Garantida",
        'icone': '↔️',
        'foco': "Contramão e ultrapassagens indevidas",
        'acoes': "Fiscalização com câmeras de monitoramento; educação sobre regras de preferência",
        'alvo': "Condutores em rodovias de pista simples",
        'cor': '#7f8c8d',
    },
}

top5 = importancias.head(5)

for _, row in top5.iterrows():
    feat = row['feature']
    pct  = row['importancia_pct']
    camp = mapa_campanhas.get(feat, {})
    if not camp:
        continue

    with st.container(border=True):
        col_icon, col_info = st.columns([1, 10])
        with col_icon:
            st.markdown(f"<h1 style='text-align:center'>{camp['icone']}</h1>", unsafe_allow_html=True)
        with col_info:
            st.markdown(
                f"**{camp['nome']}** &nbsp;&nbsp; "
                f"<span style='background-color:{camp['cor']};color:white;"
                f"padding:2px 8px;border-radius:4px;font-size:0.85em'>"
                f"`{feat}` - {pct:.2f}%</span>",
                unsafe_allow_html=True
            )
            col_c1, col_c2, col_c3 = st.columns(3)
            col_c1.markdown(f"**Foco**  \n{camp['foco']}")
            col_c2.markdown(f"**Ações**  \n{camp['acoes']}")
            col_c3.markdown(f"**Público-alvo**  \n{camp['alvo']}")

# ---------------------------------------------------------------------------
# RODAPÉ
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Fonte dos dados: Polícia Rodoviária Federal - Portal de Dados Abertos do Governo Federal &nbsp;|&nbsp; "
    "Modelo: DecisionTreeClassifier (scikit-learn) &nbsp;|&nbsp; "
    "Critério: Entropia / Ganho de Informação"
)
