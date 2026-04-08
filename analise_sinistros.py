"""
=============================================================================
TRABALHO AV1 - SISTEMAS DE APOIO À DECISÃO
Curso: Sistemas de Informação
=============================================================================

Título: Apoio à Decisão para Campanhas Educativas no Trânsito
Modelo: Árvore de Decisão (critério: Entropia / Ganho de Informação)
Dados:  Sinistros de Trânsito PRF 2025 (Portal Dados Abertos Gov Federal)

Objetivo:
    Identificar atributos críticos associados à ocorrência de vítimas em
    sinistros de trânsito e propor direcionamento de campanhas educativas.

Questão gerencial:
    Quais características dos sinistros estão mais associadas à ocorrência
    de vítimas e devem ser priorizadas como foco de campanhas educativas?

Algoritmo:
    DecisionTreeClassifier com criterion='entropy' (scikit-learn).
    Internamente implementa o cálculo de Ganho de Informação (IG),
    selecionando a cada nó o atributo que maximiza a redução de entropia,
    conforme definido em:
        IG(S, A) = Entropia(S) - Σ (|Sv|/|S|) * Entropia(Sv)
    Equivalente ao algoritmo ID3 / C4.5.
=============================================================================
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # backend sem janela, compatível com qualquer ambiente
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# CONFIGURAÇÕES GLOBAIS
# ---------------------------------------------------------------------------

# Arquivo CSV (aceita na raiz do projeto ou dentro de data/)
POSSIVEIS_CAMINHOS = [
    os.path.join(os.path.dirname(__file__), 'datatran2025.csv'),
    os.path.join(os.path.dirname(__file__), 'data', 'datatran2025.csv'),
    os.path.join(os.path.dirname(__file__), 'data', 'sinistros_2025.csv'),
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

# Features categóricas a usar como preditoras (sem data leakage)
FEATURES = [
    'dia_semana',           # dia da semana do sinistro
    'fase_dia',             # amanhecer / pleno dia / anoitecer / plena noite
    'condicao_metereologica',  # chuva, céu claro, nublado, etc.
    'tipo_pista',           # simples, dupla, múltipla
    'tracado_via',          # reta, curva, declive, etc.
    'causa_acidente',       # causa principal do sinistro
    'tipo_acidente',        # colisão traseira, tombamento, atropelamento, etc.
    'uso_solo',             # área urbana (Sim) ou rural (Não)
    'sentido_via',          # crescente / decrescente
    'uf',                   # unidade federativa
]

# Variável alvo original e codificação binária
COLUNA_ALVO = 'classificacao_acidente'

# Hiperparâmetros do modelo
MAX_DEPTH = 7          # profundidade máxima, controla overfitting
MIN_SAMPLES_LEAF = 50  # mínimo de amostras por folha, garante relevância estatística
RANDOM_STATE = 42
TEST_SIZE = 0.20       # 20% para teste, 80% para treino

# ---------------------------------------------------------------------------
# FUNÇÕES
# ---------------------------------------------------------------------------

def encontrar_arquivo(caminhos: list) -> str:
    """Retorna o primeiro caminho válido da lista."""
    for caminho in caminhos:
        if os.path.exists(caminho):
            return caminho
    raise FileNotFoundError(
        "Arquivo de dados não encontrado. Certifique-se de que 'datatran2025.csv' "
        "está na raiz do projeto ou na pasta 'data/'."
    )


def carregar_dados(caminho: str) -> pd.DataFrame:
    """
    Carrega o CSV da PRF.
    Trata automaticamente separador ';' e encoding latin-1
    (padrão dos arquivos do governo brasileiro).
    """
    print(f"\n{'='*60}")
    print("1. CARREGAMENTO DOS DADOS")
    print(f"{'='*60}")
    print(f"Arquivo: {caminho}")

    df = pd.read_csv(caminho, sep=';', encoding='latin-1', low_memory=False)

    print(f"Registros carregados : {len(df):,}")
    print(f"Colunas              : {df.shape[1]}")
    print(f"\nColunas disponíveis:\n{df.columns.tolist()}")
    return df


def explorar_dados(df: pd.DataFrame) -> None:
    """Exibe estatísticas exploratórias do dataset."""
    print(f"\n{'='*60}")
    print("2. EXPLORAÇÃO DOS DADOS")
    print(f"{'='*60}")

    print("\n--- Distribuição da variável alvo ---")
    dist = df[COLUNA_ALVO].value_counts(dropna=False)
    print(dist.to_string())

    pct_com_vitimas = (
        df[COLUNA_ALVO]
        .isin(['Com Vítimas Feridas', 'Com Vítimas Fatais'])
        .mean() * 100
    )
    print(f"\n% Com Vítimas (Feridas + Fatais) : {pct_com_vitimas:.1f}%")
    print(f"% Sem Vítimas                    : {100 - pct_com_vitimas:.1f}%")

    print("\n--- Valores nulos nas features selecionadas ---")
    nulos = df[FEATURES + [COLUNA_ALVO]].isnull().sum()
    print(nulos[nulos > 0].to_string() if nulos.sum() > 0 else "Nenhum valor nulo.")


def preprocessar(df: pd.DataFrame):
    """
    Pré-processa o dataset:
      1. Remove linhas sem classificação (alvo nulo)
      2. Cria target binário: 1 = com vítimas / 0 = sem vítimas
      3. Preenche eventuais nulos nas features com 'Ignorado'
      4. Aplica LabelEncoder em cada feature categórica
      5. Divide em treino/teste estratificado

    Retorna: X_train, X_test, y_train, y_test, encoders, feature_names
    """
    print(f"\n{'='*60}")
    print("3. PRÉ-PROCESSAMENTO")
    print(f"{'='*60}")

    # --- Remover registros sem classificação ---
    df_clean = df.dropna(subset=[COLUNA_ALVO]).copy()
    removidos = len(df) - len(df_clean)
    print(f"Registros sem classificação removidos: {removidos}")
    print(f"Registros utilizados                 : {len(df_clean):,}")

    # --- Target binário ---
    # 1 = Com Vítimas (Feridas ou Fatais)  |  0 = Sem Vítimas
    df_clean['com_vitimas'] = df_clean[COLUNA_ALVO].apply(
        lambda x: 1 if x in ('Com Vítimas Feridas', 'Com Vítimas Fatais') else 0
    )
    print(f"\nTarget após binarização:")
    print(df_clean['com_vitimas'].value_counts().rename({1: 'Com Vítimas (1)', 0: 'Sem Vítimas (0)'}))

    # --- Selecionar features e preencher nulos ---
    X_raw = df_clean[FEATURES].fillna('Ignorado')
    y = df_clean['com_vitimas']

    # --- Codificação label encoding ---
    # Cada coluna categórica recebe um LabelEncoder independente
    encoders = {}
    X_enc = X_raw.copy()
    for col in FEATURES:
        le = LabelEncoder()
        X_enc[col] = le.fit_transform(X_raw[col].astype(str))
        encoders[col] = le

    print(f"\nFeatures utilizadas ({len(FEATURES)}):")
    for f in FEATURES:
        n_cat = X_raw[f].nunique()
        print(f"  {f:<30} {n_cat} categorias")

    # --- Split treino / teste ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y          # garante proporção das classes em ambos os conjuntos
    )
    print(f"\nDivisão treino/teste: {int((1-TEST_SIZE)*100)}% / {int(TEST_SIZE*100)}%")
    print(f"  Treino : {len(X_train):,} registros")
    print(f"  Teste  : {len(X_test):,} registros")

    return X_train, X_test, y_train, y_test, encoders, FEATURES


def treinar_modelo(X_train, y_train) -> DecisionTreeClassifier:
    """
    Treina a Árvore de Decisão com critério de entropia (Ganho de Informação).

    Parâmetros principais:
        criterion='entropy' → usa IG = Entropia(S) - Σ(|Sv|/|S|)*Entropia(Sv)
        max_depth=7         → limita profundidade para evitar overfitting
        min_samples_leaf=50 → nó folha precisa de ≥50 amostras (relevância)
    """
    print(f"\n{'='*60}")
    print("4. TREINAMENTO DO MODELO")
    print(f"{'='*60}")
    print(f"Algoritmo   : Árvore de Decisão (scikit-learn)")
    print(f"Critério    : Entropia (Ganho de Informação - ID3/C4.5)")
    print(f"Max depth   : {MAX_DEPTH}")
    print(f"Min leaf    : {MIN_SAMPLES_LEAF} amostras")

    modelo = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE
    )
    modelo.fit(X_train, y_train)

    print(f"\nModelo treinado com sucesso.")
    print(f"  Nós na árvore     : {modelo.tree_.node_count}")
    print(f"  Profundidade real : {modelo.get_depth()}")
    print(f"  Folhas            : {modelo.get_n_leaves()}")

    return modelo


def avaliar_modelo(modelo: DecisionTreeClassifier,
                   X_train, X_test, y_train, y_test) -> None:
    """
    Avalia o modelo com:
      - Acurácia (treino e teste) para detectar overfitting
      - Matriz de Confusão
      - Relatório de Classificação (precision, recall, F1)
    """
    print(f"\n{'='*60}")
    print("5. AVALIAÇÃO DO MODELO")
    print(f"{'='*60}")

    acc_treino = accuracy_score(y_train, modelo.predict(X_train))
    acc_teste = accuracy_score(y_test, modelo.predict(X_test))

    print(f"\nAcurácia no Treino : {acc_treino:.4f} ({acc_treino*100:.2f}%)")
    print(f"Acurácia no Teste  : {acc_teste:.4f}  ({acc_teste*100:.2f}%)")

    gap = acc_treino - acc_teste
    if gap < 0.03:
        print("  >> Sem overfitting significativo (diferenca < 3%)")
    else:
        print(f"  >> Atencao: possivel overfitting (diferenca de {gap*100:.1f}%)")

    # Relatório detalhado
    y_pred = modelo.predict(X_test)
    print("\nRelatório de Classificação (conjunto de teste):")
    print(classification_report(
        y_test, y_pred,
        target_names=['Sem Vítimas', 'Com Vítimas']
    ))

    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Matriz de Confusão:")
    print(f"  Verdadeiros Negativos (Sem Vítimas correto) : {tn:,}")
    print(f"  Falsos Positivos (Sem -> previsto Com)       : {fp:,}")
    print(f"  Falsos Negativos (Com -> previsto Sem)       : {fn:,}")
    print(f"  Verdadeiros Positivos (Com Vítimas correto) : {tp:,}")

    return y_pred, cm


def analisar_importancia(modelo: DecisionTreeClassifier,
                         feature_names: list) -> pd.DataFrame:
    """
    Calcula e exibe a importância de cada feature na árvore.
    Importância = redução total de impureza (entropia) ponderada pelo
    número de amostras que passam por cada nó de divisão.
    """
    print(f"\n{'='*60}")
    print("6. IMPORTÂNCIA DAS FEATURES (Ganho de Informação Agregado)")
    print(f"{'='*60}")

    importancias = pd.DataFrame({
        'feature': feature_names,
        'importancia': modelo.feature_importances_
    }).sort_values('importancia', ascending=False).reset_index(drop=True)

    importancias['rank'] = importancias.index + 1
    importancias['importancia_pct'] = importancias['importancia'] * 100

    print(f"\n{'Rank':<5} {'Feature':<32} {'Importância':>12} {'%':>8}")
    print('-' * 60)
    for _, row in importancias.iterrows():
        print(f"  {int(row['rank']):<4} {row['feature']:<32} "
              f"{row['importancia']:>12.4f} {row['importancia_pct']:>7.2f}%")

    return importancias


def propor_campanhas(importancias: pd.DataFrame,
                     modelo: DecisionTreeClassifier,
                     encoders: dict,
                     feature_names: list) -> None:
    """
    Com base nas features mais importantes, propõe campanhas educativas
    e ações preventivas orientadas por dados.
    """
    print(f"\n{'='*60}")
    print("7. PROPOSTA DE CAMPANHAS EDUCATIVAS")
    print(f"{'='*60}")

    top5 = importancias.head(5)['feature'].tolist()

    mapa_campanhas = {
        'causa_acidente': (
            "CAMPANHA: 'DIRIJA COM ATENÇÃO'\n"
            "  Foco  : Comportamento do condutor (distração, reação tardia, álcool)\n"
            "  Ação  : Blitze educativas, parceria com escolas de trânsito,\n"
            "           multas com foco pedagógico e campanhas nas mídias sociais.\n"
            "  Alvo  : Condutores em geral, especialmente jovens (18-30 anos)."
        ),
        'tipo_acidente': (
            "CAMPANHA: 'DISTÂNCIA SEGURA SALVA VIDAS'\n"
            "  Foco  : Prevenção de colisões traseiras e frontais\n"
            "  Ação  : Painéis eletrônicos nas rodovias indicando distância segura,\n"
            "           fiscalização de ultrapassagem proibida e velocidade.\n"
            "  Alvo  : Motoristas de veículos pesados e de longa distância."
        ),
        'tracado_via': (
            "CAMPANHA: 'CURVAS PERIGOSAS - REDUZA A VELOCIDADE'\n"
            "  Foco  : Trechos de curva, declive e aclive\n"
            "  Ação  : Sinalização reforçada, radares em trechos críticos,\n"
            "           mapas de risco disponíveis em apps de navegação.\n"
            "  Alvo  : Todos os condutores em rodovias de relevo acentuado."
        ),
        'fase_dia': (
            "CAMPANHA: 'NOITE EXIGE MAIS CUIDADO'\n"
            "  Foco  : Plena noite e madrugada (menor visibilidade)\n"
            "  Ação  : Programas de descanso em postos rodoviários,\n"
            "           alertas luminosos em regiões de alta incidência noturna.\n"
            "  Alvo  : Caminhoneiros e motoristas de aplicativo."
        ),
        'condicao_metereologica': (
            "CAMPANHA: 'CHUVA - REDUZA A VELOCIDADE'\n"
            "  Foco  : Condições climáticas adversas (chuva, garoa, neblina)\n"
            "  Ação  : Alertas em tempo real via apps e painéis de mensagem variável,\n"
            "           treinamento sobre direção defensiva em piso molhado.\n"
            "  Alvo  : Todos os condutores, com destaque para regiões de clima úmido."
        ),
        'tipo_pista': (
            "CAMPANHA: 'PISTAS SIMPLES - MÁXIMA ATENÇÃO'\n"
            "  Foco  : Rodovias de pista simples (maior risco de colisão frontal)\n"
            "  Ação  : Faixa de ultrapassagem proibida, redutores de velocidade,\n"
            "           ampliação de trechos duplicados nas rotas mais críticas.\n"
            "  Alvo  : Gestores de infraestrutura e condutores em rodovias rurais."
        ),
        'uso_solo': (
            "CAMPANHA: 'ZONA RURAL - ATENÇÃO REDOBRADA'\n"
            "  Foco  : Áreas rurais (maior severidade dos sinistros)\n"
            "  Ação  : Sinalização de travessias, redutores de velocidade\n"
            "           em vias de acesso a propriedades.\n"
            "  Alvo  : Condutores e comunidades rurais lindeiras às rodovias."
        ),
        'dia_semana': (
            "CAMPANHA: 'FIM DE SEMANA SEGURO'\n"
            "  Foco  : Dias de maior volume de sinistros (sextas, sábados, domingos)\n"
            "  Ação  : Intensificação de blitze nos fins de semana,\n"
            "           campanhas contra álcool ao volante nas saídas de festas.\n"
            "  Alvo  : Jovens adultos e motoristas de lazer."
        ),
        'sentido_via': (
            "CAMPANHA: 'SENTIDO CORRETO - VIDA GARANTIDA'\n"
            "  Foco  : Contramão e ultrapassagens indevidas\n"
            "  Ação  : Fiscalização com câmeras de monitoramento,\n"
            "           educação sobre regras de preferência nas rodovias.\n"
            "  Alvo  : Condutores em rodovias de pista simples."
        ),
        'uf': (
            "CAMPANHA: REGIONALIZADA POR ESTADO\n"
            "  Foco  : Estados com maior concentração de sinistros com vítimas\n"
            "  Ação  : Parcerias entre DENATRAN/SENATRAN e órgãos estaduais,\n"
            "           campanhas adaptadas ao perfil regional de cada UF.\n"
            "  Alvo  : Órgãos gestores de trânsito estaduais."
        ),
    }

    print("\nCom base nos 5 atributos mais relevantes identificados pelo modelo,")
    print("propõem-se as seguintes campanhas educativas prioritárias:\n")

    for i, feat in enumerate(top5, 1):
        pct = importancias.loc[importancias['feature'] == feat, 'importancia_pct'].values[0]
        print(f"[{i}] Feature: {feat.upper()} (importância: {pct:.2f}%)")
        campanha = mapa_campanhas.get(feat, f"  Campanha genérica para o atributo '{feat}'.")
        print(f"  {campanha}")
        print()


# ---------------------------------------------------------------------------
# FUNÇÕES DE VISUALIZAÇÃO
# ---------------------------------------------------------------------------

def salvar_arvore(modelo, feature_names, output_dir):
    """Salva visualização da árvore de decisão (profundidade 3 para legibilidade)."""
    caminho = os.path.join(output_dir, 'arvore_decisao.png')
    fig, ax = plt.subplots(figsize=(28, 12))
    plot_tree(
        modelo,
        feature_names=feature_names,
        class_names=['Sem Vítimas', 'Com Vítimas'],
        filled=True,
        rounded=True,
        max_depth=3,            # exibe apenas 3 níveis para visualização limpa
        fontsize=9,
        ax=ax,
        impurity=True,          # mostra entropia de cada nó
        proportion=False
    )
    ax.set_title(
        'Árvore de Decisão - Sinistros de Trânsito PRF 2025\n'
        '(Critério: Entropia / Ganho de Informação | Exibindo até profundidade 3)',
        fontsize=13, fontweight='bold', pad=20
    )
    plt.tight_layout()
    plt.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Árvore salva em: {caminho}")


def salvar_matriz_confusao(cm, output_dir):
    """Salva heatmap da matriz de confusão."""
    caminho = os.path.join(output_dir, 'matriz_confusao.png')
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ['Sem Vítimas', 'Com Vítimas']
    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=labels, yticklabels=labels,
        cmap='Blues', ax=ax,
        linewidths=0.5, linecolor='gray'
    )
    ax.set_xlabel('Previsto pelo Modelo', fontsize=12)
    ax.set_ylabel('Valor Real', fontsize=12)
    ax.set_title('Matriz de Confusão - Conjunto de Teste', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Matriz de confusão salva em: {caminho}")


def salvar_importancia(importancias, output_dir):
    """Salva gráfico de importância das features."""
    caminho = os.path.join(output_dir, 'feature_importance.png')
    fig, ax = plt.subplots(figsize=(10, 6))
    cores = ['#c0392b' if i < 3 else '#2980b9' for i in range(len(importancias))]
    bars = ax.barh(
        importancias['feature'][::-1],
        importancias['importancia_pct'][::-1],
        color=cores[::-1], edgecolor='white', linewidth=0.8
    )
    for bar, val in zip(bars, importancias['importancia_pct'][::-1]):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}%', va='center', fontsize=9
        )
    ax.set_xlabel('Importância (%)', fontsize=11)
    ax.set_title(
        'Importância das Features - Árvore de Decisão\n'
        '(Redução de Entropia Ponderada por Amostras)',
        fontsize=12, fontweight='bold'
    )
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color='#c0392b', label='Top 3 - maior relevância'),
            plt.Rectangle((0, 0), 1, 1, color='#2980b9', label='Demais features'),
        ],
        loc='lower right', fontsize=9
    )
    plt.tight_layout()
    plt.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Importância das features salva em: {caminho}")


def salvar_distribuicao_alvo(df, output_dir):
    """Salva gráfico de pizza com distribuição da variável alvo."""
    caminho = os.path.join(output_dir, 'distribuicao_alvo.png')
    contagem = df[COLUNA_ALVO].value_counts(dropna=True)
    cores = ['#e74c3c', '#e67e22', '#27ae60']
    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        contagem.values,
        labels=contagem.index,
        autopct='%1.1f%%',
        colors=cores,
        startangle=140,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight('bold')
    ax.set_title('Distribuição dos Sinistros PRF 2025\nPor Classificação de Acidente',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Distribuição do alvo salva em: {caminho}")


# ---------------------------------------------------------------------------
# PONTO DE ENTRADA PRINCIPAL
# ---------------------------------------------------------------------------

def main():
    print("\n" + "="*60)
    print("  AV1 - ÁRVORE DE DECISÃO PARA SINISTROS DE TRÂNSITO")
    print("  Disciplina: Sistemas de Apoio à Decisão")
    print("="*60)

    # Garantir que o diretório de saída existe
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Carregar dados
    caminho_csv = encontrar_arquivo(POSSIVEIS_CAMINHOS)
    df = carregar_dados(caminho_csv)

    # 2. Exploração
    explorar_dados(df)
    salvar_distribuicao_alvo(df, OUTPUT_DIR)

    # 3. Pré-processamento
    X_train, X_test, y_train, y_test, encoders, feature_names = preprocessar(df)

    # 4. Treinamento
    modelo = treinar_modelo(X_train, y_train)

    # 5. Avaliação
    y_pred, cm = avaliar_modelo(modelo, X_train, X_test, y_train, y_test)

    # 6. Importância das features
    importancias = analisar_importancia(modelo, feature_names)

    # 7. Regras da árvore (primeiros nós)
    print(f"\n{'='*60}")
    print("REGRAS DA ÁRVORE (primeiros 2 níveis - texto)")
    print(f"{'='*60}")
    regras = export_text(modelo, feature_names=feature_names, max_depth=2)
    print(regras)

    # 8. Proposta de campanhas
    propor_campanhas(importancias, modelo, encoders, feature_names)

    # 9. Visualizações
    print(f"\n{'='*60}")
    print("8. SALVANDO VISUALIZAÇÕES")
    print(f"{'='*60}")
    salvar_arvore(modelo, feature_names, OUTPUT_DIR)
    salvar_matriz_confusao(cm, OUTPUT_DIR)
    salvar_importancia(importancias, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print("ANÁLISE CONCLUÍDA")
    print(f"Imagens salvas em: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
