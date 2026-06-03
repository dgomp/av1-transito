"""
=============================================================================
BENCHMARK: LabelEncoder vs OneHotEncoder na Árvore de Decisão
=============================================================================
Compara, sobre EXATAMENTE os mesmos dados, target e split treino/teste, duas
estratégias de codificação das 10 features categóricas nominais:

    A) LabelEncoder   (estratégia atual do projeto)
    B) OneHotEncoder  (estratégia proposta)

Tudo o mais é mantido idêntico: mesmo target binário, mesmo split estratificado
e MESMOS hiperparâmetros da árvore (entropy, max_depth=7, min_samples_leaf=50).

Para a árvore, a importância (feature_importances_) é o que dirige a proposta
de campanhas. Com one-hot a importância vira POR COLUNA (uf_SP, uf_MG…), então
ela é REAGREGADA somando-se as colunas de cada feature original — só assim os
dois rankings são comparáveis.

Uso:
    python benchmark_encoding.py
=============================================================================
"""

import os
import sys
import warnings

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)

warnings.filterwarnings('ignore')

CAMINHO_CSV = os.path.join(os.path.dirname(__file__), 'datatran2025.csv')

FEATURES = [
    'dia_semana', 'fase_dia', 'condicao_metereologica', 'tipo_pista',
    'tracado_via', 'causa_acidente', 'tipo_acidente', 'uso_solo',
    'sentido_via', 'uf',
]
COLUNA_ALVO = 'classificacao_acidente'

MAX_DEPTH = 7
MIN_SAMPLES_LEAF = 50
RANDOM_STATE = 42
TEST_SIZE = 0.20


def carregar_e_preparar():
    df = pd.read_csv(CAMINHO_CSV, sep=';', encoding='latin-1', low_memory=False)
    df = df.dropna(subset=[COLUNA_ALVO]).copy()
    df['com_vitimas'] = df[COLUNA_ALVO].apply(
        lambda x: 1 if x in ('Com Vítimas Feridas', 'Com Vítimas Fatais') else 0
    )
    X_raw = df[FEATURES].fillna('Ignorado').astype(str)
    y = df['com_vitimas']
    print(f"Registros utilizados: {len(df):,}")
    print("Cardinalidade por feature:")
    for f in FEATURES:
        print(f"  {f:<26} {X_raw[f].nunique():>3} categorias")
    return X_raw, y


def treinar_avaliar(nome, X, y, col_to_feature=None):
    """Treina a árvore e devolve métricas + importância (reagregada se one-hot)."""
    print(f"\n{'='*60}\nESTRATÉGIA: {nome}  |  colunas de entrada: {X.shape[1]}\n{'='*60}")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    modelo = DecisionTreeClassifier(
        criterion='entropy', max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF, random_state=RANDOM_STATE
    )
    modelo.fit(X_tr, y_tr)

    acc_tr = accuracy_score(y_tr, modelo.predict(X_tr))
    acc_te = accuracy_score(y_te, modelo.predict(X_te))
    y_pred = modelo.predict(X_te)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_te, y_pred, average='binary', pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_te, y_pred)

    # Importância agregada por feature original
    imp = pd.Series(modelo.feature_importances_, index=X.columns)
    if col_to_feature is not None:
        imp = imp.groupby(col_to_feature).sum()
    imp = imp.reindex(FEATURES).sort_values(ascending=False)

    print(f"Acurácia treino: {acc_tr*100:.2f}%  |  teste: {acc_te*100:.2f}%  "
          f"(gap {(acc_tr-acc_te)*100:+.2f} pp)")
    print(f"Com Vítimas → Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}")
    print(f"Árvore: {modelo.get_n_leaves()} folhas, profundidade {modelo.get_depth()}")
    print(f"Matriz de confusão [[VN, FP], [FN, VP]]: {cm.tolist()}")
    print("Importância por feature (%):")
    for f, v in imp.items():
        print(f"  {f:<26} {v*100:>6.2f}%")

    return {'acc_tr': acc_tr, 'acc_te': acc_te, 'prec': prec, 'rec': rec,
            'f1': f1, 'cm': cm, 'imp': imp}


def comparar(res_a, res_b):
    print(f"\n{'='*60}\nCOMPARAÇÃO\n{'='*60}")
    print(f"{'Métrica':<22}{'Label':>10}{'OneHot':>10}{'Δ':>10}")
    print('-' * 52)
    for k, lab in [('acc_te', 'Acurácia (teste)'), ('prec', 'Precision'),
                   ('rec', 'Recall'), ('f1', 'F1')]:
        print(f"{lab:<22}{res_a[k]:>10.4f}{res_b[k]:>10.4f}{res_b[k]-res_a[k]:>+10.4f}")

    print(f"\nRanking de importância por feature (posição):")
    rank_a = {f: i+1 for i, f in enumerate(res_a['imp'].index)}
    rank_b = {f: i+1 for i, f in enumerate(res_b['imp'].index)}
    print(f"{'Feature':<26}{'Label':>14}{'OneHot':>14}")
    print('-' * 54)
    for f in FEATURES:
        ia = res_a['imp'][f] * 100
        ib = res_b['imp'][f] * 100
        print(f"{f:<26}{f'#{rank_a[f]} ({ia:.1f}%)':>14}{f'#{rank_b[f]} ({ib:.1f}%)':>14}")

    print(f"\nTop 5 (origem das campanhas):")
    print(f"  Label  : {list(res_a['imp'].head(5).index)}")
    print(f"  OneHot : {list(res_b['imp'].head(5).index)}")


def main():
    print("=" * 60)
    print("BENCHMARK DE ENCODING — LabelEncoder vs OneHotEncoder (Árvore)")
    print("=" * 60)
    X_raw, y = carregar_e_preparar()

    # A) LabelEncoder
    X_label = X_raw.copy()
    for col in FEATURES:
        X_label[col] = LabelEncoder().fit_transform(X_raw[col])

    # B) OneHotEncoder (mapa coluna→feature original p/ reagregar a importância)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    arr = ohe.fit_transform(X_raw)
    cols = ohe.get_feature_names_out(FEATURES)
    X_onehot = pd.DataFrame(arr, columns=cols, index=X_raw.index)
    col_to_feature = {}
    for feat, cats in zip(FEATURES, ohe.categories_):
        for cat in cats:
            col_to_feature[f"{feat}_{cat}"] = feat

    res_a = treinar_avaliar('A) LabelEncoder (atual)', X_label, y)
    res_b = treinar_avaliar('B) OneHotEncoder (proposta)', X_onehot, y, col_to_feature)
    comparar(res_a, res_b)
    print()


if __name__ == '__main__':
    main()
