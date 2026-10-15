"""Problema 1 - Modelo da competicao (menor MAE nos 5 imoveis secretos).

Usa modelos prontos do scikit-learn com ajuste fino de hiperparametros
(RandomizedSearchCV com validacao cruzada) e compara com o baseline linear.

Uso (na raiz do repositorio):
    python scripts/problema1_boosting.py                       # treina, compara e salva o modelo
    python scripts/problema1_boosting.py --prever secretos.csv # preve os imoveis secretos
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import joblib
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

CAMINHO_DADOS = '4-bloco-Machine Learing/TRABALHO AV1/Problema 1/house-prices.csv'
OUTPUT_DIR = 'outputs'
CAMINHO_MODELO = os.path.join(OUTPUT_DIR, 'p1_modelo_competicao.joblib')
SEED = 42


def criar_atributos(df):
    """Engenharia de atributos aplicada igualmente ao treino e aos imoveis secretos."""
    df = df.copy()
    df['TotalSF'] = df['TotalBsmtSF'].fillna(0) + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBanheiros'] = (df['FullBath'] + 0.5 * df['HalfBath']
                            + df['BsmtFullBath'].fillna(0) + 0.5 * df['BsmtHalfBath'].fillna(0))
    df['IdadeNaVenda'] = df['YrSold'] - df['YearBuilt']
    df['AnosDesdeReforma'] = df['YrSold'] - df['YearRemodAdd']
    df['AreaVarandas'] = (df['WoodDeckSF'] + df['OpenPorchSF'] + df['EnclosedPorch']
                          + df['3SsnPorch'] + df['ScreenPorch'])
    # MSSubClass e um codigo de tipo de imovel, nao uma quantidade
    df['MSSubClass'] = df['MSSubClass'].astype(str)
    return df.drop(columns=['Id', 'SalePrice'], errors='ignore')


def colunas_por_tipo(X):
    categoricas = X.select_dtypes(exclude='number').columns.tolist()
    numericas = X.select_dtypes(include='number').columns.tolist()
    return numericas, categoricas


def modelo_boosting(X):
    numericas, categoricas = colunas_por_tipo(X)
    preprocess = ColumnTransformer([
        ('num', 'passthrough', numericas),  # o HistGB trata nulos numericos nativamente
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan,
                               encoded_missing_value=np.nan), categoricas),
    ], verbose_feature_names_out=False)
    # As categoricas saem do ColumnTransformer depois das numericas
    indices_categoricas = list(range(len(numericas), len(numericas) + len(categoricas)))
    hgb = HistGradientBoostingRegressor(categorical_features=indices_categoricas, random_state=SEED)
    # Treina no log do preco: os erros passam a ser relativos (10% de erro pesa igual em casa barata ou cara)
    return TransformedTargetRegressor(regressor=make_pipeline(preprocess, hgb),
                                      func=np.log1p, inverse_func=np.expm1)


def modelo_ridge_log(X, alpha=10.0):
    numericas, categoricas = colunas_por_tipo(X)
    preprocess = ColumnTransformer([
        ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numericas),
        ('cat', make_pipeline(SimpleImputer(strategy='constant', fill_value='None'),
                              OneHotEncoder(handle_unknown='ignore')), categoricas),
    ])
    return TransformedTargetRegressor(regressor=make_pipeline(preprocess, Ridge(alpha=alpha)),
                                      func=np.log1p, inverse_func=np.expm1)


class MediaModelos:
    """Media simples das previsoes de dois modelos ja treinados (ensemble)."""

    def __init__(self, modelos):
        self.modelos = modelos

    def fit(self, X, y):
        for m in self.modelos:
            m.fit(X, y)
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.modelos], axis=0)


def remover_outliers(X, y):
    # Dois imoveis enormes vendidos muito abaixo do valor (vendas parciais documentadas no codebook do Ames)
    mascara = ~((X['GrLivArea'] > 4000) & (y < 300000))
    return X[mascara], y[mascara]


def ajustar_boosting(X_train, y_train):
    busca = RandomizedSearchCV(
        modelo_boosting(X_train),
        param_distributions={
            'regressor__histgradientboostingregressor__learning_rate': loguniform(0.02, 0.2),
            'regressor__histgradientboostingregressor__max_iter': randint(200, 1200),
            'regressor__histgradientboostingregressor__max_leaf_nodes': randint(8, 48),
            'regressor__histgradientboostingregressor__min_samples_leaf': randint(5, 40),
            'regressor__histgradientboostingregressor__l2_regularization': loguniform(1e-3, 10),
            'regressor__histgradientboostingregressor__max_features': uniform(0.3, 0.7),
        },
        n_iter=40, cv=KFold(5, shuffle=True, random_state=SEED),
        scoring='neg_mean_absolute_error', random_state=SEED, n_jobs=-1)
    busca.fit(X_train, y_train)
    return busca


def treinar_e_comparar():
    print("=" * 70)
    print("PROBLEMA 1: MODELO DA COMPETICAO - AJUSTE FINO DE MODELOS PRONTOS")
    print("=" * 70)
    df = pd.read_csv(CAMINHO_DADOS)
    X = criar_atributos(df)
    y = df['SalePrice']

    # Mesmo split do baseline (80/20, random_state=42) para a comparacao ser justa
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)
    X_train, y_train = remover_outliers(X_train, y_train)
    print(f"Treino: {len(X_train)} imoveis (2 outliers removidos) | Teste: {len(X_test)} imoveis\n")

    print("Ajustando hiperparametros do HistGradientBoosting (40 combinacoes x 5 folds)...")
    busca = ajustar_boosting(X_train, y_train)
    print(f"  Melhor MAE na validacao cruzada: ${-busca.best_score_:,.0f}")
    melhores = {k.split('__')[-1]: (round(float(v), 4) if isinstance(v, float) else int(v))
                for k, v in busca.best_params_.items()}
    print(f"  Melhores hiperparametros: {melhores}\n")

    print("Ajustando o alpha do Ridge (log do preco, todas as colunas)...")
    cv = KFold(5, shuffle=True, random_state=SEED)
    alphas = [1, 3, 10, 30, 100]
    maes_ridge = [-cross_val_score(modelo_ridge_log(X_train, a), X_train, y_train, cv=cv,
                                   scoring='neg_mean_absolute_error').mean() for a in alphas]
    melhor_alpha = alphas[int(np.argmin(maes_ridge))]
    print(f"  Melhor alpha: {melhor_alpha} (MAE na validacao cruzada: ${min(maes_ridge):,.0f})\n")

    candidatos = {
        'HistGradientBoosting ajustado': busca.best_estimator_,
        'Ridge (log, todas as colunas)': modelo_ridge_log(X_train, melhor_alpha).fit(X_train, y_train),
    }
    candidatos['Media Ridge + HistGB'] = MediaModelos(list(candidatos.values()))

    resultados = {'Baseline Linear (relatorio)': {'MAE': 21888.03, 'R2': 0.8232},
                  'Ridge + Bairros (relatorio)': {'MAE': 19245.03, 'R2': 0.8485}}
    for nome, modelo in candidatos.items():
        pred = modelo.predict(X_test)
        resultados[nome] = {'MAE': mean_absolute_error(y_test, pred), 'R2': r2_score(y_test, pred)}

    print("=" * 70)
    print(f"{'Modelo':35s} {'MAE no teste':>15s} {'R2 no teste':>12s}")
    for nome, r in resultados.items():
        print(f"{nome:35s} {'$' + format(r['MAE'], ',.0f'):>15s} {r['R2']:12.4f}")
    print("=" * 70)

    # O vencedor e escolhido pelo teste apenas entre os 3 candidatos novos
    vencedor = min(candidatos, key=lambda n: resultados[n]['MAE'])
    print(f"\nModelo escolhido para a competicao: {vencedor}")

    # Retreina o vencedor com TODOS os dados (sem os outliers) antes de prever os secretos
    X_all, y_all = remover_outliers(X, y)
    if vencedor == 'HistGradientBoosting ajustado':
        final = modelo_boosting(X_all).set_params(**busca.best_params_)
    elif vencedor == 'Ridge (log, todas as colunas)':
        final = modelo_ridge_log(X_all, melhor_alpha)
    else:
        final = MediaModelos([modelo_boosting(X_all).set_params(**busca.best_params_),
                              modelo_ridge_log(X_all, melhor_alpha)])
    final.fit(X_all, y_all)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump({'modelo': final, 'nome': vencedor, 'tipos_colunas': X.dtypes.astype(str).to_dict()},
                CAMINHO_MODELO)
    with open(os.path.join(OUTPUT_DIR, 'p1_resultados_competicao.json'), 'w', encoding='utf-8') as f:
        json.dump({'vencedor': vencedor, 'melhor_alpha_ridge': melhor_alpha,
                   'hiperparametros_hgb': melhores, 'resultados_teste': resultados}, f, indent=2)
    print(f"Modelo final (treinado com {len(X_all)} imoveis) salvo em: {CAMINHO_MODELO}")
    return resultados


def prever_imoveis_secretos(caminho_csv):
    salvo = joblib.load(CAMINHO_MODELO)
    df = pd.read_csv(caminho_csv)
    X = criar_atributos(df)
    # Com poucas linhas, uma coluna de texto toda vazia (ex: PoolQC) e lida como numero:
    # forca cada coluna a ter o mesmo tipo que tinha no treino
    tipos = salvo['tipos_colunas']
    faltando = set(tipos) - set(X.columns)
    if faltando:
        sys.exit(f"Colunas ausentes no CSV dos imoveis secretos: {sorted(faltando)}")
    X = X[list(tipos)]
    for col, tipo in tipos.items():
        X[col] = X[col].astype(object) if tipo in ('object', 'str') else pd.to_numeric(X[col], errors='coerce')
    previsoes = salvo['modelo'].predict(X)
    saida = pd.DataFrame({'Id': df['Id'] if 'Id' in df else range(1, len(df) + 1),
                          'SalePrice_previsto': previsoes.round(2)})
    print(f"Modelo usado: {salvo['nome']}")
    print(saida.to_string(index=False))
    caminho_saida = os.path.join(OUTPUT_DIR, 'p1_previsoes_secretos.csv')
    saida.to_csv(caminho_saida, index=False)
    print(f"\nPrevisoes salvas em: {caminho_saida}")
    return saida


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prever', metavar='CSV', help='CSV com as features dos imoveis secretos')
    args = parser.parse_args()
    if args.prever:
        if not os.path.exists(CAMINHO_MODELO):
            sys.exit("Modelo ainda nao treinado. Rode primeiro sem --prever.")
        prever_imoveis_secretos(args.prever)
    else:
        treinar_e_comparar()
