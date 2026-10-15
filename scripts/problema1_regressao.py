import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# Configuracoes de visualizacao
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 11

OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def executar_problema_1():
    print("=" * 70)
    print("PROBLEMA 1: REGRESSAO LINEAR - O PRECO JUSTO")
    print("=" * 70)
    
    # 1. Carregamento dos dados
    caminho_dados = '4-bloco-Machine Learing/TRABALHO AV1/Problema 1/house-prices.csv'
    df = pd.read_csv(caminho_dados)
    print(f"Dimensoes do dataset original: {df.shape[0]} linhas e {df.shape[1]} colunas.\n")
    
    # 2. Analise de Correlacao Linear
    colunas_numericas = df.select_dtypes(include=[np.number])
    correlacoes = colunas_numericas.corr()['SalePrice'].sort_values(ascending=False)
    
    print("Top 10 variaveis com maior correlacao linear positiva com SalePrice:")
    for rank, (col, val) in enumerate(correlacoes.iloc[1:11].items(), 1):
        print(f"  {rank:2d}. {col:15s}: r = {val:.4f}")
    
    top3_cols = correlacoes.iloc[1:4].index.tolist()
    print(f"\nTop 3 variaveis selecionadas: {top3_cols}\n")
    
    # Grafico 1: Heatmap das top correlacoes
    top_corr_features = correlacoes.head(10).index
    corr_mat = df[top_corr_features].corr()
    
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap='Blues', ax=ax, cbar_kws={'label': 'Coeficiente de Pearson (r)'})
    ax.set_title("Matriz de Correlacao de Pearson - Top Variaveis vs SalePrice", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    caminho_heatmap = os.path.join(OUTPUT_DIR, 'p1_correlacoes_heatmap.png')
    fig.savefig(caminho_heatmap, dpi=300)
    plt.close(fig)
    print(f"Grafico salvo: {caminho_heatmap}")
    
    # Grafico 2: Scatter plots das Top 3 features vs SalePrice
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    nomes_descritivos = {
        'OverallQual': 'Qualidade Geral dos Materiais (1 a 10)',
        'GrLivArea': 'Area Construida Acima do Solo (sq ft)',
        'GarageCars': 'Capacidade de Veiculos na Garagem'
    }
    
    for i, col in enumerate(top3_cols):
        sns.regplot(data=df, x=col, y='SalePrice', ax=axes[i],
                    scatter_kws={'alpha': 0.4, 'color': '#1f77b4'},
                    line_kws={'color': '#d62728', 'linewidth': 2})
        axes[i].set_title(f"{col} (r = {correlacoes[col]:.3f})", fontsize=13, fontweight='bold')
        axes[i].set_xlabel(nomes_descritivos.get(col, col), fontsize=11)
        axes[i].set_ylabel("Preco de Venda ($)" if i == 0 else "", fontsize=11)
        axes[i].yaxis.set_major_formatter('${x:,.0f}')
    
    plt.suptitle("Dispersao e Tendencia Linear: Top 3 Preditores vs Preco de Venda", fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    caminho_scatter = os.path.join(OUTPUT_DIR, 'p1_scatter_top3.png')
    fig.savefig(caminho_scatter, dpi=300)
    plt.close(fig)
    print(f"Grafico salvo: {caminho_scatter}\n")
    
    # 3. Tratamento de Valores Ausentes
    print("-" * 50)
    print("Tratamento de Dados Nulos:")
    # Numerica: LotFrontage
    null_lot = df['LotFrontage'].isnull().sum()
    print(f"  Coluna Numerica 'LotFrontage': {null_lot} nulos ({null_lot/len(df)*100:.2f}%).")
    # Imputacao pela mediana agrupada por Neighborhood
    mediana_neighborhood = df.groupby('Neighborhood')['LotFrontage'].transform('median')
    df['LotFrontage'] = df['LotFrontage'].fillna(mediana_neighborhood)
    # Se restar algum nulo por bairro sem referencia, preencher com mediana global
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    print("  -> 'LotFrontage' imputada pela mediana condicional ao bairro (Neighborhood).")
    
    # Numerica secundaria: MasVnrArea
    if 'MasVnrArea' in df.columns:
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0.0)
        print("  -> 'MasVnrArea' imputada com 0.0 (ausencia de alvenaria).")
    
    # Categorica: BsmtQual
    null_bsmt = df['BsmtQual'].isnull().sum()
    print(f"  Coluna Categorica 'BsmtQual': {null_bsmt} nulos ({null_bsmt/len(df)*100:.2f}%).")
    df['BsmtQual'] = df['BsmtQual'].fillna('None')
    print("  -> 'BsmtQual' imputada com a categoria estrutural 'None' (imovel sem porao).\n")
    
    # 4. Engenharia de Atributos e Codificacao Categorica
    print("-" * 50)
    print("Engenharia de Atributos (Conversao de Categoricas para Numericas):")
    # Mapeamento ordinal para BsmtQual (Ex=5, Gd=4, TA=3, Fa=2, Po=1, None=0)
    mapa_qualidade = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    df['BsmtQual_Num'] = df['BsmtQual'].map(mapa_qualidade).fillna(0)
    print("  1. 'BsmtQual' codificada via Mapeamento Ordinal (Ex=5 a None=0).")
    
    # Selecionar outra categorica de impacto: Neighborhood (One-Hot Encoding)
    print("  2. 'Neighborhood' codificada via One-Hot Encoding (pd.get_dummies com drop_first=True).")
    df_encoded = pd.get_dummies(df, columns=['Neighborhood'], drop_first=True, dtype=float)
    
    # 5. Definicao do Modelo Baseline e Treinamento
    # Features do baseline basico: Top variaveis numericas + LotFrontage + BsmtQual_Num
    features_baseline = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 
                         'TotalBsmtSF', '1stFlrSF', 'FullBath', 'YearBuilt', 
                         'LotFrontage', 'BsmtQual_Num']
    
    # Tratar nulos residuais nas features numericas selecionadas se houver
    for col in features_baseline:
        if df_encoded[col].isnull().sum() > 0:
            df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())
            
    X_base = df_encoded[features_baseline]
    y = df_encoded['SalePrice']
    
    X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=0.20, random_state=42)
    print(f"\nDivisao Treino/Teste (80/20):")
    print(f"  Amostras de treino: {X_train.shape[0]}")
    print(f"  Amostras de teste:  {X_test.shape[0]}\n")
    
    modelo_base = LinearRegression()
    modelo_base.fit(X_train, y_train)
    
    y_pred_train = modelo_base.predict(X_train)
    y_pred_test = modelo_base.predict(X_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = root_mean_squared_error(y_test, y_pred_test)
    
    print("=" * 50)
    print("RESULTADOS DO MODELO BASELINE (REGRESSAO LINEAR):")
    print(f"  R2 no Treino: {r2_train:.4f} ({r2_train*100:.2f}%)")
    print(f"  R2 no Teste:  {r2_test:.4f} ({r2_test*100:.2f}%)")
    print(f"  MAE no Teste: ${mae_test:,.2f}")
    print(f"  RMSE no Teste: ${rmse_test:,.2f}")
    print("=" * 50)
    
    # Coeficientes do modelo baseline
    coefs = pd.DataFrame({'Atributo': features_baseline, 'Coeficiente': modelo_base.coef_})
    coefs = coefs.sort_values(by='Coeficiente', ascending=False)
    print("\nPesos (Beta) do Modelo Baseline:")
    print(f"  Intercepto (Beta_0): ${modelo_base.intercept_:,.2f}")
    for _, row in coefs.iterrows():
        print(f"  {row['Atributo']:15s}: ${row['Coeficiente']:10,.2f}")
    
    # Modelo Otimizado (incorporando One-Hot de Neighborhood e regularizacao Ridge)
    cols_neighborhood = [c for c in df_encoded.columns if c.startswith('Neighborhood_')]
    features_otimizadas = features_baseline + cols_neighborhood + ['LotArea', 'YearRemodAdd', 'Fireplaces']
    for c in features_otimizadas:
        if df_encoded[c].isnull().sum() > 0:
            df_encoded[c] = df_encoded[c].fillna(df_encoded[c].median())
            
    X_opt = df_encoded[features_otimizadas]
    X_tr_opt, X_te_opt, y_tr_opt, y_te_opt = train_test_split(X_opt, y, test_size=0.20, random_state=42)
    
    modelo_opt = Ridge(alpha=10.0)
    modelo_opt.fit(X_tr_opt, y_tr_opt)
    y_pred_opt = modelo_opt.predict(X_te_opt)
    
    r2_opt = r2_score(y_te_opt, y_pred_opt)
    mae_opt = mean_absolute_error(y_te_opt, y_pred_opt)
    rmse_opt = root_mean_squared_error(y_te_opt, y_pred_opt)
    
    print("\nRESULTADOS DO MODELO REFINADO (RIDGE REGRESSION COM BAIRROS):")
    print(f"  R2 no Teste:  {r2_opt:.4f} ({r2_opt*100:.2f}%)")
    print(f"  MAE no Teste: ${mae_opt:,.2f} (reducao de ${mae_test - mae_opt:,.2f} no erro!)")
    print(f"  RMSE no Teste: ${rmse_opt:,.2f}")
    
    # Grafico 3: Avaliacao de Residuos e Real vs Predito
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Real vs Predito
    axes[0].scatter(y_test, y_pred_test, alpha=0.6, color='#1f77b4', edgecolors='k', s=45)
    limites = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
    axes[0].plot(limites, limites, color='red', linestyle='--', linewidth=2, label='Predicao Perfeita (y = x)')
    axes[0].set_title(f"Valores Reais vs Preditos (Baseline)\nR² = {r2_test:.4f} | MAE = ${mae_test:,.0f}", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Preco de Venda Real ($)", fontsize=11)
    axes[0].set_ylabel("Preco de Venda Previsto ($)", fontsize=11)
    axes[0].xaxis.set_major_formatter('${x:,.0f}')
    axes[0].yaxis.set_major_formatter('${x:,.0f}')
    axes[0].legend(frameon=True)
    
    # Residuos vs Predito
    residuos = y_test - y_pred_test
    axes[1].scatter(y_pred_test, residuos, alpha=0.6, color='#2ca02c', edgecolors='k', s=45)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_title("Analise de Residuos (Erro de Estimativa)\nResiduo = Real - Previsto", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Preco de Venda Previsto ($)", fontsize=11)
    axes[1].set_ylabel("Residuo ($)", fontsize=11)
    axes[1].xaxis.set_major_formatter('${x:,.0f}')
    axes[1].yaxis.set_major_formatter('${x:,.0f}')
    
    plt.tight_layout()
    caminho_residuos = os.path.join(OUTPUT_DIR, 'p1_residuos_dispersao.png')
    fig.savefig(caminho_residuos, dpi=300)
    plt.close(fig)
    print(f"Grafico salvo: {caminho_residuos}\n")
    
    return {
        'top3': top3_cols,
        'r2_baseline': r2_test,
        'mae_baseline': mae_test,
        'r2_otimizado': r2_opt,
        'mae_otimizado': mae_opt
    }

if __name__ == '__main__':
    executar_problema_1()
