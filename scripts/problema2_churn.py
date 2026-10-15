import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_auc_score, roc_curve, precision_recall_curve,
                             accuracy_score, precision_score, recall_score, f1_score)

# Configuracoes de visualizacao
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 11

OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def executar_problema_2():
    print("=" * 70)
    print("PROBLEMA 2: CLASSIFICACAO - A AMEACA DO CHURN")
    print("=" * 70)
    
    caminho_dados = '4-bloco-Machine Learing/TRABALHO AV1/Problema 2/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    df = pd.read_csv(caminho_dados)
    print(f"Dimensoes do dataset original: {df.shape[0]} clientes e {df.shape[1]} colunas.\n")
    
    # 1. Compreensao do Negocio e Analise da Variavel Alvo
    target_col = 'Churn'
    contagem_alvo = df[target_col].value_counts()
    prop_alvo = df[target_col].value_counts(normalize=True) * 100
    
    print("Distribuicao da Variavel Alvo (Churn):")
    print(f"  Clientes que Ficaram (No) : {contagem_alvo['No']:5d} ({prop_alvo['No']:.2f}%)")
    print(f"  Clientes que Cancelaram (Yes): {contagem_alvo['Yes']:5d} ({prop_alvo['Yes']:.2f}%)")
    print(f"  Razao de desbalanceamento: ~{contagem_alvo['No'] / contagem_alvo['Yes']:.2f} clientes 'No' para cada 1 'Yes'.\n")
    
    # Grafico 1: Distribuicao de Churn
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cores = ['#2b5c8f', '#d95f02']
    
    sns.barplot(x=contagem_alvo.index, y=contagem_alvo.values, palette=cores, ax=axes[0], hue=contagem_alvo.index, legend=False)
    axes[0].set_title("Distribuicao Absoluta da Base (Contagem de Clientes)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Cancelou o Servico (Churn)", fontsize=11)
    axes[0].set_ylabel("Quantidade de Clientes", fontsize=11)
    for p in axes[0].patches:
        axes[0].annotate(f"{int(p.get_height())}\n({p.get_height()/len(df)*100:.1f}%)", 
                         (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', color='white', fontweight='bold', fontsize=12)
        
    axes[1].pie(contagem_alvo.values, labels=['Retidos (No)', 'Cancelados (Yes)'], autopct='%1.1f%%',
                colors=cores, explode=(0, 0.08), startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[1].set_title("Proporcao Relativa de Evasao", fontsize=13, fontweight='bold')
    
    plt.suptitle("Analise de Desbalanceamento da Variavel Alvo (Churn)", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    caminho_dist_churn = os.path.join(OUTPUT_DIR, 'p2_distribuicao_churn.png')
    fig.savefig(caminho_dist_churn, dpi=300)
    plt.close(fig)
    print(f"Grafico salvo: {caminho_dist_churn}")
    
    # 2. Analise de Risco por Tipo de Contrato
    crosstab_pct = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
    crosstab_abs = pd.crosstab(df['Contract'], df['Churn'])
    
    print("-" * 50)
    print("Taxa de Churn por Tipo de Contrato:")
    for contrato in ['Month-to-month', 'One year', 'Two year']:
        taxa = crosstab_pct.loc[contrato, 'Yes']
        total_tipo = crosstab_abs.loc[contrato].sum()
        evadidos = crosstab_abs.loc[contrato, 'Yes']
        print(f"  {contrato:15s}: {taxa:5.2f}% de evasao ({evadidos}/{total_tipo} clientes)")
        
    razao_risco = crosstab_pct.loc['Month-to-month', 'Yes'] / crosstab_pct.loc['Two year', 'Yes']
    print(f"\nFator de Risco: Clientes 'Month-to-month' evadem {razao_risco:.1f}x mais que clientes 'Two year'.\n")
    
    # Grafico 2: Taxa de Churn por Contrato
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Barras empilhadas 100%
    crosstab_pct.plot(kind='bar', stacked=True, color=['#2b5c8f', '#d95f02'], ax=axes[0], width=0.6)
    axes[0].set_title("Proporcao de Retencao vs Churn por Tipo de Contrato", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Tipo de Contrato", fontsize=11)
    axes[0].set_ylabel("Percentual (%)", fontsize=11)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    axes[0].legend(['Nao Cancelou (No)', 'Cancelou (Yes)'], title="Status", loc='upper right')
    for c in axes[0].containers:
        axes[0].bar_label(c, fmt='%.1f%%', label_type='center', color='white', fontweight='bold', fontsize=11)
        
    # Taxa de evasao comparativa
    taxa_churn_serie = crosstab_pct['Yes']
    sns.barplot(x=taxa_churn_serie.index, y=taxa_churn_serie.values, palette='Reds_r', ax=axes[1], hue=taxa_churn_serie.index, legend=False)
    axes[1].set_title("Taxa de Evasao Direta (%) por Tipo de Contrato", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Tipo de Contrato", fontsize=11)
    axes[1].set_ylabel("Taxa de Churn (%)", fontsize=11)
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 1),
                         ha='center', va='bottom', fontweight='bold', fontsize=11)
    axes[1].set_ylim(0, 50)
    
    plt.suptitle("Analise de Risco Contratual: A Fragilidade do Modelo Mensal", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    caminho_risco_contrato = os.path.join(OUTPUT_DIR, 'p2_churn_por_contrato.png')
    fig.savefig(caminho_risco_contrato, dpi=300)
    plt.close(fig)
    print(f"Grafico salvo: {caminho_risco_contrato}")
    
    # 3. Preparacao dos Dados e Tratamento de TotalCharges
    print("-" * 50)
    print("Tratamento de Dados e TotalCharges:")
    linhas_em_branco = (df['TotalCharges'].str.strip() == '').sum()
    print(f"  Linhas com espacos em branco em TotalCharges: {linhas_em_branco}")
    
    # Inspecao das linhas em branco
    amostras_branco = df[df['TotalCharges'].str.strip() == '']
    print(f"  Tempo de permanencia (tenure) dessas {linhas_em_branco} linhas: todas possuem tenure = {amostras_branco['tenure'].unique().tolist()}")
    print("  Justificativa: Sao novos clientes no primeiro ciclo de cobranca (mes 0).")
    
    # Coercao numerica e imputacao com 0.0
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    print("  -> TotalCharges convertida com sucesso para float64 com 0.0 imputado para novos clientes.\n")
    
    # Pre-processamento para modelagem
    # Descartar customerID
    df_modelo = df.drop(columns=['customerID']).copy()
    
    # Variavel alvo binarizada
    df_modelo['Churn'] = df_modelo['Churn'].map({'Yes': 1, 'No': 0})
    
    # Codificacao categorica One-Hot
    cols_categoricas = df_modelo.select_dtypes(include=['object', 'string']).columns.tolist()
    print(f"Colunas categoricas convertidas via One-Hot Encoding: {cols_categoricas}")
    df_encoded = pd.get_dummies(df_modelo, columns=cols_categoricas, drop_first=True, dtype=float)
    
    # 4. Divisao Estratificada Treino / Teste (80/20)
    X = df_encoded.drop(columns=['Churn'])
    y = df_encoded['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nDivisao Treino/Teste Estratificada (80/20):")
    print(f"  Treino: {X_train.shape[0]} amostras (Churn rate: {y_train.mean()*100:.2f}%)")
    print(f"  Teste:  {X_test.shape[0]} amostras (Churn rate: {y_test.mean()*100:.2f}%)")
    
    # Padronizacao das variaveis continuas
    cols_continuas = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[cols_continuas] = scaler.fit_transform(X_train[cols_continuas])
    X_test_scaled[cols_continuas] = scaler.transform(X_test[cols_continuas])
    
    # 5. Treinamento dos Modelos: Regressao Logistica e Random Forest
    # Modelo 1: Regressao Logistica (Interpretabilidade e calibracao probabilistica)
    modelo_lr = LogisticRegression(max_iter=1000, random_state=42)
    modelo_lr.fit(X_train_scaled, y_train)
    
    y_pred_lr = modelo_lr.predict(X_test_scaled)
    y_prob_lr = modelo_lr.predict_proba(X_test_scaled)[:, 1]
    
    # Modelo 2: Random Forest
    modelo_rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    modelo_rf.fit(X_train, y_train)
    y_pred_rf = modelo_rf.predict(X_test)
    y_prob_rf = modelo_rf.predict_proba(X_test)[:, 1]
    
    # Metricas Modelo 1 (Regressao Logistica)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    prec_lr = precision_score(y_test, y_pred_lr)
    rec_lr = recall_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr)
    auc_lr = roc_auc_score(y_test, y_prob_lr)
    
    # Metricas Modelo 2 (Random Forest)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    prec_rf = precision_score(y_test, y_pred_rf)
    rec_rf = recall_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf)
    auc_rf = roc_auc_score(y_test, y_prob_rf)
    
    print("\n" + "=" * 50)
    print("DESEMPENHO DO MODELO 1: REGRESSAO LOGISTICA")
    print(f"  Acuracia : {acc_lr:.4f} ({acc_lr*100:.2f}%)")
    print(f"  Precisao : {prec_lr:.4f} ({prec_lr*100:.2f}%)")
    print(f"  Recall   : {rec_lr:.4f} ({rec_lr*100:.2f}%)")
    print(f"  F1-Score : {f1_lr:.4f}")
    print(f"  ROC-AUC  : {auc_lr:.4f}")
    
    print("\nDESEMPENHO DO MODELO 2: RANDOM FOREST")
    print(f"  Acuracia : {acc_rf:.4f} ({acc_rf*100:.2f}%)")
    print(f"  Precisao : {prec_rf:.4f} ({prec_rf*100:.2f}%)")
    print(f"  Recall   : {rec_rf:.4f} ({rec_rf*100:.2f}%)")
    print(f"  F1-Score : {f1_rf:.4f}")
    print(f"  ROC-AUC  : {auc_rf:.4f}")
    print("=" * 50)
    
    # 6. Matriz de Confusao Detalhada e Analise de Custos
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    tn, fp, fn, tp = cm_lr.ravel()
    
    print("\nDetalhamento da Matriz de Confusao (Regressao Logistica):")
    print(f"  Verdadeiros Negativos (TN): {tn:4d} (Clientes fieis previstos corretamente como fieis)")
    print(f"  Falsos Positivos      (FP): {fp:4d} (Clientes fieis erroneamente classificados como risco de Churn)")
    print(f"  Falsos Negativos      (FN): {fn:4d} (Clientes em evasao classificados erroneamente como seguros)")
    print(f"  Verdadeiros Positivos (TP): {tp:4d} (Clientes em evasao identificados a tempo)")
    
    # Grafico 3: Matrizes de Confusao
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Matriz LR
    rotulos = np.array([[f"TN: {tn}\n({tn/len(y_test)*100:.1f}%)", f"FP: {fp}\n({fp/len(y_test)*100:.1f}%)"],
                        [f"FN: {fn}\n({fn/len(y_test)*100:.1f}%)", f"TP: {tp}\n({tp/len(y_test)*100:.1f}%)"]])
    sns.heatmap(cm_lr, annot=rotulos, fmt='', cmap='Blues', ax=axes[0], cbar=False,
                xticklabels=['Previsto: Fica (0)', 'Previsto: Churn (1)'],
                yticklabels=['Real: Fica (0)', 'Real: Churn (1)'])
    axes[0].set_title(f"Matriz de Confusao - Regressao Logistica\nRecall Churn: {rec_lr*100:.1f}% | F1: {f1_lr:.3f}", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Valor Real", fontsize=11)
    axes[0].set_xlabel("Valor Predito", fontsize=11)
    
    # Matriz RF
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()
    rotulos_rf = np.array([[f"TN: {tn_rf}\n({tn_rf/len(y_test)*100:.1f}%)", f"FP: {fp_rf}\n({fp_rf/len(y_test)*100:.1f}%)"],
                           [f"FN: {fn_rf}\n({fn_rf/len(y_test)*100:.1f}%)", f"TP: {tp_rf}\n({tp_rf/len(y_test)*100:.1f}%)"]])
    sns.heatmap(cm_rf, annot=rotulos_rf, fmt='', cmap='Greens', ax=axes[1], cbar=False,
                xticklabels=['Previsto: Fica (0)', 'Previsto: Churn (1)'],
                yticklabels=['Real: Fica (0)', 'Real: Churn (1)'])
    axes[1].set_title(f"Matriz de Confusao - Random Forest\nRecall Churn: {rec_rf*100:.1f}% | F1: {f1_rf:.3f}", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Valor Real", fontsize=11)
    axes[1].set_xlabel("Valor Predito", fontsize=11)
    
    plt.suptitle("Matrizes de Confusao e Diagnostico de Custos Operacionais", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    caminho_cm = os.path.join(OUTPUT_DIR, 'p2_matriz_confusao.png')
    fig.savefig(caminho_cm, dpi=300)
    plt.close(fig)
    print(f"Grafico salvo: {caminho_cm}")
    
    # Grafico 4: Curvas ROC comparativas
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr_lr, tpr_lr, color='#1f77b4', lw=2.5, label=f'Regressao Logistica (AUC = {auc_lr:.3f})')
    ax.plot(fpr_rf, tpr_rf, color='#2ca02c', lw=2.5, label=f'Random Forest (AUC = {auc_rf:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Classificador Aleatorio (AUC = 0.500)')
    ax.set_title("Curva ROC (Receiver Operating Characteristic) - Previsao de Churn", fontsize=13, fontweight='bold')
    ax.set_xlabel("Taxa de Falsos Positivos (1 - Especificidade)", fontsize=11)
    ax.set_ylabel("Taxa de Verdadeiros Positivos (Sensibilidade / Recall)", fontsize=11)
    ax.legend(loc='lower right', frameon=True, fontsize=11)
    plt.tight_layout()
    caminho_roc = os.path.join(OUTPUT_DIR, 'p2_curva_roc.png')
    fig.savefig(caminho_roc, dpi=300)
    plt.close(fig)
    print(f"Grafico salvo: {caminho_roc}\n")
    
    return {
        'distribuicao': prop_alvo.to_dict(),
        'crosstab_pct': crosstab_pct.to_dict(),
        'lr_metrics': {'acc': acc_lr, 'prec': prec_lr, 'rec': rec_lr, 'f1': f1_lr, 'auc': auc_lr},
        'rf_metrics': {'acc': acc_rf, 'prec': prec_rf, 'rec': rec_rf, 'f1': f1_rf, 'auc': auc_rf},
        'cm_lr': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
    }

if __name__ == '__main__':
    executar_problema_2()
