import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Configuracoes de visualizacao
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 11

OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def executar_problema_3():
    print("=" * 70)
    print("PROBLEMA 3: CLUSTERIZACAO - O PERFIL DO CONSUMIDOR (MALL CUSTOMERS)")
    print("=" * 70)
    
    caminho_dados = '4-bloco-Machine Learing/TRABALHO AV1/Problema 3/Mall_Customers.csv'
    df = pd.read_csv(caminho_dados)
    print(f"Dimensoes do dataset original: {df.shape[0]} clientes e {df.shape[1]} colunas.\n")
    print("Primeiras linhas:")
    print(df.head(3))
    
    # Selecao das variaveis especificadas no enunciado
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features].copy()
    
    print("\nEstatisticas descritivas das variaveis selecionadas (em escala original):")
    print(X.describe().T[['mean', 'std', 'min', 'max']])
    
    # 1. Escalonamento: StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\nJustificativa Matematica do Escalonamento:")
    print("O algoritmo K-Means calcula a distancia euclidiana d(p,q) = sqrt(sum (p_i - q_i)^2).")
    print("Variaveis com amplitudes desiguais causam uma distorcao geometrica desproporcional,")
    print("fazendo com que a variavel de maior variancia domine a convergencia dos centroides.")
    print("Com o StandardScaler (Z-Score), ambas passam a ter media 0 e desvio padrao 1.\n")
    
    # 2. O Dilema da Quantidade: Metodo do Cotovelo e Silhouette Score (K de 1 a 10)
    wcss = []
    silhouette_scores = []
    k_range = range(1, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
        if k >= 2:
            score = silhouette_score(X_scaled, kmeans.labels_)
            silhouette_scores.append(score)
            
    print("Valores de Inercia (WCSS) e Silhouette Score por K:")
    for k in range(1, 11):
        if k == 1:
            print(f"  K = {k:2d}: Inercia = {wcss[k-1]:10.2f} | Silhouette = N/A")
        else:
            print(f"  K = {k:2d}: Inercia = {wcss[k-1]:10.2f} | Silhouette = {silhouette_scores[k-2]:.4f}")
            
    # Grafico 1: Cotovelo e Silhouette Score
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Elbow Curve
    axes[0].plot(list(k_range), wcss, marker='o', linewidth=2.5, color='#1f77b4', markersize=8)
    axes[0].axvline(5, color='red', linestyle='--', linewidth=1.5, label='K Ideal = 5 (Inflexao)')
    axes[0].scatter(5, wcss[4], color='red', s=120, zorder=5)
    axes[0].set_title("Metodo do Cotovelo (Elbow Method) - WCSS / Inercia", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Numero de Clusters (K)", fontsize=11)
    axes[0].set_ylabel("Soma dos Quadrados Intra-Cluster (WCSS)", fontsize=11)
    axes[0].set_xticks(list(k_range))
    axes[0].legend(frameon=True)
    
    # Silhouette Curve
    axes[1].plot(range(2, 11), silhouette_scores, marker='s', linewidth=2.5, color='#2ca02c', markersize=8)
    axes[1].axvline(5, color='red', linestyle='--', linewidth=1.5, label='K = 5 (Pico = 0.555)')
    axes[1].scatter(5, silhouette_scores[3], color='red', s=120, zorder=5)
    axes[1].set_title("Coeficiente de Silhueta (Silhouette Score) por K", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Numero de Clusters (K)", fontsize=11)
    axes[1].set_ylabel("Silhouette Score Medio", fontsize=11)
    axes[1].set_xticks(range(2, 11))
    axes[1].legend(frameon=True)
    
    plt.suptitle("Determinacao do Numero Otimo de Grupos (K = 5)", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    caminho_elbow = os.path.join(OUTPUT_DIR, 'p3_elbow_silhouette.png')
    fig.savefig(caminho_elbow, dpi=300)
    plt.close(fig)
    print(f"\nGrafico salvo: {caminho_elbow}")
    
    # 3. Treinamento com K = 5
    K_ESCOLHIDO = 5
    kmeans_final = KMeans(n_clusters=K_ESCOLHIDO, init='k-means++', random_state=42, n_init=15)
    df['Cluster'] = kmeans_final.fit_predict(X_scaled)
    
    # Centroides em escala original
    centroides_scaled = kmeans_final.cluster_centers_
    centroides_originais = scaler.inverse_transform(centroides_scaled)
    
    print("\n" + "=" * 50)
    print(f"CENTROIDES EM ESCALA ORIGINAL (K = {K_ESCOLHIDO}):")
    df_centroides = pd.DataFrame(centroides_originais, columns=['Renda_Media_k$', 'Score_Gasto_Medio'])
    df_centroides['Qtd_Clientes'] = df['Cluster'].value_counts().sort_index()
    print(df_centroides)
    print("=" * 50)
    
    # 4. Mapeamento dos Rotulos Comerciais de Negocio
    # Vamos ordenar e associar os clusters com base na renda e no score
    rotulos_comerciais = {}
    for idx, row in df_centroides.iterrows():
        renda = row['Renda_Media_k$']
        score = row['Score_Gasto_Medio']
        if renda < 40 and score < 40:
            rotulo = "1. Consumidores Economicos (Baixa Renda, Baixo Gasto)"
        elif renda < 40 and score >= 60:
            rotulo = "2. Jovens Impulsivos / Entusiastas (Baixa Renda, Alto Gasto)"
        elif 40 <= renda <= 70:
            rotulo = "3. Clientes Padrao / Moderados (Media Renda, Medio Gasto)"
        elif renda > 70 and score < 40:
            rotulo = "4. Poupadores Cautelosos (Alta Renda, Baixo Gasto)"
        else:
            rotulo = "5. Clientes VIP / Alto Padrao (Alta Renda, Alto Gasto)"
        rotulos_comerciais[idx] = rotulo
        
    df['Perfil_Comercial'] = df['Cluster'].map(rotulos_comerciais)
    
    print("\nInterpretacao Comercial dos Grupos:")
    for cluster_id in sorted(rotulos_comerciais.keys()):
        c_nome = rotulos_comerciais[cluster_id]
        qtd = (df['Cluster'] == cluster_id).sum()
        r_med = df[df['Cluster'] == cluster_id]['Annual Income (k$)'].mean()
        s_med = df[df['Cluster'] == cluster_id]['Spending Score (1-100)'].mean()
        print(f"  Cluster {cluster_id}: {c_nome}")
        print(f"    Total: {qtd} clientes ({qtd/len(df)*100:.1f}%) | Renda Media: ${r_med:.1f}k | Score Medio: {s_med:.1f}")
        
    # Grafico 2: Dispersao 2D com Clusters e Centroides
    fig, ax = plt.subplots(figsize=(11, 7))
    paleta = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    
    for cluster_id in range(K_ESCOLHIDO):
        sub_df = df[df['Cluster'] == cluster_id]
        ax.scatter(sub_df['Annual Income (k$)'], sub_df['Spending Score (1-100)'],
                   s=75, alpha=0.85, label=f"Cluster {cluster_id}: {rotulos_comerciais[cluster_id]}",
                   color=paleta[cluster_id], edgecolors='white', linewidth=0.5)
                   
    # Plotar os centroides
    ax.scatter(centroides_originais[:, 0], centroides_originais[:, 1],
               s=260, c='black', marker='X', edgecolors='gold', linewidth=2.5,
               label='Centroides dos Clusters', zorder=10)
               
    ax.set_title("Segmentacao de Clientes do Shopping Center (K-Means, K = 5)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Renda Anual (Annual Income em Milhares de Dolares - k$)", fontsize=12)
    ax.set_ylabel("Pontuacao de Gastos (Spending Score - 1 a 100)", fontsize=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=10)
    plt.tight_layout()
    caminho_clusters = os.path.join(OUTPUT_DIR, 'p3_clusters_mall_customers.png')
    fig.savefig(caminho_clusters, dpi=300)
    plt.close(fig)
    print(f"\nGrafico salvo: {caminho_clusters}\n")
    
    return {
        'k_escolhido': K_ESCOLHIDO,
        'silhouette_5': silhouette_scores[3],
        'rotulos': rotulos_comerciais,
        'centroides': df_centroides.to_dict(orient='records')
    }

if __name__ == '__main__':
    executar_problema_3()
