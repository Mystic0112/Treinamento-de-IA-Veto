import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configuracoes de visualizacao
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 11

OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def executar_problema_4():
    print("=" * 70)
    print("PROBLEMA 4: CLUSTERIZACAO MUSICAL - CRIANDO A PLAYLIST PERFEITA")
    print("=" * 70)
    
    caminho_dados = '4-bloco-Machine Learing/TRABALHO AV1/Problema 4/dataset.csv'
    df = pd.read_csv(caminho_dados)
    print(f"Dimensoes do dataset original: {df.shape[0]} faixas musicais e {df.shape[1]} colunas.\n")
    
    # 1. Limpeza e Isolamento de Metadados Textuais
    print("-" * 50)
    print("1. Limpeza e Filtragem de Metadados Textuais:")
    print("Colunas de texto identificadas: ['track_id', 'artists', 'album_name', 'track_name', 'track_genre']")
    print("Justificativa Tecnica:")
    print("Textos nominais nao possuem topologia metrica nem ordem no corpo dos reais (R^n).")
    print("O calculo da distancia euclidiana em algoritmos de agrupamento exige coordenadas continuas.")
    print("Ademais, o objetivo e agrupar puramente por propriedades acusticas, sem contaminacao pelo genero.")
    print("Acao: Isolamos os metadados em dataframe separado para consulta posterior (lookup).\n")
    
    # Features acusticas continuas selecionadas
    features_acusticas = [
        'danceability', 'energy', 'loudness', 'speechiness', 
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
    ]
    
    df_clean = df.dropna(subset=features_acusticas + ['track_name']).copy()
    # A mesma faixa aparece uma vez para cada genero em que foi cadastrada (mesmo track_id,
    # mesmas features): sem remover, ela pesaria em dobro no agrupamento
    df_clean = df_clean.drop_duplicates(subset='track_id')
    print(f"Faixas unicas apos remover nulos e duplicatas de track_id: {len(df_clean)} (de {len(df)} linhas)")
    print("Colunas 'key' (tom) e 'mode' ficaram de fora: sao codigos de categoria (0 = Do, 1 = Do#...),")
    print("a distancia entre os numeros nao tem significado musical.")

    # 2. Analise de Correlacao: Energy vs Acousticness
    print("-" * 50)
    print("2. Analise da Correlacao entre Energy e Acousticness:")
    r_energy_acoustic = df_clean['energy'].corr(df_clean['acousticness'])
    print(f"  Coeficiente de Correlacao Linear de Pearson: r = {r_energy_acoustic:.4f}")
    print("  Interpretacao Fisica/Musical:")
    print(f"  Existe uma correlacao fortemente negativa (r = {r_energy_acoustic:.4f}).")
    print("  No mundo real da producao musical, faixas com alta energia apresentam ondas sintetizadas,")
    print("  saturacao de ganho, compressao dinamica pesada e amplificacao eletrica (ex: Rock, EDM).")
    print("  Em contrapartida, faixas acusticas utilizam fontes mecanicas puras (voz desplugada, violao, piano),")
    print("  que preservam dinamica natural e decaimento organico, resultando em menor potencia sonora agregada.\n")
    
    # Grafico 1: Correlacao Energy vs Acousticness
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Hexbin / Scatter com linha de regressao (amostra de 5000 para clareza visual)
    amostra_viz = df_clean.sample(n=5000, random_state=42)
    sns.regplot(data=amostra_viz, x='energy', y='acousticness', ax=axes[0],
                scatter_kws={'alpha': 0.25, 'color': '#1f77b4', 's': 15},
                line_kws={'color': '#d62728', 'linewidth': 2.5})
    axes[0].set_title(f"Dispersao: Energy vs Acousticness\nPearson r = {r_energy_acoustic:.4f} (Forte Negativa)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Energy (0 a 1)", fontsize=11)
    axes[0].set_ylabel("Acousticness (0 a 1)", fontsize=11)
    
    # Heatmap completo das features acusticas
    corr_acusticas = df_clean[features_acusticas].corr()
    sns.heatmap(corr_acusticas, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1], cbar_kws={'label': 'Pearson r'})
    axes[1].set_title("Matriz de Correlacao entre Propriedades Acusticas", fontsize=12, fontweight='bold')
    
    plt.suptitle("Analise de Propriedades Acusticas do Spotify Dataset", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    caminho_corr = os.path.join(OUTPUT_DIR, 'p4_correlacao_energy_acousticness.png')
    fig.savefig(caminho_corr, dpi=300)
    plt.close(fig)
    print(f"Grafico salvo: {caminho_corr}")
    
    # 3. Normalizacao e Treinamento do K-Means (K = 5)
    print("-" * 50)
    print("3. Normalizacao e Treinamento do Modelo (K-Means, K = 5):")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[features_acusticas])
    
    K_PLAYLISTS = 5
    kmeans_musica = KMeans(n_clusters=K_PLAYLISTS, init='k-means++', random_state=42, n_init=10)
    df_clean['cluster'] = kmeans_musica.fit_predict(X_scaled)
    
    # Centroides originais
    centroides_originais = scaler.inverse_transform(kmeans_musica.cluster_centers_)
    df_centroides = pd.DataFrame(centroides_originais, columns=features_acusticas)
    df_centroides['Qtd_Musicas'] = df_clean['cluster'].value_counts().sort_index()
    
    print("\nPerfil Acustico Medio dos 5 Clusters / Playlists (Centroides em Escala Original):")
    print(df_centroides[['energy', 'acousticness', 'danceability', 'loudness', 'valence', 'tempo', 'Qtd_Musicas']])
    
    # Denominacao das 5 Playlists ("Vibes"): cada cluster recebe o nome da feature em que
    # mais se destaca (maior centroide padronizado). A atribuicao e gulosa e sem repeticao,
    # entao dois clusters nunca ficam com o mesmo nome.
    nomes_por_feature = {
        'speechiness': "Falado & Stand-up",
        'instrumentalness': "Calmaria Instrumental",
        'acousticness': "Acustico com Voz",
        'energy': "Pesado & Adrenalina",
        'danceability': "Festa & Danca",
    }
    centroides_z = pd.DataFrame(kmeans_musica.cluster_centers_, columns=features_acusticas)
    pares = centroides_z[list(nomes_por_feature)].stack().sort_values(ascending=False)
    nomes_playlists = {}
    for (c_id, feature), _ in pares.items():
        if c_id not in nomes_playlists and nomes_por_feature[feature] not in nomes_playlists.values():
            nomes_playlists[c_id] = nomes_por_feature[feature]

    df_clean['nome_playlist'] = df_clean['cluster'].map(nomes_playlists)

    print("\nDescricao das 5 Playlists Geradas:")
    for c_id in sorted(nomes_playlists.keys()):
        qtd = (df_clean['cluster'] == c_id).sum()
        print(f"  Cluster {c_id}: {nomes_playlists[c_id]} ({qtd:,} musicas - {qtd/len(df_clean)*100:.1f}%)")

    # O genero NAO entrou no treino; aqui ele serve so para conferir se os grupos fazem sentido
    print("\nConferencia: 4 generos mais comuns em cada cluster (genero nao foi usado no treino):")
    for c_id in sorted(nomes_playlists.keys()):
        generos = df_clean.loc[df_clean['cluster'] == c_id, 'track_genre'].value_counts().head(4)
        print(f"  Cluster {c_id} ({nomes_playlists[c_id]}): {', '.join(generos.index)}")
        
    # Grafico 2: Radar ou distribuicao dos clusters
    fig, ax = plt.subplots(figsize=(12, 6))
    df_radar = df_centroides[['danceability', 'energy', 'acousticness', 'instrumentalness', 'valence', 'speechiness', 'liveness']].copy()
    
    df_radar.T.plot(kind='bar', ax=ax, colormap='tab10', width=0.8)
    ax.set_title("Comparativo das Assinaturas Acusticas Medias dos 5 Clusters", fontsize=13, fontweight='bold')
    ax.set_xlabel("Propriedades Acusticas", fontsize=11)
    ax.set_ylabel("Valor Medio Normalizado (0 a 1)", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
    ax.legend([f"Cluster {i}: {nomes_playlists[i]}" for i in range(5)], loc='upper center', bbox_to_anchor=(0.5, -0.18),
              ncol=3, frameon=False, fontsize=10)
    plt.tight_layout()
    caminho_radar = os.path.join(OUTPUT_DIR, 'p4_clusters_distribuicao.png')
    fig.savefig(caminho_radar, dpi=300)
    plt.close(fig)
    print(f"\nGrafico salvo: {caminho_radar}")
    
    # 4. A Prova de Fogo: Extracao de 5 musicas de um cluster para inspecao qualitativa
    print("\n" + "=" * 50)
    print("4. A PROVA DE FOGO: INSPECAO QUALITATIVA DE 5 MUSICAS")
    cluster_acustico_id = next(c for c, n in nomes_playlists.items() if n == "Calmaria Instrumental")
    cluster_nome = nomes_playlists[cluster_acustico_id]

    # Sorteio entre as faixas populares (popularity >= 70) para que a equipe conheca as musicas
    # e consiga julgar de ouvido se elas combinam
    populares = df_clean[(df_clean['cluster'] == cluster_acustico_id) & (df_clean['popularity'] >= 70)]
    musicas_cluster = populares.sample(n=5, random_state=42)

    print(f"\nCluster Selecionado: Cluster {cluster_acustico_id} -> {cluster_nome}")
    print(f"Amostra de 5 faixas entre as {len(populares)} populares do cluster:")
    for idx, (_, row) in enumerate(musicas_cluster.iterrows(), 1):
        print(f"  {idx}. Artista: {row['artists']} | Faixa: '{row['track_name']}' | Genero cadastrado: {row['track_genre']}")
        print(f"     [Acousticness: {row['acousticness']:.3f} | Instrumentalness: {row['instrumentalness']:.3f} | "
              f"Energy: {row['energy']:.3f} | Valence: {row['valence']:.3f} | Loudness: {row['loudness']:.1f} dB]")
    print("=" * 50 + "\n")
    
    return {
        'correlacao_r': r_energy_acoustic,
        'cluster_selecionado': cluster_acustico_id,
        'nome_cluster': cluster_nome,
        'musicas': musicas_cluster[['artists', 'track_name', 'album_name', 'acousticness', 'energy']].to_dict(orient='records')
    }

if __name__ == '__main__':
    executar_problema_4()
