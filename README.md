# Trabalho Prático AV1 — Machine Learning

Resolução dos quatro problemas do Trabalho Prático AV1 (Trilha de Machine Learning).

| Problema | Tipo | Dataset |
| --- | --- | --- |
| 1. O Preço Justo | Regressão Linear | `house-prices.csv` (Ames Housing) |
| 2. A Ameaça do Churn | Classificação | Telco Customer Churn |
| 3. O Perfil do Consumidor | Clusterização (K-Means) | Mall Customers |
| 4. Criando a Playlist Perfeita | Clusterização (K-Means) | Spotify Tracks |

## Estrutura

- `4-bloco-Machine Learing/TRABALHO AV1/Problema N/` — datasets de cada problema
- `scripts/` — um script por problema (`problemaN_*.py`) e geradores do notebook e do relatório
- `Trabalho_Pratico_AV1_ML.ipynb` — notebook com todos os problemas
- `outputs/` — gráficos gerados e relatório em `.docx`
- `RELATORIO_TRABALHO_AV1_ML.md` — relatório em Markdown

## Como rodar

Na raiz do repositório:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python scripts/problema1_regressao.py
```
