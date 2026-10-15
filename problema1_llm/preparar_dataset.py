"""Converte o house-prices.csv em texto para fazer fine-tuning de um LLM no LlamaFactory.

Usa o MESMO split 80/20 (random_state=42) do baseline e do problema1_boosting.py,
para que o MAE do LLM seja comparavel com o dos outros modelos.

Uso (na raiz do repositorio):
    python problema1_llm/preparar_dataset.py
"""
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

CAMINHO_DADOS = '4-bloco-Machine Learing/TRABALHO AV1/Problema 1/house-prices.csv'
PASTA_SAIDA = os.path.join('problema1_llm', 'dataset')
SEED = 42

INSTRUCAO = ("Voce e um avaliador de imoveis de Ames, Iowa (EUA). Com base nas caracteristicas "
             "do imovel, responda apenas o preco de venda estimado em dolares, como um numero inteiro.")


def imovel_para_texto(linha):
    # Pula campos vazios para o prompt ficar mais curto
    campos = [f"{col}={valor}" for col, valor in linha.items() if pd.notna(valor)]
    return "; ".join(campos)


def para_exemplos(df):
    exemplos = []
    for _, linha in df.iterrows():
        exemplos.append({
            'instruction': INSTRUCAO,
            'input': imovel_para_texto(linha.drop(labels=['Id', 'SalePrice'])),
            'output': str(int(linha['SalePrice'])),
            'id': int(linha['Id']),
        })
    return exemplos


def main():
    df = pd.read_csv(CAMINHO_DADOS)
    train, test = train_test_split(df, test_size=0.20, random_state=SEED)
    # Mesmos outliers removidos no problema1_boosting.py
    train = train[~((train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000))]
    # 10% do treino vira validacao para acompanhar o eval_loss durante o fine-tuning
    train, val = train_test_split(train, test_size=0.10, random_state=SEED)

    os.makedirs(PASTA_SAIDA, exist_ok=True)
    for nome, parte in [('casas_train', train), ('casas_val', val), ('casas_test', test)]:
        with open(os.path.join(PASTA_SAIDA, f'{nome}.json'), 'w', encoding='utf-8') as f:
            json.dump(para_exemplos(parte), f, ensure_ascii=False, indent=1)
        print(f"{nome}: {len(parte)} imoveis")

    colunas = {'prompt': 'instruction', 'query': 'input', 'response': 'output'}
    info = {nome: {'file_name': f'{nome}.json', 'columns': colunas}
            for nome in ['casas_train', 'casas_val', 'casas_test']}
    with open(os.path.join(PASTA_SAIDA, 'dataset_info.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    print(f"Arquivos salvos em {PASTA_SAIDA}/")


if __name__ == '__main__':
    main()
