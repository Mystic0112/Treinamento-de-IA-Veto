"""Calcula o MAE do LLM ajustado a partir das previsoes geradas pelo LlamaFactory.

Uso (na raiz do repositorio, depois de rodar qwen3-0.6b-casas-prever.yaml):
    python problema1_llm/avaliar_llm.py
"""
import os
import re
import json
import numpy as np

CAMINHO_PREVISOES = os.path.join('problema1_llm', 'saidas', 'qwen3-0.6b-casas-previsoes',
                                 'generated_predictions.jsonl')
CAMINHO_RESULTADOS_BOOSTING = os.path.join('outputs', 'p1_resultados_competicao.json')


def extrair_numero(texto):
    numeros = re.findall(r'\d+', texto.replace(',', '').replace('.', ''))
    return float(numeros[0]) if numeros else np.nan


def main():
    with open(CAMINHO_PREVISOES, encoding='utf-8') as f:
        linhas = [json.loads(l) for l in f]
    reais = np.array([extrair_numero(l['label']) for l in linhas])
    previstos = np.array([extrair_numero(l['predict']) for l in linhas])

    validos = ~np.isnan(previstos)
    print(f"Respostas com numero valido: {validos.sum()}/{len(linhas)}")
    erros = np.abs(reais[validos] - previstos[validos])
    mae = erros.mean()
    ss_res = ((reais[validos] - previstos[validos]) ** 2).sum()
    ss_tot = ((reais[validos] - reais[validos].mean()) ** 2).sum()
    print(f"LLM Qwen3-0.6B + LoRA -> MAE no teste: ${mae:,.0f} | R2: {1 - ss_res / ss_tot:.4f}")
    print(f"  Mediana do erro: ${np.median(erros):,.0f} | Maior erro: ${erros.max():,.0f}")

    if os.path.exists(CAMINHO_RESULTADOS_BOOSTING):
        with open(CAMINHO_RESULTADOS_BOOSTING, encoding='utf-8') as f:
            outros = json.load(f)['resultados_teste']
        print("\nComparacao no mesmo conjunto de teste (291 imoveis):")
        for nome, r in outros.items():
            print(f"  {nome:35s} MAE ${r['MAE']:,.0f}")
        print(f"  {'LLM Qwen3-0.6B + LoRA':35s} MAE ${mae:,.0f}")


if __name__ == '__main__':
    main()
