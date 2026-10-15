"""Calcula o MAE do LLM ajustado a partir das previsoes geradas pelo LlamaFactory.

Uso (na raiz do repositorio, depois de rodar qwen3-0.6b-casas-prever.yaml):
    python problema1_llm/avaliar_llm.py
"""
import os
import re
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt

CAMINHO_PREVISOES = os.path.join('problema1_llm', 'saidas', 'qwen3-0.6b-casas-previsoes',
                                 'generated_predictions.jsonl')
PASTA_ADAPTADOR = os.path.join('problema1_llm', 'saidas', 'qwen3-0.6b-casas-lora')
OUTPUT_DIR = 'outputs'
CAMINHO_RESULTADOS_BOOSTING = os.path.join(OUTPUT_DIR, 'p1_resultados_competicao.json')


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

    r2 = 1 - ss_res / ss_tot
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'p1_resultados_llm.json'), 'w', encoding='utf-8') as f:
        json.dump({'modelo': 'Qwen3-0.6B + LoRA', 'MAE': mae, 'R2': r2,
                   'mediana_erro': float(np.median(erros)), 'maior_erro': float(erros.max()),
                   'respostas_validas': int(validos.sum()), 'total': len(linhas)}, f, indent=2)

    # Curvas de loss geradas pelo LlamaFactory (plot_loss: true)
    for origem, destino in [('training_loss.png', 'p1_llm_training_loss.png'),
                            ('training_eval_loss.png', 'p1_llm_eval_loss.png')]:
        caminho = os.path.join(PASTA_ADAPTADOR, origem)
        if os.path.exists(caminho):
            shutil.copy(caminho, os.path.join(OUTPUT_DIR, destino))

    if os.path.exists(CAMINHO_RESULTADOS_BOOSTING):
        with open(CAMINHO_RESULTADOS_BOOSTING, encoding='utf-8') as f:
            outros = json.load(f)['resultados_teste']
        print("\nComparacao no mesmo conjunto de teste (291 imoveis):")
        for nome, r in outros.items():
            print(f"  {nome:35s} MAE ${r['MAE']:,.0f}")
        print(f"  {'LLM Qwen3-0.6B + LoRA':35s} MAE ${mae:,.0f}")
        outros['LLM Qwen3-0.6B + LoRA'] = {'MAE': mae, 'R2': r2}
        grafico_comparacao(outros)


def grafico_comparacao(resultados):
    # Um unico valor por modelo: barras horizontais, ordenadas do menor erro para o maior
    nomes = sorted(resultados, key=lambda n: resultados[n]['MAE'])
    maes = [resultados[n]['MAE'] for n in nomes]
    vencedor = nomes[0]
    cores = ['#2a78d6' if n == vencedor else '#b9b8b2' for n in nomes]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.patch.set_facecolor('#fcfcfb')
    ax.set_facecolor('#fcfcfb')
    barras = ax.barh(nomes, maes, color=cores, height=0.6, edgecolor='#fcfcfb', linewidth=2)
    ax.invert_yaxis()
    for barra, valor in zip(barras, maes):
        ax.text(valor + 400, barra.get_y() + barra.get_height() / 2, f"US$ {valor:,.0f}".replace(',', '.'),
                va='center', fontsize=10, color='#0b0b0b')
    ax.set_xlabel("MAE no conjunto de teste (US$, menor é melhor)", color='#52514e')
    ax.set_title("Problema 1: erro médio absoluto por modelo (291 imóveis de teste)",
                 loc='left', fontsize=12, fontweight='bold', color='#0b0b0b')
    ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1000:.0f} mil")
    ax.set_xlim(0, max(maes) * 1.15)
    ax.grid(axis='x', color='#e1e0d9', linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ['top', 'right', 'left']:
        ax.spines[lado].set_visible(False)
    ax.spines['bottom'].set_color('#898781')
    ax.tick_params(colors='#52514e', length=0)
    plt.tight_layout()
    caminho = os.path.join(OUTPUT_DIR, 'p1_comparacao_modelos.png')
    fig.savefig(caminho, dpi=200)
    plt.close(fig)
    print(f"\nGrafico salvo: {caminho}")


if __name__ == '__main__':
    main()
