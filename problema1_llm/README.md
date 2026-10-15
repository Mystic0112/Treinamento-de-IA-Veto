# Problema 1 — Fine-tuning de LLM (LlamaFactory)

Experimento: ajustar o **Qwen3-0.6B** com **LoRA** no LlamaFactory para prever o `SalePrice`
a partir das características do imóvel escritas como texto. O resultado é comparado com
os modelos de `scripts/problema1_boosting.py` no **mesmo conjunto de teste** (291 imóveis).

## Passo a passo

Na raiz do repositório:

```bash
python problema1_llm/preparar_dataset.py
```

Depois, de dentro de `problema1_llm/`, com o Python do ambiente onde o LlamaFactory está instalado:

```bash
llamafactory-cli train qwen3-0.6b-casas.yaml
```

```bash
llamafactory-cli train qwen3-0.6b-casas-prever.yaml
```

Por fim, de volta à raiz:

```bash
python problema1_llm/avaliar_llm.py
```

## Arquivos

- `preparar_dataset.py` — converte o CSV em exemplos `instrução → preço` (formato Alpaca), com o mesmo split 80/20 (`random_state=42`)
- `qwen3-0.6b-casas.yaml` — treino LoRA (cabe numa GPU de 8 GB)
- `qwen3-0.6b-casas-prever.yaml` — gera as previsões do conjunto de teste
- `avaliar_llm.py` — calcula o MAE e compara com os outros modelos
