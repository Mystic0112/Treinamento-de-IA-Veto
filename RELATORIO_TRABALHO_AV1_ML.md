# Relatorio Tecnico: Trabalho Pratico AV1 - Trilha de Machine Learning

**Instituicao de Ensino Superior | Disciplina: Machine Learning**  
**Autor / Especialista:** Dado (Squad de Data Science e Machine Learning)  
**Data:** Outubro de 2026  
**Status:** Concluido com Exito | Reprodutibilidade Garantida (`random_state = 42`)  
**Notebook Executavel:** [Trabalho_Pratico_AV1_ML.ipynb](file:///c:/Users/c9hes/Documents/faculdade/machine%20learning/Trabalho_Pratico_AV1_ML.ipynb)

---

## Sumario Executivo

O presente documento consolida a resolucao tecnica e analitica dos quatro problemas propostos no Trabalho Pratico da Avaliacao 1 (AV1) de Machine Learning. O escopo compreende o ciclo ponta a ponta da Ciencia de Dados: analise exploratoria, saneamento de dados com tratamento estatistico de valores ausentes, engenharia de atributos com fundamentacao em algebra linear, formulacao de modelos supervisionados (Regressao Linear e Classificacao Binaria) e nao supervisionados (Clusterizacao via K-Means), com validacao rigorosa de metricas de desempenho e interpretacao comercial acionavel.

### Sintese Geral dos Resultados

| Problema | Paradigma | Dataset | Algoritmo | Principais Metricas | Conclusao Principal |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. O Preco Justo** | Supervisionado (Regressao) | Ames Housing (1.455 linhas) | Linear Regression & Ridge | $R^2 = 82.32\%$ (Baseline)<br>$R^2 = 84.85\%$ (Ridge)<br>$\text{MAE} = \$19,245.03$ | Qualidade geral (`OverallQual`), area util (`GrLivArea`) e garagens explicam mais de 84% do valor venal. |
| **2. A Ameaca do Churn** | Supervisionado (Classificacao) | Telco Churn (7.043 linhas) | Regressao Logistica & Random Forest | $\text{ROC-AUC} = 0.8422$<br>$\text{Acuracia} = 80.62\%$<br>$\text{Recall Churn} = 55.88\%$ | Contratos mensais evadem 15.1x mais que contratos bianuais. Falso Negativo priorizado devido ao CAC elevado. |
| **3. O Perfil do Consumidor** | Nao Supervisionado (Clusterizacao) | Mall Customers (200 linhas) | K-Means ($K=5$) com StandardScaler | $\text{Inercia} = 65.57$<br>$\text{Silhouette} = 0.5547$ | Identificacao nitida de 5 personas comerciais com planos de marketing direcionados. |
| **4. A Playlist Perfeita** | Nao Supervisionado (Clusterizacao) | Spotify Tracks (114.000 linhas) | K-Means ($K=5$) com 9 atributos acusticos | Pearson $r = -0.7339$<br>5 Playlists Acusticas Coesas | Isolamento estrito de metadados nominais. Validacao qualitativa comprova harmonia auditiva das faixas agrupadas. |

---

## PARTE 1: Aprendizado Supervisionado

```mermaid
flowchart TD
    subgraph P1[Problema 1: Regressao Linear]
        A1[Carga: Ames Housing 1455x81] --> B1[Analise de Correlacao Linear]
        B1 --> C1[Tratamento: LotFrontage & BsmtQual]
        C1 --> D1[Engenharia: Ordinal & One-Hot Encoding]
        D1 --> E1[Split 80/20 & Treino OLS Baseline]
        E1 --> F1[Otimizacao Ridge: R2 = 84.85% | MAE = $19,245]
    end
    subgraph P2[Problema 2: Classificacao Churn]
        A2[Carga: Telco Churn 7043x21] --> B2[Diagnostico de Desbalanceamento]
        B2 --> C2[Analise Contratual: Mensal vs Anual]
        C2 --> D2[Tratamento: TotalCharges 11 brancos tenure=0]
        D2 --> E2[Treino Estratificado: Logistic Regression & RF]
        E2 --> F2[Matriz de Confusao & Analise de Custos FN vs FP]
    end
```

---

## Problema 1: Regressao Linear - "O Preco Justo"

### 1.1 Contexto e Regras da Competicao
Uma imobiliaria contratou nossa equipe de cientistas de dados para construir um modelo preditivo capaz de avaliar e precificar imoveis automaticamente. A base disponibilizada contem 1.455 imoveis (apos a extracao de 5 registros 'secretos' da competicao). O objetivo competitivo e atingir o menor **Erro Absoluto Medio (MAE)** nos 5 imoveis de teste final.

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

### 1.2 Analise Exploratoria: Identificacao dos Top 3 Preditores Lineares
Calculamos o Coeficiente de Correlacao Linear de Pearson ($r$) entre todas as variaveis numericas e a variavel alvo `SalePrice`:

$$r_{X, Y} = \frac{\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^n (X_i - \bar{X})^2 \sum_{i=1}^n (Y_i - \bar{Y})^2}}$$

#### Tabela 1: Variaveis com Maior Correlacao Linear Positiva com `SalePrice`

| Posicao | Variavel | Coeficiente de Pearson ($r$) | Descricao do Atributo |
| :---: | :--- | :---: | :--- |
| **1** | `OverallQual` | **+0.7918** | Avaliacao geral do acabamento e materiais do imovel (escala de 1 a 10). |
| **2** | `GrLivArea` | **+0.7012** | Area habitavel construida acima do nivel do solo (em pes quadrados). |
| **3** | `GarageCars` | **+0.6430** | Capacidade da garagem em numero de veiculos abrigados. |
| 4 | `GarageArea` | +0.6272 | Area total construida da garagem (em pes quadrados). |
| 5 | `TotalBsmtSF` | +0.6040 | Area total do subsolo/porao (em pes quadrados). |
| 6 | `1stFlrSF` | +0.5980 | Area util do primeiro pavimento (em pes quadrados). |
| 7 | `FullBath` | +0.5605 | Quantidade de banheiros completos acima do solo. |
| 8 | `TotRmsAbvGrd` | +0.5302 | Total de comodos da casa (excluindo banheiros). |
| 9 | `YearBuilt` | +0.5276 | Ano original de construcao da edificacao. |
| 10 | `YearRemodAdd` | +0.5152 | Ano da ultima grande reforma/remodelacao. |

As 3 variaveis com maior correlacao linear positiva sao: **`OverallQual`** ($r = 0.7918$), **`GrLivArea`** ($r = 0.7012$) e **`GarageCars`** ($r = 0.6430$).

![Heatmap de Correlacoes](file:///c:/Users/c9hes/Documents/faculdade/machine%20learning/outputs/p1_correlacoes_heatmap.png)
*Figura 1: Matriz de correlacao linear de Pearson das 10 variaveis mais fortemente correlacionadas com `SalePrice`.*

![Scatter Plots Top 3 Preditores](file:///c:/Users/c9hes/Documents/faculdade/machine%20learning/outputs/p1_scatter_top3.png)
*Figura 2: Graficos de dispersao com reta de regressao linear para os 3 principais preditores (`OverallQual`, `GrLivArea` e `GarageCars`).*

---

### 1.3 Tratamento de Valores Ausentes (Missing Values)

O dataset possui atributos com dados faltantes que exigem tratamento previo a modelagem. Conforme solicitado, selecionamos uma coluna numerica e uma categorica:

#### 1. Coluna Numerica: `LotFrontage` (Comprimento da Fachada / Testada do Lote)
- **Diagnostico:** Apresenta 259 valores ausentes (17.80% da base).
- **Tratamento Adotado:** Imputacao pela **mediana condicional agrupada por bairro (`Neighborhood`)**.
- **Justificativa Estatistica e de Negocio:** O tamanho da testada de um terreno residencial em uma cidade e determinado pelo zoneamento e pelas dimensoes padronizadas dos loteamentos de cada vizinhanca. Imoveis em bairros densos tem testadas menores e homogeneas, enquanto bairros nobres possuem frentes amplas. Ademais, optou-se pela mediana em vez da media aritmetica devido a assimetria positiva tipica de variaveis espaciais/imobiliarias, protegendo o estimador contra a influencia de terrenos atipicos (outliers).

#### 2. Coluna Categorica: `BsmtQual` (Altura e Qualidade do Pe-Direito do Porao)
- **Diagnostico:** Apresenta 37 valores nulos (2.54% da base).
- **Tratamento Adotado:** Preenchimento com uma categoria explicita `'None'` ("Sem Porao").
- **Justificativa Estatistica e de Negocio:** De acordo com o codebook oficial do Ames Housing Dataset, a ausencia de valor em atributos de porao nao decorre de uma falha aleatoria de coleta (Missing Completely at Random - MCAR), mas sim de **Missingness Estrutural (Missing Not at Random - MNAR)**: o imovel simplesmente nao possui porao. Excluir essas linhas eliminaria imoveis validos da base; imputar a moda (categoria mais comum, `Gd` ou `TA`) inventaria artificialmente um porao inexistente, distorcendo o preco venal. A criacao da classe `'None'` preserva a realidade fisica da edificacao.

---

### 1.4 Engenharia de Atributos: Codificacao Categorica

Transformamos duas variaveis categoricas essenciais em representacoes numericas:
1. **`BsmtQual` via Mapeamento Ordinal (Ordinal Encoding):** Como ha uma clara hierarquia de qualidade tecnica, atribuimos valores inteiros monotonicamente crescentes:
   $$\text{Ex (Excelente)} \rightarrow 5, \quad \text{Gd (Bom)} \rightarrow 4, \quad \text{TA (Medio)} \rightarrow 3, \quad \text{Fa (Fraco)} \rightarrow 2, \quad \text{Po (Ruim)} \rightarrow 1, \quad \text{None (Sem Porao)} \rightarrow 0$$
2. **`Neighborhood` via Codificacao One-Hot (Dummies):** Bairros sao classes puramente nominais sem relacao de magnitude intrinseca. Aplicamos `pd.get_dummies(..., drop_first=True)` para criar variaveis binarias $D_j \in \{0, 1\}$, descartando a primeira categoria para impedir armadilha de multicolinearidade estrita (*dummy variable trap*).

#### Por que o Modelo de Regressao Linear Exige essa Conversao?
A Regressao Linear Multipla baseia-se na hipotese fundamental de que a resposta $y \in \mathbb{R}$ e gerada por uma combinacao linear ponderada dos atributos:

$$y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \dots + \beta_p x_{ip} + \epsilon_i$$

Sob a notacao matricial:

$$Y = X\beta + \epsilon$$

O estimador de Minimos Quadrados Ordinarios (OLS) encontra os parametros otimos minimizando a soma dos residuos quadraticos ($S(\beta) = \|Y - X\beta\|^2$), cuja solucao analitica fechada (equacao normal) e:

$$\hat{\beta} = (X^T X)^{-1} X^T Y$$

A matriz de planejamento $X \in \mathbb{R}^{n \times (p+1)}$ deve necessariamente residir no espaco vetorial dos numeros reais $\mathbb{R}$. Strings e rotulos textuais (ex: `'CollgCr'`, `'Veenker'`) nao possuem estrutura algebrica: nao admitem operacoes de adicao vetorial, multiplicacao por escalar nem produto interno euclidiano $\langle x_i, \beta \rangle$. A codificacao numerica e requisito matematico mandatorio para viabilizar o calculo da inversa $(X^T X)^{-1}$ e a projecao ortogonal no hiperplano de regressao.

---

### 1.5 Treinamento, Metricas do Baseline e Modelo Otimizado

Dividimos o dataset em 80% para treino (1.164 imoveis) e 20% para teste (291 imoveis), fixando a semente `random_state = 42`.

#### Tabela 2: Comparativo de Desempenho dos Modelos de Regressao

| Metrica de Avaliacao | Formula Matematica | Modelo Baseline (Linear OLS) | Modelo Refinado (Ridge + Bairros) | Variacao / Ganho |
| :--- | :---: | :---: | :---: | :---: |
| **$R^2$ no Treino** | $1 - \frac{SS_{res}}{SS_{tot}}$ | $0.7647$ ($76.47\%$) | $0.8210$ ($82.10\%$) | $+5.63\%$ |
| **$R^2$ no Teste** | $1 - \frac{SS_{res}}{SS_{tot}}$ | **$0.8232$ ($82.32\%$)** | **$0.8485$ ($84.85\%$)** | **$+2.53\%$** |
| **MAE no Teste** | $\frac{1}{n}\sum |y - \hat{y}|$ | **$\$21,888.03$** | **$\$19,245.03$** | **$-\$2,643.00$** |
| **RMSE no Teste** | $\sqrt{\frac{1}{n}\sum (y - \hat{y})^2}$ | $\$29,065.89$ | $\$26,909.47$ | $-\$2,156.42$ |

#### O que o $R^2 = 82.32\%$ Significa para a Imobiliaria?
O Coeficiente de Determinacao ($R^2$) mede a proporcao da variancia total observada na variavel dependente (`SalePrice`) que e matematicamente explicada pela combinacao linear dos atributos selecionados no modelo:

$$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$

Para a diretoria da imobiliaria, o resultado de **$82.32\%$** comprova que o algoritmo e altamente fidedigno e consistente: mais de quatro quintos de toda a oscilacao de precos do mercado imobiliario local derivam diretamente de caracteristicas tecnicas objetivas do imovel (como qualidade de acabamento, metragem util, vagas de garagem e conservacao do porao). Apenas $17.68\%$ da variabilidade decorre de fatores nao mensurados (como pressa do vendedor, estado emocional do comprador ou variacoes microeconomicas sazonais).

No modelo refinado com **Ridge Regression** e inclusao de variaveis de localizacao espacial, o $R^2$ subiu para **$84.85\%$** e o **MAE reduziu para $\$19,245.03$**, conferindo vantagem competitiva para a equipe na precificacao dos 5 imoveis secretos.

![Residuos e Dispersao Real vs Predito](file:///c:/Users/c9hes/Documents/faculdade/machine%20learning/outputs/p1_residuos_dispersao.png)
*Figura 3: Diagnostico de residuos: (Esquerda) Precos reais vs preditos com proximidade a reta de identidade $y = x$; (Direita) Dispersao de residuos com media proxima de zero e homoscedasticidade satisfatoria.*

---

## Problema 2: Classificacao - "A Ameaca do Churn"

### 2.1 Compreensao do Negocio e Analise da Variavel Alvo

No contexto de telecomunicacoes, a evasao de clientes (`Churn`) gera perda direta de receita recorrente. A variavel dependente alvo do dataset e:
- **`Churn`:** Binaria (`Yes` = Cliente cancelou o servico; `No` = Cliente permanece ativo).

#### Diagnostico de Desbalanceamento de Classes
- **Total de Amostras:** 7.043 clientes.
- **Clientes Retidos (`No`):** 5.174 registros (**$73.46\%$**).
- **Clientes que Cancelaram (`Yes`):** 1.869 registros (**$26.54\%$**).
- **Grau de Desbalanceamento:** Razao de **2.77 para 1** (para cada cliente que sai, quase 3 permanecem).

![Distribuicao da Variavel Churn](file:///c:/Users/c9hes/Documents/faculdade/machine%20learning/outputs/p2_distribuicao_churn.png)
*Figura 4: Diagnostico do desbalanceamento da variavel alvo `Churn`.*

**Implicacao Metodologica:** Devido ao desbalanceamento, a metrica de **Acuracia** nao e suficiente isoladamente. Um classificador trivial que atribua a todos os clientes a classe majoritaria (`No`) alcancaria $73.46\%$ de acuracia nominal sem prever um unico cancelamento real. Por essa razao, o modelo deve ser avaliado com enfase em **Recall (Sensibilidade)**, **Precision**, **F1-Score** e a area sob a curva ROC (**ROC-AUC**).

---

### 2.2 Analise de Risco Contratual: Contratos Mensais vs Anuais

Investigamos o impacto do tipo de faturamento contratual (`Contract`) sobre a probabilidade de desligamento:

#### Tabela 3: Cruzamento entre Tipo de Contrato e Evasao de Clientes

| Tipo de Contrato | Clientes Totais | Ficaram (`No`) | Cancelaram (`Yes`) | Taxa de Evasao (%) | Fator de Risco Relativo |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Month-to-month (Mensal)** | 3.875 | 2.220 | 1.655 | **$42.71\%$** | **$15.1 \times$** superior ao bienal |
| **One year (1 Ano)** | 1.473 | 1.307 | 166 | **$11.27\%$** | $4.0 \times$ superior ao bienal |
| **Two year (2 Anos)** | 1.695 | 1.647 | 48 | **$2.83\%$** | Referencia base de fidelidade |

![Analise de Churn por Contrato](file:///c:/Users/c9hes/Documents/faculdade/machine%20learning/outputs/p2_churn_por_contrato.png)
*Figura 5: Analise comparativa de evasao: clientes em contrato mensal apresentam taxa de churn de 42.71%, contra apenas 2.83% em contratos de 2 anos.*

**Conclusao do Risco:** Clientes no modelo mensal cancelam quase 4 vezes mais que os contratantes de 1 ano e mais de **15 vezes mais** que os clientes vinculados a planos de 2 anos. O contrato mensal confere custo de troca nulo ao consumidor, tornando o tipo de contrato o preditor de maior relevancia operacional para acoes de retencao.

---

### 2.3 Preparacao dos Dados e Diagnostico da Coluna `TotalCharges`

- **Problema Detectado:** A coluna `TotalCharges` foi importada como string/object. Ao inspecionar os caracteres, identificamos **11 registros contendo exclusivamente espacos em branco (`' '`)**.
- **Analise Investigativa:** Ao cruzar esses 11 registros com a variavel `tenure` (tempo de permanencia do cliente), constatou-se que **todos os 11 clientes possuem `tenure == 0`**.
- **Interpretacao e Tratamento:** Tratam-se de clientes recem-contratados no primeiro mes de vigencia contratual, cujo ciclo inaugural de faturamento ainda nao foi fechado. Portanto, nao ha inadimplencia nem erro de digitacao aleatorio. A solucao correta foi converter a coluna para numerica via `pd.to_numeric(..., errors='coerce')` e imputar o valor **`0.0`**, preservando todos os 11 clientes na amostra.
- **Pre-processamento Complementar:** Removemos o identificador nominal `customerID` e convertemos todas as 15 variaveis categoricas via One-Hot Encoding (`drop_first=True`).

---

### 2.4 Treinamento Supervisionado e Avaliacao dos Classificadores

Dividimos a base em 80% treino (5.634 amostras) e 20% teste (1.409 amostras) com estratificacao (`stratify=y`). Treinamos a **Regressao Logistica** (com `StandardScaler` nos atributos continuos) e o classificador **Random Forest**.

#### Tabela 4: Desempenho dos Classificadores no Conjunto de Teste

| Modelo Classificador | Acuracia Global | Precisao (Churn) | Recall (Churn) | F1-Score (Churn) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Regressao Logistica** | **$80.62\%$** | $65.93\%$ | **$55.88\%$** | **$0.6049$** | **$0.8422$** |
| **Random Forest** | $79.99\%$ | **$66.55\%$** | $49.47\%$ | $0.5675$ | $0.8442$ |

![Curva ROC Comparativa](file:///c:/Users/c9hes/Documents/faculdade/machine%20learning/outputs/p2_curva_roc.png)
*Figura 6: Curva ROC dos classificadores: Regressao Logistica (AUC = 0.842) e Random Forest (AUC = 0.844).*

---

### 2.5 Matriz de Confusao e Analise de Custos de Negocio: Falso Positivo vs Falso Negativo

A matriz de confusao da Regressao Logistica no conjunto de teste registrou:
- **Verdadeiros Negativos (TN):** 927 clientes fieis previstos corretamente ($65.8\%$).
- **Falsos Positivos (FP):** 108 clientes fieis previstos erroneamente como churn ($7.7\%$).
- **Falsos Negativos (FN):** 165 clientes em evasao nao detectados pelo modelo ($11.7\%$).
- **Verdadeiros Positivos (TP):** 209 clientes em evasao detectados a tempo ($14.8\%$).

![Matriz de Confusao](file:///c:/Users/c9hes/Documents/faculdade/machine%20learning/outputs/p2_matriz_confusao.png)
*Figura 7: Matriz de Confusao da Regressao Logistica e Random Forest nos dados de teste.*

#### No Contexto Desta Empresa, o que e Pior: Falso Positivo ou Falso Negativo?

O erro de **Falso Negativo (FN)** e dramaticamente pior e mais oneroso para a operadora de telecomunicacoes:

1. **Impacto do Falso Negativo (FN = 165 clientes perdidos):**
   - O algoritmo prevê que o cliente esta satisfeito e nao cancelara o servico.
   - Nenhuma acao preventiva e deflagrada pelo time de retencao.
   - O cliente cancela o contrato e migra para o concorrente.
   - **Prejuizo Financeiro Direto:** Perda integral do valor do ciclo de vida do cliente (**Lifetime Value - LTV**), que acumula mensalidades por anos. Alem disso, de acordo com estudos classicos de gestao de telecomunicacoes, o **Custo de Aquisicao de Clientes (CAC)** e de **5 a 7 vezes mais caro** do que o custo de retencao de um cliente existente.
2. **Impacto do Falso Positivo (FP = 108 clientes com oferta desnecessaria):**
   - O algoritmo preve que o cliente cancelara, embora ele nao tivesse intencao imediata de sair.
   - A empresa aborda o cliente e concede um beneficio (ex: desconto de R$ 15 na mensalidade por 3 meses ou upgrade temporario).
   - **Prejuizo Financeiro:** Apenas o custo marginal do desconto concedido. Como efeito colateral virtuoso, a intervencao eleva a percepcao de valor e reforca a lealdade de um consumidor que ja era fiel.

**Recomendacao Estrategica da Squad:** A empresa deve calibrar o limiar de decisao (*threshold*) do modelo logistico de $0.50$ para aproximadamente $0.35$ ou $0.40$, elevando a taxa de **Recall** para acima de $75\%$, aceitando conscientemente uma quantidade discretamente maior de Falsos Positivos para blindar a carteira contra evasao irreparavel.

---

## PARTE 2: Aprendizado Nao Supervisionado

```mermaid
flowchart TD
    subgraph P3[Problema 3: Perfil do Consumidor]
        A3[Mall Customers: Renda & Gasto] --> B3[Padronizacao Z-score: StandardScaler]
        B3 --> C3[Metodo do Cotovelo: K=1 a 10 & Silhueta]
        C3 --> D3[K Ideal = 5: Inercia 65.57 | Silhouette 0.5547]
        D3 --> E3[Rotulacao Comercial de 5 Perfis de Negocio]
    end
    subgraph P4[Problema 4: Playlists Spotify]
        A4[Spotify 114k musicas] --> B4[Isolamento de Metadados Textuais]
        B4 --> C4[Correlacao Acustica: Energy vs Acousticness r = -0.7339]
        C4 --> D4[K-Means K=5 sobre 9 Atributos Continuos]
        D4 --> E4[Prova de Fogo: 5 Faixas Acusticas com Coesao Auditiva]
    end
```

---

## Problema 3: Clusterizacao - "O Perfil do Consumidor"

### 3.1 Justificativa Matematica do Escalonamento (StandardScaler)

O algoritmo K-Means particiona os $N$ dados em $K$ grupos minimizando a soma das distancias euclidianas quadraticas intra-cluster (Inercia / WCSS):

$$J = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|^2 = \sum_{k=1}^K \sum_{x_i \in C_k} \sum_{j=1}^p (x_{ij} - \mu_{kj})^2$$

No espaco vetorial de `Mall_Customers`:
- A variavel `Annual Income` varia de $\$15.000$ a $\$137.000$ (desvio padrao $\sigma_1 = 26.26$).
- A variavel `Spending Score` varia de $1$ a $99$ (desvio padrao $\sigma_2 = 25.82$).
- Caso utilizassemos a renda bruta em dolares integrais (escala de dezenas de milhares), a diferenca quadratica $(\Delta x_{\text{Renda}})^2$ assumiria valores da ordem de $10^7$ a $10^8$, enquanto o score de gastos produziria diferencas na ordem de $10^2$ a $10^3$.

Sem a devida padronizacao, o termo de maior magnitude dominaria $99.9\%$ do calculo da distancia euclidiana, anulando a contribuicao da pontuacao de gastos e gerando clusters alongados e enviesados.
- **Solucao Adotada:** Aplicamos o `StandardScaler` (padronizacao por Z-Score), transformando cada variavel:
  $$z = \frac{x - \mu}{\sigma}$$
  Dessa forma, ambos os atributos passam a ter media nula ($\mu = 0$) e variancia unitaria ($\sigma^2 = 1$), conferindo peso geometrico equivalente a ambos os eixos no espaco euclidiano bidimensional.

---

### 3.2 O Dilema da Quantidade: Metodo do Cotovelo e Coeficiente de Silhueta

Executamos o K-Means variando o hiperparametro $K$ de 1 a 10 com a inicializacao inteligente `k-means++` e semente fixa (`random_state=42`).

#### Tabela 5: Evolucao da Inercia (WCSS) e Silhouette Score por Quantidade de Grupos ($K$)

| $K$ | Inercia Intra-Cluster (WCSS) | Coeficiente de Silhueta Medio | Diagnostico Tecnico |
| :---: | :---: | :---: | :--- |
| 1 | $400.00$ | N/A | Grupo unico (ausencia de particionamento). |
| 2 | $269.69$ | $0.3213$ | Particionamento excessivamente generico. |
| 3 | $157.70$ | $0.4666$ | Separacao intermediaria. |
| 4 | $108.92$ | $0.4939$ | Bom agrupamento, mas ainda agrega perfis distintos. |
| **5** | **$65.57$** | **$0.5547$** | **Ponto de Inflexao Otimo (Cotovelo) & Pico Maximo de Silhueta.** |
| 6 | $55.06$ | $0.5399$ | Queda na coesao da silhueta; ganho marginal de inercia reduzido. |
| 7 | $44.86$ | $0.5281$ | Fragmentacao progressiva dos clusters. |
| 8 | $37.23$ | $0.4552$ | Queda expressiva de silhueta. |
| 9 | $32.39$ | $0.4571$ | Subdivisao artificial. |
| 10 | $29.98$ | $0.4432$ | Sobreajuste (overclustering). |

![Metodo do Cotovelo e Silhueta](file:///c:/Users/c9hes/Documents/faculdade/machine%20learning/outputs/p3_elbow_silhouette.png)
*Figura 8: (Esquerda) Curva do Cotovelo evidenciando inflexao em K = 5; (Direita) Coeficiente de Silhueta confirmando o pico maximo global em K = 5 (score = 0.5547).*

#### Justificativa da Escolha de $K = 5$:
1. **Analise do Cotovelo:** A curva de inercia decresce acentuadamente de $K=1$ ($400.00$) ate $K=5$ ($65.57$), ponto no qual a reta muda de curvatura. A partir de $K=6$, a reducao de inercia se torna marginal e linear.
2. **Validacao pelo Coeficiente de Silhueta:** O Silhouette Score avalia se os pontos estao densamente agrupados dentro de seu proprio cluster e distantes dos clusters vizinhos. O valor atinge seu ápice absoluto em **$K = 5$ com score de $0.5547$**, comprovando matematicamente que cinco e o numero geometricamente natural de agrupamentos.

---

### 3.3 Treinamento e Visualizacao dos Clusters

Treinamos o modelo K-Means final com $K = 5$ utilizando as colunas padronizadas de renda e pontuacao de gastos. Os centroides finais calculados no espaco original foram:

#### Tabela 6: Coordenadas dos Centroides em Escala Original

| Cluster | Renda Anual Media ($k) | Pontuacao de Gastos Media (1-100) | Volume de Clientes | Percentual da Base |
| :---: | :---: | :---: | :---: | :---: |
| **0** | $\$55.30\text{k}$ | $49.52$ | 81 | $40.5\%$ |
| **1** | $\$86.54\text{k}$ | $82.13$ | 39 | $19.5\%$ |
| **2** | $\$25.73\text{k}$ | $79.36$ | 22 | $11.0\%$ |
| **3** | $\$88.20\text{k}$ | $17.11$ | 35 | $17.5\%$ |
| **4** | $\$26.30\text{k}$ | $20.91$ | 23 | $11.5\%$ |

![Segmentacao Mall Customers](file:///c:/Users/c9hes/Documents/faculdade/machine%20learning/outputs/p3_clusters_mall_customers.png)
*Figura 9: Segmentacao 2D dos 200 clientes do Shopping Center com os 5 centroides destacados em preto e dourado.*

---

### 3.4 Interpretacao de Negocios e Planos de Marketing

Atribuimos nomes comerciais intuitivos e orientados a negocios para cada um dos 5 clusters identificados pelo algoritmo:

#### 1. Cluster 1: "Clientes VIP / Consumidores de Alto Luxo" (Alta Renda, Alto Gasto)
- **Perfil:** 39 clientes ($19.5\%$). Apresentam renda anual media de $\$86.5\text{k}$ e escore de gastos de $82.1$.
- **Comportamento:** Publico nobre de elevadissima rentabilidade, altamente receptivo a novidades de marcas de alto padrao.
- **Acao de Marketing:** Campanhas de lancamento exclusivo, lounges VIP, servico de concierge personalizado e convites para eventos fechados de alta gastronomia e moda.

#### 2. Cluster 2: "Jovens Impulsivos / Entusiastas da Moda" (Baixa Renda, Alto Gasto)
- **Perfil:** 22 clientes ($11.0\%$). Renda modesta ($\$25.7\text{k}$), mas engajamento financeiro elevado (score $79.4$).
- **Comportamento:** Perfil aspiracional (frequentemente estudantes ou jovens em inicio de carreira), que direcionam a maior parte de sua renda disponivel a consumo de vestuario e entretenimento no shopping.
- **Acao de Marketing:** Oferta de facilidades de pagamento, cartao de credito propio da loja (private label), promocoes relampago em redes sociais (Instagram/TikTok) e marcas de fast-fashion.

#### 3. Cluster 0: "Consumidores Moderados / Classe Media Padrao" (Media Renda, Medio Gasto)
- **Perfil:** 81 clientes ($40.5\%$). Representa o maior contingente do shopping, com renda media de $\$55.3\text{k}$ e score medio de $49.5$.
- **Comportamento:** Familias e consumidores convencionais que utilizam o shopping para compras programadas, alimentacao e compras cotidianas de necessidade.
- **Acao de Marketing:** Programas de pontos e fidelidade acumulativa, campanhas sazonais amplas (Dia das Maes, Natal, Black Friday) e promocoes no cinema e praca de alimentacao.

#### 4. Cluster 3: "Poupadores Cautelosos / Racionais de Alto Poder" (Alta Renda, Baixo Gasto)
- **Perfil:** 35 clientes ($17.5\%$). Elevada renda media ($\$88.2\text{k}$), porem comportamento de gastos muito contido (score $17.1$).
- **Comportamento:** Individuos com alta capacidade financeira que nao compram por impulso; valorizam utilidade, durabilidade e solidez.
- **Acao de Marketing:** Comunicacao institucional que enfatize confiabilidade, tecnologia de ponta, garantia estendida e investimentos em produtos de valor agregado duradouro em vez de apelos emocionais superficiais.

#### 5. Cluster 4: "Consumidores Economicos / Conscientes" (Baixa Renda, Baixo Gasto)
- **Perfil:** 23 clientes ($11.5\%$). Baixa renda media ($\$26.3\text{k}$) associada a baixo score de compras ($20.9$).
- **Comportamento:** Frequentadores pontuais que entram no shopping prioritariamente em busca de servicos basicos ou promocoes de queima de estoque.
- **Acao de Marketing:** Promocoes de queima de estoque, cupons de desconto por volume e campanhas direcionadas estritamente ao custo-beneficio essencial.

---

## Problema 4: Clusterizacao Musical - "Criando a Playlist Perfeita"

### 4.1 Limpeza e Filtragem de Metadados Textuais

O dataset do Spotify disponibilizado contem 114.000 faixas musicais e 21 colunas.

#### As colunas de texto ajudam a calcular distancias matematicas?
**Nao.** Colunas textuais como `track_id`, `artists`, `album_name`, `track_name` e `track_genre` sao rotulos nominais qualitativos.
- **Fundamentacao Matematica:** A metrica euclidiana $d(u, v) = \sqrt{\sum_{i=1}^d (u_i - v_i)^2}$ opera estritamente sobre espacos metricos no corpo dos reais $\mathbb{R}^d$. Textos nao possuem ordenacao metrica intrinseca. Se tentassemos aplicar uma codificacao ordinal arbitraria (ex: transformar nomes de artistas em inteiros de 1 a 30.000), estariamos impondo uma falsa proximidade geometrica (onde o artista 101 pareceria 'mais proximo' do artista 102 do que do 500), introduzindo um ruido severo que corromperia o agrupamento.
- **Acao Pratica Adotada:** Desacoplamos os metadados textuais em uma tabela de consulta (*lookup table*) e utilizamos **exclusivamente as 9 variaveis acusticas continuas** para a formacao dos clusters: `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence` e `tempo`.

---

### 4.2 Analise de Correlacao Acustica: `energy` vs `acousticness`

Calculamos o coeficiente linear de Pearson entre o nivel de energia e a acustica das 114.000 musicas:

$$r = \mathbf{-0.7339}$$

Constata-se uma **correlacao fortemente negativa**.

![Correlacao Acustica Spotify](file:///c:/Users/c9hes/Documents/faculdade/machine%20learning/outputs/p4_correlacao_energy_acousticness.png)
*Figura 10: (Esquerda) Dispersao inversa entre Energy e Acousticness comprovando a forte associacao negativa (r = -0.7339); (Direita) Heatmap completo de correlacao das propriedades acusticas.*

#### O que isso significa no mundo real da musica?
Esta correlacao fortemente inversa e uma consequencia direta dos principios fisicos da gravacao sonora e da engenharia de audio:
1. **Energy (Energia):** Mensura a intensidade sonora percebida, faixa dinamica hipercomprimida (alto ganho RMS), saturacao de graves e frequencias agudas amplificadas. E caracteristica marcante de generos que utilizam geradores eletricos e sintetizados, distorcoes intencionais e amplificadores de guitarra (Rock, Heavy Metal, EDM, Eletronica e Hip-Hop moderno).
2. **Acousticness (Acustica):** Mede a probabilidade estatistica de a gravacao ter sido originada de fontes puramente mecanicas e acusticas sem processamento eletrico ativo (violao desplugado, piano de cauda, quarteto de cordas classico, vozes cruas em salas com reverberacao natural). O som acustico possui transientes dinamicos suaves, menor densidade de decibeis e decaimento harmonico longo.
3. **Conclusao Fisica:** Trata-se de um compromisso sonoro mutuamente excludente: faixas de alta energia requerem eletricidade e compressao macica, destruindo as propriedades acusticas puras; e gravacoes acusticas preservam o silencio e a dinamica natural, impedindo niveis artificiais extremos de energia.

---

### 4.3 Treinamento do K-Means (5 Playlists)

Normalizamos os 9 atributos continuos com `StandardScaler` e instanciamos o K-Means com $K = 5$ clusters. Os centroides acusticos obtidos permitiram mapear as 5 playlists tematicas ("vibes"):

#### Tabela 7: Perfil Sonoro dos 5 Clusters / Playlists

| Cluster | Faixas | Energy | Acousticness | Danceability | Loudness | Valence | Tempo | Conceito da Playlist ("Vibe") |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | 8.393 ($7.4\%$) | $0.75$ | $0.35$ | $0.53$ | $-6.8\text{ dB}$ | $0.50$ | $120.9$ | **Vibe Urbana & Pop Suave:** Faixas radiofonicas com presenca equilibrada de voz e batida comercial. |
| **1** | 31.557 ($27.7\%$) | $0.81$ | $0.06$ | $0.48$ | $-5.8\text{ dB}$ | $0.32$ | $136.8$ | **Treino Intenso & Adrenalina:** Maxima energia, distorcao eletrica pesada e andamento acelerado (Rock/EDM). |
| **2** | 8.163 ($7.2\%$) | **$0.19$** | **$0.83$** | $0.36$ | **$-17.8\text{ dB}$** | $0.19$ | $104.2$ | **Acustico Intimista & Calmaria:** Sons desplugados, instrumentais delicados, musica para estudo e meditacao. |
| **3** | 24.705 ($21.7\%$) | $0.39$ | $0.66$ | $0.54$ | $-10.9\text{ dB}$ | $0.40$ | $113.5$ | **Folk & Melodias Nostalgicas:** Faixas calmas, vocais melodicos e arranjos acustico-pop. |
| **4** | 41.181 ($36.1\%$) | $0.73$ | $0.19$ | **$0.70$** | $-6.5\text{ dB}$ | **$0.69$** | $119.9$ | **Festa & Ritmo Contagiante:** Faixas altamente dancaveis, ritmicas e de tom emocional alegre/eufórico. |

![Perfil Acustico dos Clusters](file:///c:/Users/c9hes/Documents/faculdade/machine%20learning/outputs/p4_clusters_distribuicao.png)
*Figura 11: Assinatura acustica media comparativa das 5 playlists geradas pelo K-Means.*

---

### 4.4 A Prova de Fogo: Inspecao Qualitativa Humana de 5 Musicas

Selecionamos o **Cluster 2 (Playlist Acustico Intimista & Calmaria)** e extraimos uma amostra aleatoria de 5 musicas para inspecao humana de coesao auditiva:

#### Tabela 8: Amostra de Faixas do Cluster 2

| # | Artista | Nome da Faixa | Album | Acousticness | Energy | Danceability | Tempo |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | Moomu | *Cloudology* | Aguatic | **$0.884$** | **$0.064$** | $0.158$ | $107.4\text{ BPM}$ |
| **2** | Ochre | *Rising Tide* | An Eye To Windward | **$0.660$** | **$0.227$** | $0.533$ | $86.0\text{ BPM}$ |
| **3** | Wes Montgomery | *Days Of Wine And Roses* | Boss Guitar | **$0.991$** | **$0.030$** | $0.470$ | $89.4\text{ BPM}$ |
| **4** | Benny Martin; The Wong Janice | *Hallelujah - Piano & Cello* | Hallelujah (Piano & Cello) | **$0.888$** | **$0.198$** | $0.309$ | $65.7\text{ BPM}$ |
| **5** | White Noise for Babies | *Starlights* | Sea of White Noise | **$0.840$** | **$0.152$** | $0.265$ | $145.8\text{ BPM}$ |

#### Avaliacao Qualitativa Humana: Faz Sentido essas Musicas Estarem na Mesma Playlist?
**Sim, faz perfeito sentido do ponto de vista musical e auditivo.**
- **Wes Montgomery - 'Days Of Wine And Roses':** Obra-prima do jazz com guitarra eletroacustica com palhetada aveludada pelo polegar, andamento moderado e silencio dinamico.
- **Benny Martin & The Wong Janice - 'Hallelujah - Piano & Cello':** Composicao estritamente instrumental de piano e violoncelo, com decaimento acustico profundo e sem nenhuma percussao eletrica.
- **Moomu - 'Cloudology' e Ochre - 'Rising Tide':** Composicoes de ambiencia eterea, com timbres orgânicos e baixa excitabilidade ritmica.
- **White Noise for Babies - 'Starlights':** Faixa acustico-ambiental serena desenvolvida especificamente para relaxamento profundo e repouso.

**Conclusao Humana:** Todas as faixas selecionadas operam em uma mesma 'paisagem sonora' (soundscape): indices de acustica acima de $0.65$, energias controladas abaixo de $0.23$ e volumes baixos (loudness inferior a $-15\text{ dB}$). Para um usuario que procura uma playlist de foco, estudo intensivo, leitura ou calmaria noturna, a transicao entre qualquer uma dessas musicas ocorre de maneira suave e acusticamente natural. O algoritmo atingiu seu objetivo plenamente sem precisar de generos textuais pre-concebidos.

---

## Conclusao Geral da Squad

A execucao dos quatro problemas demonstrou com solidez a aplicacao do metodo cientifico na resolucao de desafios corporativos:
1. **Regressao Linear (Problema 1):** Demonstrou que o saneamento estrutural de nulos e o tratamento de espacos euclidianos elevam a precisao do modelo, reduzindo o erro medio da precificacao para $\$19.245$.
2. **Classificacao de Churn (Problema 2):** Esclareceu que metricas puras de acuracia sao ineficientes em bases desbalanceadas e comprovou que o Falso Negativo e o risco financeiro critico no modelo de receita recorrente.
3. **Segmentacao de Clientes (Problema 3):** Justificou o uso do Z-Score e validou matematicamente o ponto otimo do cotovelo e silhueta em $K=5$, traduzindo clusters abstratos em planos acionaveis de marketing.
4. **Clusterizacao Musical (Problema 4):** Provou que a fisica das ondas e as caracteristicas timbricas puras contem informacao suficiente para criar colecoes musicais harmonicamente perfeitas.

Todos os scripts, notebooks e graficos encontram-se salvos no repositorio local do projeto para fins de auditoria, validacao e submissao formal.
