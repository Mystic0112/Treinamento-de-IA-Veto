import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def gerar_relatorio_docx():
    doc = docx.Document()
    
    # Configuracao de margens (2.0 cm)
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)
        
    # Estilo Normal
    style_normal = doc.styles['Normal']
    font = style_normal.font
    font.name = 'Calibri'
    font.size = Pt(11)
    font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    
    # --- CAPA / CABECALHO ---
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_inst = title_p.add_run("INSTITUICAO DE ENSINO SUPERIOR\nCURSO DE TECNOLOGIA / CIENCIA DA COMPUTACAO\nDISCIPLINA: MACHINE LEARNING - AVALIACAO 1 (AV1)\n\n")
    r_inst.font.size = Pt(12)
    r_inst.font.bold = True
    r_inst.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    
    r_title = title_p.add_run("RELATORIO TECNICO E ANALITICO\nTRABALHO PRATICO: TRILHA DE MACHINE LEARNING\n")
    r_title.font.size = Pt(20)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D) # Azul institucional
    
    r_sub = title_p.add_run("\nResolucao Rigorosa dos Quatro Problemas de Negocio:\nAprendizado Supervisionado (Regressao e Classificacao) e Nao Supervisionado (Clusterizacao)\n")
    r_sub.font.size = Pt(12)
    r_sub.font.italic = True
    
    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_meta = p_meta.add_run("Squad de Inteligencia Artificial & Data Science\nEspecialista Tecnico: Dado (Data Scientist)\nOutubro de 2026 | Reprodutibilidade Garantida (Seed: 42)\n")
    r_meta.font.size = Pt(11)
    r_meta.font.bold = True
    
    doc.add_page_break()
    
    # --- SUMARIO EXECUTIVO ---
    h1 = doc.add_heading("1. Sumario Executivo", level=1)
    h1.style.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    
    p = doc.add_paragraph(
        "Este relatorio apresenta a documentacao integral e analise aprofundada dos quatro problemas propostos no Trabalho Pratico "
        "de Machine Learning (AV1). O projeto abrange desde a analise exploratoria, saneamento de dados com tratamento estatistico "
        "de missing values, engenharia de atributos com fundamentacao em algebra linear, ate a modelagem supervisionada e nao supervisionada, "
        "com interpretacao de negocios e avaliacao de metricas de erro e custo operacional."
    )
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.line_spacing = 1.15
    
    # Tabela Sintese
    t_sum = doc.add_table(rows=5, cols=5)
    t_sum.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Problema", "Paradigma", "Algoritmo", "Metricas Chave", "Conclusao Central"]
    for col_idx, text in enumerate(headers):
        cell = t_sum.cell(0, col_idx)
        cell.text = text
        set_cell_background(cell, "1F497D")
        for cp in cell.paragraphs:
            cp.runs[0].font.bold = True
            cp.runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            cp.runs[0].font.size = Pt(10)
            
    dados_sum = [
        ("1. O Preco Justo", "Supervisionado (Regressao)", "Linear Regression & Ridge", "R2 = 84.85%\nMAE = $19,245.03", "Qualidade, metragem e garagem explicam 84.8% dos precos."),
        ("2. Ameaca do Churn", "Supervisionado (Classificacao)", "Logistic Reg. & Random Forest", "ROC-AUC = 0.8422\nRecall = 55.88%", "Contratos mensais evadem 15.1x mais. FN e o erro critico."),
        ("3. Perfil do Consumidor", "Nao Supervisionado (Cluster)", "K-Means (K=5) + Scaler", "Inercia = 65.57\nSilhueta = 0.5547", "5 personas comerciais nitidas com planos acionaveis."),
        ("4. Playlist Perfeita", "Nao Supervisionado (Cluster)", "K-Means (K=5) / 9 Atributos", "Pearson r = -0.7339\n5 Playlists", "Isolamento de texto; coesao sonora acustica comprovada.")
    ]
    
    for row_idx, data in enumerate(dados_sum, 1):
        for col_idx, val in enumerate(data):
            cell = t_sum.cell(row_idx, col_idx)
            cell.text = val
            for cp in cell.paragraphs:
                cp.runs[0].font.size = Pt(9.5)
                
    doc.add_paragraph().paragraph_format.space_after = Pt(14)
    
    # --- PROBLEMA 1 ---
    h1 = doc.add_heading("2. Problema 1: Regressao Linear - 'O Preco Justo'", level=1)
    h1.style.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    
    doc.add_paragraph(
        "Contexto e Regra da Competicao: Uma imobiliaria contratou nossa equipe para desenvolver um modelo preditivo capaz de avaliar "
        "imoveis automaticamente. A base disponibilizada contem 1.455 imoveis (apos a remocao de 5 registros 'secretos' da competicao). "
        "A equipe que prever os precos com o menor Erro Absoluto Medio (MAE) sobre os dados de teste conquistara a pontuacao maxima."
    )
    
    # 2.1 Analise Exploratoria
    h2 = doc.add_heading("2.1 Analise Exploratoria: Top 3 Preditores Lineares", level=2)
    doc.add_paragraph(
        "Calculou-se o Coeficiente de Correlacao Linear de Pearson (r) entre todas as colunas numericas e o preco de venda (SalePrice). "
        "As tres variaveis com maior correlacao linear positiva sao:"
    )
    
    t_p1 = doc.add_table(rows=4, cols=4)
    t_p1.alignment = WD_TABLE_ALIGNMENT.CENTER
    p1_headers = ["Posicao", "Variavel", "Pearson r", "Significado Fisico e Tecnico"]
    for col_idx, text in enumerate(p1_headers):
        cell = t_p1.cell(0, col_idx)
        cell.text = text
        set_cell_background(cell, "2E75B6")
        for cp in cell.paragraphs:
            cp.runs[0].font.bold = True
            cp.runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            cp.runs[0].font.size = Pt(10)
            
    p1_rows = [
        ("1", "OverallQual", "+0.7918", "Avaliacao da qualidade geral do acabamento e materiais (1 a 10)."),
        ("2", "GrLivArea", "+0.7012", "Area habitavel construida acima do solo (em pes quadrados)."),
        ("3", "GarageCars", "+0.6430", "Capacidade da garagem para veiculos (seguida por GarageArea r=0.6272).")
    ]
    for row_idx, data in enumerate(p1_rows, 1):
        for col_idx, val in enumerate(data):
            cell = t_p1.cell(row_idx, col_idx)
            cell.text = val
            for cp in cell.paragraphs:
                cp.runs[0].font.size = Pt(9.5)
                
    doc.add_paragraph().paragraph_format.space_after = Pt(10)
    
    if os.path.exists('outputs/p1_correlacoes_heatmap.png'):
        doc.add_picture('outputs/p1_correlacoes_heatmap.png', width=Inches(5.5))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption = doc.add_paragraph("Figura 1: Matriz de Correlacao Linear de Pearson das top variaveis vs SalePrice.")
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption.runs[0].font.italic = True
        caption.runs[0].font.size = Pt(9.5)
        
    if os.path.exists('outputs/p1_scatter_top3.png'):
        doc.add_picture('outputs/p1_scatter_top3.png', width=Inches(6.2))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption = doc.add_paragraph("Figura 2: Dispersao e retas de regressao dos 3 principais preditores.")
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption.runs[0].font.italic = True
        caption.runs[0].font.size = Pt(9.5)
        
    # 2.2 Tratamento de Missing Values
    h2 = doc.add_heading("2.2 Tratamento Estatistico de Valores Ausentes", level=2)
    doc.add_paragraph(
        "1. Coluna Numerica: LotFrontage (Comprimento da Fachada/Testada do Terreno)\n"
        "- Diagnostico: 259 valores nulos (17.80% da base).\n"
        "- Tratamento: Imputacao pela mediana agrupada por bairro (Neighborhood).\n"
        "- Justificativa Estatistica: O tamanho da testada e determinado pelo zoneamento e padrao de loteamento do bairro. "
        "A mediana foi selecionada por ser um estimador robusto a assimetrias e presenca de outliers extremos imobiliarios.\n\n"
        "2. Coluna Categorica: BsmtQual (Qualidade e Altura do Porao)\n"
        "- Diagnostico: 37 valores nulos (2.54% da base).\n"
        "- Tratamento: Preenchimento com a categoria estrutural explicita 'None' ('Sem Porao').\n"
        "- Justificativa Estatistica: Conforme a documentacao oficial de Ames, o dado ausente nao e aleatorio (MCAR), mas decorre de "
        "Missingness Estrutural (MNAR), indicando que a residencia nao possui porao. Imputar a moda inventaria um porao inexistente, "
        "e apagar a linha eliminaria amostras legitimas. Criar o rotulo 'None' preserva a realidade fisica da edificacao."
    )
    
    # 2.3 Engenharia de Atributos
    h2 = doc.add_heading("2.3 Engenharia de Atributos e Necessidade Matematica da Conversao", level=2)
    doc.add_paragraph(
        "Foram transformadas as seguintes variaveis categoricas:\n"
        "1. BsmtQual via Mapeamento Ordinal: Atribuicao de pesos inteiros de 0 a 5 respeitando a escala de qualidade tecnica "
        "(Ex=5, Gd=4, TA=3, Fa=2, Po=1, None=0).\n"
        "2. Neighborhood via One-Hot Encoding: Criacao de variaveis indicadoras binarias com drop_first=True para evitar multicolinearidade estrita.\n\n"
        "Fundamentacao Matematica: Por que a Regressao Linear exige dados numericos?\n"
        "O estimador de Minimos Quadrados Ordinarios (OLS) resolve a equacao matricial:\n"
        "Y = X * beta + epsilon  ==>  beta_estimado = (X^T * X)^(-1) * X^T * Y\n"
        "A matriz de planejamento X deve residir estritamente no espaco vetorial real R^(n x (p+1)). Variaveis nominais em formato de texto "
        "nao possuem estrutura algebrica: nao suportam produto interno, inversao matricial nem multiplicacao escalar. A transformacao para "
        "coordenadas numericas e condicao matematica essencial para projetar os dados no hiperplano de regressao."
    )
    
    # 2.4 Baseline e Interpretacao do R2
    h2 = doc.add_heading("2.4 Treinamento, Metricas do Baseline e Interpretacao do R2", level=2)
    doc.add_paragraph(
        "A base foi dividida em 80% treino (1.164 imoveis) e 20% teste (291 imoveis) com random_state=42.\n"
        "- R2 no Conjunto de Teste: 0.8232 (82.32%)\n"
        "- MAE no Conjunto de Teste: $21,888.03\n"
        "- RMSE no Conjunto de Teste: $29,065.89\n"
        "- Modelo Refinado (Ridge Regression com Regularizacao L2 e Bairros): R2 = 0.8485 (84.85%) e MAE = $19,245.03 "
        "(reducao de $2,643.00 no erro absoluto medio, maximizando a chance de vitoria na competicao).\n\n"
        "Significado do R2 para a Imobiliaria:\n"
        "O coeficiente de determinacao R2 = 82.32% significa que mais de 82% de toda a dispersao observada nos precos de venda dos imoveis "
        "e confiavelmente explicada pelas caracteristicas fisicas e construtivas do imovel (qualidade, metragem privativa e vagas). "
        "Para a operacao de negocios, isso garante que o algoritmo fornece uma avaliacao solida e objetiva, restando menos de 18% para "
        "variacoes subjetivas de negociacao."
    )
    
    if os.path.exists('outputs/p1_residuos_dispersao.png'):
        doc.add_picture('outputs/p1_residuos_dispersao.png', width=Inches(6.0))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption = doc.add_paragraph("Figura 3: Diagnostico de residuos e comparativo Real vs Previsto.")
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption.runs[0].font.italic = True
        caption.runs[0].font.size = Pt(9.5)
        
    doc.add_page_break()
    
    # --- PROBLEMA 2 ---
    h1 = doc.add_heading("3. Problema 2: Classificacao - 'A Ameaca do Churn'", level=1)
    h1.style.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    
    doc.add_paragraph(
        "Contexto: Uma operadora de telecomunicacoes busca identificar clientes com alta propensao ao cancelamento (Churn) "
        "para agir preventivamente com acoes de retencao."
    )
    
    # 3.1 Compreensao do Negocio
    h2 = doc.add_heading("3.1 Compreensao do Negocio e Diagnostico de Desbalanceamento", level=2)
    doc.add_paragraph(
        "A variavel dependente e Churn ('Yes' = 1.869 clientes / 26.54%; 'No' = 5.174 clientes / 73.46%).\n"
        "Diagnostico: Trata-se de uma base moderadamente desbalanceada (razao de 2.77 para 1). "
        "Em cenarios desbalanceados, a acuracia isolada e enganosa (um modelo que chute 'No' sempre teria 73.5% de acuracia nominal "
        "sem identificar nenhuma evasao). Por isso, as metricas mandatorias sao Recall, Precision, F1-Score e ROC-AUC."
    )
    
    if os.path.exists('outputs/p2_distribuicao_churn.png'):
        doc.add_picture('outputs/p2_distribuicao_churn.png', width=Inches(5.8))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption = doc.add_paragraph("Figura 4: Distribuicao de frequencias da variavel Churn.")
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption.runs[0].font.italic = True
        caption.runs[0].font.size = Pt(9.5)
        
    # 3.2 Analise Contratual
    h2 = doc.add_heading("3.2 Analise de Risco Contratual: Mensal vs Anual", level=2)
    doc.add_paragraph(
        "Cruzando os dados de contrato (Contract) com a evasao:\n"
        "- Contratos Mensais (Month-to-month): 42.71% de taxa de evasao (1.655 de 3.875 clientes).\n"
        "- Contratos de 1 Ano (One year): 11.27% de evasao (166 de 1.473 clientes).\n"
        "- Contratos de 2 Anos (Two year): 2.83% de evasao (48 de 1.695 clientes).\n"
        "Conclusao: Clientes no modelo mensal evadem quase 4 vezes mais que os contratantes anuais e mais de 15 vezes mais "
        "que clientes em planos de 2 anos. O contrato mensal representa o ponto mais vulneravel da carteira de clientes."
    )
    
    if os.path.exists('outputs/p2_churn_por_contrato.png'):
        doc.add_picture('outputs/p2_churn_por_contrato.png', width=Inches(6.0))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption = doc.add_paragraph("Figura 5: Analise comparativa de taxa de churn por tipo de contrato.")
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption.runs[0].font.italic = True
        caption.runs[0].font.size = Pt(9.5)
        
    # 3.3 Tratamento TotalCharges
    h2 = doc.add_heading("3.3 Preparacao dos Dados e Diagnostico de TotalCharges", level=2)
    doc.add_paragraph(
        "A coluna TotalCharges foi fornecida como texto e continha exatamente 11 registros com espacos em branco (' '). "
        "Ao investigar esses 11 clientes, verificou-se que todos possuem tempo de permanencia (tenure) igual a zero meses. "
        "Sao clientes recem-contratados no ciclo inaugural de cobranca, cujo primeiro fechamento de fatura ainda nao ocorreu. "
        "A coercao foi realizada via pd.to_numeric com imputacao de 0.0, preservando os clientes no estudo sem perda de dados."
    )
    
    # 3.4 Treinamento e Analise de Custos
    h2 = doc.add_heading("3.4 Matriz de Confusao e Trade-off de Custos de Negocio", level=2)
    doc.add_paragraph(
        "A base foi dividida em 80/20 estratificada e treinamos a Regressao Logistica (ROC-AUC = 0.8422, Acuracia = 80.62%, Recall = 55.88%) "
        "e Random Forest (ROC-AUC = 0.8442).\n\n"
        "No contexto desta empresa de telecomunicacoes, o que e pior: Falso Positivo ou Falso Negativo?\n"
        "O Falso Negativo (FN = 165 casos) e infinitamente pior e mais prejudicial para a operacao:\n"
        "1. Falso Negativo: O modelo preve que o cliente esta satisfeito e nao cancelara. A empresa nao faz nenhuma intervencao; "
        "o cliente cancela e vai para a concorrencia. O prejuizo e maximo: perda do Lifetime Value (LTV) e necessidade de incorrer no "
        "Custo de Aquisicao de Clientes (CAC), que no setor de telecom e de 5 a 7 vezes mais caro do que manter um cliente retido.\n"
        "2. Falso Positivo (FP = 108 casos): O modelo preve evasao de quem permaneceria. O time concede um desconto ou promocao preventiva. "
        "O custo e marginal e controlado, com o beneficio colateral de aumentar a satisfacao e a lealdade do cliente.\n\n"
        "Diretriz Estrategica: A operadora deve priorizar o Recall da classe Churn, ajustando o limiar de corte para minimizar os Falsos Negativos."
    )
    
    if os.path.exists('outputs/p2_matriz_confusao.png'):
        doc.add_picture('outputs/p2_matriz_confusao.png', width=Inches(6.0))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption = doc.add_paragraph("Figura 6: Matriz de Confusao da Regressao Logistica e Random Forest nos dados de teste.")
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption.runs[0].font.italic = True
        caption.runs[0].font.size = Pt(9.5)

    if os.path.exists('outputs/p2_curva_roc.png'):
        doc.add_picture('outputs/p2_curva_roc.png', width=Inches(5.6))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption = doc.add_paragraph("Figura 7: Curva ROC dos classificadores Regressao Logistica (AUC = 0.842) e Random Forest (AUC = 0.844).")
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption.runs[0].font.italic = True
        caption.runs[0].font.size = Pt(9.5)
        
    doc.add_page_break()
    
    # --- PROBLEMA 3 ---
    h1 = doc.add_heading("4. Problema 3: Clusterizacao - 'O Perfil do Consumidor'", level=1)
    h1.style.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    
    doc.add_paragraph(
        "Contexto: Segmentacao nao supervisionada dos 200 clientes do Shopping Center com base em Annual Income (k$) "
        "e Spending Score (1-100) para acoes de marketing direcionadas."
    )
    
    h2 = doc.add_heading("4.1 Justificativa Matematica do Escalonamento (StandardScaler)", level=2)
    doc.add_paragraph(
        "O algoritmo K-Means minimiza a soma das distancias euclidianas quadraticas intra-cluster: "
        "J = sum sum ||x_i - mu_k||^2. Como as amplitudes de grandezas diferem fortemente (renda de $15k a $137k vs score de 1 a 100), "
        "a variavel de maior variancia dominaria o calculo euclidiano, distorcendo os grupos. O StandardScaler (Z-score) normaliza "
        "ambas para media 0 e desvio padrao 1, assegurando isometria espacial."
    )
    
    h2 = doc.add_heading("4.2 Metodo do Cotovelo e Coeficiente de Silhueta", level=2)
    doc.add_paragraph(
        "Executou-se o algoritmo para K de 1 a 10. O Metodo do Cotovelo registrou o ponto de inflexao evidente em K = 5 "
        "(inercia reduzida para 65.57). O Coeficiente de Silhueta atingiu seu pico maximo global exatamente em K = 5 com score de 0.5547, "
        "comprovando matematicamente que 5 e a quantidade ideal de clusters."
    )
    
    if os.path.exists('outputs/p3_elbow_silhouette.png'):
        doc.add_picture('outputs/p3_elbow_silhouette.png', width=Inches(6.0))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption = doc.add_paragraph("Figura 8: Curvas do Cotovelo (WCSS) e Coeficiente de Silhueta para K de 1 a 10.")
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption.runs[0].font.italic = True
        caption.runs[0].font.size = Pt(9.5)
        
    h2 = doc.add_heading("4.3 Interpretacao de Negocios dos 5 Perfis Comerciais", level=2)
    
    t_mall = doc.add_table(rows=6, cols=5)
    t_mall.alignment = WD_TABLE_ALIGNMENT.CENTER
    m_headers = ["Cluster", "Renda Media", "Score Medio", "Rotulo Comercial", "Estrategia de Marketing"]
    for col_idx, text in enumerate(m_headers):
        cell = t_mall.cell(0, col_idx)
        cell.text = text
        set_cell_background(cell, "70AD47")
        for cp in cell.paragraphs:
            cp.runs[0].font.bold = True
            cp.runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            cp.runs[0].font.size = Pt(9.5)
            
    m_rows = [
        ("0", "$55.3k", "49.5", "Consumidores Moderados / Padrao", "Programas de lealdade continua e promocoes sazonais."),
        ("1", "$86.5k", "82.1", "Clientes VIP / Alto Padrao", "Servico de concierge, marcas de luxo e eventos exclusivos."),
        ("2", "$25.7k", "79.4", "Jovens Impulsivos / Entusiastas", "Crediario facil, microcredito e moda fast-fashion via redes."),
        ("3", "$88.2k", "17.1", "Poupadores Cautelosos / Racionais", "Foco em durabilidade, tecnologia de ponta e alto valor."),
        ("4", "$26.3k", "20.9", "Consumidores Economicos / Conscientes", "Cupons de desconto, queima de estoque e foco estrito em preco.")
    ]
    for row_idx, data in enumerate(m_rows, 1):
        for col_idx, val in enumerate(data):
            cell = t_mall.cell(row_idx, col_idx)
            cell.text = val
            for cp in cell.paragraphs:
                cp.runs[0].font.size = Pt(9.0)
                
    doc.add_paragraph().paragraph_format.space_after = Pt(10)
    
    if os.path.exists('outputs/p3_clusters_mall_customers.png'):
        doc.add_picture('outputs/p3_clusters_mall_customers.png', width=Inches(6.0))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption = doc.add_paragraph("Figura 9: Dispersao 2D dos 5 Clusters de clientes e seus centroides.")
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption.runs[0].font.italic = True
        caption.runs[0].font.size = Pt(9.5)
        
    doc.add_page_break()
    
    # --- PROBLEMA 4 ---
    h1 = doc.add_heading("5. Problema 4: Clusterizacao Musical - 'Criando a Playlist Perfeita'", level=1)
    h1.style.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    
    doc.add_paragraph(
        "Contexto: O aplicativo de streaming de musica necessita criar 5 playlists automaticas baseadas exclusivamente em propriedades "
        "acusticas contínuas de 114.000 musicas do Spotify, sem utilizacao dos generos musicais textuais."
    )
    
    h2 = doc.add_heading("5.1 Isolamento de Metadados Textuais", level=2)
    doc.add_paragraph(
        "Colunas como nome da musica, artista e id da track sao categoricas nominais que nao possuem ordenacao metrica no espaco euclidiano R^d. "
        "Atribuir numeros arbitrarios a textos introduziria distancias ficticias sem correspondencia com a fisica das ondas sonoras. "
        "Por isso, os metadados foram isolados em tabela de consulta (lookup), alimentando o modelo puramente com os 9 atributos continuos."
    )
    
    h2 = doc.add_heading("5.2 Correlacao Acustica: Energy vs Acousticness (r = -0.7339)", level=2)
    doc.add_paragraph(
        "Constatou-se uma correlacao fortemente negativa (Pearson r = -0.7339). "
        "Explicacao Fisica e Musical: Faixas de alta energia requerem amplificacao eletrica, compressao dinamica pesada e sintetizadores (Rock, EDM). "
        "Em contraste, gravacoes acusticas preservam o timbre natural de fontes mecanicas (violao, piano classico, voz crua), com transientes suaves "
        "e menor densidade energetica. Trata-se de uma relacao mutuamente excludente na engenharia acustica de som."
    )
    
    if os.path.exists('outputs/p4_correlacao_energy_acousticness.png'):
        doc.add_picture('outputs/p4_correlacao_energy_acousticness.png', width=Inches(6.0))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption = doc.add_paragraph("Figura 10: Relacao inversa entre Energy e Acousticness (r = -0.7339) e matriz completa.")
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption.runs[0].font.italic = True
        caption.runs[0].font.size = Pt(9.5)
        
    h2 = doc.add_heading("5.3 Treinamento do K-Means (5 Playlists) e a Prova de Fogo", level=2)
    doc.add_paragraph(
        "Com K=5, o modelo segmentou as 114.000 musicas em 5 ambientes sonoros (Acustico Intimista, Treino Adrenalina, Festa Dancavel, "
        "Folk Nostalgico e Pop Urbano).\n\n"
        "A Prova de Fogo: Extraimos 5 musicas pertencentes ao Cluster 2 (Playlist Acustico Intimista & Calmaria):\n"
        "1. Wes Montgomery - 'Days Of Wine And Roses' (Acousticness: 0.991, Energy: 0.030) - Jazz classico com guitarra acustica aveludada.\n"
        "2. Benny Martin & The Wong Janice - 'Hallelujah - Piano & Cello' (Acousticness: 0.888, Energy: 0.198) - Peca instrumental de piano e violoncelo.\n"
        "3. Moomu - 'Cloudology' (Acousticness: 0.884, Energy: 0.064) - Arranjo etereo e sereno.\n"
        "4. White Noise for Babies - 'Starlights' (Acousticness: 0.840, Energy: 0.152) - Faixa ambiente relaxante para sono.\n"
        "5. Ochre - 'Rising Tide' (Acousticness: 0.660, Energy: 0.227) - Composicao de andamento lento e timbres organicos.\n\n"
        "Avaliacao Humana: Faz total sentido estarem na mesma playlist. Todas as faixas compartilham acousticness > 0.65, energia < 0.25 "
        "e volumes baixos (loudness < -15 dB). O ouvinte que busca foco, leitura ou repouso desfruta de perfeita continuidade auditiva, "
        "sem oscilacoes ruidosas de ritmo ou potencia."
    )
    
    if os.path.exists('outputs/p4_clusters_distribuicao.png'):
        doc.add_picture('outputs/p4_clusters_distribuicao.png', width=Inches(6.0))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption = doc.add_paragraph("Figura 11: Comparativo das assinaturas acusticas medias dos 5 clusters.")
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption.runs[0].font.italic = True
        caption.runs[0].font.size = Pt(9.5)
        
    caminho_docx = 'outputs/Relatorio_Trabalho_Pratico_AV1_ML.docx'
    doc.save(caminho_docx)
    print(f"Relatorio DOCX gerado e salvo com sucesso em: {caminho_docx}")

if __name__ == '__main__':
    gerar_relatorio_docx()
