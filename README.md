# Previsão de Casos de Dengue via Variáveis Climáticas — Ourinhos/SP

Este projeto de Ciência de Dados investiga a correlação entre fatores climáticos e a incidência de dengue no município de Ourinhos, São Paulo. O objetivo principal é desenvolver um modelo preditivo para auxiliar na gestão de recursos de saúde pública local.

## Contexto e Objetivos

A dengue é um problema de saúde pública sazonal em Ourinhos. Este projeto utiliza dados históricos (2015–2026) para:
- Identificar padrões climáticos (temperatura, umidade, pluviosidade) que antecedem surtos.
- Comparar o desempenho de diferentes modelos de machine learning na predição de novos casos.
- Gerar previsões para as próximas semanas epidemiológicas.
- Fornecer um **dashboard interativo** para visualização e tomada de decisão.

## Dados Utilizados

Os dados foram extraídos de fontes oficiais e processados para análise:
1. **Epidemiológicos:** Microdados de casos de dengue (SINAN) via DATASUS e estimativas do InfoDengue.
2. **Climáticos:** Séries históricas da estação automática **A716 (Ourinhos)** do INMET.

## Estrutura do Projeto

```
├── Dados_sinan.py                  # Extração e limpeza dos microdados SINAN
├── tratamento_dados_dengue.py      # Tratamento e junção dos dados climáticos + epidemiológicos
├── tratamento semanais 2.py        # Agregação semanal dos dados
├── _merge_infodengue.py            # Integração com dados do InfoDengue
├── Modelagem.py                    # Treinamento, comparação de modelos e previsão
├── dashboard.py                    # Dashboard interativo (Streamlit)
├── requirements.txt                # Dependências do projeto
├── Data/                           # Datasets tratados
│   ├── dataset_final_ourinhos.csv  # Dataset final usado na modelagem
│   ├── DENGBR*_OURINHOS.csv        # Dados SINAN filtrados por ano
│   └── ...
├── dados_inmet/                    # Dados brutos do INMET
├── Documentação/                   # Gráficos e imagens geradas
├── previsao_proximas_semanas.csv   # Previsões geradas pelo modelo
└── df_projeto.csv                  # Dataset intermediário do projeto
```

## Tecnologias e Metodologia

- **Linguagem:** Python
- **Modelagem:** Scikit-Learn (Random Forest), XGBoost (Poisson, Tweedie)
- **Dashboard:** Streamlit, Plotly, PyDeck
- **Tratamento de Dados:** Pandas, NumPy
- **Visualização:** Matplotlib, Seaborn, Plotly

### Modelos Comparados

| Modelo | Descrição |
|--------|-----------|
| Random Forest (RobustScaler) | Ensemble de árvores de decisão com escalonamento robusto a outliers |
| XGBoost Poisson | Gradient boosting com distribuição Poisson (ideal para contagens) |
| XGBoost Tweedie | Gradient boosting com distribuição Tweedie (lida com excesso de zeros) |

O pipeline seleciona automaticamente o melhor modelo com base no MAE (Erro Médio Absoluto) e gera previsões encadeadas para as **próximas 8 semanas epidemiológicas**.

### Features Utilizadas

- Variáveis climáticas: `chuva`, `temp_ar`, `chuva_lag_3`, `chuva_lag_4`, `temp_lag_4`
- Sazonalidade: `mes_seno` (codificação cíclica)
- Memória epidemiológica: `casos_lag_1`, `casos_lag_2`, `casos_mm4`
- Perfil etário: `idade_media`

## Resultados e Análises

O trabalho demonstrou que variáveis climáticas isoladas explicam apenas 32% da variação de casos de dengue em Ourinhos. A integração dos microdados do SINAN elevou esse valor para 73%, evidenciando que o clima prepara o terreno, mas é a memória epidemiológica que decide o surto.

O modelo performou bem em anos dentro do padrão histórico, com erro médio de apenas 13 casos por semana em 2023. Seu limite foi encontrado em 2025, o ano mais grave de dengue já registrado em Ourinhos (10.041 casos estimados), onde o modelo subestimou sistematicamente — atingindo erro máximo de 627 casos em uma única semana.

> **Nota:** Os microdados do SINAN de 2024 não estavam disponíveis no período de coleta. Para esse ano, o modelo utilizou exclusivamente variáveis climáticas e estimativas do InfoDengue.

Essa limitação delimita a fronteira de confiabilidade do modelo e aponta o caminho para trabalhos futuros: incorporação de dados de circulação viral, sorotipo e cobertura vacinal.

## Dashboard

O projeto inclui um **dashboard interativo** desenvolvido com Streamlit para visualização em tempo real dos dados e previsões:

```bash
streamlit run dashboard.py
```

O dashboard apresenta:
- Mapa de calor com localização de Ourinhos
- Evolução temporal dos casos de dengue
- Métricas e indicadores epidemiológicos
- Previsões para as próximas semanas

## Como Executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/GalardOnly/previsao-dengue-ourinhos.git
   cd previsao-dengue-ourinhos
   ```

2. Crie um ambiente virtual e instale as dependências:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```

3. Execute o pipeline de dados (se quiser refazer o tratamento):
   ```bash
   python Dados_sinan.py
   python tratamento_dados_dengue.py
   python "tratamento semanais 2.py"
   ```

4. Execute a modelagem e previsão:
   ```bash
   python Modelagem.py
   ```

5. Inicie o dashboard:
   ```bash
   streamlit run dashboard.py
   ```

## Autor

**Gabriel** — Estudante de Ciência de Dados, 5º Semestre
