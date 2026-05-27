# Previsão de Surtos de Dengue em Ourinhos-SP

Modelo de previsão semanal de casos de dengue integrando dados epidemiológicos do SINAN com séries climáticas do INMET. Trabalho de Conclusão de Curso em Ciência de Dados — FATEC Ourinhos.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![scikit--learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-yellow) ![XGBoost](https://img.shields.io/badge/XGBoost-Poisson%20%2F%20Tweedie-orange) ![Pandas](https://img.shields.io/badge/Pandas-Data%20Pipeline-green) ![License](https://img.shields.io/badge/license-MIT-green)

📄 **Documentação completa:** [`Documentação/`](./Documentação)

---

## TL;DR

Três modelos comparados em conjunto de teste temporal (118 semanas, sem shuffle):

| Modelo | R² | MAE | RMSE | MAPE |
|---|---|---|---|---|
| **Random Forest (RobustScaler)** | **0,731** | **53,65** | **120,31** | 47,8% |
| XGBoost Poisson | 0,605 | 62,79 | 145,91 | 47,5% |
| XGBoost Tweedie | 0,598 | 64,04 | 147,14 | 48,3% |

> **Achado central:** clima isolado explica apenas 32% da variação semanal de casos. A integração de lags de casos do SINAN eleva esse valor para 73% — confirmando que *o clima prepara o terreno, mas é a memória epidemiológica que decide o surto*.

---

## Demonstração

### Casos reais vs preditos — 3 modelos comparados

![Série temporal de casos reais vs preditos](./docs/serie_temporal.png)

O conjunto de teste cobre 2024 a 2026, incluindo o surto histórico de 2025. Os três modelos acompanham bem períodos de baixa transmissão e o surto de 2024, mas todos subestimam sistematicamente o pico de 2025 (real ~1080 casos/semana, predito ~580 pelo melhor modelo). Esta limitação é discutida na seção [Limitações](#limitações-e-honestidade-do-modelo).

### Importância das variáveis

![Importância das features](./docs/feature_importance.png)

Variáveis derivadas do histórico de casos (`casos_lag_1`, `casos_mm4`) respondem por **~89% da importância total**. Variáveis climáticas isoladas têm contribuição marginal — coerente com a literatura: dengue tem forte componente autoregressivo.

---

## O Problema

A dengue é um problema de saúde pública sazonal em Ourinhos-SP. Apesar do conhecimento epidemiológico geral sobre a doença (clima quente e úmido favorece o vetor *Aedes aegypti*), gestores municipais carecem de ferramentas preditivas calibradas para a realidade local que orientem alocação de recursos de combate ao mosquito, campanhas de prevenção e dimensionamento da rede de saúde.

A maior parte dos modelos públicos existentes (como o InfoDengue) opera em escala estadual ou nacional, perdendo o sinal específico de municípios médios como Ourinhos. Este TCC investiga se é possível construir, com dados públicos disponíveis, um modelo útil em escala municipal.

---

## A Solução

Pipeline preditivo semanal integrando duas fontes oficiais:

1. **Microdados do SINAN** via DATASUS/InfoDengue — notificações de casos
2. **Séries climáticas do INMET** — estação automática **A716 (Ourinhos)**, com temperatura, umidade, pluviosidade e pressão

A engenharia de atributos centrou em **defasagens temporais (lags)** das duas fontes e em **encoding cíclico de sazonalidade** (`mes_seno`, `mes_cosseno`) — garantindo que dezembro e janeiro fiquem próximos no espaço de features, em vez de serem tratados como meses distantes.

### Pipeline

```
SINAN/InfoDengue          INMET (estação A716)
       │                          │
       ▼                          ▼
   tratamento_dados_dengue.py     tratamento semanais 2.py
   (parsing, agregação semanal,   (limpeza, agregação semanal,
    enriquecimento via InfoDengue) derivação de features climáticas)
       │                          │
       └──────────┬───────────────┘
                  ▼
            Dataset semanal
            (casos + clima + lags + mes_seno/cosseno)
                  │
                  ▼
              Modelagem.py
        ┌─────────┴─────────┐
        ▼                   ▼
   RobustScaler        Validação temporal
        │              (80/20 sem shuffle)
        ▼
  ┌────────────┬──────────────┬──────────────┐
  ▼            ▼              ▼              ▼
Random      XGBoost        XGBoost      Métricas +
Forest      Poisson        Tweedie       gráficos
(n=500)   (count:poisson) (reg:tweedie)
```

---

## Dados Utilizados

| Fonte | Origem | Granularidade | Período |
|---|---|---|---|
| **Casos de dengue (target = `casos_est`)** | SINAN via DATASUS + InfoDengue (nowcasting) | Semanal, municipal | 2014 → 2026 |
| **Clima** | INMET — estação automática A716 (Ourinhos) | Horária, agregada para semanal | Mesmo período |

**Sobre a escolha do alvo:** o modelo prevê `casos_est` (estimativa nowcasting do InfoDengue), e não `casos_reais` do SINAN diretamente. Esta decisão foi consciente: os microdados do SINAN para Ourinhos em 2024 não estavam consolidados durante a coleta, e usar `casos_est` permite cobertura contínua até 2026. Na prática, o modelo aprende a antecipar a estimativa que o InfoDengue produzirá, a partir de clima + lags de casos.

**Features finais (10):**
- Climáticas: `chuva`, `temp_ar`, `chuva_lag_3`, `chuva_lag_4`, `temp_lag_4`
- Sazonalidade: `mes_seno`
- Histórico de casos: `casos_lag_1`, `casos_lag_2`, `casos_mm4` (média móvel de 4 semanas)
- Demográfica: `idade_media`

---

## Metodologia

### Engenharia de atributos

- **Agregação semanal** das séries climáticas (médias e somas conforme a variável)
- **Defasagens temporais (lags)** de até 4 semanas em casos e clima
- **Média móvel de 4 semanas** (`casos_mm4`) para capturar tendência de curto prazo
- **Encoding cíclico** de sazonalidade com `sen`/`cos` do mês

### Modelos comparados

Três modelos foram testados deliberadamente, cada um motivado por uma hipótese estatística sobre os dados:

| Modelo | Motivação |
|---|---|
| **Random Forest** (n=500, RobustScaler) | Robusto a outliers, captura interações não-lineares, fornece importância interpretável |
| **XGBoost Poisson** (`count:poisson`) | Dados de contagem (casos) seguem distribuição de Poisson melhor que normal — adequação teórica esperada |
| **XGBoost Tweedie** (`reg:tweedie`, `power=1.5`) | Generalização da Poisson que lida com excesso de zeros (períodos de baixa transmissão) |

### Validação

Divisão **80/20 com `shuffle=False`** — separação cronológica entre treino (468 semanas) e teste (118 semanas), respeitando a natureza de série temporal dos dados. **Não há k-fold aleatório** porque embaralhar tempo causa data leakage (modelo veria "futuro" treinando).

O código contém verificação explícita de leakage por sobreposição de índices ao final de `Modelagem.py`.

---

## Resultados

### Comparação entre modelos

| Modelo | R² | MAE | RMSE | MAPE |
|---|---|---|---|---|
| **Random Forest (RobustScaler)** | **0,731** | **53,65** | **120,31** | 47,8% |
| XGBoost Poisson | 0,605 | 62,79 | 145,91 | 47,5% |
| XGBoost Tweedie | 0,598 | 64,04 | 147,14 | 48,3% |

**Observação contraintuitiva:** os modelos com função objetivo teoricamente mais adequada para contagem (Poisson, Tweedie) tiveram desempenho **inferior** ao Random Forest padrão. Uma hipótese é que a regularização das funções Poisson/Tweedie tende a achatar predições nos extremos para otimizar log-likelihood — e em surtos epidêmicos, *os extremos são justamente a informação relevante*. O Random Forest, sem essa regularização, captura melhor os picos.

### Predito vs Real (Random Forest)

![Predito vs Real](./docs/predito_vs_real.png)

A maior parte dos pontos se concentra próxima da diagonal, especialmente em valores baixos (períodos de baixa transmissão). Para valores acima de ~600 casos/semana o modelo subestima sistematicamente, formando um "achatamento" abaixo da diagonal — comportamento característico de modelos de árvore em regiões extrapoladas.

### Achado central

Clima isolado (temperatura, umidade, pluviosidade) explica apenas **~32%** da variação semanal de casos em Ourinhos quando testado em modelos prévios. A integração dos lags de casos do SINAN eleva esse valor para **0,731 (R²)** — um salto de mais de **2x** em poder explicativo.

A interpretação substantiva é que o clima **viabiliza** a ocorrência da dengue (cria condições para o vetor) mas não a **determina** isoladamente. O preditor mais forte para a quantidade de casos na semana N é a quantidade de casos nas semanas anteriores (`casos_lag_1` = 56,2% da importância; `casos_mm4` = 33,2%). Em termos epidemiológicos: vírus em circulação é o que sustenta surtos.

### Análise de resíduos

![Análise de resíduos](./docs/residuos.png)

A análise revelou três achados importantes:

1. **Distribuição assimétrica** — o histograma mostra concentração de resíduos próximos a zero (modelo acerta a maior parte do tempo) com cauda longa à direita (subestimação em surtos).
2. **Não-normalidade** — o QQ-plot evidencia desvio claro da reta diagonal nos extremos superiores, especialmente acima de +200 casos.
3. **Heterocedasticidade** — o gráfico de resíduos vs preditos forma um padrão de "funil": variância dos erros cresce com o valor predito. Isto é **esperado** em dados de contagem, onde a variância depende da média (propriedade Poisson-like). Esta característica motivou justamente o teste de XGBoost com `objective` adequado para contagem.

**Estatísticas dos resíduos (Random Forest):**

| Métrica | Valor |
|---|---|
| Resíduo médio | 40,26 (modelo tende a subestimar) |
| Desvio padrão | 113,38 |
| Resíduo máximo | 612,84 (semana de pico em 2025) |
| Resíduo mínimo | -118,15 |

---

## Limitações e Honestidade do Modelo

O modelo performou bem em períodos dentro do padrão histórico observado durante o treino, com a maioria dos resíduos abaixo de 100 casos/semana.

Seu limite foi encontrado em **2025**, o ano mais grave de dengue já registrado em Ourinhos: **10.041 casos estimados**, superando o pico histórico anterior de 6.545 casos em 2015. Diante de um surto sem precedente no período de treino e da ausência dos microdados do SINAN de 2024 totalmente consolidados, o modelo subestimou sistematicamente o pico, atingindo erro máximo de **~613 casos em uma única semana** (real ~1080, predito ~470).

Essa limitação não invalida o modelo — ela delimita sua fronteira de confiabilidade e aponta direções para trabalhos futuros:

- **Incorporação de dados de circulação viral e sorotipo** (DENV-1, DENV-2, etc.) para capturar mudanças de sorotipo dominante, fator conhecido de surtos atípicos
- **Cobertura vacinal** após implementação do programa público de vacinação contra dengue
- **Atualização contínua com microdados do SINAN** assim que liberados pelo DATASUS
- **Variáveis socioeconômicas e de saneamento** (cobertura de coleta de lixo, infestação predial)
- **Comparação com baseline naïve** (prever semana N = semana N-1) para mensurar ganho real do modelo sobre persistência autoregressiva
- **Modelos sequenciais** (LSTM, Prophet, ARIMAX) para comparação com a abordagem de árvores

---

## Aplicação Prática

Além do modelo preditivo, o projeto entrega:

- **Previsão encadeada de N semanas à frente** com intervalo de confiança via `iteration_range` do XGBoost, classificando a confiança da predição em alta (semanas 1-2), média (3-4) e baixa (5+) à medida que o erro acumula.
- **Relatório PDF automatizado para a Secretaria de Saúde** ([reportlab](https://www.reportlab.com/)) com situação atual (Rt, nível de incidência), série histórica, previsão das próximas semanas e nota metodológica.

---

## Como Executar

```bash
# 1. Clone o repositório e instale dependências
git clone https://github.com/GalardOnly/previsao-dengue-ourinhos.git
cd previsao-dengue-ourinhos
pip install -r requirements.txt

# 2. Execute o tratamento dos dados
python tratamento_dados_dengue.py
python "tratamento semanais 2.py"

# 3. Execute a modelagem (treina os 3 modelos, gera previsão e PDF)
python Modelagem.py

# 4. (Opcional) Regere os gráficos do README
python gerar_graficos_e_metricas.py
```

---

## Estrutura do Repositório

```
previsao-dengue-ourinhos/
├── Data/                          # Datasets processados
│   └── dataset_final_ourinhos.csv # Dataset consolidado (casos + clima + lags)
├── dados_inmet/                   # Séries climáticas brutas do INMET
├── Documentação/                  # Documentação técnica do TCC
├── docs/                          # Gráficos do README
├── tratamento_dados_dengue.py     # ETL dos dados SINAN/InfoDengue
├── tratamento semanais 2.py       # ETL e agregação dos dados climáticos
├── Dados_sinan.py                 # Utilitários de extração SINAN
├── Modelagem.py                   # Treinamento, comparação, previsão, PDF
├── gerar_graficos_e_metricas.py   # Gera gráficos do README
├── requirements.txt
└── README.md
```

---

## Stack

- **Linguagem**: Python 3.10+
- **Processamento**: Pandas, NumPy
- **Modelagem**: scikit-learn (Random Forest, RobustScaler), XGBoost (Poisson, Tweedie)
- **Visualização**: Matplotlib, Seaborn
- **Relatório**: ReportLab
- **Calendário epidemiológico**: epiweeks
- **Fontes de dados**: SINAN/DATASUS, InfoDengue, INMET

---

## Autor

**Gabriel Costuchenco**
Estudante de Ciência de Dados — FATEC Ourinhos (conclusão Dez/2026)

- 💼 [LinkedIn](https://www.linkedin.com/in/gabrielcostuchenco)
- 📧 gabrielolivcos8@gmail.com
- 🐙 [GitHub](https://github.com/GalardOnly)

---

## Licença

MIT — veja [LICENSE](./LICENSE) para detalhes.
