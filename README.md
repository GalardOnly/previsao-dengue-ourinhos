# Previsão de Casos de Dengue via Variáveis Climáticas: Estudo de Caso Ourinhos-SP

Este projeto de Ciência de Dados investiga a correlação entre fatores climáticos e a incidência de dengue no município de Ourinhos, São Paulo. O objetivo principal é desenvolver um modelo preditivo para auxiliar na gestão de recursos de saúde pública local.

## Contexto e Objetivos
A dengue é um problema de saúde pública sazonal em Ourinhos. Este projeto utiliza dados históricos para:
- Identificar padrões climáticos (temperatura, umidade, pluviosidade) que antecedem surtos.
- Comparar o desempenho de modelos de regressão na predição de novos casos.
- Fornecer visualizações claras para tomada de decisão estratégica.

## Dados Utilizados
Os dados foram extraídos de fontes oficiais e processados para análise:
1. **Epidemiológicos:** Dados de casos de dengue (SINAN) via plataforma DATASUS e InfoDengue.
2. **Climáticos:** Séries históricas da estação automática **A716 (Ourinhos)** do INMET.

## Tecnologias e Metodologia
- **Linguagem:** Python
- **Bibliotecas Principais:** Pandas (Tratamento), Scikit-Learn (Modelagem), Matplotlib/Seaborn (Visualização).
- **Abordagem:** O projeto seguiu as etapas de Limpeza de Dados (Data Cleaning), Engenharia de Atributos (Feature Engineering), Treinamento de Modelos e Análise de Erro (Resíduos).


## Resultados e Análises
Abaixo, algumas das análises geradas durante o desenvolvimento do modelo:

O trabalho demonstrou que variáveis climáticas isoladas explicam apenas 32% da variação de casos de dengue em Ourinhos. A integração dos microdados do SINAN elevou esse valor para 73%, evidenciando que o clima prepara o terreno, mas é a memória epidemiológica que decide o surto. Um sistema de alerta precoce para dengue em Ourinhos precisa necessariamente integrar dados climáticos históricos com o histórico recente de notificações nenhuma das duas fontes isoladas é suficiente. O modelo performou bem em anos dentro do padrão histórico, com erro médio de apenas 13 casos por semana em 2023. Vale ressaltar que os microdados do SINAN referentes ao município de Ourinhos para o ano de 2024 não estavam disponíveis para download no período de coleta deste trabalho, em razão do atraso natural de consolidação do sistema. Para este período, o modelo utilizou exclusivamente as variáveis climáticas e os casos estimados pelo info.dengue como base preditiva. Seu limite foi encontrado em 2025, o ano mais grave de dengue já registrado em Ourinhos, com 10.041 casos estimados, superando o pico histórico anterior de 6.545 casos em 2015. Diante de um surto sem precedente no período de treino e da ausência dos dados do SINAN de 2024, o modelo subestimou sistematicamente, atingindo erro máximo de 627 casos em uma única semana. Essa limitação não invalida o modelo — ela delimita com precisão sua fronteira de confiabilidade e aponta o caminho para trabalhos futuros: a incorporação de dados de circulação viral, sorotipo e cobertura vacinal, além da atualização contínua com microdados do SINAN, seria o próximo passo para capturar surtos de magnitude histórica.

### Análise de Resíduos
A análise de resíduos foi fundamental para validar a homocedasticidade e a normalidade dos erros do modelo preditivo.

##  Como Executar
1. Instale as dependências: `pip install -r requirements.txt`
2. Execute o tratamento de dados: `python src/tratamento_dados_dengue.py`
3. Execute o script de modelagem: `python src/modelagem.py`

##  Autor
**Gabriel** *Estudante de Ciência de Dados - 5º Semestre*
