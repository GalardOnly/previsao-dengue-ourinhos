import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

data_treino = pd.read_csv('dataset_final_ourinhos.csv', sep=';', encoding='latin-1')

#convertendo a coluna Data_Inicio_Semanas para datetime e  usando o .dt.month para extrair o mes e criar a coluna "mes" para usar na criação das colunas seno e cosseno
data_treino['Data_Inicio_Semanas'] = pd.to_datetime(data_treino['Data_Inicio_Semanas'], format='%Y-%m-%d')
data_treino['mes'] = data_treino['Data_Inicio_Semanas'].dt.month

#criando colunas mes_seno mes_cosseno para variação de sazonalidade, usando a função para criar uma representação onde 1 e 12 ficam proximos 
#para que o modelo entenda as estações do ano
data_treino['mes_seno'] = np.sin(2 * np.pi * data_treino['mes'] / 12)
data_treino['mes_cosseno'] = np.cos(2 * np.pi * data_treino['mes'] / 12)

data_treino = data_treino.sort_values(['ano', 'Semana_Epidemiologica']).reset_index(drop=True)

#  Seleção de colunas e limpeza de nulos 
features = [
    'chuva', 'temp_ar', 'chuva_lag_3', 'chuva_lag_4', 'temp_lag_4', 'mes_seno',
    'casos_lag_1', 'casos_lag_2', 'casos_mm4','idade_media'             
]
target = 'casos_est'

data_treino = data_treino.dropna(subset=features + [target])

X = data_treino[features]
y = data_treino[target]


# Divisão (Para o TCC, o ideal é manter a ordem cronológica, 
# mas manter o shuffle agora para testar o código, use shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

#utilizando o robust scaler para lidar com outliers
robust_scaler = RobustScaler()
X_train_robust = robust_scaler.fit_transform(X_train[features])
X_test_robust = robust_scaler.transform(X_test[features])
modelo_rf_robust = RandomForestRegressor(n_estimators=500, random_state=42,)
modelo_rf_robust.fit(X_train_robust, y_train)
y_pred_robust = modelo_rf_robust.predict(X_test_robust)
print(f"Erro Médio Absoluto (MAE) com RobustScaler: {mean_absolute_error(y_test, y_pred_robust):.2f}")
print(f"R² Score (Precisão) com RobustScaler: {r2_score(y_test, y_pred_robust):.2f}")

# Criando e Treinando o Modelo
modelo_rf = RandomForestRegressor(n_estimators=500, random_state=42,)
modelo_rf.fit(X_train, y_train)

# Avaliando o modelo
y_pred = modelo_rf.predict(X_test)

print(f"Erro Médio Absoluto (MAE): {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R² Score (Precisão): {r2_score(y_test, y_pred):.2f}")


importancias = pd.Series(modelo_rf_robust.feature_importances_, index=features).sort_values()
importancias.plot(kind='barh', figsize=(8, 5), title='Importância das Features')
plt.tight_layout()
plt.show()

residuos = y_test.values - y_pred_robust

plt.figure(figsize=(10, 5))
plt.scatter(y_pred_robust, residuos, alpha=0.5)
plt.axhline(0, color='red', linestyle='--', label='Resíduo zero')
plt.xlabel("Valores Previstos")
plt.ylabel("Resíduo (Real - Previsto)")
plt.title("Análise de Resíduos")
plt.legend()
plt.grid(True)
plt.show()

# Estatísticas dos resíduos
print(f"Resíduo médio: {residuos.mean():.2f}")
print(f"Desvio padrão dos resíduos: {residuos.std():.2f}")
print(f"Resíduo máximo: {residuos.max():.2f}")
print(f"Resíduo mínimo: {residuos.min():.2f}")

#gerando grafico para mostrar qual a faixa etaria mais afetada pelos casos reais

plt.figure(figsize=(10, 5))
sns.histplot(data=data_treino, x='idade_media', bins=20, kde=True)
plt.title('Distribuição da Idade Média dos Casos Reais')
plt.xlabel('Idade Média')
plt.ylabel('Frequência')
plt.show()

# Verificação de leakage simplificada
print("verificação de leakage:")

treino_anos = data_treino.loc[X_train.index, 'ano']
teste_anos = data_treino.loc[X_test.index, 'ano']

treino_semanas = data_treino.loc[X_train.index, 'Semana_Epidemiologica']
teste_semanas = data_treino.loc[X_test.index, 'Semana_Epidemiologica']

print(f"Treino: {treino_anos.min()} semana {treino_semanas[treino_anos == treino_anos.min()].iloc[0]} "
      f"até {treino_anos.max()} semana {treino_semanas[treino_anos == treino_anos.max()].iloc[-1]}")

print(f"Teste:  {teste_anos.min()} semana {teste_semanas[teste_anos == teste_anos.min()].iloc[0]} "
      f"até {teste_anos.max()} semana {teste_semanas[teste_anos == teste_anos.max()].iloc[-1]}")

sobreposicao = set(X_train.index) & set(X_test.index)
print(f"\nSobreposição de índices entre treino e teste: {len(sobreposicao)} linhas")
print("Sem leakage por sobreposição!" if len(sobreposicao) == 0 else " HÁ LEAKAGE!")






