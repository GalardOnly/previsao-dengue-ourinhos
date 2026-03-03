import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Data_Dengue = pd.read_csv("dengue_3-6.csv", encoding="latin-1")
Data_Dengue.head()
Data_Dengue.info()
Data_Dengue.describe()


#tratamento de colunas vazias Substituindo 0 por NaN e remove as colunas (axis=1) onde todos os valores são NaN

df_limpo = Data_Dengue.replace(0, np.nan).dropna(axis=1, how='all')
df_limpo.info()

# renomeando as colunas para facilitar a leitura

df_limpo = df_limpo.rename(columns={
    'data_iniSE': 'Data_Inicio_Semanas',
    'SE': 'Semana_Epidemiologica',
    'casos_est': 'Casos_Estimados',
    'casos_est_min': 'Casos_Estimados_Minimo',
    'casos_est_max': 'Casos_Estimados_Maximo',
    'p_rt1': 'Probabilidade_RT_Alto',
    'p_inc100k': 'Probabilidade_Incidencia_100k',
    'id': 'ID_Municipio',
    'tweet': 'mencoes_twitter',
    'Rt': 'taxa_transmissao_rt',
    'pop': 'populacao',
    'tempmin': 'temperatura_minima',
    'umidmax': 'umidade_maxima',
    'nivel_inc': 'nivel_incidencia',
    'umidmed': 'umidade_media',
    'tempmax': 'temperatura_maxima',
    'tempmed': 'temperatura_media',
    'casoprov': 'casos_provaveis',
    'notif_accum_year': 'notificacoes_acumuladas_ano'
},)

print(df_limpo.columns)

df_limpo['Data_Inicio_Semanas'] = pd.to_datetime(df_limpo['Data_Inicio_Semanas'], errors='coerce')
df_limpo.info()
print(df_limpo['Data_Inicio_Semanas'].head())

df_limpo['nivel_incidencia'] = df_limpo['nivel_incidencia'].fillna(0).astype(int)

print(df_limpo['nivel_incidencia'].value_counts())


# substituindo os valores de 'nivel_incidencia' por categorias mais legiveis

mapa_incidencia = {
    0: 'incidencia nula',
    1: 'baixa incidencia',
    2: 'média incidencia',
    3: 'altissima incidencia'
}

df_limpo['nivel_incidencia'] = df_limpo['nivel_incidencia'].map(mapa_incidencia)
print(df_limpo['nivel_incidencia'].value_counts())

#checando coluna transmissao e receptivo

print(df_limpo['transmissao'].value_counts())
print(df_limpo['receptivo'].value_counts())

#calculando a correlaçao entre  temperatura media e casos

correlacao_temp_casos = df_limpo['temperatura_media'].corr(df_limpo['casos'], method='pearson')
print(f'correlação entre temperatura media e casos: {correlacao_temp_casos}')


#calculando a correlacao entre temperatura media receptivo e casos estimados

correlacao_temp_media_recep_casos_est = df_limpo['temperatura_media'].corr(df_limpo['Casos_Estimados'])
print(f'correlação entre temperatura media e casos estimados: {correlacao_temp_media_recep_casos_est}')


#criando  uma nova coluna chamada temp_lag_3

df_limpo['temp_lag_3'] = df_limpo['temperatura_media'].shift(3)
print(df_limpo[['temperatura_media', 'temp_lag_3']].head(10))

#criando correlaçao entre temp_lag_3 e casos_estimados

correlacao_temp_lag_casos_est = df_limpo['temp_lag_3'].corr(df_limpo['Casos_Estimados'])
print(f'correlação entre temp_lag_3 e casos estimados: {correlacao_temp_lag_casos_est}')


#plotando grafico temp_lag_ e casos estimados

plt.figure(figsize=(10, 6))
plt.scatter(df_limpo['temp_lag_3'], df_limpo['Casos_Estimados'], alpha=0.5)
plt.title('relação entre temperatura média com lag de 3 semanas e casos estimados de dengue')
plt.xlabel('temperatura média com lag de 3 semanas')
plt.ylabel('casos estimados')
plt.grid()
plt.show()

# Garanta que a Semana Epidemiológica da saúde seja quebrada em Ano e SE (0-52)
df_limpo['ano'] = df_limpo['Semana_Epidemiologica'] // 100
df_limpo['Semana_Epidemiologica'] = df_limpo['Semana_Epidemiologica'] % 100

#salvando o dataset limpo para um novo arquivo csv
df_limpo.to_csv('dengue_limpo.csv', index=False, encoding='latin-1')