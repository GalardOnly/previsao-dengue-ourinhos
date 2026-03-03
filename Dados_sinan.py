import pandas as pd
import numpy as np
import glob

#carregando os arquivos csv do sinan e deixando em apenas um dataframe com a função glob
arquivos_csv = glob.glob('C:\\Projeto tcc\\csv\\*.csv')
df_projeto = pd.read_csv('C:\\Projeto tcc\\df_projeto.csv', encoding='latin-1', sep=';')

data_sus = pd.DataFrame()
for arquivo in arquivos_csv:
    df_temp = pd.read_csv(arquivo, sep=';', encoding='latin-1')
    data_sus = pd.concat([data_sus, df_temp], ignore_index=True)

data_sus.to_csv('C:\\Projeto tcc\\dados_sinan_completo.csv', index=False, encoding='latin-1', sep=';')
print("Processamento concluído. O arquivo 'dados_sinan_completo.csv' foi criado com sucesso.")
print(data_sus.columns.tolist())

# seleção das colunas que serão usadas
colunas_uteis = ['SEM_PRI', 'NU_IDADE_N', 'CS_SEXO', 'CLASSI_FIN', 'EVOLUCAO']
df_sinan = data_sus[colunas_uteis].copy()

# def para corrigir a idade que vem em formato  diferente
def decodificar_idade(v):
    try:
        s = str(int(v))
        if len(s) == 4:
            unidade = int(s[0])
            valor = int(s[1:])
            if unidade in [3, 4]:  # anos (1-9) e anos (10-99)
                return valor
    except:
        pass
    return np.nan  # ignora dias, meses e inválidos

df_sinan['idade_anos'] = df_sinan['NU_IDADE_N'].apply(decodificar_idade)
df_sinan['ano'] = df_sinan['SEM_PRI'] // 100
df_sinan['Semana_Epidemiologica'] = df_sinan['SEM_PRI'] % 100

# filtro para nao ter anos menos que 2014
df_sinan = df_sinan[
    (df_sinan['ano'] >= 2014) &
    (df_sinan['ano'] <= 2026) &
    (df_sinan['Semana_Epidemiologica'] >= 1) &
    (df_sinan['Semana_Epidemiologica'] <= 53)
].copy()

print(f"Registros após limpeza: {len(df_sinan)}")
print(f"Anos restantes: {sorted(df_sinan['ano'].unique())}")

#  agregação semanal para criar as features de saude (função .agg para agregar os dados e aplicar funções de media soma)
df_semanal = df_sinan.groupby(['ano', 'Semana_Epidemiologica']).agg(
    casos_reais=('SEM_PRI', 'count'),
    idade_media=('idade_anos', 'mean'),
    pct_mulheres=('CS_SEXO', lambda x: (x == 'F').mean() * 100),
    pct_confirmados=('CLASSI_FIN', lambda x: (x == 1).mean() * 100),
    obitos=('EVOLUCAO', lambda x: (x == 2).sum())
).reset_index()

#criando os lags e a media movel de 4 semana para casos reais
# , a media movel serve para que o modelo consiga captar a melhor tendencia dos casos 
df_semanal = df_semanal.sort_values(['ano', 'Semana_Epidemiologica'])

df_semanal['casos_lag_1'] = df_semanal.groupby('ano')['casos_reais'].shift(1)
df_semanal['casos_lag_2'] = df_semanal.groupby('ano')['casos_reais'].shift(2)
df_semanal['casos_mm4'] = (
    df_semanal.groupby('ano')['casos_reais']
    .transform(lambda x: x.rolling(window=4, min_periods=1).mean())
)

print("\nEstrutura pronta para o Merge:")
print(df_semanal.head())
print(f"\nShape: {df_semanal.shape}")
print(f"Anos cobertos: {sorted(df_semanal['ano'].unique())}")
print(f"\nNulos por coluna:\n{df_semanal.isnull().sum()[df_semanal.isnull().sum() > 0]}")

df_final = pd.merge(
    df_projeto,
    df_semanal,
    on=['ano', 'Semana_Epidemiologica'],
    how='left'  # left para não perder semanas sem notificação no SINAN
)

# Imputação por mediana do ano mais robusta que média em dados com surtos
colunas_clima = ['temp_ar', 'umidade', 'temp_lag_1', 'temp_lag_2', 
                 'temp_lag_3', 'temp_lag_4', 'chuva_lag_1', 
                 'chuva_lag_2', 'chuva_lag_3', 'chuva_lag_4']

for col in colunas_clima:
    df_final[col] = df_final.groupby('ano')[col].transform(
        lambda x: x.fillna(x.median())
    )

# Para as colunas do SINAN, zero faz sentido biológico semana sem notificação = zero casos
colunas_sinan = ['casos_reais', 'idade_media', 'pct_mulheres', 
                 'pct_confirmados', 'obitos', 'casos_lag_1', 
                 'casos_lag_2', 'casos_mm4']

df_final[['casos_reais', 'obitos', 'casos_lag_1', 'casos_lag_2', 'casos_mm4']] = \
    df_final[['casos_reais', 'obitos', 'casos_lag_1', 'casos_lag_2', 'casos_mm4']].fillna(0)

# Médias ficam como NaN — o modelo ignora ou imputa pela mediana do ano
df_final['idade_media'] = df_final.groupby('ano')['idade_media'].transform(
    lambda x: x.fillna(x.median())
)
df_final['pct_mulheres'] = df_final.groupby('ano')['pct_mulheres'].transform(
    lambda x: x.fillna(x.median())
)
df_final['pct_confirmados'] = df_final.groupby('ano')['pct_confirmados'].transform(
    lambda x: x.fillna(x.median())
)

print(f"Dataset final: {df_final.shape[0]} semanas e {df_final.shape[1]} colunas")
print(f"\nNulos por coluna:\n{df_final.isnull().sum()[df_final.isnull().sum() > 0]}")
print(f'merge completo. Anos no final: {sorted(df_final["ano"].unique())}')

df_final.to_csv('C:\\Projeto tcc\\dataset_final_ourinhos.csv', index=False, encoding='latin-1', sep=';')

print("✅ Dataset final criado com sucesso! O arquivo 'dataset_final_ourinhos.csv' está pronto para análise.")