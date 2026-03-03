import pandas as pd
from epiweeks import Week, Year
from datetime import date
import glob
import os

# CONSOLIDAÇÃO DO CLIMA 
caminho_clima = 'dados_inmet/*.csv' 
arquivos = glob.glob(caminho_clima)

lista_clima = []

for arquivo in arquivos:
    # Lógica de limpeza aplicada a todos
    temp_df = pd.read_csv(arquivo, sep=';', encoding='latin-1', skiprows=9, decimal=',')
    
    # Seleção e renomeação dinâmica
    temp_df = temp_df.iloc[:, [0, 2, 7, 13]].copy() 
    temp_df.columns = ['data', 'chuva', 'temp_ar', 'umidade'] # Padronizado como 'temp_ar'
    
    lista_clima.append(temp_df)

# Empilhando todos os anos

df_clima_bruto = pd.concat(lista_clima, ignore_index=True)

df_clima_bruto['data'] = pd.to_datetime(df_clima_bruto['data'], format='mixed', dayfirst=False)
df_clima_bruto['ano'] = df_clima_bruto['data'].dt.year

def get_epiweek(data):
    try:
        w = Week.fromdate(data.date(), system="cdc")  # CDC = começa no domingo
        return w.week
    except:
        return None

df_clima_bruto['Semana_Epidemiologica'] = df_clima_bruto['data'].apply(get_epiweek)
df_clima_bruto = df_clima_bruto.dropna(subset=['Semana_Epidemiologica']) 

#  Usando os nomes corretos
colunas_clima = ['temp_ar', 'chuva', 'umidade'] 

# Varredura de limpeza e Type Casting(é o processo de transformar um dado de um tipo para outro, como converter uma string em número ou um ponto flutuante em inteiro.)
for col in colunas_clima:
    df_clima_bruto[col] = df_clima_bruto[col].astype(str).str.replace(',', '.')
    df_clima_bruto[col] = pd.to_numeric(df_clima_bruto[col], errors='coerce')

# Agrupamento semanal 
df_clima_semanal = df_clima_bruto.groupby(['ano', 'Semana_Epidemiologica']).agg({
    'temp_ar': 'mean',
    'chuva': 'sum',
    'umidade': 'mean'
}).reset_index()

# Ordenação para os Lags
df_clima_semanal = df_clima_semanal.sort_values(['ano', 'Semana_Epidemiologica'])

for lag in range(1, 5):
    df_clima_semanal[f'chuva_lag_{lag}'] = df_clima_semanal.groupby('ano')['chuva'].shift(lag)
    df_clima_semanal[f'temp_lag_{lag}'] = df_clima_semanal.groupby('ano')['temp_ar'].shift(lag)

df_clima_final = df_clima_semanal[
    ~(df_clima_semanal['temp_ar'].isna() & df_clima_semanal['chuva'].isna())
].reset_index(drop=True)

#  DADOS DE SAÚDE
df_saude = pd.read_csv("dengue_3-6.csv", encoding="latin-1")
df_saude['Data_Inicio_Semanas'] = pd.to_datetime(df_saude['data_iniSE'])

# Desmembrando a coluna SE para garantir o Merge
df_saude['ano'] = df_saude['SE'] // 100
df_saude['Semana_Epidemiologica'] = df_saude['SE'] % 100



#  partindo para o merge final.
df_projeto = pd.merge(
    df_saude, 
    df_clima_final, 
    on=['ano', 'Semana_Epidemiologica'], 
    how='inner' 
)

# retirando colunas vazias
colunas_vazias = ['casprov_est', 'casprov_est_min', 'casprov_est_max', 'casconf', 'tweet']
df_projeto = df_projeto.drop(columns=[c for c in colunas_vazias if c in df_projeto.columns])

# Imputação dos 17 nulos das colunas originais do info.dengue
# Usando mediana por ano para não deixar surtos distorcerem
for col in ['umidmed', 'umidmin', 'tempmed', 'tempmax']:
    if col in df_projeto.columns:
        df_projeto[col] = df_projeto.groupby('ano')[col].transform(
            lambda x: x.fillna(x.median())
        )

df_projeto.to_csv('C:\\Projeto tcc\\df_projeto.csv', index=False, encoding='latin-1', sep=';')

print(f"Sucesso! Dataset de Ourinhos criado com {df_projeto.shape[0]} semanas e {df_projeto.shape[1]} colunas.")

# verificar se o merge não gerou NaNs indesejados no clima
print("\nVerificando dados nulos pós-soldagem:")
print(df_projeto[['temp_ar', 'chuva']].isnull().sum())

print(f"Semanas no clima:  {df_clima_final.shape[0]}")
print(f"Semanas na saúde:  {df_saude.shape[0]}")
print(f"Semanas no merge:  {df_projeto.shape[0]}")
print(f"\nAnos cobertos: {sorted(df_projeto['ano'].unique())}")
print(f"\nNulos por coluna:\n{df_projeto.isnull().sum()[df_projeto.isnull().sum() > 0]}")

