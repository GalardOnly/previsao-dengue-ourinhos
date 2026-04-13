"""Script temporário para merge dos dados novos do InfoDengue."""
import pandas as pd

df_old = pd.read_csv('Data/dengue_3-6.csv', encoding='latin-1')
df_new = pd.read_csv('Data/dengue_53-15.csv', encoding='latin-1')

# Remover coluna extra do novo arquivo
if 'municipio_nome' in df_new.columns:
    df_new = df_new.drop(columns=['municipio_nome'])

# Remover do antigo as SEs que existem no novo (priorizar dados mais recentes)
ses_novas = set(df_new['SE'].values)
df_old_limpo = df_old[~df_old['SE'].isin(ses_novas)]

# Concatenar
df_concat = pd.concat([df_old_limpo, df_new], ignore_index=True)
df_concat = df_concat.sort_values('SE').reset_index(drop=True)

print(f"Antigo: {len(df_old)} semanas (SE {df_old['SE'].min()}-{df_old['SE'].max()})")
print(f"Novo:   {len(df_new)} semanas (SE {df_new['SE'].min()}-{df_new['SE'].max()})")
print(f"Removidas do antigo (sobrepostas): {len(df_old) - len(df_old_limpo)}")
print(f"Final:  {len(df_concat)} semanas (SE {df_concat['SE'].min()}-{df_concat['SE'].max()})")
print(f"Semanas novas adicionadas: {len(df_concat) - len(df_old)}")

df_concat.to_csv('Data/dengue_3-6.csv', index=False, encoding='latin-1')
print("\ndengue_3-6.csv atualizado com sucesso!")
