import pandas as pd

df_pso = pd.read_csv('data/processed/PSO_master.csv')
df_mebl = pd.read_csv('data/processed/MEBL_master.csv')

pso_cols = set(df_pso.columns)
mebl_cols = set(df_mebl.columns)

missing_in_mebl = pso_cols - mebl_cols
missing_in_pso = mebl_cols - pso_cols

print(f"PSO cols: {len(pso_cols)}, MEBL cols: {len(mebl_cols)}")
print(f"Missing in MEBL: {missing_in_mebl}")
print(f"Missing in PSO: {missing_in_pso}")

print(f"PSO is_synthetic_rate values: {df_pso['is_synthetic_rate'].unique()}")
print(f"MEBL is_synthetic_rate values: {df_pso['is_synthetic_rate'].unique()}")
print(f"PSO middle_east_conflict_flag positive %: {df_pso['middle_east_conflict_flag'].mean()*100:.2f}%")
print(f"MEBL middle_east_conflict_flag positive %: {df_mebl['middle_east_conflict_flag'].mean()*100:.2f}%")
