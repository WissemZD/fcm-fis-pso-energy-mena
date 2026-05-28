import pandas as pd
import os

RESULTS_DIR = "03_results"
FCM_FILE = "fcm_global_output_20260510_183404.csv"
FCM_PATH = os.path.join(RESULTS_DIR, FCM_FILE)

# Charger
df = pd.read_csv(FCM_PATH)

# Calculer rendement
df['rendement'] = df['P_mesuree'] / df['P_ref']

# Sauvegarder
df.to_csv(FCM_PATH, index=False)
print(f"✅ Colonne 'rendement' ajoutée à {FCM_FILE}")
print(f"   Rendement moyen: {df['rendement'].mean():.2%}")
print(f"   Rendement min: {df['rendement'].min():.2%}")
print(f"   Rendement max: {df['rendement'].max():.2%}")