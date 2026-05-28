# fis_inference.py - VERSION CHEMINS CORRIGÉS ✅
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os

print("🔍 FIS Inference System — Mode CSV direct\n")

# =============================================================================
# Configuration des chemins (robuste)
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Remonte d'un niveau
DATA_DIR = os.path.join(PROJECT_ROOT, "01_data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "03_results")
os.makedirs(f"{RESULTS_DIR}/figures", exist_ok=True)

# Fichier d'entrée (sortie de FCM)
FCM_OUTPUT = os.path.join(RESULTS_DIR, "fcm_output_Site_1_regime2.csv")

print(f"📂 Racine: {PROJECT_ROOT}")
print(f"📥 Entrée: {FCM_OUTPUT}")

# =============================================================================
# Charger les résultats FCM
# =============================================================================
try:
    df = pd.read_csv(FCM_OUTPUT)
    print(f"✅ Données FCM chargées: {len(df)} lignes")
except FileNotFoundError:
    print(f"❌ Fichier non trouvé: {FCM_OUTPUT}")
    print("💡 Exécute d'abord fcm_pipeline_direct.py")
    exit(1)

# =============================================================================
# Variables floues (mêmes univers que précédemment)
# =============================================================================
print("\n📊 Création des variables floues...")

T_amb = ctrl.Antecedent(np.linspace(20, 50, 100), 'T_amb')
T_amb['basse'] = fuzz.trimf(T_amb.universe, [20, 20, 30])
T_amb['moyenne'] = fuzz.trimf(T_amb.universe, [25, 35, 40])
T_amb['haute'] = fuzz.trimf(T_amb.universe, [35, 50, 50])

HR = ctrl.Antecedent(np.linspace(20, 80, 100), 'HR')
HR['faible'] = fuzz.trimf(HR.universe, [20, 20, 40])
HR['moyenne'] = fuzz.trimf(HR.universe, [30, 50, 60])
HR['élevée'] = fuzz.trimf(HR.universe, [50, 80, 80])

delta_P = ctrl.Antecedent(np.linspace(0, 30, 100), 'delta_P')
delta_P['faible'] = fuzz.trimf(delta_P.universe, [0, 0, 10])
delta_P['moyen'] = fuzz.trimf(delta_P.universe, [5, 15, 20])
delta_P['élevé'] = fuzz.trimf(delta_P.universe, [15, 30, 30])

Performance = ctrl.Consequent(np.linspace(0, 100, 100), 'Performance')
Performance['mauvaise'] = fuzz.trimf(Performance.universe, [0, 0, 40])
Performance['moyenne'] = fuzz.trimf(Performance.universe, [30, 60, 70])
Performance['bonne'] = fuzz.trimf(Performance.universe, [60, 100, 100])

# =============================================================================
# Règles floues
# =============================================================================
print("📜 Définition des règles...")
rule1 = ctrl.Rule(T_amb['basse'] & HR['faible'] & delta_P['faible'], Performance['bonne'])
rule2 = ctrl.Rule(T_amb['moyenne'] & HR['moyenne'] & delta_P['moyen'], Performance['moyenne'])
rule3 = ctrl.Rule(T_amb['haute'] & HR['élevée'] & delta_P['élevé'], Performance['mauvaise'])
rule4 = ctrl.Rule(T_amb['haute'] & delta_P['élevé'], Performance['mauvaise'])
rule5 = ctrl.Rule(T_amb['basse'] & delta_P['faible'], Performance['bonne'])

fis_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
fis = ctrl.ControlSystemSimulation(fis_ctrl)

# =============================================================================
# Application FIS
# =============================================================================
print("\n🔧 Application du FIS...")
performance_results = []

for idx, row in df.iterrows():
    try:
        fis.input['T_amb'] = row['T_amb']
        fis.input['HR'] = row['HR']
        fis.input['delta_P'] = row['delta_P']
        fis.compute()
        performance_results.append(fis.output['Performance'])
    except:
        performance_results.append(np.nan)

df['FIS_Performance'] = performance_results

# =============================================================================
# Statistiques et visualisation
# =============================================================================
print("\n📈 Statistiques FIS:")
print(df['FIS_Performance'].describe())

# Graphique
plt.figure(figsize=(10, 6))
plt.scatter(range(len(df)), df['FIS_Performance'], 
           c=df.get('fcm_cluster', np.zeros(len(df))), 
           cmap='viridis', alpha=0.7, s=20)
plt.xlabel('Index des données')
plt.ylabel('Performance FIS (%)')
plt.title('Performance prédite par FIS — Site_1, regime 2')
plt.colorbar(label='Cluster FCM')
plt.grid(alpha=0.3)
plt.tight_layout()

# ✅ Sauvegarde dans le bon dossier
output_fig = f"{RESULTS_DIR}/figures/fis_performance_Site_1_regime2.png"
plt.savefig(output_fig, dpi=300, bbox_inches='tight')
plt.close()
print(f"📊 Graphique: {output_fig}")

# =============================================================================
# Sauvegarde CSV + JSON
# =============================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_csv = f"{RESULTS_DIR}/fis_output_Site_1_regime2_{timestamp}.csv"
df.to_csv(output_csv, index=False)
print(f"💾 CSV: {output_csv}")

# Résumé JSON
summary = {
    "timestamp": timestamp,
    "input_file": FCM_OUTPUT,
    "total_samples": len(df),
    "valid_predictions": int(df['FIS_Performance'].notna().sum()),
    "mean_performance": float(df['FIS_Performance'].mean()),
    "std_performance": float(df['FIS_Performance'].std()),
    "output_csv": output_csv,
    "output_fig": output_fig
}
with open(f"{RESULTS_DIR}/fis_summary_{timestamp}.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"📁 Résumé: {RESULTS_DIR}/fis_summary_{timestamp}.json")

print("\n✅ FIS terminé ! Prêt pour PSO 🚀")