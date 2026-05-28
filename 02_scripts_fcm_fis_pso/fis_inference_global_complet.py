# fis_inference_global_complet.py
# ✅ Version complète avec calcul du rendement + métriques
# ✅ Compatible avec pso_optimization_global.py

import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("🔍 FIS Inference — Clustering GLOBAL + Biais Climatique\n")

# =============================================================================
# Configuration des chemins
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "03_results")
os.makedirs(f"{RESULTS_DIR}/figures", exist_ok=True)

# Fichier d'entrée (sortie du clustering global)
# 👉 Mets à jour ce nom avec ton fichier le plus récent
import glob
fcm_files = glob.glob(os.path.join(RESULTS_DIR, "fcm_global_output_*.csv"))
if not fcm_files:
    print("❌ Aucun fichier FCM trouvé dans 03_results/")
    exit(1)
FCM_PATH = sorted(fcm_files)[-1]  # Prend le plus récent

print(f"📥 Chargement: {FCM_PATH}")

# =============================================================================
# Charger les données FCM
# =============================================================================
try:
    df = pd.read_csv(FCM_PATH)
    print(f"✅ Données chargées: {len(df)} lignes")
except Exception as e:
    print(f"❌ Erreur lecture: {e}")
    exit(1)

# =============================================================================
# Calculer le rendement (si absent)
# =============================================================================
if 'rendement' not in df.columns:
    print("📊 Calcul du rendement (P_mesuree / P_ref)...")
    df['rendement'] = df['P_mesuree'] / df['P_ref']
    df['rendement'] = df['rendement'].clip(0, 2)  # Évite les valeurs aberrantes
    print(f"   Rendement moyen: {df['rendement'].mean():.2%}")
    print(f"   Rendement min: {df['rendement'].min():.2%}")
    print(f"   Rendement max: {df['rendement'].max():.2%}")
    
    # Sauvegarder le CSV avec rendement
    df.to_csv(FCM_PATH, index=False)
    print(f"   ✅ CSV mis à jour avec colonne 'rendement'")

# =============================================================================
# Variables floues
# =============================================================================
print("\n📊 Création des variables floues...")

T_amb = ctrl.Antecedent(np.linspace(20, 50, 100), 'T_amb')
T_amb['fraiche'] = fuzz.trimf(T_amb.universe, [20, 20, 28])
T_amb['moderee'] = fuzz.trimf(T_amb.universe, [25, 35, 40])
T_amb['chaude'] = fuzz.trimf(T_amb.universe, [35, 50, 50])

HR = ctrl.Antecedent(np.linspace(20, 80, 100), 'HR')
HR['seche'] = fuzz.trimf(HR.universe, [20, 20, 35])
HR['normale'] = fuzz.trimf(HR.universe, [30, 50, 60])
HR['humide'] = fuzz.trimf(HR.universe, [50, 80, 80])

LF = ctrl.Antecedent(np.linspace(0, 1, 100), 'LF')
LF['faible'] = fuzz.trimf(LF.universe, [0, 0, 0.5])
LF['moyen'] = fuzz.trimf(LF.universe, [0.4, 0.6, 0.8])
LF['eleve'] = fuzz.trimf(LF.universe, [0.7, 1, 1])

delta_P = ctrl.Antecedent(np.linspace(0, 30, 100), 'delta_P')
delta_P['minime'] = fuzz.trimf(delta_P.universe, [0, 0, 8])
delta_P['modere'] = fuzz.trimf(delta_P.universe, [5, 12, 20])
delta_P['important'] = fuzz.trimf(delta_P.universe, [15, 30, 30])

Performance = ctrl.Consequent(np.linspace(0, 100, 100), 'Performance_Climat')
Performance['degradee'] = fuzz.trimf(Performance.universe, [0, 0, 40])
Performance['nominale'] = fuzz.trimf(Performance.universe, [30, 60, 70])
Performance['optimale'] = fuzz.trimf(Performance.universe, [60, 100, 100])

# =============================================================================
# Règles floues
# =============================================================================
print("📜 Définition des règles...")

rule1 = ctrl.Rule(T_amb['fraiche'] & HR['seche'] & LF['eleve'] & delta_P['minime'], Performance['optimale'])
rule2 = ctrl.Rule(T_amb['chaude'] & HR['seche'] & delta_P['modere'], Performance['nominale'])
rule3 = ctrl.Rule(HR['humide'] & delta_P['important'], Performance['degradee'])
rule4 = ctrl.Rule(T_amb['chaude'] & HR['humide'], Performance['degradee'])
rule5 = ctrl.Rule(LF['eleve'] & delta_P['minime'], Performance['optimale'])
rule6 = ctrl.Rule(T_amb['moderee'] & HR['normale'] & LF['moyen'], Performance['nominale'])

fis_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
fis = ctrl.ControlSystemSimulation(fis_ctrl)

# =============================================================================
# Application FIS
# =============================================================================
print("\n🔧 Application du FIS...")

predictions = []
for idx, row in df.iterrows():
    try:
        fis.input['T_amb'] = float(row['T_amb'])
        fis.input['HR'] = float(row['HR'])
        fis.input['LF'] = float(row['LF'])
        fis.input['delta_P'] = float(row['delta_P'])
        fis.compute()
        predictions.append(fis.output['Performance_Climat'])
    except:
        predictions.append(np.nan)

df['FIS_Performance_Climat'] = predictions
print(f"✅ Inférence terminée: {np.sum(~np.isnan(predictions))}/{len(df)} prédictions valides")

# =============================================================================
# Métriques de validation (AVANT PSO)
# =============================================================================
print("\n📈 Métriques de validation (avant optimisation)...")

y_true = df['rendement'].values * 100  # Conversion en %
y_pred = np.array(predictions)
mask = ~np.isnan(y_pred) & ~np.isnan(y_true)

if np.sum(mask) > 10:
    rmse_before = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
    r2_before = r2_score(y_true[mask], y_pred[mask])
    mae_before = mean_absolute_error(y_true[mask], y_pred[mask])
    
    print(f"   • RMSE (avant PSO): {rmse_before:.2f}%")
    print(f"   • R² (avant PSO): {r2_before:.3f}")
    print(f"   • MAE (avant PSO): {mae_before:.2f}%")
    
    metrics_before = {"rmse": float(rmse_before), "r2": float(r2_before), "mae": float(mae_before)}
else:
    print("   ⚠️ Pas assez de données valides")
    metrics_before = {}

# =============================================================================
# Visualisation
# =============================================================================
print("\n📊 Génération des graphiques...")

# Graphique 1: Performance FIS vs Rendement réel
plt.figure(figsize=(10, 6))
plt.scatter(y_true[mask], y_pred[mask], alpha=0.5, s=20)
plt.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Ligne de référence')
plt.xlabel('Rendement réel (%)')
plt.ylabel('Performance prédite par FIS (%)')
plt.title('FIS vs Rendement Réel (Avant Optimisation PSO)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
fig1_path = f"{RESULTS_DIR}/figures/fis_vs_reel_avant_pso.png"
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Graphique 1: {fig1_path}")

# Graphique 2: Heatmap T_amb × HR
plt.figure(figsize=(8, 6))
T_grid = np.linspace(df['T_amb'].min(), df['T_amb'].max(), 50)
HR_grid = np.linspace(df['HR'].min(), df['HR'].max(), 50)
Z = np.zeros((len(T_grid), len(HR_grid)))

for i, t in enumerate(T_grid):
    for j, h in enumerate(HR_grid):
        try:
            fis.input['T_amb'] = t
            fis.input['HR'] = h
            fis.input['LF'] = df['LF'].median()
            fis.input['delta_P'] = df['delta_P'].median()
            fis.compute()
            Z[i, j] = fis.output['Performance_Climat']
        except:
            Z[i, j] = np.nan

plt.contourf(HR_grid, T_grid, Z, levels=20, cmap='RdYlGn')
plt.colorbar(label='Performance FIS (%)')
plt.xlabel('Humidité Relative (%)')
plt.ylabel('Température ambiante (°C)')
plt.title('Carte de Performance — Biais Climatique (T_amb × HR)')
plt.tight_layout()
fig2_path = f"{RESULTS_DIR}/figures/fis_heatmap_climat.png"
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Graphique 2: {fig2_path}")

# =============================================================================
# Sauvegarde
# =============================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_csv = f"{RESULTS_DIR}/fis_global_output_{timestamp}.csv"
df.to_csv(output_csv, index=False)
print(f"\n💾 CSV sauvegardé: {output_csv}")

summary = {
    "timestamp": timestamp,
    "input_file": os.path.basename(FCM_PATH),
    "strategy": "Global clustering + FIS (avant PSO)",
    "features": ["T_amb", "HR", "LF", "delta_P"],
    "total_samples": len(df),
    "valid_predictions": int(np.sum(mask)),
    "metrics_before_pso": metrics_before,
    "rules_count": 6,
    "output_csv": output_csv,
    "figures": [fig1_path, fig2_path]
}
with open(f"{RESULTS_DIR}/fis_global_summary_{timestamp}.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"📁 Résumé JSON: {RESULTS_DIR}/fis_global_summary_{timestamp}.json")

print("\n✅ FIS GLOBAL terminé ! Prêt pour PSO 🚀")
print(f"\n📌 PROCHAINE ÉTAPE:")
print(f"   Lance: python .\\02_scripts_fcm_fis_pso\\pso_optimization_global.py")
print(f"   PSO va optimiser les poids des règles pour réduire le RMSE de {rmse_before:.2f}%")