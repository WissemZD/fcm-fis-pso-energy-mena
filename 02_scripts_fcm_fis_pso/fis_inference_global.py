# fis_inference_global.py
# ✅ Inférence floue sur clustering FCM GLOBAL
# ✅ Intègre explicitement le biais climatique (T_amb + HR)
# ✅ Métriques de validation incluses

import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os
from sklearn.metrics import mean_squared_error, r2_score

print("🔍 FIS Inference — Clustering GLOBAL + Biais Climatique\n")

# =============================================================================
# Configuration des chemins
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "03_results")
os.makedirs(f"{RESULTS_DIR}/figures", exist_ok=True)

# Fichier d'entrée (sortie du clustering global)
# 👉 Mets à jour ce nom si ton timestamp est différent
FCM_OUTPUT = "fcm_global_output_20260510_183404.csv"
FCM_PATH = os.path.join(RESULTS_DIR, FCM_OUTPUT)

print(f"📥 Chargement: {FCM_PATH}")

# =============================================================================
# Charger les données FCM
# =============================================================================
try:
    df = pd.read_csv(FCM_PATH)
    print(f"✅ Données chargées: {len(df)} lignes")
    print(f"📋 Colonnes: {list(df.columns)}")
except FileNotFoundError:
    print(f"❌ Fichier non trouvé. Liste des fichiers dans {RESULTS_DIR}:")
    print([f for f in os.listdir(RESULTS_DIR) if f.startswith("fcm_global")])
    exit(1)

# =============================================================================
# Variables floues — Calibrées sur LES données réelles
# =============================================================================
print("\n📊 Création des variables floues (calibrées sur dataset réel)...")

# Input 1: Température ambiante [°C] — plages observées dans dataset
T_amb = ctrl.Antecedent(np.linspace(20, 50, 100), 'T_amb')
T_amb['fraiche'] = fuzz.trimf(T_amb.universe, [20, 20, 28])
T_amb['moderee'] = fuzz.trimf(T_amb.universe, [25, 35, 40])
T_amb['chaude'] = fuzz.trimf(T_amb.universe, [35, 50, 50])

# Input 2: Humidité Relative [%] — impact sur condensation/corrosion
HR = ctrl.Antecedent(np.linspace(20, 80, 100), 'HR')
HR['seche'] = fuzz.trimf(HR.universe, [20, 20, 35])
HR['normale'] = fuzz.trimf(HR.universe, [30, 50, 60])
HR['humide'] = fuzz.trimf(HR.universe, [50, 80, 80])

# Input 3: Load Factor [0-1] — ratio performance réelle/théorique
LF = ctrl.Antecedent(np.linspace(0, 1, 100), 'LF')
LF['faible'] = fuzz.trimf(LF.universe, [0, 0, 0.5])
LF['moyen'] = fuzz.trimf(LF.universe, [0.4, 0.6, 0.8])
LF['eleve'] = fuzz.trimf(LF.universe, [0.7, 1, 1])

# Input 4: Delta P [W/m²] — pertes de puissance
delta_P = ctrl.Antecedent(np.linspace(0, 30, 100), 'delta_P')
delta_P['minime'] = fuzz.trimf(delta_P.universe, [0, 0, 8])
delta_P['modere'] = fuzz.trimf(delta_P.universe, [5, 12, 20])
delta_P['important'] = fuzz.trimf(delta_P.universe, [15, 30, 30])

# Output: Performance corrigée du biais climatique [0-100]
Performance = ctrl.Consequent(np.linspace(0, 100, 100), 'Performance_Climat')
Performance['degradee'] = fuzz.trimf(Performance.universe, [0, 0, 40])
Performance['nominale'] = fuzz.trimf(Performance.universe, [30, 60, 70])
Performance['optimale'] = fuzz.trimf(Performance.universe, [60, 100, 100])

# =============================================================================
# Règles floues — Intégrant le BIAIS CLIMATIQUE
# =============================================================================
print("📜 Définition des règles (biais climatique explicite)...")

# Règle 1: Conditions idéales → performance optimale
rule1 = ctrl.Rule(
    T_amb['fraiche'] & HR['seche'] & LF['eleve'] & delta_P['minime'],
    Performance['optimale']
)

# Règle 2: Stress thermique (chaud + sec) → dégradation modérée
rule2 = ctrl.Rule(
    T_amb['chaude'] & HR['seche'] & delta_P['modere'],
    Performance['nominale']
)

# Règle 3: Stress hygrométrique (humide) → dégradation forte
rule3 = ctrl.Rule(
    HR['humide'] & delta_P['important'],
    Performance['degradee']
)

# Règle 4: Biais climatique combiné (chaud + humide = pire cas)
rule4 = ctrl.Rule(
    T_amb['chaude'] & HR['humide'],
    Performance['degradee']
)

# Règle 5: Compensation par bon LF
rule5 = ctrl.Rule(
    LF['eleve'] & delta_P['minime'],
    Performance['optimale']
)

# Règle 6: Conditions intermédiaires → performance nominale
rule6 = ctrl.Rule(
    T_amb['moderee'] & HR['normale'] & LF['moyen'],
    Performance['nominale']
)

# =============================================================================
# Système de contrôle FIS
# =============================================================================
fis_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
fis = ctrl.ControlSystemSimulation(fis_ctrl)

# =============================================================================
# Application FIS à chaque échantillon
# =============================================================================
print("\n🔧 Application du FIS sur les 3000 échantillons...")

predictions = []
confidences = []

for idx, row in df.iterrows():
    try:
        # Inputs normalisés
        fis.input['T_amb'] = float(row['T_amb'])
        fis.input['HR'] = float(row['HR'])
        fis.input['LF'] = float(row['LF'])
        fis.input['delta_P'] = float(row['delta_P'])
        
        # Inférence
        fis.compute()
        
        predictions.append(fis.output['Performance_Climat'])
        confidences.append(1.0)  # À remplacer par métrique de confiance si disponible
        
    except Exception as e:
        predictions.append(np.nan)
        confidences.append(0.0)

df['FIS_Performance_Climat'] = predictions
df['FIS_Confidence'] = confidences

print(f"✅ Inférence terminée: {np.sum(~np.isnan(predictions))}/{len(df)} prédictions valides")

# =============================================================================
# Métriques de validation (si référence disponible)
# =============================================================================
print("\n📈 Métriques de validation...")

# Si tu as une colonne de référence (ex: P_mesuree/P_ref * 100)
if 'rendement' in df.columns:
    reference = df['rendement'] * 100  # Conversion en %
    valid_mask = ~np.isnan(predictions) & ~np.isnan(reference)
    
    if np.sum(valid_mask) > 0:
        y_true = reference[valid_mask]
        y_pred = np.array(predictions)[valid_mask]
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mae = np.mean(np.abs(y_true - y_pred))
        
        print(f"   • RMSE: {rmse:.2f}%")
        print(f"   • R²: {r2:.3f}")
        print(f"   • MAE: {mae:.2f}%")
        
        metrics = {"rmse": float(rmse), "r2": float(r2), "mae": float(mae)}
    else:
        print("   ⚠️ Pas assez de données valides pour métriques")
        metrics = {}
else:
    print("   ℹ️ Aucune colonne 'rendement' — métriques internes uniquement")
    metrics = {
        "mean_prediction": float(np.nanmean(predictions)),
        "std_prediction": float(np.nanstd(predictions))
    }

# =============================================================================
# Visualisation : Performance vs Cluster FCM + Biais Climatique
# =============================================================================
print("\n📊 Génération des graphiques...")

# Graphique 1: Performance FIS par cluster FCM
plt.figure(figsize=(10, 6))
for cluster in sorted(df['fcm_cluster_global'].unique()):
    mask = df['fcm_cluster_global'] == cluster
    plt.scatter(
        df.loc[mask, 'T_amb'], 
        df.loc[mask, 'FIS_Performance_Climat'],
        label=f'Cluster {cluster}', 
        alpha=0.6, s=30
    )
plt.xlabel('Température ambiante (°C)')
plt.ylabel('Performance corrigée FIS (%)')
plt.title('Performance FIS par Cluster FCM — Biais Climatique')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
fig1_path = f"{RESULTS_DIR}/figures/fis_performance_vs_cluster.png"
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Graphique 1: {fig1_path}")

# Graphique 2: Heatmap T_amb × HR → Performance
plt.figure(figsize=(8, 6))
# Créer une grille pour heatmap
T_grid = np.linspace(df['T_amb'].min(), df['T_amb'].max(), 50)
HR_grid = np.linspace(df['HR'].min(), df['HR'].max(), 50)
Z = np.zeros((len(T_grid), len(HR_grid)))

# Inférence rapide sur grille (simplifiée)
for i, t in enumerate(T_grid):
    for j, h in enumerate(HR_grid):
        try:
            fis.input['T_amb'] = t
            fis.input['HR'] = h
            fis.input['LF'] = df['LF'].median()  # Valeur médiane fixe
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
# Sauvegarde des résultats
# =============================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_csv = f"{RESULTS_DIR}/fis_global_output_{timestamp}.csv"
df.to_csv(output_csv, index=False)
print(f"\n💾 CSV sauvegardé: {output_csv}")

# Résumé JSON pour le mémoire
summary = {
    "timestamp": timestamp,
    "input_file": FCM_OUTPUT,
    "strategy": "Global clustering + FIS biais climatique",
    "features": ["T_amb", "HR", "LF", "delta_P"],
    "total_samples": len(df),
    "valid_predictions": int(np.sum(~np.isnan(predictions))),
    "fpc_score": 0.6582,  # Valeur du clustering FCM
    "metrics": metrics,
    "rules_count": 6,
    "output_csv": output_csv,
    "figures": [fig1_path, fig2_path]
}
with open(f"{RESULTS_DIR}/fis_global_summary_{timestamp}.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"📁 Résumé JSON: {RESULTS_DIR}/fis_global_summary_{timestamp}.json")

print("\n✅ FIS GLOBAL terminé ! Prêt pour PSO 🚀")