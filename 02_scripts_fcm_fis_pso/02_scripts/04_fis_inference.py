#!/usr/bin/env python3
"""
fis_inference.py
================
Système d'Inférence Floue (FIS) pour modéliser le biais climatique.

Améliorations :
  - Fonctions d'appartenance gaussiennes (plus lisses)
  - 5 niveaux de performance au lieu de 3
  - Calcul automatique du rendement si absent
  - Métriques de validation (RMSE, R², MAE)

Auteur : Wissem ZD
Date : 2026
"""

import os
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import json
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("="*70)
print("🔍 SYSTÈME D'INFÉRENCE FLOUE (FIS)")
print("="*70)

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = r"C:\projets\memoire_FCM"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "03_results")

# Choisir la configuration (triplet ou quadruplet)
CONFIG = "triplet"  # Change en "triplet" pour comparer
INPUT_DIR = os.path.join(RESULTS_DIR, CONFIG)

print(f"\n📂 Configuration: {CONFIG}")
print(f"📁 Dossier entrée: {INPUT_DIR}")

# Trouver le fichier FCM le plus récent
import glob
fcm_files = glob.glob(os.path.join(INPUT_DIR, "fcm_output_*.csv"))
if not fcm_files:
    print(f"❌ Aucun fichier FCM trouvé dans {INPUT_DIR}")
    print("💡 Lance d'abord: python 02_scripts/01_fcm_clustering.py")
    exit(1)

FCM_FILE = sorted(fcm_files)[-1]
print(f"📥 Fichier FCM: {os.path.basename(FCM_FILE)}")

# =============================================================================
# Charger les données
# =============================================================================
df = pd.read_csv(FCM_FILE)
print(f"✅ Données chargées: {len(df)} lignes")

# Calculer rendement si absent
if 'rendement' not in df.columns:
    print("\n📊 Calcul du rendement (P_mesuree / P_ref)...")
    if 'P_mesuree' in df.columns and 'P_ref' in df.columns:
        df['rendement'] = (df['P_mesuree'] / df['P_ref']).clip(0, 1.0)
        print(f"   Rendement moyen: {df['rendement'].mean():.2%}")
    else:
        print("⚠️ Colonnes P_mesuree ou P_ref manquantes")
        df['rendement'] = np.nan

# =============================================================================
# Variables floues (améliorées)
# =============================================================================
print("\n📊 Création des variables floues (fonctions gaussiennes)...")

# Input 1: Température ambiante
T_amb = ctrl.Antecedent(np.linspace(15, 55, 100), 'T_amb')
T_amb['fraiche'] = fuzz.gaussmf(T_amb.universe, 22, 5)
T_amb['moderee'] = fuzz.gaussmf(T_amb.universe, 32, 6)
T_amb['chaude'] = fuzz.gaussmf(T_amb.universe, 42, 5)

# Input 2: Humidité Relative (seulement si quadruplet)
if CONFIG == "quadruplet" and 'HR' in df.columns:
    HR = ctrl.Antecedent(np.linspace(20, 80, 100), 'HR')
    HR['seche'] = fuzz.gaussmf(HR.universe, 30, 8)
    HR['normale'] = fuzz.gaussmf(HR.universe, 50, 10)
    HR['humide'] = fuzz.gaussmf(HR.universe, 70, 8)
    use_hr = True
else:
    use_hr = False
    print("ℹ️ Mode TRIPLET (sans HR)")

# Input 3: Load Factor
LF = ctrl.Antecedent(np.linspace(0, 1.2, 100), 'LF')
LF['faible'] = fuzz.gaussmf(LF.universe, 0.3, 0.15)
LF['moyen'] = fuzz.gaussmf(LF.universe, 0.6, 0.15)
LF['eleve'] = fuzz.gaussmf(LF.universe, 0.9, 0.15)

# Input 4: Delta P
delta_P = ctrl.Antecedent(np.linspace(0, 35, 100), 'delta_P')
delta_P['minime'] = fuzz.gaussmf(delta_P.universe, 3, 2)
delta_P['modere'] = fuzz.gaussmf(delta_P.universe, 12, 4)
delta_P['important'] = fuzz.gaussmf(delta_P.universe, 22, 5)

# Output: Performance (5 niveaux pour plus de précision)
Performance = ctrl.Consequent(np.linspace(0, 100, 100), 'Performance_Climat')
Performance['tres_faible'] = fuzz.gaussmf(Performance.universe, 15, 8)
Performance['faible'] = fuzz.gaussmf(Performance.universe, 35, 10)
Performance['moyenne'] = fuzz.gaussmf(Performance.universe, 50, 10)
Performance['bonne'] = fuzz.gaussmf(Performance.universe, 70, 10)
Performance['excellente'] = fuzz.gaussmf(Performance.universe, 90, 8)

# =============================================================================
# Règles floues (adaptées à la configuration)
# =============================================================================
print("📜 Définition des règles floues...")

if use_hr:
    # Règles complètes (quadruplet)
    rules = [
        ctrl.Rule(T_amb['fraiche'] & HR['seche'] & LF['eleve'] & delta_P['minime'], Performance['excellente']),
        ctrl.Rule(T_amb['moderee'] & HR['normale'] & LF['moyen'] & delta_P['modere'], Performance['moyenne']),
        ctrl.Rule(T_amb['chaude'] & HR['humide'] & delta_P['important'], Performance['tres_faible']),
        ctrl.Rule(T_amb['chaude'] & HR['humide'], Performance['faible']),  # Biais T×HR
        ctrl.Rule(T_amb['fraiche'] & LF['eleve'] & delta_P['minime'], Performance['bonne']),
        ctrl.Rule(LF['faible'] | delta_P['important'], Performance['faible']),
    ]
else:
    # Règles simplifiées (triplet)
    rules = [
        ctrl.Rule(T_amb['fraiche'] & LF['eleve'] & delta_P['minime'], Performance['excellente']),
        ctrl.Rule(T_amb['moderee'] & LF['moyen'] & delta_P['modere'], Performance['moyenne']),
        ctrl.Rule(T_amb['chaude'] & delta_P['important'], Performance['tres_faible']),
        ctrl.Rule(T_amb['chaude'], Performance['faible']),
        ctrl.Rule(LF['eleve'] & delta_P['minime'], Performance['bonne']),
        ctrl.Rule(LF['faible'] | delta_P['important'], Performance['faible']),
    ]

fis_ctrl = ctrl.ControlSystem(rules)
fis = ctrl.ControlSystemSimulation(fis_ctrl)

# =============================================================================
# Application FIS
# =============================================================================
print("\n🔧 Application du FIS sur tous les échantillons...")

predictions = []
errors_count = 0

for idx, row in df.iterrows():
    try:
        fis.input['T_amb'] = float(row['T_amb'])
        if use_hr and 'HR' in row:
            fis.input['HR'] = float(row['HR'])
        fis.input['LF'] = float(row['LF'])
        fis.input['delta_P'] = float(row['delta_P'])
        
        fis.compute()
        predictions.append(fis.output['Performance_Climat'])
    except Exception as e:
        predictions.append(np.nan)
        errors_count += 1

df['FIS_Performance'] = predictions
print(f"✅ Inférence terminée: {np.sum(~np.isnan(predictions))}/{len(df)} prédictions valides")
if errors_count > 0:
    print(f"⚠️ {errors_count} erreurs d'inférence")

# =============================================================================
# Métriques de validation
# =============================================================================
print("\n📈 Calcul des métriques de validation...")

y_true = df['rendement'].values * 100  # Conversion en %
y_pred = np.array(predictions)
mask = ~np.isnan(y_pred) & ~np.isnan(y_true)

if np.sum(mask) > 10:
    rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
    r2 = r2_score(y_true[mask], y_pred[mask])
    mae = mean_absolute_error(y_true[mask], y_pred[mask])
    
    print(f"   • RMSE: {rmse:.2f}%")
    print(f"   • R²: {r2:.3f}")
    print(f"   • MAE: {mae:.2f}%")
    
    metrics = {"rmse": float(rmse), "r2": float(r2), "mae": float(mae)}
else:
    print("⚠️ Pas assez de données valides pour les métriques")
    metrics = {}

# =============================================================================
# Visualisations
# =============================================================================
print("\n📊 Génération des graphiques...")

# Créer dossier figures
figures_dir = f"{INPUT_DIR}/figures"
os.makedirs(figures_dir, exist_ok=True)

# Graphique 1: FIS vs Réel
plt.figure(figsize=(10, 6))
plt.scatter(y_true[mask], y_pred[mask], alpha=0.5, s=20, c='blue')
plt.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Référence (y=x)')
plt.xlabel('Rendement réel (%)')
plt.ylabel('Performance prédite par FIS (%)')
plt.title(f'FIS vs Rendement Réel — Configuration {CONFIG}')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

fig1_path = f"{figures_dir}/fis_vs_reel_{CONFIG}.png"
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ FIS vs Réel: {fig1_path}")

# Graphique 2: Heatmap (si quadruplet)
if use_hr:
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
    
    fig2_path = f"{figures_dir}/fis_heatmap_{CONFIG}.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Heatmap: {fig2_path}")

# =============================================================================
# Sauvegarde
# =============================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# CSV
output_csv = f"{INPUT_DIR}/fis_output_{timestamp}.csv"
df.to_csv(output_csv, index=False)
print(f"\n💾 CSV sauvegardé: {output_csv}")

# JSON summary
summary = {
    "timestamp": timestamp,
    "configuration": CONFIG,
    "input_file": os.path.basename(FCM_FILE),
    "total_samples": len(df),
    "valid_predictions": int(np.sum(mask)),
    "metrics": metrics,
    "rules_count": len(rules),
    "output_csv": output_csv,
    "figures": [fig1_path] + ([fig2_path] if use_hr else [])
}

summary_json = f"{INPUT_DIR}/fis_summary_{timestamp}.json"
with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"📁 Summary JSON: {summary_json}")

print("\n" + "="*70)
print("✅ FIS TERMINÉ")
print("="*70)
print(f"\n📌 PROCHAINE ÉTAPE:")
print(f"   Lance: python 02_scripts/05_pso_optimization.py")