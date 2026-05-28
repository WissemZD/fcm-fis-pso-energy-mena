#!/usr/bin/env python3
"""
pso_optimization.py
===================
Optimisation des coefficients de règles FIS par Particle Swarm Optimization (PSO).

Objectif : Ajuster les poids des règles pour minimiser le RMSE entre 
           la prédiction FIS et le rendement réel.

Auteur : Wissem ZD
Date : 2026
"""

import os
import sys
import numpy as np
import pandas as pd
import pyswarms as ps
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import json
from datetime import datetime
from sklearn.metrics import mean_squared_error

print("="*70)
print("🐦 OPTIMISATION PSO DES COEFFICIENTS DE BIAIS CLIMATIQUE")
print("="*70)

# =============================================================================
# DÉTECTION ROBUSTE DU PROJET ROOT
# =============================================================================
def find_project_root(start_path=None, marker=".git"):
    if start_path is None:
        start_path = os.path.dirname(os.path.abspath(__file__))
    current = os.path.abspath(start_path)
    while True:
        if os.path.isdir(os.path.join(current, "01_data")):
            return current
        parent = os.path.dirname(current)
        if parent == current: break
        current = parent
    return os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = find_project_root()
RESULTS_DIR = os.path.join(PROJECT_ROOT, "03_results")

print(f"📂 Projet root détecté: {PROJECT_ROOT}")

# =============================================================================
# Configuration
# =============================================================================
CONFIG = "quadruplet"  # Change en "triplet" pour optimiser l'autre
INPUT_DIR = os.path.join(RESULTS_DIR, CONFIG)

print(f"🔍 Configuration cible: {CONFIG}")

# Charger fichier FIS le plus récent
import glob
fis_files = glob.glob(os.path.join(INPUT_DIR, "fis_output_*.csv"))
if not fis_files:
    print(f"❌ Aucun fichier FIS trouvé dans {INPUT_DIR}")
    exit(1)

FIS_FILE = sorted(fis_files)[-1]
df = pd.read_csv(FIS_FILE)
print(f"📥 Données FIS chargées: {len(df)} lignes")

# =============================================================================
# Définition du Système FIS (Identique à 04_fis_inference.py)
# =============================================================================
def create_fis_system(use_hr=True):
    """Recrée l'architecture FIS pour évaluation"""
    
    # Variables
    T_amb = ctrl.Antecedent(np.linspace(15, 55, 100), 'T_amb')
    T_amb['fraiche'] = fuzz.gaussmf(T_amb.universe, 22, 5)
    T_amb['moderee'] = fuzz.gaussmf(T_amb.universe, 32, 6)
    T_amb['chaude'] = fuzz.gaussmf(T_amb.universe, 42, 5)
    
    if use_hr:
        HR = ctrl.Antecedent(np.linspace(20, 80, 100), 'HR')
        HR['seche'] = fuzz.gaussmf(HR.universe, 30, 8)
        HR['normale'] = fuzz.gaussmf(HR.universe, 50, 10)
        HR['humide'] = fuzz.gaussmf(HR.universe, 70, 8)
    
    LF = ctrl.Antecedent(np.linspace(0, 1.2, 100), 'LF')
    LF['faible'] = fuzz.gaussmf(LF.universe, 0.3, 0.15)
    LF['moyen'] = fuzz.gaussmf(LF.universe, 0.6, 0.15)
    LF['eleve'] = fuzz.gaussmf(LF.universe, 0.9, 0.15)
    
    delta_P = ctrl.Antecedent(np.linspace(0, 35, 100), 'delta_P')
    delta_P['minime'] = fuzz.gaussmf(delta_P.universe, 3, 2)
    delta_P['modere'] = fuzz.gaussmf(delta_P.universe, 12, 4)
    delta_P['important'] = fuzz.gaussmf(delta_P.universe, 22, 5)
    
    Performance = ctrl.Consequent(np.linspace(0, 100, 100), 'Performance_Climat')
    Performance['tres_faible'] = fuzz.gaussmf(Performance.universe, 15, 8)
    Performance['faible'] = fuzz.gaussmf(Performance.universe, 35, 10)
    Performance['moyenne'] = fuzz.gaussmf(Performance.universe, 50, 10)
    Performance['bonne'] = fuzz.gaussmf(Performance.universe, 70, 10)
    Performance['excellente'] = fuzz.gaussmf(Performance.universe, 90, 8)
    
    # Règles
    if use_hr:
        rules = [
            ctrl.Rule(T_amb['fraiche'] & HR['seche'] & LF['eleve'] & delta_P['minime'], Performance['excellente']),
            ctrl.Rule(T_amb['moderee'] & HR['normale'] & LF['moyen'] & delta_P['modere'], Performance['moyenne']),
            ctrl.Rule(T_amb['chaude'] & HR['humide'] & delta_P['important'], Performance['tres_faible']),
            ctrl.Rule(T_amb['chaude'] & HR['humide'], Performance['faible']),
            ctrl.Rule(T_amb['fraiche'] & LF['eleve'] & delta_P['minime'], Performance['bonne']),
            ctrl.Rule(LF['faible'] | delta_P['important'], Performance['faible']),
        ]
    else:
        rules = [
            ctrl.Rule(T_amb['fraiche'] & LF['eleve'] & delta_P['minime'], Performance['excellente']),
            ctrl.Rule(T_amb['moderee'] & LF['moyen'] & delta_P['modere'], Performance['moyenne']),
            ctrl.Rule(T_amb['chaude'] & delta_P['important'], Performance['tres_faible']),
            ctrl.Rule(T_amb['chaude'], Performance['faible']),
            ctrl.Rule(LF['eleve'] & delta_P['minime'], Performance['bonne']),
            ctrl.Rule(LF['faible'] | delta_P['important'], Performance['faible']),
        ]
        
    return ctrl.ControlSystem(rules), use_hr

# =============================================================================
# Fonction Coût pour PSO (RMSE à minimiser)
# =============================================================================
def objective_function(weights, df_data, fis_template, use_hr):
    """
    Calcule le RMSE pour un ensemble de poids de règles
    """
    fis_ctrl, _ = fis_template
    sim = ctrl.ControlSystemSimulation(fis_ctrl)
    
    predictions = []
    
    # Appliquer les poids aux règles
    for i, rule in enumerate(fis_ctrl.rules):
        rule.weight = weights[i]
    
    # Simulation rapide sur un échantillon (pour accélérer PSO)
    sample = df_data.sample(min(500, len(df_data)))
    
    for _, row in sample.iterrows():
        try:
            sim.input['T_amb'] = row['T_amb']
            if use_hr and 'HR' in row:
                sim.input['HR'] = row['HR']
            sim.input['LF'] = row['LF']
            sim.input['delta_P'] = row['delta_P']
            sim.compute()
            predictions.append(sim.output['Performance_Climat'])
        except:
            predictions.append(0)
            
    y_true = sample['rendement'].values * 100
    y_pred = np.array(predictions)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

# =============================================================================
# Lancement PSO
# =============================================================================
print("\n⚙️ Configuration PSO...")
fis_sys, use_hr_flag = create_fis_system(CONFIG == "quadruplet")

# 6 poids (un par règle), bornes [0.5, 3.0] pour permettre d'amplifier les règles
n_rules = len(fis_sys.rules)
bounds = (np.ones(n_rules) * 0.5, np.ones(n_rules) * 3.0)
options = {'c1': 2.0, 'c2': 2.0, 'w': 0.9}

print(f"   • Nombre de règles à optimiser: {n_rules}")
print(f"   • Bornes: [0.5, 3.0]")
print(f"   • Itérations: 50")

optimizer = ps.single.GlobalBestPSO(n_particles=30, dimensions=n_rules, options=options, bounds=bounds)

# Wrapper pour pyswarms (qui attend un vecteur de coûts)
def cost_wrapper(weights):
    return np.array([objective_function(w, df, fis_sys, use_hr_flag) for w in weights])

print("\n🔍 Lancement de l'optimisation...")
best_cost, best_weights = optimizer.optimize(cost_wrapper, iters=50)

print(f"\n✅ Optimisation terminée !")
print(f"📉 Meilleur RMSE: {best_cost:.2f}%")
print(f"⚖️ Poids optimaux: {best_weights}")

# =============================================================================
# Visualisation Convergence
# =============================================================================
plt.figure(figsize=(10, 5))
plt.plot(optimizer.cost_history, linewidth=2, color='purple')
plt.title(f'Convergence PSO — Configuration {CONFIG}')
plt.xlabel('Itération')
plt.ylabel('RMSE (%)')
plt.grid(alpha=0.3)
plt.tight_layout()

fig_conv = os.path.join(INPUT_DIR, "figures", f"pso_convergence_{CONFIG}.png")
os.makedirs(os.path.dirname(fig_conv), exist_ok=True)
plt.savefig(fig_conv, dpi=300, bbox_inches='tight')
print(f"📊 Graphique de convergence sauvegardé: {fig_conv}")

# =============================================================================
# Sauvegarde Résultats
# =============================================================================
rule_names = [f"Règle {i+1}" for i in range(n_rules)]
summary = {
    "timestamp": datetime.now().isoformat(),
    "configuration": CONFIG,
    "best_rmse": float(best_cost),
    "optimal_weights": dict(zip(rule_names, best_weights.tolist())),
    "convergence_history_length": len(optimizer.cost_history)
}

out_json = os.path.join(INPUT_DIR, f"pso_optimization_{CONFIG}.json")
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n💾 Résultats PSO sauvegardés: {out_json}")
print("\n📌 PROCHAINE ÉTAPE:")
print("   Lance: python 02_scripts/06_compare_results.py")