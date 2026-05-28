#!/usr/bin/env python3
"""
pso_optimization.py
===================
Optimisation des coefficients de règles FIS par Particle Swarm Optimization (PSO).

Objectif : Ajuster les poids des règles pour minimiser le RMSE entre 
           la prédiction FIS et le rendement réel des machines industrielles.

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
CONFIG = "triplet"  # Change en "triplet" pour optimiser l'autre
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
    
    # ✅ CORRECTION: Créer les règles comme une liste explicite
    if use_hr:
        rules_list = [
            ctrl.Rule(T_amb['fraiche'] & HR['seche'] & LF['eleve'] & delta_P['minime'], Performance['excellente']),
            ctrl.Rule(T_amb['moderee'] & HR['normale'] & LF['moyen'] & delta_P['modere'], Performance['moyenne']),
            ctrl.Rule(T_amb['chaude'] & HR['humide'] & delta_P['important'], Performance['tres_faible']),
            ctrl.Rule(T_amb['chaude'] & HR['humide'], Performance['faible']),
            ctrl.Rule(T_amb['fraiche'] & LF['eleve'] & delta_P['minime'], Performance['bonne']),
            ctrl.Rule(LF['faible'] | delta_P['important'], Performance['faible']),
        ]
    else:
        rules_list = [
            ctrl.Rule(T_amb['fraiche'] & LF['eleve'] & delta_P['minime'], Performance['excellente']),
            ctrl.Rule(T_amb['moderee'] & LF['moyen'] & delta_P['modere'], Performance['moyenne']),
            ctrl.Rule(T_amb['chaude'] & delta_P['important'], Performance['tres_faible']),
            ctrl.Rule(T_amb['chaude'], Performance['faible']),
            ctrl.Rule(LF['eleve'] & delta_P['minime'], Performance['bonne']),
            ctrl.Rule(LF['faible'] | delta_P['important'], Performance['faible']),
        ]
    
    # Créer le système avec la liste de règles
    fis_ctrl = ctrl.ControlSystem(rules_list)
    
    return fis_ctrl, use_hr, len(rules_list)

# =============================================================================
# Fonction Coût pour PSO (RMSE à minimiser)
# =============================================================================
def objective_function(weights, df_data, use_hr):
    """
    Calcule le RMSE pour un ensemble de poids de règles
    """
    # Recréer le FIS avec les poids actuels
    fis_sys, _, n_rules = create_fis_system(use_hr)
    sim = ctrl.ControlSystemSimulation(fis_sys)
    
    predictions = []
    
    # Appliquer les poids aux règles
    for i, rule in enumerate(fis_sys.rules):
        if i < len(weights):
            rule.weight = weights[i]
    
    # Simulation rapide sur un échantillon (pour accélérer PSO)
    sample = df_data.sample(min(300, len(df_data)), random_state=42)
    
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
            predictions.append(50)  # Valeur par défaut en cas d'erreur
            
    y_true = sample['rendement'].values * 100
    y_pred = np.array(predictions)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

# =============================================================================
# Lancement PSO
# =============================================================================
print("\n⚙️ Configuration PSO...")
fis_sys, use_hr_flag, n_rules = create_fis_system(CONFIG == "quadruplet")

# Bornes: [0.1, 5.0] pour permettre d'amplifier ou réduire fortement les règles
bounds = (np.ones(n_rules) * 0.1, np.ones(n_rules) * 5.0)
options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}  # Paramètres plus stables

print(f"   • Nombre de règles à optimiser: {n_rules}")
print(f"   • Bornes des poids: [0.1, 5.0]")
print(f"   • Particules: 30")
print(f"   • Itérations: 100")

optimizer = ps.single.GlobalBestPSO(n_particles=30, dimensions=n_rules, options=options, bounds=bounds)

# Wrapper pour pyswarms
def cost_wrapper(weights):
    return np.array([objective_function(w, df, use_hr_flag) for w in weights])

print("\n🔍 Lancement de l'optimisation (cela peut prendre 5-10 minutes)...")
best_cost, best_weights = optimizer.optimize(cost_wrapper, iters=100)

print(f"\n✅ Optimisation terminée !")
print(f"📉 Meilleur RMSE: {best_cost:.2f}%")
print(f"⚖️ Poids optimaux: {best_weights}")

# =============================================================================
# Visualisation Convergence
# =============================================================================
plt.figure(figsize=(10, 5))
plt.plot(optimizer.cost_history, linewidth=2, color='purple')
plt.title(f'Convergence PSO — Configuration {CONFIG}\nRMSE optimal: {best_cost:.2f}%')
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
rule_names = [f"Règle_{i+1}" for i in range(n_rules)]
summary = {
    "timestamp": datetime.now().isoformat(),
    "configuration": CONFIG,
    "best_rmse": float(best_cost),
    "optimal_weights": dict(zip(rule_names, [float(w) for w in best_weights])),
    "convergence_iterations": len(optimizer.cost_history),
    "pso_parameters": {
        "n_particles": 30,
        "n_iterations": 100,
        "c1": 1.5,
        "c2": 1.5,
        "w": 0.7,
        "bounds": [0.1, 5.0]
    }
}

out_json = os.path.join(INPUT_DIR, f"pso_optimization_{CONFIG}.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\n💾 Résultats PSO sauvegardés: {out_json}")

# Affichage interprétation
print("\n" + "="*70)
print("📊 INTERPRÉTATION DES POIDS OPTIMAUX")
print("="*70)
for i, (name, weight) in enumerate(summary["optimal_weights"].items()):
    impact = "renforcée" if weight > 1.5 else "réduite" if weight < 0.8 else "stable"
    print(f"   • {name}: poids = {weight:.2f} → Règle {impact}")

print("\n📌 PROCHAINE ÉTAPE:")
print("   1. Lance le même script avec CONFIG = 'triplet'")
print("   2. Puis lance: python 02_scripts/06_compare_results.py")