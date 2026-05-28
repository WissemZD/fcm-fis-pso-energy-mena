# 02_scripts/05_pso_sugeno_optimization.py
#!/usr/bin/env python3
"""
pso_sugeno_optimization.py
==========================
Optimisation PSO des coefficients d'un FIS SUGENO (1er ordre).
Chaque règle : y_i = a*T_amb + b*HR + c*LF + d*delta_P + e
PSO optimise [a,b,c,d,e] pour chaque règle.
"""
import os
import numpy as np
import pandas as pd
import pyswarms as ps
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import json
from datetime import datetime
from sklearn.metrics import mean_squared_error

print("="*70)
print("🐦 OPTIMISATION PSO — FIS SUGENO (1er ORDRE)")
print("="*70)

def find_project_root():
    current = os.path.dirname(os.path.abspath(__file__))
    while not os.path.isdir(os.path.join(current, "01_data")):
        parent = os.path.dirname(current)
        if parent == current: break
        current = parent
    return current

PROJECT_ROOT = find_project_root()
RESULTS_DIR = os.path.join(PROJECT_ROOT, "03_results")

CONFIG = "quadruplet"  # Change en "triplet" si besoin
INPUT_DIR = os.path.join(RESULTS_DIR, CONFIG)

import glob
fis_files = glob.glob(os.path.join(INPUT_DIR, "fis_output_*.csv"))
if not fis_files:
    print(f"❌ Aucun fichier FIS trouvé dans {INPUT_DIR}"); exit(1)

df = pd.read_csv(sorted(fis_files)[-1])
print(f"📥 Données chargées: {len(df)} lignes")

# Features selon config
features = ["T_amb", "HR", "LF", "delta_P"] if CONFIG == "quadruplet" else ["T_amb", "LF", "delta_P"]
X = df[features].values
y_true = df["rendement"].values * 100  # Rendement en %

# =============================================================================
# 1. Définition des fonctions d'appartenance (identique au clustering FCM)
# =============================================================================
def compute_firing_strengths(X, config):
    """Calcule les forces d'activation des 6 règles (produit)"""
    T = X[:, 0]
    if config == "quadruplet":
        HR = X[:, 1]; LF = X[:, 2]; dP = X[:, 3]
    else:
        LF = X[:, 1]; dP = X[:, 2]
        
    # Triangulaires simplifiées pour rapidité
    def tri(x, a, b, c):
        return np.maximum(0, np.minimum((x-a)/(b-a), (c-x)/(c-b)))
        
    T_fraiche = tri(T, 15, 15, 30)
    T_moderee = tri(T, 20, 35, 45)
    T_chaude  = tri(T, 35, 50, 50)
    
    if config == "quadruplet":
        HR_seche   = tri(HR, 20, 20, 40)
        HR_normale = tri(HR, 30, 50, 65)
        HR_humide  = tri(HR, 55, 80, 80)
        
        f1 = T_fraiche * HR_seche   * tri(LF, 0.7, 0.9, 1.0) * tri(dP, 0, 0, 5)
        f2 = T_moderee * HR_normale * tri(LF, 0.4, 0.6, 0.8) * tri(dP, 5, 12, 20)
        f3 = T_chaude  * HR_humide  * tri(LF, 0, 0, 0.4)     * tri(dP, 15, 25, 35)
        f4 = T_chaude  * HR_humide
        f5 = T_fraiche * tri(LF, 0.7, 0.9, 1.0) * tri(dP, 0, 0, 5)
        f6 = (1-tri(LF, 0, 0, 0.4)) * (1-tri(dP, 15, 25, 35)) # Simplification OU
    else:
        f1 = T_fraiche * tri(LF, 0.7, 0.9, 1.0) * tri(dP, 0, 0, 5)
        f2 = T_moderee * tri(LF, 0.4, 0.6, 0.8) * tri(dP, 5, 12, 20)
        f3 = T_chaude  * tri(LF, 0, 0, 0.4)     * tri(dP, 15, 25, 35)
        f4 = T_chaude
        f5 = T_fraiche * tri(LF, 0.7, 0.9, 1.0) * tri(dP, 0, 0, 5)
        f6 = (1-tri(LF, 0, 0, 0.4)) * (1-tri(dP, 15, 25, 35))
        
    return np.column_stack([f1, f2, f3, f4, f5, f6])

F = compute_firing_strengths(X, CONFIG)
F_sum = F.sum(axis=1, keepdims=True)
F_norm = F / (F_sum + 1e-8)  # [N, 6]

# =============================================================================
# 2. Fonction coût PSO (Vectorisée → très rapide)
# =============================================================================
def sugeno_predict(coeffs, F_norm, X):
    """coeffs shape: (6, n_features+1)"""
    # X_extended: ajoute une colonne de 1 pour le biais e_i
    X_ext = np.column_stack([X, np.ones(len(X))])
    # Prediction = sum( F_norm_i * (X_ext @ coeffs_i) )
    # Vectorisé : (N, 6) @ (6, D) -> (N, D) puis somme sur features
    linear_outputs = X_ext @ coeffs.T  # (N, 6)
    return np.sum(F_norm * linear_outputs, axis=1)

def objective_function(weights):
    # Reshape weights en (6 règles × 5 coeffs)
    coeffs = weights.reshape(6, -1)
    y_pred = sugeno_predict(coeffs, F_norm, X)
    return np.sqrt(mean_squared_error(y_true, y_pred))

# =============================================================================
# 3. Lancement PSO
# =============================================================================
n_rules = 6
n_coeffs = len(features) + 1  # 4 ou 5
dims = n_rules * n_coeffs

bounds = (np.full(dims, -5.0), np.full(dims, 5.0))
options = {'c1': 2.0, 'c2': 2.0, 'w': 0.8}

print(f"\n⚙️ PSO Config: {dims} paramètres, bornes [-5, 5], 50 itérations")
optimizer = ps.single.GlobalBestPSO(n_particles=30, dimensions=dims, options=options, bounds=bounds)

print("🔍 Optimisation en cours (vectorisée ~2-3 min)...")
best_cost, best_pos = optimizer.optimize(objective_function, iters=50)

best_coeffs = best_pos.reshape(6, n_coeffs)
print(f"\n✅ Terminé ! RMSE optimal: {best_cost:.2f}%")

# =============================================================================
# 4. Sauvegarde & Visualisation
# =============================================================================
plt.figure(figsize=(8,4))
plt.plot(optimizer.cost_history, linewidth=2)
plt.title(f'Convergence PSO Sugeno — {CONFIG}'); plt.xlabel('Itération'); plt.ylabel('RMSE (%)')
plt.grid(alpha=0.3); plt.tight_layout()
fig_path = os.path.join(INPUT_DIR, "figures", f"pso_sugeno_convergence_{CONFIG}.png")
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
plt.savefig(fig_path, dpi=300); plt.close()

rule_names = [f"R{i+1}" for i in range(6)]
feature_names = features + ["bias"]
summary = {
    "config": CONFIG,
    "best_rmse": float(best_cost),
    "coefficients": {rule: dict(zip(feature_names, best_coeffs[i].tolist())) for i, rule in enumerate(rule_names)},
    "timestamp": datetime.now().isoformat()
}
out_json = os.path.join(INPUT_DIR, f"pso_sugeno_{CONFIG}.json")
with open(out_json, "w") as f: json.dump(summary, f, indent=2)

print(f"💾 Sauvegardé: {out_json}")
print("📌 Lance maintenant 06_compare_results.py")