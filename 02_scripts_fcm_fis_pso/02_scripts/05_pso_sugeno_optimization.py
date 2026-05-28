"""
## 🐛 2. Correction du Script `05_pso_sugeno_optimization.py`

**Cause de l'erreur** :
1. `ValueError: matmul... size 150` → `pyswarms` passe un tableau `(30 particules, 30 paramètres)`. Ton code faisait `weights.reshape(6, -1)` sur l'ensemble du tableau, créant une matrice `(6, 150)` au lieu de traiter chaque particule individuellement.
2. `divide by zero` → La fonction triangulaire `tri(x, 15, 15, 30)` divise par `(b-a) = 0`.

**✅ Script corrigé & optimisé** :

```python
# 02_scripts/05_pso_sugeno_optimization.py
#!/usr/bin/env python3

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
import matplotlib.pyplot as plt
import json
from datetime import datetime
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")  # Ignore les warnings numpy pour lisibilité

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

CONFIG = "triplet"  # Change en "triplet" si besoin
INPUT_DIR = os.path.join(RESULTS_DIR, CONFIG)

import glob
fis_files = glob.glob(os.path.join(INPUT_DIR, "fis_output_*.csv"))
if not fis_files:
    print(f"❌ Aucun fichier FIS trouvé dans {INPUT_DIR}"); exit(1)

df = pd.read_csv(sorted(fis_files)[-1])
print(f"📥 Données chargées: {len(df)} lignes")

features = ["T_amb", "HR", "LF", "delta_P"] if CONFIG == "quadruplet" else ["T_amb", "LF", "delta_P"]
X = df[features].values
y_true = df["rendement"].values * 100

# =============================================================================
# 1. Fonctions d'appartenance (Sécurisées)
# =============================================================================
def safe_trimf(x, a, b, c, eps=1e-8):
    return np.maximum(0, np.minimum((x - a) / (b - a + eps), (c - x) / (c - b + eps)))

def compute_firing_strengths(X, config):
    T = X[:, 0]
    if config == "quadruplet":
        HR = X[:, 1]; LF = X[:, 2]; dP = X[:, 3]
    else:
        LF = X[:, 1]; dP = X[:, 2]
        
    T_fraiche = safe_trimf(T, 15, 16, 30)
    T_moderee = safe_trimf(T, 20, 35, 45)
    T_chaude  = safe_trimf(T, 35, 50, 50)
    
    if config == "quadruplet":
        HR_seche   = safe_trimf(HR, 20, 21, 40)
        HR_normale = safe_trimf(HR, 30, 50, 65)
        HR_humide  = safe_trimf(HR, 55, 80, 80)
        
        f1 = T_fraiche * HR_seche   * safe_trimf(LF, 0.7, 0.9, 1.0) * safe_trimf(dP, 0, 0, 5)
        f2 = T_moderee * HR_normale * safe_trimf(LF, 0.4, 0.6, 0.8) * safe_trimf(dP, 5, 12, 20)
        f3 = T_chaude  * HR_humide  * safe_trimf(LF, 0, 0, 0.4)     * safe_trimf(dP, 15, 25, 35)
        f4 = T_chaude  * HR_humide
        f5 = T_fraiche * safe_trimf(LF, 0.7, 0.9, 1.0) * safe_trimf(dP, 0, 0, 5)
        f6 = (1 - safe_trimf(LF, 0, 0, 0.4)) * (1 - safe_trimf(dP, 15, 25, 35))
    else:
        f1 = T_fraiche * safe_trimf(LF, 0.7, 0.9, 1.0) * safe_trimf(dP, 0, 0, 5)
        f2 = T_moderee * safe_trimf(LF, 0.4, 0.6, 0.8) * safe_trimf(dP, 5, 12, 20)
        f3 = T_chaude  * safe_trimf(LF, 0, 0, 0.4)     * safe_trimf(dP, 15, 25, 35)
        f4 = T_chaude
        f5 = T_fraiche * safe_trimf(LF, 0.7, 0.9, 1.0) * safe_trimf(dP, 0, 0, 5)
        f6 = (1 - safe_trimf(LF, 0, 0, 0.4)) * (1 - safe_trimf(dP, 15, 25, 35))
        
    F = np.column_stack([f1, f2, f3, f4, f5, f6])
    return F / (F.sum(axis=1, keepdims=True) + 1e-8)

F_norm = compute_firing_strengths(X, CONFIG)
X_ext = np.column_stack([X, np.ones(len(X))])  # Ajout biais
n_rules = 6
n_coeffs = X_ext.shape[1]  # 4 ou 5
dims = n_rules * n_coeffs

# =============================================================================
# 2. Fonction Coût (Batch-friendly pour pyswarms)
# =============================================================================
def objective_function(weights):
    n_particles = weights.shape[0]
    costs = np.zeros(n_particles)
    
    for i in range(n_particles):
        # Reshape par particule : (6 règles, 5 coeffs)
        coeffs = weights[i].reshape(n_rules, n_coeffs)
        
        # Prédiction vectorisée : (3000, 5) @ (5, 6) -> (3000, 6)
        linear_outputs = X_ext @ coeffs.T
        y_pred = np.sum(F_norm * linear_outputs, axis=1)
        
        costs[i] = np.sqrt(mean_squared_error(y_true, y_pred))
    return costs

# =============================================================================
# 3. Lancement PSO
# =============================================================================
bounds = (np.full(dims, -5.0), np.full(dims, 5.0))
options = {'c1': 2.0, 'c2': 2.0, 'w': 0.8}

print(f"\n⚙️ PSO Config: {dims} paramètres, bornes [-5, 5], 50 itérations")
optimizer = ps.single.GlobalBestPSO(n_particles=30, dimensions=dims, options=options, bounds=bounds)

print("🔍 Optimisation en cours (vectorisée ~2-3 min)...")
best_cost, best_pos = optimizer.optimize(objective_function, iters=50)

best_coeffs = best_pos.reshape(n_rules, n_coeffs)
print(f"\n✅ Terminé ! RMSE optimal: {best_cost:.2f}%")

# =============================================================================
# 4. Sauvegarde & Visualisation
# =============================================================================
plt.figure(figsize=(8,4))
plt.plot(optimizer.cost_history, linewidth=2, color='purple')
plt.title(f'Convergence PSO Sugeno — {CONFIG}\nRMSE optimal: {best_cost:.2f}%')
plt.xlabel('Itération'); plt.ylabel('RMSE (%)')
plt.grid(alpha=0.3); plt.tight_layout()

fig_path = os.path.join(INPUT_DIR, "figures", f"pso_sugeno_convergence_{CONFIG}.png")
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
plt.savefig(fig_path, dpi=300); plt.close()
print(f"📊 Graphique sauvegardé: {fig_path}")

rule_names = [f"R{i+1}" for i in range(6)]
feature_names = features + ["bias"]
summary = {
    "config": CONFIG,
    "best_rmse": float(best_cost),
    "coefficients": {rule: dict(zip(feature_names, best_coeffs[i].tolist())) for i, rule in enumerate(rule_names)},
    "timestamp": datetime.now().isoformat()
}

out_json = os.path.join(INPUT_DIR, f"pso_sugeno_{CONFIG}.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"💾 Sauvegardé: {out_json}")
print("📌 Lance maintenant: python 02_scripts/06_compare_results.py")