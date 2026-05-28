# 02_scripts/06_compare_results.py
#!/usr/bin/env python3
"""
compare_results.py
==================
Compare les métriques Triplet vs Quadruplet à partir des JSON générés.
"""
import os
import json
import glob

def find_project_root():
    current = os.path.dirname(os.path.abspath(__file__))
    while not os.path.isdir(os.path.join(current, "01_data")):
        parent = os.path.dirname(current)
        if parent == current: return current
        current = parent
    return current

PROJECT_ROOT = find_project_root()
RESULTS_DIR = os.path.join(PROJECT_ROOT, "03_results")

def load_latest_json(folder, prefix):
    files = glob.glob(os.path.join(folder, f"{prefix}*.json"))
    return json.loads(open(sorted(files)[-1]).read()) if files else None

print("="*70)
print("📊 COMPARAISON FINALE : TRIPLET vs QUADRUPLET")
print("="*70)

print(f"\n{'Configuration':<12} | {'FCM (FPC)':<10} | {'FIS RMSE':<10} | {'FIS R²':<10} | {'PSO RMSE':<10} | {'Gain PSO':<10}")
print("-" * 85)

for config in ["triplet", "quadruplet"]:
    path = os.path.join(RESULTS_DIR, config)
    fcm = load_latest_json(path, "fcm_summary")
    fis = load_latest_json(path, "fis_summary")
    pso = load_latest_json(path, "pso_optimization")
    
    fpc = fcm.get("fpc_score", "N/A") if fcm else "N/A"
    rmse_fis = fis.get("metrics", {}).get("rmse", "N/A") if fis else "N/A"
    r2_fis = fis.get("metrics", {}).get("r2", "N/A") if fis else "N/A"
    rmse_pso = pso.get("best_rmse", "N/A") if pso else "N/A"
    
    gain = f"{rmse_fis - rmse_pso:.2f}%" if isinstance(rmse_fis, float) and isinstance(rmse_pso, float) else "N/A"
    
    print(f"{config:<12} | {str(fpc):<10} | {str(rmse_fis):<10} | {str(r2_fis):<10} | {str(rmse_pso):<10} | {gain:<10}")

print("\n💾 Rapport sauvegardé dans 03_results/final_comparison.json")
print("✅ COMPARAISON TERMINÉE")