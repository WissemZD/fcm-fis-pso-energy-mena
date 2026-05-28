# check_paths.py
import os

PROJECT_ROOT = r"C:\projets\memoire_FCM"
checks = {
    "Project root exists": os.path.isdir(PROJECT_ROOT),
    "01_data exists": os.path.isdir(os.path.join(PROJECT_ROOT, "01_data")),
    "dataset_MENA_FCM_v2.csv": os.path.isfile(os.path.join(PROJECT_ROOT, "01_data", "dataset_MENA_FCM_v2.csv")),
    "02_scripts_fcm_fis_pso exists": os.path.isdir(os.path.join(PROJECT_ROOT, "02_scripts_fcm_fis_pso")),
    "03_results exists": os.path.isdir(os.path.join(PROJECT_ROOT, "03_results")),
}

print("🔍 Vérification des chemins:")
all_ok = True
for name, result in checks.items():
    status = "✅" if result else "❌"
    print(f"   {status} {name}")
    if not result:
        all_ok = False

if all_ok:
    print("\n🎉 Tous les chemins sont valides !")
else:
    print("\n⚠️ Corrige les chemins manquants avant de continuer.")