# fcm_pipeline_direct.py - VERSION ROBUSTE ✅
import os
import sys
import pandas as pd
import numpy as np
try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("Erreur: sklearn n'est pas installé. Installez-le avec 'pip install scikit-learn'")
    sys.exit(1)
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# 📍 Détection automatique du dossier racine du projet
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "01_data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "03_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CSV_FILE = os.path.join(DATA_DIR, "dataset_MENA_FCM_v2.csv")

print(" Pipeline FCM — Mode CSV direct\n")
print(f"📂 Racine projet : {PROJECT_ROOT}")

# Charger les données
df = pd.read_csv(CSV_FILE)
print(f"✅ Dataset chargé: {len(df)} lignes")
print(f"📋 Colonnes: {list(df.columns)}")

# =============================================================================
# 2. Filtrer pour un site et régime spécifique (exemple: Sfax, regime_ref=1)
# =============================================================================
site_cible = "Site_1"
regime_cible = 2

df_filtered = df[(df["site"] == site_cible) & (df["regime_ref"] == regime_cible)].copy()
print(f"\n🎯 Données filtrées: {len(df_filtered)} lignes pour {site_cible} (regime {regime_cible})")

if len(df_filtered) == 0:
    print("⚠️ Aucune donnée pour ce filtre — essaye un autre site/regime")
    exit(0)

# =============================================================================
# 3. Préparer les features pour FCM
# =============================================================================
features = ["T_amb", "HR", "LF", "delta_P"]  # À adapter selon ton modèle
X = df_filtered[features].dropna().values

print(f"\n📊 Features pour FCM: {features}")
print(f"   Shape: {X.shape}")

# Normalisation (important pour FCM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =============================================================================
# 4. Appliquer Fuzzy C-Means (FCM)
# =============================================================================
print(f"\n🔍 Clustering FCM (c=3 clusters)...")

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data=X_scaled.T,  # FCM attend (features, samples)
    c=3,              # Nombre de clusters
    m=2,              # Fuzziness parameter
    error=0.005,
    maxiter=1000,
    init=None
)

# Assigner chaque point au cluster de plus forte appartenance
cluster_labels = np.argmax(u, axis=0)
df_filtered["fcm_cluster"] = cluster_labels

print(f"✅ FCM terminé — Distribution des clusters:")
print(df_filtered["fcm_cluster"].value_counts().sort_index())

# =============================================================================
# 5. (Optionnel) Visualisation rapide
# =============================================================================
if len(features) >= 2:
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                         c=cluster_labels, cmap='viridis', alpha=0.7)
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title(f"FCM Clustering — {site_cible} (regime {regime_cible})")
    plt.colorbar(label="Cluster")
    plt.tight_layout()
    plt.savefig(f"fcm_{site_cible}_regime{regime_cible}.png", dpi=300)
    print(f"📊 Graphique sauvegardé: fcm_{site_cible}_regime{regime_cible}.png")
    plt.close()

# =============================================================================
# 6. Sauvegarder les résultats pour la suite (FIS/PSO)
# =============================================================================
output_file = f"fcm_output_{site_cible}_regime{regime_cible}.csv"
df_filtered.to_csv(output_file, index=False)
print(f"\n💾 Résultats sauvegardés dans '{output_file}'")

print("\n✅ Étape FCM terminée ! Prêt pour FIS → PSO 🚀")