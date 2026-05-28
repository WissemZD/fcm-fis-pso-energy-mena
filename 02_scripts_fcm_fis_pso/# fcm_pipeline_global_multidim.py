# fcm_pipeline_global_multidim.py
# ✅ Clustering FCM GLOBAL + Visualisation Multi-Dimensionnelle
# ✅ Compatible thèse FCM-FIS-PSO-KMENA

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import json
from datetime import datetime

print("🌍 Clustering FCM GLOBAL — Multi-Dimensionnel\n")

# =============================================================================
# Configuration des chemins
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "01_data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "03_results")
os.makedirs(f"{RESULTS_DIR}/figures", exist_ok=True)

CSV_FILE = os.path.join(DATA_DIR, "dataset_MENA_FCM_v2.csv")

# =============================================================================
# 1. Chargement & Pré-traitement
# =============================================================================
print("📥 Chargement du dataset global...")
df = pd.read_csv(CSV_FILE)
print(f"✅ Données brutes: {len(df)} lignes, {len(df.columns)} colonnes")

# Features sélectionnées (Quadruplet optimal)
features = ["T_amb", "HR", "LF", "delta_P"]
print(f"🎯 Features retenues: {features}")

# Nettoyage: supprimer lignes avec NaN sur les features critiques
df_clean = df.dropna(subset=features).copy()
print(f"🧹 Après nettoyage NaN: {len(df_clean)} lignes conservées\n")

# Normalisation (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[features].values)

# =============================================================================
# 2. Clustering Fuzzy C-Means (Global)
# =============================================================================
print(" Application FCM (c=3 clusters) sur l'ensemble global...")
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data=X_scaled.T,  # FCM attend (features, samples)
    c=3,
    m=2.0,
    error=0.005,
    maxiter=1000,
    init=None
)

# Attribution des clusters
cluster_labels = np.argmax(u, axis=0)
df_clean["fcm_cluster_global"] = cluster_labels

print("✅ FCM terminé — Distribution globale:")
print(df_clean["fcm_cluster_global"].value_counts().sort_index())
print(f"📊 Indice de compacité (FPC): {fpc:.4f} (proche de 1 = bon clustering)\n")

# =============================================================================
# 3. Visualisation Multi-Dimensionnelle (6 combinaisons de paires)
# =============================================================================
print("📈 Génération des graphiques multi-dimensionnels...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

feature_pairs = [
    (0, 1, "T_amb", "HR"),
    (0, 2, "T_amb", "LF"),
    (0, 3, "T_amb", "delta_P"),
    (1, 2, "HR", "LF"),
    (1, 3, "HR", "delta_P"),
    (2, 3, "LF", "delta_P")
]

for idx, (i, j, label_x, label_y) in enumerate(feature_pairs):
    scatter = axes[idx].scatter(
        X_scaled[:, i], X_scaled[:, j],
        c=cluster_labels, cmap='viridis', alpha=0.7, s=25, edgecolors='k', linewidth=0.3
    )
    axes[idx].set_xlabel(f"{label_x} (normalisé)")
    axes[idx].set_ylabel(f"{label_y} (normalisé)")
    axes[idx].set_title(f"{label_x} vs {label_y}")
    axes[idx].grid(alpha=0.3)

# Légende unique
plt.figlegend(*scatter.legend_elements(), loc="lower right", title="Clusters FCM", bbox_to_anchor=(0.92, 0.15))
plt.suptitle("Clustering FCM Global — Visualisation Multi-Dimensionnelle (Toutes Sites/Régimes)", fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

output_fig = f"{RESULTS_DIR}/figures/fcm_global_multidim.png"
plt.savefig(output_fig, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Graphique sauvegardé: {output_fig}\n")

# =============================================================================
# 4. Sauvegarde des résultats
# =============================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_csv = f"{RESULTS_DIR}/fcm_global_output_{timestamp}.csv"
df_clean.to_csv(output_csv, index=False)
print(f" CSV sauvegardé: {output_csv}")

# Résumé JSON pour le mémoire
summary = {
    "timestamp": timestamp,
    "strategy": "Global (all sites & regimes)",
    "features": features,
    "total_raw_rows": len(df),
    "clean_rows": len(df_clean),
    "n_clusters": 3,
    "fpc_score": float(fpc),
    "cluster_distribution": df_clean["fcm_cluster_global"].value_counts().to_dict(),
    "output_csv": output_csv,
    "output_fig": output_fig
}
with open(f"{RESULTS_DIR}/fcm_global_summary_{timestamp}.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"📁 Résumé JSON sauvegardé\n")

print("✅ Clustering FCM GLOBAL terminé ! Prêt pour validation FIS 🚀")