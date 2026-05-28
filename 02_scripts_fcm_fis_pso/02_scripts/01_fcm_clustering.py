#!/usr/bin/env python3
"""
fcm_clustering.py
=================
Clustering Fuzzy C-Means (FCM) sur données énergétiques MENA.

Deux configurations :
  - Triplet: T_amb, LF, delta_P
  - Quadruplet: T_amb, HR, LF, delta_P

Objectif : Comparer l'impact de l'humidité (HR) sur la qualité du clustering.

Auteur : Wissem ZD
Date : 2026
"""

import os
import sys
import pandas as pd
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler

print("="*70)
print("🌍 CLUSTERING FUZZY C-MEANS (FCM)")
print("="*70)

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = r"C:\projets\memoire_FCM"
DATA_DIR = os.path.join(PROJECT_ROOT, "01_data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "03_results")

# Dataset corrigé
CSV_FILE = os.path.join(RESULTS_DIR, "dataset_corrected.csv")
if not os.path.exists(CSV_FILE):
    print(f"❌ Dataset corrigé non trouvé. Lance d'abord 03_fix_rendement.py")
    exit(1)

print(f"\n📥 Dataset: {CSV_FILE}")

# =============================================================================
# Charger les données
# =============================================================================
df = pd.read_csv(CSV_FILE)
print(f"✅ Données chargées: {len(df)} lignes")

# =============================================================================
# Configurations à tester
# =============================================================================
configurations = {
    "triplet": {
        "name": "Triplet (sans HR)",
        "features": ["T_amb", "LF", "delta_P"],
        "output_dir": os.path.join(RESULTS_DIR, "triplet")
    },
    "quadruplet": {
        "name": "Quadruplet (avec HR)",
        "features": ["T_amb", "HR", "LF", "delta_P"],
        "output_dir": os.path.join(RESULTS_DIR, "quadruplet")
    }
}

# =============================================================================
# Fonction FCM
# =============================================================================
def run_fcm_clustering(df, features, config_name, output_dir):
    """Exécute FCM sur les features spécifiées"""
    
    print(f"\n{'='*70}")
    print(f"🔬 Configuration: {config_name}")
    print(f"   Features: {features}")
    print(f"{'='*70}")
    
    # Créer dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    
    # Préparer données
    df_clean = df.dropna(subset=features).copy()
    X = df_clean[features].values
    
    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"📊 Données: {len(df_clean)} échantillons, {len(features)} features")
    
    # FCM
    print("\n🔄 Exécution FCM (c=3 clusters, m=2.0)...")
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data=X_scaled.T,
        c=3,
        m=2.0,
        error=0.005,
        maxiter=1000,
        init=None
    )
    
    # Attribution clusters
    cluster_labels = np.argmax(u, axis=0)
    df_clean["fcm_cluster"] = cluster_labels
    
    # Sauvegarder matrice d'appartenance
    for i in range(3):
        df_clean[f"u_{i}"] = u[i]
    
    print(f"✅ FCM terminé")
    print(f"   Distribution: {df_clean['fcm_cluster'].value_counts().sort_index().to_dict()}")
    print(f"   FPC Score: {fpc:.4f}")
    
    # =============================================================================
    # Visualisations
    # =============================================================================
    print("\n📊 Génération des visualisations...")
    
    # 1. Projection 2D (2 premières features)
    if len(features) >= 2:
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            X_scaled[:, 0], X_scaled[:, 1],
            c=cluster_labels, cmap='viridis', alpha=0.7, s=30
        )
        plt.xlabel(f"{features[0]} (normalisé)")
        plt.ylabel(f"{features[1]} (normalisé)")
        plt.title(f"Clustering FCM — {config_name}")
        plt.colorbar(label="Cluster")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        fig_path = f"{output_dir}/figures/fcm_projection_2d.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Projection 2D: {fig_path}")
    
    # 2. Distribution des clusters
    plt.figure(figsize=(8, 5))
    df_clean['fcm_cluster'].value_counts().sort_index().plot(
        kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c']
    )
    plt.xlabel("Cluster")
    plt.ylabel("Nombre d'échantillons")
    plt.title(f"Distribution des clusters — {config_name}")
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    fig_path = f"{output_dir}/figures/fcm_distribution.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Distribution: {fig_path}")
    
    # =============================================================================
    # Sauvegarde
    # =============================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CSV
    output_csv = f"{output_dir}/fcm_output_{timestamp}.csv"
    df_clean.to_csv(output_csv, index=False)
    print(f"   💾 CSV: {output_csv}")
    
    # JSON summary
    summary = {
        "timestamp": timestamp,
        "configuration": config_name,
        "features": features,
        "n_clusters": 3,
        "fpc_score": float(fpc),
        "n_samples": len(df_clean),
        "cluster_distribution": df_clean['fcm_cluster'].value_counts().sort_index().to_dict(),
        "output_csv": output_csv,
        "figures": [
            f"{output_dir}/figures/fcm_projection_2d.png" if len(features) >= 2 else None,
            f"{output_dir}/figures/fcm_distribution.png"
        ]
    }
    
    summary_json = f"{output_dir}/fcm_summary_{timestamp}.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"   📁 Summary: {summary_json}")
    
    return output_csv, fpc

# =============================================================================
# Exécuter les deux configurations
# =============================================================================
results = {}

for config_key, config in configurations.items():
    output_csv, fpc = run_fcm_clustering(
        df, 
        config["features"], 
        config["name"],
        config["output_dir"]
    )
    results[config_key] = {
        "csv": output_csv,
        "fpc": fpc,
        "features": config["features"]
    }

# =============================================================================
# Comparaison finale
# =============================================================================
print(f"\n{'='*70}")
print("📊 COMPARAISON TRIplet vs QUADRUPLET")
print(f"{'='*70}")

print(f"\n{'Configuration':<20} {'FPC Score':<15} {'Features'}")
print("-"*70)
for config_key, result in results.items():
    print(f"{config_key:<20} {result['fpc']:.4f}{'':<10} {result['features']}")

# Sauvegarder comparaison
comparison = {
    "timestamp": datetime.now().isoformat(),
    "triplet": {
        "fpc": results["triplet"]["fpc"],
        "features": results["triplet"]["features"],
        "output_csv": results["triplet"]["csv"]
    },
    "quadruplet": {
        "fpc": results["quadruplet"]["fpc"],
        "features": results["quadruplet"]["features"],
        "output_csv": results["quadruplet"]["csv"]
    },
    "conclusion": "Quadruplet meilleur" if results["quadruplet"]["fpc"] > results["triplet"]["fpc"] else "Triplet suffisant"
}

comparison_file = f"{RESULTS_DIR}/fcm_comparison_summary.json"
with open(comparison_file, "w", encoding="utf-8") as f:
    json.dump(comparison, f, indent=2, ensure_ascii=False)

print(f"\n💾 Comparaison sauvegardée: {comparison_file}")

print("\n" + "="*70)
print("✅ CLUSTERING FCM TERMINÉ")
print("="*70)
print(f"\n📌 PROCHAINE ÉTAPE:")
print(f"   Lance: python 02_scripts/04_fis_inference.py")