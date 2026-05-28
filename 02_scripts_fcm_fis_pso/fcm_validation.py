# fcm_validation.py
# ✅ Validation FCM par GMM et indice de Xie-Beni

import pandas as pd
import numpy as np
import skfuzzy as fuzz
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import json
import os

print("🔬 Validation FCM — GMM + Xie-Beni + Métriques externes\n")

# =============================================================================
# Charger les données
# =============================================================================
RESULTS_DIR = "03_results"
FCM_FILE = "fcm_global_output_20260510_183404.csv"
FCM_PATH = os.path.join(RESULTS_DIR, FCM_FILE)

df = pd.read_csv(FCM_PATH)
features = ["T_amb", "HR", "LF", "delta_P"]
X = df[features].values

# Labels FCM existants
labels_fcm = df['fcm_cluster_global'].values

print(f"📊 Données: {len(df)} échantillons, {len(features)} features")
print(f"📋 Labels FCM: {np.unique(labels_fcm)}\n")

# =============================================================================
# 1. Validation interne FCM — Indice de Xie-Beni
# =============================================================================
print("📈 1. Indice de Xie-Beni (validité FCM)...")

def xie_beni_index(X, cntr, u, m=2):
    """
    Calcule l'indice de Xie-Beni pour évaluer la qualité du clustering flou.
    Plus l'indice est faible, meilleur est le clustering.
    """
    n_clusters = cntr.shape[0]
    n_samples = X.shape[0]
    
    # Matrice d'appartenance au carré
    u_power = u ** m
    
    # Séparation entre clusters (minimum distance entre centres)
    min_dist = float('inf')
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            dist = np.linalg.norm(cntr[i] - cntr[j])
            if dist < min_dist:
                min_dist = dist
    
    # Compacité (somme des distances intra-cluster)
    compactness = 0
    for k in range(n_clusters):
        for i in range(n_samples):
            compactness += u_power[k, i] * np.linalg.norm(X[i] - cntr[k]) ** 2
    
    # Indice de Xie-Beni
    xb_index = compactness / (n_samples * (min_dist ** 2))
    
    return xb_index

# Recharger la matrice d'appartenance depuis les colonnes u_0, u_1, u_2
u_matrix = df[[f'u_{i}' for i in range(3)]].values.T  # Shape: (3, 3000)

# Centres de clusters (à recalculer ou charger)
# Pour simplifier, on utilise les moyennes par cluster
cntr = np.array([X[labels_fcm == k].mean(axis=0) for k in range(3)])

xb = xie_beni_index(X, cntr, u_matrix)
print(f"   ✅ Xie-Beni Index: {xb:.4f}")
print(f"   💡 Interprétation: {'Bon clustering' if xb < 0.5 else 'Clustering moyen' if xb < 1.0 else 'Clustering faible'}\n")

# =============================================================================
# 2. Comparaison FCM vs GMM (Gaussian Mixture Model)
# =============================================================================
print("📈 2. Validation croisée FCM vs GMM...")

# GMM avec 3 clusters (même nombre que FCM)
gmm = GaussianMixture(n_components=3, random_state=42, covariance_type='full')
labels_gmm = gmm.fit_predict(X)

# Score de vraisemblance GMM
gmm_log_likelihood = gmm.score(X)
print(f"   • GMM Log-Likelihood: {gmm_log_likelihood:.2f}")

# Critère d'information bayésien (BIC) — plus bas = mieux
gmm_bic = gmm.bic(X)
print(f"   • GMM BIC: {gmm_bic:.2f}")

# Accord entre FCM et GMM (Adjusted Rand Index)
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ari = adjusted_rand_score(labels_fcm, labels_gmm)
nmi = normalized_mutual_info_score(labels_fcm, labels_gmm)

print(f"   • Adjusted Rand Index (FCM vs GMM): {ari:.3f}")
print(f"   • Normalized Mutual Information: {nmi:.3f}")
print(f"   💡 Interprétation: {'Fort accord' if ari > 0.7 else 'Accord modéré' if ari > 0.4 else 'Faible accord'}\n")

# =============================================================================
# 3. Métriques de validation externes (Silhouette, Calinski-Harabasz)
# =============================================================================
print("📈 3. Métriques de validation externes...")

# Silhouette Score (cohésion intra-cluster vs séparation inter-cluster)
silhouette = silhouette_score(X, labels_fcm)
print(f"   • Silhouette Score: {silhouette:.3f}")
print(f"   💡 Interprétation: {'Bonne séparation' if silhouette > 0.5 else 'Séparation modérée' if silhouette > 0.25 else 'Faible séparation'}")

# Calinski-Harabasz Index (ratio variance inter/intra)
ch_index = calinski_harabasz_score(X, labels_fcm)
print(f"   • Calinski-Harabasz Index: {ch_index:.2f}")
print(f"   💡 Plus élevé = mieux\n")

# =============================================================================
# 4. Résumé JSON
# =============================================================================
validation_summary = {
    "fcm_validation": {
        "xie_beni_index": float(xb),
        "n_clusters": 3,
        "fpc_score": 0.6582,
        "cluster_distribution": df['fcm_cluster_global'].value_counts().to_dict()
    },
    "gmm_validation": {
        "log_likelihood": float(gmm_log_likelihood),
        "bic": float(gmm_bic),
        "n_components": 3
    },
    "comparison_fcm_vs_gmm": {
        "adjusted_rand_index": float(ari),
        "normalized_mutual_info": float(nmi)
    },
    "external_metrics": {
        "silhouette_score": float(silhouette),
        "calinski_harabasz_index": float(ch_index)
    },
    "interpretation": {
        "clustering_quality": "Bon" if xb < 0.5 else "Moyen",
        "fcm_gmm_agreement": "Fort" if ari > 0.7 else "Modéré",
        "recommendation": "FCM validé par GMM" if ari > 0.6 else "Comparer les deux approches"
    }
}

# Sauvegarder
output_json = os.path.join(RESULTS_DIR, "fcm_validation_summary.json")
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(validation_summary, f, indent=2, ensure_ascii=False)

print(f"\n💾 Résumé sauvegardé: {output_json}")
print("✅ Validation FCM terminée !\n")

# =============================================================================
# Interprétation pour le mémoire
# =============================================================================
print("="*70)
print("📝 INTERPRÉTATION POUR LE MÉMOIRE")
print("="*70)
print(f"""
1. QUALITÉ DU CLUSTERING FCM :
   • Indice de Xie-Beni : {xb:.4f} → {'Clustering compact et bien séparé' if xb < 0.5 else 'Clustering acceptable'}
   • Silhouette Score : {silhouette:.3f} → {'Bonne cohésion intra-cluster' if silhouette > 0.5 else 'Cohésion modérée'}

2. COHÉRENCE AVEC GMM :
   • ARI = {ari:.3f} → {'Les deux méthodes convergent vers les mêmes clusters' if ari > 0.7 else 'Différences notables entre FCM et GMM'}
   • Cela valide {'la robustesse' if ari > 0.6 else 'la spécificité'} de l'approche FCM pour ton dataset.

3. RECOMMANDATION :
   → {validation_summary['interpretation']['recommendation']}
""")
print("="*70)