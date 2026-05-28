#!/usr/bin/env python3
"""
fix_rendement.py
================
Correction et validation du calcul du rendement énergétique.

Problème identifié : P_mesuree > P_ref dans le dataset simulé
Solution : 
  - Clipper le rendement à [0, 1] OU
  - Utiliser une référence théorique ajustée

Auteur : Wissem ZD
Date : 2026
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

print("="*70)
print("🔧 CORRECTION DU RENDEMENT ÉNERGÉTIQUE")
print("="*70)

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT = r"C:\projets\memoire_FCM"
DATA_DIR = os.path.join(PROJECT_ROOT, "01_data")
CSV_FILE = os.path.join(DATA_DIR, "dataset_MENA_FCM_v2.csv")

print(f"\n📂 Dataset: {CSV_FILE}")

# =============================================================================
# Charger les données
# =============================================================================
if not os.path.exists(CSV_FILE):
    print(f"❌ Fichier non trouvé: {CSV_FILE}")
    exit(1)

df = pd.read_csv(CSV_FILE)
print(f"✅ Données chargées: {len(df)} lignes")

# =============================================================================
# Analyse P_mesuree vs P_ref
# =============================================================================
print("\n📊 Analyse des puissances:")
print(f"   P_mesuree - Min: {df['P_mesuree'].min():.2f} W/m²")
print(f"   P_mesuree - Max: {df['P_mesuree'].max():.2f} W/m²")
print(f"   P_mesuree - Moy: {df['P_mesuree'].mean():.2f} W/m²")
print(f"\n   P_ref - Min: {df['P_ref'].min():.2f} W/m²")
print(f"   P_ref - Max: {df['P_ref'].max():.2f} W/m²")
print(f"   P_ref - Moy: {df['P_ref'].mean():.2f} W/m²")

# Calcul ratio brut
df['ratio_brut'] = df['P_mesuree'] / df['P_ref']

print(f"\n⚠️ Ratio P_mesuree/P_ref:")
print(f"   Min: {df['ratio_brut'].min():.2f}")
print(f"   Max: {df['ratio_brut'].max():.2f}")
print(f"   Moy: {df['ratio_brut'].mean():.2f}")
print(f"   Nombre de valeurs > 1.0: {(df['ratio_brut'] > 1.0).sum()}")
print(f"   Pourcentage > 100%: {(df['ratio_brut'] > 1.0).sum()/len(df)*100:.1f}%")

# =============================================================================
# Stratégie de correction
# =============================================================================
print("\n🔧 Stratégies de correction disponibles:")
print("   1. Clipper à 100% (rendement = min(ratio, 1.0))")
print("   2. Normaliser par le max observé")
print("   3. Utiliser P_ref théorique ajusté")

# Option recommandée : Clipper + garder trace du ratio brut
df['rendement'] = df['ratio_brut'].clip(0, 1.0)

print(f"\n✅ Rendement corrigé (clippé à 100%):")
print(f"   Min: {df['rendement'].min():.2%}")
print(f"   Max: {df['rendement'].max():.2%}")
print(f"   Moy: {df['rendement'].mean():.2%}")

# =============================================================================
# Sauvegarde
# =============================================================================
RESULTS_DIR = os.path.join(PROJECT_ROOT, "03_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Sauvegarder le dataset corrigé
output_file = os.path.join(RESULTS_DIR, "dataset_corrected.csv")
df.to_csv(output_file, index=False)
print(f"\n💾 Dataset corrigé sauvegardé: {output_file}")

# Résumé JSON
summary = {
    "timestamp": datetime.now().isoformat(),
    "original_file": CSV_FILE,
    "corrected_file": output_file,
    "statistics": {
        "total_samples": len(df),
        "p_mesuree": {
            "min": float(df['P_mesuree'].min()),
            "max": float(df['P_mesuree'].max()),
            "mean": float(df['P_mesuree'].mean())
        },
        "p_ref": {
            "min": float(df['P_ref'].min()),
            "max": float(df['P_ref'].max()),
            "mean": float(df['P_ref'].mean())
        },
        "ratio_brut": {
            "min": float(df['ratio_brut'].min()),
            "max": float(df['ratio_brut'].max()),
            "mean": float(df['ratio_brut'].mean()),
            "values_above_1": int((df['ratio_brut'] > 1.0).sum()),
            "percentage_above_100": float((df['ratio_brut'] > 1.0).sum()/len(df)*100)
        },
        "rendement_corrected": {
            "min": float(df['rendement'].min()),
            "max": float(df['rendement'].max()),
            "mean": float(df['rendement'].mean())
        }
    },
    "correction_method": "clip_to_100",
    "note": "P_mesuree > P_ref peut indiquer un biais climatique ou une référence mal définie"
}

summary_file = os.path.join(RESULTS_DIR, "rendement_correction_summary.json")
with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"📁 Résumé sauvegardé: {summary_file}")

print("\n" + "="*70)
print("✅ CORRECTION TERMINÉE")
print("="*70)
print(f"\n📌 PROCHAINE ÉTAPE:")
print(f"   Lance: python 02_scripts/01_fcm_clustering.py")