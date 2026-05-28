#!/usr/bin/env python3
"""
archive_results.py
==================
Sauvegarde tous les résultats actuels avec timestamp avant nouvelle exécution.
"""

import os
import shutil
from datetime import datetime

PROJECT_ROOT = "."
RESULTS_DIR = os.path.join(PROJECT_ROOT, "03_results")
ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "03_results_archive")

# Créer dossier archive avec timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
archive_path = f"{ARCHIVE_DIR}/{timestamp}"

if os.path.exists(RESULTS_DIR):
    shutil.copytree(RESULTS_DIR, archive_path)
    print(f"✅ Résultats archivés dans: {archive_path}")
else:
    print("ℹ️ Aucun résultat à archiver")