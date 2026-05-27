# 🏭 FCM-FIS-PSO Energy MENA
> **Hybrid Intelligent System for Modeling Climate-Induced Energy Inefficiency in MENA Industrial Machines**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Status](https://img.shields.io/badge/Status-Research_Prototype-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Branch](https://img.shields.io/badge/Branch-pipeline--sugeno--pso-orange)

---

## 📖 Abstract
This project proposes a **hybrid computational intelligence pipeline** to quantify and correct the **climatic bias** affecting industrial machine efficiency in the MENA region. By coupling **Fuzzy C-Means (FCM)** clustering, **Sugeno-type Fuzzy Inference System (FIS)**, and **Particle Swarm Optimization (PSO)**, the system dynamically estimates a climate correction factor ($K_{MENA}$) from ambient temperature ($T_{amb}$), relative humidity ($HR$), load factor ($LF$), and power deviation ($\Delta P$).

## 🎯 Objectives
- ️ Model the non-linear impact of thermo-hygrometric stress on industrial performance
- 🧩 Compare `Triplet` vs `Quadruplet` feature spaces to validate humidity's discriminative role
- ⚡ Optimize Sugeno rule coefficients via PSO for real-time bias compensation
- 📊 Provide a reproducible, open-source pipeline for industrial energy analytics

## 🏗️ Pipeline Architecture
```mermaid
graph LR
  A[Raw Sensor Data] --> B(Preprocessing & Yield Clipping)
  B --> C{Feature Selection}
  C -->|Triplet| D1[T_amb, LF, ΔP]
  C -->|Quadruplet| D2[T_amb, HR, LF, ΔP]
  D1 --> E[FCM Clustering c=3]
  D2 --> E
  E --> F[Sugeno FIS Inference]
  F --> G[PSO Coefficient Optimization]
  G --> H[K_MENA Climate Bias Map]
  H --> I[Dashboard & Reports]
```
## 🧠 Methodology
| Step | Algorithm | Purpose | Metrics |
|------|-----------|---------|---------|
| **1. Clustering** | Fuzzy C-Means (`skfuzzy`) | Unsupervised pattern discovery in operational regimes | FPC, Silhouette Score |
| **2. Inference** | Sugeno FIS (1st order) | Rule-based mapping: `y = a·T + b·HR + c·LF + d·ΔP + e` | RMSE, R², MAE |
| **3. Optimization** | Global Best PSO (`pyswarms`) | Minimize RMSE between FIS output and actual yield | Convergence speed, Best cost |
| **4. Validation** | Triplet vs Quadruplet comparison | Quantify humidity's discriminative role | ΔRMSE, Statistical significance |

## 🚀 Quick Start
```bash
# 1. Environment setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Run pipeline sequentially
python 02_scripts/03_fix_rendement.py
python 02_scripts/01_fcm_clustering.py
python 02_scripts/05_pso_sugeno_optimization.py
python 02_scripts/06_compare_results.py

📈 Key Results
✅ PSO convergence: RMSE reduced from 50% → 12.13% (quadruplet) in < 2 minutes
📊 Humidity impact: Quadruplet outperforms Triplet by 0.89% RMSE
🔥 Vectorization: 100× speedup vs Mamdani approach (1h45 → 1s)
📉 FPC Score: 0.72 (Triplet) vs 0.66 (Quadruplet) — acceptable clustering quality
🔬 Scientific Contributions
First application of FCM-Sugeno-PSO hybrid for MENA industrial climate bias
Quantitative validation of humidity as a discriminative feature
Open-source reproducible pipeline with timestamped outputs
🤝 Contributing & Academic Use
This repository is designed for reproducible research. All scripts use robust path detection and timestamped outputs. Feel free to:
🔀 Fork and adapt feature sets
📡 Integrate real-time MQTT/InfluxDB streams
📊 Extend to other climate zones (EU, Asia, etc.)
📜 License
MIT License © 2026 Wissem ZD. Free for academic & industrial research.
📧 Contact
Wissem ZD — wissem.zdini@u-virtuelle.tn
