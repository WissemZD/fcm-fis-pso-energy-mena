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
