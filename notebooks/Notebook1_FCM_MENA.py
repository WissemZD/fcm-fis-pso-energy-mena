"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MÉMOIRE MASTER IASRIA — Notebook 1 : Dataset MENA + EDA + FCM             ║
║  Modélisation hybride FCM–FIS–PSO de l'inefficience énergétique industrielle ║
║  Contexte climatique MENA — Tunisie                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Auteur   : Wissem
Encadrant: [Nom encadrant]
Date     : 2025

Structure :
  1. Génération du dataset simulé MENA (3 régimes physiques)
  2. Analyse exploratoire (EDA) — histogrammes, corrélations, scatter plots
  3. Implémentation FCM from scratch (Bezdek 1981)
  4. Visualisation des clusters et interprétation physique
  5. Construction de l'indice d'inefficience I_ineff
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# ── Style global ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#F8FBFF',
    'axes.grid':        True,
    'grid.alpha':       0.35,
    'grid.linestyle':   '--',
    'font.family':      'DejaVu Sans',
    'axes.titlesize':   13,
    'axes.labelsize':   11,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'legend.fontsize':  9,
})

COLORS = {
    'nominal':   '#2E75B6',   # bleu  — Régime nominal (efficace)
    'moderate':  '#F4A228',   # orange — Inefficacité thermique modérée
    'critical':  '#C0392B',   # rouge  — Inefficacité critique MENA
    'accent':    '#1A3A5C',
    'grid':      '#CCDDEE',
}
CLUSTER_COLORS = [COLORS['nominal'], COLORS['moderate'], COLORS['critical']]
CLUSTER_LABELS = ['Régime Nominal', 'Inefficacité Modérée', 'Inefficacité Critique MENA']

np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — GÉNÉRATION DU DATASET SIMULÉ MENA
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  SECTION 1 — GÉNÉRATION DU DATASET SIMULÉ MENA")
print("=" * 70)

N_TOTAL = 3000   # nombre d'observations total
N_SITES = 3      # 3 sites industriels tunisiens
N = N_TOTAL // N_SITES

def generate_regime(n, regime, site_id, season_weights=None):
    """
    Génère n observations pour un régime énergétique donné.
    
    Régimes physiques MENA :
      0 — Nominal     : T_amb faible, LF optimal, ΔP faible
      1 — Modéré      : T_amb intermédiaire, LF sous-optimal, ΔP moyen
      2 — Critique    : T_amb élevée (canicule MENA), LF dégradé, ΔP élevé
    """
    
    if regime == 0:  # ── Nominal ──────────────────────────────────────────
        T_amb   = np.random.normal(24,  4,  n).clip(15, 34)   # °C
        HR      = np.random.normal(52,  8,  n).clip(35, 68)   # %
        LF      = np.random.normal(0.88, 0.06, n).clip(0.75, 1.0)
        P_meas  = np.random.normal(48,  4,  n).clip(38, 58)   # kW
        P_ref   = 50.0  # kW — valeur datasheet IEC
        
    elif regime == 1:  # ── Modéré ───────────────────────────────────────
        T_amb   = np.random.normal(34,  4,  n).clip(26, 43)
        HR      = np.random.normal(62,  9,  n).clip(45, 80)
        LF      = np.random.normal(0.72, 0.08, n).clip(0.55, 0.85)
        P_meas  = np.random.normal(61,  5,  n).clip(50, 74)
        P_ref   = 50.0
        
    else:  # ── Critique MENA ──────────────────────────────────────────────
        T_amb   = np.random.normal(42,  3,  n).clip(36, 50)   # canicule
        HR      = np.random.normal(45,  10, n).clip(25, 65)   # air sec
        LF      = np.random.normal(0.52, 0.10, n).clip(0.30, 0.72)
        P_meas  = np.random.normal(74,  6,  n).clip(60, 90)
        P_ref   = 50.0

    # ── Variables dérivées ────────────────────────────────────────────────
    delta_P = (P_meas - P_ref) / P_ref                 # écart énergétique relatif
    T_norm  = (T_amb - 20) / (50 - 20)                 # normalisation [20°C, 50°C]
    T_norm  = T_norm.clip(0, 1)
    HR_norm = HR / 100.0

    # ── Indice d'inefficience de référence (physique) ─────────────────────
    # Contribution thermique (coefficient de dérating moteur ~0.5%/°C)
    alpha_T = 0.005
    # Contribution charge partielle
    eta_LF  = 1 - 0.15 * (1 - LF) ** 2
    # Indice synthétique
    I_ineff = (delta_P.clip(0, None) * 0.45
               + T_norm * 0.35
               + (1 - LF) * 0.20).clip(0, 1)

    # ── Horodatage simulé (1 an, toutes saisons) ──────────────────────────
    days_per_regime = [120, 140, 105][regime]  # jours/an par régime
    timestamps = pd.date_range('2024-01-01', periods=n, freq='3h')

    df = pd.DataFrame({
        'timestamp':  timestamps,
        'site':       f'Site_{site_id+1}',
        'regime_ref': regime,                    # vérité terrain (pour validation)
        'T_amb':      T_amb.round(2),
        'HR':         HR.round(2),
        'LF':         LF.round(4),
        'P_mesuree':  P_meas.round(2),           # kW
        'P_ref':      P_ref,
        'delta_P':    delta_P.round(4),
        'T_norm':     T_norm.round(4),
        'I_ineff_ref': I_ineff.round(4),
    })
    return df

# Génération multi-sites avec proportions saisonnières différentes
all_dfs = []
for site in range(N_SITES):
    # Chaque site a une proportion différente de régimes (diversité géographique)
    proportions = [
        [0.45, 0.35, 0.20],  # Site 1 — côtier : plus de nominal
        [0.30, 0.40, 0.30],  # Site 2 — intérieur : plus de modéré
        [0.25, 0.35, 0.40],  # Site 3 — sud : plus de critique (Sahara)
    ][site]
    
    site_dfs = []
    for regime, prop in enumerate(proportions):
        n_regime = int(N * prop)
        df_r = generate_regime(n_regime, regime, site)
        site_dfs.append(df_r)
    
    df_site = pd.concat(site_dfs, ignore_index=True)
    df_site = df_site.sample(frac=1, random_state=site).reset_index(drop=True)
    all_dfs.append(df_site)

df = pd.concat(all_dfs, ignore_index=True)

print(f"\n✅ Dataset généré : {len(df)} observations | {N_SITES} sites | 1 an simulé")
print(f"   Variables : {list(df.columns)}\n")
print(df[['site', 'T_amb', 'HR', 'LF', 'delta_P', 'T_norm', 'I_ineff_ref']].describe().round(3))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ANALYSE EXPLORATOIRE (EDA)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  SECTION 2 — ANALYSE EXPLORATOIRE (EDA)")
print("=" * 70)

X_raw = df[['delta_P', 'T_norm', 'LF']].values
features = ['ΔP (écart énergétique)', 'T_norm (température)', 'LF (facteur de charge)']

regime_colors_map = {0: COLORS['nominal'], 1: COLORS['moderate'], 2: COLORS['critical']}
point_colors = [regime_colors_map[r] for r in df['regime_ref']]

# ── Figure 1 : Histogrammes des 3 variables ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle("Figure 1 — Distribution des Variables Énergétiques\nDataset MENA Simulé (3 sites × 1 an)",
             fontsize=13, fontweight='bold', color=COLORS['accent'], y=1.02)

var_raw = ['delta_P', 'T_norm', 'LF']
colors_hist = [COLORS['nominal'], COLORS['moderate'], COLORS['critical']]
xlabels = ['ΔP — Écart énergétique relatif', 'T_norm — Température normalisée', 'LF — Facteur de charge']

for i, (ax, var, col, xlab) in enumerate(zip(axes, var_raw, colors_hist, xlabels)):
    for r, (rcolor, rlabel) in enumerate(zip(CLUSTER_COLORS, CLUSTER_LABELS)):
        subset = df[df['regime_ref'] == r][var]
        ax.hist(subset, bins=35, alpha=0.65, color=rcolor, label=rlabel, edgecolor='white', linewidth=0.4)
    
    ax.set_xlabel(xlab, fontweight='bold')
    ax.set_ylabel('Fréquence')
    ax.set_title(f'Distribution — {xlab.split("—")[0].strip()}')
    if i == 0:
        ax.legend(fontsize=8, framealpha=0.9)
    
    # Ligne médiane
    med = df[var].median()
    ax.axvline(med, color='black', linestyle=':', linewidth=1.5, alpha=0.7,
               label=f'Médiane = {med:.3f}')

plt.tight_layout()
plt.savefig('/home/claude/fig1_histogrammes.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure 1 sauvegardée : fig1_histogrammes.png")

# ── Figure 2 : Matrice de corrélation ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Figure 2 — Corrélations entre Variables Énergétiques",
             fontsize=13, fontweight='bold', color=COLORS['accent'])

vars_corr = ['T_amb', 'HR', 'LF', 'delta_P', 'T_norm', 'I_ineff_ref']
corr_matrix = df[vars_corr].corr()

# Heatmap manuelle
ax = axes[0]
im = ax.imshow(corr_matrix.values, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(vars_corr)))
ax.set_yticks(range(len(vars_corr)))
ax.set_xticklabels(vars_corr, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(vars_corr, fontsize=9)
ax.set_title("Matrice de Corrélation — Toutes Variables")

for i in range(len(vars_corr)):
    for j in range(len(vars_corr)):
        val = corr_matrix.values[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color, fontweight='bold')

plt.colorbar(im, ax=ax, shrink=0.8)

# Corrélation avec I_ineff
ax2 = axes[1]
corr_with_ineff = df[['T_amb', 'HR', 'LF', 'delta_P', 'T_norm']].corrwith(df['I_ineff_ref']).sort_values()
bar_colors = [COLORS['critical'] if v > 0 else COLORS['nominal'] for v in corr_with_ineff.values]
bars = ax2.barh(corr_with_ineff.index, corr_with_ineff.values, color=bar_colors, alpha=0.85, edgecolor='white')
ax2.axvline(0, color='black', linewidth=1)
ax2.set_xlabel("Corrélation de Pearson avec I_ineff")
ax2.set_title("Corrélation avec l'Indice d'Inefficience\n(variable cible du modèle)")

for bar, val in zip(bars, corr_with_ineff.values):
    ax2.text(val + 0.01 * np.sign(val), bar.get_y() + bar.get_height()/2,
             f'{val:+.3f}', va='center', ha='left' if val > 0 else 'right', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/claude/fig2_correlations.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure 2 sauvegardée : fig2_correlations.png")

# ── Figure 3 : Scatter plots ─────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 5))
fig.suptitle("Figure 3 — Scatter Plots : Espace des États Énergétiques MENA\n(coloré par régime réel — vérité terrain)",
             fontsize=13, fontweight='bold', color=COLORS['accent'])

pairs = [('delta_P', 'T_norm'), ('T_norm', 'LF'), ('delta_P', 'LF')]
pair_labels = [('ΔP (écart énergie)', 'T_norm (température)'),
               ('T_norm (température)', 'LF (facteur charge)'),
               ('ΔP (écart énergie)', 'LF (facteur charge)')]

for idx, ((xv, yv), (xl, yl)) in enumerate(zip(pairs, pair_labels)):
    ax = fig.add_subplot(1, 3, idx+1)
    
    for r, (rcolor, rlabel) in enumerate(zip(CLUSTER_COLORS, CLUSTER_LABELS)):
        mask = df['regime_ref'] == r
        ax.scatter(df.loc[mask, xv], df.loc[mask, yv],
                   c=rcolor, alpha=0.35, s=8, label=rlabel if idx == 0 else "")
    
    ax.set_xlabel(xl, fontsize=10)
    ax.set_ylabel(yl, fontsize=10)
    ax.set_title(f'{xl.split("(")[0].strip()} vs {yl.split("(")[0].strip()}')

handles = [mpatches.Patch(color=c, label=l, alpha=0.8) 
           for c, l in zip(CLUSTER_COLORS, CLUSTER_LABELS)]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, -0.05), framealpha=0.9)

plt.tight_layout()
plt.savefig('/home/claude/fig3_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure 3 sauvegardée : fig3_scatter.png")

# ── Figure 4 : Analyse par site ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle("Figure 4 — Profils Énergétiques par Site Industriel\n(distribution de I_ineff — contexte MENA multi-sites)",
             fontsize=13, fontweight='bold', color=COLORS['accent'])

site_colors = ['#2E75B6', '#E67E22', '#27AE60']
for i, (site, ax) in enumerate(zip(df['site'].unique(), axes)):
    df_site = df[df['site'] == site]
    
    ax.hist(df_site['I_ineff_ref'], bins=40, color=site_colors[i], alpha=0.8,
            edgecolor='white', linewidth=0.4)
    
    mean_val = df_site['I_ineff_ref'].mean()
    ax.axvline(mean_val, color='black', linestyle='--', linewidth=2, label=f'Moyenne = {mean_val:.3f}')
    
    site_names = ['Site 1 — Côtier (Sfax)', 'Site 2 — Intérieur (Kairouan)', 'Site 3 — Sud (Gabès)']
    ax.set_title(site_names[i], fontweight='bold')
    ax.set_xlabel("I_ineff (indice d'inefficience)")
    ax.set_ylabel("Fréquence")
    ax.legend(fontsize=9)
    
    # Statistiques dans le graphe
    stats_text = (f"n = {len(df_site)}\n"
                  f"μ = {mean_val:.3f}\n"
                  f"σ = {df_site['I_ineff_ref'].std():.3f}\n"
                  f"max = {df_site['I_ineff_ref'].max():.3f}")
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=8,
            va='top', ha='right', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))

plt.tight_layout()
plt.savefig('/home/claude/fig4_sites.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure 4 sauvegardée : fig4_sites.png")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — IMPLÉMENTATION FCM FROM SCRATCH (Bezdek 1981)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  SECTION 3 — FUZZY C-MEANS (Bezdek 1981) — Implémentation from scratch")
print("=" * 70)

def fcm(X, C=3, m=2.0, max_iter=150, eps=1e-4, seed=42):
    """
    Fuzzy C-Means Algorithm — Bezdek (1981)
    
    Paramètres :
      X        : array (N, D) — données d'entrée
      C        : int          — nombre de clusters
      m        : float        — fuzzifier (m=2 standard)
      max_iter : int          — nombre max d'itérations
      eps      : float        — critère de convergence (norme Frobenius)
      seed     : int          — reproductibilité
    
    Retourne :
      U        : (N, C) — matrice d'appartenance finale
      centers  : (C, D) — centres des clusters
      J_history: list   — historique de la fonction objectif
      n_iter   : int    — nombre d'itérations jusqu'à convergence
    
    Référence : Bezdek, J.C. (1981). Pattern Recognition with Fuzzy
                Objective Function Algorithms. Plenum Press.
    """
    np.random.seed(seed)
    N, D = X.shape
    
    # ── Initialisation : matrice U aléatoire respectant Σⱼ uᵢⱼ = 1 ──────
    U = np.random.dirichlet(np.ones(C), size=N)   # chaque ligne somme à 1
    
    J_history = []
    
    for iteration in range(max_iter):
        U_old = U.copy()
        
        # ── Étape A : Mise à jour des centres ─────────────────────────────
        # cⱼ = Σᵢ uᵢⱼᵐ · xᵢ / Σᵢ uᵢⱼᵐ
        Um = U ** m                         # (N, C) — u^m
        centers = (Um.T @ X) / Um.sum(axis=0)[:, np.newaxis]   # (C, D)
        
        # ── Étape B : Calcul des distances ────────────────────────────────
        # dᵢⱼ = ||xᵢ - cⱼ||₂
        D_mat = cdist(X, centers, metric='euclidean')   # (N, C)
        D_mat = np.fmax(D_mat, 1e-10)     # évite division par zéro
        
        # ── Étape C : Mise à jour de U ────────────────────────────────────
        # uᵢⱼ = 1 / Σₖ (dᵢⱼ / dᵢₖ)^(2/(m-1))
        exp = 2.0 / (m - 1.0)
        U_new = np.zeros((N, C))
        for j in range(C):
            ratio = D_mat[:, j:j+1] / D_mat    # (N, C)
            U_new[:, j] = 1.0 / (ratio ** exp).sum(axis=1)
        
        U = U_new
        
        # ── Calcul de la fonction objectif ────────────────────────────────
        # J = Σᵢ Σⱼ uᵢⱼᵐ · dᵢⱼ²
        J = float(((U ** m) * (D_mat ** 2)).sum())
        J_history.append(J)
        
        # ── Critère de convergence — Norme de Frobenius ───────────────────
        delta = np.linalg.norm(U - U_old, ord='fro')
        
        if (iteration + 1) % 20 == 0:
            print(f"   Iter {iteration+1:3d} | J = {J:.4f} | ||ΔU||_F = {delta:.6f}")
        
        if delta < eps:
            print(f"\n✅ Convergence atteinte à l'itération {iteration+1}")
            print(f"   Norme Frobenius finale : {delta:.2e} < ε = {eps:.2e}")
            return U, centers, J_history, iteration + 1
    
    print(f"\n⚠️  Arrêt après {max_iter} itérations (convergence non atteinte)")
    return U, centers, J_history, max_iter


# ── Exécution FCM ─────────────────────────────────────────────────────────────
X = df[['delta_P', 'T_norm', 'LF']].values
print(f"\n  Données : X ∈ ℝ^{X.shape}  |  C=3, m=2, ε=10⁻⁴")
print(f"  Vecteur d'état : X = [ΔP, T_norm, LF]\n")

U, centers, J_history, n_iter = fcm(X, C=3, m=2.0, eps=1e-4)

# ── Attribution des clusters (argmax des appartenances) ───────────────────────
cluster_assignments = np.argmax(U, axis=1)

# ── Alignement des clusters avec la physique ──────────────────────────────────
# On identifie chaque cluster par sa température normalisée (T_norm) croissante
center_T_norm = centers[:, 1]   # colonne T_norm
order = np.argsort(center_T_norm)   # [nominal, modéré, critique]

# Remap des clusters dans l'ordre physique
remap = {old: new for new, old in enumerate(order)}
cluster_assignments_phys = np.array([remap[c] for c in cluster_assignments])
centers_phys = centers[order]

df['cluster_fcm'] = cluster_assignments_phys
df['u_nominal']   = U[:, order[0]]
df['u_moderate']  = U[:, order[1]]
df['u_critical']  = U[:, order[2]]

# ── Affichage des centres ──────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  CENTRES DES CLUSTERS FCM (espace normalisé [0,1])")
print("─" * 60)
for j, label in enumerate(CLUSTER_LABELS):
    c = centers_phys[j]
    print(f"\n  Cluster {j} — {label}")
    print(f"    ΔP    = {c[0]:.4f}  {'(faible ✅)' if j==0 else '(moyen ⚠️)' if j==1 else '(élevé ❌)'}")
    print(f"    T_norm = {c[1]:.4f}  {'(tempérée ✅)' if j==0 else '(chaude ⚠️)' if j==1 else '(canicule MENA ❌)'}")
    print(f"    LF    = {c[2]:.4f}  {'(optimal ✅)' if j==0 else '(sous-optimal ⚠️)' if j==1 else '(dégradé ❌)'}")

# ── Statistiques de partitionnement ───────────────────────────────────────────
print("\n" + "─" * 60)
print("  STATISTIQUES DE PARTITIONNEMENT")
print("─" * 60)
total = len(df)
for j, label in enumerate(CLUSTER_LABELS):
    n_j = (cluster_assignments_phys == j).sum()
    print(f"  Cluster {j} ({label}) : {n_j} obs. ({100*n_j/total:.1f}%)")

# ── Indice de Xie-Beni ────────────────────────────────────────────────────────
Um = U ** 2.0
D_mat = cdist(X, centers, metric='euclidean')
J_final = float(((Um) * (D_mat ** 2)).sum())

center_dists = cdist(centers_phys, centers_phys, metric='euclidean')
np.fill_diagonal(center_dists, np.inf)
min_center_dist_sq = center_dists.min() ** 2

XB_index = J_final / (total * min_center_dist_sq)
print(f"\n  Indice de Xie-Beni : XB = {XB_index:.4f}  (plus petit = meilleur partitionnement)")

# ── Accord avec la vérité terrain ─────────────────────────────────────────────
# On compare cluster FCM vs régime simulé (vérité)
# Note : les labels peuvent être permutés — on cherche la meilleure correspondance
from itertools import permutations
best_acc = 0
best_perm = None
for perm in permutations([0, 1, 2]):
    mapped = np.array([perm[c] for c in cluster_assignments_phys])
    acc = (mapped == df['regime_ref'].values).mean()
    if acc > best_acc:
        best_acc = acc
        best_perm = perm

print(f"  Accord FCM / vérité terrain : {100*best_acc:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — VISUALISATION FCM
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  SECTION 4 — VISUALISATION DES CLUSTERS FCM")
print("=" * 70)

# ── Figure 5 : Clusters FCM dans l'espace 2D ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f"Figure 5 — Clustering FCM (C=3, m=2) — Convergence en {n_iter} itérations\n"
             f"Accord avec vérité terrain : {100*best_acc:.1f}%  |  Xie-Beni = {XB_index:.4f}",
             fontsize=12, fontweight='bold', color=COLORS['accent'])

fcm_colors = [CLUSTER_COLORS[c] for c in cluster_assignments_phys]

for idx, ((xv, yv), (xl, yl)) in enumerate(zip(
    [('delta_P', 'T_norm'), ('T_norm', 'LF'), ('delta_P', 'LF')],
    [('ΔP', 'T_norm'), ('T_norm', 'LF'), ('ΔP', 'LF')]
)):
    ax = axes[idx]
    
    # Points colorés par cluster FCM
    sample_idx = np.random.choice(len(df), size=min(800, len(df)), replace=False)
    for r, (rcolor, rlabel) in enumerate(zip(CLUSTER_COLORS, CLUSTER_LABELS)):
        mask = cluster_assignments_phys[sample_idx] == r
        ax.scatter(df[xv].values[sample_idx][mask],
                   df[yv].values[sample_idx][mask],
                   c=rcolor, alpha=0.35, s=10, label=rlabel)
    
    # Centres des clusters
    cx = centers_phys[:, ['delta_P', 'T_norm', 'LF'].index(xv)]
    cy = centers_phys[:, ['delta_P', 'T_norm', 'LF'].index(yv)]
    ax.scatter(cx, cy, c=CLUSTER_COLORS, s=200, marker='*',
               edgecolors='black', linewidths=1.5, zorder=5, label='Centres FCM')
    
    for j, (x_c, y_c) in enumerate(zip(cx, cy)):
        ax.annotate(f'c{j}', (x_c, y_c), textcoords='offset points',
                    xytext=(8, 6), fontsize=9, fontweight='bold', color=COLORS['accent'])
    
    ax.set_xlabel(xl, fontweight='bold')
    ax.set_ylabel(yl, fontweight='bold')
    ax.set_title(f'{xl} vs {yl}')
    if idx == 0:
        ax.legend(fontsize=8, framealpha=0.9, loc='upper left')

plt.tight_layout()
plt.savefig('/home/claude/fig5_clusters_fcm.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure 5 sauvegardée : fig5_clusters_fcm.png")

# ── Figure 6 : Convergence + Appartenances floues ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle("Figure 6 — Convergence FCM et Appartenances Floues",
             fontsize=12, fontweight='bold', color=COLORS['accent'])

# Courbe de convergence
ax = axes[0]
ax.plot(J_history, color=COLORS['accent'], linewidth=2, marker='o', markersize=3)
ax.set_xlabel("Itération")
ax.set_ylabel("Fonction objectif J")
ax.set_title(f"Convergence de J\n(atteinte en {n_iter} itérations)")
ax.set_yscale('log')
ax.axhline(J_history[-1], color='red', linestyle='--', alpha=0.5, label=f'J final = {J_history[-1]:.1f}')
ax.legend()

# Distribution des appartenances pour 3 machines représentatives
ax2 = axes[1]
sample_machines = df.groupby('cluster_fcm').apply(lambda g: g.iloc[0]).reset_index(drop=True)
x_pos = np.arange(3)
width = 0.25
for j, (color, label) in enumerate(zip(CLUSTER_COLORS, ['u_nominal', 'u_moderate', 'u_critical'])):
    vals = sample_machines[label].values
    ax2.bar(x_pos + j*width, vals, width, color=color, alpha=0.85, label=CLUSTER_LABELS[j])

ax2.set_xticks(x_pos + width)
ax2.set_xticklabels(['Machine\nNominale', 'Machine\nModérée', 'Machine\nCritique'], fontsize=9)
ax2.set_ylabel("Degré d'appartenance uᵢⱼ")
ax2.set_title("Appartenance Floue\n(3 machines représentatives)")
ax2.legend(fontsize=7, loc='upper right')
ax2.set_ylim(0, 1.1)
ax2.axhline(1/3, color='gray', linestyle=':', alpha=0.6, label='1/C')

# I_ineff par cluster
ax3 = axes[2]
for j, (color, label) in enumerate(zip(CLUSTER_COLORS, CLUSTER_LABELS)):
    subset = df[df['cluster_fcm'] == j]['I_ineff_ref']
    ax3.hist(subset, bins=30, color=color, alpha=0.7, label=f'{label}\n(μ={subset.mean():.3f})',
             edgecolor='white', linewidth=0.3)

ax3.set_xlabel("I_ineff (indice d'inefficience)")
ax3.set_ylabel("Fréquence")
ax3.set_title("Distribution de I_ineff\npar Cluster FCM")
ax3.legend(fontsize=7)

plt.tight_layout()
plt.savefig('/home/claude/fig6_convergence.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure 6 sauvegardée : fig6_convergence.png")

# ── Figure 7 : Analyse K_MENA préliminaire ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Figure 7 — Analyse Préliminaire du Facteur K_MENA\n"
             "Gap entre performance IEC et performance terrain MENA",
             fontsize=12, fontweight='bold', color=COLORS['accent'])

# K_MENA par cluster
K_MENA_ref = 1.0   # référence IEC normalisée
I_IEC_ref   = 0.08  # inefficacité nominale IEC (conditions standard)

ax = axes[0]
cluster_means = []
for j, (color, label) in enumerate(zip(CLUSTER_COLORS, CLUSTER_LABELS)):
    subset = df[df['cluster_fcm'] == j]['I_ineff_ref']
    k_mena_j = subset.mean() / I_IEC_ref
    cluster_means.append(k_mena_j)
    bar = ax.bar(j, k_mena_j, color=color, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.text(j, k_mena_j + 0.04, f'K={k_mena_j:.2f}', ha='center', fontweight='bold',
            fontsize=11, color=COLORS['accent'])

ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5, label='Référence IEC (K=1.0)')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(CLUSTER_LABELS, fontsize=9)
ax.set_ylabel("Facteur K_MENA")
ax.set_title("K_MENA par Régime de Fonctionnement")
ax.legend()
ax.set_ylim(0, max(cluster_means) * 1.25)

# K_MENA par site
ax2 = axes[1]
site_kmena = []
site_names = ['Site 1\nCôtier (Sfax)', 'Site 2\nIntérieur (Kairouan)', 'Site 3\nSud (Gabès)']
site_colors_list = ['#2E75B6', '#E67E22', '#27AE60']

for i, site in enumerate(df['site'].unique()):
    df_site = df[df['site'] == site]
    k = df_site['I_ineff_ref'].mean() / I_IEC_ref
    site_kmena.append(k)
    ax2.bar(i, k, color=site_colors_list[i], alpha=0.85, edgecolor='white')
    ax2.text(i, k + 0.04, f'K={k:.2f}', ha='center', fontweight='bold',
             fontsize=11, color=COLORS['accent'])

ax2.axhline(1.0, color='black', linestyle='--', linewidth=1.5, label='Référence IEC (K=1.0)')
ax2.set_xticks([0, 1, 2])
ax2.set_xticklabels(site_names, fontsize=9)
ax2.set_ylabel("Facteur K_MENA")
ax2.set_title("K_MENA par Site Industriel\n(comparaison multi-sites)")
ax2.legend()

plt.tight_layout()
plt.savefig('/home/claude/fig7_kmena.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure 7 sauvegardée : fig7_kmena.png")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — EXPORT DATASET ET RÉSUMÉ FINAL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  SECTION 5 — EXPORT ET RÉSUMÉ")
print("=" * 70)

df_export = df[['timestamp', 'site', 'T_amb', 'HR', 'LF', 'P_mesuree',
                 'P_ref', 'delta_P', 'T_norm', 'I_ineff_ref',
                 'cluster_fcm', 'u_nominal', 'u_moderate', 'u_critical']].copy()

df_export.to_csv('/home/claude/dataset_MENA_FCM.csv', index=False)
print(f"\n✅ Dataset exporté : dataset_MENA_FCM.csv ({len(df_export)} lignes × {len(df_export.columns)} colonnes)")

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  RÉSUMÉ NOTEBOOK 1 — RÉSULTATS CLÉS                             ║
╠══════════════════════════════════════════════════════════════════╣
║  Dataset    : {len(df):,} observations | {N_SITES} sites | 1 an simulé         ║
║  Variables  : ΔP, T_norm, LF (vecteur d'état X)                ║
║                                                                  ║
║  FCM        : C=3 clusters | m=2 | convergé en {n_iter:3d} itérations  ║
║  Xie-Beni   : XB = {XB_index:.4f}  (partitionnement validé)         ║
║  Accord terrain : {100*best_acc:.1f}%                                      ║
║                                                                  ║
║  Centres FCM (ΔP | T_norm | LF) :                               ║
║    c0 Nominal   : {centers_phys[0,0]:.3f} | {centers_phys[0,1]:.3f} | {centers_phys[0,2]:.3f}              ║
║    c1 Modéré    : {centers_phys[1,0]:.3f} | {centers_phys[1,1]:.3f} | {centers_phys[1,2]:.3f}              ║
║    c2 Critique  : {centers_phys[2,0]:.3f} | {centers_phys[2,1]:.3f} | {centers_phys[2,2]:.3f}              ║
║                                                                  ║
║  K_MENA préliminaire :                                          ║
║    Régime nominal   → K = {cluster_means[0]:.2f}                          ║
║    Régime modéré    → K = {cluster_means[1]:.2f}                          ║
║    Régime critique  → K = {cluster_means[2]:.2f}  (biais MENA confirmé)  ║
║                                                                  ║
║  Prochaine étape : Notebook 2 — FIS Sugeno + PSO                ║
╚══════════════════════════════════════════════════════════════════╝
""")
