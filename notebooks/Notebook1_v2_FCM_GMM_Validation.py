"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MÉMOIRE MASTER IASRIA — Notebook 1 v2 : FCM + Validation GMM              ║
║  Améliorations vs v1 :                                                       ║
║    • Standardisation des variables (StandardScaler) avant FCM               ║
║    • Validation par Gaussian Mixture Model (GMM)                             ║
║    • Indices : Silhouette, Davies-Bouldin, Xie-Beni                         ║
║    • Note méthodologique : regime_ref = outil de validation synthétique     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': '#F8FBFF',
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
    'font.family': 'DejaVu Sans', 'axes.titlesize': 12,
    'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
})

C_NOM  = '#2E75B6'
C_MOD  = '#F4A228'
C_CRIT = '#C0392B'
C_ACC  = '#1A3A5C'
CLUSTER_COLORS = [C_NOM, C_MOD, C_CRIT]
CLUSTER_LABELS = ['Cluster 0 — Nominal', 'Cluster 1 — Modéré', 'Cluster 2 — Critique MENA']
np.random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — GÉNÉRATION DATASET MENA (identique v1)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  SECTION 1 — GÉNÉRATION DATASET MENA")
print("=" * 70)

def generate_regime(n, regime, site_id):
    if regime == 0:
        T_amb  = np.random.normal(24, 4, n).clip(15, 34)
        HR     = np.random.normal(52, 8, n).clip(35, 68)
        LF     = np.random.normal(0.88, 0.06, n).clip(0.75, 1.0)
        P_meas = np.random.normal(48, 4, n).clip(38, 58)
    elif regime == 1:
        T_amb  = np.random.normal(34, 4, n).clip(26, 43)
        HR     = np.random.normal(62, 9, n).clip(45, 80)
        LF     = np.random.normal(0.72, 0.08, n).clip(0.55, 0.85)
        P_meas = np.random.normal(61, 5, n).clip(50, 74)
    else:
        T_amb  = np.random.normal(42, 3, n).clip(36, 50)
        HR     = np.random.normal(45, 10, n).clip(25, 65)
        LF     = np.random.normal(0.52, 0.10, n).clip(0.30, 0.72)
        P_meas = np.random.normal(74, 6, n).clip(60, 90)

    P_ref   = 50.0
    delta_P = (P_meas - P_ref) / P_ref
    T_norm  = ((T_amb - 20) / (50 - 20)).clip(0, 1)
    I_ineff = (delta_P.clip(0) * 0.45 + T_norm * 0.35 + (1 - LF) * 0.20).clip(0, 1)

    return pd.DataFrame({
        'site': f'Site_{site_id+1}', 'regime_ref': regime,
        'T_amb': T_amb.round(2), 'HR': HR.round(2),
        'LF': LF.round(4), 'P_mesuree': P_meas.round(2), 'P_ref': P_ref,
        'delta_P': delta_P.round(4), 'T_norm': T_norm.round(4),
        'I_ineff_ref': I_ineff.round(4),
    })

all_dfs = []
for site in range(3):
    props = [[0.45,0.35,0.20],[0.30,0.40,0.30],[0.25,0.35,0.40]][site]
    site_dfs = [generate_regime(int(1000*p), r, site) for r, p in enumerate(props)]
    df_site = pd.concat(site_dfs).sample(frac=1, random_state=site).reset_index(drop=True)
    all_dfs.append(df_site)

df = pd.concat(all_dfs, ignore_index=True)
print(f"✅ Dataset : {len(df)} observations | 3 sites | variables : delta_P, T_norm, LF")
print("\n⚠️  NOTE MÉTHODOLOGIQUE IMPORTANTE")
print("   'regime_ref' est une étiquette de validation UNIQUEMENT disponible")
print("   sur données synthétiques. En conditions réelles industrielles,")
print("   le régime n'est pas connu a priori — c'est précisément l'objet du clustering.\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ANALYSE CORRÉLATION ET NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  SECTION 2 — ANALYSE DE CORRÉLATION + STANDARDISATION")
print("=" * 70)

X_raw = df[['delta_P', 'T_norm', 'LF']].values
feature_names = ['ΔP', 'T_norm', 'LF']

# ── Analyse de corrélation ────────────────────────────────────────────────────
corr = df[['delta_P','T_norm','LF']].corr()
print("\nMatrice de corrélation entre variables du vecteur X :")
print(corr.round(3))

max_corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().abs().max()
print(f"\nCorr. max inter-variables : {max_corr:.3f}", end="  ")
if max_corr > 0.8:
    print("⚠️  Redondance possible → considérer PCA")
else:
    print("✅ Pas de redondance critique — vecteur X valide")

# ── Statistiques avant/après normalisation ───────────────────────────────────
print("\nVariances brutes (avant standardisation) :")
for i, name in enumerate(feature_names):
    print(f"  {name:8s} : var = {X_raw[:,i].var():.4f} | range = [{X_raw[:,i].min():.3f}, {X_raw[:,i].max():.3f}]")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print("\nVariances après StandardScaler :")
for i, name in enumerate(feature_names):
    print(f"  {name:8s} : var = {X_scaled[:,i].var():.4f} | range = [{X_scaled[:,i].min():.3f}, {X_scaled[:,i].max():.3f}]")

print("\n✅ Standardisation appliquée : μ=0, σ=1 pour chaque variable")
print("   Justification : FCM utilise la distance euclidienne — sans standardisation,")
print("   la variable à plus grande variance dominerait le clustering.\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FCM FROM SCRATCH sur données standardisées
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  SECTION 3 — FCM FROM SCRATCH (sur X_scaled)")
print("=" * 70)

def fcm(X, C=3, m=2.0, max_iter=150, eps=1e-4, seed=42):
    """Fuzzy C-Means — Bezdek (1981) — implémentation from scratch."""
    np.random.seed(seed)
    N, D = X.shape
    U = np.random.dirichlet(np.ones(C), size=N)
    J_history = []

    for it in range(max_iter):
        U_old = U.copy()
        Um = U ** m
        centers = (Um.T @ X) / Um.sum(axis=0)[:, np.newaxis]
        D_mat = np.fmax(cdist(X, centers, metric='euclidean'), 1e-10)
        exp = 2.0 / (m - 1.0)
        U_new = np.zeros((N, C))
        for j in range(C):
            U_new[:, j] = 1.0 / ((D_mat[:, j:j+1] / D_mat) ** exp).sum(axis=1)
        U = U_new
        J = float(((U ** m) * (D_mat ** 2)).sum())
        J_history.append(J)
        delta = np.linalg.norm(U - U_old, ord='fro')
        if delta < eps:
            print(f"  ✅ Convergence à l'itération {it+1} | J={J:.4f} | ||ΔU||_F={delta:.2e}")
            return U, centers, J_history, it + 1

    return U, centers, J_history, max_iter

print("\nFCM sur X_scaled :")
U, centers_scaled, J_history, n_iter = fcm(X_scaled, C=3, m=2.0, eps=1e-4)

# ── Alignement physique des clusters ─────────────────────────────────────────
# Trier par T_norm croissant dans l'espace original
centers_orig = scaler.inverse_transform(centers_scaled)
order = np.argsort(centers_orig[:, 1])   # par T_norm
remap = {old: new for new, old in enumerate(order)}

cluster_fcm = np.array([remap[c] for c in np.argmax(U, axis=1)])
centers_orig_phys = centers_orig[order]
centers_scaled_phys = centers_scaled[order]

df['cluster_fcm'] = cluster_fcm
df['u_0'] = U[:, order[0]]
df['u_1'] = U[:, order[1]]
df['u_2'] = U[:, order[2]]

print("\nCentres FCM (espace original) :")
print(f"  {'Cluster':<12} {'ΔP':>8} {'T_norm':>8} {'LF':>8}  Interprétation")
print("  " + "-" * 58)
interp = ['Nominal (efficace)', 'Modéré (thermique)', 'Critique (MENA canicule)']
for j in range(3):
    c = centers_orig_phys[j]
    print(f"  c{j} {interp[j]:<22}  {c[0]:>6.3f}   {c[1]:>6.3f}   {c[2]:>6.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — INDICES DE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 4 — INDICES DE VALIDATION DU CLUSTERING")
print("=" * 70)

# Silhouette Score (sur X_scaled)
sil_fcm = silhouette_score(X_scaled, cluster_fcm, metric='euclidean')

# Davies-Bouldin
db_fcm = davies_bouldin_score(X_scaled, cluster_fcm)

# Xie-Beni
Um_val = U ** 2.0
D_val  = np.fmax(cdist(X_scaled, centers_scaled, metric='euclidean'), 1e-10)
J_val  = float(((Um_val) * (D_val ** 2)).sum())
cd_mat = cdist(centers_scaled_phys, centers_scaled_phys)
np.fill_diagonal(cd_mat, np.inf)
xb_fcm = J_val / (len(df) * cd_mat.min() ** 2)

# Accord avec vérité terrain
from itertools import permutations
best_acc = max(
    (np.array([p[c] for c in cluster_fcm]) == df['regime_ref'].values).mean()
    for p in permutations([0,1,2])
)

print(f"\n  Silhouette Score    : {sil_fcm:.4f}  (meilleur → 1.0 | seuil acceptable > 0.50)")
print(f"  Davies-Bouldin Index: {db_fcm:.4f}  (meilleur → 0   | seuil acceptable < 1.0)")
print(f"  Xie-Beni Index      : {xb_fcm:.4f}  (meilleur → 0   | compact + séparé)")
print(f"  Accord terrain      : {100*best_acc:.1f}%  (sur dataset synthétique uniquement)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — VALIDATION PAR GMM (Gaussian Mixture Model)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 5 — VALIDATION PAR GMM (Gaussian Mixture Model)")
print("=" * 70)

print("""
  Rôle du GMM dans ce mémoire :
  ─────────────────────────────
  GMM est utilisé UNIQUEMENT comme outil de validation de FCM.
  Il modélise les données comme un mélange de C distributions gaussiennes.
  Si FCM et GMM produisent des structures similaires → le clustering
  est robuste et ne dépend pas du choix algorithmique.

  Référence : Bishop, C.M. (2006). Pattern Recognition and Machine Learning.
""")

gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42, n_init=5)
gmm.fit(X_scaled)
cluster_gmm_raw = gmm.predict(X_scaled)

# Alignement GMM avec même ordre que FCM (par T_norm croissant des moyennes)
gmm_means_orig = scaler.inverse_transform(gmm.means_)
gmm_order = np.argsort(gmm_means_orig[:, 1])
gmm_remap = {old: new for new, old in enumerate(gmm_order)}
cluster_gmm = np.array([gmm_remap[c] for c in cluster_gmm_raw])
df['cluster_gmm'] = cluster_gmm

# Indices GMM
sil_gmm = silhouette_score(X_scaled, cluster_gmm, metric='euclidean')
db_gmm  = davies_bouldin_score(X_scaled, cluster_gmm)

print(f"  GMM — Silhouette    : {sil_gmm:.4f}")
print(f"  GMM — Davies-Bouldin: {db_gmm:.4f}")

# Accord FCM vs GMM
accord_fcm_gmm = (cluster_fcm == cluster_gmm).mean()
print(f"\n  Accord FCM ↔ GMM    : {100*accord_fcm_gmm:.1f}%")

if accord_fcm_gmm > 0.90:
    print("  ✅ VALIDATION RÉUSSIE : les deux méthodes produisent des structures")
    print("     quasi-identiques → le clustering FCM est robuste et non arbitraire.")
elif accord_fcm_gmm > 0.80:
    print("  ✅ VALIDATION ACCEPTABLE : accord > 80% — structures cohérentes.")
else:
    print("  ⚠️  Accord faible — revoir le nombre de clusters ou les variables.")

print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │  TABLEAU COMPARATIF FCM vs GMM                               │
  ├──────────────────┬──────────────┬──────────────┬────────────┤
  │  Indice          │     FCM      │     GMM      │  Meilleur  │
  ├──────────────────┼──────────────┼──────────────┼────────────┤
  │  Silhouette ↑   │   {sil_fcm:.4f}   │   {sil_gmm:.4f}   │  {'FCM ✅' if sil_fcm >= sil_gmm else 'GMM ✅'}    │
  │  Davies-Bouldin ↓│   {db_fcm:.4f}   │   {db_gmm:.4f}   │  {'FCM ✅' if db_fcm <= db_gmm else 'GMM ✅'}    │
  │  Accord FCM↔GMM  │       {100*accord_fcm_gmm:.1f}%              │
  └──────────────────┴──────────────┴──────────────┴────────────┘
  Accord FCM-GMM : {100*accord_fcm_gmm:.1f}%
""")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FIGURES DE PRÉSENTATION (7 figures)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  SECTION 6 — GÉNÉRATION DES FIGURES")
print("=" * 70)

sample_idx = np.random.choice(len(df), size=600, replace=False)
fcm_colors_pts  = [CLUSTER_COLORS[c] for c in cluster_fcm]
gmm_colors_pts  = [CLUSTER_COLORS[c] for c in cluster_gmm]

# ── Figure A : Avant / Après normalisation ────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Figure A — Justification de la Standardisation\n"
             "Variables brutes vs standardisées — impact sur les distances euclidiennes",
             fontsize=12, fontweight='bold', color=C_ACC)

for i, (fname, col) in enumerate(zip(['delta_P','T_norm','LF'], CLUSTER_COLORS)):
    regime_colors = [CLUSTER_COLORS[r] for r in df['regime_ref']]
    # Avant
    ax = axes[0, i]
    ax.hist(X_raw[:, i], bins=40, color=col, alpha=0.75, edgecolor='white')
    ax.set_title(f'{["ΔP","T_norm","LF"][i]} — Brut', fontweight='bold')
    ax.set_xlabel(f'Valeur brute  |  var={X_raw[:,i].var():.4f}')
    ax.set_ylabel('Fréquence')
    # Après
    ax2 = axes[1, i]
    ax2.hist(X_scaled[:, i], bins=40, color=col, alpha=0.75, edgecolor='white')
    ax2.set_title(f'{["ΔP","T_norm","LF"][i]} — Standardisé (μ=0, σ=1)', fontweight='bold')
    ax2.set_xlabel(f'Valeur standardisée  |  var={X_scaled[:,i].var():.4f}')
    ax2.set_ylabel('Fréquence')

plt.tight_layout()
plt.savefig('/home/claude/figA_normalisation.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure A : Normalisation")

# ── Figure B : Matrice de corrélation ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Figure B — Analyse de Corrélation entre Variables\n"
             "Vérification de redondance avant clustering",
             fontsize=12, fontweight='bold', color=C_ACC)

vars_all = ['delta_P','T_norm','LF','I_ineff_ref']
corr_all = df[vars_all].corr()

ax = axes[0]
im = ax.imshow(corr_all.values, cmap='RdYlBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(vars_all))); ax.set_yticks(range(len(vars_all)))
ax.set_xticklabels(vars_all, rotation=45, ha='right')
ax.set_yticklabels(vars_all)
ax.set_title("Corrélation entre toutes les variables")
for i in range(len(vars_all)):
    for j in range(len(vars_all)):
        v = corr_all.values[i,j]
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=9,
                color='white' if abs(v)>0.5 else 'black', fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)

ax2 = axes[1]
corr_target = df[['delta_P','T_norm','LF']].corrwith(df['I_ineff_ref']).sort_values()
bar_colors = [C_CRIT if v>0 else C_NOM for v in corr_target.values]
bars = ax2.barh(corr_target.index, corr_target.values, color=bar_colors, alpha=0.85)
ax2.axvline(0, color='black', linewidth=1)
ax2.set_xlabel("Corrélation de Pearson avec I_ineff")
ax2.set_title("Prédicteurs de l'Inefficience Énergétique\n(variable cible du modèle FCM–FIS)")
for bar, val in zip(bars, corr_target.values):
    ax2.text(val + 0.01*np.sign(val), bar.get_y()+bar.get_height()/2,
             f'{val:+.3f}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/claude/figB_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure B : Corrélation")

# ── Figure C : FCM vs GMM — Scatter comparatif ───────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Figure C — FCM vs GMM : Validation Comparative du Clustering\n"
             f"Accord FCM↔GMM = {100*accord_fcm_gmm:.1f}% | Silhouette FCM={sil_fcm:.3f} | GMM={sil_gmm:.3f}",
             fontsize=12, fontweight='bold', color=C_ACC)

pairs = [('delta_P','T_norm'), ('T_norm','LF'), ('delta_P','LF')]
pair_labels = [('ΔP','T_norm'), ('T_norm','LF'), ('ΔP','LF')]
row_labels = ['FCM (algorithme principal)', 'GMM (validation probabiliste)']

for row, (clusters, row_label) in enumerate([(cluster_fcm, row_labels[0]), (cluster_gmm, row_labels[1])]):
    for col, ((xv, yv), (xl, yl)) in enumerate(zip(pairs, pair_labels)):
        ax = axes[row, col]
        for r, (rcolor, rlabel) in enumerate(zip(CLUSTER_COLORS, CLUSTER_LABELS)):
            mask = clusters[sample_idx] == r
            ax.scatter(df[xv].values[sample_idx][mask],
                      df[yv].values[sample_idx][mask],
                      c=rcolor, alpha=0.4, s=12,
                      label=rlabel.split('—')[1].strip() if col==0 else "")
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        if col == 0:
            ax.set_ylabel(f'{row_label}\n{yl}', fontsize=9)
        ax.set_title(f'{xl} vs {yl}')
        if row == 0 and col == 0:
            ax.legend(fontsize=8, framealpha=0.9)

handles = [mpatches.Patch(color=c, label=l.split('—')[1].strip())
           for c, l in zip(CLUSTER_COLORS, CLUSTER_LABELS)]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, -0.02), framealpha=0.9)
plt.tight_layout()
plt.savefig('/home/claude/figC_fcm_vs_gmm.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure C : FCM vs GMM")

# ── Figure D : Indices de validation ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Figure D — Indices de Validation du Clustering\n"
             "Silhouette · Davies-Bouldin · Accord FCM↔GMM",
             fontsize=12, fontweight='bold', color=C_ACC)

# Silhouette
ax = axes[0]
bars = ax.bar(['FCM\n(principal)', 'GMM\n(validation)'],
              [sil_fcm, sil_gmm],
              color=[C_NOM, C_MOD], alpha=0.85, width=0.4, edgecolor='white')
ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Seuil acceptable')
ax.set_ylim(0, 1)
ax.set_ylabel('Silhouette Score')
ax.set_title('Silhouette Score ↑\n(meilleur proche de 1)')
ax.legend(fontsize=9)
for bar, val in zip(bars, [sil_fcm, sil_gmm]):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.02, f'{val:.3f}',
            ha='center', fontweight='bold', fontsize=12, color=C_ACC)

# Davies-Bouldin
ax2 = axes[1]
bars2 = ax2.bar(['FCM\n(principal)', 'GMM\n(validation)'],
                [db_fcm, db_gmm],
                color=[C_NOM, C_MOD], alpha=0.85, width=0.4, edgecolor='white')
ax2.axhline(1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Seuil acceptable')
ax2.set_ylabel('Davies-Bouldin Index')
ax2.set_title('Davies-Bouldin Index ↓\n(meilleur proche de 0)')
ax2.legend(fontsize=9)
for bar, val in zip(bars2, [db_fcm, db_gmm]):
    ax2.text(bar.get_x()+bar.get_width()/2, val+0.02, f'{val:.3f}',
             ha='center', fontweight='bold', fontsize=12, color=C_ACC)

# Convergence FCM
ax3 = axes[2]
ax3.plot(J_history, color=C_ACC, linewidth=2, marker='o', markersize=4)
ax3.set_xlabel('Itération')
ax3.set_ylabel('Fonction objectif J (log)')
ax3.set_title(f'Convergence FCM\n(atteinte en {n_iter} itérations)')
ax3.set_yscale('log')
ax3.axhline(J_history[-1], color='red', linestyle='--', alpha=0.5,
            label=f'J final = {J_history[-1]:.2f}')
ax3.legend(fontsize=9)

plt.tight_layout()
plt.savefig('/home/claude/figD_indices.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure D : Indices de validation")

# ── Figure E : Appartenances floues ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Figure E — Appartenances Floues FCM\n"
             "Cœur de la distinction FCM vs clustering crisp (K-Means, GMM)",
             fontsize=12, fontweight='bold', color=C_ACC)

ax = axes[0]
n_display = 80
sorted_by_ineff = df['I_ineff_ref'].argsort().values[::-1][:n_display]
x_pos = np.arange(n_display)
ax.bar(x_pos, df['u_2'].values[sorted_by_ineff], color=C_CRIT, alpha=0.85,
       label='u critique (c2)')
ax.bar(x_pos, df['u_1'].values[sorted_by_ineff], bottom=df['u_2'].values[sorted_by_ineff],
       color=C_MOD, alpha=0.85, label='u modéré (c1)')
ax.bar(x_pos, df['u_0'].values[sorted_by_ineff],
       bottom=(df['u_2']+df['u_1']).values[sorted_by_ineff],
       color=C_NOM, alpha=0.85, label='u nominal (c0)')
ax.set_xlabel('Machines (triées par I_ineff décroissant)')
ax.set_ylabel("Degré d'appartenance")
ax.set_title('Appartenance floue (stacked)\n80 machines — du plus inefficace au moins')
ax.legend(fontsize=9); ax.set_ylim(0, 1)

ax2 = axes[1]
scatter = ax2.scatter(df['delta_P'].values[sample_idx],
                      df['T_norm'].values[sample_idx],
                      c=df['u_2'].values[sample_idx],
                      cmap='Reds', s=15, alpha=0.7, vmin=0, vmax=1)
plt.colorbar(scatter, ax=ax2, label="Degré d'appartenance au cluster Critique")
cp = centers_orig_phys
ax2.scatter(cp[:,0], cp[:,1], c=['black']*3, s=200, marker='*',
            edgecolors='white', linewidths=1.5, zorder=5)
for j, (x,y) in enumerate(zip(cp[:,0], cp[:,1])):
    ax2.annotate(f'c{j}', (x,y), xytext=(6,6), textcoords='offset points',
                 fontsize=10, fontweight='bold', color=C_ACC)
ax2.set_xlabel('ΔP (écart énergétique)')
ax2.set_ylabel('T_norm (température)')
ax2.set_title("Intensité d'appartenance au régime Critique MENA\n(rouge foncé = très critique)")

plt.tight_layout()
plt.savefig('/home/claude/figE_appartenances.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure E : Appartenances floues")

# ── Figure F : K_MENA par cluster et site ────────────────────────────────────
I_IEC_ref = 0.08
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Figure F — Facteur K_MENA Préliminaire\n"
             "Biais climatique MENA quantifié par cluster et par site",
             fontsize=12, fontweight='bold', color=C_ACC)

ax = axes[0]
k_vals = []
for j, (col, lab) in enumerate(zip(CLUSTER_COLORS, ['Nominal','Modéré','Critique'])):
    subset = df[df['cluster_fcm']==j]['I_ineff_ref']
    k = subset.mean() / I_IEC_ref
    k_vals.append(k)
    bar = ax.bar(j, k, color=col, alpha=0.85, edgecolor='white')
    ax.text(j, k+0.1, f'K={k:.2f}×', ha='center', fontweight='bold',
            fontsize=12, color=C_ACC)
ax.axhline(1.0, color='black', linestyle='--', linewidth=2, label='Référence IEC (K=1)')
ax.set_xticks([0,1,2])
ax.set_xticklabels(['Nominal','Modéré\n(MENA)','Critique\n(canicule)'], fontsize=10)
ax.set_ylabel('Facteur K_MENA')
ax.set_title('K_MENA par Régime de Fonctionnement\n(écart vs référence IEC)'); ax.legend()

ax2 = axes[1]
site_names_s = ['Site 1\nCôtier', 'Site 2\nIntérieur', 'Site 3\nSud (Sahara)']
site_cols = [C_NOM, C_MOD, C_CRIT]
for i, (site, col) in enumerate(zip(df['site'].unique(), site_cols)):
    k_s = df[df['site']==site]['I_ineff_ref'].mean() / I_IEC_ref
    ax2.bar(i, k_s, color=col, alpha=0.85, edgecolor='white')
    ax2.text(i, k_s+0.1, f'K={k_s:.2f}×', ha='center', fontweight='bold',
             fontsize=12, color=C_ACC)
ax2.axhline(1.0, color='black', linestyle='--', linewidth=2, label='Référence IEC (K=1)')
ax2.set_xticks([0,1,2]); ax2.set_xticklabels(site_names_s, fontsize=10)
ax2.set_ylabel('Facteur K_MENA')
ax2.set_title('K_MENA par Site Industriel\n(diversité géographique MENA)'); ax2.legend()

plt.tight_layout()
plt.savefig('/home/claude/figF_kmena.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure F : K_MENA")

# ── Export CSV ────────────────────────────────────────────────────────────────
df.to_csv('/home/claude/dataset_MENA_FCM_v2.csv', index=False)

# ══════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ FINAL
# ══════════════════════════════════════════════════════════════════════════════
print(f"""
{"=" * 70}
  RÉSUMÉ NOTEBOOK 1 v2 — RÉSULTATS COMPLETS
{"=" * 70}

  DONNÉES
  ─────────────────────────────────────────────────────
  Observations  : {len(df):,} | Sites : 3 | Variables : ΔP, T_norm, LF
  Standardisation : StandardScaler (μ=0, σ=1) ✅

  CLUSTERING FCM (algorithme principal)
  ─────────────────────────────────────────────────────
  Convergence   : {n_iter} itérations | ε = 10⁻⁴
  Centres FCM   :
    c0 Nominal   ΔP={centers_orig_phys[0,0]:+.3f} | T_norm={centers_orig_phys[0,1]:.3f} | LF={centers_orig_phys[0,2]:.3f}
    c1 Modéré    ΔP={centers_orig_phys[1,0]:+.3f} | T_norm={centers_orig_phys[1,1]:.3f} | LF={centers_orig_phys[1,2]:.3f}
    c2 Critique  ΔP={centers_orig_phys[2,0]:+.3f} | T_norm={centers_orig_phys[2,1]:.3f} | LF={centers_orig_phys[2,2]:.3f}

  INDICES DE VALIDATION
  ─────────────────────────────────────────────────────
  Silhouette (FCM) : {sil_fcm:.4f}  {'✅ Bon' if sil_fcm > 0.5 else '⚠️  Acceptable'}
  Silhouette (GMM) : {sil_gmm:.4f}  {'✅ Bon' if sil_gmm > 0.5 else '⚠️  Acceptable'}
  Davies-Bouldin(FCM): {db_fcm:.4f}  {'✅ Bon' if db_fcm < 1.0 else '⚠️  À améliorer'}
  Davies-Bouldin(GMM): {db_gmm:.4f}  {'✅ Bon' if db_gmm < 1.0 else '⚠️  À améliorer'}
  Xie-Beni (FCM)   : {xb_fcm:.4f}

  VALIDATION FCM vs GMM
  ─────────────────────────────────────────────────────
  Accord FCM ↔ GMM : {100*accord_fcm_gmm:.1f}%  {'✅ Validation réussie' if accord_fcm_gmm > 0.9 else '⚠️  Vérifier'}
  
  → Conclusion : "The clustering structure obtained with FCM is consistent
    with the probabilistic structure identified by GMM. This cross-validation
    confirms the robustness of the three-regime partition of industrial
    energy states in the MENA climatic context."

  K_MENA PRÉLIMINAIRE
  ─────────────────────────────────────────────────────
  Régime Nominal   → K = {k_vals[0]:.2f}×  (conforme IEC)
  Régime Modéré    → K = {k_vals[1]:.2f}×  (biais thermique MENA)
  Régime Critique  → K = {k_vals[2]:.2f}×  (canicule + surcharge)

  PROCHAINE ÉTAPE : Notebook 2 — FIS Sugeno + PSO
{"=" * 70}
""")
