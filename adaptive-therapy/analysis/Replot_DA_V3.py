"""
Replot DA V3 results from saved .npz with improved figure (2x2):

REPRODUCES IN THE THESIS
    Regenerates the dual-agent figures from saved results. Utility, produces no new numbers.

PROVENANCE
    Copied verbatim from the working tree of the thesis
    (Modelos_Replicator/Codigos/Replot_DA_V3.py). The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md
    at the repository root for the complete list of differences.

--------------------------------------------------------------------------
Replot DA V3 results from saved .npz with improved figure (2x2):
  - Panel [0,0]: Cell Dynamics (linear, all 4 populations, T^{-/-} dashed)
  - Panel [0,1]: Population Frequencies (Replicator)
  - Panel [1,0]: Treatment Schedule (Abi blue + Doc green, combined)
  - Panel [1,1]: Tumor Burden
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Load data ---
data = np.load('Results_DA_V3_Hp140.npz', allow_pickle=True)

Xhist_abs  = data['Xhist_abs']
Xhist_freq = data['Xhist_freq']
Uhist      = data['Uhist']
t_vec      = data['t_vec']
N_max      = float(data['N_max'])
Hp         = int(data['Hp'])
t_elapsed  = float(data['t_elapsed'])

# --- Derived quantities ---
N_total = np.sum(Xhist_abs, axis=0)
N_zf    = N_max * (1 - Xhist_freq[0, :])

drug_abi = np.mean(Uhist[0, :])
drug_doc = np.mean(Uhist[1, :])

FSIZE_AX = 14
FSIZE_LG = 11

# --- Figure 2x2 ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f'Replicator-MPC DA V3 | Hp={Hp} | $N_{{max}}$={N_max:.0e} | '
    f'Time={t_elapsed:.1f}s | Abi={drug_abi:.3f} | Doc={drug_doc:.3f}',
    fontsize=15
)

# ===== Panel [0,0]: Cell Dynamics - ALL 4 (T^{-/-} dashed) =====
ax = axes[0, 0]
ax.plot(t_vec, Xhist_abs[0, :], 'b-',  lw=2, label='$T^+$')
ax.plot(t_vec, Xhist_abs[1, :], 'g-',  lw=2, label='$T^P$')
ax.plot(t_vec, Xhist_abs[2, :], 'r-',  lw=2, label='$T^{-/+}$')
ax.plot(t_vec, Xhist_abs[3, :], 'c--', lw=2, label='$T^{-/-}$')
ax.set_xlabel('Time (days)', fontsize=FSIZE_AX)
ax.set_ylabel('Population', fontsize=FSIZE_AX)
ax.set_title('Cell Dynamics (Plant - LV)', fontsize=FSIZE_AX)
ax.legend(fontsize=FSIZE_LG)
ax.tick_params(labelsize=FSIZE_AX)
ax.set_ylim(-1, 10000)
ax.grid(True)

# ===== Panel [0,1]: Population Frequencies =====
ax = axes[0, 1]
ax.plot(t_vec, Xhist_freq[0, :], 'k--', lw=1.5, label='$x_0$ (fict.)')
ax.plot(t_vec, Xhist_freq[1, :], 'b-',  lw=2,   label='$x_{T^+}$')
ax.plot(t_vec, Xhist_freq[2, :], 'g-',  lw=2,   label='$x_{T^P}$')
ax.plot(t_vec, Xhist_freq[3, :], 'r-',  lw=2,   label='$x_{T^{-/+}}$')
ax.plot(t_vec, Xhist_freq[4, :], 'c--', lw=2,   label='$x_{T^{-/-}}$')
ax.set_xlabel('Time (days)', fontsize=FSIZE_AX)
ax.set_ylabel('Frequency', fontsize=FSIZE_AX)
ax.set_title('Population Frequencies (Replicator)', fontsize=FSIZE_AX)
ax.legend(fontsize=FSIZE_LG)
ax.tick_params(labelsize=FSIZE_AX)
ax.set_ylim(-0.05, 1.05)
ax.grid(True)

# ===== Panel [1,0]: Treatment Schedule (COMBINED) =====
ax = axes[1, 0]
ax.plot(t_vec, Uhist[0, :], 'b-', lw=1.5, label=f'Abiraterone (mean={drug_abi:.3f})')
ax.plot(t_vec, Uhist[1, :], 'g-', lw=1.5, label=f'Docetaxel (mean={drug_doc:.3f})')
ax.set_xlabel('Time (days)', fontsize=FSIZE_AX)
ax.set_ylabel('Normalized dose', fontsize=FSIZE_AX)
ax.set_title('Treatment Schedule', fontsize=FSIZE_AX)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=FSIZE_LG)
ax.tick_params(labelsize=FSIZE_AX)
ax.grid(True)

# ===== Panel [1,1]: Tumor Burden =====
ax = axes[1, 1]
ax.plot(t_vec, N_total, 'purple', lw=2, label='$N$ (plant)')
ax.plot(t_vec, N_zf, 'purple', lw=1.5, ls='--', label='$N_{max}(1-x_0)$')
ax.set_xlabel('Time (days)', fontsize=FSIZE_AX)
ax.set_ylabel('Total population', fontsize=FSIZE_AX)
ax.set_title('Tumor Burden', fontsize=FSIZE_AX)
ax.legend(fontsize=FSIZE_LG)
ax.tick_params(labelsize=FSIZE_AX)
ax.grid(True)

plt.tight_layout()
plt.savefig('Replicator_MPC_DA_V3_Hp140.png', dpi=150)
plt.close()
print('Figure saved: Replicator_MPC_DA_V3_Hp140.png')
