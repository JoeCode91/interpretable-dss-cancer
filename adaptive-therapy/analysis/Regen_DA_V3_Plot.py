"""
Regen_DA_V3_Plot.py - Regenerate DA V3 figure with dual-axis frequency subplot

REPRODUCES IN THE THESIS
    Regenerates the dual-agent figures from saved results. Utility.

PROVENANCE
    Copied verbatim from the working tree of the thesis. The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md at the
    repository root.

--------------------------------------------------------------------------
Regen_DA_V3_Plot.py - Regenerate DA V3 figure with dual-axis frequency subplot
Loads saved results and replots with x0 on secondary axis.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

IMG_DIR = os.path.dirname(os.path.abspath(__file__))
FSIZE_AX = 14

# Load saved results
data = np.load(os.path.join(IMG_DIR, 'Results_DA_V3_Hp140.npz'), allow_pickle=True)
print("Keys:", list(data.keys()))

t_vec = data['t_vec']
Xhist_abs = data['Xhist_abs']
Xhist_freq = data['Xhist_freq']
Uhist = data['Uhist']
N_max = float(data['N_max'])
Hp = int(data['Hp'])
Bw = int(data['Bw']) if 'Bw' in data else 80
t_elapsed = float(data['t_elapsed']) if 't_elapsed' in data else 0.0

N_total_mpc = np.sum(Xhist_abs, axis=0)
N_from_zf = N_max * (1 - Xhist_freq[0, :])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Replicator-MPC DA V3 | Hp={Hp}, Bw={Bw} | $N_{{max}}$={N_max:.0e} | Time={t_elapsed:.1f}s',
             fontsize=15)

# Top-left: Cell Dynamics (Plant - LV)
ax = axes[0, 0]
ax.plot(t_vec, Xhist_abs[0, :], 'b-', lw=2, label='T+')
ax.plot(t_vec, Xhist_abs[1, :], 'g-', lw=2, label='TP')
ax.plot(t_vec, Xhist_abs[2, :], 'r-', lw=2, label='T-/+')
ax.plot(t_vec, Xhist_abs[3, :], 'c-', lw=2, label='T-/-')
ax.set_xlabel('Time (days)', fontsize=FSIZE_AX)
ax.set_ylabel('Population', fontsize=FSIZE_AX)
ax.set_title('Cell Dynamics (Plant - LV)', fontsize=FSIZE_AX)
ax.legend(fontsize=11)
ax.tick_params(labelsize=FSIZE_AX)
ax.grid(True)

# Top-right: Population Frequencies (Replicator) - DUAL AXIS
ax = axes[0, 1]
ax.plot(t_vec, Xhist_freq[1, :], 'b-', lw=2, label='$x_{T+}$')
ax.plot(t_vec, Xhist_freq[2, :], 'g-', lw=2, label='$x_{TP}$')
ax.plot(t_vec, Xhist_freq[3, :], 'r-', lw=2, label='$x_{T-/+}$')
ax.plot(t_vec, Xhist_freq[4, :], 'c-', lw=2, label='$x_{T-/-}$')
ax.set_xlabel('Time (days)', fontsize=FSIZE_AX)
ax.set_ylabel('Frequency (populations)', fontsize=FSIZE_AX)
ax.set_title('Population Frequencies (Replicator)', fontsize=FSIZE_AX)
ax.tick_params(labelsize=FSIZE_AX)
ax.grid(True)
ax2 = ax.twinx()
ax2.plot(t_vec, Xhist_freq[0, :], 'k--', lw=1.5, label='$x_0$ (fictitious)')
ax2.set_ylabel('$x_0$ (fictitious zone)', fontsize=FSIZE_AX)
ax2.tick_params(labelsize=FSIZE_AX)
# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='center right')

# Bottom-left: Treatment Schedule (Abiraterone)
ax = axes[1, 0]
ax.plot(t_vec, Uhist[0, :] if Uhist.ndim > 1 else Uhist, 'b.', ms=2, label='Abiraterone')
if Uhist.ndim > 1:
    ax.plot(t_vec, Uhist[1, :], 'g.', ms=2, label='Docetaxel')
ax.set_xlabel('Time (days)', fontsize=FSIZE_AX)
ax.set_ylabel('Drug dose', fontsize=FSIZE_AX)
ax.set_title('Treatment Schedule', fontsize=FSIZE_AX)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=11)
ax.tick_params(labelsize=FSIZE_AX)
ax.grid(True)

# Bottom-right: Tumor Burden
ax = axes[1, 1]
ax.plot(t_vec, N_total_mpc, 'purple', lw=2, label='$N_{tumor}$ (plant)')
ax.plot(t_vec, N_from_zf, 'purple', lw=1.5, ls='--', label='$N_{max}(1-x_0)$')
ax.set_xlabel('Time (days)', fontsize=FSIZE_AX)
ax.set_ylabel('Total population', fontsize=FSIZE_AX)
ax.set_title('Tumor Burden', fontsize=FSIZE_AX)
ax.legend(fontsize=11)
ax.tick_params(labelsize=FSIZE_AX)
ax.grid(True)

plt.tight_layout()
out_path = os.path.join(IMG_DIR, f'Replicator_MPC_DA_V3_Hp{Hp}.png')
plt.savefig(out_path, dpi=150)
plt.close()
print(f"Saved: {out_path}")
