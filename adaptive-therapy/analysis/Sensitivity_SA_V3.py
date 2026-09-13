"""
Sensitivity_SA_V3.py - Sensitivity Analysis for Single-Agent Replicator-MPC

REPRODUCES IN THE THESIS
    Single-agent sensitivity sweep: Tables tab:rep_sa_sensitivity and tab:rep_sa_full.

PROVENANCE
    Copied verbatim from the working tree of the thesis
    (Modelos_Replicator/Codigos/Sensitivity_SA_V3.py). The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md
    at the repository root for the complete list of differences.

--------------------------------------------------------------------------
Sensitivity_SA_V3.py - Sensitivity Analysis for Single-Agent Replicator-MPC
===========================================================================
V3: Fixes Hp>=80 divergence by using u_prev=0.0 (initial guess fix).
     Uses RK4 integration + max_iter=1000 (same as V2).

Hypothesis confirmed by Test_SA_InitGuess.py:
  u_prev=1.0 + B_rate=0.05 trapped IPOPT in local minimum for Hp>=80.
  u_prev=0.0 allows the optimizer to find the adaptive therapy solution.

Full factorial parameter sweep matching Table 3 of the article:
  5 gamma configurations x 6 Hp values = 30 configurations

Gamma configs (Table 3, BiologicalModels_EA_250930.pdf):
  Sim 1: gamma_1=3.6e-7, gamma_2=0.1, gamma_3=1.6e-3 (Baseline)
  Sim 2: gamma_1=3.6e-7, gamma_2=0.2, gamma_3=1.6e-3
  Sim 3: gamma_1=3.6e-7, gamma_2=0.5, gamma_3=1.6e-3
  Sim 4: gamma_1=3.6e-7, gamma_2=1.0, gamma_3=1.6e-3
  Sim 5: gamma_1=1,      gamma_2=100, gamma_3=1e6    (Extreme)

Hp values: [20, 50, 80, 100, 140, 200]

Output:
  Sensitivity_SA_V3/Datos/Results_SA_Sim{i}_Hp{hp}.npz    -- per configuration
  Sensitivity_SA_V3/Datos/Summary_SA.csv                   -- summary table
  Sensitivity_SA_V3/Imagenes/SA_Sim{i}_Hp{hp}_Overview.png -- per-config 2x2 plot
  Sensitivity_SA_V3/Imagenes/Sensitivity_SA_Comparison.png -- comparison plots

Based on: Sensitivity_SA_V2.py + Test_SA_InitGuess.py results
Author: Johan Pena-Campos, PhD Project
Date: 2026-03
"""

import casadi as cs
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import traceback
import sys
import os
import csv

from telegram_notifier import MPCNotifier

# Force unbuffered output for real-time logging
print = lambda *args, **kwargs: __builtins__.__dict__['print'](*args, **{**kwargs, 'flush': True})

def DM2Arr(dm):
    return np.array(dm.full())


# ================================================================
# CONFIGURATION (Table 3 of BiologicalModels_EA_250930.pdf)
# ================================================================

OUTPUT_DIR = "Sensitivity_SA_V3"
IMG_DIR = os.path.join(OUTPUT_DIR, "Imagenes")
DATA_DIR = os.path.join(OUTPUT_DIR, "Datos")
SKIP_EXISTING = True   # Skip configs that already have .npz files

# Parse --sims argument: e.g. --sims 1,2,3 or --sims=1,2,3
SELECTED_SIMS = None
for i, arg in enumerate(sys.argv):
    if arg.startswith('--sims='):
        SELECTED_SIMS = [int(s) for s in arg.split('=')[1].split(',')]
    elif arg == '--sims' and i + 1 < len(sys.argv):
        SELECTED_SIMS = [int(s) for s in sys.argv[i + 1].split(',')]

# Fixed parameters (same for all configs)
N_max = 8e4
dt = 10
Hs = 4000
B_rate = 0.05
y0_abs = np.array([560.36, 747.59, 47.10])

# --- Hp values (Table 3, also confirmed in Process_V3.pdf) ---
HP_VALUES = [20, 50, 80, 100, 140, 200]

# --- Gamma configurations (Table 3, eq. 4a) ---
# SA objective: J = gamma_1 * ||x_tilde||^2_Q + gamma_2 * ||u||^2_R + gamma_3 * x_{T-}
# Mapped to code weights: w_sens = gamma_1 * N_max^2, w_drug = gamma_2, w_res = gamma_3 * N_max
GAMMA_CONFIGS = [
    {'name': 'Sim1_Baseline', 'gamma_1': 3.6e-7, 'gamma_2': 0.1, 'gamma_3': 1.6e-3},
    {'name': 'Sim2_R02',      'gamma_1': 3.6e-7, 'gamma_2': 0.2, 'gamma_3': 1.6e-3},
    {'name': 'Sim3_R05',      'gamma_1': 3.6e-7, 'gamma_2': 0.5, 'gamma_3': 1.6e-3},
    {'name': 'Sim4_R10',      'gamma_1': 3.6e-7, 'gamma_2': 1.0, 'gamma_3': 1.6e-3},
    {'name': 'Sim5_Extreme',  'gamma_1': 1.0,    'gamma_2': 100,  'gamma_3': 1e6},
]

FSIZE = 14  # Font size for all plots


# ================================================================
# MODEL DEFINITIONS (from MPC_Replicator_SA_V3.py)
# ================================================================

r_lv = cs.DM([2.7726e-3, 3.4657e-3, 6.6542e-3])
A_lv = cs.DM([[1, 0.7, 0.8],
               [0.4, 1, 0.6],
               [0.5, 0.9, 1]])
mu_param = 1.5

# --- LV Plant model ---
y_s = cs.SX.sym('y', 3)
d_s = cs.SX.sym('d', 1)
K_lv = cs.vertcat(
    (mu_param - d_s) * y_s[1],
    1e4 - 9.9e3 * d_s,
    1e4
)
rhs_lv = cs.vertcat(
    (1 - cs.dot(A_lv[0,:], y_s.T) / K_lv[0]) * float(r_lv[0]) * y_s[0],
    (1 - cs.dot(A_lv[1,:], y_s.T) / K_lv[1]) * float(r_lv[1]) * y_s[1],
    (1 - cs.dot(A_lv[2,:], y_s.T) / K_lv[2]) * float(r_lv[2]) * y_s[2]
)
F_LV = cs.Function('F_LV', [y_s, d_s], [rhs_lv])

# --- Replicator + Zona Ficticia ---
n_pop = 3
n_total = n_pop + 1

p_s = cs.SX.sym('p', n_total)
u_s = cs.SX.sym('u', 1)
y_from_p = N_max * p_s[1:]
K_rep = cs.vertcat(
    (mu_param - u_s) * cs.fmax(y_from_p[1], 1e-10),
    1e4 - 9.9e3 * u_s,
    1e4
)
f_real = cs.SX.zeros(n_pop)
for i in range(n_pop):
    f_real[i] = float(r_lv[i]) * (1 - cs.dot(A_lv[i,:], y_from_p.T) / K_rep[i])
f_all = cs.vertcat(0, f_real)
phi_bar = cs.dot(p_s, f_all)
rhs_rep = p_s * (f_all - phi_bar)
F_REP = cs.Function('F_REP', [p_s, u_s], [rhs_rep])


def abs_to_freq(y_abs_val):
    y_arr = np.array(y_abs_val).flatten()
    p_real = y_arr / N_max
    p_0 = max(1.0 - np.sum(p_real), 0.0)
    return np.concatenate([[p_0], p_real])


# ================================================================
# SOLVER AND SIMULATION FUNCTIONS
# ================================================================

def build_sa_solver(Hp, w_sens, w_drug, w_res):
    """Build CasADi NLP solver for SA with RK4 integration + max_iter=1000."""
    U = cs.SX.sym('U', 1, Hp)
    X = cs.SX.sym('X', n_total, Hp + 1)
    n_par = n_total + 1  # state (4) + u_prev (1)
    P = cs.SX.sym('P', n_par)

    obj = 0
    X[:, 0] = P[:n_total]

    for t in range(Hp):
        st = X[:, t]
        con = U[:, t]
        # RK4 integration (replaces Euler for numerical stability)
        k1 = F_REP(st, con)
        k2 = F_REP(st + dt/2 * k1, con)
        k3 = F_REP(st + dt/2 * k2, con)
        k4 = F_REP(st + dt * k3, con)
        X[:, t+1] = st + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        obj += w_sens * (X[1, t+1]**2 + X[2, t+1]**2)
        obj += w_drug * (con[0]**2)
        obj += w_res * X[3, t+1]

    g = []
    for t in range(1, Hp + 1):
        for i in range(n_total):
            g = cs.vertcat(g, -X[i, t])
            g = cs.vertcat(g, X[i, t] - 1.0)
        g = cs.vertcat(g, -U[:, t-1])
        g = cs.vertcat(g, U[:, t-1] - 1.0)
        if t != 1:
            g = cs.vertcat(g, -(U[:, t-1] - U[:, t-2]) - B_rate)
            g = cs.vertcat(g, (U[:, t-1] - U[:, t-2]) - B_rate)
        else:
            g = cs.vertcat(g, -(P[n_total] - U[:, t-1]) - B_rate)
            g = cs.vertcat(g, (P[n_total] - U[:, t-1]) - B_rate)

    OPT_vars = cs.reshape(U, Hp, 1)
    nlp = {'f': obj, 'x': OPT_vars, 'g': g, 'p': P}
    opts = {
        'ipopt': {'max_iter': 1000, 'print_level': 0,
                  'acceptable_tol': 1e-8, 'acceptable_obj_change_tol': 1e-6},
        'print_time': 0,
        'error_on_fail': False,
    }
    solver = cs.nlpsol('solver', 'ipopt', nlp, opts)
    return solver, g.shape[0]


def run_sa_simulation(solver, Hp, n_g, label="", notifier=None,
                      config_idx=0, total_configs=1):
    """Run SA MPC simulation. Returns results dict with KPIs."""
    n_steps = int(Hs / dt)

    x0 = cs.DM(y0_abs)
    u_prev = 0.0
    U0 = u_prev * cs.DM.ones(1, Hp)

    Xhist_abs = np.zeros((3, n_steps + 1))
    Xhist_freq = np.zeros((n_total, n_steps + 1))
    Uhist = np.zeros(n_steps + 1)

    Xhist_abs[:, 0] = y0_abs
    Xhist_freq[:, 0] = abs_to_freq(y0_abs)
    Uhist[0] = u_prev

    lbg = [-float('inf')] * n_g
    ubg = [0] * n_g

    t0 = time.time()

    # Telegram milestones: every 25%
    notify_milestones = {25, 50, 75, 100}
    last_pct = 0

    for sim in tqdm(range(n_steps), desc=label, leave=True):
        p_curr = abs_to_freq(x0)
        Par = cs.DM(np.concatenate([p_curr, [u_prev]]))
        sol = solver(lbg=lbg, ubg=ubg, p=Par, x0=cs.reshape(U0, Hp, 1))
        U0 = sol['x'].full().reshape(1, Hp)

        u_opt = float(U0[0, 0])
        u_prev = u_opt

        x0 = x0 + dt * F_LV(x0, u_opt)

        Xhist_abs[:, sim + 1] = np.array(x0).flatten()
        Xhist_freq[:, sim + 1] = abs_to_freq(x0)
        Uhist[sim + 1] = u_opt

        U0 = np.hstack([U0[:, 1:], U0[:, -1:]])

        # Telegram progress
        pct = int(100 * (sim + 1) / n_steps)
        if notifier and pct in notify_milestones and pct > last_pct:
            last_pct = pct
            elapsed = time.time() - t0
            y_now = np.array(x0).flatten()
            notifier.notify_progress(sim + 1, n_steps, elapsed,
                extra_info=(
                    f"Config [{config_idx}/{total_configs}]: {label}\n"
                    f"u_a={u_opt:.3f}\n"
                    f"N_tumor={np.sum(y_now):.0f}\n"
                    f"Day {(sim+1)*dt}/{Hs}"
                ))

    t_elapsed = time.time() - t0

    # Compute KPIs
    N_tot = np.sum(Xhist_abs, axis=0)
    frac_res = Xhist_abs[2, :] / N_tot
    frac_sens = np.sum(Xhist_abs[:2, :], axis=0) / N_tot

    kpi1_idx = np.where(frac_res > frac_sens)[0]
    kpi1 = int(kpi1_idx[0] * dt) if len(kpi1_idx) > 0 else -1

    kpi2_idx = np.where(Xhist_abs[0, :] > y0_abs[0] * 2.5)[0]
    kpi2 = int(kpi2_idx[0] * dt) if len(kpi2_idx) > 0 else -1

    kpi3_idx = np.where(Xhist_abs[1, :] > y0_abs[1] * 2.5)[0]
    kpi3 = int(kpi3_idx[0] * dt) if len(kpi3_idx) > 0 else -1

    drug = float(np.mean(Uhist))

    return {
        'Xhist_abs': Xhist_abs, 'Xhist_freq': Xhist_freq,
        'Uhist': Uhist, 't_elapsed': t_elapsed,
        'kpi1': kpi1, 'kpi2': kpi2, 'kpi3': kpi3,
        'drug_exposure': drug,
    }


# ================================================================
# PLOT FUNCTIONS
# ================================================================

def generate_config_plot(res, cfg, img_dir, t_vec):
    """Generate 2x2 overview plot for a single config (matching SA V3 panorama).
    Layout:
      [0,0] Cell Dynamics (absolute populations T+, TP, T-)
      [0,1] Population Frequencies (x0, xT+, xTP, xT-)
      [1,0] Treatment Schedule
      [1,1] Tumor Burden (N_tumor + N_max*(1-x0))
    """
    Xhist_abs = res['Xhist_abs']
    Xhist_freq = res['Xhist_freq']
    Uhist = res['Uhist']
    N_total_mpc = np.sum(Xhist_abs, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sim_name = cfg['name']
    hp = cfg['Hp']
    fig.suptitle(
        f"Replicator-MPC SA | {sim_name} | Hp={hp} | "
        f"$\\gamma$=({cfg['gamma_1']:.1e}, {cfg['gamma_2']}, {cfg['gamma_3']:.1e}) | "
        f"Time={res['t_elapsed']:.1f}s",
        fontsize=FSIZE)

    # [0,0] Cell Dynamics
    ax = axes[0, 0]
    ax.plot(t_vec, Xhist_abs[0, :], 'b-', lw=2, label='T+')
    ax.plot(t_vec, Xhist_abs[1, :], 'g-', lw=2, label='TP')
    ax.plot(t_vec, Xhist_abs[2, :], 'r-', lw=2, label='T-')
    ax.set_xlabel('Time (days)', fontsize=FSIZE)
    ax.set_ylabel('Population', fontsize=FSIZE)
    ax.set_title('Cell Dynamics (Plant - LV)', fontsize=FSIZE)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=FSIZE)
    ax.grid(True)

    # [0,1] Population Frequencies
    ax = axes[0, 1]
    ax.plot(t_vec, Xhist_freq[0, :], 'k--', lw=1.5, label='$x_0$ (fictitious)')
    ax.plot(t_vec, Xhist_freq[1, :], 'b-', lw=2, label='$x_{T+}$')
    ax.plot(t_vec, Xhist_freq[2, :], 'g-', lw=2, label='$x_{TP}$')
    ax.plot(t_vec, Xhist_freq[3, :], 'r-', lw=2, label='$x_{T-}$')
    ax.set_xlabel('Time (days)', fontsize=FSIZE)
    ax.set_ylabel('Frequency', fontsize=FSIZE)
    ax.set_title('Population Frequencies (Replicator)', fontsize=FSIZE)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=FSIZE)
    ax.grid(True)
    ax.set_ylim(-0.05, 1.05)

    # [1,0] Treatment Schedule
    ax = axes[1, 0]
    ax.plot(t_vec, Uhist, 'b.', ms=3)
    ax.set_xlabel('Time (days)', fontsize=FSIZE)
    ax.set_ylabel('Abiraterone dose', fontsize=FSIZE)
    ax.set_title('Treatment Schedule', fontsize=FSIZE)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=FSIZE)
    ax.grid(True)

    # [1,1] Tumor Burden
    ax = axes[1, 1]
    ax.plot(t_vec, N_total_mpc, 'purple', lw=2, label='$N_{tumor}$ (plant)')
    N_from_zf = N_max * (1 - Xhist_freq[0, :])
    ax.plot(t_vec, N_from_zf, 'purple', lw=1.5, ls='--', label='$N_{max}(1-x_0)$')
    ax.set_xlabel('Time (days)', fontsize=FSIZE)
    ax.set_ylabel('Total population', fontsize=FSIZE)
    ax.set_title('Tumor Burden', fontsize=FSIZE)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=FSIZE)
    ax.grid(True)

    plt.tight_layout()
    plot_name = f"SA_{sim_name}_Hp{hp}_Overview.png"
    plot_path = os.path.join(img_dir, plot_name)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Overview plot saved: {plot_path}")


def generate_comparison_plots(all_results, img_dir):
    """Generate comparison plots for full factorial SA sensitivity analysis.
    2x2 layout:
      [0,0] KPI-1 (T- dominance day) vs Hp for each gamma config
      [0,1] KPI-3 (TP > 250% x0) vs Hp for each gamma config
      [1,0] Drug exposure vs Hp for each gamma config
      [1,1] Computation time vs Hp for each gamma config
    """
    valid = [r for r in all_results if 'error' not in r]
    if not valid:
        print("No valid results for comparison plots.")
        return

    NOT_REACHED = Hs + 500

    # Organize results by gamma config name
    sim_names = [gc['name'] for gc in GAMMA_CONFIGS]
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    markers = ['o', 's', '^', 'D', 'v']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Sensitivity Analysis SA: Full Factorial (Table 3)', fontsize=16)

    for si, sname in enumerate(sim_names):
        # Get all results for this gamma config, sorted by Hp
        sim_results = sorted(
            [r for r in valid if r['config']['name'] == sname],
            key=lambda r: r['config']['Hp']
        )
        if not sim_results:
            continue

        hps = [r['config']['Hp'] for r in sim_results]
        kpi1s = [r['kpi1'] if r['kpi1'] >= 0 else NOT_REACHED for r in sim_results]
        kpi3s = [r['kpi3'] if r['kpi3'] >= 0 else NOT_REACHED for r in sim_results]
        drugs = [r['drug_exposure'] for r in sim_results]
        times = [r['t_elapsed'] for r in sim_results]

        label = sname.replace('_', ' ')
        c = colors[si % len(colors)]
        m = markers[si % len(markers)]

        # [0,0] KPI-1 vs Hp
        ax = axes[0, 0]
        ax.plot(hps, kpi1s, f'-{m}', color=c, lw=1.5, ms=7, label=label)

        # [0,1] KPI-3 vs Hp
        ax = axes[0, 1]
        ax.plot(hps, kpi3s, f'-{m}', color=c, lw=1.5, ms=7, label=label)

        # [1,0] Drug exposure vs Hp
        ax = axes[1, 0]
        ax.plot(hps, drugs, f'-{m}', color=c, lw=1.5, ms=7, label=label)

        # [1,1] Computation time vs Hp
        ax = axes[1, 1]
        ax.plot(hps, times, f'-{m}', color=c, lw=1.5, ms=7, label=label)

    # Format axes
    for ax in axes.flat:
        ax.set_xlabel('$H_p$', fontsize=FSIZE)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=FSIZE - 1)
        ax.legend(fontsize=9)
        ax.set_xticks(HP_VALUES)

    axes[0, 0].set_ylabel('Day', fontsize=FSIZE)
    axes[0, 0].set_title('KPI-1: T- Dominance Day', fontsize=FSIZE)
    axes[0, 0].axhline(y=NOT_REACHED, color='gray', ls=':', lw=1, alpha=0.5)
    axes[0, 0].text(HP_VALUES[0], NOT_REACHED + 50, 'Not Reached', fontsize=9, color='gray')

    axes[0, 1].set_ylabel('Day', fontsize=FSIZE)
    axes[0, 1].set_title('KPI-3: TP > 250% $x_0$ Day', fontsize=FSIZE)
    axes[0, 1].axhline(y=NOT_REACHED, color='gray', ls=':', lw=1, alpha=0.5)
    axes[0, 1].text(HP_VALUES[0], NOT_REACHED + 50, 'Not Reached', fontsize=9, color='gray')

    axes[1, 0].set_ylabel('Mean drug dose', fontsize=FSIZE)
    axes[1, 0].set_title('Drug Exposure', fontsize=FSIZE)

    axes[1, 1].set_ylabel('Time (s)', fontsize=FSIZE)
    axes[1, 1].set_title('Computation Time', fontsize=FSIZE)

    plt.tight_layout()
    plot_path = os.path.join(img_dir, "Sensitivity_SA_Comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved: {plot_path}")


# ================================================================
# MAIN EXECUTION
# ================================================================
if __name__ == "__main__":

    notifier = MPCNotifier()
    all_results = []

    try:
        os.makedirs(IMG_DIR, exist_ok=True)
        os.makedirs(DATA_DIR, exist_ok=True)

        # ---- Filter by --sims if specified ----
        gamma_configs = GAMMA_CONFIGS
        if SELECTED_SIMS is not None:
            gamma_configs = [gc for i, gc in enumerate(GAMMA_CONFIGS) if (i + 1) in SELECTED_SIMS]
            print(f"  PARALLEL MODE: Running sims {SELECTED_SIMS} only")

        # ---- Build full factorial configuration list ----
        configs = []
        for gc in gamma_configs:
            for hp in HP_VALUES:
                w_s = gc['gamma_1'] * N_max**2
                w_d = gc['gamma_2']
                w_r = gc['gamma_3'] * N_max
                fname = f"Results_SA_{gc['name']}_Hp{hp}.npz"
                configs.append({
                    'Hp': hp,
                    'w_sens': w_s, 'w_drug': w_d, 'w_res': w_r,
                    'gamma_1': gc['gamma_1'], 'gamma_2': gc['gamma_2'],
                    'gamma_3': gc['gamma_3'],
                    'name': gc['name'], 'filename': fname,
                    'label': f"{gc['name']} Hp={hp}",
                })

        n_total_configs = len(configs)
        print("=" * 60)
        print("  SENSITIVITY ANALYSIS: Single-Agent (SA) Replicator-MPC")
        print("  Full Factorial -- Table 3 of BiologicalModels_EA_250930")
        print("=" * 60)
        print(f"  Output: {OUTPUT_DIR}/")
        print(f"  Total configs: {n_total_configs} ({len(gamma_configs)} sims x {len(HP_VALUES)} Hp)")
        print(f"  Skip existing: {SKIP_EXISTING}")
        print(f"  N_max={N_max:.0e}, dt={dt}, Hs={Hs}")
        print(f"  u_init: 0.0 (V3 fix for Hp>=80, confirmed by Test_SA_InitGuess)")
        print(f"  Integration: RK4, max_iter: 1000")
        print(f"  Hp values: {HP_VALUES}")
        print(f"  Gamma configs:")
        for gc in gamma_configs:
            print(f"    {gc['name']}: gamma=({gc['gamma_1']:.1e}, {gc['gamma_2']}, {gc['gamma_3']:.1e})")
        print()

        notifier.notify_start("Sensitivity Analysis SA V3 (Table 3, u_init=0.0, RK4)", {
            "Total configs": f"{n_total_configs} ({len(gamma_configs)} sims x {len(HP_VALUES)} Hp)",
            "Hp values": str(HP_VALUES),
            "Gamma configs": str(len(gamma_configs)),
            "Hs": Hs, "dt": dt, "N_max": f"{N_max:.0e}",
        })

        t_total_start = time.time()
        n_completed = 0
        n_skipped = 0
        n_errors = 0

        for ci, cfg in enumerate(configs):
            filepath = os.path.join(DATA_DIR, cfg['filename'])

            # Skip if exists
            if SKIP_EXISTING and os.path.exists(filepath):
                print(f"\n[{ci+1}/{n_total_configs}] SKIP (exists): {cfg['filename']}")
                try:
                    data = np.load(filepath, allow_pickle=True)
                    all_results.append({
                        'config': cfg,
                        'kpi1': int(data['kpi1']),
                        'kpi2': int(data['kpi2']),
                        'kpi3': int(data['kpi3']),
                        'drug_exposure': float(data['drug_exposure']),
                        't_elapsed': float(data['t_elapsed']),
                        'skipped': True,
                    })
                except Exception:
                    all_results.append({'config': cfg, 'error': 'load_failed', 'skipped': True})
                n_skipped += 1
                continue

            print(f"\n{'='*60}")
            print(f"  [{ci+1}/{n_total_configs}] {cfg['label']}")
            print(f"  Hp={cfg['Hp']}, gamma=({cfg['gamma_1']:.1e}, {cfg['gamma_2']}, {cfg['gamma_3']:.1e})")
            print(f"  w_sens={cfg['w_sens']:.1f}, w_drug={cfg['w_drug']:.2f}, w_res={cfg['w_res']:.1f}")
            print(f"{'='*60}")

            try:
                # Notify config start
                notifier.notify_progress(ci, n_total_configs, time.time() - t_total_start,
                    extra_info=(
                        f"Starting: {cfg['label']}\n"
                        f"Hp={cfg['Hp']}\n"
                        f"gamma=({cfg['gamma_1']:.1e}, {cfg['gamma_2']}, {cfg['gamma_3']:.1e})"
                    ))

                # Build solver
                t_build = time.time()
                solver, n_g = build_sa_solver(cfg['Hp'], cfg['w_sens'], cfg['w_drug'], cfg['w_res'])
                print(f"  Solver built in {time.time()-t_build:.1f}s")

                # Run simulation
                res = run_sa_simulation(
                    solver, cfg['Hp'], n_g, label=cfg['label'],
                    notifier=notifier, config_idx=ci+1, total_configs=n_total_configs
                )

                # Print KPIs
                print(f"  KPI-1 (T- dom):    {'day '+str(res['kpi1']) if res['kpi1']>=0 else 'NOT reached'}")
                print(f"  KPI-2 (T+>250%):   {'day '+str(res['kpi2']) if res['kpi2']>=0 else 'NOT reached'}")
                print(f"  KPI-3 (TP>250%):   {'day '+str(res['kpi3']) if res['kpi3']>=0 else 'NOT reached'}")
                print(f"  Drug exposure:      {res['drug_exposure']:.4f}")
                print(f"  Computation time:   {res['t_elapsed']:.1f}s")

                # Save .npz
                t_vec = np.arange(0, Hs + dt, dt)
                np.savez(filepath,
                    Xhist_abs=res['Xhist_abs'], Xhist_freq=res['Xhist_freq'],
                    Uhist=res['Uhist'], t_vec=t_vec, y0_abs=y0_abs,
                    Hp=cfg['Hp'], Hs=Hs, dt=dt, N_max=N_max,
                    t_elapsed=res['t_elapsed'],
                    w_sens=cfg['w_sens'], w_drug=cfg['w_drug'], w_res=cfg['w_res'],
                    gamma_1=cfg['gamma_1'], gamma_2=cfg['gamma_2'], gamma_3=cfg['gamma_3'],
                    kpi1=res['kpi1'], kpi2=res['kpi2'], kpi3=res['kpi3'],
                    drug_exposure=res['drug_exposure'],
                    name=cfg['name'],
                )
                print(f"  Saved: {filepath}")

                # Generate per-config 2x2 overview plot
                generate_config_plot(res, cfg, IMG_DIR, t_vec)

                all_results.append({
                    'config': cfg,
                    'kpi1': res['kpi1'], 'kpi2': res['kpi2'], 'kpi3': res['kpi3'],
                    'drug_exposure': res['drug_exposure'], 't_elapsed': res['t_elapsed'],
                    'skipped': False,
                })
                n_completed += 1

                # Notify config completion
                notifier.notify_progress(ci + 1, n_total_configs, time.time() - t_total_start,
                    extra_info=(
                        f"Completed: {cfg['label']}\n"
                        f"KPI-1: {'day '+str(res['kpi1']) if res['kpi1']>=0 else 'NR'}\n"
                        f"KPI-3: {'day '+str(res['kpi3']) if res['kpi3']>=0 else 'NR'}\n"
                        f"Drug: {res['drug_exposure']:.4f}\n"
                        f"Time: {res['t_elapsed']:.1f}s"
                    ))

            except Exception as e:
                tb = traceback.format_exc()
                print(f"  ERROR in {cfg['label']}: {e}")
                print(tb)
                all_results.append({'config': cfg, 'error': str(e), 'skipped': False})
                n_errors += 1
                notifier.notify_error("Config Error", f"{cfg['label']}: {e}", tb)

        t_total = time.time() - t_total_start

        # ---- Summary CSV (matching Table 3 format) ----
        csv_path = os.path.join(DATA_DIR, "Summary_SA.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Sim', 'Name', 'Hp', 'gamma_1', 'gamma_2', 'gamma_3',
                'w_sens', 'w_drug', 'w_res',
                'KPI-1', 'KPI-2', 'KPI-3', 'Drug_exposure', 'Time_s', 'Status'
            ])
            for r in all_results:
                c = r['config']
                # Extract sim number from name
                sim_num = c['name'].split('_')[0]  # e.g., "Sim1"
                if 'error' in r:
                    writer.writerow([
                        sim_num, c['name'], c['Hp'],
                        c['gamma_1'], c['gamma_2'], c['gamma_3'],
                        '', '', '',
                        'ERROR', '', '', '', '', r['error']
                    ])
                else:
                    kpi1_str = str(r['kpi1']) if r['kpi1'] >= 0 else 'NR'
                    kpi2_str = str(r['kpi2']) if r['kpi2'] >= 0 else 'NR'
                    kpi3_str = str(r['kpi3']) if r['kpi3'] >= 0 else 'NR'
                    status = 'skipped' if r.get('skipped') else 'ok'
                    writer.writerow([
                        sim_num, c['name'], c['Hp'],
                        c['gamma_1'], c['gamma_2'], c['gamma_3'],
                        f"{c['w_sens']:.2f}", f"{c['w_drug']:.4f}", f"{c['w_res']:.2f}",
                        kpi1_str, kpi2_str, kpi3_str,
                        f"{r['drug_exposure']:.4f}", f"{r['t_elapsed']:.1f}", status
                    ])
        print(f"\nSummary CSV saved: {csv_path}")

        # ---- Comparison Plots ----
        valid_results = [r for r in all_results if 'error' not in r]
        if valid_results:
            generate_comparison_plots(all_results, IMG_DIR)

        # ---- Final Summary ----
        print(f"\n{'='*60}")
        print(f"  SENSITIVITY ANALYSIS SA COMPLETE (Table 3)")
        print(f"{'='*60}")
        print(f"  Completed: {n_completed} | Skipped: {n_skipped} | Errors: {n_errors}")
        print(f"  Total time: {t_total:.1f}s ({t_total/60:.1f} min)")
        print(f"  Output: {OUTPUT_DIR}/")
        print(f"  Images: {IMG_DIR}/")
        print(f"  Data:   {DATA_DIR}/")

        notifier.notify_complete("Sensitivity Analysis SA (Table 3)", t_total, {
            "Completed": n_completed,
            "Skipped": n_skipped,
            "Errors": n_errors,
        }, {
            "Total configs": f"{n_total_configs} ({len(GAMMA_CONFIGS)} sims x {len(HP_VALUES)} Hp)",
            "Output": OUTPUT_DIR,
        })

    except Exception as e:
        tb = traceback.format_exc()
        print(f"\nFATAL ERROR: {e}\n{tb}")
        notifier.notify_error("Fatal", str(e), tb)
        sys.exit(1)
