"""
Sensitivity_DA.py - Sensitivity Analysis for Dual-Agent Replicator-MPC

REPRODUCES IN THE THESIS
    Base dual-agent sensitivity sweep.
    [TO CONFIRM: mapping to the thesis table has not been verified by the author.]

PROVENANCE
    Copied verbatim from the working tree of the thesis
    (Modelos_Replicator/Codigos/Sensitivity_DA.py). The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md
    at the repository root for the complete list of differences.

--------------------------------------------------------------------------
Sensitivity_DA.py - Sensitivity Analysis for Dual-Agent Replicator-MPC
======================================================================
Performs systematic parameter sweeps for DA (West, 4 pop, Abi+Doc):
  Phase 1: Hp sweep [20, 50, 80, 100, 140] with Bw=80, baseline weights
  Phase 2: Bw sweep [20, 50, 80, 100] with Hp=140, baseline weights
  Phase 3: Weight sweep at Hp=140, Bw=80 with 5 configurations

Output (in Sensitivity_DA/ folder):
  Results_DA_Hp{}_Bw{}_G{N}_{name}.npz  -- per configuration
  Summary_DA.csv                          -- summary table
  Sensitivity_DA_Comparison.png           -- comparison plots

Note: Bw=1 is extremely expensive (~12 hours). Set INCLUDE_BW1=True to enable.

Based on: MPC_Replicator_DA_V3.py
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
# CONFIGURATION
# ================================================================

OUTPUT_DIR = "Sensitivity_DA"
IMG_DIR = os.path.join(OUTPUT_DIR, "Imagenes")
DATA_DIR = os.path.join(OUTPUT_DIR, "Datos")
SKIP_EXISTING = True
INCLUDE_BW1 = False  # Bw=1 adds ~12 hours; enable only if server allows

# Fixed parameters (same for all configs)
N_max = 8e4
dt = 1
Hs = 30000
B_rate = [0.05, 0.2]
y0_abs = np.array([2062.0, 2617.0, 3.97e-11, 3.97e-12])

# Active-set reference
Xs_abs = np.array([3000.0, 3000.0, 1800.0])
n_Eps = len(Xs_abs)

# Internal weight matrices (fixed across all configs)
Q_diag = cs.DM([1, 1, 0.5, 0.1])
R_diag = cs.DM([0.3, 1])

# --- Phase 1: Hp sweep ---
HP_VALUES = [20, 50, 80, 100, 140]
BASELINE_BW = 80
BASELINE_WEIGHTS = {'w_drug': 0.3, 'w_slack': 0.45, 'w_state': 0.25}

# --- Phase 2: Bw sweep ---
BW_HP = 140
BW_VALUES = [20, 50, 80, 100]
if INCLUDE_BW1:
    BW_VALUES = [1] + BW_VALUES

# --- Phase 3: Weight/Gamma sweep ---
WEIGHT_HP = 140
WEIGHT_BW = 80
WEIGHT_CONFIGS = [
    {'name': 'Baseline',       'w_drug': 0.30, 'w_slack': 0.45, 'w_state': 0.25},
    {'name': 'Drug_economy',   'w_drug': 0.60, 'w_slack': 0.25, 'w_state': 0.15},
    {'name': 'Tumor_ctrl',     'w_drug': 0.15, 'w_slack': 0.35, 'w_state': 0.50},
    {'name': 'Slack_priority', 'w_drug': 0.20, 'w_slack': 0.60, 'w_state': 0.20},
    {'name': 'Balanced',       'w_drug': 0.33, 'w_slack': 0.34, 'w_state': 0.33},
]


# ================================================================
# MODEL DEFINITIONS (from MPC_Replicator_DA_V3.py)
# ================================================================

r_lv = cs.DM([2.78e-3, 3.55e-3, 6.65e-3, 6.65e-3])
delta_comp = 0.05
A_lv = cs.DM([[1,                0.7,              0.8,              0.8 - 2*delta_comp],
               [0.4,              1,                0.5,              0.5 - 2*delta_comp],
               [0.6,              0.9,              1,                1   - 2*delta_comp],
               [0.6 - delta_comp, 0.9 - delta_comp, 1 - delta_comp,  1]])
mu_param = 1.5
d_v = 5e-4

# --- LV Plant model ---
y_s = cs.SX.sym('y', 4)
d_s = cs.SX.sym('d', 2)
K_lv = cs.vertcat(
    (mu_param - d_s[0]) * y_s[1],
    1e4 / (1 * (10**(2 * d_s[0]))),
    1e4,
    6e3
)
rhs_lv = cs.vertcat(
    (1 - cs.dot(A_lv[0,:], y_s.T)/K_lv[0]) * float(r_lv[0]) * y_s[0] * (1 - 1.5*d_s[1]) - d_v*y_s[0],
    (1 - cs.dot(A_lv[1,:], y_s.T)/K_lv[1]) * float(r_lv[1]) * y_s[1] * (1 - 1.5*d_s[1]) - d_v*y_s[1],
    (1 - cs.dot(A_lv[2,:], y_s.T)/K_lv[2]) * float(r_lv[2]) * y_s[2] * (1 - 1.5*d_s[1]) - d_v*y_s[2],
    (1 - cs.dot(A_lv[3,:], y_s.T)/K_lv[3]) * float(r_lv[3]) * y_s[3]                     - d_v*y_s[3]
)
F_LV = cs.Function('F_LV', [y_s, d_s], [rhs_lv])

# --- Replicator + Zona Ficticia ---
n_pop = 4
n_total = n_pop + 1
n_controls = 2

p_s = cs.SX.sym('p', n_total)
u_s = cs.SX.sym('u', 2)
y_from_p = N_max * p_s[1:]
K_rep = cs.vertcat(
    (mu_param - u_s[0]) * cs.fmax(y_from_p[1], 1e-10),
    1e4 / (1 * (10**(2 * u_s[0]))),
    1e4,
    6e3
)
f_real = cs.SX.zeros(n_pop)
for i in range(n_pop):
    growth = float(r_lv[i]) * (1 - cs.dot(A_lv[i,:], y_from_p.T) / K_rep[i])
    if i < 3:
        growth = growth * (1 - 1.5 * u_s[1])
    f_real[i] = growth - d_v
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

def build_da_solver(Hp, w_drug, w_slack, w_state):
    """Build CasADi NLP solver for DA with given Hp and weights."""
    U = cs.SX.sym('U', n_controls, Hp)
    E = cs.SX.sym('E', n_Eps, Hp)
    X = cs.SX.sym('X', n_total, Hp + 1)

    n_par = n_total + n_controls  # state (5) + u_prev (2)
    P = cs.SX.sym('P', n_par)

    obj = 0
    X[:, 0] = P[:n_total]

    Q_mat = cs.diagcat(float(Q_diag[0]), float(Q_diag[1]), float(Q_diag[2]), float(Q_diag[3]))
    R_mat = cs.diagcat(float(R_diag[0]), float(R_diag[1]))

    for t in range(Hp):
        st = X[:, t]
        con = U[:, t]
        f_val = F_REP(st, con)
        X[:, t+1] = st + dt * f_val

        p_real = X[1:, t+1]
        obj += w_drug * (con.T @ R_mat @ con)
        obj += w_slack * (E[:, t].T @ cs.DM.eye(n_Eps) @ E[:, t])
        obj += w_state * (p_real.T @ Q_mat @ p_real)

    Xs_freq = Xs_abs / N_max

    g = []
    for t in range(1, Hp + 1):
        # State bounds [0, 1]
        for i in range(n_total):
            g = cs.vertcat(g, -X[i, t])
            g = cs.vertcat(g, X[i, t] - 1.0)

        # Control bounds [0, 1]
        g = cs.vertcat(g, -U[:, t-1])
        g = cs.vertcat(g, U[:, t-1] - cs.DM([1, 1]))

        # Rate constraints
        if t != 1:
            g = cs.vertcat(g, -(U[:, t-1] - U[0, t-2]) - cs.DM(B_rate))
            g = cs.vertcat(g, (U[:, t-1] - U[0, t-2]) - cs.DM(B_rate))
        else:
            g = cs.vertcat(g, -(P[n_total:] - U[:, t-1]) - cs.DM(B_rate))
            g = cs.vertcat(g, (P[n_total:] - U[:, t-1]) - cs.DM(B_rate))

        # Slack constraints (active-set)
        g = cs.vertcat(g, -E[:, t-1])
        g = cs.vertcat(g, X[1:n_Eps+1, t] - cs.DM(Xs_freq) - E[:, t-1])

    OPT_vars = cs.vertcat(
        cs.reshape(U, n_controls * Hp, 1),
        cs.reshape(E, n_Eps * Hp, 1)
    )

    nlp = {'f': obj, 'x': OPT_vars, 'g': g, 'p': P}
    opts = {
        'ipopt': {'max_iter': 200, 'print_level': 0,
                  'acceptable_tol': 1e-8, 'acceptable_obj_change_tol': 1e-6},
        'print_time': 0
    }
    solver = cs.nlpsol('solver', 'ipopt', nlp, opts)
    return solver, g.shape[0]


def run_da_simulation(solver, Hp, Block_V, n_g, label="",
                      notifier=None, config_idx=0, total_configs=1):
    """Run DA MPC simulation with blocking. Returns results dict."""
    n_steps = int(Hs / dt)

    x0 = cs.DM(y0_abs)
    u_prev = np.array([0.6, 0.0])
    U0 = np.tile(u_prev.reshape(n_controls, 1), (1, Hp))
    E0 = 0.5 * np.ones((n_Eps, Hp))

    Xhist_abs = np.zeros((4, n_steps + 1))
    Xhist_freq = np.zeros((n_total, n_steps + 1))
    Uhist = np.zeros((n_controls, n_steps + 1))

    Xhist_abs[:, 0] = y0_abs
    Xhist_freq[:, 0] = abs_to_freq(y0_abs)
    Uhist[:, 0] = u_prev

    lbg = [-float('inf')] * n_g
    ubg = [0] * n_g

    t0 = time.time()
    UT = None

    # Telegram: notify every 10% for long runs
    notify_pct_interval = 10
    next_notify_pct = notify_pct_interval

    for sim in tqdm(range(n_steps), desc=label, leave=True):
        if sim % Block_V == 0:
            if sim > 0:
                U0 = UT
            p_curr = abs_to_freq(x0)
            Par = cs.DM(np.concatenate([p_curr, U0[:, 0]]))

            x0_nlp = cs.vertcat(
                cs.DM(U0.reshape(n_controls * Hp, 1)),
                cs.DM(E0.reshape(n_Eps * Hp, 1))
            )
            sol = solver(lbg=lbg, ubg=ubg, p=Par, x0=x0_nlp)

            sol_x = sol['x'].full().flatten()
            U0 = sol_x[:n_controls * Hp].reshape(n_controls, Hp)
            E0 = sol_x[n_controls * Hp:].reshape(n_Eps, Hp)

        u_apply = U0[:, 0]
        x0 = x0 + dt * F_LV(x0, cs.DM(u_apply))

        Xhist_abs[:, sim + 1] = np.array(x0).flatten()
        Xhist_freq[:, sim + 1] = abs_to_freq(x0)
        Uhist[:, sim + 1] = u_apply

        if sim % Block_V == 0:
            UT = np.hstack([U0[:, 1:], U0[:, -1:]])
            E0 = np.hstack([E0[:, 1:], E0[:, -1:]])

        # Telegram progress
        pct = 100.0 * (sim + 1) / n_steps
        if notifier and pct >= next_notify_pct:
            elapsed = time.time() - t0
            y_now = np.array(x0).flatten()
            N_now = np.sum(y_now)
            notifier.notify_progress(sim + 1, n_steps, elapsed,
                extra_info=(
                    f"Config [{config_idx}/{total_configs}]: {label}\n"
                    f"u_a={u_apply[0]:.3f}, u_d={u_apply[1]:.3f}\n"
                    f"N_tumor={N_now:.0f}\n"
                    f"Day {(sim+1)*dt}/{Hs}"
                ))
            next_notify_pct += notify_pct_interval

    t_elapsed = time.time() - t0

    # Compute KPIs
    N_tot = np.sum(Xhist_abs, axis=0)
    frac_res1 = Xhist_abs[2, :] / N_tot
    frac_res2 = Xhist_abs[3, :] / N_tot
    frac_sens = np.sum(Xhist_abs[:2, :], axis=0) / N_tot

    kpi1a_idx = np.where(frac_res1 > frac_sens)[0]
    kpi1a = int(kpi1a_idx[0]) if len(kpi1a_idx) > 0 else -1

    kpi1b_idx = np.where(frac_res2 > frac_sens)[0]
    kpi1b = int(kpi1b_idx[0]) if len(kpi1b_idx) > 0 else -1

    kpi2_idx = np.where(Xhist_abs[0, :] > y0_abs[0] * 2.5)[0]
    kpi2 = int(kpi2_idx[0]) if len(kpi2_idx) > 0 else -1

    kpi3_idx = np.where(Xhist_abs[1, :] > y0_abs[1] * 2.5)[0]
    kpi3 = int(kpi3_idx[0]) if len(kpi3_idx) > 0 else -1

    drug_abi = float(np.mean(Uhist[0, :]))
    drug_doc = float(np.mean(Uhist[1, :]))

    return {
        'Xhist_abs': Xhist_abs, 'Xhist_freq': Xhist_freq,
        'Uhist': Uhist, 't_elapsed': t_elapsed,
        'kpi1a': kpi1a, 'kpi1b': kpi1b, 'kpi2': kpi2, 'kpi3': kpi3,
        'drug_abi': drug_abi, 'drug_doc': drug_doc,
    }


def generate_comparison_plots(all_results, img_dir):
    """Generate comparison plots for DA sensitivity analysis."""
    FSIZE = 14
    NOT_REACHED = Hs + 1000

    ph1 = [r for r in all_results if r['config']['phase'] == 1 and 'error' not in r]
    ph2 = [r for r in all_results if r['config']['phase'] == 2 and 'error' not in r]
    ph3 = [r for r in all_results if r['config']['phase'] == 3 and 'error' not in r]

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Sensitivity Analysis DA: Replicator-MPC (West, Dual-Agent)', fontsize=16)

    def plot_phase(ax_row, results, x_labels, x_label_name, phase_title):
        if not results:
            for ax in ax_row:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            return

        kpi1a = [r['kpi1a'] if r['kpi1a'] >= 0 else NOT_REACHED for r in results]
        kpi1b = [r['kpi1b'] if r['kpi1b'] >= 0 else NOT_REACHED for r in results]
        drugs_a = [r['drug_abi'] for r in results]
        drugs_d = [r['drug_doc'] for r in results]
        times = [r['t_elapsed'] for r in results]

        x = np.arange(len(x_labels))
        w = 0.35

        # KPIs
        ax = ax_row[0]
        b1 = ax.bar(x - w/2, kpi1a, w, label='KPI-1a (T-/+)', color='red', alpha=0.7)
        b2 = ax.bar(x + w/2, kpi1b, w, label='KPI-1b (T-/-)', color='cyan', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30 if len(x_labels[0]) > 4 else 0,
                           ha='right' if len(x_labels[0]) > 4 else 'center', fontsize=9)
        ax.set_xlabel(x_label_name, fontsize=FSIZE)
        ax.set_ylabel('Day', fontsize=FSIZE)
        ax.set_title(f'KPIs - {phase_title}', fontsize=FSIZE)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        for bars, vals in [(b1, kpi1a), (b2, kpi1b)]:
            for bar, val in zip(bars, vals):
                if val == NOT_REACHED:
                    bar.set_hatch('//')
                    bar.set_edgecolor('black')

        # Drug exposure (grouped: Abi + Doc)
        ax = ax_row[1]
        ax.bar(x - w/2, drugs_a, w, label='Abiraterone', color='blue', alpha=0.7)
        ax.bar(x + w/2, drugs_d, w, label='Docetaxel', color='green', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30 if len(x_labels[0]) > 4 else 0,
                           ha='right' if len(x_labels[0]) > 4 else 'center', fontsize=9)
        ax.set_xlabel(x_label_name, fontsize=FSIZE)
        ax.set_ylabel('Mean dose', fontsize=FSIZE)
        ax.set_title(f'Drug Exposure - {phase_title}', fontsize=FSIZE)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Computation time
        ax = ax_row[2]
        ax.bar(x, times, 0.5, color='orange', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30 if len(x_labels[0]) > 4 else 0,
                           ha='right' if len(x_labels[0]) > 4 else 'center', fontsize=9)
        ax.set_xlabel(x_label_name, fontsize=FSIZE)
        ax.set_ylabel('Time (s)', fontsize=FSIZE)
        ax.set_title(f'Time - {phase_title}', fontsize=FSIZE)
        ax.grid(True, alpha=0.3)

    # Row 0: Phase 1 - Hp sweep
    plot_phase(axes[0], ph1,
               [str(r['config']['Hp']) for r in ph1], 'Hp', f'Hp sweep (Bw={BASELINE_BW})')

    # Row 1: Phase 2 - Bw sweep
    plot_phase(axes[1], ph2,
               [str(r['config']['Block_V']) for r in ph2], 'Bw', f'Bw sweep (Hp={BW_HP})')

    # Row 2: Phase 3 - Weight sweep
    plot_phase(axes[2], ph3,
               [r['config']['tag'].split('_', 1)[1] for r in ph3], 'Config',
               f'Weight sweep (Hp={WEIGHT_HP}, Bw={WEIGHT_BW})')

    plt.tight_layout()
    plot_path = os.path.join(img_dir, "Sensitivity_DA_Comparison.png")
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

        # ---- Build full configuration list ----
        configs = []

        # Phase 1: Hp sweep (Bw=80, baseline weights)
        for hp in HP_VALUES:
            w = BASELINE_WEIGHTS
            tag = "G1_Baseline"
            fname = f"Results_DA_Hp{hp}_Bw{BASELINE_BW}_{tag}.npz"
            configs.append({
                'phase': 1, 'Hp': hp, 'Block_V': BASELINE_BW,
                'w_drug': w['w_drug'], 'w_slack': w['w_slack'], 'w_state': w['w_state'],
                'tag': tag, 'filename': fname,
                'label': f"Ph1 Hp={hp} Bw={BASELINE_BW}",
            })

        # Phase 2: Bw sweep (Hp=140, baseline weights)
        for bw in BW_VALUES:
            w = BASELINE_WEIGHTS
            tag = "G1_Baseline"
            fname = f"Results_DA_Hp{BW_HP}_Bw{bw}_{tag}.npz"
            configs.append({
                'phase': 2, 'Hp': BW_HP, 'Block_V': bw,
                'w_drug': w['w_drug'], 'w_slack': w['w_slack'], 'w_state': w['w_state'],
                'tag': tag, 'filename': fname,
                'label': f"Ph2 Hp={BW_HP} Bw={bw}",
            })

        # Phase 3: Weight/Gamma sweep (Hp=140, Bw=80)
        for gi, wc in enumerate(WEIGHT_CONFIGS, 1):
            tag = f"G{gi}_{wc['name']}"
            fname = f"Results_DA_Hp{WEIGHT_HP}_Bw{WEIGHT_BW}_{tag}.npz"
            configs.append({
                'phase': 3, 'Hp': WEIGHT_HP, 'Block_V': WEIGHT_BW,
                'w_drug': wc['w_drug'], 'w_slack': wc['w_slack'], 'w_state': wc['w_state'],
                'tag': tag, 'filename': fname,
                'label': f"Ph3 {wc['name']}",
            })

        n_total_configs = len(configs)

        # Count unique configs (some overlap between phases)
        unique_files = set(c['filename'] for c in configs)
        n_unique = len(unique_files)

        print("=" * 60)
        print("  SENSITIVITY ANALYSIS: Dual-Agent (DA) Replicator-MPC")
        print("=" * 60)
        print(f"  Output: {OUTPUT_DIR}/")
        print(f"  Total configs: {n_total_configs} ({n_unique} unique, {n_total_configs - n_unique} overlap)")
        print(f"  Skip existing: {SKIP_EXISTING}")
        print(f"  N_max={N_max:.0e}, dt={dt}, Hs={Hs}")
        print(f"  Phase 1: Hp sweep {HP_VALUES} (Bw={BASELINE_BW})")
        print(f"  Phase 2: Bw sweep {BW_VALUES} (Hp={BW_HP})")
        print(f"  Phase 3: Weight sweep ({len(WEIGHT_CONFIGS)} configs, Hp={WEIGHT_HP}, Bw={WEIGHT_BW})")
        if INCLUDE_BW1:
            print(f"  WARNING: Bw=1 included -- expect ~12 hours for that config!")
        print()

        notifier.notify_start("Sensitivity Analysis DA", {
            "Total configs": f"{n_total_configs} ({n_unique} unique)",
            "Phase 1": f"Hp sweep {HP_VALUES}, Bw={BASELINE_BW}",
            "Phase 2": f"Bw sweep {BW_VALUES}, Hp={BW_HP}",
            "Phase 3": f"Weight sweep ({len(WEIGHT_CONFIGS)} configs)",
            "Hs": Hs, "dt": dt, "N_max": f"{N_max:.0e}",
            "Include Bw=1": INCLUDE_BW1,
        })

        t_total_start = time.time()
        n_completed = 0
        n_skipped = 0
        n_errors = 0

        for ci, cfg in enumerate(configs):
            filepath = os.path.join(DATA_DIR, cfg['filename'])

            # Skip if exists (also handles phase overlap)
            if SKIP_EXISTING and os.path.exists(filepath):
                print(f"\n[{ci+1}/{n_total_configs}] SKIP (exists): {cfg['filename']}")
                try:
                    data = np.load(filepath, allow_pickle=True)
                    all_results.append({
                        'config': cfg,
                        'kpi1a': int(data['kpi1a']),
                        'kpi1b': int(data['kpi1b']),
                        'kpi2': int(data['kpi2']),
                        'kpi3': int(data['kpi3']),
                        'drug_abi': float(data['drug_abi']),
                        'drug_doc': float(data['drug_doc']),
                        't_elapsed': float(data['t_elapsed']),
                        'skipped': True,
                    })
                except Exception:
                    all_results.append({'config': cfg, 'error': 'load_failed', 'skipped': True})
                n_skipped += 1
                continue

            print(f"\n{'='*60}")
            print(f"  [{ci+1}/{n_total_configs}] {cfg['label']}")
            print(f"  Hp={cfg['Hp']}, Bw={cfg['Block_V']}")
            print(f"  w_drug={cfg['w_drug']:.2f}, w_slack={cfg['w_slack']:.2f}, w_state={cfg['w_state']:.2f}")
            n_nlp_solves = int(Hs / dt / cfg['Block_V'])
            print(f"  NLP solves: {n_nlp_solves}")
            print(f"{'='*60}")

            try:
                # Notify config start
                notifier.notify_progress(ci, n_total_configs, time.time() - t_total_start,
                    extra_info=(
                        f"Starting: {cfg['label']}\n"
                        f"Hp={cfg['Hp']}, Bw={cfg['Block_V']}\n"
                        f"w=({cfg['w_drug']:.2f}, {cfg['w_slack']:.2f}, {cfg['w_state']:.2f})\n"
                        f"NLP solves: {n_nlp_solves}"
                    ))

                # Build solver
                t_build = time.time()
                solver, n_g = build_da_solver(
                    cfg['Hp'], cfg['w_drug'], cfg['w_slack'], cfg['w_state']
                )
                print(f"  Solver built in {time.time()-t_build:.1f}s")

                # Run simulation
                res = run_da_simulation(
                    solver, cfg['Hp'], cfg['Block_V'], n_g,
                    label=cfg['label'], notifier=notifier,
                    config_idx=ci+1, total_configs=n_total_configs
                )

                # Print KPIs
                fmt_kpi = lambda v: f'day {v}' if v >= 0 else 'NOT reached'
                print(f"  KPI-1a (T-/+ dom): {fmt_kpi(res['kpi1a'])}")
                print(f"  KPI-1b (T-/- dom): {fmt_kpi(res['kpi1b'])}")
                print(f"  KPI-2  (T+>250%):  {fmt_kpi(res['kpi2'])}")
                print(f"  KPI-3  (TP>250%):  {fmt_kpi(res['kpi3'])}")
                print(f"  Drug Abi:          {res['drug_abi']:.4f}")
                print(f"  Drug Doc:          {res['drug_doc']:.4f}")
                print(f"  Computation time:  {res['t_elapsed']:.1f}s")

                # Save .npz
                t_vec = np.arange(0, Hs + dt, dt)
                np.savez(filepath,
                    Xhist_abs=res['Xhist_abs'], Xhist_freq=res['Xhist_freq'],
                    Uhist=res['Uhist'], t_vec=t_vec, y0_abs=y0_abs,
                    Hp=cfg['Hp'], Hs=Hs, dt=dt, N_max=N_max,
                    Block_V=cfg['Block_V'],
                    t_elapsed=res['t_elapsed'],
                    w_drug=cfg['w_drug'], w_slack=cfg['w_slack'], w_state=cfg['w_state'],
                    kpi1a=res['kpi1a'], kpi1b=res['kpi1b'],
                    kpi2=res['kpi2'], kpi3=res['kpi3'],
                    drug_abi=res['drug_abi'], drug_doc=res['drug_doc'],
                    tag=cfg['tag'],
                )
                print(f"  Saved: {filepath}")

                all_results.append({
                    'config': cfg,
                    'kpi1a': res['kpi1a'], 'kpi1b': res['kpi1b'],
                    'kpi2': res['kpi2'], 'kpi3': res['kpi3'],
                    'drug_abi': res['drug_abi'], 'drug_doc': res['drug_doc'],
                    't_elapsed': res['t_elapsed'],
                    'skipped': False,
                })
                n_completed += 1

                # Notify config completion
                notifier.notify_progress(ci + 1, n_total_configs, time.time() - t_total_start,
                    extra_info=(
                        f"Completed: {cfg['label']}\n"
                        f"KPI-1a: {fmt_kpi(res['kpi1a'])}\n"
                        f"KPI-1b: {fmt_kpi(res['kpi1b'])}\n"
                        f"Drug: Abi={res['drug_abi']:.4f}, Doc={res['drug_doc']:.4f}\n"
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

        # ---- Summary CSV ----
        csv_path = os.path.join(DATA_DIR, "Summary_DA.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Phase', 'Tag', 'Hp', 'Bw',
                'w_drug', 'w_slack', 'w_state',
                'KPI-1a', 'KPI-1b', 'KPI-2', 'KPI-3',
                'Drug_Abi', 'Drug_Doc', 'Drug_Total', 'Time_s', 'Status'
            ])
            for r in all_results:
                c = r['config']
                if 'error' in r:
                    writer.writerow([
                        c['phase'], c['tag'], c['Hp'], c['Block_V'],
                        '', '', '',
                        'ERROR', '', '', '', '', '', '', '', r['error']
                    ])
                else:
                    fmt = lambda v: str(v) if v >= 0 else 'NR'
                    drug_total = r['drug_abi'] + r['drug_doc']
                    status = 'skipped' if r.get('skipped') else 'ok'
                    writer.writerow([
                        c['phase'], c['tag'], c['Hp'], c['Block_V'],
                        f"{c['w_drug']:.2f}", f"{c['w_slack']:.2f}", f"{c['w_state']:.2f}",
                        fmt(r['kpi1a']), fmt(r['kpi1b']), fmt(r['kpi2']), fmt(r['kpi3']),
                        f"{r['drug_abi']:.4f}", f"{r['drug_doc']:.4f}",
                        f"{drug_total:.4f}", f"{r['t_elapsed']:.1f}", status
                    ])
        print(f"\nSummary CSV saved: {csv_path}")

        # ---- Comparison Plots ----
        valid_results = [r for r in all_results if 'error' not in r]
        if valid_results:
            generate_comparison_plots(all_results, IMG_DIR)

        # ---- Final Summary ----
        print(f"\n{'='*60}")
        print(f"  SENSITIVITY ANALYSIS DA COMPLETE")
        print(f"{'='*60}")
        print(f"  Completed: {n_completed} | Skipped: {n_skipped} | Errors: {n_errors}")
        print(f"  Total time: {t_total:.1f}s ({t_total/60:.1f} min)")
        print(f"  Output: {OUTPUT_DIR}/")

        notifier.notify_complete("Sensitivity Analysis DA", t_total, {
            "Completed": n_completed,
            "Skipped": n_skipped,
            "Errors": n_errors,
        }, {
            "Total configs": n_total_configs,
            "Unique configs": n_unique,
            "Output": OUTPUT_DIR,
        })

    except Exception as e:
        tb = traceback.format_exc()
        print(f"\nFATAL ERROR: {e}\n{tb}")
        notifier.notify_error("Fatal", str(e), tb)
        sys.exit(1)
