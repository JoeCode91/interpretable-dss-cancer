"""
Sensitivity_DA_T5B_V2.py - Table 5 Sims 15-20: CORRECTED configs matching PLoS ONE

REPRODUCES IN THE THESIS
    Dual-agent EMO-Opt sweep, simulations 15-20: Table tab:rep_da_full, and the
    Rep-MPC column of Table tab:rep_da_gamma.

PROVENANCE
    Copied verbatim from the working tree of the thesis
    (Modelos_Replicator/Codigos/Sensitivity_DA_T5B_V2.py). The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md
    at the repository root for the complete list of differences.

--------------------------------------------------------------------------
Sensitivity_DA_T5B_V2.py - Table 5 Sims 15-20: CORRECTED configs matching PLoS ONE
=====================================================================================
V2 fixes config mismatch in Sims 16, 17, 18 relative to the published paper
(BioModels_PLoSOne_251029.tex, tab:Exp_Art2_P2).

Changes from V1:
  Sim 16: gamma_4 = 2.5e-9 -> 0       (V1 had gamma_4 swapped with Sim17)
  Sim 17: gamma_4 = 0 -> 2.5e-9       (V1 had gamma_4 swapped with Sim16)
  Sim 18: gamma_1 = 2.5e-9 -> 0       (V1 had wrong gamma_1)
          gamma_5 = 0.2 -> 0           (V1 had wrong gamma_5)

Dual-Agent Replicator-MPC with active-set, blocking, and additional objective terms:
  gamma_4: Inter-variability between population pairs
  gamma_5: Smoothness penalty ||Delta_u||^2

RK4 integration inside NLP for numerical stability.

Table 5 Sims 15-20 (CORRECTED - matching PLoS ONE paper):
  Sim 15: gamma=(1e-8, 0.1, 1, 0, 0),           Hp=140, Bw=80   [unchanged]
  Sim 16: gamma=(2.5e-9, 0.1, 1, 0, 0),          Hp=140, Bw=80   [FIXED g4]
  Sim 17: gamma=(0, 0.1, 1, 2.5e-9, 0),          Hp=140, Bw=80   [FIXED g4]
  Sim 18: gamma=(0, 0.1, 0.65, 2.5e-9, 0),       Hp=140, Bw=80   [FIXED g1,g5]
  Sim 19: gamma=(2.5e-9, 0.1, 0.45, 0, 0.2),     Hp=140, Bw=80   [unchanged]
  Sim 20: gamma=(2.5e-9, 0.1, 0.45, 0, 0.2),     Hp=[20..200], Bw=140  [unchanged]
  Total: 11 configs

Output: Sensitivity_DA_T5B_V2/{Datos,Imagenes}/

Runs on: Home PC (overnight)
Usage: python Sensitivity_DA_T5B_V2.py [--test] [--sims 16,17,18]

Based on: Sensitivity_DA_T5B.py (V1)
Author: Johan Pena-Campos, PhD Project
Date: 2026-03-24
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
import math

from telegram_notifier import MPCNotifier

# Force unbuffered output
print = lambda *args, **kwargs: __builtins__.__dict__['print'](*args, **{**kwargs, 'flush': True})

def DM2Arr(dm):
    return np.array(dm.full())


# ================================================================
# CONFIGURATION (Table 5, Sims 15-20)
# ================================================================

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Sensitivity_DA_T5B_V2")
IMG_DIR = os.path.join(OUTPUT_DIR, "Imagenes")
DATA_DIR = os.path.join(OUTPUT_DIR, "Datos")
SKIP_EXISTING = True
TEST_MODE = '--test' in sys.argv

# Parse --sims argument: e.g. --sims 15,16 or --sims=15,16
SELECTED_SIMS = None
for i, arg in enumerate(sys.argv):
    if arg.startswith('--sims='):
        SELECTED_SIMS = [int(s) for s in arg.split('=')[1].split(',')]
    elif arg == '--sims' and i + 1 < len(sys.argv):
        SELECTED_SIMS = [int(s) for s in sys.argv[i + 1].split(',')]

# Fixed parameters
N_max = 8e4
dt = 1
Hs = 30000
B_rate = [0.05, 0.2]
y0_abs = np.array([2062.0, 2617.0, 3.97e-11, 3.97e-12])

# Active-set reference
Xs_abs = np.array([3000.0, 3000.0, 1800.0])
n_Eps = len(Xs_abs)

# Internal weight matrices
Q_diag = cs.DM([1, 1, 0.5, 0.1])
R_diag = cs.DM([0.3, 1])

FSIZE = 14
SCALE = (N_max / 1e4)**2  # = 64

# Inter-variability pairs: all (i,j) with i<j for 4 populations
INTER_PAIRS = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

# --- Table 5 Sims 15-20 configs ---
CONFIGS = [
    # Sim 15
    {'sim': 15, 'Hp': 140, 'Bw': 80,
     'gamma_1': 1e-8, 'gamma_2': 0.1, 'gamma_3': 1, 'gamma_4': 0, 'gamma_5': 0},
    # Sim 16  [FIXED: gamma_4 was 2.5e-9, paper has 0]
    {'sim': 16, 'Hp': 140, 'Bw': 80,
     'gamma_1': 2.5e-9, 'gamma_2': 0.1, 'gamma_3': 1, 'gamma_4': 0, 'gamma_5': 0},
    # Sim 17  [FIXED: gamma_4 was 0, paper has 2.5e-9]
    {'sim': 17, 'Hp': 140, 'Bw': 80,
     'gamma_1': 0, 'gamma_2': 0.1, 'gamma_3': 1, 'gamma_4': 2.5e-9, 'gamma_5': 0},
    # Sim 18  [FIXED: gamma_1 was 2.5e-9->0, gamma_5 was 0.2->0]
    {'sim': 18, 'Hp': 140, 'Bw': 80,
     'gamma_1': 0, 'gamma_2': 0.1, 'gamma_3': 0.65, 'gamma_4': 2.5e-9, 'gamma_5': 0},
    # Sim 19
    {'sim': 19, 'Hp': 140, 'Bw': 80,
     'gamma_1': 2.5e-9, 'gamma_2': 0.1, 'gamma_3': 0.45, 'gamma_4': 0, 'gamma_5': 0.2},
]

# Sim 20: Hp sweep with Bw=140
for hp in [20, 50, 80, 100, 140, 200]:
    CONFIGS.append({
        'sim': 20, 'Hp': hp, 'Bw': 140,
        'gamma_1': 2.5e-9, 'gamma_2': 0.1, 'gamma_3': 0.45, 'gamma_4': 0, 'gamma_5': 0.2,
    })

# Add derived fields
for cfg in CONFIGS:
    cfg['w_state'] = cfg['gamma_1'] * 1e8 * SCALE
    cfg['w_drug'] = cfg['gamma_2']
    cfg['w_slack'] = cfg['gamma_3'] * 1e8 * SCALE
    cfg['w_inter'] = cfg['gamma_4'] * 1e8 * SCALE
    cfg['w_smooth'] = cfg['gamma_5']
    cfg['name'] = f"Sim{cfg['sim']}_Hp{cfg['Hp']}_Bw{cfg['Bw']}"
    cfg['filename'] = f"Results_DA_T5B_V2_{cfg['name']}.npz"
    cfg['label'] = f"Sim{cfg['sim']} Hp={cfg['Hp']} Bw={cfg['Bw']}"


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
# SOLVER (RK4 + Active-Set + Blocking + Inter-var + Smoothness)
# ================================================================

def build_da_solver_V2_full(Hp, Bw, w_state, w_drug, w_slack, w_inter, w_smooth):
    """Build CasADi NLP for DA with RK4, active-set, blocking, and extended objectives.

    Args:
        Hp: Prediction horizon
        Bw: Blocking window
        w_state: Weight for state objective (p^T Q p)
        w_drug: Weight for drug objective (u^T R u)
        w_slack: Weight for slack objective (E^T E)
        w_inter: Weight for inter-variability (population pair differences)
        w_smooth: Weight for control smoothness (||Delta_u||^2)
    """
    n_blocks = math.ceil(Hp / Bw)

    U = cs.SX.sym('U', n_controls, n_blocks)
    E = cs.SX.sym('E', n_Eps, Hp)
    X = cs.SX.sym('X', n_total, Hp + 1)

    n_par = n_total + n_controls  # state (5) + u_prev (2)
    P = cs.SX.sym('P', n_par)

    obj = 0
    X[:, 0] = P[:n_total]

    Q_mat = cs.diagcat(float(Q_diag[0]), float(Q_diag[1]), float(Q_diag[2]), float(Q_diag[3]))
    R_mat = cs.diagcat(float(R_diag[0]), float(R_diag[1]))

    Xs_freq = Xs_abs / N_max

    for t in range(Hp):
        st = X[:, t]
        block_idx = min(t // Bw, n_blocks - 1)
        con = U[:, block_idx]

        # RK4 integration
        k1 = F_REP(st, con)
        k2 = F_REP(st + dt/2 * k1, con)
        k3 = F_REP(st + dt/2 * k2, con)
        k4 = F_REP(st + dt * k3, con)
        X[:, t+1] = st + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        p_real = X[1:, t+1]  # 4 real populations (indices 1..4)

        # Standard terms
        obj += w_drug * (con.T @ R_mat @ con)
        obj += w_slack * (E[:, t].T @ cs.DM.eye(n_Eps) @ E[:, t])
        obj += w_state * (p_real.T @ Q_mat @ p_real)

        # Inter-variability: sum over 6 pairs (x_i - x_j)^2 / 2
        if w_inter > 0:
            for (i, j) in INTER_PAIRS:
                obj += w_inter * (p_real[i] - p_real[j])**2 / 2

    # Smoothness: ||Delta_u||^2 between consecutive blocks
    if w_smooth > 0:
        for b in range(n_blocks):
            if b == 0:
                # First block vs u_prev (from parameter)
                delta_u = U[:, 0] - P[n_total:]
            else:
                delta_u = U[:, b] - U[:, b-1]
            obj += w_smooth * (delta_u.T @ R_mat @ delta_u)

    # Constraints
    g = []
    for t in range(1, Hp + 1):
        block_idx = min((t - 1) // Bw, n_blocks - 1)

        # State bounds [0, 1]
        for i in range(n_total):
            g = cs.vertcat(g, -X[i, t])
            g = cs.vertcat(g, X[i, t] - 1.0)

        # Control bounds [0, 1]
        g = cs.vertcat(g, -U[:, block_idx])
        g = cs.vertcat(g, U[:, block_idx] - cs.DM([1, 1]))

        # Rate constraints (between blocks)
        if block_idx == 0 and t == 1:
            g = cs.vertcat(g, -(P[n_total:] - U[:, 0]) - cs.DM(B_rate))
            g = cs.vertcat(g, (P[n_total:] - U[:, 0]) - cs.DM(B_rate))
        elif t > 1:
            prev_block_idx = min((t - 2) // Bw, n_blocks - 1)
            if block_idx != prev_block_idx:
                g = cs.vertcat(g, -(U[:, block_idx] - U[:, prev_block_idx]) - cs.DM(B_rate))
                g = cs.vertcat(g, (U[:, block_idx] - U[:, prev_block_idx]) - cs.DM(B_rate))

        # Slack constraints (active-set)
        g = cs.vertcat(g, -E[:, t-1])
        g = cs.vertcat(g, X[1:n_Eps+1, t] - cs.DM(Xs_freq) - E[:, t-1])

    OPT_vars = cs.vertcat(
        cs.reshape(U, n_controls * n_blocks, 1),
        cs.reshape(E, n_Eps * Hp, 1)
    )

    nlp = {'f': obj, 'x': OPT_vars, 'g': g, 'p': P}
    opts = {
        'ipopt': {'max_iter': 200, 'print_level': 0,
                  'acceptable_tol': 1e-8, 'acceptable_obj_change_tol': 1e-6},
        'print_time': 0,
        'error_on_fail': False,
    }
    solver = cs.nlpsol('solver', 'ipopt', nlp, opts)
    return solver, g.shape[0], n_blocks


# ================================================================
# SIMULATION
# ================================================================

def run_da_simulation(solver, Hp, Bw, n_blocks, n_g, label="",
                      notifier=None, config_idx=0, total_configs=1):
    """Run DA MPC simulation with blocking. Returns results dict."""
    n_steps = int(Hs / dt)

    x0 = cs.DM(y0_abs)
    u_prev = np.array([0.6, 0.0])
    U0 = np.tile(u_prev.reshape(n_controls, 1), (1, n_blocks))
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

    notify_pct_interval = 10
    next_notify_pct = notify_pct_interval

    for sim in tqdm(range(n_steps), desc=label, leave=True):
        if sim % Bw == 0:
            if sim > 0:
                U0 = UT
            p_curr = abs_to_freq(x0)
            Par = cs.DM(np.concatenate([p_curr, U0[:, 0]]))

            x0_nlp = cs.vertcat(
                cs.DM(U0.reshape(n_controls * n_blocks, 1)),
                cs.DM(E0.reshape(n_Eps * Hp, 1))
            )
            sol = solver(lbg=lbg, ubg=ubg, p=Par, x0=x0_nlp)

            sol_x = sol['x'].full().flatten()
            U0 = sol_x[:n_controls * n_blocks].reshape(n_controls, n_blocks)
            E0 = sol_x[n_controls * n_blocks:].reshape(n_Eps, Hp)

        u_apply = U0[:, 0]
        x0 = x0 + dt * F_LV(x0, cs.DM(u_apply))

        Xhist_abs[:, sim + 1] = np.array(x0).flatten()
        Xhist_freq[:, sim + 1] = abs_to_freq(x0)
        Uhist[:, sim + 1] = u_apply

        if sim % Bw == 0:
            if n_blocks > 1:
                UT = np.hstack([U0[:, 1:], U0[:, -1:]])
            else:
                UT = U0.copy()
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


# ================================================================
# PLOT FUNCTIONS
# ================================================================

def generate_config_plot(res, cfg, img_dir, t_vec):
    """Generate 2x3 overview plot for a single DA config."""
    Xhist_abs = res['Xhist_abs']
    Xhist_freq = res['Xhist_freq']
    Uhist = res['Uhist']
    N_total_mpc = np.sum(Xhist_abs, axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Replicator-MPC DA | {cfg['name']} | "
        f"$\\gamma$=({cfg['gamma_1']:.1e}, {cfg['gamma_2']}, {cfg['gamma_3']}, "
        f"{cfg['gamma_4']:.1e}, {cfg['gamma_5']}) | "
        f"Time={res['t_elapsed']:.1f}s",
        fontsize=FSIZE - 1)

    ax = axes[0, 0]
    ax.plot(t_vec, Xhist_abs[0, :], 'b-', lw=2, label='T+')
    ax.plot(t_vec, Xhist_abs[1, :], 'g-', lw=2, label='TP')
    ax.plot(t_vec, Xhist_abs[2, :], 'r-', lw=2, label='T-/+')
    ax.plot(t_vec, Xhist_abs[3, :], 'c-', lw=2, label='T-/-')
    ax.set_xlabel('Time (days)', fontsize=FSIZE)
    ax.set_ylabel('Population', fontsize=FSIZE)
    ax.set_title('Cell Dynamics (Plant - LV)', fontsize=FSIZE)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=FSIZE)
    ax.grid(True)

    ax = axes[0, 1]
    ax.plot(t_vec, Xhist_freq[0, :], 'k--', lw=1.5, label='$x_0$ (fict.)')
    ax.plot(t_vec, Xhist_freq[1, :], 'b-', lw=2, label='$x_{T+}$')
    ax.plot(t_vec, Xhist_freq[2, :], 'g-', lw=2, label='$x_{TP}$')
    ax.plot(t_vec, Xhist_freq[3, :], 'r-', lw=2, label='$x_{T-/+}$')
    ax.plot(t_vec, Xhist_freq[4, :], 'c-', lw=2, label='$x_{T-/-}$')
    ax.set_xlabel('Time (days)', fontsize=FSIZE)
    ax.set_ylabel('Frequency', fontsize=FSIZE)
    ax.set_title('Population Frequencies', fontsize=FSIZE)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=FSIZE)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)

    ax = axes[0, 2]
    ax.plot(t_vec, N_total_mpc, 'purple', lw=2, label='$N_{tumor}$ (plant)')
    N_zf = N_max * (1 - Xhist_freq[0, :])
    ax.plot(t_vec, N_zf, 'purple', lw=1.5, ls='--', label='$N_{max}(1-x_0)$')
    ax.set_xlabel('Time (days)', fontsize=FSIZE)
    ax.set_ylabel('Total population', fontsize=FSIZE)
    ax.set_title('Tumor Burden', fontsize=FSIZE)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=FSIZE)
    ax.grid(True)

    ax = axes[1, 0]
    ax.plot(t_vec, Uhist[0, :], 'b.', ms=1)
    ax.set_xlabel('Time (days)', fontsize=FSIZE)
    ax.set_ylabel('Abiraterone', fontsize=FSIZE)
    ax.set_title('Abiraterone Schedule', fontsize=FSIZE)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=FSIZE)
    ax.grid(True)

    ax = axes[1, 1]
    ax.plot(t_vec, Uhist[1, :], 'g.', ms=1)
    ax.set_xlabel('Time (days)', fontsize=FSIZE)
    ax.set_ylabel('Docetaxel', fontsize=FSIZE)
    ax.set_title('Docetaxel Schedule', fontsize=FSIZE)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=FSIZE)
    ax.grid(True)

    ax = axes[1, 2]
    N_safe = np.where(N_total_mpc > 0, N_total_mpc, 1.0)
    frac_sens = np.sum(Xhist_abs[:2, :], axis=0) / N_safe
    frac_res1 = Xhist_abs[2, :] / N_safe
    frac_res2 = Xhist_abs[3, :] / N_safe
    ax.plot(t_vec, frac_sens, 'b-', lw=1.5, label='Sensitive (T+ + TP)')
    ax.plot(t_vec, frac_res1, 'r-', lw=1.5, label='T-/+')
    ax.plot(t_vec, frac_res2, 'c-', lw=1.5, label='T-/-')
    if res['kpi1a'] >= 0:
        ax.axvline(x=res['kpi1a'], color='r', ls=':', alpha=0.7, label=f"KPI-1a: day {res['kpi1a']}")
    if res['kpi1b'] >= 0:
        ax.axvline(x=res['kpi1b'], color='c', ls=':', alpha=0.7, label=f"KPI-1b: day {res['kpi1b']}")
    ax.set_xlabel('Time (days)', fontsize=FSIZE)
    ax.set_ylabel('Population fraction', fontsize=FSIZE)
    ax.set_title('KPI Fractions', fontsize=FSIZE)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=FSIZE)
    ax.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(img_dir, f"DA_T5B_V2_{cfg['name']}_Overview.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Overview plot saved: {plot_path}")


def generate_comparison_plots(all_results, img_dir):
    """Generate comparison plots grouped by Sim number."""
    valid = [r for r in all_results if 'error' not in r]
    if not valid:
        return

    NOT_REACHED = Hs + 1000
    sim_nums = sorted(set(r['config']['sim'] for r in valid))
    colors_map = {15: 'blue', 16: 'green', 17: 'orange', 18: 'red', 19: 'purple', 20: 'brown'}
    markers_map = {15: 'o', 16: 's', 17: '^', 18: 'D', 19: 'v', 20: 'p'}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Sensitivity DA T5B: Table 5 Sims 15-20', fontsize=16)

    for sn in sim_nums:
        sim_results = sorted(
            [r for r in valid if r['config']['sim'] == sn],
            key=lambda r: r['config']['Hp']
        )
        if not sim_results:
            continue

        c = colors_map.get(sn, 'gray')
        m = markers_map.get(sn, 'o')
        label = f'Sim {sn}'

        if len(sim_results) == 1:
            # Single point: use bar-like marker
            x_vals = [sn]
        else:
            x_vals = [r['config']['Hp'] for r in sim_results]

        kpi1a = [r['kpi1a'] if r['kpi1a'] >= 0 else NOT_REACHED for r in sim_results]
        kpi1b = [r['kpi1b'] if r['kpi1b'] >= 0 else NOT_REACHED for r in sim_results]
        drugs_a = [r['drug_abi'] for r in sim_results]
        drugs_d = [r['drug_doc'] for r in sim_results]
        times = [r['t_elapsed'] for r in sim_results]

        axes[0, 0].plot(x_vals, kpi1a, f'-{m}', color=c, lw=1.5, ms=7, label=f'{label} (T-/+)')
        axes[0, 1].plot(x_vals, kpi1b, f'-{m}', color=c, lw=1.5, ms=7, label=f'{label} (T-/-)')
        axes[0, 2].plot(x_vals, times, f'-{m}', color=c, lw=1.5, ms=7, label=label)
        axes[1, 0].plot(x_vals, drugs_a, f'-{m}', color=c, lw=1.5, ms=7, label=label)
        axes[1, 1].plot(x_vals, drugs_d, f'-{m}', color=c, lw=1.5, ms=7, label=label)

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=FSIZE - 1)
        ax.legend(fontsize=8)

    axes[0, 0].set_ylabel('Day', fontsize=FSIZE)
    axes[0, 0].set_title('KPI-1a: T-/+ Dominance', fontsize=FSIZE)
    axes[0, 1].set_ylabel('Day', fontsize=FSIZE)
    axes[0, 1].set_title('KPI-1b: T-/- Dominance', fontsize=FSIZE)
    axes[0, 2].set_ylabel('Time (s)', fontsize=FSIZE)
    axes[0, 2].set_title('Computation Time', fontsize=FSIZE)
    axes[1, 0].set_ylabel('Mean dose', fontsize=FSIZE)
    axes[1, 0].set_title('Abiraterone Exposure', fontsize=FSIZE)
    axes[1, 1].set_ylabel('Mean dose', fontsize=FSIZE)
    axes[1, 1].set_title('Docetaxel Exposure', fontsize=FSIZE)
    axes[1, 2].set_visible(False)

    plt.tight_layout()
    plot_path = os.path.join(img_dir, "Comparison_DA_T5B.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved: {plot_path}")


# ================================================================
# MAIN EXECUTION
# ================================================================
if __name__ == "__main__":

    notifier = MPCNotifier()
    all_results = []

    configs = CONFIGS
    if SELECTED_SIMS is not None:
        configs = [c for c in configs if c['sim'] in SELECTED_SIMS]
        print(f"*** PARALLEL MODE: Running sims {SELECTED_SIMS} only ({len(configs)} configs) ***")
    if TEST_MODE:
        # Sim 17 is simplest (gamma_1=0)
        test_cfgs = [c for c in configs if c['sim'] == 17]
        configs = test_cfgs if test_cfgs else [configs[0]]
        print("*** TEST MODE: Running Sim 17 only for validation ***")

    try:
        os.makedirs(IMG_DIR, exist_ok=True)
        os.makedirs(DATA_DIR, exist_ok=True)

        n_total_configs = len(configs)
        print("=" * 60)
        print("  SENSITIVITY ANALYSIS DA: Table 5 Sims 15-20 (T5B)")
        print("  Extended objectives: inter-variability + smoothness")
        print("=" * 60)
        print(f"  Output: {OUTPUT_DIR}/")
        print(f"  Total configs: {n_total_configs}")
        print(f"  Skip existing: {SKIP_EXISTING}")
        print(f"  N_max={N_max:.0e}, dt={dt}, Hs={Hs}, SCALE={SCALE:.0f}")
        if TEST_MODE:
            print("  *** TEST MODE ***")
        print()

        notifier.notify_start("Sensitivity DA T5B_V2 (Table 5, Sims 15-20, FIXED configs)", {
            "Total configs": n_total_configs,
            "Hs": Hs, "dt": dt, "N_max": f"{N_max:.0e}",
            "Features": "inter-variability + smoothness",
            "Test mode": TEST_MODE,
        })

        t_total_start = time.time()
        n_completed = 0
        n_skipped = 0
        n_errors = 0

        for ci, cfg in enumerate(configs):
            filepath = os.path.join(DATA_DIR, cfg['filename'])

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
            print(f"  Hp={cfg['Hp']}, Bw={cfg['Bw']}")
            print(f"  gamma=({cfg['gamma_1']:.1e}, {cfg['gamma_2']}, {cfg['gamma_3']}, "
                  f"{cfg['gamma_4']:.1e}, {cfg['gamma_5']})")
            print(f"  w_state={cfg['w_state']:.2f}, w_drug={cfg['w_drug']:.2f}, "
                  f"w_slack={cfg['w_slack']:.2e}")
            print(f"  w_inter={cfg['w_inter']:.2e}, w_smooth={cfg['w_smooth']:.2f}")
            n_nlp_solves = int(Hs / dt / cfg['Bw'])
            n_blocks = math.ceil(cfg['Hp'] / cfg['Bw'])
            print(f"  NLP blocks: {n_blocks}, NLP solves: {n_nlp_solves}")
            print(f"{'='*60}")

            try:
                notifier.notify_progress(ci, n_total_configs, time.time() - t_total_start,
                    extra_info=(
                        f"Starting: {cfg['label']}\n"
                        f"gamma=({cfg['gamma_1']:.1e}, {cfg['gamma_2']}, {cfg['gamma_3']}, "
                        f"{cfg['gamma_4']:.1e}, {cfg['gamma_5']})\n"
                        f"NLP solves: {n_nlp_solves}"
                    ))

                t_build = time.time()
                solver, n_g, n_blk = build_da_solver_V2_full(
                    cfg['Hp'], cfg['Bw'],
                    cfg['w_state'], cfg['w_drug'], cfg['w_slack'],
                    cfg['w_inter'], cfg['w_smooth']
                )
                print(f"  Solver built in {time.time()-t_build:.1f}s")

                res = run_da_simulation(
                    solver, cfg['Hp'], cfg['Bw'], n_blk, n_g,
                    label=cfg['label'], notifier=notifier,
                    config_idx=ci+1, total_configs=n_total_configs
                )

                fmt_kpi = lambda v: f'day {v}' if v >= 0 else 'NOT reached'
                print(f"  KPI-1a (T-/+ dom): {fmt_kpi(res['kpi1a'])}")
                print(f"  KPI-1b (T-/- dom): {fmt_kpi(res['kpi1b'])}")
                print(f"  KPI-2  (T+>250%):  {fmt_kpi(res['kpi2'])}")
                print(f"  KPI-3  (TP>250%):  {fmt_kpi(res['kpi3'])}")
                print(f"  Drug Abi:          {res['drug_abi']:.4f}")
                print(f"  Drug Doc:          {res['drug_doc']:.4f}")
                print(f"  Computation time:  {res['t_elapsed']:.1f}s")

                t_vec = np.arange(0, Hs + dt, dt)
                np.savez(filepath,
                    Xhist_abs=res['Xhist_abs'], Xhist_freq=res['Xhist_freq'],
                    Uhist=res['Uhist'], t_vec=t_vec, y0_abs=y0_abs,
                    Hp=cfg['Hp'], Hs=Hs, dt=dt, N_max=N_max,
                    Bw=cfg['Bw'],
                    t_elapsed=res['t_elapsed'],
                    w_state=cfg['w_state'], w_drug=cfg['w_drug'], w_slack=cfg['w_slack'],
                    w_inter=cfg['w_inter'], w_smooth=cfg['w_smooth'],
                    gamma_1=cfg['gamma_1'], gamma_2=cfg['gamma_2'], gamma_3=cfg['gamma_3'],
                    gamma_4=cfg['gamma_4'], gamma_5=cfg['gamma_5'],
                    kpi1a=res['kpi1a'], kpi1b=res['kpi1b'],
                    kpi2=res['kpi2'], kpi3=res['kpi3'],
                    drug_abi=res['drug_abi'], drug_doc=res['drug_doc'],
                    sim=cfg['sim'],
                )
                print(f"  Saved: {filepath}")

                generate_config_plot(res, cfg, IMG_DIR, t_vec)

                all_results.append({
                    'config': cfg,
                    'kpi1a': res['kpi1a'], 'kpi1b': res['kpi1b'],
                    'kpi2': res['kpi2'], 'kpi3': res['kpi3'],
                    'drug_abi': res['drug_abi'], 'drug_doc': res['drug_doc'],
                    't_elapsed': res['t_elapsed'],
                    'skipped': False,
                })
                n_completed += 1

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

        # Summary CSV
        csv_path = os.path.join(DATA_DIR, "Summary_DA_T5B_V2.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Sim', 'Name', 'Hp', 'Bw',
                'gamma_1', 'gamma_2', 'gamma_3', 'gamma_4', 'gamma_5',
                'w_state', 'w_drug', 'w_slack', 'w_inter', 'w_smooth',
                'KPI-1a', 'KPI-1b', 'KPI-2', 'KPI-3',
                'Drug_Abi', 'Drug_Doc', 'Drug_Total', 'Time_s', 'Status'
            ])
            for r in all_results:
                c = r['config']
                if 'error' in r:
                    writer.writerow([
                        c['sim'], c['name'], c['Hp'], c['Bw'],
                        c['gamma_1'], c['gamma_2'], c['gamma_3'], c['gamma_4'], c['gamma_5'],
                        '', '', '', '', '',
                        'ERROR', '', '', '', '', '', '', '', r['error']
                    ])
                else:
                    fmt = lambda v: str(v) if v >= 0 else 'NR'
                    drug_total = r['drug_abi'] + r['drug_doc']
                    status = 'skipped' if r.get('skipped') else 'ok'
                    writer.writerow([
                        c['sim'], c['name'], c['Hp'], c['Bw'],
                        c['gamma_1'], c['gamma_2'], c['gamma_3'], c['gamma_4'], c['gamma_5'],
                        f"{c['w_state']:.2f}", f"{c['w_drug']:.2f}", f"{c['w_slack']:.2e}",
                        f"{c['w_inter']:.2e}", f"{c['w_smooth']:.2f}",
                        fmt(r['kpi1a']), fmt(r['kpi1b']), fmt(r['kpi2']), fmt(r['kpi3']),
                        f"{r['drug_abi']:.4f}", f"{r['drug_doc']:.4f}",
                        f"{drug_total:.4f}", f"{r['t_elapsed']:.1f}", status
                    ])
        print(f"\nSummary CSV saved: {csv_path}")

        valid_results = [r for r in all_results if 'error' not in r]
        if valid_results and not TEST_MODE:
            generate_comparison_plots(all_results, IMG_DIR)

        print(f"\n{'='*60}")
        print(f"  SENSITIVITY DA T5B_V2 COMPLETE (Table 5, FIXED configs)")
        print(f"{'='*60}")
        print(f"  Completed: {n_completed} | Skipped: {n_skipped} | Errors: {n_errors}")
        print(f"  Total time: {t_total:.1f}s ({t_total/60:.1f} min)")
        print(f"  Output: {OUTPUT_DIR}/")

        notifier.notify_complete("Sensitivity DA T5B_V2 (Table 5, Sims 15-20, FIXED configs)", t_total, {
            "Completed": n_completed,
            "Skipped": n_skipped,
            "Errors": n_errors,
        }, {
            "Total configs": n_total_configs,
            "Output": OUTPUT_DIR,
        })

    except Exception as e:
        tb = traceback.format_exc()
        print(f"\nFATAL ERROR: {e}\n{tb}")
        notifier.notify_error("Fatal", str(e), tb)
        sys.exit(1)
