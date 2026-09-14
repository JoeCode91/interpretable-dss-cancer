"""
MPC_Replicator_DA_V3.py - Dual-Agent Replicator-MPC (Calibrated N_max)

REPRODUCES IN THE THESIS
    Dual-agent closed-loop results of Section 6.6.2, and the dual-agent column of Table 6.5 (tab:rep_comparison).

PROVENANCE
    Copied verbatim from the working tree of the thesis
    (Modelos_Replicator/Codigos/MPC_Replicator_DA_V3.py). The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md
    at the repository root for the complete list of differences.

--------------------------------------------------------------------------
V3: Uses calibrated N_max from Calibracion_Nmax_OpenLoop.py so that
Replicator open-loop dynamics match LV open-loop (Carlos's requirement).
Improved open-loop validation with corrected frequency comparison.

Replicator Dynamics + Zona Ficticia as MPC internal model.
Plant: Lotka-Volterra (West, 4 populations, Abiraterone + Docetaxel).
Includes active-set (slack variables E) for population reference tracking.

Architecture:
  Plant(LV) --> y_abs --> Convert(y->p) --> MPC(Replicator) --> u* --> Plant(LV)

Reference base code: TJ/MPC_Code_Art2_E.py
Date: 2026-02 (V2), 2026-02-25 (V3 calibrated)

AUTHORSHIP
    Johan S. Pena-Campos
    PhD thesis, "Interpretable Decision Support Systems for Early Cancer Diagnosis and
    Personalized Treatment", Universitat Politecnica de Catalunya and Pontificia
    Universidad Javeriana, 2026.

USE OF AI ASSISTANCE
    Generative AI (Claude, Anthropic) was used as a tool for code review, for verifying
    the code against the results reported in the thesis, for detecting and correcting
    errors, and for writing the documentation headers of this repository. The research
    design, the models, the experiments and the interpretation of the results are the
    author's. Every suggestion was reviewed by the author, who takes full responsibility
    for the content of this file.
"""

import casadi as cs
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for unattended execution
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import traceback
import sys
import os

# Telegram notifications
from telegram_notifier import MPCNotifier

# Force unbuffered output for real-time logging
print = lambda *args, **kwargs: __builtins__.__dict__['print'](*args, **{**kwargs, 'flush': True})


def DM2Arr(dm):
    return np.array(dm.full())


# ================================================================
# 1. PARAMETERS
# ================================================================

# --- Lotka-Volterra plant (West, 4 populations, dual-agent) ---
r_lv = cs.DM([2.78e-3, 3.55e-3, 6.65e-3, 6.65e-3])
delta_comp = 0.05
A_lv = cs.DM([[1,                0.7,              0.8,              0.8 - 2*delta_comp],
               [0.4,              1,                0.5,              0.5 - 2*delta_comp],
               [0.6,              0.9,              1,                1   - 2*delta_comp],
               [0.6 - delta_comp, 0.9 - delta_comp, 1 - delta_comp,  1]])
mu_param = 1.5
d_v = 5e-4

# --- Replicator parameters ---
# Calibrated value from Calibracion_Nmax_OpenLoop.py (2026-02-25)
# V2 used 1.0e4 (NRMSE=112.5%). Calibrated: 8e4 (NRMSE=3.59%)
N_max = 8e4

# --- Simulation ---
dt = 1
Hs = 30000
Hp = 140
B_rate = [0.05, 0.2]

# --- Initial conditions ---
y0_abs = np.array([2062.0, 2617.0, 3.97e-11, 3.97e-12])

# --- Active-set reference ---
Xs_abs = np.array([3000.0, 3000.0, 1800.0])
n_Eps = len(Xs_abs)

# --- MPC objective weights ---
Q_diag = cs.DM([1, 1, 0.5, 0.1])
R_diag = cs.DM([0.3, 1])
w_drug  = 0.3
w_slack = 0.45
w_state = 0.25

# ================================================================
# 2. MODEL DEFINITIONS
# ================================================================

# --- 2a. LV Plant model ---
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

# --- 2b. Replicator model with Zona Ficticia ---
n_pop = 4
n_total = n_pop + 1

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

# --- 2c. Conversion ---
def abs_to_freq(y_abs_val):
    y_arr = np.array(y_abs_val).flatten()
    p_real = y_arr / N_max
    p_0 = max(1.0 - np.sum(p_real), 0.0)
    return np.concatenate([[p_0], p_real])


# ================================================================
# OUTPUT DIRECTORIES
# ================================================================
IMG_DIR = os.path.join("Resultados", "DA", "Imagenes")
DATA_DIR = os.path.join("Resultados", "DA", "Datos")


# ================================================================
# MAIN EXECUTION
# ================================================================
if __name__ == "__main__":

    notifier = MPCNotifier()

    try:
        os.makedirs(IMG_DIR, exist_ok=True)
        os.makedirs(DATA_DIR, exist_ok=True)
        # Notify start
        notifier.notify_start("Replicator-MPC Dual-Agent (DA) V3", {
            "Hp": Hp, "Hs": Hs, "dt": dt,
            "N_max": N_max,
            "Block": 1,
            "Active-set Xs": str(Xs_abs),
            "y0": str(y0_abs),
            "Steps": int(Hs/dt),
        })

        # ============================================================
        # 3. OPEN-LOOP VALIDATION (Corrected in V3)
        # ============================================================
        print("=" * 60)
        print("  OPEN-LOOP VALIDATION: LV vs Replicator (u=0)")
        print("  V3: Corrected frequency comparison p/(1-x0) vs freq_LV")
        print("=" * 60)

        T_val = 5000  # V3: extended validation horizon
        n_val = int(T_val / dt)

        y_lv_val = np.zeros((4, n_val + 1))
        y_lv_val[:, 0] = y0_abs
        for k in range(n_val):
            y_curr = cs.DM(y_lv_val[:, k])
            dy = F_LV(y_curr, cs.DM([0, 0]))
            y_lv_val[:, k+1] = y_lv_val[:, k] + dt * np.array(dy).flatten()

        N_lv_val = np.sum(y_lv_val, axis=0)
        N_lv_safe = np.where(N_lv_val > 0, N_lv_val, 1.0)
        freq_lv_val = y_lv_val / N_lv_safe

        p_rep_val = np.zeros((n_total, n_val + 1))
        p_rep_val[:, 0] = abs_to_freq(y0_abs)
        for k in range(n_val):
            p_curr = cs.DM(p_rep_val[:, k])
            dp = F_REP(p_curr, cs.DM([0, 0]))
            p_rep_val[:, k+1] = p_rep_val[:, k] + dt * np.array(dp).flatten()
            p_rep_val[:, k+1] = np.maximum(p_rep_val[:, k+1], 0)
            p_rep_val[:, k+1] /= np.sum(p_rep_val[:, k+1])

        # V3: Corrected comparison -- p_i/(1-x0) vs freq_LV
        x0_val = p_rep_val[0, :]
        denom_val = np.where((1 - x0_val) > 1e-12, 1 - x0_val, 1e-12)
        freq_rep_corrected = p_rep_val[1:, :] / denom_val

        # V3: Absolute comparison -- N_max * p_i vs y_LV
        y_rep_abs = N_max * p_rep_val[1:, :]

        # V3: NRMSE metrics
        labels_pop = ['T+', 'TP', 'T-/+', 'T-/-']
        colors_pop = ['blue', 'green', 'red', 'cyan']

        nrmse_per_pop = np.zeros(4)
        for i in range(4):
            diff = y_lv_val[i, :] - y_rep_abs[i, :]
            rmse = np.sqrt(np.mean(diff**2))
            rng = np.max(y_lv_val[i, :]) - np.min(y_lv_val[i, :])
            nrmse_per_pop[i] = rmse / rng if rng > 0 else 0.0
        total_nrmse = np.mean(nrmse_per_pop)

        N_rep_val = N_max * (1 - x0_val)

        print(f"  N_max = {N_max:.0e}")
        print(f"  N_total(0) / N_max = {np.sum(y0_abs)/N_max*100:.1f}%")
        print(f"  N_lv_max = {np.max(N_lv_val):.0f}")
        for i in range(4):
            print(f"  {labels_pop[i]}: NRMSE = {nrmse_per_pop[i]*100:.2f}%")
        print(f"  TOTAL NRMSE = {total_nrmse*100:.2f}%")
        if total_nrmse < 0.05:
            print("  >>> CALIBRATION OK (NRMSE < 5%)")
        else:
            print("  >>> WARNING: NRMSE >= 5% -- consider re-calibrating N_max")

        t_val_vec = np.arange(0, T_val + dt, dt)
        FSIZE_AX = 14

        # Plot 1: Absolute comparison (V3 new)
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
        for i in range(4):
            ax = axes[i]
            ax.plot(t_val_vec, y_lv_val[i, :], '-', color=colors_pop[i], lw=2, label='LV')
            ax.plot(t_val_vec, y_rep_abs[i, :], '--', color=colors_pop[i], lw=2,
                    label=f'Rep ($N_{{max}}$={N_max:.0e})')
            ax.set_xlabel('Time (days)', fontsize=FSIZE_AX)
            ax.set_ylabel('Population', fontsize=FSIZE_AX)
            ax.set_title(f'{labels_pop[i]} (NRMSE={nrmse_per_pop[i]*100:.1f}%)', fontsize=FSIZE_AX)
            ax.legend(fontsize=9)
            ax.tick_params(labelsize=FSIZE_AX)
            ax.grid(True)
        plt.suptitle(f'Open-Loop Absolute Comparison | NRMSE={total_nrmse*100:.1f}% | $N_{{max}}$={N_max:.0e}',
                     fontsize=FSIZE_AX + 1)
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, 'Validation_DA_V3_Absolute.png'), dpi=150)
        plt.close()

        # Plot 2: Corrected frequency comparison (V3 improved)
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
        for i in range(4):
            ax = axes[i]
            ax.plot(t_val_vec, freq_lv_val[i, :], '-', color=colors_pop[i], lw=2, label='LV freq')
            ax.plot(t_val_vec, freq_rep_corrected[i, :], '--', color=colors_pop[i], lw=2,
                    label='Rep freq (corrected)')
            ax.set_xlabel('Time (days)', fontsize=FSIZE_AX)
            ax.set_ylabel('Frequency', fontsize=FSIZE_AX)
            ax.set_title(f'{labels_pop[i]}', fontsize=FSIZE_AX)
            ax.legend(fontsize=9)
            ax.tick_params(labelsize=FSIZE_AX)
            ax.grid(True)
        plt.suptitle(f'Open-Loop Frequency Comparison (V3 corrected) | $N_{{max}}$={N_max:.0e}',
                     fontsize=FSIZE_AX + 1)
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, 'Validation_DA_V3_Frequency.png'), dpi=150)
        plt.close()

        # Plot 3: Total population comparison (V3 new)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(t_val_vec, N_lv_val, 'b-', lw=2, label='$N_{LV}$')
        ax.plot(t_val_vec, N_rep_val, 'r--', lw=2, label=f'$N_{{max}}(1-x_0)$')
        ax.set_xlabel('Time (days)', fontsize=FSIZE_AX)
        ax.set_ylabel('Total population', fontsize=FSIZE_AX)
        ax.set_title(f'Total Population: LV vs Replicator | $N_{{max}}$={N_max:.0e}', fontsize=FSIZE_AX)
        ax.legend(fontsize=11)
        ax.tick_params(labelsize=FSIZE_AX)
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, 'Validation_DA_V3_Ntotal.png'), dpi=150)
        plt.close()

        max_err = 0
        for i in range(4):
            err = np.max(np.abs(freq_lv_val[i, :] - freq_rep_corrected[i, :]))
            max_err = max(max_err, err)
            print(f"  {labels_pop[i]}: max corrected freq error = {err:.6f}")
        print(f"  Overall max corrected error: {max_err:.6f}")

        notifier.notify_progress(0, 1, 0,
            extra_info=f"Open-loop validation V3\nNRMSE: {total_nrmse*100:.2f}%\nMax freq err: {max_err:.6f}")

        # ============================================================
        # 4. MPC SETUP
        # ============================================================
        print("\n" + "=" * 60)
        print(f"  MPC SETUP: Replicator-MPC (Dual-Agent), Hp={Hp}")
        print("=" * 60)

        n_controls = 2

        U_mpc = cs.SX.sym('U', n_controls, Hp)
        E_mpc = cs.SX.sym('E', n_Eps, Hp)
        X_mpc = cs.SX.sym('X', n_total, Hp + 1)

        n_par = n_total + n_controls
        P_mpc = cs.SX.sym('P', n_par)

        obj = 0
        X_mpc[:, 0] = P_mpc[:n_total]

        Q_mat = cs.diagcat(float(Q_diag[0]), float(Q_diag[1]), float(Q_diag[2]), float(Q_diag[3]))
        R_mat = cs.diagcat(float(R_diag[0]), float(R_diag[1]))

        for t in range(Hp):
            st = X_mpc[:, t]
            con = U_mpc[:, t]
            f_value = F_REP(st, con)
            X_mpc[:, t+1] = st + dt * f_value

            p_real = X_mpc[1:, t+1]
            obj += w_drug * (con.T @ R_mat @ con)
            obj += w_slack * (E_mpc[:, t].T @ cs.DM.eye(n_Eps) @ E_mpc[:, t])
            obj += w_state * (p_real.T @ Q_mat @ p_real)

        ff_mpc = cs.Function('ff_mpc', [U_mpc, P_mpc], [X_mpc])

        g = []
        Xs_freq = Xs_abs / N_max

        for t in range(1, Hp + 1):
            for i in range(n_total):
                g = cs.vertcat(g, -X_mpc[i, t])
                g = cs.vertcat(g, X_mpc[i, t] - 1.0)

            g = cs.vertcat(g, -U_mpc[:, t-1])
            g = cs.vertcat(g, U_mpc[:, t-1] - cs.DM([1, 1]))

            if t != 1:
                g = cs.vertcat(g, -(U_mpc[:, t-1] - U_mpc[0, t-2]) - cs.DM(B_rate))
                g = cs.vertcat(g, (U_mpc[:, t-1] - U_mpc[0, t-2]) - cs.DM(B_rate))
            else:
                g = cs.vertcat(g, -(P_mpc[n_total:] - U_mpc[:, t-1]) - cs.DM(B_rate))
                g = cs.vertcat(g, (P_mpc[n_total:] - U_mpc[:, t-1]) - cs.DM(B_rate))

            g = cs.vertcat(g, -E_mpc[:, t-1])
            g = cs.vertcat(g, X_mpc[1:n_Eps+1, t] - cs.DM(Xs_freq) - E_mpc[:, t-1])

        OPT_variables = cs.vertcat(
            cs.reshape(U_mpc, n_controls * Hp, 1),
            cs.reshape(E_mpc, n_Eps * Hp, 1)
        )

        nlp_prob = {'f': obj, 'x': OPT_variables, 'g': g, 'p': P_mpc}

        opts = {
            'ipopt': {'max_iter': 200, 'print_level': 0,
                      'acceptable_tol': 1e-8, 'acceptable_obj_change_tol': 1e-6},
            'print_time': 0
        }

        solver = cs.nlpsol('solver', 'ipopt', nlp_prob, opts)

        # ============================================================
        # 5. MPC SIMULATION LOOP
        # ============================================================
        n_steps = int(Hs / dt)

        x0_plant = cs.DM(y0_abs)
        u_prev = np.array([0.6, 0.0])
        U0 = np.tile(u_prev.reshape(n_controls, 1), (1, Hp))
        E0 = 0.5 * np.ones((n_Eps, Hp))

        Xhist_abs = np.zeros((4, n_steps + 1))
        Xhist_freq = np.zeros((n_total, n_steps + 1))
        Uhist = np.zeros((n_controls, n_steps + 1))

        Xhist_abs[:, 0] = y0_abs
        Xhist_freq[:, 0] = abs_to_freq(y0_abs)
        Uhist[:, 0] = u_prev

        lbg = [-float('inf')] * g.shape[0]
        ubg = [0] * g.shape[0]

        Block_V = 80

        print(f"\nRunning Replicator-MPC (Dual-Agent) V3")
        print(f"  Hp={Hp}, Hs={Hs}, dt={dt}, N_max={N_max}, Block={Block_V}")
        print(f"  Total MPC steps: {n_steps}")
        t_start = time.time()

        # Notification milestones: every 5% for long simulation
        notify_pct_interval = 5
        next_notify_pct = notify_pct_interval

        for sim in tqdm(range(n_steps)):
            if sim % Block_V == 0:
                if sim > 0:
                    U0 = UT
                p_current = abs_to_freq(x0_plant)
                Par = cs.DM(np.concatenate([p_current, U0[:, 0]]))

                x0_nlp = cs.vertcat(
                    cs.DM(U0.reshape(n_controls * Hp, 1)),
                    cs.DM(E0.reshape(n_Eps * Hp, 1))
                )
                sol = solver(lbg=lbg, ubg=ubg, p=Par, x0=x0_nlp)

                sol_x = sol['x'].full().flatten()
                U0 = sol_x[:n_controls * Hp].reshape(n_controls, Hp)
                E0 = sol_x[n_controls * Hp:].reshape(n_Eps, Hp)

            u_apply = U0[:, 0]
            x0_plant = x0_plant + dt * F_LV(x0_plant, cs.DM(u_apply))

            Xhist_abs[:, sim + 1] = np.array(x0_plant).flatten()
            Xhist_freq[:, sim + 1] = abs_to_freq(x0_plant)
            Uhist[:, sim + 1] = u_apply

            if sim % Block_V == 0:
                UT = np.hstack([U0[:, 1:], U0[:, -1:]])
                E0 = np.hstack([E0[:, 1:], E0[:, -1:]])

            # Telegram progress notification every N%
            pct = 100.0 * (sim + 1) / n_steps
            if pct >= next_notify_pct:
                elapsed = time.time() - t_start
                y_now = np.array(x0_plant).flatten()
                N_now = np.sum(y_now)
                notifier.notify_progress(sim + 1, n_steps, elapsed,
                    extra_info=(
                        f"u_a={u_apply[0]:.3f}, u_d={u_apply[1]:.3f}\n"
                        f"N_tumor={N_now:.0f}\n"
                        f"Day {(sim+1)*dt}/{Hs}"
                    ))
                next_notify_pct += notify_pct_interval

        t_elapsed = time.time() - t_start
        print(f"\nSimulation completed in {t_elapsed:.1f} seconds")

        # ============================================================
        # 6. RESULTS: KPIs AND PLOTS
        # ============================================================
        t_vec = np.arange(0, Hs + dt, dt)
        N_total_mpc = np.sum(Xhist_abs, axis=0)

        frac_res1 = Xhist_abs[2, :] / N_total_mpc
        frac_res2 = Xhist_abs[3, :] / N_total_mpc
        frac_sens = np.sum(Xhist_abs[:2, :], axis=0) / N_total_mpc

        kpi1a_idx = np.where(frac_res1 > frac_sens)[0]
        kpi1a = kpi1a_idx[0] if len(kpi1a_idx) > 0 else None

        kpi1b_idx = np.where(frac_res2 > frac_sens)[0]
        kpi1b = kpi1b_idx[0] if len(kpi1b_idx) > 0 else None

        kpi2_idx = np.where(Xhist_abs[0, :] > y0_abs[0] * 2.5)[0]
        kpi2 = kpi2_idx[0] if len(kpi2_idx) > 0 else None

        kpi3_idx = np.where(Xhist_abs[1, :] > y0_abs[1] * 2.5)[0]
        kpi3 = kpi3_idx[0] if len(kpi3_idx) > 0 else None

        drug_abi = np.mean(Uhist[0, :])
        drug_doc = np.mean(Uhist[1, :])

        print("\n" + "=" * 60)
        print("  RESULTS: Replicator-MPC (Dual-Agent) V3")
        print("=" * 60)
        print(f"  KPI-1a (T-/+ dominance): {f'day {kpi1a}' if kpi1a else 'not reached'}")
        print(f"  KPI-1b (T-/- dominance): {f'day {kpi1b}' if kpi1b else 'not reached'}")
        print(f"  KPI-2  (T+ > 250% x0):  {f'day {kpi2}' if kpi2 else 'not reached'}")
        print(f"  KPI-3  (TP > 250% x0):  {f'day {kpi3}' if kpi3 else 'not reached'}")
        print(f"  Drug exposure (Abi):     {drug_abi:.4f}")
        print(f"  Drug exposure (Doc):     {drug_doc:.4f}")
        print(f"  Computation time:        {t_elapsed:.1f}s")
        print(f"  Hp={Hp}, N_max={N_max}, Block={Block_V}")
        print(f"  Open-loop NRMSE:         {total_nrmse*100:.2f}%")

        # Generate .txt report
        report_file = os.path.join(DATA_DIR, f'Reporte_Replicator_DA_V3_Hp{Hp}_B{Block_V}.txt')
        with open(report_file, 'w') as file:
            file.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
            file.write("MODELO REPLICATOR-MPC DUAL-AGENT V3 (CALIBRATED)\n")
            file.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
            file.write("Resultados de la evolucion poblacional con Tratamiento.\n")
            file.write("Medicacion definida a traves de MPC con modelo interno Replicator+ZF.\n")
            file.write("Planta: Lotka-Volterra (West, 4 poblaciones, Abiraterone + Docetaxel).\n")
            file.write(f"Horizonte de prediccion: {Hp}\n")
            file.write(f"Ventana de bloqueo: {Block_V}\n")
            file.write(f"N_max: {N_max}\n")
            file.write(f"Open-loop NRMSE: {total_nrmse*100:.2f}%\n")
            file.write(f"Tiempo de computo: {t_elapsed:.1f}s\n")
            file.write(f"Exposicion farmaco (Abi): {drug_abi:.4f}\n")
            file.write(f"Exposicion farmaco (Doc): {drug_doc:.4f}\n")
            if kpi1a is not None:
                file.write(f"La poblacion celular resistente supera la suma de la poblacion celular sensible y productora en {kpi1a} dias\n")
            else:
                file.write("La poblacion celular resistente NO supera la suma de la poblacion celular sensible y productora\n")
            if kpi1b is not None:
                file.write(f"La poblacion celular resistente/resistente supera la suma de la poblacion celular sensible y productora en {kpi1b} dias\n")
            else:
                file.write("La poblacion celular resistente/resistente NO supera la suma de la poblacion celular sensible y productora\n")
            if kpi2 is not None:
                file.write(f"La poblacion celular sensible supera el 250% de la condicion inicial ({y0_abs[0]:.0f}) en {kpi2} dias\n")
            if kpi3 is not None:
                file.write(f"La poblacion celular productora supera el 250% de la condicion inicial ({y0_abs[1]:.0f}) en {kpi3} dias\n")
        print(f"Report saved: {report_file}")

        # Plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Replicator-MPC DA V3 | Hp={Hp} | $N_{{max}}$={N_max:.0e} | Time={t_elapsed:.1f}s',
                     fontsize=15)

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
        ax.set_ylim(-1, 10000)
        ax.grid(True)

        ax = axes[0, 1]
        ax.plot(t_vec, Xhist_freq[0, :], 'k--', lw=1.5, label='$x_0$ (fict.)')
        ax.plot(t_vec, Xhist_freq[1, :], 'b-', lw=2, label='$x_{T+}$')
        ax.plot(t_vec, Xhist_freq[2, :], 'g-', lw=2, label='$x_{TP}$')
        ax.plot(t_vec, Xhist_freq[3, :], 'r-', lw=2, label='$x_{T-/+}$')
        ax.plot(t_vec, Xhist_freq[4, :], 'c-', lw=2, label='$x_{T-/-}$')
        ax.set_xlabel('Time (days)', fontsize=FSIZE_AX)
        ax.set_ylabel('Frequency', fontsize=FSIZE_AX)
        ax.set_title('Population Frequencies (Replicator)', fontsize=FSIZE_AX)
        ax.legend(fontsize=11)
        ax.tick_params(labelsize=FSIZE_AX)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True)

        ax = axes[0, 2]
        ax.plot(t_vec, N_total_mpc, 'purple', lw=2, label='$N$ (plant)')
        N_zf = N_max * (1 - Xhist_freq[0, :])
        ax.plot(t_vec, N_zf, 'purple', lw=1.5, ls='--', label='$N_{max}(1-x_0)$')
        ax.set_xlabel('Time (days)', fontsize=FSIZE_AX)
        ax.set_ylabel('Total population', fontsize=FSIZE_AX)
        ax.set_title('Tumor Burden', fontsize=FSIZE_AX)
        ax.legend(fontsize=11)
        ax.tick_params(labelsize=FSIZE_AX)
        ax.grid(True)

        ax = axes[1, 0]
        ax.plot(t_vec, Uhist[0, :], 'b.', ms=1)
        ax.set_xlabel('Time (days)', fontsize=FSIZE_AX)
        ax.set_ylabel('Abiraterone', fontsize=FSIZE_AX)
        ax.set_title('Abiraterone Schedule', fontsize=FSIZE_AX)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=FSIZE_AX)
        ax.grid(True)

        ax = axes[1, 1]
        ax.plot(t_vec, Uhist[1, :], 'g.', ms=1)
        ax.set_xlabel('Time (days)', fontsize=FSIZE_AX)
        ax.set_ylabel('Docetaxel', fontsize=FSIZE_AX)
        ax.set_title('Docetaxel Schedule', fontsize=FSIZE_AX)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=FSIZE_AX)
        ax.grid(True)

        ax = axes[1, 2]
        ax.plot(t_vec, frac_sens, 'b-', lw=1.5, label='Sensitive (T+ + TP)')
        ax.plot(t_vec, frac_res1, 'r-', lw=1.5, label='T-/+')
        ax.plot(t_vec, frac_res2, 'c-', lw=1.5, label='T-/-')
        if kpi1a:
            ax.axvline(x=kpi1a, color='r', ls=':', alpha=0.7, label=f'KPI-1a: day {kpi1a}')
        if kpi1b:
            ax.axvline(x=kpi1b, color='c', ls=':', alpha=0.7, label=f'KPI-1b: day {kpi1b}')
        ax.set_xlabel('Time (days)', fontsize=FSIZE_AX)
        ax.set_ylabel('Population fraction', fontsize=FSIZE_AX)
        ax.set_title('KPI Fractions', fontsize=FSIZE_AX)
        ax.legend(fontsize=11)
        ax.tick_params(labelsize=FSIZE_AX)
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, f'Replicator_MPC_DA_V3_Hp{Hp}.png'), dpi=150)
        plt.close()

        # Save numerical results for post-processing
        results_file = os.path.join(DATA_DIR, f'Results_DA_V3_Hp{Hp}.npz')
        np.savez(results_file,
                 Xhist_abs=Xhist_abs, Xhist_freq=Xhist_freq,
                 Uhist=Uhist, t_vec=t_vec, y0_abs=y0_abs,
                 Hp=Hp, Hs=Hs, dt=dt, N_max=N_max, t_elapsed=t_elapsed,
                 w_drug=w_drug, w_slack=w_slack, w_state=w_state,
                 Block_V=Block_V,
                 kpi1a=kpi1a if kpi1a is not None else -1,
                 kpi1b=kpi1b if kpi1b is not None else -1,
                 kpi2=kpi2 if kpi2 is not None else -1,
                 kpi3=kpi3 if kpi3 is not None else -1,
                 drug_abi=drug_abi, drug_doc=drug_doc,
                 open_loop_nrmse=total_nrmse)
        print(f"\nResults saved: {results_file}")
        print(f"Figures saved: {IMG_DIR}/Validation_DA_V3_*.png, Replicator_MPC_DA_V3_Hp{Hp}.png")

        # Telegram: completion with KPIs
        notifier.notify_complete("Replicator-MPC DA V3", t_elapsed, {
            "KPI-1a (T-/+ dom)": f"day {kpi1a}" if kpi1a else "not reached",
            "KPI-1b (T-/- dom)": f"day {kpi1b}" if kpi1b else "not reached",
            "KPI-2 (T+>250%)": f"day {kpi2}" if kpi2 else "not reached",
            "KPI-3 (TP>250%)": f"day {kpi3}" if kpi3 else "not reached",
            "Drug Abi": f"{drug_abi:.4f}",
            "Drug Doc": f"{drug_doc:.4f}",
            "OL NRMSE": f"{total_nrmse*100:.2f}%",
        }, {
            "Hp": Hp, "Hs": Hs, "dt": dt, "N_max": N_max, "Block": Block_V,
        })

    except Exception as e:
        tb = traceback.format_exc()
        print(f"\nERROR: {e}\n{tb}")
        notifier.notify_error(type(e).__name__, str(e), tb)
        sys.exit(1)
