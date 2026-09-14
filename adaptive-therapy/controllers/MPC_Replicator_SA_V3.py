"""
MPC_Replicator_SA_V3.py - Single-Agent Replicator-MPC (Calibrated N_max)

REPRODUCES IN THE THESIS
    Single-agent closed-loop results of Section 6.6.1, and the single-agent column of Table 6.5 (tab:rep_comparison).

PROVENANCE
    Copied verbatim from the working tree of the thesis
    (Modelos_Replicator/Codigos/MPC_Replicator_SA_V3.py). The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md
    at the repository root for the complete list of differences.

--------------------------------------------------------------------------
V3: Uses calibrated N_max from Calibracion_Nmax_OpenLoop.py so that
Replicator open-loop dynamics match LV open-loop (Carlos's requirement).
Improved open-loop validation with corrected frequency comparison.

Replicator Dynamics + Zona Ficticia as MPC internal model.
Plant: Lotka-Volterra (Cunningham, 3 populations, Abiraterone only).

Architecture:
  Plant(LV) --> y_abs --> Convert(y->p) --> MPC(Replicator) --> u* --> Plant(LV)

Reference base code: TJ/MPC_Code.py
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

# --- Lotka-Volterra plant (Cunningham, 3 populations) ---
r_lv = cs.DM([2.7726e-3, 3.4657e-3, 6.6542e-3])
A_lv = cs.DM([[1, 0.7, 0.8],
               [0.4, 1, 0.6],
               [0.5, 0.9, 1]])
mu_param = 1.5

# --- Replicator parameters ---
# Calibrated value from Calibracion_Nmax_OpenLoop.py (2026-02-25)
# V2 used 1.5e4 (NRMSE=22.7%). Calibrated: 8e4 (NRMSE=4.99%)
N_max = 8e4

# --- Simulation ---
dt = 10
Hs = 4000
Hp = 50
B_rate = 0.05

# --- Initial conditions ---
y0_abs = np.array([560.36, 747.59, 47.10])

# --- MPC objective weights ---
# Note: w_sens and w_res auto-scale with N_max
gamma_1_abs = 3.6e-7
gamma_2_abs = 0.1
gamma_3_abs = 1.6e-3

w_sens = gamma_1_abs * N_max**2
w_drug = gamma_2_abs
w_res  = gamma_3_abs * N_max

# ================================================================
# 2. MODEL DEFINITIONS
# ================================================================

# --- 2a. LV Plant model ---
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

# --- 2b. Replicator model with Zona Ficticia ---
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

# --- 2c. Conversion ---
def abs_to_freq(y_abs_val):
    y_arr = np.array(y_abs_val).flatten()
    p_real = y_arr / N_max
    p_0 = max(1.0 - np.sum(p_real), 0.0)
    return np.concatenate([[p_0], p_real])


# ================================================================
# OUTPUT DIRECTORIES
# ================================================================
IMG_DIR = os.path.join("Resultados", "SA", "Imagenes")
DATA_DIR = os.path.join("Resultados", "SA", "Datos")


# ================================================================
# MAIN EXECUTION
# ================================================================
if __name__ == "__main__":

    notifier = MPCNotifier()

    try:
        os.makedirs(IMG_DIR, exist_ok=True)
        os.makedirs(DATA_DIR, exist_ok=True)
        # Notify start
        notifier.notify_start("Replicator-MPC Single-Agent (SA) V3", {
            "Hp": Hp, "Hs": Hs, "dt": dt,
            "N_max": N_max,
            "w_sens": f"{w_sens:.2f}", "w_drug": f"{w_drug:.2f}", "w_res": f"{w_res:.2f}",
            "y0": str(y0_abs)
        })

        # ============================================================
        # 3. OPEN-LOOP VALIDATION (Corrected in V3)
        # ============================================================
        print("=" * 60)
        print("  OPEN-LOOP VALIDATION: LV vs Replicator (u=0)")
        print("  V3: Corrected frequency comparison p/(1-x0) vs freq_LV")
        print("=" * 60)

        T_val = 4000  # V3: full simulation horizon for validation
        n_val = int(T_val / dt)

        y_lv = np.zeros((3, n_val + 1))
        y_lv[:, 0] = y0_abs
        for k in range(n_val):
            y_curr = cs.DM(y_lv[:, k])
            dy = F_LV(y_curr, 0.0)
            y_lv[:, k+1] = y_lv[:, k] + dt * np.array(dy).flatten()

        N_lv = np.sum(y_lv, axis=0)
        N_lv_safe = np.where(N_lv > 0, N_lv, 1.0)
        freq_lv = y_lv / N_lv_safe

        p_rep = np.zeros((n_total, n_val + 1))
        p_rep[:, 0] = abs_to_freq(y0_abs)
        for k in range(n_val):
            p_curr = cs.DM(p_rep[:, k])
            dp = F_REP(p_curr, 0.0)
            p_rep[:, k+1] = p_rep[:, k] + dt * np.array(dp).flatten()
            p_rep[:, k+1] = np.maximum(p_rep[:, k+1], 0)
            p_rep[:, k+1] /= np.sum(p_rep[:, k+1])

        # V3: Corrected comparison -- p_i/(1-x0) vs freq_LV
        x0_val = p_rep[0, :]
        denom_val = np.where((1 - x0_val) > 1e-12, 1 - x0_val, 1e-12)
        freq_rep_corrected = p_rep[1:, :] / denom_val

        # V3: Absolute comparison -- N_max * p_i vs y_LV
        y_rep_abs = N_max * p_rep[1:, :]

        # V3: NRMSE metrics
        nrmse_per_pop = np.zeros(3)
        for i in range(3):
            diff = y_lv[i, :] - y_rep_abs[i, :]
            rmse = np.sqrt(np.mean(diff**2))
            rng = np.max(y_lv[i, :]) - np.min(y_lv[i, :])
            nrmse_per_pop[i] = rmse / rng if rng > 0 else 0.0
        total_nrmse = np.mean(nrmse_per_pop)

        N_rep = N_max * (1 - x0_val)

        labels_pop = ['T+', 'TP', 'T-']
        colors_pop = ['blue', 'green', 'red']

        print(f"  N_max = {N_max:.0e}")
        print(f"  N_total(0) / N_max = {np.sum(y0_abs)/N_max*100:.1f}%")
        print(f"  N_lv_max = {np.max(N_lv):.0f}")
        for i in range(3):
            print(f"  {labels_pop[i]}: NRMSE = {nrmse_per_pop[i]*100:.2f}%")
        print(f"  TOTAL NRMSE = {total_nrmse*100:.2f}%")
        if total_nrmse < 0.05:
            print("  >>> CALIBRATION OK (NRMSE < 5%)")
        else:
            print("  >>> WARNING: NRMSE >= 5% -- consider re-calibrating N_max")

        t_val = np.arange(0, T_val + dt, dt)
        FSIZE_AX = 14

        # Plot 1: Absolute comparison (V3 new)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for i in range(3):
            ax = axes[i]
            ax.plot(t_val, y_lv[i, :], '-', color=colors_pop[i], lw=2, label='LV')
            ax.plot(t_val, y_rep_abs[i, :], '--', color=colors_pop[i], lw=2,
                    label=f'Rep ($N_{{max}}$={N_max:.0e})')
            ax.set_xlabel('Time (days)', fontsize=FSIZE_AX)
            ax.set_ylabel('Population', fontsize=FSIZE_AX)
            ax.set_title(f'{labels_pop[i]} (NRMSE={nrmse_per_pop[i]*100:.1f}%)', fontsize=FSIZE_AX)
            ax.legend(fontsize=11)
            ax.tick_params(labelsize=FSIZE_AX)
            ax.grid(True)
        plt.suptitle(f'Open-Loop Absolute Comparison | NRMSE={total_nrmse*100:.1f}% | $N_{{max}}$={N_max:.0e}',
                     fontsize=FSIZE_AX + 1)
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, 'Validation_SA_V3_Absolute.png'), dpi=150)
        plt.close()

        # Plot 2: Corrected frequency comparison (V3 improved)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for i in range(3):
            ax = axes[i]
            ax.plot(t_val, freq_lv[i, :], '-', color=colors_pop[i], lw=2, label='LV freq')
            ax.plot(t_val, freq_rep_corrected[i, :], '--', color=colors_pop[i], lw=2,
                    label='Rep freq (corrected)')
            ax.set_xlabel('Time (days)', fontsize=FSIZE_AX)
            ax.set_ylabel('Frequency', fontsize=FSIZE_AX)
            ax.set_title(f'{labels_pop[i]}', fontsize=FSIZE_AX)
            ax.legend(fontsize=11)
            ax.tick_params(labelsize=FSIZE_AX)
            ax.grid(True)
        plt.suptitle(f'Open-Loop Frequency Comparison (V3 corrected) | $N_{{max}}$={N_max:.0e}',
                     fontsize=FSIZE_AX + 1)
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, 'Validation_SA_V3_Frequency.png'), dpi=150)
        plt.close()

        # Plot 3: Total population comparison (V3 new)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(t_val, N_lv, 'b-', lw=2, label='$N_{LV}$')
        ax.plot(t_val, N_rep, 'r--', lw=2, label=f'$N_{{max}}(1-x_0)$')
        ax.set_xlabel('Time (days)', fontsize=FSIZE_AX)
        ax.set_ylabel('Total population', fontsize=FSIZE_AX)
        ax.set_title(f'Total Population: LV vs Replicator | $N_{{max}}$={N_max:.0e}', fontsize=FSIZE_AX)
        ax.legend(fontsize=11)
        ax.tick_params(labelsize=FSIZE_AX)
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, 'Validation_SA_V3_Ntotal.png'), dpi=150)
        plt.close()

        max_err_freq = np.max(np.abs(freq_lv - freq_rep_corrected))
        print(f"  Max corrected frequency discrepancy: {max_err_freq:.6f}")

        notifier.notify_progress(0, 1, 0,
            extra_info=f"Open-loop validation V3\nNRMSE: {total_nrmse*100:.2f}%\nMax freq err: {max_err_freq:.6f}")

        # ============================================================
        # 4. MPC SETUP
        # ============================================================
        print("\n" + "=" * 60)
        print(f"  MPC SETUP: Replicator-MPC, Hp={Hp}")
        print("=" * 60)

        U_mpc = cs.SX.sym('U', 1, Hp)
        X_mpc = cs.SX.sym('X', n_total, Hp + 1)
        n_par = n_total + 1
        P_mpc = cs.SX.sym('P', n_par)

        obj = 0
        X_mpc[:, 0] = P_mpc[:n_total]

        for t in range(Hp):
            st = X_mpc[:, t]
            con = U_mpc[:, t]
            f_value = F_REP(st, con)
            X_mpc[:, t+1] = st + dt * f_value
            obj += w_sens * (X_mpc[1, t+1]**2 + X_mpc[2, t+1]**2)
            obj += w_drug * (con[0]**2)
            obj += w_res * X_mpc[3, t+1]

        ff_mpc = cs.Function('ff_mpc', [U_mpc, P_mpc], [X_mpc])

        g = []
        for t in range(1, Hp + 1):
            for i in range(n_total):
                g = cs.vertcat(g, -X_mpc[i, t])
                g = cs.vertcat(g, X_mpc[i, t] - 1.0)
            g = cs.vertcat(g, -U_mpc[:, t-1])
            g = cs.vertcat(g, U_mpc[:, t-1] - 1.0)
            if t != 1:
                g = cs.vertcat(g, -(U_mpc[:, t-1] - U_mpc[:, t-2]) - B_rate)
                g = cs.vertcat(g, (U_mpc[:, t-1] - U_mpc[:, t-2]) - B_rate)
            else:
                g = cs.vertcat(g, -(P_mpc[n_total] - U_mpc[:, t-1]) - B_rate)
                g = cs.vertcat(g, (P_mpc[n_total] - U_mpc[:, t-1]) - B_rate)

        OPT_variables = cs.reshape(U_mpc, Hp, 1)
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
        u_prev = 1.0
        U0 = u_prev * cs.DM.ones(1, Hp)

        Xhist_abs = np.zeros((3, n_steps + 1))
        Xhist_freq = np.zeros((n_total, n_steps + 1))
        Uhist = np.zeros(n_steps + 1)

        Xhist_abs[:, 0] = y0_abs
        Xhist_freq[:, 0] = abs_to_freq(y0_abs)
        Uhist[0] = u_prev

        lbg = [-float('inf')] * g.shape[0]
        ubg = [0] * g.shape[0]

        print(f"\nRunning Replicator-MPC (Single-Agent) V3")
        print(f"  Hp={Hp}, Hs={Hs}, dt={dt}, N_max={N_max}")
        print(f"  Weights: sens={w_sens}, drug={w_drug}, res={w_res}")
        t_start = time.time()

        # Notification milestones: every 10%
        notify_milestones = set(range(10, 101, 10))
        last_pct_notified = 0

        for sim in tqdm(range(n_steps)):
            p_current = abs_to_freq(x0_plant)

            Par = cs.DM(np.concatenate([p_current, [u_prev]]))
            sol = solver(lbg=lbg, ubg=ubg, p=Par, x0=cs.reshape(U0, Hp, 1))
            U0 = sol['x'].full().reshape(1, Hp)

            u_opt = float(U0[0, 0])
            u_prev = u_opt

            x0_plant = x0_plant + dt * F_LV(x0_plant, u_opt)

            Xhist_abs[:, sim + 1] = np.array(x0_plant).flatten()
            Xhist_freq[:, sim + 1] = abs_to_freq(x0_plant)
            Uhist[sim + 1] = u_opt

            U0 = np.hstack([U0[:, 1:], U0[:, -1:]])

            # Telegram progress notification every 10%
            pct = int(100 * (sim + 1) / n_steps)
            if pct in notify_milestones and pct > last_pct_notified:
                last_pct_notified = pct
                elapsed = time.time() - t_start
                y_now = np.array(x0_plant).flatten()
                notifier.notify_progress(sim + 1, n_steps, elapsed,
                    extra_info=f"u_a={u_opt:.3f}\nN_tumor={np.sum(y_now):.0f}")

        t_elapsed = time.time() - t_start
        print(f"\nSimulation completed in {t_elapsed:.1f} seconds")

        # ============================================================
        # 6. RESULTS: KPIs AND PLOTS
        # ============================================================
        t_vec = np.arange(0, Hs + dt, dt)
        N_total_mpc = np.sum(Xhist_abs, axis=0)
        frac_res = Xhist_abs[2, :] / N_total_mpc
        frac_sens = np.sum(Xhist_abs[:2, :], axis=0) / N_total_mpc

        kpi1_idx = np.where(frac_res > frac_sens)[0]
        kpi1 = kpi1_idx[0] * dt if len(kpi1_idx) > 0 else None

        kpi2_idx = np.where(Xhist_abs[0, :] > y0_abs[0] * 2.5)[0]
        kpi2 = kpi2_idx[0] * dt if len(kpi2_idx) > 0 else None

        kpi3_idx = np.where(Xhist_abs[1, :] > y0_abs[1] * 2.5)[0]
        kpi3 = kpi3_idx[0] * dt if len(kpi3_idx) > 0 else None

        drug_exposure = np.mean(Uhist)

        print("\n" + "=" * 60)
        print("  RESULTS: Replicator-MPC (Single-Agent) V3")
        print("=" * 60)
        print(f"  KPI-1 (T- dominance):   {f'day {kpi1}' if kpi1 else 'not reached'}")
        print(f"  KPI-2 (T+ > 250% x0):  {f'day {kpi2}' if kpi2 else 'not reached'}")
        print(f"  KPI-3 (TP > 250% x0):  {f'day {kpi3}' if kpi3 else 'not reached'}")
        print(f"  Drug exposure (mean):    {drug_exposure:.4f}")
        print(f"  Computation time:        {t_elapsed:.1f}s")
        print(f"  Hp={Hp}, N_max={N_max}")
        print(f"  Open-loop NRMSE:         {total_nrmse*100:.2f}%")

        # Plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Replicator-MPC SA V3 | Hp={Hp} | $N_{{max}}$={N_max:.0e} | Time={t_elapsed:.1f}s',
                     fontsize=15)

        ax = axes[0, 0]
        ax.plot(t_vec, Xhist_abs[0, :], 'b-', lw=2, label='T+')
        ax.plot(t_vec, Xhist_abs[1, :], 'g-', lw=2, label='TP')
        ax.plot(t_vec, Xhist_abs[2, :], 'r-', lw=2, label='T-')
        ax.set_xlabel('Time (days)', fontsize=FSIZE_AX)
        ax.set_ylabel('Population', fontsize=FSIZE_AX)
        ax.set_title('Cell Dynamics (Plant - LV)', fontsize=FSIZE_AX)
        ax.legend(fontsize=11)
        ax.tick_params(labelsize=FSIZE_AX)
        ax.grid(True)

        ax = axes[0, 1]
        ax.plot(t_vec, Xhist_freq[1, :], 'b-', lw=2, label='$x_{T+}$')
        ax.plot(t_vec, Xhist_freq[2, :], 'g-', lw=2, label='$x_{TP}$')
        ax.plot(t_vec, Xhist_freq[3, :], 'r-', lw=2, label='$x_{T-}$')
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
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='center right')

        ax = axes[1, 0]
        ax.plot(t_vec, Uhist, 'b.', ms=3)
        ax.set_xlabel('Time (days)', fontsize=FSIZE_AX)
        ax.set_ylabel('Abiraterone dose', fontsize=FSIZE_AX)
        ax.set_title('Treatment Schedule', fontsize=FSIZE_AX)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=FSIZE_AX)
        ax.grid(True)

        ax = axes[1, 1]
        ax.plot(t_vec, N_total_mpc, 'purple', lw=2, label='$N_{tumor}$ (plant)')
        N_from_zf = N_max * (1 - Xhist_freq[0, :])
        ax.plot(t_vec, N_from_zf, 'purple', lw=1.5, ls='--', label='$N_{max}(1-x_0)$')
        ax.set_xlabel('Time (days)', fontsize=FSIZE_AX)
        ax.set_ylabel('Total population', fontsize=FSIZE_AX)
        ax.set_title('Tumor Burden', fontsize=FSIZE_AX)
        ax.legend(fontsize=11)
        ax.tick_params(labelsize=FSIZE_AX)
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, f'Replicator_MPC_SA_V3_Hp{Hp}.png'), dpi=150)
        plt.close()

        # Save numerical results for post-processing
        results_file = os.path.join(DATA_DIR, f'Results_SA_V3_Hp{Hp}.npz')
        np.savez(results_file,
                 Xhist_abs=Xhist_abs, Xhist_freq=Xhist_freq,
                 Uhist=Uhist, t_vec=t_vec, y0_abs=y0_abs,
                 Hp=Hp, Hs=Hs, dt=dt, N_max=N_max, t_elapsed=t_elapsed,
                 w_sens=w_sens, w_drug=w_drug, w_res=w_res,
                 kpi1=kpi1 if kpi1 is not None else -1,
                 kpi2=kpi2 if kpi2 is not None else -1,
                 kpi3=kpi3 if kpi3 is not None else -1,
                 drug_exposure=drug_exposure,
                 open_loop_nrmse=total_nrmse)
        print(f"\nResults saved: {results_file}")
        print(f"Figures saved: {IMG_DIR}/Validation_SA_V3_*.png, Replicator_MPC_SA_V3_Hp{Hp}.png")

        # Telegram: completion with KPIs
        notifier.notify_complete("Replicator-MPC SA V3", t_elapsed, {
            "KPI-1 (T- dom)": f"day {kpi1}" if kpi1 else "not reached",
            "KPI-2 (T+>250%)": f"day {kpi2}" if kpi2 else "not reached",
            "KPI-3 (TP>250%)": f"day {kpi3}" if kpi3 else "not reached",
            "Drug exposure": f"{drug_exposure:.4f}",
            "OL NRMSE": f"{total_nrmse*100:.2f}%",
        }, {
            "Hp": Hp, "Hs": Hs, "dt": dt, "N_max": N_max,
        })

    except Exception as e:
        tb = traceback.format_exc()
        print(f"\nERROR: {e}\n{tb}")
        notifier.notify_error(type(e).__name__, str(e), tb)
        sys.exit(1)
