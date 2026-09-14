"""
Calibracion_Nmax_OpenLoop.py - N_max Sweep for Replicator-LV Open-Loop Calibration

REPRODUCES IN THE THESIS
    Calibration of N_max by open-loop NRMSE, Section 6.5.1 (subsec:rep_nmax_calibration).

PROVENANCE
    Copied verbatim from the working tree of the thesis
    (Modelos_Replicator/Codigos/Calibracion_Nmax_OpenLoop.py). The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md
    at the repository root for the complete list of differences.

--------------------------------------------------------------------------
Sweeps N_max values and compares Lotka-Volterra vs Replicator+ZF open-loop dynamics
(u=0) to find the optimal N_max that minimizes discrepancy.

Both SA (Cunningham, 3 pop) and DA (West, 4 pop) cases.

Mathematical basis:
  LV scaled:       dp_i/dt = p_i * f_i
  Replicator+ZF:   dp_i/dt = p_i * (f_i - phi_bar)
  Difference:      -p_i * phi_bar, where phi_bar = sum(p_j * f_j)
  When N_max >> N_total: p_j small => phi_bar small => Replicator ~ LV

Date: 2026-02-25

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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys
import traceback

# Telegram notifications
from telegram_notifier import MPCNotifier

# Force unbuffered output
print = lambda *args, **kwargs: __builtins__.__dict__['print'](*args, **{**kwargs, 'flush': True})

FSIZE_AX = 14
FSIZE_LG = 11


# ================================================================
# 1. PARAMETER DEFINITIONS
# ================================================================

def get_params_SA():
    """Cunningham single-agent (3 populations, Abiraterone only)."""
    return {
        'name': 'SA_Cunningham',
        'n_pop': 3,
        'n_controls': 1,
        'r_lv': [2.7726e-3, 3.4657e-3, 6.6542e-3],
        'A_lv': [[1, 0.7, 0.8],
                  [0.4, 1, 0.6],
                  [0.5, 0.9, 1]],
        'mu_param': 1.5,
        'd_v': 0.0,
        'y0_abs': [560.36, 747.59, 47.10],
        'dt': 10,
        'T_val': 4000,
        'labels': ['$T^+$', '$T^P$', '$T^-$'],
        'colors': ['blue', 'green', 'red'],
        'has_docetaxel': False,
    }


def get_params_DA():
    """West dual-agent (4 populations, Abiraterone + Docetaxel)."""
    delta = 0.05
    return {
        'name': 'DA_West',
        'n_pop': 4,
        'n_controls': 2,
        'r_lv': [2.78e-3, 3.55e-3, 6.65e-3, 6.65e-3],
        'A_lv': [[1,           0.7,         0.8,         0.8 - 2*delta],
                  [0.4,         1,           0.5,         0.5 - 2*delta],
                  [0.6,         0.9,         1,           1   - 2*delta],
                  [0.6 - delta, 0.9 - delta, 1 - delta,   1]],
        'mu_param': 1.5,
        'd_v': 5e-4,
        'y0_abs': [2062.0, 2617.0, 3.97e-11, 3.97e-12],
        'dt': 1,
        'T_val': 5000,
        'labels': ['$T^+$', '$T^P$', '$T^{-/+}$', '$T^{-/-}$'],
        'colors': ['blue', 'green', 'red', 'cyan'],
        'has_docetaxel': True,
    }


# ================================================================
# 2. MODEL BUILDERS
# ================================================================

def build_LV(params):
    """Build CasADi LV plant function F_LV(y, u) -> dy/dt."""
    n = params['n_pop']
    nc = params['n_controls']
    r = cs.DM(params['r_lv'])
    A = cs.DM(params['A_lv'])
    mu = params['mu_param']
    dv = params['d_v']

    y = cs.SX.sym('y', n)
    u = cs.SX.sym('u', nc)

    u_a = u[0] if nc >= 1 else 0
    u_d = u[1] if nc >= 2 else 0

    if n == 3:  # SA
        K = cs.vertcat(
            (mu - u_a) * y[1],
            1e4 - 9.9e3 * u_a,
            1e4
        )
        rhs = cs.vertcat(*[
            float(r[i]) * y[i] * (1 - cs.dot(A[i, :], y.T) / K[i])
            for i in range(3)
        ])
    else:  # DA (4 pop)
        K = cs.vertcat(
            (mu - u_a) * y[1],
            1e4 / (1 * (10**(2 * u_a))),
            1e4,
            6e3
        )
        rhs_list = []
        for i in range(4):
            growth = float(r[i]) * y[i] * (1 - cs.dot(A[i, :], y.T) / K[i])
            if i < 3:
                growth = growth * (1 - 1.5 * u_d)
            rhs_list.append(growth - dv * y[i])
        rhs = cs.vertcat(*rhs_list)

    return cs.Function('F_LV', [y, u], [rhs])


def build_REP(params, N_max):
    """Build CasADi Replicator+ZF function F_REP(p, u) -> dp/dt."""
    n = params['n_pop']
    nc = params['n_controls']
    n_total = n + 1
    r = cs.DM(params['r_lv'])
    A = cs.DM(params['A_lv'])
    mu = params['mu_param']
    dv = params['d_v']

    p = cs.SX.sym('p', n_total)
    u = cs.SX.sym('u', nc)

    u_a = u[0] if nc >= 1 else 0
    u_d = u[1] if nc >= 2 else 0

    y_from_p = N_max * p[1:]

    if n == 3:  # SA
        K = cs.vertcat(
            (mu - u_a) * cs.fmax(y_from_p[1], 1e-10),
            1e4 - 9.9e3 * u_a,
            1e4
        )
        f_real = cs.SX.zeros(n)
        for i in range(n):
            f_real[i] = float(r[i]) * (1 - cs.dot(A[i, :], y_from_p.T) / K[i])
    else:  # DA (4 pop)
        K = cs.vertcat(
            (mu - u_a) * cs.fmax(y_from_p[1], 1e-10),
            1e4 / (1 * (10**(2 * u_a))),
            1e4,
            6e3
        )
        f_real = cs.SX.zeros(n)
        for i in range(n):
            growth = float(r[i]) * (1 - cs.dot(A[i, :], y_from_p.T) / K[i])
            if i < 3:
                growth = growth * (1 - 1.5 * u_d)
            f_real[i] = growth - dv

    f_all = cs.vertcat(0, f_real)
    phi_bar = cs.dot(p, f_all)
    rhs = p * (f_all - phi_bar)

    return cs.Function('F_REP', [p, u], [rhs])


# ================================================================
# 3. SIMULATION FUNCTIONS
# ================================================================

def simulate_LV(params, F_LV):
    """Simulate LV open-loop (u=0) for T_val days."""
    n = params['n_pop']
    nc = params['n_controls']
    dt = params['dt']
    T = params['T_val']
    n_steps = int(T / dt)
    y0 = np.array(params['y0_abs'])
    u_zero = cs.DM.zeros(nc)

    y = np.zeros((n, n_steps + 1))
    y[:, 0] = y0
    for k in range(n_steps):
        dy = F_LV(cs.DM(y[:, k]), u_zero)
        y[:, k + 1] = y[:, k] + dt * np.array(dy).flatten()

    return y


def simulate_REP(params, F_REP, N_max):
    """Simulate Replicator+ZF open-loop (u=0) for T_val days."""
    n = params['n_pop']
    nc = params['n_controls']
    n_total = n + 1
    dt = params['dt']
    T = params['T_val']
    n_steps = int(T / dt)
    y0 = np.array(params['y0_abs'])
    u_zero = cs.DM.zeros(nc)

    # Convert IC
    p_real = y0 / N_max
    p_0 = max(1.0 - np.sum(p_real), 0.0)
    p0 = np.concatenate([[p_0], p_real])

    p = np.zeros((n_total, n_steps + 1))
    p[:, 0] = p0
    for k in range(n_steps):
        dp = F_REP(cs.DM(p[:, k]), u_zero)
        p[:, k + 1] = p[:, k] + dt * np.array(dp).flatten()
        p[:, k + 1] = np.maximum(p[:, k + 1], 0)
        s = np.sum(p[:, k + 1])
        if s > 0:
            p[:, k + 1] /= s

    return p


# ================================================================
# 4. METRICS
# ================================================================

def compute_metrics(y_lv, p_rep, N_max, params):
    """Compute calibration metrics between LV and Replicator trajectories."""
    n = params['n_pop']
    n_steps = y_lv.shape[1]

    # Replicator back to absolute
    y_rep_abs = N_max * p_rep[1:, :]

    # LV frequencies
    N_lv = np.sum(y_lv, axis=0)
    N_lv_safe = np.where(N_lv > 0, N_lv, 1.0)
    freq_lv = y_lv / N_lv_safe

    # Replicator real frequencies (excluding x_0)
    x0 = p_rep[0, :]
    denom = np.where((1 - x0) > 1e-12, 1 - x0, 1e-12)
    freq_rep = p_rep[1:, :] / denom

    # N from Replicator
    N_rep = N_max * (1 - x0)

    metrics = {}

    # Per-population absolute RMSE and NRMSE
    rmse_abs = np.zeros(n)
    nrmse_abs = np.zeros(n)
    max_abs = np.zeros(n)
    rmse_freq = np.zeros(n)
    max_freq = np.zeros(n)

    for i in range(n):
        diff_abs = y_lv[i, :] - y_rep_abs[i, :]
        rmse_abs[i] = np.sqrt(np.mean(diff_abs**2))
        rng = np.max(y_lv[i, :]) - np.min(y_lv[i, :])
        nrmse_abs[i] = rmse_abs[i] / rng if rng > 0 else 0.0
        max_abs[i] = np.max(np.abs(diff_abs))

        diff_freq = freq_lv[i, :] - freq_rep[i, :]
        rmse_freq[i] = np.sqrt(np.mean(diff_freq**2))
        max_freq[i] = np.max(np.abs(diff_freq))

    # Total population
    diff_N = N_lv - N_rep
    rmse_N = np.sqrt(np.mean(diff_N**2))
    max_rel_N = np.max(np.abs(diff_N) / N_lv_safe)

    # phi_bar magnitude (from Replicator)
    # phi_bar = sum(p_j * f_j) -- approximate as (N_rep_dot) / N_max
    # For simplicity, use the NRMSE as aggregate
    total_nrmse = np.mean(nrmse_abs)

    metrics['rmse_abs'] = rmse_abs
    metrics['nrmse_abs'] = nrmse_abs
    metrics['max_abs'] = max_abs
    metrics['rmse_freq'] = rmse_freq
    metrics['max_freq'] = max_freq
    metrics['rmse_N'] = rmse_N
    metrics['max_rel_N'] = max_rel_N
    metrics['total_nrmse'] = total_nrmse
    metrics['N_lv_max'] = np.max(N_lv)

    return metrics


# ================================================================
# 5. SWEEP AND PLOTTING
# ================================================================

def run_sweep(params, N_max_values):
    """Run N_max sweep for one scenario."""
    name = params['name']
    n = params['n_pop']
    print(f"\n{'='*60}")
    print(f"  CALIBRATION SWEEP: {name}")
    print(f"  N_max values: {len(N_max_values)}")
    print(f"{'='*60}")

    # Build LV (independent of N_max)
    F_LV = build_LV(params)
    y_lv = simulate_LV(params, F_LV)
    N_lv_max = np.max(np.sum(y_lv, axis=0))
    print(f"  LV simulated: N_total_max = {N_lv_max:.0f}")

    all_metrics = []
    valid_Nmax = []

    for N_max in N_max_values:
        if N_max < N_lv_max * 0.5:
            print(f"  N_max={N_max:.0e}: SKIP (< 0.5 * N_lv_max={N_lv_max:.0f})")
            continue

        F_REP = build_REP(params, N_max)
        p_rep = simulate_REP(params, F_REP, N_max)

        # Check for NaN
        if np.any(np.isnan(p_rep)):
            print(f"  N_max={N_max:.0e}: NaN detected, skipping")
            continue

        m = compute_metrics(y_lv, p_rep, N_max, params)
        all_metrics.append(m)
        valid_Nmax.append(N_max)

        ratio = np.sum(np.array(params['y0_abs'])) / N_max * 100
        print(f"  N_max={N_max:.0e}: NRMSE={m['total_nrmse']:.4f} "
              f"({m['total_nrmse']*100:.1f}%), "
              f"max_abs={np.max(m['max_abs']):.0f}, "
              f"ratio_0={ratio:.1f}%")

    if not all_metrics:
        print("  ERROR: No valid N_max values!")
        return None

    valid_Nmax = np.array(valid_Nmax)
    nrmse_arr = np.array([m['total_nrmse'] for m in all_metrics])

    # Find optimal: smallest N_max with NRMSE < 5%
    threshold = 0.05
    below = np.where(nrmse_arr < threshold)[0]
    if len(below) > 0:
        opt_idx = below[0]
    else:
        opt_idx = np.argmin(nrmse_arr)
    opt_Nmax = valid_Nmax[opt_idx]
    opt_nrmse = nrmse_arr[opt_idx]

    print(f"\n  >>> OPTIMAL N_max = {opt_Nmax:.0e} (NRMSE = {opt_nrmse*100:.2f}%)")

    # ---- PLOTS ----
    dt = params['dt']
    T = params['T_val']
    t_vec = np.arange(0, T + dt, dt)

    # Plot 1: Calibration curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogx(valid_Nmax, nrmse_arr * 100, 'bo-', lw=2, ms=8)
    ax.axhline(y=threshold * 100, color='r', ls='--', lw=1.5, label=f'Threshold ({threshold*100:.0f}%)')
    ax.axvline(x=opt_Nmax, color='g', ls='--', lw=1.5, label=f'Optimal: {opt_Nmax:.0e}')
    ax.set_xlabel('$N_{max}$', fontsize=FSIZE_AX)
    ax.set_ylabel('NRMSE (%)', fontsize=FSIZE_AX)
    ax.set_title(f'Calibration Curve: {name}', fontsize=FSIZE_AX)
    ax.legend(fontsize=FSIZE_LG)
    ax.tick_params(labelsize=FSIZE_AX)
    ax.grid(True, which='both', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'Calibracion_{name}_curve.png', dpi=150)
    plt.close()

    # Plot 2: Best-match absolute comparison
    F_REP_opt = build_REP(params, opt_Nmax)
    p_rep_opt = simulate_REP(params, F_REP_opt, opt_Nmax)
    y_rep_opt = opt_Nmax * p_rep_opt[1:, :]

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for i in range(n):
        ax = axes[i]
        ax.plot(t_vec, y_lv[i, :], '-', color=params['colors'][i], lw=2,
                label=f'LV')
        ax.plot(t_vec, y_rep_opt[i, :], '--', color=params['colors'][i], lw=2,
                label=f'Rep ($N_{{max}}$={opt_Nmax:.0e})')
        ax.set_xlabel('Time (days)', fontsize=FSIZE_AX)
        ax.set_ylabel('Population', fontsize=FSIZE_AX)
        ax.set_title(f'{params["labels"][i]}', fontsize=FSIZE_AX)
        ax.legend(fontsize=FSIZE_LG)
        ax.tick_params(labelsize=FSIZE_AX)
        ax.grid(True)
    plt.suptitle(f'Absolute Comparison at $N_{{max}}$={opt_Nmax:.0e} | {name}',
                 fontsize=FSIZE_AX + 1)
    plt.tight_layout()
    plt.savefig(f'Calibracion_{name}_absolute.png', dpi=150)
    plt.close()

    # Plot 3: Best-match frequency comparison (corrected)
    N_lv_total = np.sum(y_lv, axis=0)
    N_lv_safe = np.where(N_lv_total > 0, N_lv_total, 1.0)
    freq_lv = y_lv / N_lv_safe
    x0_opt = p_rep_opt[0, :]
    denom = np.where((1 - x0_opt) > 1e-12, 1 - x0_opt, 1e-12)
    freq_rep = p_rep_opt[1:, :] / denom

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for i in range(n):
        ax = axes[i]
        ax.plot(t_vec, freq_lv[i, :], '-', color=params['colors'][i], lw=2,
                label='LV freq')
        ax.plot(t_vec, freq_rep[i, :], '--', color=params['colors'][i], lw=2,
                label='Rep freq (corrected)')
        ax.set_xlabel('Time (days)', fontsize=FSIZE_AX)
        ax.set_ylabel('Frequency', fontsize=FSIZE_AX)
        ax.set_title(f'{params["labels"][i]}', fontsize=FSIZE_AX)
        ax.legend(fontsize=FSIZE_LG)
        ax.tick_params(labelsize=FSIZE_AX)
        ax.grid(True)
    plt.suptitle(f'Frequency Comparison (corrected) at $N_{{max}}$={opt_Nmax:.0e} | {name}',
                 fontsize=FSIZE_AX + 1)
    plt.tight_layout()
    plt.savefig(f'Calibracion_{name}_frequency.png', dpi=150)
    plt.close()

    # Plot 4: Total population comparison
    N_rep_opt = opt_Nmax * (1 - x0_opt)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t_vec, N_lv_total, 'b-', lw=2, label='$N_{LV}$')
    ax.plot(t_vec, N_rep_opt, 'r--', lw=2, label=f'$N_{{max}}(1-x_0)$, $N_{{max}}$={opt_Nmax:.0e}')
    ax.set_xlabel('Time (days)', fontsize=FSIZE_AX)
    ax.set_ylabel('Total population', fontsize=FSIZE_AX)
    ax.set_title(f'Total Population Comparison | {name}', fontsize=FSIZE_AX)
    ax.legend(fontsize=FSIZE_LG)
    ax.tick_params(labelsize=FSIZE_AX)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f'Calibracion_{name}_Ntotal.png', dpi=150)
    plt.close()

    # Plot 5: Multi-N_max overlay (pick 5 representative values)
    n_show = min(5, len(valid_Nmax))
    indices = np.linspace(0, len(valid_Nmax) - 1, n_show, dtype=int)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, n_show))
    for idx_j, j in enumerate(indices):
        Nm = valid_Nmax[j]
        F_R = build_REP(params, Nm)
        p_r = simulate_REP(params, F_R, Nm)
        y_r = Nm * p_r[1:, :]
        for i in range(n):
            axes[i].plot(t_vec, y_r[i, :], '--', color=cmap[idx_j], lw=1.5,
                         label=f'$N_{{max}}$={Nm:.0e}')
    for i in range(n):
        axes[i].plot(t_vec, y_lv[i, :], 'k-', lw=2.5, label='LV (reference)')
        axes[i].set_xlabel('Time (days)', fontsize=FSIZE_AX)
        axes[i].set_ylabel('Population', fontsize=FSIZE_AX)
        axes[i].set_title(f'{params["labels"][i]}', fontsize=FSIZE_AX)
        axes[i].legend(fontsize=9, loc='best')
        axes[i].tick_params(labelsize=FSIZE_AX)
        axes[i].grid(True)
    plt.suptitle(f'Multi-$N_{{max}}$ Overlay | {name}', fontsize=FSIZE_AX + 1)
    plt.tight_layout()
    plt.savefig(f'Calibracion_{name}_overlay.png', dpi=150)
    plt.close()

    return {
        'valid_Nmax': valid_Nmax,
        'metrics': all_metrics,
        'nrmse_arr': nrmse_arr,
        'opt_Nmax': opt_Nmax,
        'opt_nrmse': opt_nrmse,
        'opt_idx': opt_idx,
        'y_lv': y_lv,
        'N_lv_max': N_lv_max,
    }


# ================================================================
# 6. MAIN
# ================================================================

if __name__ == "__main__":

    notifier = MPCNotifier()

    try:
        t_global = time.time()
        notifier.notify_start("Calibracion N_max Open-Loop", {
            "Escenarios": "SA + DA",
            "Objetivo": "Match Replicator open-loop con LV",
        })

        # ---- SA CALIBRATION ----
        params_SA = get_params_SA()
        N_max_SA = [5e3, 7.5e3, 1e4, 1.5e4, 2e4, 3e4, 5e4,
                    7.5e4, 1e5, 2e5, 5e5, 1e6]
        results_SA = run_sweep(params_SA, N_max_SA)

        # ---- DA CALIBRATION ----
        params_DA = get_params_DA()
        N_max_DA = [5e3, 1e4, 1.5e4, 2e4, 3e4, 5e4,
                    7.5e4, 1e5, 2e5, 5e5, 1e6]
        results_DA = run_sweep(params_DA, N_max_DA)

        t_total = time.time() - t_global

        # ---- PRINT SUMMARY ----
        print("\n" + "=" * 60)
        print("  CALIBRATION SUMMARY")
        print("=" * 60)

        if results_SA:
            print(f"\n  SA ({params_SA['name']}):")
            print(f"    N_lv_max = {results_SA['N_lv_max']:.0f}")
            print(f"    Optimal N_max = {results_SA['opt_Nmax']:.0e}")
            print(f"    NRMSE = {results_SA['opt_nrmse']*100:.2f}%")
            print(f"    All NRMSE: {', '.join(f'{v*100:.1f}%' for v in results_SA['nrmse_arr'])}")

        if results_DA:
            print(f"\n  DA ({params_DA['name']}):")
            print(f"    N_lv_max = {results_DA['N_lv_max']:.0f}")
            print(f"    Optimal N_max = {results_DA['opt_Nmax']:.0e}")
            print(f"    NRMSE = {results_DA['opt_nrmse']*100:.2f}%")
            print(f"    All NRMSE: {', '.join(f'{v*100:.1f}%' for v in results_DA['nrmse_arr'])}")

        print(f"\n  Total time: {t_total:.1f}s")

        # ---- SAVE ----
        save_data = {'t_total': t_total}
        if results_SA:
            save_data.update({
                'SA_Nmax_values': results_SA['valid_Nmax'],
                'SA_nrmse': results_SA['nrmse_arr'],
                'SA_opt_Nmax': results_SA['opt_Nmax'],
                'SA_opt_nrmse': results_SA['opt_nrmse'],
                'SA_N_lv_max': results_SA['N_lv_max'],
            })
        if results_DA:
            save_data.update({
                'DA_Nmax_values': results_DA['valid_Nmax'],
                'DA_nrmse': results_DA['nrmse_arr'],
                'DA_opt_Nmax': results_DA['opt_Nmax'],
                'DA_opt_nrmse': results_DA['opt_nrmse'],
                'DA_N_lv_max': results_DA['N_lv_max'],
            })

        np.savez('Calibracion_Nmax_Results.npz', **save_data)
        print("\nResults saved: Calibracion_Nmax_Results.npz")
        print("Figures saved: Calibracion_*_curve.png, *_absolute.png, *_frequency.png, *_Ntotal.png, *_overlay.png")

        # Telegram
        msg = "CALIBRACION COMPLETADA\n"
        if results_SA:
            msg += f"SA: N_max_opt={results_SA['opt_Nmax']:.0e}, NRMSE={results_SA['opt_nrmse']*100:.2f}%\n"
        if results_DA:
            msg += f"DA: N_max_opt={results_DA['opt_Nmax']:.0e}, NRMSE={results_DA['opt_nrmse']*100:.2f}%\n"
        msg += f"Tiempo: {t_total:.1f}s"
        notifier.notify_complete("Calibracion N_max", t_total, {
            "SA_opt": f"{results_SA['opt_Nmax']:.0e}" if results_SA else "N/A",
            "DA_opt": f"{results_DA['opt_Nmax']:.0e}" if results_DA else "N/A",
        }, {})

    except Exception as e:
        tb = traceback.format_exc()
        print(f"\nERROR: {e}\n{tb}")
        notifier.notify_error(type(e).__name__, str(e), tb)
        sys.exit(1)
