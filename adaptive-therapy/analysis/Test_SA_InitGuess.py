"""
Test_SA_InitGuess.py - Test hypothesis: u_prev=0.0 fixes Hp>=80 divergence

REPRODUCES IN THE THESIS
    Tests the hypothesis that u(-1) = 0 removes the divergence seen at H_p >= 80.
    Supports the choice of previous input documented in Section 6.4.3.

PROVENANCE
    Copied verbatim from the working tree of the thesis. The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md at the
    repository root.

--------------------------------------------------------------------------
Test_SA_InitGuess.py - Test hypothesis: u_prev=0.0 fixes Hp>=80 divergence
==========================================================================
Quick test: runs Sim1_Baseline with Hp=80 using u_prev=0.0 (instead of 1.0).

Hypothesis:
  The initial guess u_prev=1.0 + rate constraint B_rate=0.05 traps IPOPT
  in a local minimum where drug=1.0 for the entire horizon. Changing to
  u_prev=0.0 should allow the optimizer to find the adaptive therapy solution.

Expected outcome if hypothesis is correct:
  - Drug exposure ~0.25-0.35 (similar to Hp=50 baseline)
  - KPI-1 ~300-500 days (not 200)
  - Computation time >> 6s (solver actually working)

Expected outcome if hypothesis is wrong:
  - Drug still ~1.0, KPI-1 ~200 -> problem is deeper (scaling, NLP structure)

Usage:
  python Test_SA_InitGuess.py              # Run on server or local
  python Test_SA_InitGuess.py --also-50    # Also run Hp=50 for comparison

Based on: Sensitivity_SA_V2.py
Author: Johan Pena-Campos, PhD Project
Date: 2026-03-23
"""

import casadi as cs
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import sys
import os

# Force unbuffered output
print = lambda *args, **kwargs: __builtins__.__dict__['print'](*args, **{**kwargs, 'flush': True})

def DM2Arr(dm):
    return np.array(dm.full())


# ================================================================
# CONFIGURATION
# ================================================================

OUTPUT_DIR = "Test_SA_InitGuess"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALSO_HP50 = '--also-50' in sys.argv

# Fixed parameters (same as Sensitivity_SA_V2.py)
N_max = 8e4
dt = 10
Hs = 4000
B_rate = 0.05
y0_abs = np.array([560.36, 747.59, 47.10])

# Baseline gamma (Sim1)
gamma_1 = 3.6e-7
gamma_2 = 0.1
gamma_3 = 1.6e-3
w_sens = gamma_1 * N_max**2   # 2304.0
w_drug = gamma_2              # 0.1
w_res  = gamma_3 * N_max      # 128.0

# Test configs
TEST_CONFIGS = [
    {'Hp': 80, 'u_init': 0.0, 'label': 'Hp80_u0'},
    {'Hp': 80, 'u_init': 0.5, 'label': 'Hp80_u05'},
    {'Hp': 80, 'u_init': 1.0, 'label': 'Hp80_u1_CONTROL'},
]

if ALSO_HP50:
    TEST_CONFIGS.append({'Hp': 50, 'u_init': 0.0, 'label': 'Hp50_u0_REF'})
    TEST_CONFIGS.append({'Hp': 50, 'u_init': 1.0, 'label': 'Hp50_u1_REF'})


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
# SOLVER (RK4 + max_iter=1000, same as Sensitivity_SA_V2)
# ================================================================

def build_sa_solver(Hp):
    """Build CasADi NLP solver for SA with RK4 integration."""
    U = cs.SX.sym('U', 1, Hp)
    X = cs.SX.sym('X', n_total, Hp + 1)
    n_par = n_total + 1  # state (4) + u_prev (1)
    P = cs.SX.sym('P', n_par)

    obj = 0
    X[:, 0] = P[:n_total]

    for t in range(Hp):
        st = X[:, t]
        con = U[:, t]
        # RK4 integration
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


# ================================================================
# SIMULATION (with configurable u_init)
# ================================================================

def run_sa_simulation(solver, Hp, n_g, u_init, label=""):
    """Run SA MPC simulation with configurable initial guess."""
    n_steps = int(Hs / dt)

    x0 = cs.DM(y0_abs)
    u_prev = u_init                                   # <-- KEY CHANGE: configurable
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
# PLOT
# ================================================================

def generate_plot(res, cfg, t_vec):
    """Generate 2x2 overview plot."""
    Xhist_abs = res['Xhist_abs']
    Xhist_freq = res['Xhist_freq']
    Uhist = res['Uhist']
    N_total_mpc = np.sum(Xhist_abs, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Test InitGuess | Hp={cfg['Hp']} | u_init={cfg['u_init']} | "
        f"Drug={res['drug_exposure']:.4f} | Time={res['t_elapsed']:.1f}s | "
        f"KPI-1={'day '+str(res['kpi1']) if res['kpi1']>=0 else 'NR'}",
        fontsize=13)

    # [0,0] Cell Dynamics
    ax = axes[0, 0]
    ax.plot(t_vec, Xhist_abs[0, :], 'b-', lw=2, label='T+')
    ax.plot(t_vec, Xhist_abs[1, :], 'g-', lw=2, label='TP')
    ax.plot(t_vec, Xhist_abs[2, :], 'r-', lw=2, label='T-')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Population')
    ax.set_title('Cell Dynamics (Plant - LV)')
    ax.legend()
    ax.grid(True)

    # [0,1] Frequencies
    ax = axes[0, 1]
    ax.plot(t_vec, Xhist_freq[0, :], 'k--', lw=1.5, label='$x_0$ (fict)')
    ax.plot(t_vec, Xhist_freq[1, :], 'b-', lw=2, label='$x_{T+}$')
    ax.plot(t_vec, Xhist_freq[2, :], 'g-', lw=2, label='$x_{TP}$')
    ax.plot(t_vec, Xhist_freq[3, :], 'r-', lw=2, label='$x_{T-}$')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Frequency')
    ax.set_title('Population Frequencies')
    ax.legend()
    ax.grid(True)
    ax.set_ylim(-0.05, 1.05)

    # [1,0] Treatment
    ax = axes[1, 0]
    ax.plot(t_vec, Uhist, 'b.', ms=3)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Abiraterone dose')
    ax.set_title('Treatment Schedule')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)

    # [1,1] Tumor Burden
    ax = axes[1, 1]
    ax.plot(t_vec, N_total_mpc, 'purple', lw=2, label='$N_{tumor}$ (plant)')
    N_from_zf = N_max * (1 - Xhist_freq[0, :])
    ax.plot(t_vec, N_from_zf, 'purple', lw=1.5, ls='--', label='$N_{max}(1-x_0)$')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Total population')
    ax.set_title('Tumor Burden')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"Test_{cfg['label']}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {plot_path}")


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":

    print("=" * 70)
    print("  TEST: Initial Guess Hypothesis for Hp>=80 Divergence")
    print("  Hypothesis: u_prev=0.0 fixes Hp>=80 (u_prev=1.0 causes local min)")
    print("=" * 70)
    print(f"  Configs to test: {len(TEST_CONFIGS)}")
    print(f"  Baseline: gamma=({gamma_1:.1e}, {gamma_2}, {gamma_3:.1e})")
    print(f"  w_sens={w_sens:.1f}, w_drug={w_drug:.2f}, w_res={w_res:.1f}")
    print(f"  Output: {OUTPUT_DIR}/")
    if ALSO_HP50:
        print("  Including Hp=50 reference configs (--also-50)")
    print()

    t_total_start = time.time()
    results = []

    for ci, cfg in enumerate(TEST_CONFIGS):
        print(f"\n{'='*60}")
        print(f"  [{ci+1}/{len(TEST_CONFIGS)}] {cfg['label']}")
        print(f"  Hp={cfg['Hp']}, u_init={cfg['u_init']}")
        print(f"{'='*60}")

        # Build solver (only rebuild if Hp changes)
        solver, n_g = build_sa_solver(cfg['Hp'])

        # Run simulation
        res = run_sa_simulation(
            solver, cfg['Hp'], n_g,
            u_init=cfg['u_init'],
            label=cfg['label']
        )

        # Print results
        kpi1_str = f"day {res['kpi1']}" if res['kpi1'] >= 0 else "NOT reached"
        kpi3_str = f"day {res['kpi3']}" if res['kpi3'] >= 0 else "NOT reached"
        print(f"  KPI-1 (T- dom):   {kpi1_str}")
        print(f"  KPI-3 (TP>250%):  {kpi3_str}")
        print(f"  Drug exposure:    {res['drug_exposure']:.4f}")
        print(f"  Time:             {res['t_elapsed']:.1f}s")

        # Verdict
        if res['drug_exposure'] < 0.95:
            print(f"  >>> PASS: Drug < 0.95 -- adaptive therapy found!")
        else:
            print(f"  >>> FAIL: Drug >= 0.95 -- still saturated")

        # Generate plot
        t_vec = np.arange(0, Hs + dt, dt)
        generate_plot(res, cfg, t_vec)

        results.append({'config': cfg, **res})

    t_total = time.time() - t_total_start

    # ---- Summary ----
    print(f"\n{'='*70}")
    print(f"  SUMMARY -- Initial Guess Hypothesis Test")
    print(f"{'='*70}")
    print(f"  {'Label':<22} {'Hp':>4} {'u_init':>7} {'KPI-1':>8} {'Drug':>8} {'Time':>8}")
    print(f"  {'-'*62}")

    any_pass = False
    for r in results:
        c = r['config']
        kpi1_str = str(r['kpi1']) if r['kpi1'] >= 0 else 'NR'
        verdict = 'PASS' if r['drug_exposure'] < 0.95 else 'FAIL'
        if r['drug_exposure'] < 0.95:
            any_pass = True
        print(f"  {c['label']:<22} {c['Hp']:>4} {c['u_init']:>7.1f} {kpi1_str:>8} "
              f"{r['drug_exposure']:>8.4f} {r['t_elapsed']:>7.1f}s  [{verdict}]")

    print(f"\n  Total time: {t_total:.1f}s ({t_total/60:.1f} min)")

    if any_pass:
        print(f"\n  HYPOTHESIS CONFIRMED: u_init=0.0 (or 0.5) fixes Hp>=80!")
        print(f"  Next step: update Sensitivity_SA_V2.py line 247: u_prev = 0.0")
        print(f"  Then re-run full sensitivity with Sensitivity_SA_V3.py")
    else:
        print(f"\n  HYPOTHESIS REJECTED: initial guess is NOT the root cause.")
        print(f"  Need to investigate: NLP scaling, constraint structure, or solver options.")

    print()
