"""
Integrator_Ablation_SA.py -- RK4 vs Euler ablation for the single-agent Rep-MPC

REPRODUCES IN THE THESIS
    Euler against RK4 ablation for the single-agent scenario: quantifies how much of the
    difference the integrator explains, Section 6.4.3.

PROVENANCE
    Copied verbatim from the working tree of the thesis. The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md at the
    repository root.

--------------------------------------------------------------------------
PURPOSE
    Quantify, in isolation, how much of the Rep-MPC result depends on the
    discretization used for the *prediction model* inside the NLP.

    Context: the two nominal closed-loop runs (MPC_Replicator_SA_V3.py /
    MPC_Replicator_DA_V3.py) discretize the replicator predictor with forward
    Euler, whereas every sensitivity sweep (Sensitivity_SA_V3.py,
    Sensitivity_DA_T4.py, T5A/T5B, SS20_BwSweep) uses RK4. The LV-MPC baseline
    (TJ/Cratos_Code/MPC_Code_Art2_*) uses Euler. The sensitivity comparisons are
    therefore Euler-baseline vs RK4-Rep-MPC, and this script measures whether the
    integrator alone accounts for any meaningful part of the reported gains.

METHOD
    Everything is imported from Sensitivity_SA_V3 so the model, the plant loop,
    the KPI definitions, the weights, the initial condition and u_prev=0.0 are
    bit-for-bit identical. The ONLY thing this script changes is the integrator
    used to build the NLP dynamics constraint. IPOPT options are held fixed at
    max_iter=1000 for both arms.

    Note the plant is always propagated with Euler (x0 += dt*F_LV(x0,u)), exactly
    as in the sweeps; only the internal predictor changes.

USAGE
    Run from the directory that holds Sensitivity_SA_V3.py (on cratos that is
    ~/Code/Replicator), with the conda environment active:

        conda activate BD_Doc
        cd ~/Code/Replicator
        bash lanzar_ablation.sh          # one process per Hp, in parallel

    Or a single Hp by hand:
        nohup python -u Integrator_Ablation_SA.py --hp 20 > logs_abl/Hp20.log 2>&1 &

    Or everything sequentially in one process:
        nohup python -u Integrator_Ablation_SA.py --hp 20,50,80 > logs_abl/all.log 2>&1 &

    Optional flags:
        --sim N          gamma config index, 1-5 (default 1 = Baseline)
        --integrators X  comma list, default "rk4,euler"
        --threads N      BLAS/OpenMP threads (default 1, see note above)
        --no-telegram    disable notifications

OUTPUT
    Integrator_Ablation_SA/Datos/Ablation_SA_Sim<N>_Hp<H>_<INT>.npz
    Integrator_Ablation_SA/Ablation_Summary.csv
    A comparison table printed at the end of the run.

COST ESTIMATE (from the stored RK4 sweep times for Sim1)
    Hp=20 ~ 230 s RK4  + ~80 s Euler
    Hp=50 ~ 3300 s RK4 + ~1100 s Euler
    Hp=80 ~ 6145 s RK4 + ~2050 s Euler
    => ~2.3 h wall-clock if the three Hp run in parallel.

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

import os
import sys

# Pin BLAS/OpenMP to a single thread BEFORE importing numpy/casadi.
# Two reasons: (i) it prevents IPOPT/MUMPS from over-subscribing cores when
# several Hp run in parallel, same rationale as lanzar_hp100_paralelo.sh;
# (ii) this script measures wall-clock ratios between two integrators, and
# variable thread counts would contaminate that measurement. Override with
# --threads N only if a deliberately multi-threaded timing is wanted.
_THREADS = '1'
for _i, _a in enumerate(sys.argv):
    if _a.startswith('--threads='):
        _THREADS = _a.split('=', 1)[1]
    elif _a == '--threads' and _i + 1 < len(sys.argv):
        _THREADS = sys.argv[_i + 1]
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = _THREADS

import csv
import time
import numpy as np
import casadi as cs

# Reuse the EXACT model, plant loop and KPI definitions from the sweep.
import Sensitivity_SA_V3 as SW

OUTPUT_DIR = "Integrator_Ablation_SA"
DATA_DIR = os.path.join(OUTPUT_DIR, "Datos")
SKIP_EXISTING = True


# ----------------------------------------------------------------------
# Argument parsing (same lightweight style as the other scripts)
# ----------------------------------------------------------------------
def _arg(flag, default=None):
    for i, a in enumerate(sys.argv):
        if a.startswith(flag + '='):
            return a.split('=', 1)[1]
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


HP_LIST = [int(s) for s in _arg('--hp', '20,50,80').split(',')]
SIM_IDX = int(_arg('--sim', '1'))
INTEGRATORS = [s.strip().lower() for s in _arg('--integrators', 'rk4,euler').split(',')]
USE_TELEGRAM = '--no-telegram' not in sys.argv


# ----------------------------------------------------------------------
# Solver builder -- identical to SW.build_sa_solver except the integrator
# ----------------------------------------------------------------------
def build_sa_solver_ablation(Hp, w_sens, w_drug, w_res, integrator):
    """Build the SA NLP with a selectable integrator for the predictor.

    integrator = 'rk4'   -> classic 4th-order Runge-Kutta  (as in the sweeps)
    integrator = 'euler' -> forward Euler                  (as in the nominal
                            runs and in the LV-MPC baseline)
    Everything else, objective, constraints, bounds, IPOPT options, matches
    Sensitivity_SA_V3.build_sa_solver exactly.
    """
    U = cs.SX.sym('U', 1, Hp)
    X = cs.SX.sym('X', SW.n_total, Hp + 1)
    n_par = SW.n_total + 1                      # state (4) + u_prev (1)
    P = cs.SX.sym('P', n_par)

    obj = 0
    X[:, 0] = P[:SW.n_total]
    dt = SW.dt

    for t in range(Hp):
        st = X[:, t]
        con = U[:, t]

        if integrator == 'rk4':
            k1 = SW.F_REP(st, con)
            k2 = SW.F_REP(st + dt / 2 * k1, con)
            k3 = SW.F_REP(st + dt / 2 * k2, con)
            k4 = SW.F_REP(st + dt * k3, con)
            X[:, t + 1] = st + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        elif integrator == 'euler':
            X[:, t + 1] = st + dt * SW.F_REP(st, con)
        else:
            raise ValueError(f"unknown integrator '{integrator}'")

        obj += w_sens * (X[1, t + 1] ** 2 + X[2, t + 1] ** 2)
        obj += w_drug * (con[0] ** 2)
        obj += w_res * X[3, t + 1]

    g = []
    for t in range(1, Hp + 1):
        for i in range(SW.n_total):
            g = cs.vertcat(g, -X[i, t])
            g = cs.vertcat(g, X[i, t] - 1.0)
        g = cs.vertcat(g, -U[:, t - 1])
        g = cs.vertcat(g, U[:, t - 1] - 1.0)
        if t != 1:
            g = cs.vertcat(g, -(U[:, t - 1] - U[:, t - 2]) - SW.B_rate)
            g = cs.vertcat(g, (U[:, t - 1] - U[:, t - 2]) - SW.B_rate)
        else:
            g = cs.vertcat(g, -(P[SW.n_total] - U[:, t - 1]) - SW.B_rate)
            g = cs.vertcat(g, (P[SW.n_total] - U[:, t - 1]) - SW.B_rate)

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


# ----------------------------------------------------------------------
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    cfg = SW.GAMMA_CONFIGS[SIM_IDX - 1]
    w_sens = cfg['gamma_1'] * SW.N_max ** 2
    w_drug = cfg['gamma_2']
    w_res = cfg['gamma_3'] * SW.N_max

    notifier = None
    if USE_TELEGRAM:
        try:
            from telegram_notifier import TelegramNotifier
            notifier = TelegramNotifier()
        except Exception as e:
            print(f"  [Telegram] disabled ({e})")

    print("=" * 64)
    print("  INTEGRATOR ABLATION -- Single-Agent Replicator-MPC")
    print("  RK4 vs forward Euler for the NLP prediction model")
    print("=" * 64)
    print(f"  Config      : {cfg['name']} (gamma_2={cfg['gamma_2']})")
    print(f"  Hp values   : {HP_LIST}")
    print(f"  Integrators : {INTEGRATORS}")
    print(f"  Fixed       : N_max={SW.N_max:.0e}, dt={SW.dt}, Hs={SW.Hs}, "
          f"u_prev=0.0, max_iter=1000")
    print(f"  Plant       : forward Euler (unchanged, as in every sweep)")
    print("=" * 64)

    if notifier:
        try:
            notifier.notify_start("Integrator Ablation SA (RK4 vs Euler)", {
                'Config': cfg['name'],
                'Hp': str(HP_LIST),
                'Integrators': str(INTEGRATORS),
            })
        except Exception:
            pass

    results = {}
    total = len(HP_LIST) * len(INTEGRATORS)
    idx = 0

    for Hp in HP_LIST:
        for integ in INTEGRATORS:
            idx += 1
            tag = f"Sim{SIM_IDX}_Hp{Hp}_{integ.upper()}"
            npz_path = os.path.join(DATA_DIR, f"Ablation_SA_{tag}.npz")

            if SKIP_EXISTING and os.path.exists(npz_path):
                print(f"[{idx}/{total}] SKIP {tag} (already exists)")
                d = np.load(npz_path, allow_pickle=True)
                results[(Hp, integ)] = {
                    'kpi1': int(d['kpi1']), 'kpi2': int(d['kpi2']),
                    'kpi3': int(d['kpi3']),
                    'drug_exposure': float(d['drug_exposure']),
                    't_elapsed': float(d['t_elapsed']),
                }
                continue

            print(f"\n[{idx}/{total}] RUN  {tag}")
            t_build = time.time()
            solver, n_g = build_sa_solver_ablation(Hp, w_sens, w_drug, w_res, integ)
            print(f"     NLP built in {time.time() - t_build:.1f}s "
                  f"({n_g} constraints)")

            res = SW.run_sa_simulation(solver, Hp, n_g, label=tag,
                                       notifier=notifier,
                                       config_idx=idx, total_configs=total)

            np.savez_compressed(
                npz_path,
                Xhist_abs=res['Xhist_abs'], Xhist_freq=res['Xhist_freq'],
                Uhist=res['Uhist'], t_elapsed=res['t_elapsed'],
                kpi1=res['kpi1'], kpi2=res['kpi2'], kpi3=res['kpi3'],
                drug_exposure=res['drug_exposure'],
                Hp=Hp, Hs=SW.Hs, dt=SW.dt, N_max=SW.N_max,
                integrator=integ, u_prev_init=0.0, max_iter=1000,
                gamma_1=cfg['gamma_1'], gamma_2=cfg['gamma_2'],
                gamma_3=cfg['gamma_3'], name=cfg['name'],
                y0_abs=SW.y0_abs,
            )
            results[(Hp, integ)] = res
            print(f"     KPI-1={res['kpi1']}  KPI-3={res['kpi3']}  "
                  f"u_a={res['drug_exposure']:.4f}  t={res['t_elapsed']:.1f}s")

    # ------------------------------------------------------------------
    # Comparison table + CSV
    # ------------------------------------------------------------------
    def pct(a, b):
        if a in (None, -1) or b in (None, -1) or a == 0:
            return None
        return (b - a) / a * 100.0

    print("\n" + "=" * 88)
    print("  ABLATION SUMMARY -- effect of the predictor integrator alone")
    print("=" * 88)
    hdr = (f"{'Hp':>5} | {'KPI-1 rk4':>10} {'euler':>8} {'d%':>8} | "
           f"{'KPI-3 rk4':>10} {'euler':>8} {'d%':>8} | "
           f"{'u_a rk4':>8} {'euler':>8} {'d%':>8} | {'t rk4':>9} {'t euler':>9}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for Hp in HP_LIST:
        r = results.get((Hp, 'rk4'))
        e = results.get((Hp, 'euler'))
        if not r or not e:
            continue
        d1, d3 = pct(r['kpi1'], e['kpi1']), pct(r['kpi3'], e['kpi3'])
        du = pct(r['drug_exposure'], e['drug_exposure'])
        f = lambda v: f"{v:+8.1f}" if v is not None else "     n/a"
        print(f"{Hp:>5} | {r['kpi1']:>10} {e['kpi1']:>8} {f(d1)} | "
              f"{r['kpi3']:>10} {e['kpi3']:>8} {f(d3)} | "
              f"{r['drug_exposure']:>8.4f} {e['drug_exposure']:>8.4f} {f(du)} | "
              f"{r['t_elapsed']:>9.1f} {e['t_elapsed']:>9.1f}")
        rows.append({
            'Hp': Hp,
            'kpi1_rk4': r['kpi1'], 'kpi1_euler': e['kpi1'], 'kpi1_delta_pct': d1,
            'kpi3_rk4': r['kpi3'], 'kpi3_euler': e['kpi3'], 'kpi3_delta_pct': d3,
            'drug_rk4': r['drug_exposure'], 'drug_euler': e['drug_exposure'],
            'drug_delta_pct': du,
            't_rk4': r['t_elapsed'], 't_euler': e['t_elapsed'],
            'speedup_rk4_over_euler': (r['t_elapsed'] / e['t_elapsed']
                                       if e['t_elapsed'] else None),
        })

    if rows:
        csv_path = os.path.join(OUTPUT_DIR, "Ablation_Summary.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            if write_header:
                w.writeheader()
            w.writerows(rows)
        print(f"\n  CSV -> {csv_path}")

        sp = [r['speedup_rk4_over_euler'] for r in rows
              if r['speedup_rk4_over_euler']]
        if sp:
            print(f"  Mean RK4/Euler wall-clock ratio: {np.mean(sp):.2f}x")

    print("=" * 88)

    if notifier:
        try:
            notifier.notify_complete("Integrator Ablation SA finished", {
                'Configs': str(len(rows)),
                'Hp': str(HP_LIST),
            })
        except Exception:
            pass


if __name__ == "__main__":
    main()
