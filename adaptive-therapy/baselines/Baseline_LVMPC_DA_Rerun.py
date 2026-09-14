"""
Baseline_LVMPC_DA_Rerun.py -- re-corrida instrumentada del baseline LV-MPC dual-agent

REPRODUCES IN THE THESIS
    LV-MPC dual-agent baseline column of Table 6.5 (tab:rep_comparison).

PROVENANCE
    Copied verbatim from the working tree of the thesis
    (Modelos_Replicator/Codigos/Baseline_LVMPC_DA_Rerun.py). The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md
    at the repository root for the complete list of differences.

--------------------------------------------------------------------------
POR QUE
    La Tabla 3 del articulo CBM (y la tabla equivalente del Cap. 6 del libro) da,
    para el baseline LV-MPC dual-agent, KPI-1a = 8012 y KPI-1b = 9126 junto con
    exposiciones u_a ~ 0.19 y u_d ~ 0.17, y de ahi sale el "81% less drug".

    Los KPI 8012/9126 corresponden al **experimento 19** (Hp=140, Bw=80) de
    `TJ/Cratos_Code/MPC_Code_Art2_EB_CO_V2.py`. Pero:
      - el reporte que genero esa corrida
        (`TJ/Reportes/Reporte_Opti_DM_MPC_AS_HP140_B80.txt`) NO contiene ninguna
        linea de exposicion de farmaco, aunque el script las calcula;
      - la fila SS 19 del Cap. 5 (`5_Prostate_Treatment.tex` L730) da
        u_a = 0.3057 y u_d = 0.2846, que NO son 0.19/0.17;
      - el tiempo de computo que usa la Tabla 3 (8523 s) sale del **experimento 14**
        (`TJ/Reportes/Results.xlsx`, Hp=140 con barrido de Bw), o sea de OTRA
        configuracion de pesos.

    Es decir, la columna del baseline esta armada con datos de mas de una corrida y
    el 0.19/0.17 no se ha podido reproducir. Este script re-corre el experimento 19
    exactamente como estaba y registra TODO lo que faltaba.

QUE CAMBIA RESPECTO AL ORIGINAL
    NADA del modelo, de la funcion objetivo, de las restricciones ni del solver.
    El bucle de simulacion es identico. Solo se anade instrumentacion:
      1. `time.time()` alrededor del bucle -> tiempo de computo de ESTA corrida.
      2. Guardado de `Uhist`, `Xhist` y KPIs en un `.npz`.
      3. Calculo de las exposiciones medias, que el original computaba pero cuyo
         `file.write` final quedo truncado.
      4. Exposiciones acumuladas parciales, para poder ver si 0.19/0.17 corresponde
         a una ventana temporal distinta de los 14000 dias completos.
      5. `matplotlib` en backend Agg (sin `plt.show()`), imprescindible para correr
         con nohup en un servidor sin display.
      6. Hilos BLAS/OpenMP fijados a 1, como en `lanzar_hp100_paralelo.sh`.

    Las figuras del original se omiten: no aportan a esta verificacion y ahorran
    tiempo. Los KPI se calculan con las MISMAS formulas del original.

USO (en cratos, tras entrar por ssh)
    conda activate BD_Doc
    cd ~/Code/Replicator
    nohup python -u Baseline_LVMPC_DA_Rerun.py --exp 19 > logs_base/exp19.log 2>&1 &

    --exp N   experimento a correr, 15..19 (por defecto 19, el de la Tabla 3)
    --hs N    horizonte de simulacion en dias (por defecto 14000, el original)

COSTE ESPERADO
    El exp 14 a Hp=140, Bw=80 tardo 8523 s (~2.4 h). El exp 19 deberia estar en el
    mismo orden. Sin contencion en cratos, probablemente menos.

SALIDA
    Baseline_LVMPC_DA/Datos/Baseline_DA_Exp<N>_Hp140_Bw80.npz
    Baseline_LVMPC_DA/Baseline_Summary.csv
    y un resumen impreso al final del log.

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

# Hilos a 1 ANTES de importar numpy/casadi (mismo criterio que lanzar_hp100_paralelo.sh)
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'

import csv
import time
import numpy as np
import casadi as cs
import matplotlib
matplotlib.use('Agg')          # sin display: obligatorio bajo nohup
from tqdm import tqdm

OUTPUT_DIR = "Baseline_LVMPC_DA"
DATA_DIR = os.path.join(OUTPUT_DIR, "Datos")


def _arg(flag, default):
    for i, a in enumerate(sys.argv):
        if a.startswith(flag + '='):
            return a.split('=', 1)[1]
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


EXP = int(_arg('--exp', '19'))        # 15..19  (k = EXP - 15)
HS = int(_arg('--hs', '14000'))
k = EXP - 15
assert 0 <= k <= 4, "--exp debe estar entre 15 y 19"

os.makedirs(DATA_DIR, exist_ok=True)

# Notificaciones opcionales (si telegram_notifier.py no esta, sigue sin avisar)
notifier = None
if '--no-telegram' not in sys.argv:
    try:
        from telegram_notifier import TelegramNotifier
        notifier = TelegramNotifier()
    except Exception as _e:
        print(f"  [Telegram] desactivado ({_e})")

# =====================================================================
# MODELO Y PARAMETROS -- copia literal de MPC_Code_Art2_EB_CO_V2.py
# =====================================================================
r = cs.DM([2.78e-3, 3.55e-3, 6.65e-3, 6.65e-3])
delta = 0.05
A = cs.DM([[1, 0.7, 0.8, 8 - (2 * delta)],
           [0.4, 1, 0.5, 5 - (2 * delta)],
           [0.6, 0.9, 1, 1 - (2 * delta)],
           [0.6 - delta, 0.9 - delta, 1 - delta, 1]])
x0 = cs.DM([2062, 2617, 3.97e-11, 3.97e-12])
d_v = 5e-4
mu = 1.5
dt = 1
Hp = 140
Q = cs.diagcat(1, 1, .5, .1)
R = cs.diagcat(.3, 1)
Xs = cs.DM([3000, 3000, 1800])
n_Eps = Xs.numel()
Hs = HS
umin = [0, 0]; umax = [1, 1]
xmin = 0; xmax = 10e3
B = [0.05, 0.2]
Block_V = 80

y = cs.SX.sym('y', 4)
n_states = y.numel()
d = cs.SX.sym('d', 2)
n_controls = d.numel()

K = [(mu - d[0]) * y[1], 1e4 / (1 * (10 ** (2 * d[0]))), 1e4, 6e3]

rhs = cs.vertcat(
    (1 - (cs.dot(A[0, :], y.T) / K[0])) * r[0] * y[0] * (1 - (1.5 * d[1])) - d_v * y[0],
    (1 - (cs.dot(A[1, :], y.T) / K[1])) * r[1] * y[1] * (1 - (1.5 * d[1])) - d_v * y[1],
    (1 - (cs.dot(A[2, :], y.T) / K[2])) * r[2] * y[2] * (1 - (1.5 * d[1])) - d_v * y[2],
    (1 - (cs.dot(A[3, :], y.T) / K[3])) * r[3] * y[3] - d_v * y[3])

F_Y = cs.Function('F_Y', [y, d], [rhs])

U = cs.SX.sym('U', n_controls, Hp)
X = cs.SX.sym('X', n_states, (Hp + 1))
P = cs.SX.sym('P', 6)
E = cs.SX.sym('E', n_Eps, Hp)

obj = 0
X[:, 0] = P[:4]

# --- forward Euler, exactamente como el original (el bloque RK4 esta comentado alli)
for t in range(Hp):
    st = X[:, t]; con = U[:, t]
    f_value = F_Y(st, con)
    X[:, t + 1] = st + (dt * f_value)
    if k == 0:
        obj += 0.1 * (con.T @ R @ con) + ((E[:, t].T @ cs.diagcat(1, 1, 1) @ E[:, t])) \
               + (((X[:, t + 1]).T @ Q @ (X[:, t + 1])) / 1e4 ** 2)
    elif k == 1:
        obj += 0.1 * (con.T @ R @ con) + 0.65 * ((E[:, t].T @ cs.diagcat(1, 1, 1) @ E[:, t])) \
               + 0.25 * (((X[:, t + 1]).T @ Q @ (X[:, t + 1])) / 1e4 ** 2)
    elif k == 2:
        if t == 0:
            obj += 0.1 * (con.T @ R @ con) + ((E[:, t].T @ cs.diagcat(1, 1, 1) @ E[:, t])) + ((
                ((X[0, t] - X[1, t + 1]) ** 2) + ((X[0, t] - X[2, t + 1]) ** 2) +
                ((X[0, t] - X[3, t + 1]) ** 2) + ((X[1, t] - X[2, t + 1]) ** 2) +
                ((X[1, t] - X[3, t + 1]) ** 2) + ((X[2, t] - X[3, t + 1]) ** 2)) / 1e4 ** 2)
        else:
            obj += 0.1 * (con.T @ R @ con) + ((E[:, t].T @ cs.diagcat(1, 1, 1) @ E[:, t])) + ((
                ((((cs.mmax(X[0, :t + 1]) - cs.mmin(X[0, :t + 1])) / 2) - X[1, t + 1]) ** 2) +
                ((((cs.mmax(X[0, :t + 1]) - cs.mmin(X[0, :t + 1])) / 2) - X[2, t + 1]) ** 2) +
                ((((cs.mmax(X[0, :t + 1]) - cs.mmin(X[0, :t + 1])) / 2) - X[3, t + 1]) ** 2) +
                ((((cs.mmax(X[1, :t + 1]) - cs.mmin(X[1, :t + 1])) / 2) - X[2, t + 1]) ** 2) +
                ((((cs.mmax(X[1, :t + 1]) - cs.mmin(X[1, :t + 1])) / 2) - X[3, t + 1]) ** 2) +
                ((((cs.mmax(X[2, :t + 1]) - cs.mmin(X[2, :t + 1])) / 2) - X[3, t + 1]) ** 2)) / 1e4 ** 2)
    elif k == 3:
        if t == 0:
            obj += 0.1 * (con.T @ R @ con) + 0.65 * ((E[:, t].T @ cs.diagcat(1, 1, 1) @ E[:, t])) + 0.25 * ((
                ((X[0, t] - X[1, t + 1]) ** 2) + ((X[0, t] - X[2, t + 1]) ** 2) +
                ((X[0, t] - X[3, t + 1]) ** 2) + ((X[1, t] - X[2, t + 1]) ** 2) +
                ((X[1, t] - X[3, t + 1]) ** 2) + ((X[2, t] - X[3, t + 1]) ** 2)) / 1e4 ** 2)
        else:
            obj += 0.1 * (con.T @ R @ con) + 0.65 * ((E[:, t].T @ cs.diagcat(1, 1, 1) @ E[:, t])) + 0.25 * ((
                ((((cs.mmax(X[0, :t + 1]) - cs.mmin(X[0, :t + 1])) / 2) - X[1, t + 1]) ** 2) +
                ((((cs.mmax(X[0, :t + 1]) - cs.mmin(X[0, :t + 1])) / 2) - X[2, t + 1]) ** 2) +
                ((((cs.mmax(X[0, :t + 1]) - cs.mmin(X[0, :t + 1])) / 2) - X[3, t + 1]) ** 2) +
                ((((cs.mmax(X[1, :t + 1]) - cs.mmin(X[1, :t + 1])) / 2) - X[2, t + 1]) ** 2) +
                ((((cs.mmax(X[1, :t + 1]) - cs.mmin(X[1, :t + 1])) / 2) - X[3, t + 1]) ** 2) +
                ((((cs.mmax(X[2, :t + 1]) - cs.mmin(X[2, :t + 1])) / 2) - X[3, t + 1]) ** 2)) / 1e4 ** 2)
    elif k == 4:
        if t != 0:
            obj += 0.1 * (con.T @ R @ con) + .45 * ((E[:, t].T @ cs.diagcat(1, 1, 1) @ E[:, t])) \
                   + .25 * (((X[:, t + 1]).T @ Q @ (X[:, t + 1])) / 1e4 ** 2) \
                   + .2 * ((U[:, t] - U[:, t - 1]).T @ R @ (U[:, t] - U[:, t - 1]))
        else:
            obj += 0.1 * (con.T @ R @ con) + .45 * ((E[:, t].T @ cs.diagcat(1, 1, 1) @ E[:, t])) \
                   + .25 * (((X[:, t + 1]).T @ Q @ (X[:, t + 1])) / 1e4 ** 2)

ff = cs.Function('ff', [U, P], [X])

g = []
for t in range(1, Hp + 1):
    g = cs.vertcat(g, -X[:, t] + xmin)
    g = cs.vertcat(g, X[:, t] - xmax)
    g = cs.vertcat(g, -U[:, t - 1] + umin)
    g = cs.vertcat(g, U[:, t - 1] - umax)
    if t != 1:
        g = cs.vertcat(g, -(U[:, t - 1] - U[0, t - 2]) - B)
        g = cs.vertcat(g, (U[:, t - 1] - U[0, t - 2]) - B)
    else:
        g = cs.vertcat(g, -(P[4:] - U[:, t - 1]) - B)
        g = cs.vertcat(g, (P[4:] - U[:, t - 1]) - B)
    g = cs.vertcat(g, -E[:, t - 1])
    g = cs.vertcat(g, X[:n_Eps, t] - Xs - E[:, t - 1])

OPT_variables = cs.vertcat(cs.reshape(U, n_controls * (Hp), 1),
                           cs.reshape(E, n_Eps * (Hp), 1))
nlp_prob = {'f': obj, 'x': OPT_variables, 'g': g, 'p': P}
opts = {'ipopt': {'max_iter': 200, 'print_level': 0,
                  'acceptable_tol': 1e-8, 'acceptable_obj_change_tol': 1e-6},
        'print_time': 0}
solver = cs.nlpsol('solver', 'ipopt', nlp_prob, opts)

u0 = 0.6
U0 = u0 * cs.DM.ones((n_controls, Hp))
E0 = .5 * cs.DM.ones((n_Eps, Hp))
Xhist = x0
Uhist = U0[:, 0]
lbg = [-float('inf')] * g.shape[0]
ubg = [0] * (g.shape[0])

print("=" * 70)
print("  BASELINE LV-MPC dual-agent -- re-corrida instrumentada")
print("=" * 70)
print(f"  Experimento : {EXP}  (k={k})")
print(f"  Hp={Hp}  Bw={Block_V}  dt={dt}  Hs={Hs}  u(-1)=[{u0}, {u0}]")
print(f"  Integrador  : forward Euler (identico al original)")
print(f"  IPOPT       : max_iter=200, hilos BLAS/OpenMP = 1")
print("=" * 70)

# =====================================================================
# BUCLE DE SIMULACION -- identico al original, con cronometro
# =====================================================================
if notifier:
    try:
        notifier.notify_start("Baseline LV-MPC DA re-run (instrumentado)", {
            'Experimento': str(EXP), 'Hp': str(Hp), 'Bw': str(Block_V),
            'Hs': str(Hs), 'Integrador': 'Euler'})
    except Exception:
        pass

t_start = time.time()
_milestones = {25, 50, 75, 100}
_last_pct = 0

for sim in tqdm(range(int(Hs / dt)), desc=f"Exp{EXP}"):
    if sim % Block_V == 0:
        if sim > 0:
            U0 = UT
        Par = cs.vertcat(x0, U0[:, 0])
        sol = solver(lbg=lbg, ubg=ubg, p=Par,
                     x0=cs.vertcat(cs.reshape(U0, n_controls * (Hp), 1),
                                   cs.reshape(E0, n_Eps * (Hp), 1)))
        U0 = sol['x'][:n_controls * Hp].reshape((n_controls, Hp)).full()
        E0 = sol['x'][n_controls * Hp:].reshape((n_Eps, Hp)).full()
    Uhist = np.hstack((Uhist, np.reshape(U0[:, 0], (n_controls, 1))))
    x0 = x0 + (dt * F_Y(x0, U0[:, 0]))
    Xhist = np.hstack((Xhist, x0))
    if sim % Block_V == 0:
        UT = np.hstack((U0[:, 1:], np.reshape(U0[:, -1], (n_controls, 1))))
        E0 = np.hstack((E0[:, 1:], np.reshape(E0[:, -1], (n_Eps, 1))))

    # progreso por Telegram cada 25%
    if notifier:
        _pct = int(100 * (sim + 1) / int(Hs / dt))
        if _pct in _milestones and _pct > _last_pct:
            _last_pct = _pct
            try:
                notifier.notify_progress(sim + 1, int(Hs / dt), time.time() - t_start,
                                         extra_info=f"Exp {EXP} | dia {sim + 1}/{Hs}")
            except Exception:
                pass

t_elapsed = time.time() - t_start

# =====================================================================
# KPIs -- mismas formulas del original
# =====================================================================
Xh = np.array(Xhist)
Uh = np.array(Uhist)

KPI_N_1 = Xh[2, :] / np.sum(Xh, 0)
KPI_N_2 = Xh[3, :] / np.sum(Xh, 0)
KPI_D = np.sum(Xh[:2, :], 0) / np.sum(Xh, 0)

idx1 = np.where(KPI_N_1 > KPI_D)[0]
idx2 = np.where(KPI_N_2 > KPI_D)[0]
idx3 = np.where(Xh[0, :] > 2062 * 2.5)[0]
idx4 = np.where(Xh[1, :] > 2617 * 2.5)[0]

kpi1a = int(idx1[0]) if len(idx1) else -1
kpi1b = int(idx2[0]) if len(idx2) else -1
kpi2 = int(idx3[0]) if len(idx3) else -1
kpi3 = int(idx4[0]) if len(idx4) else -1

# --- exposiciones medias sobre TODO el horizonte (formula del original)
u_a = float(np.sum(Uh[0, :]) / len(Uh[0, :]))
u_d = float(np.sum(Uh[1, :]) / len(Uh[0, :]))

print()
print("=" * 70)
print("  RESULTADOS")
print("=" * 70)
print(f"  KPI-1a = {kpi1a:6d}   (Tabla 3 dice 8012 para el exp 19)")
print(f"  KPI-1b = {kpi1b:6d}   (Tabla 3 dice 9126 para el exp 19)")
print(f"  KPI-2  = {kpi2:6d}    KPI-3 = {kpi3:6d}")
print()
print(f"  u_a (media sobre {Hs} dias) = {u_a:.4f}   (Tabla 3 dice ~0.19)")
print(f"  u_d (media sobre {Hs} dias) = {u_d:.4f}   (Tabla 3 dice ~0.17)")
print(f"  u total                      = {u_a + u_d:.4f}   (Tabla 3 dice ~0.36)")
print()
print(f"  Tiempo de computo = {t_elapsed:.1f} s   (Tabla 3 usa 8523 s, del exp 14)")

# --- medias acumuladas parciales: para ver si 0.19/0.17 corresponde a otra ventana
print()
print("  Media acumulada por ventana temporal:")
print(f"    {'dias':>7} {'u_a':>8} {'u_d':>8} {'total':>8}")
for lim in (2000, 4000, 6000, 8000, 10000, 12000, 14000, 20000, 30000):
    if lim > Uh.shape[1]:
        continue
    a = float(np.mean(Uh[0, :lim])); b = float(np.mean(Uh[1, :lim]))
    print(f"    {lim:>7} {a:>8.4f} {b:>8.4f} {a + b:>8.4f}")

# --- reduccion frente al Rep-MPC (valores fijos de Results_DA_V3_Hp140.npz)
REP_A, REP_D, REP_T = 0.045466744185262795, 0.021371537424514926, 541.0107147693634
tot = u_a + u_d
print()
print("  Frente al Rep-MPC (u_a=0.0455, u_d=0.0214, total=0.0668, t=541 s):")
print(f"    reduccion de farmaco = {(tot - (REP_A + REP_D)) / tot * 100:.1f}%   "
      f"(el articulo afirma 81%)")
print(f"    reduccion de tiempo  = {(t_elapsed - REP_T) / t_elapsed * 100:.1f}%   "
      f"(el articulo afirma 94%)")
print("=" * 70)

# =====================================================================
# GUARDADO
# =====================================================================
npz = os.path.join(DATA_DIR, f"Baseline_DA_Exp{EXP}_Hp140_Bw80.npz")
np.savez_compressed(
    npz, Xhist=Xh, Uhist=Uh, t_elapsed=t_elapsed,
    kpi1a=kpi1a, kpi1b=kpi1b, kpi2=kpi2, kpi3=kpi3,
    drug_abi=u_a, drug_doc=u_d, drug_total=tot,
    Hp=Hp, Bw=Block_V, dt=dt, Hs=Hs, exp=EXP, u_prev_init=u0,
    integrator='euler', max_iter=200)
print(f"\n  npz -> {npz}")

csv_path = os.path.join(OUTPUT_DIR, "Baseline_Summary.csv")
row = {'exp': EXP, 'Hp': Hp, 'Bw': Block_V, 'Hs': Hs,
       'kpi1a': kpi1a, 'kpi1b': kpi1b, 'kpi2': kpi2, 'kpi3': kpi3,
       'u_a': u_a, 'u_d': u_d, 'u_total': tot, 't_elapsed': t_elapsed}
write_header = not os.path.exists(csv_path)
with open(csv_path, 'a', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(row.keys()))
    if write_header:
        w.writeheader()
    w.writerow(row)
print(f"  csv -> {csv_path}")

if notifier:
    try:
        notifier.notify_complete("Baseline LV-MPC DA re-run terminado", {
            'Experimento': str(EXP),
            'KPI-1a': str(kpi1a), 'KPI-1b': str(kpi1b),
            'u_a': f"{u_a:.4f}", 'u_d': f"{u_d:.4f}", 'u_total': f"{tot:.4f}",
            'Tiempo (s)': f"{t_elapsed:.1f}",
            'Reduccion farmaco': f"{(tot - (REP_A + REP_D)) / tot * 100:.1f}%"})
    except Exception:
        pass
