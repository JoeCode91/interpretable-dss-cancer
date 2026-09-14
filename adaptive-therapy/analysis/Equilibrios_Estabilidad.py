"""
=========================================================================

REPRODUCES IN THE THESIS
    Equilibria and stability analysis: Table tab:rep_equilibria, Section 6.6.4.

PROVENANCE
    Copied verbatim from the working tree of the thesis
    (Modelos_Replicator/Codigos/Equilibrios_Estabilidad.py). The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md
    at the repository root for the complete list of differences.

--------------------------------------------------------------------------
 Equilibrios y estabilidad de los modelos LV y replicador  --  R1-7(c)
=========================================================================

 Responde al punto R1-7(c) del informe del Revisor 1: la tesis justifica la
 terapia adaptativa diciendo que conduce a un equilibrio estable de
 coexistencia, pero no calcula los equilibrios de ninguno de los dos modelos
 ni analiza su estabilidad. R1 pregunta cuales son, y donde se situa el punto
 de operacion del controlador respecto del de coexistencia.

 QUE HACE
 --------
 Para los cuatro sistemas del Cap. 6 -- planta LV y modelo replicador, en sus
 versiones mono-agente (Cunningham, 3 poblaciones) y dual-agente (West, 4
 poblaciones) -- y para una rejilla de dosis CONSTANTES:

   1. Enumera los equilibrios. No basta con buscar el interior: en un sistema
      Lotka-Volterra la mayoria de los equilibrios viven en las caras del
      ortante (subconjuntos de poblaciones extintas), y son precisamente esos
      los que describen la perdida de una poblacion bajo tratamiento. El
      script recorre los 2^n subconjuntos de soporte, resuelve el sistema
      reducido en cada cara con arranques multiples, y descarta duplicados.

   2. Evalua el jacobiano COMPLETO en cada equilibrio, no el de la cara. Esta
      es la parte que importa: los autovalores transversales a la cara dicen
      si una poblacion ausente puede reinvadir, que es la pregunta biologica.

   3. Clasifica cada equilibrio por el signo de la parte real de sus
      autovalores: estable, inestable, o silla.

   4. Si se le pasa un fichero de resultados de lazo cerrado, localiza el
      estado final del controlador y lo compara con cada equilibrio, para
      responder la segunda mitad de la pregunta de R1.

 COMO SE CORRE
 -------------
   python Equilibrios_Estabilidad.py

 Opciones (todas opcionales):
   --sistema {todos,lv_sa,lv_da,rep_sa,rep_da}   por defecto: todos
   --dosis-sa  0 0.5 1.0        dosis de Abiraterona a evaluar (mono-agente)
   --dosis-da  0 0.5 1.0        idem para el dual-agente
   --dosis-doc 0 0.5            dosis de Docetaxel (solo dual-agente)
   --resultados RUTA.npz        fichero de lazo cerrado para comparar
   --salida CARPETA             por defecto: Equilibrios_Resultados

 Ejemplo con comparacion contra el lazo cerrado:
   python Equilibrios_Estabilidad.py --resultados Results_DA_V3_Hp140.npz

 DEPENDENCIAS: numpy, scipy. (No usa CasADi: el jacobiano se obtiene por
 diferencias centradas de alta precision, que para 3 y 4 estados es exacto a
 efectos practicos y evita arrastrar el solver simbolico.)

 SALIDAS: un CSV por sistema y dosis en la carpeta de salida, mas un resumen
 por pantalla y un fichero RESUMEN.txt con todo lo anterior.

 Autor: Johan S. Pena-Campos   |   Generado para R1-7(c), septiembre 2026
=========================================================================

AUTHORSHIP
    Johan S. Pena-Campos
    PhD thesis, "Interpretable Decision Support Systems for Early Cancer Diagnosis and
    Personalized Treatment", Universitat Politecnica de Catalunya and Pontificia
    Universidad Javeriana, 2026.

USE OF AI ASSISTANCE
    This file was drafted with generative AI (Claude, Anthropic) and then reviewed,
    corrected and validated by the author, who takes full responsibility for it. The
    research question this script answers, and the interpretation of what it returns,
    are the author's.
"""

import argparse
import itertools
import os
import sys

import numpy as np
from scipy.optimize import fsolve

# =========================================================================
# 1. PARAMETROS
#    Extraidos literalmente de MPC_Replicator_SA_V3.py y
#    MPC_Replicator_DA_V3.py, que son los que produjeron los resultados
#    reportados en el Cap. 6. Si se cambian alli, hay que cambiarlos aqui.
# =========================================================================

# --- Mono-agente: Cunningham et al. 3 poblaciones (T+, T^P, T^-) ---
SA = {
    "nombre": "Mono-agente (Cunningham)",
    "poblaciones": ["T+", "TP", "T-"],
    "r": np.array([2.7726e-3, 3.4657e-3, 6.6542e-3]),
    "A": np.array([[1.0, 0.7, 0.8],
                   [0.4, 1.0, 0.6],
                   [0.5, 0.9, 1.0]]),
    "mu": 1.5,
    "lam": 0.0,                     # sin muerte basal en el mono-agente
    "c": np.array([0.0, 0.0, 0.0]),  # sin Docetaxel
    "N_max": 8e4,
    "y0": np.array([560.36, 747.59, 47.10]),
}

# --- Dual-agente: West et al. 4 poblaciones (T+, T^P, T^-/+, T^-/-) ---
_delta = 0.05
DA = {
    "nombre": "Dual-agente (West)",
    "poblaciones": ["T+", "TP", "T-/+", "T-/-"],
    "r": np.array([2.78e-3, 3.55e-3, 6.65e-3, 6.65e-3]),
    "A": np.array([[1.0,           0.7,           0.8,           0.8 - 2*_delta],
                   [0.4,           1.0,           0.5,           0.5 - 2*_delta],
                   [0.6,           0.9,           1.0,           1.0 - 2*_delta],
                   [0.6 - _delta,  0.9 - _delta,  1.0 - _delta,  1.0]]),
    "mu": 1.5,
    "lam": 5e-4,                            # d_v en el codigo, lambda en el Cap. 5
    "c": np.array([1.5, 1.5, 1.5, 0.0]),    # el T^-/- no responde al Docetaxel
    "N_max": 8e4,
    "y0": np.array([2062.0, 2617.0, 3.97e-11, 3.97e-12]),
}

TOL_RESIDUO = 1e-9    # residuo maximo para aceptar una raiz
TOL_DUPLICADO = 1e-6  # distancia relativa por debajo de la cual dos raices son la misma
TOL_CERO = 1e-12      # por debajo de esto una poblacion se considera extinta


# =========================================================================
# 2. CAPACIDADES DE CARGA
#    Aqui esta la distincion que el Cap. 6 no hacia y que R1-4(a) obligo a
#    explicitar: la forma de K_TP depende del escenario.
# =========================================================================

def K_sa(y, u_a, P):
    """Capacidades de carga del modelo mono-agente. K_TP es LINEAL (Cunningham)."""
    return np.array([
        (P["mu"] - u_a) * y[1],      # K_T+ depende de la poblacion productora
        1e4 - 9.9e3 * u_a,           # K_TP lineal
        1e4,
    ])


def K_da(y, u_a, P):
    """Capacidades de carga del modelo dual-agente. K_TP es EXPONENCIAL (West)."""
    return np.array([
        (P["mu"] - u_a) * y[1],
        1e4 / (10.0 ** (2.0 * u_a)),  # K_TP exponencial
        1e4,
        6e3,
    ])


# =========================================================================
# 3. CAMPOS VECTORIALES
# =========================================================================

def campo_lv(y, u, P, K_fun):
    """
    Planta Lotka-Volterra en poblaciones absolutas.

        dy_i/dt = r_i (1 - sum_j a_ij y_j / K_i) (1 - c_i u_d) y_i - lam y_i

    Se protege K_i contra el cero: cuando la poblacion productora se extingue,
    K_T+ tiende a cero y el termino de competencia diverge. Biologicamente esa
    rama corresponde a T+ extinguiendose, asi que se satura en lugar de
    devolver un NaN que rompa el buscador de raices.
    """
    y = np.asarray(y, dtype=float)
    u_a = u[0]
    u_d = u[1] if len(u) > 1 else 0.0

    K = K_fun(y, u_a, P)
    K = np.where(np.abs(K) < 1e-30, 1e-30, K)

    competencia = P["A"] @ y
    crecimiento = P["r"] * (1.0 - competencia / K) * (1.0 - P["c"] * u_d)
    return (crecimiento - P["lam"]) * y


def campo_rep(x, u, P, K_fun):
    """
    Replicador con zona ficticia, en el simplex de dimension n+1.

    x[0] es la zona ficticia, con fitness identicamente nulo; x[1:] son las
    frecuencias de las n poblaciones reales. La conversion a poblaciones
    absolutas es y = N_max * x[1:], que es la normalizacion por constante fija
    que R1-3 discute.

        dx_i/dt = x_i (f_i - phi_barra),    phi_barra = sum_j x_j f_j
    """
    x = np.asarray(x, dtype=float)
    u_a = u[0]
    u_d = u[1] if len(u) > 1 else 0.0

    y = P["N_max"] * x[1:]
    K = K_fun(y, u_a, P)
    K = np.where(np.abs(K) < 1e-30, 1e-30, K)

    competencia = P["A"] @ y
    f_real = P["r"] * (1.0 - competencia / K) * (1.0 - P["c"] * u_d) - P["lam"]

    f = np.concatenate(([0.0], f_real))   # la zona ficticia tiene fitness nulo
    phi = float(np.dot(x, f))
    return x * (f - phi)


# =========================================================================
# 4. BUSQUEDA DE EQUILIBRIOS POR CARAS
# =========================================================================

def _jacobiano(F, z, h_rel=1e-6):
    """Jacobiano por diferencias centradas con paso relativo al estado."""
    z = np.asarray(z, dtype=float)
    n = len(z)
    J = np.zeros((n, n))
    for j in range(n):
        h = h_rel * max(abs(z[j]), 1.0)
        zp, zm = z.copy(), z.copy()
        zp[j] += h
        zm[j] -= h
        J[:, j] = (np.asarray(F(zp)) - np.asarray(F(zm))) / (2.0 * h)
    return J


def _clasificar(autovalores, en_simplex=False):
    """
    Clasifica por el signo de la parte real. En el replicador se descarta un
    autovalor nulo, que corresponde a la direccion normal al simplex y no
    informa sobre estabilidad dentro de el.

    Aviso sobre un artefacto: cuando la poblacion productora de testosterona se
    extingue, K_T+ = (mu - u_a) y_TP tiende a cero y el termino de competencia
    diverge. La saturacion numerica que evita el NaN produce entonces
    autovalores de magnitud absurda (10^20 y mayores). Esos equilibrios no son
    puntos regulares del campo, asi que se marcan como degenerados en lugar de
    clasificarlos: el modelo simplemente no esta definido ahi.
    """
    if np.max(np.abs(autovalores)) > 1e12:
        return "degenerado"
    re = np.real(autovalores)
    if en_simplex:
        orden = np.argsort(np.abs(re))
        re = re[orden][1:]           # fuera el modo trivial del simplex
    if len(re) == 0:
        return "indeterminado"
    if np.all(re < -1e-12):
        return "ESTABLE"
    if np.all(re > 1e-12):
        return "inestable"
    if np.any(re > 1e-12) and np.any(re < -1e-12):
        return "silla"
    return "marginal"


def _deduplicar(puntos, escala):
    unicos = []
    for p in puntos:
        if not any(np.linalg.norm((p - q) / escala) < TOL_DUPLICADO for q in unicos):
            unicos.append(p)
    return unicos


def equilibrios_lv(P, u, K_fun, n_arranques=40, semilla=0):
    """
    Recorre los 2^n soportes posibles. En cada cara, las poblaciones fuera del
    soporte se fijan a cero y se resuelve el sistema reducido; despues se
    evalua el jacobiano COMPLETO, que es el que dice si las poblaciones
    ausentes pueden reinvadir.
    """
    rng = np.random.default_rng(semilla)
    n = len(P["r"])
    F = lambda y: campo_lv(y, u, P, K_fun)
    escala = np.maximum(np.abs(P["y0"]), 1.0)

    encontrados = []
    for k in range(n + 1):
        for soporte in itertools.combinations(range(n), k):
            soporte = list(soporte)
            if not soporte:
                encontrados.append(np.zeros(n))
                continue

            def F_cara(v):
                y = np.zeros(n)
                y[soporte] = v
                return F(y)[soporte]

            for _ in range(n_arranques):
                v0 = rng.uniform(1.0, 1.5e4, size=len(soporte))
                try:
                    v, info, ok, _ = fsolve(F_cara, v0, full_output=True)
                except Exception:
                    continue
                if ok != 1:
                    continue
                if np.max(np.abs(F_cara(v))) > TOL_RESIDUO:
                    continue
                if np.any(v < -TOL_CERO):
                    continue          # fuera del ortante positivo: no es biologico
                y = np.zeros(n)
                y[soporte] = np.maximum(v, 0.0)
                encontrados.append(y)

    unicos = _deduplicar(encontrados, escala)
    salida = []
    for y in unicos:
        J = _jacobiano(F, y)
        w = np.linalg.eigvals(J)
        salida.append({"estado": y, "autovalores": w, "clase": _clasificar(w)})
    return salida


def equilibrios_rep(P, u, K_fun, n_arranques=40, semilla=0):
    """
    En el simplex, los equilibrios cumplen x_i (f_i - phi) = 0 para todo i: o
    bien la frecuencia es nula, o bien su fitness iguala al medio. Se recorren
    los soportes igual que en el caso LV, imponiendo ademas sum(x) = 1.
    """
    rng = np.random.default_rng(semilla)
    n_tot = len(P["r"]) + 1
    F = lambda x: campo_rep(x, u, P, K_fun)
    escala = np.ones(n_tot)

    encontrados = []
    for k in range(1, n_tot + 1):
        for soporte in itertools.combinations(range(n_tot), k):
            soporte = list(soporte)

            def F_cara(v):
                x = np.zeros(n_tot)
                x[soporte] = v
                res = F(x)[soporte]
                res[-1] = np.sum(v) - 1.0      # cierre del simplex
                return res

            for _ in range(n_arranques):
                v0 = rng.dirichlet(np.ones(len(soporte)))
                try:
                    v, info, ok, _ = fsolve(F_cara, v0, full_output=True)
                except Exception:
                    continue
                if ok != 1:
                    continue
                if np.max(np.abs(F_cara(v))) > TOL_RESIDUO:
                    continue
                if np.any(v < -TOL_CERO) or np.any(v > 1.0 + 1e-9):
                    continue
                x = np.zeros(n_tot)
                x[soporte] = np.clip(v, 0.0, 1.0)
                if abs(np.sum(x) - 1.0) > 1e-8:
                    continue
                encontrados.append(x)

    unicos = _deduplicar(encontrados, escala)
    salida = []
    for x in unicos:
        J = _jacobiano(F, x)
        w = np.linalg.eigvals(J)
        salida.append({"estado": x, "autovalores": w,
                       "clase": _clasificar(w, en_simplex=True)})
    return salida


# =========================================================================
# 5. INFORME
# =========================================================================

def _etiqueta_soporte(estado, nombres, umbral):
    vivos = [nombres[i] for i, v in enumerate(estado) if v > umbral]
    return "{" + ", ".join(vivos) + "}" if vivos else "{extincion total}"


def informar(sistema, P, u, equilibrios, es_rep, f_txt, carpeta):
    nombres = (["zona"] + P["poblaciones"]) if es_rep else P["poblaciones"]
    umbral = 1e-8 if es_rep else 1e-6
    dosis = "u_a=%.2f" % u[0] + (", u_d=%.2f" % u[1] if len(u) > 1 else "")

    cab = "\n" + "=" * 78 + "\n%s  |  %s  |  %s\n" % (sistema, P["nombre"], dosis) + "=" * 78
    print(cab); f_txt.write(cab + "\n")

    if not equilibrios:
        msg = "  No se encontro ningun equilibrio admisible."
        print(msg); f_txt.write(msg + "\n")
        return

    orden = {"ESTABLE": 0, "silla": 1, "marginal": 2, "inestable": 3,
             "degenerado": 5, "indeterminado": 4}
    equilibrios = sorted(equilibrios, key=lambda e: orden.get(e["clase"], 9))

    filas = []
    for e in equilibrios:
        soporte = _etiqueta_soporte(e["estado"], nombres, umbral)
        estado = "  ".join("%s=%.4g" % (nm, v) for nm, v in zip(nombres, e["estado"]))
        re_max = float(np.max(np.real(e["autovalores"])))
        linea = ("  [%-10s] %-34s  max Re(lambda) = %+.3e\n"
                 "               %s" % (e["clase"], soporte, re_max, estado))
        print(linea); f_txt.write(linea + "\n")
        filas.append([e["clase"], soporte, re_max] + list(e["estado"]))

    nombre_csv = "%s_ua%.2f%s.csv" % (sistema, u[0],
                                      ("_ud%.2f" % u[1]) if len(u) > 1 else "")
    ruta = os.path.join(carpeta, nombre_csv)
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("clase,soporte,max_Re_lambda," + ",".join(nombres) + "\n")
        for fila in filas:
            fh.write("%s,\"%s\",%.6e," % (fila[0], fila[1], fila[2]) +
                     ",".join("%.6e" % v for v in fila[3:]) + "\n")


def comparar_lazo_cerrado(ruta_npz, equilibrios_por_sistema, P, f_txt):
    """
    Segunda mitad de la pregunta de R1: donde queda el punto de operacion del
    controlador respecto del equilibrio de coexistencia.
    """
    if not os.path.exists(ruta_npz):
        print("\n[aviso] No existe %s; se omite la comparacion." % ruta_npz)
        return
    d = np.load(ruta_npz, allow_pickle=True)

    # Los ficheros del Rep-MPC guardan 'Xhist_abs'; los del baseline LV-MPC,
    # 'Xhist'. Se admiten ambos, en poblaciones absolutas.
    clave = next((k for k in ("Xhist_abs", "Xhist") if k in d), None)
    if clave is None:
        print("\n[aviso] %s no contiene 'Xhist_abs' ni 'Xhist'; se omite." % ruta_npz)
        return

    X = np.asarray(d[clave])
    if X.ndim != 2:
        print("\n[aviso] '%s' no es bidimensional; se omite." % clave)
        return
    estado_final = X[:, -1] if X.shape[0] < X.shape[1] else X[-1, :]

    cab = "\n" + "=" * 78 + "\nPUNTO DE OPERACION DEL CONTROLADOR  (%s)\n" % os.path.basename(ruta_npz) + "=" * 78
    print(cab); f_txt.write(cab + "\n")
    linea = "  Estado final del lazo cerrado [%s]: " % clave + "  ".join("%.6g" % v for v in estado_final)
    print(linea); f_txt.write(linea + "\n")

    # La dosis media del ultimo decimo de la simulacion dice contra que
    # equilibrio conviene comparar: el controlador no opera a dosis cero.
    if "Uhist" in d:
        U = np.asarray(d["Uhist"])
        if U.ndim == 2:
            U = U if U.shape[0] < U.shape[1] else U.T
            cola = U[:, int(0.9 * U.shape[1]):]
            u_med = cola.mean(axis=1)
            linea = ("  Dosis media del ultimo 10% del horizonte: " +
                     "  ".join("u%d=%.4f" % (i, v) for i, v in enumerate(u_med)))
            print(linea); f_txt.write(linea + "\n")

    for sistema, equilibrios in equilibrios_por_sistema.items():
        for e in equilibrios:
            if len(e["estado"]) != len(estado_final):
                continue
            escala = np.maximum(np.abs(e["estado"]), 1.0)
            dist = float(np.linalg.norm((estado_final - e["estado"]) / escala))
            linea = "  [%s] %-11s distancia relativa = %.4g" % (sistema, e["clase"], dist)
            print(linea); f_txt.write(linea + "\n")


# =========================================================================
# 6. PRINCIPAL
# =========================================================================

def main():
    ap = argparse.ArgumentParser(description="Equilibrios y estabilidad -- R1-7(c)")
    ap.add_argument("--sistema", default="todos",
                    choices=["todos", "lv_sa", "lv_da", "rep_sa", "rep_da"])
    ap.add_argument("--dosis-sa", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    ap.add_argument("--dosis-da", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    ap.add_argument("--dosis-doc", nargs="+", type=float, default=[0.0, 0.5])
    ap.add_argument("--resultados", default=None)
    ap.add_argument("--salida", default="Equilibrios_Resultados")
    ap.add_argument("--arranques", type=int, default=40)
    args = ap.parse_args()

    os.makedirs(args.salida, exist_ok=True)
    ruta_txt = os.path.join(args.salida, "RESUMEN.txt")

    with open(ruta_txt, "w", encoding="utf-8") as f_txt:
        enc = ("ANALISIS DE EQUILIBRIOS Y ESTABILIDAD -- R1-7(c)\n"
               "Modelos del Capitulo 6: planta Lotka-Volterra y replicador con zona ficticia.\n"
               "Los equilibrios se buscan cara por cara y se clasifican con el jacobiano completo.\n")
        print(enc); f_txt.write(enc + "\n")

        acumulado = {}

        if args.sistema in ("todos", "lv_sa"):
            for ua in args.dosis_sa:
                eq = equilibrios_lv(SA, [ua], K_sa, args.arranques)
                informar("LV_SA", SA, [ua], eq, False, f_txt, args.salida)
                acumulado["LV_SA(u_a=%.2f)" % ua] = eq

        if args.sistema in ("todos", "rep_sa"):
            for ua in args.dosis_sa:
                eq = equilibrios_rep(SA, [ua], K_sa, args.arranques)
                informar("REP_SA", SA, [ua], eq, True, f_txt, args.salida)

        if args.sistema in ("todos", "lv_da"):
            for ua in args.dosis_da:
                for ud in args.dosis_doc:
                    eq = equilibrios_lv(DA, [ua, ud], K_da, args.arranques)
                    informar("LV_DA", DA, [ua, ud], eq, False, f_txt, args.salida)
                    acumulado["LV_DA(u_a=%.2f,u_d=%.2f)" % (ua, ud)] = eq

        if args.sistema in ("todos", "rep_da"):
            for ua in args.dosis_da:
                for ud in args.dosis_doc:
                    eq = equilibrios_rep(DA, [ua, ud], K_da, args.arranques)
                    informar("REP_DA", DA, [ua, ud], eq, True, f_txt, args.salida)

        if args.resultados:
            comparar_lazo_cerrado(args.resultados, acumulado, DA, f_txt)

        fin = "\n\nHecho. CSV y RESUMEN.txt en: %s" % os.path.abspath(args.salida)
        print(fin); f_txt.write(fin + "\n")


if __name__ == "__main__":
    sys.exit(main())
