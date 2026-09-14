"""
analisis_atribucion_individual.py

REPRODUCES IN THE THESIS
    Per-patient attribution, Section 4.9.

PROVENANCE
    Copied verbatim from the working tree of the thesis. The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md at the
    repository root.
"""

# -*- coding: utf-8 -*-
"""
Cuanto varia la atribucion ENTRE PACIENTES.

La pregunta de fondo: el articulo usa la interpretabilidad como auditoria de
poblacion. Para que sirva a un clinico ante UN paciente, la explicacion tiene
que ser distinta entre pacientes. Si a todos les sale lo mismo, la atribucion
individual no aporta informacion aunque exista.

Se mide sobre los ficheros que ya estan en Drive:
  nobsp_results_v4/demo_explainer.npz        contributions (500, 15, 2)
  nobsp_results_v4/input_event_explainer.npz contributions (500, 100, 2)
"""
import numpy as np

BASE = (r"G:\Mi unidad\Doctorado\Proyecto_CC\Test\Book\XAI_Model"
        r"\nobsp_results_v4")

demo = np.load(BASE + r"\demo_explainer.npz", allow_pickle=True)
C = demo['contributions']                     # (500, 15, 2)
nombres = [str(x) for x in demo['feature_names']]
clases = [str(x) for x in demo['class_names']]
print("clases:", clases, " -> se usa la columna de la clase cancer (indice 1)")
D = C[:, :, 1]                                # (500, 15) contribucion a cancer
n = D.shape[0]

print("=" * 78)
print("(1) CUAL ES LA VARIABLE DOMINANTE, PACIENTE A PACIENTE")
print("=" * 78)
dom = np.argmax(np.abs(D), axis=1)
cuenta = np.bincount(dom, minlength=len(nombres))
orden = np.argsort(-cuenta)
for i in orden:
    if cuenta[i]:
        print("  %-8s dominante en %3d de %d pacientes  (%.1f%%)"
              % (nombres[i], cuenta[i], n, 100 * cuenta[i] / n))

print()
print("=" * 78)
print("(2) IMPORTANCIA MEDIA FRENTE A DISPERSION ENTRE PACIENTES")
print("=" * 78)
imp = np.abs(D)
media = imp.mean(axis=0)
media_norm = 100 * media / media.sum()
print("%-8s %10s %10s %10s %10s" % ("variable", "media %", "sd", "P5", "P95"))
for i in np.argsort(-media):
    print("%-8s %10.1f %10.4f %10.4f %10.4f"
          % (nombres[i], media_norm[i], imp[:, i].std(),
             np.percentile(imp[:, i], 5), np.percentile(imp[:, i], 95)))

print()
print("=" * 78)
print("(3) CUANTO SE PARECEN ENTRE SI LOS PERFILES DE ATRIBUCION")
print("=" * 78)
P = imp / np.maximum(imp.sum(axis=1, keepdims=True), 1e-12)   # perfil normalizado
rng = np.random.default_rng(7)
idx = rng.choice(n, size=min(300, n), replace=False)
Pn = P[idx] / np.maximum(np.linalg.norm(P[idx], axis=1, keepdims=True), 1e-12)
S = Pn @ Pn.T
iu = np.triu_indices(S.shape[0], k=1)
cos = S[iu]
print("  similitud coseno entre perfiles de pacientes distintos:")
print("     media %.4f   mediana %.4f   P5 %.4f   P95 %.4f"
      % (cos.mean(), np.median(cos), np.percentile(cos, 5), np.percentile(cos, 95)))
print("  (1,0 significaria que todos los pacientes reciben la misma explicacion)")

# cuantas variables hacen falta para explicar el 80% del perfil, por paciente
ordp = -np.sort(-P, axis=1)
acum = np.cumsum(ordp, axis=1)
k80 = (acum < 0.80).sum(axis=1) + 1
print()
print("  variables necesarias para cubrir el 80%% del perfil de un paciente:")
print("     mediana %d, P25 %d, P75 %d, maximo %d"
      % (np.median(k80), np.percentile(k80, 25), np.percentile(k80, 75), k80.max()))

print()
print("=" * 78)
print("(4) LO MISMO PARA LAS 100 POSICIONES DE EVENTO")
print("=" * 78)
ev = np.load(BASE + r"\input_event_explainer.npz", allow_pickle=True)
E = np.abs(ev['contributions'][:, :, 1])      # (500, 100)
Pe = E / np.maximum(E.sum(axis=1, keepdims=True), 1e-12)
Pen = Pe[idx] / np.maximum(np.linalg.norm(Pe[idx], axis=1, keepdims=True), 1e-12)
Se = Pen @ Pen.T
cose = Se[np.triu_indices(Se.shape[0], k=1)]
print("  similitud coseno entre perfiles temporales de pacientes distintos:")
print("     media %.4f   mediana %.4f   P5 %.4f   P95 %.4f"
      % (cose.mean(), np.median(cose), np.percentile(cose, 5), np.percentile(cose, 95)))
pos_dom = np.argmax(E, axis=1)
c2 = np.bincount(pos_dom, minlength=100)
top = np.argsort(-c2)[:5]
print("  posicion de evento dominante, las 5 mas frecuentes:")
for p_ in top:
    print("     e%03d en %3d de %d pacientes (%.1f%%)" % (p_, c2[p_], n, 100 * c2[p_] / n))
