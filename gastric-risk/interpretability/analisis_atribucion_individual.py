"""
analisis_atribucion_individual.py -- how much the attribution varies between patients.

REPRODUCES IN THE THESIS
    Per-patient attribution, Section 4.9.

PROVENANCE
    Taken from the working tree that produced the reported results. The computational
    content is unchanged; absolute paths were replaced by relative defaults and
    environment variables, notification calls were removed, and comments were edited.
    See CHANGES-FROM-THESIS.md at the repository root.

--------------------------------------------------------------------------

Cuanto varia la atribucion ENTRE PACIENTES.

La pregunta de fondo: el articulo usa la interpretabilidad como auditoria de
poblacion. Para que sirva a un clinico ante UN paciente, la explicacion tiene
que ser distinta entre pacientes. Si a todos les sale lo mismo, la atribucion
individual no aporta informacion aunque exista.

Se mide sobre las salidas de NObSP de la muestra B (NOBSP_RUNB_DIR):
  demo_explainer.npz         W1, contributions (n_pacientes, 15, 2)
  input_event_explainer.npz  W2, contributions (n_sondas, 100, 2)

Muestra B: todos los cancer de train (100:1) y test, y el mismo numero de no cancer
de train+test (RandomState(42)). Las filas de W2 son las 500 sondas multiplicativas
alrededor de la media de esa muestra, no pacientes.

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
# Las rutas de este fichero eran absolutas en el arbol de trabajo del autor y se han
# sustituido por rutas relativas. Ajustalas a tu entorno antes de ejecutar: apuntan a
# datos y modelos que este repositorio no distribuye (ver el README de gastric-risk).
import os

import numpy as np

BASE = os.environ.get('NOBSP_RUNB_DIR', "./batches/Resultados/nobsp_runB_traintest")

demo = np.load(os.path.join(BASE, "demo_explainer.npz"), allow_pickle=True)
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
ev = np.load(os.path.join(BASE, "input_event_explainer.npz"), allow_pickle=True)
E = np.abs(ev['contributions'][:, :, 1])      # (500, 100)
# Las filas de W2 son 500 SONDAS (perturbaciones de la media de la muestra), no
# pacientes; su numero no coincide con el de W1 (1.734), asi que se usa un indice
# propio. Se informan como sondas, no como pacientes.
print("  AVISO: filas de W2 = sondas de perturbacion, no pacientes")
n_e = E.shape[0]
idx_e = rng.choice(n_e, size=min(300, n_e), replace=False)
Pe = E / np.maximum(E.sum(axis=1, keepdims=True), 1e-12)
Pen = Pe[idx_e] / np.maximum(np.linalg.norm(Pe[idx_e], axis=1, keepdims=True), 1e-12)
Se = Pen @ Pen.T
cose = Se[np.triu_indices(Se.shape[0], k=1)]
print("  similitud coseno entre perfiles de SONDAS de perturbacion alrededor de una referencia (no pacientes);")
print("  mide sensibilidad local alrededor de la referencia, NO variabilidad entre pacientes:")
print("     media %.4f   mediana %.4f   P5 %.4f   P95 %.4f"
      % (cose.mean(), np.median(cose), np.percentile(cose, 5), np.percentile(cose, 95)))
pos_dom = np.argmax(E, axis=1)
c2 = np.bincount(pos_dom, minlength=100)
top = np.argsort(-c2)[:5]
print("  posicion de evento dominante, las 5 mas frecuentes:")
for p_ in top:
    print("     e%03d en %3d de %d sondas (%.1f%%)" % (p_, c2[p_], n_e, 100 * c2[p_] / n_e))
