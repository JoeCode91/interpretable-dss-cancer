"""
diagnostico_metricas_aucpr.py

REPRODUCES IN THE THESIS
    Diagnoses the discrepancy that motivated the recomputation above.

PROVENANCE
    Copied verbatim from the working tree of the thesis. The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md at the
    repository root.

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

# -*- coding: utf-8 -*-
"""
Diagnostico de las metricas del Cap. 4, con numpy puro (este PC no tiene sklearn).

(1) Cuanto difiere auc(recall, precision), que interpola linealmente en el
    espacio PR, de average_precision, que no interpola. Davis y Goadrich
    demuestran que lo primero es incorrecto. El codigo del Cap. 4 usa AMBAS
    segun el modelo, luego los dos AUC-PR que el articulo compara no se
    calcularon con la misma funcion.

(2) Anchura de los intervalos de confianza con 96 positivos, por bootstrap
    estratificado.
"""
import numpy as np

NPZ = (r"G:\Mi unidad\Doctorado\Proyecto_CC\Test\Book\XAI_Model"
       r"\threshold_sweep_results\probas_test.npz")
RNG = np.random.default_rng(20260827)
B = 2000


def curva_pr(y, p):
    """Replica precision_recall_curve de scikit-learn."""
    o = np.argsort(-p, kind='mergesort')
    ys, ps = y[o], p[o]
    distintos = np.flatnonzero(np.diff(ps))
    idx = np.r_[distintos, ys.size - 1]
    tp = np.cumsum(ys)[idx]
    fp = np.cumsum(1 - ys)[idx]
    prec = tp / np.maximum(tp + fp, 1e-12)
    rec = tp / max(y.sum(), 1)
    return prec, rec


def ap_score(y, p):
    """average_precision: suma escalonada, sin interpolar."""
    prec, rec = curva_pr(y, p)
    drec = np.diff(np.r_[0.0, rec])
    return float(np.sum(drec * prec))


def auc_trapecio(y, p):
    """auc(recall, precision): trapecios, que es lo que Davis y Goadrich objetan."""
    prec, rec = curva_pr(y, p)
    return float(np.trapezoid(prec, rec)) if hasattr(np, 'trapezoid') \
        else float(np.trapz(prec, rec))


def auc_roc(y, p):
    """Mann-Whitney con correccion de empates."""
    o = np.argsort(p, kind='mergesort')
    r = np.empty(len(p), float)
    ps = p[o]
    i = 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def youden_max(y, p, th):
    tp = ((p[:, None] >= th) & (y[:, None] == 1)).sum(0)
    fp = ((p[:, None] >= th) & (y[:, None] == 0)).sum(0)
    fn = y.sum() - tp
    tn = (y == 0).sum() - fp
    return float(((tp / np.maximum(tp + fn, 1)) +
                  (tn / np.maximum(tn + fp, 1)) - 1).max())


d = np.load(NPZ)
y = d['y_test'].astype(int)
modelos = {'V2': d['proba_v2'].astype(float), 'V3.1': d['proba_v3'].astype(float)}

print("=" * 78)
print("(1) auc(recall, precision) FRENTE A average_precision")
print("=" * 78)
print("n=%d  positivos=%d  prevalencia=%.4f" % (len(y), y.sum(), y.mean()))
print("Valores de referencia en summary.json: auc_pr_v2=0.05238  auc_pr_v3=0.05145")
print("  (ese fichero se genero con auc(rc,pr), asi que la columna izquierda")
print("   deberia reproducirlos, lo que valida esta implementacion)")
print()
print("%-7s %14s %16s %12s %10s" % ("modelo", "auc(rc,pr)", "avg_precision",
                                    "diferencia", "sesgo"))
for nom, p in modelos.items():
    a_i, a_p = auc_trapecio(y, p), ap_score(y, p)
    print("%-7s %14.5f %16.5f %12.5f %+9.1f%%"
          % (nom, a_i, a_p, a_i - a_p, 100 * (a_i - a_p) / a_p))

print()
print("=" * 78)
print("(2) INTERVALOS DE CONFIANZA POR BOOTSTRAP (%d remuestreos, estratificado)" % B)
print("=" * 78)
pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
th = np.linspace(0.01, 0.99, 99)
print("%-7s %-12s %10s %24s %10s" % ("modelo", "metrica", "punto", "IC 95%", "anchura"))
for nom, p in modelos.items():
    acc = {'AUC-ROC': [], 'AUC-PR': [], 'Youden J': []}
    for _ in range(B):
        ib = np.concatenate([RNG.choice(pos, pos.size, replace=True),
                             RNG.choice(neg, neg.size, replace=True)])
        yb, pb = y[ib], p[ib]
        acc['AUC-ROC'].append(auc_roc(yb, pb))
        acc['AUC-PR'].append(ap_score(yb, pb))
        acc['Youden J'].append(youden_max(yb, pb, th))
    puntos = {'AUC-ROC': auc_roc(y, p), 'AUC-PR': ap_score(y, p),
              'Youden J': youden_max(y, p, th)}
    for met, vals in acc.items():
        v = np.array(vals)
        lo, hi = np.percentile(v, [2.5, 97.5])
        print("%-7s %-12s %10.4f  [%7.4f , %7.4f] %10.4f"
              % (nom, met, puntos[met], lo, hi, hi - lo))

print()
print("Nota: es el MISMO conjunto de prueba (9.727 pacientes, 96 positivos) que")
print("usan los dos modelos finales del articulo, asi que la anchura de estos")
print("intervalos es representativa de la que tendrian 0,086 y 0,091.")
