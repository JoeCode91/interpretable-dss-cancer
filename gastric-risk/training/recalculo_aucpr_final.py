# -*- coding: utf-8 -*-
"""
recalculo_aucpr_final.py -- AUC-PR (average precision) de los dos modelos y bootstrap pareado.

REPRODUCES IN THE THESIS
    Computes AUC-PR as average precision, without interpolation between operating
    points, for the CNN-2D and BiLSTM-Attention models, with paired bootstrap confidence
    intervals for each metric and for their difference. Produces the AUC-PR values
    reported in Section 4.7.

PROVENANCE
    Taken from the working tree that produced the reported results. The computational
    content is unchanged; absolute paths were replaced by relative defaults, and comments
    were edited. See CHANGES-FROM-THESIS.md at the repository root.

--------------------------------------------------------------------------
  - Probabilidades de test de los DOS modelos desde sus *_probas.npz (los guardan los
    barridos de umbral).
  - tau* leido de `tau_youden_val` de cada *_evaluacion_sweep.json.
  - Pareado comprobado por ID: se leen los ID del test HDF5 que uso cada modelo (campo
    test_data de su *_config.json) y se exige igualdad exacta (y tambien de y); si no, error.
  - Checkpoints fijados por CNN2D_TS y BILSTM_TS (obligatorias; ver checkpoint_utils.py);
    rutas y SHA-256 en el JSON.
  - AP = average_precision_score; bootstrap estratificado por clase y pareado, B = 2.000.
  - Salida en METRICS_OUT_DIR (por defecto ./batches/Resultados/metricas_finales).
El BiLSTM-Attention se entrena con un script que este repositorio no publica; este script
solo lee sus salidas (_probas.npz, _evaluacion_sweep.json, _config.json, _best.pt).

Uso:  CNN2D_TS=<ts> BILSTM_TS=<ts> python recalculo_aucpr_final.py

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
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, auc)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checkpoint_utils as ck

OUT = Path(os.environ.get("METRICS_OUT_DIR", "./batches/Resultados/metricas_finales"))
B_BOOT = 2000
SEED = 20260827


def ap(y, p):
    return float(average_precision_score(y, p))


def auc_pr_interpolada(y, p):
    pr, rc, _ = precision_recall_curve(y, p)
    return float(auc(rc, pr))


def j_en_tau(y, p, tau):
    yhat = p >= tau
    tp = int((yhat & (y == 1)).sum()); fn = int((~yhat & (y == 1)).sum())
    tn = int((~yhat & (y == 0)).sum()); fp = int((yhat & (y == 0)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    return sens + spec - 1.0, sens, spec


def cargar(nombre, fichero):
    probas = fichero("_probas.npz")
    ev = json.load(open(fichero("_evaluacion_sweep.json"), encoding='utf-8'))
    cfg = json.load(open(fichero("_config.json"), encoding='utf-8'))
    d = np.load(probas)
    y, p = d["y_test"].astype(int), d["p_test"].astype(float)
    with h5py.File(cfg["test_data"], 'r') as f:
        ids = f["ID"][:]
        y_h5 = f["y"][:].astype(int)
    if not np.array_equal(y, y_h5):
        raise RuntimeError(f"{nombre}: y de probas != y del test HDF5 {cfg['test_data']}")
    print(f"  {nombre}: n={len(y)} pos={y.sum()} tau*={ev['tau_youden_val']:.2f} test={cfg['test_data']}", flush=True)
    return y, p, float(ev["tau_youden_val"]), ids, {"probas": str(probas), "test_data": cfg["test_data"]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 78, flush=True)
    print("RECALCULO AUC-PR (AP) Y BOOTSTRAP PAREADO", flush=True)
    print("=" * 78, flush=True)
    pt_b, pt_c = ck.bilstm_pt(), ck.cnn_pt()
    y_b, p_b, tau_b, id_b, src_b = cargar("BiLSTM-Attention", ck.bilstm_fichero)
    y_c, p_c, tau_c, id_c, src_c = cargar("CNN-2D", ck.cnn_fichero)
    if not (np.array_equal(id_b, id_c) and np.array_equal(y_b, y_c)):
        raise RuntimeError("los test de los dos modelos no son los mismos pacientes en el mismo orden")
    print("  pareado por ID: OK", flush=True)
    y = y_b
    modelos = [("BiLSTM-Attention", p_b, tau_b), ("CNN-2D", p_c, tau_c)]

    puntos = {}
    for nom, p, tau in modelos:
        a_roc = float(roc_auc_score(y, p)); a_ap = ap(y, p); a_in = auc_pr_interpolada(y, p)
        J, sens, spec = j_en_tau(y, p, tau)
        puntos[nom] = dict(auc_roc=a_roc, ap=a_ap, auc_pr_interpolada=a_in,
                           sesgo_rel=(a_in - a_ap) / a_ap, tau=tau, J=J, sens=sens, spec=spec)
        print("%-18s AUC-ROC %.4f  AP %.4f  trapecio %.4f  J(tau*=%.2f) %.4f  sens %.4f  spec %.4f"
              % (nom, a_roc, a_ap, a_in, tau, J, sens, spec), flush=True)

    rng = np.random.default_rng(SEED)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    acc = {nom: {"auc_roc": [], "ap": [], "J": []} for nom, _, _ in modelos}
    dif = {"auc_roc": [], "ap": [], "J": []}
    t0 = time.time()
    for b in range(B_BOOT):
        ib = np.concatenate([rng.choice(pos, pos.size, replace=True), rng.choice(neg, neg.size, replace=True)])
        yb = y[ib]; vals = {}
        for nom, p, tau in modelos:
            pb = p[ib]
            v = dict(auc_roc=float(roc_auc_score(yb, pb)), ap=ap(yb, pb), J=j_en_tau(yb, pb, tau)[0])
            vals[nom] = v
            for k in acc[nom]:
                acc[nom][k].append(v[k])
        for k in dif:
            dif[k].append(vals["CNN-2D"][k] - vals["BiLSTM-Attention"][k])

    def ic(v):
        lo, hi = np.percentile(np.asarray(v), [2.5, 97.5]); return float(lo), float(hi), float(hi - lo)

    resumen = {"fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "n": int(len(y)),
               "positivos": int(y.sum()), "B_bootstrap": B_BOOT, "semilla": SEED,
               "pareado_por_id": True, "puntos": puntos, "ic": {}, "diferencias": {},
               "procedencia": {"BiLSTM-Attention": {**ck.registro(pt_b), **src_b},
                               "CNN-2D": {**ck.registro(pt_c), **src_c}},
               "nota": "AP = average_precision_score (sin interpolar); 'auc_pr_interpolada' = trapecios, solo referencia"}
    for nom, _, _ in modelos:
        resumen["ic"][nom] = {}
        for k in ("auc_roc", "ap", "J"):
            lo, hi, w = ic(acc[nom][k])
            resumen["ic"][nom][k] = dict(punto=puntos[nom][k], lo=lo, hi=hi, anchura=w)
            print("%-18s %-8s %.4f [%.4f, %.4f]" % (nom, k, puntos[nom][k], lo, hi), flush=True)
    for k in ("auc_roc", "ap", "J"):
        lo, hi, w = ic(dif[k])
        pt = puntos["CNN-2D"][k] - puntos["BiLSTM-Attention"][k]
        resumen["diferencias"][k] = dict(punto=float(pt), lo=lo, hi=hi, anchura=w, cubre_cero=bool(lo <= 0 <= hi))
        print("DIF CNN-BiLSTM %-8s %.4f [%.4f, %.4f]" % (k, pt, lo, hi), flush=True)
    with io.open(OUT / "metricas_finales.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print("salidas en: %s (%.0f s)" % (OUT, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
