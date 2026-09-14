"""
recalculo_aucpr_final.py

REPRODUCES IN THE THESIS
    Recomputes AUC-PR as average precision, without interpolation between operating
    points. Produces the 0.088 and 0.095 reported in Section 4.7.

PROVENANCE
    Copied verbatim from the working tree of the thesis. The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md at the
    repository root.
"""

# -*- coding: utf-8 -*-
"""
Cierra el TODO 2 del articulo Cap.4/AIM: AUC-PR de los DOS modelos finales
calculado con la MISMA funcion, e intervalos de confianza por bootstrap sobre
sus probabilidades reales.

Por que hace falta. El AUC-PR del BiLSTM que aparece en la tabla del articulo
(0,0855) lo produjo recover_v32_and_sweep.py con auc(rc, pr), que interpola
linealmente en el espacio precision-recall, algo que Davis y Goadrich declaran
incorrecto. El de la CNN-2D (0,0912) lo produjo Modelo_Predictivo_CNN2D_V4.py
con average_precision_score, que no interpola. Las dos cifras que el articulo
compara no se calcularon con la misma funcion.

Que hace:
  1. Reconstruye las probabilidades de test del BiLSTM V3.2 (no se guardaron en
     su momento) y las deja en un .npz para no repetirlo nunca mas.
  2. Recalcula todo con average_precision_score, y de paso con auc(rc, pr),
     para medir el sesgo de la interpolacion sobre estos datos concretos.
  3. Bootstrap estratificado PAREADO: el mismo remuestreo se aplica a los dos
     modelos, de modo que ademas de los IC de cada metrica se obtiene el IC de
     la DIFERENCIA entre modelos, que es la afirmacion que el articulo necesita.

Salidas en XAI_Model/metricas_finales/.
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
import torch
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, auc)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================== CONFIGURACION ==============================
BILSTM_PT = Path(r"D:\Proyecto_CC_Batches\Modelos_V32_PyTorch"
                 r"\modelo_v32_pytorch_20260510_151626_best.pt")
BILSTM_TEST = Path(r"D:\Proyecto_CC_Batches\Datos_Modelo"
                   r"\datos_modelo_test_20260506_120925.h5")
CNN_PROBAS = Path(r"D:\Proyecto_CC_Batches\Modelos_CNN2D_V4_PyTorch"
                  r"\modelo_cnn2d_v4_pytorch_20260519_043106_probas.npz")

OUT = Path(os.path.dirname(os.path.abspath(__file__))) / "metricas_finales"
OUT.mkdir(exist_ok=True)
NPZ_BILSTM = OUT / "probas_test_v32.npz"

TAU_BILSTM = 0.78          # tau* Youden fijado en validacion, el de la tabla
TAU_CNN = 0.44
BATCH = 64
B_BOOT = 2000
SEED = 20260827
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================== METRICAS ==============================
def ap(y, p):
    """average_precision_score: escalonada, sin interpolar. La correcta."""
    return float(average_precision_score(y, p))


def auc_pr_interpolada(y, p):
    """auc(rc, pr): trapecios. Lo que objetan Davis y Goadrich."""
    pr, rc, _ = precision_recall_curve(y, p)
    return float(auc(rc, pr))


def j_en_tau(y, p, tau):
    yhat = p >= tau
    tp = int((yhat & (y == 1)).sum())
    fn = int((~yhat & (y == 1)).sum())
    tn = int((~yhat & (y == 0)).sum())
    fp = int((yhat & (y == 0)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return sens + spec - 1.0, sens, spec


# ============================== PROBAS DEL BiLSTM ==============================
def probas_bilstm():
    if NPZ_BILSTM.exists():
        d = np.load(NPZ_BILSTM)
        print("  reutilizando %s" % NPZ_BILSTM.name, flush=True)
        return d["y_test"].astype(int), d["p_test"].astype(float)

    from Modelo_Predictivo_V3_PyTorch import (BiLSTMAttentionModelV3,
                                              DROPOUT_LSTM, DROPOUT_MHA,
                                              DROPOUT_DENSE)
    t0 = time.time()
    with h5py.File(BILSTM_TEST, "r") as f:
        n = f["y"].shape[0]
        n_feat = f["X_ventanas"].shape[3]
        n_demo = f["X_demograficos"].shape[1]
        y = np.asarray(f["y"][:]).ravel().astype(int)
        print("  n=%d  positivos=%d  feat=%d  demo=%d"
              % (n, y.sum(), n_feat, n_demo), flush=True)

        model = BiLSTMAttentionModelV3(
            n_features=n_feat, n_demograficos=n_demo,
            dropout_lstm=DROPOUT_LSTM, dropout_mha=DROPOUT_MHA,
            dropout_dense=DROPOUT_DENSE).to(DEVICE)
        model.load_state_dict(torch.load(BILSTM_PT, map_location=DEVICE,
                                         weights_only=True))
        model.eval()
        print("  parametros: %d" % sum(q.numel() for q in model.parameters()),
              flush=True)

        p = np.empty(n, dtype=np.float64)
        with torch.no_grad():
            for i0 in range(0, n, BATCH):
                i1 = min(i0 + BATCH, n)
                xv = torch.from_numpy(f["X_ventanas"][i0:i1].astype(np.float32)).to(DEVICE)
                xm = torch.from_numpy(f["X_mascaras"][i0:i1].astype(np.float32)).to(DEVICE)
                xd = torch.from_numpy(f["X_demograficos"][i0:i1].astype(np.float32)).to(DEVICE)
                p[i0:i1] = model(xv, xm, xd).flatten().cpu().numpy()
                if (i0 // BATCH) % 20 == 0:
                    print("    %d/%d  (%.0f s)" % (i1, n, time.time() - t0), flush=True)

    np.savez_compressed(NPZ_BILSTM, y_test=y, p_test=p)
    print("  inferencia en %.1f s, guardado %s" % (time.time() - t0, NPZ_BILSTM.name),
          flush=True)
    return y, p


# ============================== PRINCIPAL ==============================
def main():
    print("=" * 78, flush=True)
    print("RECALCULO DEL AUC-PR Y BOOTSTRAP SOBRE LOS MODELOS FINALES", flush=True)
    print("inicio: %s   device: %s"
          % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), DEVICE), flush=True)
    print("=" * 78, flush=True)

    print("\n[1/4] Probabilidades de test del BiLSTM V3.2", flush=True)
    y_b, p_b = probas_bilstm()

    print("\n[2/4] Probabilidades de test de la CNN-2D V4", flush=True)
    d = np.load(CNN_PROBAS)
    y_c, p_c = d["y_test"].astype(int), d["p_test"].astype(float)
    print("  n=%d  positivos=%d" % (len(y_c), y_c.sum()), flush=True)

    pareado = bool(len(y_b) == len(y_c) and np.array_equal(y_b, y_c))
    print("  etiquetas identicas y en el mismo orden: %s" % pareado, flush=True)
    if not pareado:
        print("  AVISO: sin pareado no se puede dar el IC de la diferencia.", flush=True)

    y = y_b
    modelos = [("BiLSTM-Attention", p_b, TAU_BILSTM),
               ("CNN-2D", p_c, TAU_CNN)]

    print("\n[3/4] Puntos, con las dos funciones", flush=True)
    print("%-18s %9s %12s %12s %9s %9s %8s %8s"
          % ("modelo", "AUC-ROC", "AP (buena)", "auc(rc,pr)", "sesgo", "J(tau*)",
             "sens", "spec"), flush=True)
    puntos = {}
    for nom, p, tau in modelos:
        a_roc = float(roc_auc_score(y, p))
        a_ap = ap(y, p)
        a_in = auc_pr_interpolada(y, p)
        J, sens, spec = j_en_tau(y, p, tau)
        puntos[nom] = dict(auc_roc=a_roc, ap=a_ap, auc_pr_interpolada=a_in,
                           sesgo_rel=(a_in - a_ap) / a_ap, tau=tau,
                           J=J, sens=sens, spec=spec)
        print("%-18s %9.4f %12.4f %12.4f %+8.1f%% %9.4f %8.4f %8.4f"
              % (nom, a_roc, a_ap, a_in, 100 * (a_in - a_ap) / a_ap, J, sens, spec),
              flush=True)

    print("\n[4/4] Bootstrap estratificado pareado, %d remuestreos" % B_BOOT, flush=True)
    rng = np.random.default_rng(SEED)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    acc = {nom: {"auc_roc": [], "ap": [], "J": []} for nom, _, _ in modelos}
    dif = {"auc_roc": [], "ap": [], "J": []}
    t0 = time.time()
    for b in range(B_BOOT):
        ib = np.concatenate([rng.choice(pos, pos.size, replace=True),
                             rng.choice(neg, neg.size, replace=True)])
        yb = y[ib]
        vals = {}
        for nom, p, tau in modelos:
            pb = p[ib]
            v = dict(auc_roc=float(roc_auc_score(yb, pb)),
                     ap=ap(yb, pb),
                     J=j_en_tau(yb, pb, tau)[0])
            vals[nom] = v
            for k in acc[nom]:
                acc[nom][k].append(v[k])
        for k in dif:
            dif[k].append(vals["CNN-2D"][k] - vals["BiLSTM-Attention"][k])
        if (b + 1) % 500 == 0:
            print("    %d/%d  (%.0f s)" % (b + 1, B_BOOT, time.time() - t0), flush=True)

    def ic(v):
        v = np.asarray(v)
        lo, hi = np.percentile(v, [2.5, 97.5])
        return float(lo), float(hi), float(hi - lo)

    print("\n%-18s %-9s %9s %24s %9s"
          % ("modelo", "metrica", "punto", "IC 95%", "anchura"), flush=True)
    resumen = {"fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               "n": int(len(y)), "positivos": int(y.sum()),
               "B_bootstrap": B_BOOT, "pareado": pareado,
               "puntos": puntos, "ic": {}, "diferencias": {}}
    etiq = {"auc_roc": "AUC-ROC", "ap": "AUC-PR", "J": "Youden J"}
    for nom, _, _ in modelos:
        resumen["ic"][nom] = {}
        for k in ("auc_roc", "ap", "J"):
            lo, hi, w = ic(acc[nom][k])
            pt = puntos[nom][k if k != "ap" else "ap"]
            resumen["ic"][nom][k] = dict(punto=pt, lo=lo, hi=hi, anchura=w)
            print("%-18s %-9s %9.4f  [%7.4f , %7.4f] %9.4f"
                  % (nom, etiq[k], pt, lo, hi, w), flush=True)

    print("\nDIFERENCIA (CNN-2D menos BiLSTM), mismo remuestreo en las dos ramas:",
          flush=True)
    print("%-10s %10s %24s %10s %14s"
          % ("metrica", "punto", "IC 95%", "anchura", "cubre el cero"), flush=True)
    for k in ("auc_roc", "ap", "J"):
        lo, hi, w = ic(dif[k])
        pt = puntos["CNN-2D"][k] - puntos["BiLSTM-Attention"][k]
        cubre = bool(lo <= 0.0 <= hi)
        resumen["diferencias"][k] = dict(punto=float(pt), lo=lo, hi=hi,
                                         anchura=w, cubre_cero=cubre)
        print("%-10s %10.4f  [%7.4f , %7.4f] %10.4f %14s"
              % (etiq[k], pt, lo, hi, w, "SI" if cubre else "NO"), flush=True)

    with io.open(OUT / "metricas_finales.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print("\nsalidas en: %s" % OUT, flush=True)
    print("duracion total: %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
