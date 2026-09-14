"""
Threshold sweep V4 (A6) -- Cap. 4 NObSP, CNN 2D V4 Camino 2 Alex.

REPRODUCES IN THE THESIS
    Operating-threshold sweep: the Youden-optimal thresholds of Section 4.7.

PROVENANCE
    Copied verbatim from the working tree of the thesis. The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md at the
    repository root.

--------------------------------------------------------------------------
Threshold sweep V4 (A6) -- Cap. 4 NObSP, CNN 2D V4 Camino 2 Alex.

El run de training V4 (modelo_cnn2d_v4_pytorch_20260519_043106) guardo
evaluacion.json con metricas en thresholds 0.5 y 0.3 unicamente. Este
script:
  1. Carga el checkpoint best
  2. Re-corre val + test sets con CNN2DModelV4
  3. Threshold sweep fino (paso 0.01) sobre val para encontrar tau* Youden
  4. Aplica tau* a test
  5. Reescribe evaluacion.json (completo) + threshold_sweep_val.csv
  6. Imprime resumen comparativo V2 / V3.1 / V3.2 / Camino 1 / V4

Uso (en la PowerShell con D:\venv_torch_cu121 activado):
    D:\venv_torch_cu121\Scripts\python.exe -u threshold_sweep_v4.py

Autor: Johan S. Pena-Campos
Fecha: 2026-05-21 (sesion 33, A6)
"""
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)

sys.path.insert(0, str(Path(__file__).parent))
from modelo_cnn2d_v4_pytorch import CNN2DModelV4
from Modelo_Predictivo_CNN2D_V4 import (
    HDF5DatasetV4, find_latest_v4_split,
    BATCH_SIZE, C1, C2, C3, C_POOL, N_DENSE, DROPOUT,
)

# ===== CONFIG =====
TIMESTAMP = "20260519_043106"
MODEL_NAME = f"modelo_cnn2d_v4_pytorch_{TIMESTAMP}"
OUTPUT_DIR = Path(os.environ.get(
    "CNN2D_V4_OUT", r"D:\Proyecto_CC_Batches\Modelos_CNN2D_V4_PyTorch"
))
BEST_PT = OUTPUT_DIR / f"{MODEL_NAME}_best.pt"
HISTORY_JSON = OUTPUT_DIR / f"{MODEL_NAME}_history.json"
EVALUACION_JSON = OUTPUT_DIR / f"{MODEL_NAME}_evaluacion.json"


@torch.no_grad()
def evaluate_with_probs(model, loader, device):
    model.eval()
    ys, ps = [], []
    for X_e, X_d, y in loader:
        X_e = X_e.to(device, non_blocking=True)
        X_d = X_d.to(device, non_blocking=True)
        p = model(X_e, X_d).flatten().cpu().numpy()
        ys.append(y.numpy().flatten())
        ps.append(p)
    return np.concatenate(ys), np.concatenate(ps)


def metrics_at_threshold(y, p, threshold):
    yhat = (p >= threshold).astype(int)
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    J = sens + spec - 1
    return dict(
        threshold=float(threshold), sens=sens, spec=spec, prec=prec,
        f1=f1, J=J, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def main():
    print("=" * 78, flush=True)
    print(f"V4 Threshold Sweep ({MODEL_NAME})", flush=True)
    print("=" * 78, flush=True)

    if not BEST_PT.exists():
        print(f"ERROR: no encontrado {BEST_PT}", flush=True)
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {BEST_PT}", flush=True)

    # Resolver paths HDF5 V4 (auto-detect latest)
    val_path = os.environ.get("CNN2D_V4_VAL", "")
    test_path = os.environ.get("CNN2D_V4_TEST", "")
    if not val_path:
        val_path = find_latest_v4_split("val")
    if not test_path:
        test_path = find_latest_v4_split("test")
    print(f"Val:  {val_path}", flush=True)
    print(f"Test: {test_path}", flush=True)

    # Cargar datasets (preload a RAM, son chicos)
    print("\n[1/4] Cargando val + test...", flush=True)
    val_ds = HDF5DatasetV4(val_path, preload=True)
    test_ds = HDF5DatasetV4(test_path, preload=True)
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # Cargar modelo
    print("\n[2/4] Construyendo CNN2DModelV4 + cargando pesos...", flush=True)
    n_features = val_ds.shape_e[1]   # 1541
    n_eventos = val_ds.shape_e[2]    # 100
    n_demograficos = val_ds.shape_d[1]  # 15
    model = CNN2DModelV4(
        n_features=n_features,
        n_eventos=n_eventos,
        n_demograficos=n_demograficos,
        c1=C1, c2=C2, c3=C3, c_pool=C_POOL,
        n_dense=N_DENSE, dropout=DROPOUT,
    ).to(device)
    model.load_state_dict(
        torch.load(BEST_PT, map_location=device, weights_only=True)
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parametros: {n_params:,}", flush=True)

    # Inferencia val + test
    print("\n[3/4] Inferencia val + test...", flush=True)
    t0 = time.time()
    y_val, p_val = evaluate_with_probs(model, val_loader, device)
    print(
        f"  val: n={len(y_val)} pos={int(y_val.sum())} "
        f"en {time.time()-t0:.1f}s",
        flush=True,
    )
    t0 = time.time()
    y_test, p_test = evaluate_with_probs(model, test_loader, device)
    print(
        f"  test: n={len(y_test)} pos={int(y_test.sum())} "
        f"en {time.time()-t0:.1f}s",
        flush=True,
    )

    # AUC global
    auc_roc_val = float(roc_auc_score(y_val, p_val))
    pr, rc, _ = precision_recall_curve(y_val, p_val)
    auc_pr_val = float(auc(rc, pr))
    auc_roc_test = float(roc_auc_score(y_test, p_test))
    pr_t, rc_t, _ = precision_recall_curve(y_test, p_test)
    auc_pr_test = float(auc(rc_t, pr_t))

    # Threshold sweep val (paso 0.01)
    print("\n[4/4] Threshold sweep val (paso 0.01, 99 puntos)...", flush=True)
    sweep_taus = np.arange(0.01, 1.00, 0.01)
    sweep_val = [metrics_at_threshold(y_val, p_val, t) for t in sweep_taus]
    sweep_test = [metrics_at_threshold(y_test, p_test, t) for t in sweep_taus]

    # tau* Youden val
    Js_val = [s['J'] for s in sweep_val]
    best_idx = int(np.argmax(Js_val))
    tau_youden = sweep_val[best_idx]['threshold']
    print(
        f"  tau* Youden (val) = {tau_youden:.2f}  J = "
        f"{sweep_val[best_idx]['J']:+.4f}",
        flush=True,
    )
    print(
        f"    -> val sens={sweep_val[best_idx]['sens']:.3f}  "
        f"spec={sweep_val[best_idx]['spec']:.3f}",
        flush=True,
    )

    # Aplicar a test
    test_05 = metrics_at_threshold(y_test, p_test, 0.5)
    test_03 = metrics_at_threshold(y_test, p_test, 0.3)
    test_youden = metrics_at_threshold(y_test, p_test, tau_youden)

    # Cargar history para extraer best_epoch / best_composite
    history = {}
    if HISTORY_JSON.exists():
        with open(HISTORY_JSON) as f:
            history = json.load(f)
    epochs_trained = len(history.get('loss', []))
    best_epoch = -1
    best_composite = -1.0
    if 'val_composite' in history and history['val_composite']:
        best_epoch = int(np.argmax(history['val_composite'])) + 1
        best_composite = float(max(history['val_composite']))

    # Sobrescribir evaluacion.json (mantener auc_roc/auc_pr top-level, threshold_05/30,
    # agregar tau Youden y metricas val)
    eval_data = {
        'auc_roc_val': auc_roc_val,
        'auc_pr_val':  auc_pr_val,
        'auc_roc_test': auc_roc_test,
        'auc_pr_test':  auc_pr_test,
        # Mantener compatibilidad con el evaluacion.json original
        'auc_roc': auc_roc_test,
        'auc_pr':  auc_pr_test,
        'threshold_05': {
            **{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
               for k, v in test_05.items()},
            'auc_roc': auc_roc_test,
            'auc_pr':  auc_pr_test,
        },
        'threshold_03': {
            **{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
               for k, v in test_03.items()},
            'auc_roc': auc_roc_test,
            'auc_pr':  auc_pr_test,
        },
        'tau_youden_val': float(tau_youden),
        'threshold_youden_test': {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in test_youden.items()
        },
        'val_metrics_at_youden': {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in sweep_val[best_idx].items()
        },
        'best_epoch': best_epoch,
        'best_score_composite': best_composite,
        'epochs_trained': epochs_trained,
        'sweep_added_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(EVALUACION_JSON, 'w') as f:
        json.dump(eval_data, f, indent=2, default=float)
    print(f"\nGuardado: {EVALUACION_JSON}", flush=True)

    # CSV con sweep completo (val + test)
    out_csv = OUTPUT_DIR / f"{MODEL_NAME}_threshold_sweep.csv"
    with open(out_csv, 'w') as f:
        f.write(
            "threshold,"
            "sens_val,spec_val,prec_val,f1_val,J_val,tp_val,fn_val,fp_val,tn_val,"
            "sens_test,spec_test,prec_test,f1_test,J_test,tp_test,fn_test,fp_test,tn_test\n"
        )
        for sv, st in zip(sweep_val, sweep_test):
            f.write(
                f"{sv['threshold']:.2f},"
                f"{sv['sens']:.4f},{sv['spec']:.4f},{sv['prec']:.4f},"
                f"{sv['f1']:.4f},{sv['J']:+.4f},"
                f"{sv['tp']},{sv['fn']},{sv['fp']},{sv['tn']},"
                f"{st['sens']:.4f},{st['spec']:.4f},{st['prec']:.4f},"
                f"{st['f1']:.4f},{st['J']:+.4f},"
                f"{st['tp']},{st['fn']},{st['fp']},{st['tn']}\n"
            )
    print(f"Guardado: {out_csv}", flush=True)

    # Guardar probas para uso futuro (NObSP, comparativas)
    out_npz = OUTPUT_DIR / f"{MODEL_NAME}_probas.npz"
    np.savez(out_npz, y_val=y_val, p_val=p_val, y_test=y_test, p_test=p_test)
    print(f"Guardado: {out_npz}", flush=True)

    # Resumen
    n_pos_test = int(y_test.sum())
    print("\n" + "=" * 78, flush=True)
    print(
        f" V4 SUMMARY (test set, n={len(y_test)}, {n_pos_test} positives)",
        flush=True,
    )
    print("=" * 78, flush=True)
    print(f"  AUC-ROC test: {auc_roc_test:.4f}", flush=True)
    print(f"  AUC-PR  test: {auc_pr_test:.4f}", flush=True)
    print(f"  Best epoch:   {best_epoch} / {epochs_trained}", flush=True)
    print(f"  Best composite val: {best_composite:.4f}", flush=True)
    print("", flush=True)
    print(
        "  Threshold  | Sens   | Spec   | Prec   | F1     | J       | "
        "TP/FN | FP/TN",
        flush=True,
    )
    print(
        "  -----------|--------|--------|--------|--------|---------|"
        "-------|-------",
        flush=True,
    )
    for name, m in [
        ("0.50", test_05),
        ("0.30", test_03),
        (f"{tau_youden:.2f}*Y", test_youden),
    ]:
        print(
            f"  {name:>10} | {m['sens']:.3f}  | {m['spec']:.3f}  | "
            f"{m['prec']:.4f} | {m['f1']:.4f} | {m['J']:+.4f} | "
            f"{m['tp']:>3}/{m['fn']:<3} | {m['fp']:>4}/{m['tn']:<4}",
            flush=True,
        )

    print("\n" + "=" * 78, flush=True)
    print(" COMPARATIVO V2 / V3.1 / V3.2 / Camino 1 / V4 (Youden tau*)", flush=True)
    print("=" * 78, flush=True)
    print(
        "  Model     | tau*   | AUC-ROC | AUC-PR | Sens   | Spec   | J",
        flush=True,
    )
    print(
        "  ----------|--------|---------|--------|--------|--------|--------",
        flush=True,
    )
    print(
        "  V2        | 0.49   | 0.815   | 0.052  | 0.781  | 0.744  | +0.525",
        flush=True,
    )
    print(
        "  V3.1      | 0.53   | 0.813   | 0.051  | 0.781  | 0.723  | +0.504",
        flush=True,
    )
    print(
        "  V3.2      | 0.78   | 0.836   | 0.086  | 0.656  | 0.866  | +0.522",
        flush=True,
    )
    print(
        "  Camino 1  |  --    |  --     | 0.123  |  --    |  --    |  --     "
        "(pendiente sweep formal)",
        flush=True,
    )
    print(
        f"  V4 (este) | {tau_youden:.2f}   | {auc_roc_test:.3f}   | "
        f"{auc_pr_test:.3f}  | {test_youden['sens']:.3f}  | "
        f"{test_youden['spec']:.3f}  | {test_youden['J']:+.4f}",
        flush=True,
    )
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
