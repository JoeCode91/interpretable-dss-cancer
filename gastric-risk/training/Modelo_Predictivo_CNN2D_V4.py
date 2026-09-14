"""
Modelo_Predictivo_CNN2D_V4 -- training script para CNN 2D V4 (Camino 2,

REPRODUCES IN THE THESIS
    Trains the final CNN-2D model. Section 4.4 and the results reported for it.

PROVENANCE
    Copied verbatim from the working tree of the thesis. The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md at the
    repository root.

--------------------------------------------------------------------------
Modelo_Predictivo_CNN2D_V4 -- training script para CNN 2D V4 (Camino 2,
refinamiento Alex 18 May 2026 sobre tensor (1541, 100) sin overlap, sin masking).

Diferencias respecto a Modelo_Predictivo_CNN2D.py (Camino 1):
  - HDF5 schema nuevo: X_events (n, 1541, 100), X_demograficos (n, 15),
    y (n,). Sin X_mascaras.
  - Dataset V4 propio (HDF5DatasetV4) que devuelve 3-tupla en vez de 4.
  - Modelo CNN2DModelV4 (single-channel, MaxPool entre bloques).
  - BATCH_SIZE=64 (en vez de 128, restriccion VRAM RTX 3060 Ti verificada).
  - Reusa hybrid_loss + composite_score + threshold sweep helpers de V3 PyTorch.

Autor: Johan S. Pena-Campos, Cap. 4 Camino 2 V4
Fecha: 2026-05-18 (sesion 29)
"""
# Las rutas de este fichero eran absolutas en el arbol de trabajo del autor y se han
# sustituido por rutas relativas. Ajustalas a tu entorno antes de ejecutar: apuntan a
# datos y modelos que este repositorio no distribuye (ver el README de gastric-risk).


import os
import sys
import json
import time
import glob
from pathlib import Path
from datetime import datetime

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))

from modelo_cnn2d_v4_pytorch import CNN2DModelV4
# Reusar loss y composite del V3 PyTorch
from Modelo_Predictivo_V3_PyTorch import hybrid_loss, composite_score, J_WEIGHT

# Telegram notifications (silent fail)
try:
    from telegram_notifier import MPCNotifier
except Exception as _e:
    print(f"[Telegram] modulo no disponible ({_e}); notificaciones OFF", flush=True)
    class MPCNotifier:
        def __init__(self, **kw): self.enabled = False
        def notify_start(self, *a, **kw): return False
        def notify_progress(self, *a, **kw): return False
        def notify_error(self, *a, **kw): return False
        def notify_complete(self, *a, **kw): return False
        def notify_warning(self, *a, **kw): return False


# ==================== CONFIG ====================

V4_DIR = Path("./batches/Datos_Modelo_V4")

def find_latest_v4_split(split: str) -> str:
    """Encuentra el HDF5 V4 mas reciente para train/val/test (por timestamp)."""
    pattern = str(V4_DIR / f"datos_modelo_{split}_v4_*.h5")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No se encontraron HDF5 V4 en {V4_DIR} con pattern {pattern}. "
            f"Ejecuta Preparar_HDF5_V4.py primero."
        )
    return matches[-1]  # mas reciente por sort alfabetico timestamp


TRAIN_DATA = os.environ.get("CNN2D_V4_TRAIN", "")
VAL_DATA   = os.environ.get("CNN2D_V4_VAL",   "")
TEST_DATA  = os.environ.get("CNN2D_V4_TEST",  "")
OUTPUT_DIR = Path(os.environ.get(
    "CNN2D_V4_OUT", "./batches/Modelos_CNN2D_V4_PyTorch/"
))
TAG = os.environ.get("CNN2D_V4_TAG", "cnn2d_v4")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Hiperparametros
BATCH_SIZE = 64          # vs 128 V3.2/Camino 1, restriccion VRAM (peak 6.2 GB en RTX 3060 Ti)
EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 10
LR_REDUCE_PATIENCE = 5
LR_REDUCE_FACTOR = 0.5
DROPOUT = 0.4

# Arquitectura CNN 2D V4
C1 = 32
C2 = 64
C3 = 128
C_POOL = 64
N_DENSE = 64


# ==================== DATASET V4 ====================

class HDF5DatasetV4(Dataset):
    """Lee HDF5 V4 con keys X_events, X_demograficos, y (sin X_mascaras).

    preload=True carga todo a RAM (recomendado para train balanceado 20:1).
    """

    def __init__(self, path, preload: bool = False):
        self.path = path
        self.preload = preload
        with h5py.File(path, 'r') as f:
            self.n = f['y'].shape[0]
            self.shape_e = f['X_events'].shape
            self.shape_d = f['X_demograficos'].shape
            if preload:
                t0 = time.time()
                print(f"  preload {path}...", flush=True)
                self._x_e = f['X_events'][:].astype(np.float32)
                self._x_d = f['X_demograficos'][:].astype(np.float32)
                self._y = f['y'][:].astype(np.float32)
                size_gb = (self._x_e.nbytes + self._x_d.nbytes + self._y.nbytes) / (1024 ** 3)
                print(f"  preload OK: {size_gb:.1f} GB en RAM, {time.time()-t0:.1f}s", flush=True)
        self._fh = None

    def _ensure_open(self):
        if self._fh is None:
            self._fh = h5py.File(self.path, 'r', swmr=True)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.preload:
            return (
                torch.from_numpy(self._x_e[idx]),
                torch.from_numpy(self._x_d[idx]),
                torch.tensor(self._y[idx], dtype=torch.float32),
            )
        self._ensure_open()
        X_e = self._fh['X_events'][idx].astype(np.float32)
        X_d = self._fh['X_demograficos'][idx].astype(np.float32)
        y = self._fh['y'][idx].astype(np.float32)
        return (
            torch.from_numpy(X_e),
            torch.from_numpy(X_d),
            torch.tensor(y, dtype=torch.float32),
        )


# ==================== EVALUACION V4 ====================

@torch.no_grad()
def evaluate_v4(model, loader, device, threshold=0.5):
    """Evaluacion completa adaptada al forward V4 (sin mascara)."""
    model.eval()
    ys = []; ps = []
    for X_e, X_d, y in loader:
        X_e = X_e.to(device, non_blocking=True)
        X_d = X_d.to(device, non_blocking=True)
        p = model(X_e, X_d).flatten().cpu().numpy()
        ys.append(y.numpy()); ps.append(p)
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    y_pred = (y_prob >= threshold).astype(int)

    try:    auc_roc = roc_auc_score(y_true, y_prob)
    except: auc_roc = float('nan')
    try:    auc_pr  = average_precision_score(y_true, y_prob)
    except: auc_pr  = float('nan')
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'sens': float(sens),
        'spec': float(spec),
        'tp': int(tp), 'fn': int(fn), 'fp': int(fp), 'tn': int(tn),
    }


# ==================== MAIN ====================

def main():
    global TRAIN_DATA, VAL_DATA, TEST_DATA
    if not TRAIN_DATA: TRAIN_DATA = find_latest_v4_split("train")
    if not VAL_DATA:   VAL_DATA   = find_latest_v4_split("val")
    if not TEST_DATA:  TEST_DATA  = find_latest_v4_split("test")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"modelo_{TAG}_pytorch_{ts}"

    print("=" * 72)
    print(f"CNN 2D V4 PyTorch -- {model_name}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Train: {TRAIN_DATA}")
    print(f"Val:   {VAL_DATA}")
    print(f"Test:  {TEST_DATA}")
    print(f"Out:   {OUTPUT_DIR}")
    print(f"Batch: {BATCH_SIZE} (reducido por VRAM)")
    print("=" * 72)

    notifier = MPCNotifier(enabled=True)
    notifier.notify_start(model_name, {
        'device': str(device),
        'tag': TAG,
        'arch': 'CNN2D V4 (Camino 2, Alex 18 May)',
        'batch_size': BATCH_SIZE,
        'max_epochs': EPOCHS,
        'patience': PATIENCE,
        'train_data': Path(TRAIN_DATA).name,
        'criterion': f'composite = AUC-PR + {J_WEIGHT}*Youden_J',
        'channels': f'1 -> {C1} -> {C2} -> {C3} -> {C_POOL}',
    })

    # --- Datasets ---
    print("\n[1/5] Cargando datasets V4...", flush=True)
    train_ds = HDF5DatasetV4(TRAIN_DATA, preload=True)
    val_ds   = HDF5DatasetV4(VAL_DATA,   preload=True)
    test_ds  = HDF5DatasetV4(TEST_DATA,  preload=True)
    print(f"  Train: n={len(train_ds):,}  X_e={train_ds.shape_e}  X_d={train_ds.shape_d}", flush=True)
    print(f"  Val:   n={len(val_ds):,}",  flush=True)
    print(f"  Test:  n={len(test_ds):,}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    # --- Modelo ---
    print("\n[2/5] Construyendo CNN2DModelV4...")
    n_features = train_ds.shape_e[1]   # 1541
    n_eventos  = train_ds.shape_e[2]   # 100
    n_demo     = train_ds.shape_d[1]   # 15
    print(f"  n_features={n_features}, n_eventos={n_eventos}, n_demograficos={n_demo}")

    model = CNN2DModelV4(
        n_features=n_features,
        n_eventos=n_eventos,
        n_demograficos=n_demo,
        c1=C1, c2=C2, c3=C3, c_pool=C_POOL,
        n_dense=N_DENSE, dropout=DROPOUT,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parametros: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=LR_REDUCE_FACTOR, patience=LR_REDUCE_PATIENCE,
    )

    # --- Training loop ---
    print("\n[3/5] Entrenando...", flush=True)
    history = {k: [] for k in [
        'loss', 'auc_roc', 'auc_pr', 'sens', 'spec',
        'val_auc_roc', 'val_auc_pr', 'val_sens', 'val_spec',
        'val_composite', 'lr',
    ]}
    best_score = -float('inf')
    best_metrics = None
    best_epoch = 0
    epochs_no_improve = 0
    best_path = OUTPUT_DIR / f"{model_name}_best.pt"
    t_start = time.time()

    for ep in range(1, EPOCHS + 1):
        model.train()
        ep_t0 = time.time()
        ep_loss = 0.0
        ep_n = 0
        for bi, (X_e, X_d, y) in enumerate(train_loader):
            if bi == 0 and ep == 1:
                print(f"  ep {ep} bi=0 batch shape={tuple(X_e.shape)} "
                      f"obtenido en {time.time()-ep_t0:.1f}s", flush=True)
            X_e = X_e.to(device, non_blocking=True)
            X_d = X_d.to(device, non_blocking=True)
            y   = y.to(device,   non_blocking=True)
            p = model(X_e, X_d).flatten()
            loss = hybrid_loss(p, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * y.size(0)
            ep_n += y.size(0)
        ep_loss /= ep_n

        train_metrics = evaluate_v4(model, train_loader, device, threshold=0.5)
        val_metrics   = evaluate_v4(model, val_loader,   device, threshold=0.5)
        score = composite_score(val_metrics)
        J_val = val_metrics['sens'] + val_metrics['spec'] - 1.0

        history['loss'].append(ep_loss)
        history['auc_roc'].append(train_metrics['auc_roc'])
        history['auc_pr'].append(train_metrics['auc_pr'])
        history['sens'].append(train_metrics['sens'])
        history['spec'].append(train_metrics['spec'])
        history['val_auc_roc'].append(val_metrics['auc_roc'])
        history['val_auc_pr'].append(val_metrics['auc_pr'])
        history['val_sens'].append(val_metrics['sens'])
        history['val_spec'].append(val_metrics['spec'])
        history['val_composite'].append(score)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        ep_t = time.time() - ep_t0
        print(f"  Ep {ep:>3}/{EPOCHS} | loss {ep_loss:.4f} | "
              f"val AUC {val_metrics['auc_roc']:.4f} "
              f"AUC-PR {val_metrics['auc_pr']:.4f} "
              f"sens {val_metrics['sens']:.3f} spec {val_metrics['spec']:.3f} "
              f"J {J_val:+.3f} composite {score:.4f} | "
              f"lr {optimizer.param_groups[0]['lr']:.1e} | {ep_t:.1f}s", flush=True)

        if score > best_score:
            best_score = score
            best_metrics = dict(val_metrics)
            best_epoch = ep
            torch.save(model.state_dict(), best_path)
            epochs_no_improve = 0
            print(f"     -> nuevo best (composite={best_score:.4f}, "
                  f"AUC-PR={val_metrics['auc_pr']:.4f}, J={J_val:+.3f}), guardado", flush=True)
        else:
            epochs_no_improve += 1

        scheduler.step(score)

        elapsed_total = time.time() - t_start
        notifier.notify_progress(
            ep, EPOCHS, elapsed_total,
            extra_info=(
                f"<b>CNN2D V4 Epoch {ep}/{EPOCHS}</b>\n"
                f"loss {ep_loss:.4f} | val AUC {val_metrics['auc_roc']:.4f}\n"
                f"AUC-PR {val_metrics['auc_pr']:.4f} | J {J_val:+.3f}\n"
                f"composite {score:.4f} (best {best_score:.4f} @ ep {best_epoch})\n"
                f"sens {val_metrics['sens']:.3f}  spec {val_metrics['spec']:.3f}"
            ),
        )

        if epochs_no_improve >= PATIENCE:
            print(f"     -> EarlyStopping en epoch {ep}")
            break

    train_time = time.time() - t_start

    # --- Test final ---
    print("\n[4/5] Evaluacion en test (best checkpoint)...")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_metrics    = evaluate_v4(model, test_loader, device, threshold=0.5)
    test_metrics_03 = evaluate_v4(model, test_loader, device, threshold=0.3)

    print(f"\n  Test results @ threshold 0.5:")
    for k, v in test_metrics.items():
        print(f"    {k:10}: {v}")

    # --- Guardar metadatos ---
    print("\n[5/5] Guardando outputs...")
    with open(OUTPUT_DIR / f"{model_name}_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    with open(OUTPUT_DIR / f"{model_name}_evaluacion.json", 'w') as f:
        json.dump({
            'auc_roc': test_metrics['auc_roc'],
            'auc_pr': test_metrics['auc_pr'],
            'threshold_05': test_metrics,
            'threshold_30': test_metrics_03,
            'best_score_composite': best_score,
            'best_epoch': best_epoch,
            'best_val_metrics': best_metrics,
            'training_time_seconds': train_time,
        }, f, indent=2, default=float)
    with open(OUTPUT_DIR / f"{model_name}_config.json", 'w') as f:
        json.dump({
            'timestamp': ts,
            'tag': TAG,
            'architecture': 'CNN2D V4 (Camino 2, Alex 18 May)',
            'channels': {'c1': C1, 'c2': C2, 'c3': C3, 'c_pool': C_POOL, 'n_dense': N_DENSE},
            'epochs_trained': len(history['loss']),
            'n_parameters': n_params,
            'best_score_composite': best_score,
            'best_epoch': best_epoch,
            'training_time_seconds': train_time,
            'batch_size': BATCH_SIZE,
            'lr': LEARNING_RATE,
            'dropout': DROPOUT,
            'patience': PATIENCE,
            'configuration': f'{TAG} -- PyTorch GPU CNN 2D V4',
            'parent_version': 'V4 HDF5 from scratch (sin overlap, sin masking, lat/lon + edad continua)',
            'train_data': TRAIN_DATA,
            'val_data':   VAL_DATA,
            'test_data':  TEST_DATA,
        }, f, indent=2)

    print(f"\nDONE -- best composite={best_score:.4f} at ep {best_epoch}, "
          f"test AUC-ROC={test_metrics['auc_roc']:.4f}, "
          f"test AUC-PR={test_metrics['auc_pr']:.4f}, "
          f"training={train_time/60:.1f} min", flush=True)

    notifier.notify_complete(
        model_name, train_time,
        kpis={
            'best_composite': f"{best_score:.4f} @ ep {best_epoch}/{len(history['loss'])}",
            'test_AUC_ROC': f"{test_metrics['auc_roc']:.4f}",
            'test_AUC_PR':  f"{test_metrics['auc_pr']:.4f}",
            'sens@0.5': f"{test_metrics['sens']:.3f}",
            'spec@0.5': f"{test_metrics['spec']:.3f}",
            'TP/FN @ 0.5': f"{test_metrics['tp']}/{test_metrics['fn']}",
        },
        config={
            'tag': TAG,
            'arch': 'CNN2D V4',
            'n_params': f"{n_params:,}",
            'epochs_trained': len(history['loss']),
        },
    )


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        tb = traceback.format_exc()
        try:
            n = MPCNotifier(enabled=True)
            n.notify_error(type(e).__name__, str(e), traceback_text=tb)
        except Exception:
            pass
        print(tb, flush=True)
        raise
