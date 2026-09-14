"""
R2 -- NObSP sobre TRAIN + TEST (solicitud S2 de Alex): "estas curvas las puedes

REPRODUCES IN THE THESIS
    NObSP over train and test together, Section 4.9: what the model learned, not only
    what it generalises.

PROVENANCE
    Copied verbatim from the working tree of the thesis. The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md at the
    repository root.

--------------------------------------------------------------------------
sacar con training tambien... es ver que aprendio el modelo".

Reutiliza los wrappers y run_nobsp de aplicar_nobsp_v4.py (sin tocarlo). Construye
un conjunto de calibracion BALANCEADO ~50/50: TODOS los casos de cancer de
train+test (~867) + igual numero de no-cancer muestreados (seed 42). Corre los 4
wrappers (W1-W4) y guarda el INDICE EXACTO del conjunto (`calib_index.npz`) para
que R1/R3/R4 se re-corran sobre las mismas muestras en el mismo orden.

Salidas en nobsp_results_v4_traintest/:
    {demo,input_event,pre_pool,post_pool}_explainer.npz  + importancias + plots
    calib_index.npz   (split, row, y -- el conjunto exacto usado, en orden)

Ejecutar (GPU, ~4-5 h por el wrapper W2):
    python -u aplicar_nobsp_v4_r2.py

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


import glob
import os
import time
from pathlib import Path

import numpy as np
import h5py
import torch
import torch.nn.functional as F

import aplicar_nobsp_v4 as base   # reutiliza wrappers, run_nobsp, plots, scalers

OUTPUT_DIR = Path(r".\nobsp_results_v4_traintest")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
BERT2 = 2 * 768
# W2/W3/W4 usan PESOS DE PERTURBACION aleatorios (no los datos reales): su N es
# solo cuantas muestras de estimacion. Fijarlo (=metodologia original) evita el
# OOM de W2 al reconstruir tensores (1541x100) completos en la GPU de 8 GB.
# Solo W1 (demo) usa los datos reales (las N del conjunto balanceado).
N_PERTURB = int(os.environ.get('R2_N_PERTURB', '500'))   # bajar a 300 si W2 da OOM


def split_files():
    tr = sorted(glob.glob(str(base.V4_DATA_DIR / "datos_modelo_train_v4_*.h5")))[-1]
    te = sorted(glob.glob(str(base.V4_DATA_DIR / "datos_modelo_test_v4_*.h5")))[-1]
    return tr, te


def build_balanced_index():
    """TODOS los cancer de train+test + igual no-cancer (seed). Devuelve
    arreglo ordenado (split, row) y las etiquetas, ya barajado."""
    tr, te = split_files()
    rng = np.random.RandomState(SEED)
    cancer, noncancer = [], []
    for split_code, path in [(0, tr), (1, te)]:
        with h5py.File(path, 'r') as f:
            y = f['y'][:].astype(int)
        for r in np.where(y == 1)[0]:
            cancer.append((split_code, int(r)))
        for r in np.where(y == 0)[0]:
            noncancer.append((split_code, int(r)))
    n_cancer = len(cancer)
    sel_non = [noncancer[i] for i in rng.choice(len(noncancer), size=n_cancer, replace=False)]
    combined = cancer + sel_non                       # 2*n_cancer
    order = rng.permutation(len(combined))
    combined = [combined[i] for i in order]
    idx = np.array(combined, dtype=np.int64)          # (N,2): split, row
    yv = np.array([1] * n_cancer + [0] * n_cancer, dtype=np.int64)[order]
    print(f"  balanceado: {n_cancer} cancer + {n_cancer} no-cancer = {len(idx)} muestras", flush=True)
    return idx, yv


def gather(idx):
    """Lee X_events/X_demo para las filas de idx, en el orden de idx."""
    tr, te = split_files()
    paths = {0: tr, 1: te}
    N = len(idx)
    X_events = np.zeros((N, 1541, 100), dtype=np.float32)
    X_demo = np.zeros((N, 15), dtype=np.float32)
    for split_code, path in paths.items():
        pos = np.where(idx[:, 0] == split_code)[0]    # posiciones en el conjunto
        if len(pos) == 0:
            continue
        rows = idx[pos, 1]
        order = np.argsort(rows)                       # h5py requiere indices crecientes
        rows_sorted = rows[order]
        with h5py.File(path, 'r') as f:
            ev = f['X_events'][rows_sorted].astype(np.float32)
            dm = f['X_demograficos'][rows_sorted].astype(np.float32)
        X_events[pos[order]] = ev
        X_demo[pos[order]] = dm
        print(f"  split {split_code}: {len(rows)} filas leidas", flush=True)
    return X_events, X_demo


def main():
    t0 = time.time()
    notifier = base.MPCNotifier(enabled=True)
    print(f"Salida: {OUTPUT_DIR}", flush=True)

    # 1. Conjunto balanceado + guardar indice
    idx, y = build_balanced_index()
    np.savez(OUTPUT_DIR / "calib_index.npz", index=idx, y=y, seed=SEED)
    print(f"  indice guardado en calib_index.npz", flush=True)
    X_events, X_demo = gather(idx)
    N = len(idx)

    # 2. Scalers + denormalizacion (igual que base)
    meta_pt = base.find_latest_v4_metadata(base.V4_DATA_DIR)
    scalers, _ = base.load_scalers_from_metadata(meta_pt)
    X_demo_real = base.denormalize_demo_continuous(X_demo, scalers)

    # 3. Modelo
    model_pt = base.find_latest_v4_checkpoint(base.MODEL_DIR)
    model = base.CNN2DModelV4(n_features=1541, n_eventos=100, n_demograficos=15).to(base.DEVICE)
    model.load_state_dict(torch.load(model_pt, map_location=base.DEVICE, weights_only=True))
    model.eval()
    print(f"  modelo cargado: {model_pt}", flush=True)

    # 4. Referencias (media del conjunto)
    x_events_ref = torch.from_numpy(X_events.mean(0)).to(base.DEVICE)
    x_demo_ref = torch.from_numpy(X_demo.mean(0)).to(base.DEVICE)
    with torch.no_grad():
        feat_map_ref = model.forward_conv(x_events_ref.unsqueeze(0)).squeeze(0)
        pooled_ref = F.adaptive_avg_pool2d(feat_map_ref.unsqueeze(0), 1).squeeze(-1).squeeze(-1).squeeze(0)
    W_pool = feat_map_ref.shape[-1]
    bucket_names = [f"tb{i:02d}" for i in range(W_pool)]
    rng = np.random.RandomState(SEED)

    notifier.notify_start("NObSP V4 R2 (train+test balanceado)",
                          {'N': N, 'output': str(OUTPUT_DIR), 'device': str(base.DEVICE)})

    # 5. W1 demo
    wrapper_demo = base.V4DemoWrapper(model, pooled_ref).to(base.DEVICE).eval()
    base.run_nobsp("demo", wrapper_demo, X_demo, base.FEATURE_NAMES_DEMO, OUTPUT_DIR, notifier,
                   top_n_curves=None, x_for_plot=X_demo_real)

    # 6. W2 input por evento (CARO) -- N_PERTURB perturbaciones (no datos reales)
    if base.DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    wrapper_input = base.V4InputEventWrapper(model, x_events_ref, x_demo_ref).to(base.DEVICE).eval()
    X_in = (1.0 + rng.uniform(-0.5, 0.5, size=(N_PERTURB, 100))).astype(np.float32)
    base.run_nobsp("input_event", wrapper_input, X_in, base.EVENT_NAMES, OUTPUT_DIR, notifier,
                   top_n_curves=base.TOP_N_W2,
                   extra_plot_fn=lambda c, d, n: base.plot_temporal_contribution_heatmap(
                       c, d / f"{n}_temporal_heatmap.png"))

    # 7. W3 pre-pool
    wrapper_pre = base.V4PreToolWrapper(model, feat_map_ref, x_demo_ref).to(base.DEVICE).eval()
    X_pre = (1.0 + rng.uniform(-0.5, 0.5, size=(N_PERTURB, W_pool))).astype(np.float32)
    base.run_nobsp("pre_pool", wrapper_pre, X_pre, bucket_names, OUTPUT_DIR, notifier,
                   top_n_curves=None,
                   extra_plot_fn=lambda c, d, n: base.plot_bucket_bar(
                       np.abs(c).mean(axis=(0, 2)), bucket_names, d / f"{n}_bucket_bar.png",
                       f"NObSP W3 R2: importancia por bucket (W_pool={W_pool})"))

    # 8. W4 post-pool
    wrapper_pool = base.V4PostPoolWrapper(model, pooled_ref, x_demo_ref).to(base.DEVICE).eval()
    X_pool = (1.0 + rng.uniform(-0.5, 0.5, size=(N_PERTURB, 64))).astype(np.float32)
    base.run_nobsp("post_pool", wrapper_pool, X_pool, base.POOL_NAMES, OUTPUT_DIR, notifier,
                   top_n_curves=base.TOP_N_W4)

    dt = time.time() - t0
    print(f"\nDONE R2 en {dt/60:.1f} min. Salida: {OUTPUT_DIR}", flush=True)
    notifier.notify_complete("NObSP V4 R2", dt, kpis={'N': N}, config={'output_dir': str(OUTPUT_DIR)})


if __name__ == "__main__":
    main()
