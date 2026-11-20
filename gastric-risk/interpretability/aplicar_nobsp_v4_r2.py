"""
R2 -- NObSP sobre TRAIN + TEST (muestra B): que aprendio el modelo, no solo que generaliza.

REPRODUCES IN THE THESIS
    NObSP over train and test together, Section 4.9: what the model learned, not only
    what it generalises.

PROVENANCE
    Taken from the working tree that produced the reported results. The computational
    content is unchanged; absolute paths were replaced by relative defaults and
    environment variables, notification calls were removed, and comments were edited.
    See CHANGES-FROM-THESIS.md at the repository root.

--------------------------------------------------------------------------

Reutiliza los wrappers y run_nobsp de aplicar_nobsp_v4.py. Construye un conjunto de
calibracion BALANCEADO 50/50 (muestra B): TODOS los casos de cancer de train (100:1)
+ test (867) + igual numero de no-cancer de train+test muestreados con
RandomState(42), barajado. Corre los 4 wrappers (W1-W4) y guarda el INDICE EXACTO
del conjunto (`calib_index.npz`) para que la reconstruccion, el mapa, la
decodificacion de diagnosticos y la colinealidad se calculen sobre las mismas
muestras en el mismo orden.

W1 usa los datos reales de la muestra; W2-W4 usan 500 sondas multiplicativas
U[0.5, 1.5] alrededor de la MEDIA de la muestra.

La CNN se entrena con el submuestreo 20:1 del train; la muestra B toma sus no
cancer del train 100:1 y del test, de modo que parte de ellos no se vio en el
entrenamiento. `calib_index.npz` lleva por fila `en_20a1` para estratificar.

Los ficheros de datos salen del config del modelo fijado con CNN2D_TS (test) y del
atributo stage2_src de su train 20:1 (train 100:1 de origen).

Salidas en NOBSP_RUNB_DIR (por defecto ./batches/Resultados/nobsp_runB_traintest):
    {demo,input_event,pre_pool,post_pool}_explainer.npz  + importancias + plots
    calib_index.npz   (split, row, y, en_20a1 -- el conjunto exacto usado, en orden)
    calib_ids.npy     (ID de paciente de cada fila)
    muestra.json      (definicion de la muestra, ficheros y checkpoint con SHA-256)

Opcional: NOBSP_CALIB_REF apunta a un calib_index.npz de referencia; si existe, se
informa si la muestra coincide con el (mismas filas, mismo orden).

Ejecutar (GPU, ~1.5-5 h por el wrapper W2):
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
import sys
import time
from pathlib import Path

import numpy as np
import h5py
import torch
import torch.nn.functional as F

# Modulos hermanos del repositorio
_HERE = Path(__file__).resolve().parent
for _d in (_HERE, _HERE.parent / "models", _HERE.parent / "training"):
    sys.path.insert(0, str(_d))

import aplicar_nobsp_v4 as base   # wrappers, run_nobsp, plots, scalers

OUTPUT_DIR = Path(os.environ.get('NOBSP_RUNB_DIR', "./batches/Resultados/nobsp_runB_traintest"))
# Indice de referencia opcional, para comprobar que la muestra es la misma (si no existe, se omite)
CALIB_V4 = Path(os.environ.get('NOBSP_CALIB_REF', "./nobsp_reference/calib_index.npz"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
BERT2 = 2 * 768
# W2/W3/W4 usan PESOS DE PERTURBACION aleatorios (no los datos reales): su N es
# solo cuantas muestras de estimacion. Fijarlo (=metodologia original) evita el
# OOM de W2 al reconstruir tensores (1541x100) completos en la GPU de 8 GB.
# Solo W1 (demo) usa los datos reales (las N del conjunto balanceado).
N_PERTURB = int(os.environ.get('R2_N_PERTURB', '500'))   # bajar a 300 si W2 da OOM


def split_files():
    # Los ficheros salen del config del modelo fijado por CNN2D_TS (test) y del atributo
    # stage2_src de su train 20:1 (train 100:1 de origen), no de "el mas reciente"
    import checkpoint_utils as ck
    d = ck.datos_cnn()
    return str(d['train']), str(d['test'])


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
    print(f"Salida: {OUTPUT_DIR}", flush=True)

    # 1. Conjunto balanceado + guardar indice
    idx, y = build_balanced_index()
    np.savez(OUTPUT_DIR / "calib_index.npz", index=idx, y=y, seed=SEED)
    print(f"  indice guardado en calib_index.npz", flush=True)
    # Comparacion opcional con un indice de referencia (mismas filas, mismo orden) + IDs
    import json as _json
    misma = None
    if CALIB_V4.exists():
        old = np.load(CALIB_V4)
        misma = bool(np.array_equal(old['index'], idx) and np.array_equal(old['y'], y))
        print(f"  indice identico al de referencia ({CALIB_V4}): {misma}", flush=True)
    tr_, te_ = split_files()
    ids = np.zeros(len(idx), dtype=np.int64)
    for sc, pth in ((0, tr_), (1, te_)):
        with h5py.File(pth, 'r') as fh:
            if 'ID' in fh:
                pos = np.where(idx[:, 0] == sc)[0]
                ids[pos] = fh['ID'][:][idx[pos, 1]]
    np.save(OUTPUT_DIR / "calib_ids.npy", ids)
    # Marca por fila de si el paciente esta en el 20:1 con el que se entreno la CNN
    import checkpoint_utils as ck
    filas20 = set(np.load(ck.datos_cnn()['filas_20a1'])['filas_train_100a1'].tolist())
    en_20a1 = np.array([(sc == 0 and int(r) in filas20) for sc, r in idx], dtype=bool)
    np.savez(OUTPUT_DIR / "calib_index.npz", index=idx, y=y, seed=SEED, en_20a1=en_20a1)
    print(f"  en el 20:1 de entrenamiento: cancer {int(en_20a1[y == 1].sum())}/{int((y == 1).sum())}, "
          f"no cancer {int(en_20a1[y == 0].sum())}/{int((y == 0).sum())}", flush=True)
    model_pt_ = ck.cnn_pt()
    with open(OUTPUT_DIR / "muestra.json", "w", encoding="utf-8") as fh:
        _json.dump({"muestra": "B", "descripcion": "muestra B = conjunto balanceado train+test (867 cancer + 867 no cancer, seed 42, calib_index.npz); W2-W4 con 500 sondas alrededor de la media de esa muestra",
                    "n": int(len(idx)), "n_cancer": int(y.sum()),
                    "split_codes": {"0": tr_, "1": te_},
                    "identica_a_referencia": misma,
                    "modelo": ck.registro(model_pt_),
                    "en_20a1": {"cancer": int(en_20a1[y == 1].sum()), "no_cancer": int(en_20a1[y == 0].sum())},
                    "nota_20a1": "el modelo se entreno con el 20:1 (16.191); los no cancer de la muestra que "
                                 "vienen de train pueden no haber sido vistos en entrenamiento",
                    "salidas": ["demo_*", "input_event_*", "pre_pool_*", "post_pool_*", "recon_input_*",
                                "recon_w3_*", "r1_geo_por_departamento.csv (Tabla S1)", "pub_w1_geo_*",
                                "r3_*", "atribucion_individual.txt"]},
                   fh, indent=2)
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

    # 5. W1 demo
    wrapper_demo = base.V4DemoWrapper(model, pooled_ref).to(base.DEVICE).eval()
    base.run_nobsp("demo", wrapper_demo, X_demo, base.FEATURE_NAMES_DEMO, OUTPUT_DIR,
                   top_n_curves=None, x_for_plot=X_demo_real)

    # 6. W2 input por evento (CARO) -- N_PERTURB perturbaciones (no datos reales)
    if base.DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    wrapper_input = base.V4InputEventWrapper(model, x_events_ref, x_demo_ref).to(base.DEVICE).eval()
    X_in = (1.0 + rng.uniform(-0.5, 0.5, size=(N_PERTURB, 100))).astype(np.float32)
    base.run_nobsp("input_event", wrapper_input, X_in, base.EVENT_NAMES, OUTPUT_DIR,
                   top_n_curves=base.TOP_N_W2,
                   extra_plot_fn=lambda c, d, n: base.plot_temporal_contribution_heatmap(
                       c, d / f"{n}_temporal_heatmap.png"))

    # 7. W3 pre-pool
    wrapper_pre = base.V4PreToolWrapper(model, feat_map_ref, x_demo_ref).to(base.DEVICE).eval()
    X_pre = (1.0 + rng.uniform(-0.5, 0.5, size=(N_PERTURB, W_pool))).astype(np.float32)
    base.run_nobsp("pre_pool", wrapper_pre, X_pre, bucket_names, OUTPUT_DIR,
                   top_n_curves=None,
                   extra_plot_fn=lambda c, d, n: base.plot_bucket_bar(
                       np.abs(c).mean(axis=(0, 2)), bucket_names, d / f"{n}_bucket_bar.png",
                       f"NObSP W3 R2: importancia por bucket (W_pool={W_pool})"))

    # 8. W4 post-pool
    wrapper_pool = base.V4PostPoolWrapper(model, pooled_ref, x_demo_ref).to(base.DEVICE).eval()
    X_pool = (1.0 + rng.uniform(-0.5, 0.5, size=(N_PERTURB, 64))).astype(np.float32)
    base.run_nobsp("post_pool", wrapper_pool, X_pool, base.POOL_NAMES, OUTPUT_DIR,
                   top_n_curves=base.TOP_N_W4)

    dt = time.time() - t0
    print(f"\nDONE R2 en {dt/60:.1f} min. Salida: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
