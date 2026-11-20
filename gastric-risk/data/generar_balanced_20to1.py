"""
generar_balanced_20to1.py -- second-stage undersampling of the training partition to 1:20.

REPRODUCES IN THE THESIS
    The 1:20 training set on which the CNN-2D model is trained, Section 4.4. Validation
    and test partitions stay at 1:100.

PROVENANCE
    Taken from the working tree that produced the reported results. The computational
    content is unchanged; absolute paths were replaced by relative defaults and
    environment variables, and comments were edited. See CHANGES-FROM-THESIS.md at the
    repository root.

--------------------------------------------------------------------------
Submuestreo de la particion de entrenamiento 100:1 a 20:1:
    RandomState(42).choice(negativos, 20 * n_positivos) + todos los positivos, ordenados.

  - SRC/DST por argumento: sirve para el train de la CNN (X_events) y para el del
    BiLSTM (X_ventanas).
  - Copia TODOS los datasets del origen (incluidos ID, gidx, X_temporal_raw) y
    los attrs (medias/desv. del z-score temporal), y anade stage2_ratio,
    stage2_seed y stage2_src (ruta del train 100:1 de origen).
  - Guarda junto al HDF5 un .filas.npz con las filas elegidas del train 100:1 y
    sus gidx/ID.
  - Comprobaciones opcionales de identidad de pacientes: contra un 20:1 de
    referencia (REF_TRAIN_H5, REF_20A1_H5) y, para la CNN, gidx/ID iguales a los
    del 20:1 del BiLSTM (--comparar-gidx). Sin referencia se avisa y se sigue.

Uso (solo h5py/numpy):
    python generar_balanced_20to1.py --src <datos_modelo_train_cnn_<ts>.h5> --dst-dir <dir>

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
from __future__ import annotations
import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

# ===== CONFIG =====
TARGET_RATIO = 20            # negatives:positives
RANDOM_SEED = 42
CHUNK_PATIENTS = 256         # patients copiados por chunk (limita RAM)
# Ficheros de referencia opcionales (otra construccion de la misma particion) para comprobar
# identidad: el train 100:1 de referencia y su 20:1. Sin la variable de entorno no se comparan.
V32_TRAIN = Path(os.environ["REF_TRAIN_H5"]) if os.environ.get("REF_TRAIN_H5") else None
V32_20A1 = Path(os.environ["REF_20A1_H5"]) if os.environ.get("REF_20A1_H5") else None
BILSTM_DATA_DIR_20A1 = Path("./batches/Datos_Modelo_BiLSTM")
N_BLOQUE_VENTANAS = 200   # filas del 20:1 en las que se comparan los embeddings de X_ventanas


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def filas_20to1(y, ratio=TARGET_RATIO, seed=RANDOM_SEED):
    """Todos los positivos + RandomState(seed).choice(negativos, ratio * n_pos), ordenados."""
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng = np.random.RandomState(seed)
    neg_chosen = rng.choice(neg_idx, size=len(pos_idx) * ratio, replace=False)
    return np.sort(np.concatenate([pos_idx, neg_chosen]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst-dir', required=True)
    ap.add_argument('--tag', default='')
    ap.add_argument('--sin-check-v32', action='store_true',
                    help='no hacer las comprobaciones contra los ficheros de referencia')
    ap.add_argument('--comparar-gidx', default=None,
                    help='(CNN) 20:1 BiLSTM con el que exigir gidx identicos; por defecto el mas reciente')
    a = ap.parse_args()
    SRC = Path(a.src)
    DST_DIR = Path(a.dst_dir); DST_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = SRC.stem.replace('datos_modelo_train_', '')
    dst = DST_DIR / f"datos_modelo_train_balanced_20to1_{stem}_{ts}.h5"
    log(f"Source: {SRC}")
    log(f"Dest:   {dst}")
    t0 = time.time()

    with h5py.File(SRC, 'r') as f:
        y = f['y'][:]
        keys = list(f.keys())
    selected = filas_20to1(y)
    n_out = len(selected)
    log(f"  Total {len(y):,} | pos {int(y.sum()):,} | seleccionados {n_out:,}")

    # Comprobacion opcional contra el train de referencia (mismas filas del train 100:1)
    if not a.sin_check_v32 and V32_TRAIN is not None and V32_TRAIN.exists():
        with h5py.File(V32_TRAIN, 'r') as f:
            y32 = f['y'][:]
        if len(y32) == len(y) and np.array_equal(y32, y):
            # Cierto por construccion si y es igual (no prueba pacientes); la prueba real se
            # hace despues contra el fichero 20:1 de referencia / el gidx del BiLSTM.
            log("  y del train identica a la referencia (las filas 20:1 coinciden por construccion; "
                "ver comprobacion directa al final)")
        else:
            log("  AVISO: y del train distinta de la referencia (build parcial): no se comprueba")

    dst_tmp = dst.with_name(dst.name + '.partial')        # escritura atomica: .partial y os.replace
    with h5py.File(SRC, 'r') as fi, h5py.File(dst_tmp, 'w') as fo:
        for k in keys:
            d = fi[k]
            if d.ndim == 1 or d.shape[0] != len(y):
                if d.shape[0] == len(y):
                    fo.create_dataset(k, data=d[:][selected])
                else:
                    fo.create_dataset(k, data=d[:])
                continue
            shp = (n_out, *d.shape[1:])
            ch = tuple(min(c, s_) for c, s_ in zip(d.chunks, shp)) if d.chunks else None
            fo.create_dataset(k, shape=shp, dtype=d.dtype,
                              chunks=ch, compression=d.compression,
                              compression_opts=d.compression_opts)
        for k, v in fi.attrs.items():
            fo.attrs[k] = v
        fo.attrs['stage2_ratio'] = TARGET_RATIO
        fo.attrs['stage2_seed'] = RANDOM_SEED
        fo.attrs['stage2_src'] = str(SRC)
        big = [k for k in keys if fi[k].ndim > 1 and fi[k].shape[0] == len(y)]
        for ci in range(0, n_out, CHUNK_PATIENTS):
            b = min(ci + CHUNK_PATIENTS, n_out)
            idxs = selected[ci:b].tolist()
            for k in big:
                fo[k][ci:b] = fi[k][idxs]
            if (ci // CHUNK_PATIENTS + 1) % 10 == 0 or b == n_out:
                log(f"   {b:,}/{n_out:,}")
        ids = fi['ID'][:][selected] if 'ID' in fi else None
        gidx = fi['gidx'][:][selected] if 'gidx' in fi else None
    os.replace(dst_tmp, dst)
    np.savez(dst.with_suffix('.filas.npz'), filas_train_100a1=selected,
             **({'ID': ids} if ids is not None else {}), **({'gidx': gidx} if gidx is not None else {}))
    with h5py.File(dst, 'r') as f:
        yo = f['y'][:]
        log(f"  pos/neg {int(yo.sum())} / {len(yo) - int(yo.sum())}  datasets {list(f.keys())}")
        es_bilstm = 'X_ventanas' in f
    comprobar_directo(dst, es_bilstm, a)
    log(f"DONE en {(time.time() - t0)/60:.1f} min -> {dst}")


def comprobar_directo(dst, es_bilstm, a):
    """Comparacion directa de pacientes (no solo de y).
    BiLSTM: el 20:1 contra el fichero 20:1 de referencia (X_demograficos y X_mascaras completos,
    y, y X_ventanas[..., :1536] en un bloque de filas).
    CNN: gidx del 20:1 CNN == gidx del 20:1 BiLSTM (mismos pacientes, mismo orden)."""
    if a.sin_check_v32:
        return
    if es_bilstm:
        if V32_20A1 is None or not V32_20A1.exists():
            log("  AVISO: no hay 20:1 de referencia; no se compara"); return
        with h5py.File(dst, 'r') as n5, h5py.File(V32_20A1, 'r') as o:
            if n5['y'].shape[0] != o['y'].shape[0]:
                log(f"  AVISO: tamanos distintos ({n5['y'].shape[0]} vs {o['y'].shape[0]}): no se compara")
                return
            r = {'y': np.array_equal(n5['y'][:], o['y'][:]),
                 'X_demograficos': np.array_equal(n5['X_demograficos'][:], o['X_demograficos'][:]),
                 'X_mascaras': np.array_equal(n5['X_mascaras'][:], o['X_mascaras'][:]),
                 'X_ventanas_emb_bloque': np.array_equal(n5['X_ventanas'][:N_BLOQUE_VENTANAS, ..., :1536],
                                                         o['X_ventanas'][:N_BLOQUE_VENTANAS, ..., :1536])}
        log(f"  20:1 BiLSTM vs 20:1 de referencia (comparacion directa): {r}")
        if not all(r.values()):
            raise RuntimeError("el 20:1 BiLSTM no coincide con el 20:1 de referencia")
    else:
        ref = a.comparar_gidx
        if not ref:
            c = sorted(BILSTM_DATA_DIR_20A1.glob("datos_modelo_train_balanced_20to1_bilstm_*.h5"))
            ref = str(c[-1]) if c else None
        if not ref:
            log("  AVISO: no hay 20:1 BiLSTM con el que comparar gidx (construirlo y repetir con "
                "--comparar-gidx)"); return
        with h5py.File(dst, 'r') as n5, h5py.File(ref, 'r') as b:
            ok = np.array_equal(n5['gidx'][:], b['gidx'][:]) and np.array_equal(n5['ID'][:], b['ID'][:])
        log(f"  gidx/ID del 20:1 CNN == 20:1 BiLSTM ({Path(ref).name}): {ok}")
        if not ok:
            raise RuntimeError("los 20:1 de CNN y BiLSTM no son los mismos pacientes")


if __name__ == "__main__":
    main()
