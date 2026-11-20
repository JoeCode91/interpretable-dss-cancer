"""
seleccion.py -- seleccion de pacientes y escalado de las variables temporales.

REPRODUCES IN THE THESIS
    Nothing directly. Shared by the cohort builder: the 1:100 undersampling, the
    80/10/10 stratified split and the z-scoring of the five temporal features of the
    event vector, Section 4.2.

PROVENANCE
    Taken from the working tree that produced the reported results. The computational
    content is unchanged; comments were edited. See CHANGES-FROM-THESIS.md at the
    repository root.

--------------------------------------------------------------------------
1) y_poblacion(): etiqueta de los 4.915.663 pacientes en el orden de filas de la
   carga por batches: pd.read_csv por chunks de 150.000, zip de los 4 CSV, merge
   inner por ID Diag->Proc->Fecha->Gen, leyendo solo la columna ID de
   Diag/Proc/Fecha y ID+Estomago_x de Gen. El merge no depende de las columnas no
   clave, asi que el orden de filas es el mismo que con las columnas completas.
   y = Estomago_x.fillna(0).astype(int32); se comprueba que no hay nulos y se
   aborta si los hay.

2) seleccionar(): undersampling y split:
     np.random.seed(42); choice(idx_no_cancer, 100*n_cancer, replace=False);
     concat(cancer, no_cancer); np.random.shuffle; train_test_split 0,1 test
     estratificado seed 42; train_test_split 0,1/0,9 val estratificado seed 42.
   Devuelve, por split, el indice de fila en la poblacion (gidx) en el orden
   exacto de las filas del HDF5 de ese split.

3) stats_temporales() / escalar_*(): z-score de las 5 temporales con
   estadisticos de TRAIN sobre eventos reales. Evento real <=> columna C
   (indice del evento, 1..n) > 0; el relleno tiene C = 0 y se deja en 0.

AUTHORSHIP
    Johan S. Pena-Campos
    PhD thesis, "Interpretable Decision Support Systems for Early Cancer Diagnosis and
    Personalized Treatment", Universitat Politecnica de Catalunya and Pontificia
    Universidad Javeriana, 2026.

USE OF AI ASSISTANCE
    This file was written with the assistance of generative AI (Claude, Anthropic),
    to the author's specification, and reviewed, corrected and validated by the
    author, who takes full responsibility for it.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CHUNKSIZE = 150000          # chunks de lectura de los CSV fuente
COL_INDICE = 2              # posicion de C (indice del evento) dentro de las 5 temporales


def y_poblacion(matrices_path: Path, solo_gen: bool = False, log=print,
                solo_batches=None) -> Tuple[np.ndarray, np.ndarray, list]:
    """Devuelve (y_all int32, ids_all, filas_por_batch). solo_batches: solo para pruebas."""
    diag_batches = sorted(Path(matrices_path).glob("MatChar_Diag_batch_*.csv"))
    ys, ids, filas = [], [], []
    for b, diag_file in enumerate(diag_batches, 1):
        if solo_batches and b not in solo_batches:
            continue
        suf = diag_file.name.replace("MatChar_Diag_batch_", "").replace(".csv", "")
        p = {k: Path(matrices_path) / f"MatChar_{k}_batch_{suf}.csv" for k in ("Diag", "Proc", "Fecha", "Gen")}
        n_b = 0
        if solo_gen:
            # Atajo SOLO para pruebas: asume que los 4 CSV tienen los mismos IDs en el
            # mismo orden (lo que el merge inner preserva).
            g = pd.read_csv(p["Gen"], usecols=["ID", "Estomago_x"], low_memory=False)
            ys.append(g["Estomago_x"].values); ids.append(g["ID"].values); n_b = len(g)
        else:
            rd = {k: pd.read_csv(p[k], chunksize=CHUNKSIZE, usecols=[0], low_memory=False)
                  for k in ("Diag", "Proc", "Fecha")}
            rg = pd.read_csv(p["Gen"], chunksize=CHUNKSIZE, low_memory=False)
            for c, (dc, pc, fc, gc_) in enumerate(zip(rd["Diag"], rd["Proc"], rd["Fecha"], rg), 1):
                dc = dc.rename(columns={dc.columns[0]: 'ID'})
                pc = pc.rename(columns={pc.columns[0]: 'ID'})
                fc = fc.rename(columns={fc.columns[0]: 'ID'})
                if gc_.columns[0] != 'ID':
                    gc_ = gc_.rename(columns={gc_.columns[0]: 'ID'})
                gc_ = gc_[['ID', 'Estomago_x']]
                m = dc.merge(pc, on='ID', how='inner', suffixes=('_diag', '_proc'))
                m = m.merge(fc, on='ID', how='inner')
                m = m.merge(gc_, on='ID', how='inner')
                ys.append(m['Estomago_x'].values); ids.append(m['ID'].values); n_b += len(m)
                del dc, pc, fc, gc_, m
                gc.collect()
        filas.append(n_b)
        log(f"  [y poblacion] batch {b}: {n_b:,} filas")
    y_raw = np.concatenate(ys)
    if pd.isna(y_raw).any():
        raise ValueError("Estomago_x tiene nulos")
    y_all = pd.Series(y_raw).fillna(0).astype(np.int32).values
    ids_all = np.concatenate(ids)
    return y_all, ids_all, filas


def seleccionar(y_all: np.ndarray, ratio: int = 100, seed: int = 42,
                train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1) -> Dict:
    idx_c = np.where(y_all == 1)[0]
    idx_nc = np.where(y_all == 0)[0]
    n_keep = min(len(idx_c) * ratio, len(idx_nc))
    np.random.seed(seed)
    idx_nc_keep = np.random.choice(idx_nc, size=n_keep, replace=False)
    idx_final = np.concatenate([idx_c, idx_nc_keep])
    np.random.shuffle(idx_final)
    y_bal = y_all[idx_final]
    pos = np.arange(len(idx_final))
    pos_trval, pos_test = train_test_split(pos, test_size=test_ratio, stratify=y_bal, random_state=seed)
    val_adj = val_ratio / (train_ratio + val_ratio)
    pos_train, pos_val = train_test_split(pos_trval, test_size=val_adj,
                                          stratify=y_bal[pos_trval], random_state=seed)
    out = {'idx_final': idx_final, 'y_bal': y_bal}
    for name, p in (('train', pos_train), ('val', pos_val), ('test', pos_test)):
        out[name] = {'pos_balanced': p, 'gidx': idx_final[p], 'y': y_bal[p]}
    return out


def guardar_seleccion(path: Path, sel: Dict, ids_all: np.ndarray, y_all: np.ndarray):
    np.savez(path,
             idx_final=sel['idx_final'],
             **{f"{s}_gidx": sel[s]['gidx'] for s in ('train', 'val', 'test')},
             **{f"{s}_y": sel[s]['y'] for s in ('train', 'val', 'test')},
             **{f"{s}_id": ids_all[sel[s]['gidx']] for s in ('train', 'val', 'test')},
             n_poblacion=len(y_all), n_cancer_poblacion=int((y_all == 1).sum()))


# ------------------------- escalado temporal -------------------------

def stats_temporales(T: np.ndarray, eje_temporal: int) -> Dict:
    """T: temporales crudas de TRAIN. eje_temporal = eje de longitud 5.
    Media y desviacion (ddof=0, como StandardScaler) sobre eventos reales."""
    Tm = np.moveaxis(T, eje_temporal, -1).reshape(-1, 5).astype(np.float64)
    real = Tm[:, COL_INDICE] > 0
    R = Tm[real]
    mu = R.mean(axis=0)
    sd = R.std(axis=0)
    if np.any(sd <= 0):
        raise ValueError(f"desviacion nula en alguna temporal: {sd}")
    return {'mean': mu.tolist(), 'std': sd.tolist(), 'n_eventos_reales_train': int(real.sum()),
            'min': R.min(axis=0).tolist(), 'max': R.max(axis=0).tolist(),
            'p50': np.percentile(R, 50, axis=0).tolist(),
            'p95': np.percentile(R, 95, axis=0).tolist()}


def escalar_cnn(X: np.ndarray, T: np.ndarray, st: Dict, fila0: int = 1536) -> np.ndarray:
    """X (b,1541,100), T (b,5,100) crudas -> X con filas 1536-1540 z-score; relleno = 0."""
    mu = np.asarray(st['mean'], np.float64)[None, :, None]
    sd = np.asarray(st['std'], np.float64)[None, :, None]
    real = (T[:, COL_INDICE, :] > 0)[:, None, :]
    X[:, fila0:fila0 + 5, :] = np.where(real, (T - mu) / sd, 0.0).astype(np.float32)
    # En todo evento real z_C != 0 (C entera >= 1, media de train no entera), asi que
    # ninguna columna real queda a cero exacto aunque sus embeddings sean nulos (fuera de vocabulario)
    if np.any(X[:, fila0 + COL_INDICE, :][real[:, 0, :]] == 0):
        raise RuntimeError("evento real con z_C == 0: el relleno dejaria de ser distinguible")
    return X


def escalar_bilstm(X: np.ndarray, T: np.ndarray, st: Dict, fila0: int = 1536) -> np.ndarray:
    """X (b,10,40,1541), T (b,10,40,5) crudas -> X con features 1536-1540 z-score; relleno = 0."""
    mu = np.asarray(st['mean'], np.float64)
    sd = np.asarray(st['std'], np.float64)
    real = (T[..., COL_INDICE] > 0)[..., None]
    X[..., fila0:fila0 + 5] = np.where(real, (T - mu) / sd, 0.0).astype(np.float32)
    if np.any(X[..., fila0 + COL_INDICE][real[..., 0]] == 0):
        raise RuntimeError("evento real con z_C == 0: el relleno dejaria de ser distinguible")
    return X


def guardar_stats_json(path: Path, st: Dict, nombres, extra: Dict = None, n_train: int = None):
    d = {'columnas': nombres, 'filas_en_vector_1541': [1536, 1537, 1538, 1539, 1540],
         'ajustado_en': 'particion de entrenamiento 100:1 (%s pacientes), solo eventos reales'
                        % (f"{n_train:,}".replace(',', '.') if n_train is not None else '?'),
         'evento_real': 'columna C (indice del evento) > 0',
         'politica_relleno': 'relleno = 0 exacto en las 5 temporales tras el z-score '
                             '(igual que las 1536 filas de embedding). En un evento real z_C != 0 '
                             'siempre (C entera >= 1, media no entera; se comprueba), asi que el relleno '
                             'es distinguible aunque el evento tenga embeddings nulos (fuera de '
                             'vocabulario). Un 0 en otra temporal de un evento real significa '
                             '"valor medio de train"; el relleno queda en el centro de la distribucion '
                             'temporal y la red solo lo distingue por los embeddings y por C',
         **st}
    if extra:
        d.update(extra)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=2)
