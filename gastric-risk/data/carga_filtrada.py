"""
carga_filtrada.py -- carga por batches que conserva solo las filas de los pacientes elegidos.

REPRODUCES IN THE THESIS
    Nothing directly. Shared by the cohort builder: reads the source matrices batch by
    batch and keeps only the rows of the selected patients, Section 4.2.

PROVENANCE
    Taken from the working tree that produced the reported results. The computational
    content is unchanged; comments were edited. See CHANGES-FROM-THESIS.md at the
    repository root.

--------------------------------------------------------------------------
Cada chunk de 150.000 filas se lee y se fusiona con read_csv por chunks y merge inner
por ID (Diag -> Proc -> Fecha -> Gen); de cada chunk fusionado se guardan solo las filas
elegidas (por indice de poblacion) y, del chunk entero, solo ID y Estomago_x (para los
controles de alineacion y de y del batch completo). La memoria por batch es la de un
chunk (~7-20 GB transitorios) mas ~2 % del batch, en lugar del batch completo (~70-80 GB).

Por que filtrar no cambia los datos:
  - Las columnas todo-NaN de cada chunk no se borran. Un evento exige diag, proc y fecha
    no nulos, asi que una columna NaN no aporta eventos; el dtype de cada columna lo fija
    read_csv sobre el chunk completo, antes de filtrar.
  - La conversion a category se hace sobre las filas elegidas (no cambia los valores).
  - Los encoders se ajustan con el Gen del batch 1 completo (mismas filas y valores que el
    batch 1 fusionado), porque el batch completo no se retiene en memoria.

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
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import psutil


def cargar_batches_filtrados(matrices_path, elegidos, filas_batch, solo_batches=None, max_chunks=0,
                             chunksize=150000, log=print):
    """Genera (batch_idx, offset, sub_df, ids_batch, y_batch_raw) por batch.
    elegidos: gidx elegidos (ordenados). filas_batch: filas por batch del paso 1.
    sub_df: filas elegidas del batch, en orden de poblacion, columnas como en el merge completo."""
    diag_batches = sorted(Path(matrices_path).glob("MatChar_Diag_batch_*.csv"))
    bounds = np.cumsum([0] + list(filas_batch))
    for batch_idx, diag_file in enumerate(diag_batches, 1):
        if solo_batches and batch_idx not in solo_batches:
            continue
        suf = diag_file.name.replace("MatChar_Diag_batch_", "").replace(".csv", "")
        paths = {k: Path(matrices_path) / f"MatChar_{k}_batch_{suf}.csv" for k in ("Diag", "Proc", "Fecha", "Gen")}
        readers = {k: pd.read_csv(p, chunksize=chunksize, low_memory=False) for k, p in paths.items()}
        offset = int(bounds[batch_idx - 1])
        pos = offset                               # gidx de la primera fila del chunk siguiente
        subs, ids_l, y_l = [], [], []
        log(f"\n[Batch {batch_idx}/{len(diag_batches)}] (carga filtrada a los elegidos)")
        for c, (dc, pc, fc, gc_) in enumerate(zip(readers['Diag'], readers['Proc'], readers['Fecha'], readers['Gen']), 1):
            dc = dc.rename(columns={dc.columns[0]: 'ID'})
            pc = pc.rename(columns={pc.columns[0]: 'ID'})
            fc = fc.rename(columns={fc.columns[0]: 'ID'})
            if gc_.columns[0] != 'ID':
                gc_ = gc_.rename(columns={gc_.columns[0]: 'ID'})
            m = dc.merge(pc, on='ID', how='inner', suffixes=('_diag', '_proc'))
            m = m.merge(fc, on='ID', how='inner')
            m = m.merge(gc_, on='ID', how='inner')
            del dc, pc, fc, gc_
            n = len(m)
            ids_l.append(m['ID'].values.copy())
            y_l.append(m['Estomago_x'].values.copy())
            sel = elegidos[(elegidos >= pos) & (elegidos < pos + n)] - pos
            sub = m.iloc[sel].copy()
            del m
            gc.collect()
            for col in sub.columns:                     # columnas object -> category (memoria)
                if sub[col].dtype == 'object':
                    try:
                        sub[col] = sub[col].astype('category')
                    except Exception:
                        pass
            subs.append(sub)
            pos += n
            log(f"  Chunk {c}: {n:,} filas, elegidas {len(sel):,} | mem {psutil.Process().memory_info().rss / 1024**3:.1f} GB")
            if max_chunks and c >= max_chunks:
                break
        sub_df = pd.concat(subs, ignore_index=True, copy=False) if subs else pd.DataFrame()
        del subs
        gc.collect()
        yield batch_idx, offset, sub_df, np.concatenate(ids_l), np.concatenate(y_l)
        del sub_df
        gc.collect()
