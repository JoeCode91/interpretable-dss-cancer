"""
Helper compartido: carga el conjunto de calibracion train+test exacto

REPRODUCES IN THE THESIS
    Nothing directly. Loads, in the exact row order of calib_index.npz, the sample on
    which NObSP is computed (sample B), so that the scripts derived from it run on the
    same patients. Section 4.9.

PROVENANCE
    Taken from the working tree that produced the reported results. The computational
    content is unchanged; comments were edited. See CHANGES-FROM-THESIS.md at the
    repository root.

--------------------------------------------------------------------------
(guardado por aplicar_nobsp_v4_r2.py). Muestra B: todos los cancer de train (100:1) y
test, y el mismo numero de no cancer de train+test (RandomState(42)). Las rutas exactas
de los splits se leen de muestra.json (split_codes); el glob del mas reciente solo se
usa si falta ese fichero.

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
import glob
from pathlib import Path

import numpy as np
import h5py


def _split_files(v4_data_dir):
    tr = sorted(glob.glob(str(Path(v4_data_dir) / "datos_modelo_train_cnn_*.h5")))[-1]
    te = sorted(glob.glob(str(Path(v4_data_dir) / "datos_modelo_test_cnn_*.h5")))[-1]
    return tr, te


def load_r2_calib(r2_dir, v4_data_dir, need_events=True):
    """Devuelve (X_events|None, X_demo, y) en el orden exacto de calib_index.npz."""
    d = np.load(Path(r2_dir) / "calib_index.npz")
    idx, y = d['index'], d['y'].astype(int)
    # rutas exactas de la muestra B (muestra.json -> split_codes); glob solo si falta
    import json as _json
    mj = Path(r2_dir) / "muestra.json"
    if mj.exists():
        sc = _json.load(open(mj, encoding='utf-8'))['split_codes']
        paths = {0: sc['0'], 1: sc['1']}
    else:
        print("  AVISO r2_calib: sin muestra.json; se usa el glob (el mas reciente)")
        paths = dict(zip((0, 1), _split_files(v4_data_dir)))
    N = len(idx)
    X_events = np.zeros((N, 1541, 100), np.float32) if need_events else None
    X_demo = np.zeros((N, 15), np.float32)
    for sc, path in paths.items():
        pos = np.where(idx[:, 0] == sc)[0]
        if len(pos) == 0:
            continue
        rows = idx[pos, 1]
        order = np.argsort(rows)
        rs = rows[order]
        with h5py.File(path, 'r') as f:
            if need_events:
                X_events[pos[order]] = f['X_events'][rs].astype(np.float32)
            X_demo[pos[order]] = f['X_demograficos'][rs].astype(np.float32)
    return X_events, X_demo, y
