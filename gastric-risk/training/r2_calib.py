"""
Helper compartido R2: carga el conjunto de calibracion train+test exacto

REPRODUCES IN THE THESIS
    Calibration diagnostics on the held-out partition.

PROVENANCE
    Copied verbatim from the working tree of the thesis. The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md at the
    repository root.

--------------------------------------------------------------------------
Helper compartido R2: carga el conjunto de calibracion train+test exacto
(guardado por aplicar_nobsp_v4_r2.py) para que R1/R3/R4 se re-corran alineados.
"""
import glob
from pathlib import Path

import numpy as np
import h5py


def _split_files(v4_data_dir):
    tr = sorted(glob.glob(str(Path(v4_data_dir) / "datos_modelo_train_v4_*.h5")))[-1]
    te = sorted(glob.glob(str(Path(v4_data_dir) / "datos_modelo_test_v4_*.h5")))[-1]
    return tr, te


def load_r2_calib(r2_dir, v4_data_dir, need_events=True):
    """Devuelve (X_events|None, X_demo, y) en el orden exacto de calib_index.npz."""
    d = np.load(Path(r2_dir) / "calib_index.npz")
    idx, y = d['index'], d['y'].astype(int)
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
