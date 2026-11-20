"""
checkpoint_utils.py -- fixes the model checkpoint, and the data it was trained on, for
every script that runs after training.

REPRODUCES IN THE THESIS
    Nothing directly. Shared helper of the scripts that evaluate and interpret the trained
    models: it resolves the checkpoint by timestamp and records its SHA-256 in their
    outputs, so that every reported figure can be traced to one set of weights.

PROVENANCE
    Taken from the working tree that produced the reported results. The computational
    content is unchanged; absolute paths were replaced by relative defaults, and comments
    were edited. See CHANGES-FROM-THESIS.md at the repository root.

--------------------------------------------------------------------------
El checkpoint de cada script posterior al entrenamiento se fija por timestamp
(variable de entorno), nunca por "el mas reciente", y su ruta + SHA-256 se
registran en las salidas.

  CNN    : CNN2D_TS (obligatoria), CNN2D_OUT (por defecto ./batches/Modelos_CNN2D_PyTorch),
           CNN2D_TAG (por defecto cnn2d)
  BiLSTM : BILSTM_TS (obligatoria donde se use), BILSTM_OUT, BILSTM_TAG (por defecto bilstm)

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
import hashlib
import json
import os
from pathlib import Path

CNN_DIR_DEF = "./batches/Modelos_CNN2D_PyTorch"
BILSTM_DIR_DEF = "./batches/Modelos_BiLSTM_PyTorch"


def _ruta(var_ts, var_out, var_tag, dir_def, tag_def, sufijo):
    ts = os.environ.get(var_ts, "").strip()
    if not ts:
        raise SystemExit(f"Falta la variable de entorno {var_ts} (timestamp del modelo a usar). "
                         f"Se exige para no tomar por error otro checkpoint.")
    d = Path(os.environ.get(var_out, dir_def))
    p = d / f"modelo_{os.environ.get(var_tag, tag_def)}_pytorch_{ts}{sufijo}"
    if not p.exists():
        raise SystemExit(f"No existe {p}")
    return p


def cnn_pt():
    return _ruta("CNN2D_TS", "CNN2D_OUT", "CNN2D_TAG", CNN_DIR_DEF, "cnn2d", "_best.pt")


def bilstm_pt():
    return _ruta("BILSTM_TS", "BILSTM_OUT", "BILSTM_TAG", BILSTM_DIR_DEF, "bilstm", "_best.pt")


def cnn_fichero(sufijo):
    return _ruta("CNN2D_TS", "CNN2D_OUT", "CNN2D_TAG", CNN_DIR_DEF, "cnn2d", sufijo)


def bilstm_fichero(sufijo):
    return _ruta("BILSTM_TS", "BILSTM_OUT", "BILSTM_TAG", BILSTM_DIR_DEF, "bilstm", sufijo)


def _datos(fichero_cfg):
    """Ficheros de datos con los que se entreno el modelo fijado: val/test del config, el 20:1
    de entrenamiento, el train 100:1 de origen (attr stage2_src del 20:1) y sus filas (.filas.npz)."""
    import h5py
    cfg = json.load(open(fichero_cfg("_config.json"), encoding='utf-8'))
    tr20 = Path(cfg['train_data'])
    with h5py.File(tr20, 'r') as f:
        tr100 = Path(str(f.attrs['stage2_src']))
    return {'train': tr100, 'train_20a1': tr20, 'val': Path(cfg['val_data']),
            'test': Path(cfg['test_data']), 'filas_20a1': tr20.with_suffix('.filas.npz')}


def datos_cnn():
    return _datos(cnn_fichero)


def datos_bilstm():
    return _datos(bilstm_fichero)


def sha256(path, bloque=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(bloque), b''):
            h.update(b)
    return h.hexdigest()


def registro(path):
    """Diccionario para guardar en los JSON de salida."""
    return {"checkpoint": str(path), "sha256": sha256(path)}


def escribir_registro(out_dir, nombre, extra=None):
    """Anade/actualiza out_dir/procedencia.json con una entrada por script."""
    out_dir = Path(out_dir)
    p = out_dir / "procedencia.json"
    d = json.load(open(p, encoding='utf-8')) if p.exists() else {}
    d[nombre] = extra or {}
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=2, default=str)
