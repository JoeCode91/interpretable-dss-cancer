"""
Colinealidad de las representaciones atendidas -- evidencia para el PIVOTE del

REPRODUCES IN THE THESIS
    Collinearity of the attention representation, first run. Section 4.3.

PROVENANCE
    Copied verbatim from the working tree of the thesis. The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md at the
    repository root.

--------------------------------------------------------------------------
articulo del Cap. 4 (Seccion "Attention collinearity degrades attribution").

QUE MIDE
--------
El articulo afirma que la salida de la atencion multi-cabeza es una combinacion
lineal de la secuencia de entrada, de modo que las representaciones de distintas
posiciones temporales quedan casi colineales y una descomposicion que asigne
contribuciones a posiciones individuales no puede separarlas limpiamente.

Hasta ahora esa afirmacion es cualitativa. Este script la convierte en medida.

Tres comparaciones, de mas a menos concluyente:

  (A) CONTROL INTERNO, el argumento fuerte.
      En el MISMO modelo BiLSTM-Attention, con los MISMOS pacientes y la MISMA
      dimensionalidad (512), se comparan las representaciones de ventana
      ANTES de la atencion (lstm_masked) y DESPUES (attn_masked). Si la
      colinealidad aparece despues y no antes, la causa es la atencion y no el
      dato ni la arquitectura recurrente.

  (B) COMPARACION ENTRE ARQUITECTURAS.
      Las 12 columnas temporales del mapa de caracteristicas pre-pooling de la
      CNN-2D, que es el punto donde se aplica el wrapper W3 de NObSP.
      OJO: la dimension no es la misma (64 frente a 512), asi que el numero de
      condicion NO es directamente comparable entre arquitecturas. La similitud
      coseno si lo es, porque es invariante a la escala y robusta a la dimension.
      El script lo deja escrito en la salida para que no se compare de mas.

  (C) Rango efectivo por entropia de valores singulares, que resume cuantas
      direcciones independientes quedan realmente ocupadas.

METRICAS POR PACIENTE
---------------------
  - similitud coseno entre pares de posiciones validas (sin padding)
  - separacion angular en grados, que es arccos de la anterior
  - numero de condicion de la matriz (n_validas x D)
  - rango efectivo = exp(entropia de Shannon de los valores singulares
    normalizados). Va de 1 (todo colineal) a n_validas (todo ortogonal)

DONDE SE EJECUTA
----------------
PC de casa. Requiere los checkpoints de D:\\Proyecto_CC_Batches\\, que no estan
en Drive. No modifica ningun fichero existente: solo lee modelos y datos y
escribe resultados nuevos en XAI_Model\\colinealidad_results\\.

USO
---
    conda activate <entorno torch>   # el mismo de aplicar_nobsp_v4.py
    cd "G:\\Mi unidad\\Doctorado\\Proyecto_CC\\Test\\Book\\XAI_Model"
    python colinealidad_atencion.py

    # opcional, menos pacientes para una prueba rapida:
    python colinealidad_atencion.py --n 50

Autor: Johan Pena-Campos. Cap. 4, articulo AIM.
Fecha: 2026-08-25

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
import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import h5py
import torch

sys.path.insert(0, str(Path(__file__).parent))

# ==================== CONFIGURACION ====================

# BiLSTM-Attention (V3.2). Rutas tomadas de aplicar_nobsp_v32.py
BILSTM_PT = Path(r"D:\Proyecto_CC_Batches\Modelos_V32_PyTorch"
                 r"\modelo_v32_pytorch_20260510_151626_best.pt")
BILSTM_DATA_DIR = Path(r"D:\Proyecto_CC_Batches\Datos_Modelo")
BILSTM_DATA_GLOB = "datos_modelo_test_*.h5"

# CNN-2D (V4). Rutas tomadas de aplicar_nobsp_v4.py
CNN_MODEL_DIR = Path(r"D:\Proyecto_CC_Batches\Modelos_CNN2D_V4_PyTorch")
CNN_MODEL_GLOB = "modelo_cnn2d_v4_pytorch_*_best.pt"
CNN_DATA_DIR = Path(r"D:\Proyecto_CC_Batches\Datos_Modelo_V4")
CNN_DATA_GLOB = "datos_modelo_test_v4_*.h5"

OUTPUT_DIR = Path(__file__).parent / "colinealidad_results"

N_SAMPLES_DEFAULT = 500          # misma convencion que los scripts de NObSP
BATCH = 32                        # lotes para no saturar los 8 GB de VRAM
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================== METRICAS ====================

def metricas_colinealidad(V: np.ndarray):
    """Metricas de colinealidad de un conjunto de vectores fila.

    Args:
        V: (n, D) con n vectores de dimension D. Se asume que ya se han
           descartado las posiciones de padding.

    Returns:
        dict con las metricas, o None si n < 2 o los vectores son nulos.
    """
    if V.shape[0] < 2:
        return None

    norms = np.linalg.norm(V, axis=1)
    keep = norms > 1e-8
    V = V[keep]
    norms = norms[keep]
    if V.shape[0] < 2:
        return None

    # --- similitud coseno entre pares ---
    Vn = V / norms[:, None]
    S = Vn @ Vn.T
    iu = np.triu_indices(S.shape[0], k=1)
    cos = np.clip(S[iu], -1.0, 1.0)

    # --- numero de condicion y rango efectivo ---
    sv = np.linalg.svd(V, compute_uv=False)
    sv = sv[sv > 1e-12]
    cond = float(sv[0] / sv[-1]) if sv.size >= 2 else float('inf')
    p = sv / sv.sum()
    ent = float(-(p * np.log(p)).sum())
    eff_rank = float(np.exp(ent))

    return {
        'n_pos': int(V.shape[0]),
        'cos_mean': float(cos.mean()),
        'cos_median': float(np.median(cos)),
        'cos_p95': float(np.percentile(cos, 95)),
        'cos_abs_mean': float(np.abs(cos).mean()),
        'ang_mean_deg': float(np.degrees(np.arccos(np.clip(cos, -1, 1))).mean()),
        'ang_min_deg': float(np.degrees(np.arccos(np.clip(cos.max(), -1, 1)))),
        'cond': cond,
        'eff_rank': eff_rank,
        'eff_rank_frac': eff_rank / float(V.shape[0]),
    }


def resumir(nombre, filas, notas=""):
    """Agrega las metricas por paciente en un resumen legible."""
    if not filas:
        return {'bloque': nombre, 'n_pacientes': 0}
    keys = ['cos_mean', 'cos_median', 'cos_p95', 'cos_abs_mean',
            'ang_mean_deg', 'ang_min_deg', 'eff_rank', 'eff_rank_frac']
    out = {'bloque': nombre, 'n_pacientes': len(filas), 'notas': notas}
    for k in keys:
        v = np.array([f[k] for f in filas], dtype=float)
        out[k + '_mean'] = float(v.mean())
        out[k + '_median'] = float(np.median(v))
        out[k + '_q1'] = float(np.percentile(v, 25))
        out[k + '_q3'] = float(np.percentile(v, 75))
    conds = np.array([f['cond'] for f in filas if np.isfinite(f['cond'])], dtype=float)
    if conds.size:
        out['cond_median'] = float(np.median(conds))
        out['cond_q1'] = float(np.percentile(conds, 25))
        out['cond_q3'] = float(np.percentile(conds, 75))
    out['n_pos_median'] = float(np.median([f['n_pos'] for f in filas]))
    return out


def _resolver(dirpath: Path, patron: str, etiqueta: str) -> Path:
    cands = sorted(dirpath.glob(patron))
    if not cands:
        raise FileNotFoundError(
            "No se encontro %s en %s con el patron %s" % (etiqueta, dirpath, patron))
    return cands[-1]


# ==================== BLOQUE A: BiLSTM antes y despues de la atencion ====

def analizar_bilstm(n_samples):
    from Modelo_Predictivo_V3_PyTorch import BiLSTMAttentionModelV3

    data_pt = _resolver(BILSTM_DATA_DIR, BILSTM_DATA_GLOB, "datos test del BiLSTM")
    print("  datos  : %s" % data_pt.name, flush=True)
    print("  modelo : %s" % BILSTM_PT.name, flush=True)

    with h5py.File(data_pt, 'r') as f:
        X_vent = f['X_ventanas'][:n_samples].astype(np.float32)
        X_mask = f['X_mascaras'][:n_samples].astype(np.float32)
        y_all = f['y'][:n_samples]
    n, n_win, seq_len, n_feat = X_vent.shape
    print("  n=%d ventanas=%d seq=%d feat=%d  positivos=%d"
          % (n, n_win, seq_len, n_feat, int(y_all.sum())), flush=True)

    model = BiLSTMAttentionModelV3(n_features=n_feat).to(DEVICE)
    model.load_state_dict(torch.load(BILSTM_PT, map_location=DEVICE, weights_only=True))
    model.eval()

    filas_pre, filas_post = [], []
    with torch.no_grad():
        for i0 in range(0, n, BATCH):
            xb = torch.from_numpy(X_vent[i0:i0 + BATCH]).to(DEVICE)
            mb = torch.from_numpy(X_mask[i0:i0 + BATCH]).to(DEVICE)
            B = xb.shape[0]

            # Reproduce exactamente el forward de Modelo_Predictivo_V3_PyTorch
            x = xb.reshape(B * n_win, seq_len, n_feat)
            _, (h_n, _) = model.bilstm(x)
            lstm_out = torch.cat([h_n[0], h_n[1]], dim=-1).view(B, n_win, -1)
            mask_3d = mb.unsqueeze(-1)
            lstm_masked = lstm_out * mask_3d           # ANTES de la atencion
            attn_out = model.mha(lstm_masked)
            attn_masked = attn_out * mask_3d           # DESPUES de la atencion

            pre = lstm_masked.cpu().numpy()
            post = attn_masked.cpu().numpy()
            msk = mb.cpu().numpy()

            for j in range(B):
                val = msk[j] > 0.5
                if val.sum() < 2:
                    continue
                r_pre = metricas_colinealidad(pre[j][val])
                r_post = metricas_colinealidad(post[j][val])
                if r_pre and r_post:
                    r_pre['idx'] = i0 + j
                    r_post['idx'] = i0 + j
                    filas_pre.append(r_pre)
                    filas_post.append(r_post)
            print("    %d/%d" % (min(i0 + BATCH, n), n), end='\r', flush=True)
    print(" " * 30, end='\r')
    return filas_pre, filas_post


# ==================== BLOQUE B: CNN-2D pre-pooling ====================

def analizar_cnn(n_samples):
    from modelo_cnn2d_v4_pytorch import CNN2DModelV4

    model_pt = _resolver(CNN_MODEL_DIR, CNN_MODEL_GLOB, "modelo CNN-2D")
    data_pt = _resolver(CNN_DATA_DIR, CNN_DATA_GLOB, "datos test V4")
    print("  datos  : %s" % data_pt.name, flush=True)
    print("  modelo : %s" % model_pt.name, flush=True)

    with h5py.File(data_pt, 'r') as f:
        X_events = f['X_events'][:n_samples].astype(np.float32)
        y_all = f['y'][:n_samples]
    n = X_events.shape[0]
    print("  n=%d  positivos=%d" % (n, int(y_all.sum())), flush=True)

    model = CNN2DModelV4(n_features=1541, n_eventos=100, n_demograficos=15).to(DEVICE)
    model.load_state_dict(torch.load(model_pt, map_location=DEVICE, weights_only=True))
    model.eval()

    filas = []
    with torch.no_grad():
        for i0 in range(0, n, BATCH):
            xb = torch.from_numpy(X_events[i0:i0 + BATCH]).to(DEVICE)
            fm = model.forward_conv(xb)               # (B, C, H', W')
            # Un vector por columna temporal: se promedia el eje de embedding H'
            # y queda (B, W', C). Es el analogo del vector por ventana del BiLSTM.
            cols = fm.mean(dim=2).permute(0, 2, 1).cpu().numpy()
            for j in range(cols.shape[0]):
                r = metricas_colinealidad(cols[j])
                if r:
                    r['idx'] = i0 + j
                    filas.append(r)
            print("    %d/%d" % (min(i0 + BATCH, n), n), end='\r', flush=True)
    print(" " * 30, end='\r')
    return filas, int(fm.shape[1])


# ==================== FIGURA ====================

def figura(filas_pre, filas_post, filas_cnn, path_png):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print("  (sin figura, matplotlib no disponible: %s)" % e)
        return

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    datos = [[f['cos_mean'] for f in filas_pre],
             [f['cos_mean'] for f in filas_post]]
    etiquetas = ['BiLSTM\nbefore attention', 'BiLSTM\nafter attention']
    if filas_cnn:
        datos.append([f['cos_mean'] for f in filas_cnn])
        etiquetas.append('CNN-2D\npre-pooling')

    bp = ax[0].boxplot(datos, labels=etiquetas, showfliers=False, patch_artist=True)
    for patch, c in zip(bp['boxes'], ['#8ecae6', '#e07a5f', '#81b29a']):
        patch.set_facecolor(c)
    ax[0].set_ylabel('Mean pairwise cosine similarity')
    ax[0].axhline(0.0, color='grey', lw=0.8, ls=':')
    ax[0].set_title('(a) Similarity between temporal positions')

    datos2 = [[f['eff_rank_frac'] for f in filas_pre],
              [f['eff_rank_frac'] for f in filas_post]]
    if filas_cnn:
        datos2.append([f['eff_rank_frac'] for f in filas_cnn])
    bp2 = ax[1].boxplot(datos2, labels=etiquetas, showfliers=False, patch_artist=True)
    for patch, c in zip(bp2['boxes'], ['#8ecae6', '#e07a5f', '#81b29a']):
        patch.set_facecolor(c)
    ax[1].set_ylabel('Effective rank / number of valid positions')
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title('(b) Independent directions retained')

    fig.tight_layout()
    fig.savefig(path_png, dpi=300)
    fig.savefig(str(path_png).replace('.png', '.pdf'))
    print("  figura -> %s" % path_png.name, flush=True)


# ==================== MAIN ====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=N_SAMPLES_DEFAULT,
                    help='numero de pacientes (por defecto 500)')
    ap.add_argument('--skip-cnn', action='store_true',
                    help='solo el control interno del BiLSTM')
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()
    print("=" * 72)
    print("COLINEALIDAD DE LAS REPRESENTACIONES ATENDIDAS")
    print("inicio: %s   device: %s   n=%d"
          % (t0.strftime('%Y-%m-%d %H:%M:%S'), DEVICE, args.n))
    print("=" * 72)

    print("\n[1/3] BiLSTM-Attention: antes y despues de la atencion", flush=True)
    filas_pre, filas_post = analizar_bilstm(args.n)
    print("  pacientes con al menos 2 ventanas validas: %d" % len(filas_pre), flush=True)

    filas_cnn, c_pool = [], 0
    if not args.skip_cnn:
        print("\n[2/3] CNN-2D: columnas temporales pre-pooling", flush=True)
        try:
            filas_cnn, c_pool = analizar_cnn(args.n)
            print("  pacientes procesados: %d" % len(filas_cnn), flush=True)
        except Exception as e:
            print("  AVISO: bloque CNN omitido (%s)" % e, flush=True)

    print("\n[3/3] Resumen y guardado", flush=True)
    nota_dim = ("La dimension de los vectores difiere entre bloques "
                "(BiLSTM 512, CNN %d), luego el numero de condicion NO es "
                "comparable entre arquitecturas; la similitud coseno si lo es."
                % c_pool)
    resumen = {
        'fecha': t0.strftime('%Y-%m-%d %H:%M:%S'),
        'n_solicitado': args.n,
        'device': str(DEVICE),
        'bloques': [
            resumir('bilstm_pre_attention', filas_pre,
                    'Representacion de ventana a la salida del BiLSTM, antes de la atencion.'),
            resumir('bilstm_post_attention', filas_post,
                    'Misma representacion despues del bloque de atencion multi-cabeza. '
                    'Control interno: mismo modelo, mismos pacientes, misma dimension.'),
        ],
    }
    if filas_cnn:
        resumen['bloques'].append(
            resumir('cnn2d_pre_pooling', filas_cnn, nota_dim))

    # Delta del control interno, que es la cifra que va al articulo
    a = resumen['bloques'][0]
    b = resumen['bloques'][1]
    resumen['control_interno'] = {
        'cos_mean_antes': a['cos_mean_mean'],
        'cos_mean_despues': b['cos_mean_mean'],
        'delta_cos': b['cos_mean_mean'] - a['cos_mean_mean'],
        'ang_mean_deg_antes': a['ang_mean_deg_mean'],
        'ang_mean_deg_despues': b['ang_mean_deg_mean'],
        'eff_rank_frac_antes': a['eff_rank_frac_mean'],
        'eff_rank_frac_despues': b['eff_rank_frac_mean'],
    }

    with open(OUTPUT_DIR / 'resumen_colinealidad.json', 'w', encoding='utf-8') as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    import csv as _csv
    for nombre, filas in [('bilstm_pre', filas_pre),
                          ('bilstm_post', filas_post),
                          ('cnn2d_prepool', filas_cnn)]:
        if not filas:
            continue
        cols = ['idx', 'n_pos', 'cos_mean', 'cos_median', 'cos_p95', 'cos_abs_mean',
                'ang_mean_deg', 'ang_min_deg', 'cond', 'eff_rank', 'eff_rank_frac']
        with open(OUTPUT_DIR / ('por_paciente_%s.csv' % nombre), 'w',
                  newline='', encoding='utf-8') as f:
            w = _csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in filas:
                w.writerow({k: r.get(k, '') for k in cols})

    figura(filas_pre, filas_post, filas_cnn, OUTPUT_DIR / 'colinealidad_atencion.png')

    print("\n" + "=" * 72)
    print("RESULTADO PRINCIPAL (control interno, mismo modelo y pacientes)")
    print("=" * 72)
    ci = resumen['control_interno']
    print("  coseno medio entre posiciones   antes: %+.4f    despues: %+.4f   (delta %+.4f)"
          % (ci['cos_mean_antes'], ci['cos_mean_despues'], ci['delta_cos']))
    print("  separacion angular media        antes: %6.2f deg despues: %6.2f deg"
          % (ci['ang_mean_deg_antes'], ci['ang_mean_deg_despues']))
    print("  rango efectivo / n posiciones   antes: %.3f      despues: %.3f"
          % (ci['eff_rank_frac_antes'], ci['eff_rank_frac_despues']))
    if filas_cnn:
        c = resumen['bloques'][2]
        print("\n  CNN-2D pre-pooling  coseno medio: %+.4f   rango efectivo: %.3f"
              % (c['cos_mean_mean'], c['eff_rank_frac_mean']))
        print("  " + nota_dim)
    print("\nsalidas en: %s" % OUTPUT_DIR)
    print("duracion: %s" % (datetime.now() - t0))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
