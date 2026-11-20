"""
Colinealidad de las representaciones atendidas: BiLSTM-Attention frente a CNN-2D.

REPRODUCES IN THE THESIS
    Collinearity of the attention representation, paired control. Section 4.3.

PROVENANCE
    Taken from the working tree that produced the reported results. The computational
    content is unchanged; absolute paths were replaced by relative defaults and
    environment variables, notification calls were removed, and comments were edited.
    See CHANGES-FROM-THESIS.md at the repository root.

--------------------------------------------------------------------------

Mide cuan colineales son las representaciones por posicion temporal de los dos
modelos, paciente a paciente: similitud coseno media entre pares de posiciones,
angulo, numero de condicion y rango efectivo (normalizado por n).

  Bloque A, control interno del BiLSTM: salida del BiLSTM por ventana antes y
  despues del bloque de atencion. Mismo modelo, mismos pacientes, misma dimension
  (512) y mismo n en ambas ramas; solo ventanas validas (mascara).

  Bloque B, CNN-2D: columnas temporales del mapa pre-pooling. El tensor de la CNN
  es fijo, 1541 x 100, con los eventos empaquetados a la izquierda y ceros a la
  derecha, y el relleno tras convolucion, normalizacion y ReLU NO es cero (los
  sesgos producen activaciones constantes), de modo que las columnas de relleno son
  casi identicas entre si e inflarian la colinealidad. Por eso se calcula la
  ocupacion real por paciente y se descartan las columnas cuyo campo receptivo no
  toca ningun evento real. Campo receptivo de la columna c de conv4: [8c-15, 8c+22]
  recortado a 0-99 (paso 8, campo 38). Como sensibilidad se informa tambien el
  criterio aproximado [8c, 8c+22) y el bloque sin enmascarar.

  Comparacion entre arquitecturas: el rango efectivo va normalizado por n y el
  BiLSTM aporta una mediana de pocas ventanas validas, asi que se calcula un bloque
  de la CNN SUBMUESTREADO al mismo numero de posiciones que la mediana del BiLSTM,
  y para comparar se usa SOLO la similitud coseno, que es robusta a la dimension.

Se guarda la etiqueta y se informa cuantos positivos hay en cada bloque, y la
DISTRIBUCION DE OCUPACION de la representacion ventaneada (cuantos pacientes tienen
k ventanas validas), evidencia para la seccion del confusor de longitud de historia.

MUESTRA
-------
Muestra B (calib_index.npz en NOBSP_RUNB_DIR): todos los cancer de train (100:1) y
test, y el mismo numero de no cancer de train+test (RandomState(42)), la misma para
los dos modelos y en el mismo orden; se exige que los IDs coincidan. --n por defecto
toma toda la muestra. Los checkpoints se fijan con CNN2D_TS y BILSTM_TS
(training/checkpoint_utils.py) y su SHA-256 queda en el JSON.

El bloque A necesita el modelo BiLSTM-Attention (BiLSTMAttentionModelV3, del modulo
Modelo_Predictivo_V3_PyTorch) y sus HDF5 ventaneados (X_ventanas, X_mascaras), que
no forman parte de este repositorio.

SALIDAS
-------
COLLINEARITY_DIR (por defecto ./batches/Resultados/colinealidad_runB):
resumen_colinealidad.json, por_paciente_*.csv, colinealidad_v2.{png,pdf}.

USO
---
    python colinealidad_atencion_v2.py [--n N] [--skip-cnn]

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

import argparse
import collections
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import h5py
import torch

# Modulos hermanos del repositorio
_HERE = Path(__file__).resolve().parent
for _d in (_HERE, _HERE.parent / "models", _HERE.parent / "training"):
    sys.path.insert(0, str(_d))

# ==================== CONFIGURACION ====================

BILSTM_MODEL_DIR = Path("./batches/Modelos_BiLSTM_PyTorch")
BILSTM_MODEL_GLOB = "modelo_bilstm_pytorch_*_best.pt"
BILSTM_DATA_DIR = Path("./batches/Datos_Modelo_BiLSTM")
BILSTM_DATA_GLOB = "datos_modelo_{split}_bilstm_*.h5"

CNN_MODEL_DIR = Path("./batches/Modelos_CNN2D_PyTorch")
CNN_MODEL_GLOB = "modelo_cnn2d_pytorch_*_best.pt"
CNN_DATA_DIR = Path(os.environ.get('CNN_DATA_DIR', "./batches/Datos_Modelo_CNN"))
CNN_DATA_GLOB = "datos_modelo_{split}_cnn_*.h5"

CALIB_B = Path(os.path.join(os.environ.get('NOBSP_RUNB_DIR', "./batches/Resultados/nobsp_runB_traintest"), 'calib_index.npz'))
OUTPUT_DIR = Path(os.environ.get('COLLINEARITY_DIR', "./batches/Resultados/colinealidad_runB"))

N_SAMPLES_DEFAULT = 100000          # toda la muestra B (1.734)
BATCH = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Geometria del backbone convolucional, de la tabla de campo receptivo:
# cada columna del mapa pre-pooling avanza 8 eventos e integra 22.
CNN_STRIDE = 8
CNN_RECEPTIVE = 22          # criterio aproximado [8c, 8c+22): solo como sensibilidad
# Campo receptivo real de conv4: [8c-15, 8c+22] recortado a 0-99
CNN_RF_LO, CNN_RF_HI = -15, 22

SEED = 20260827


# ==================== METRICAS ====================

def metricas_colinealidad(V):
    if V.shape[0] < 2:
        return None
    norms = np.linalg.norm(V, axis=1)
    keep = norms > 1e-8
    V = V[keep]
    norms = norms[keep]
    if V.shape[0] < 2:
        return None

    Vn = V / norms[:, None]
    S = Vn @ Vn.T
    iu = np.triu_indices(S.shape[0], k=1)
    cos = np.clip(S[iu], -1.0, 1.0)

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
    if not filas:
        return {'bloque': nombre, 'n_pacientes': 0, 'notas': notas}
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
    out['n_pos_min'] = int(min(f['n_pos'] for f in filas))
    out['n_pos_max'] = int(max(f['n_pos'] for f in filas))
    pos = [f for f in filas if f.get('y') == 1]
    out['n_positivos'] = len(pos)
    return out


def _resolver(dirpath, patron, etiqueta):
    cands = sorted(dirpath.glob(patron))
    if not cands:
        raise FileNotFoundError(
            "No se encontro %s en %s con el patron %s" % (etiqueta, dirpath, patron))
    return cands[-1]


# ============ Carga de la muestra B en el orden de calib_index ============

def cargar_muestra_b(archivos, claves, n_max):   # archivos = ck.datos_*() del modelo fijado
    """Devuelve {clave: array} y y para las filas (split, fila) de calib_index.npz,
    en ese orden. split 0 = train 100:1, 1 = test (igual que en aplicar_nobsp_v4_r2.py)."""
    d = np.load(CALIB_B)
    idx, y_cal = d['index'][:n_max], d['y'][:n_max].astype(int)
    out = {k: None for k in claves}
    ids = np.zeros(len(idx), dtype=np.int64)
    y = np.zeros(len(idx), dtype=int)
    for sc, split in ((0, 'train'), (1, 'test')):
        f = archivos[split]
        pos = np.where(idx[:, 0] == sc)[0]
        if len(pos) == 0:
            continue
        rows = idx[pos, 1]
        o = np.argsort(rows)
        with h5py.File(f, 'r') as h:
            for k in claves:
                a = h[k][rows[o]].astype(np.float32)
                if out[k] is None:
                    out[k] = np.zeros((len(idx), *a.shape[1:]), np.float32)
                out[k][pos[o]] = a
            y[pos[o]] = h['y'][rows[o]]
            if 'ID' in h:
                ids[pos[o]] = h['ID'][rows[o]]
    if not np.array_equal(y, y_cal):
        raise RuntimeError("y de los HDF5 no coincide con calib_index.npz")
    return out, y, ids


# ============ BLOQUE A: BiLSTM antes y despues de la atencion ============

def analizar_bilstm(n_samples):
    from Modelo_Predictivo_V3_PyTorch import BiLSTMAttentionModelV3

    # Muestra B (train+test) en el orden de calib_index
    import checkpoint_utils as ck
    datos, y_all, ids_b = cargar_muestra_b(ck.datos_bilstm(), ['X_ventanas', 'X_mascaras'], n_samples)
    X_vent, X_mask = datos['X_ventanas'], datos['X_mascaras']
    print("  datos  : muestra B, %d pacientes" % len(y_all), flush=True)
    n, n_win, seq_len, n_feat = X_vent.shape
    print("  n=%d ventanas=%d seq=%d feat=%d  positivos en el lote=%d"
          % (n, n_win, seq_len, n_feat, int(y_all.sum())), flush=True)

    # --- ocupacion de la representacion ventaneada, evidencia para la Sec. 4.6 ---
    n_val = (X_mask > 0.5).sum(axis=1).astype(int)
    dist = collections.Counter(n_val.tolist())
    ocupacion = {
        'n_cargados': int(n),
        'distribucion_ventanas_validas': {str(k): int(v) for k, v in sorted(dist.items())},
        'mediana_ventanas_validas': float(np.median(n_val)),
        'frac_con_1_o_menos': float((n_val <= 1).mean()),
        'frac_con_2_o_mas': float((n_val >= 2).mean()),
        'max_ventanas_validas': int(n_val.max()),
        'positivos_lote': int(y_all.sum()),
        'positivos_con_2_o_mas': int(y_all[n_val >= 2].sum()),
        'mediana_ventanas_positivos': float(np.median(n_val[y_all == 1])) if y_all.sum() else None,
        'mediana_ventanas_negativos': float(np.median(n_val[y_all == 0])),
    }
    print("  ocupacion: mediana %.1f ventanas validas de %d; %.1f%% tiene 1 o menos"
          % (ocupacion['mediana_ventanas_validas'], n_win,
             100 * ocupacion['frac_con_1_o_menos']), flush=True)

    model = BiLSTMAttentionModelV3(n_features=n_feat).to(DEVICE)
    bilstm_pt = ck.bilstm_pt()          # fijado por BILSTM_TS
    print("  modelo : %s" % bilstm_pt.name, flush=True)
    model.load_state_dict(torch.load(bilstm_pt, map_location=DEVICE, weights_only=True))
    model.eval()

    filas_pre, filas_post = [], []
    with torch.no_grad():
        for i0 in range(0, n, BATCH):
            xb = torch.from_numpy(X_vent[i0:i0 + BATCH]).to(DEVICE)
            mb = torch.from_numpy(X_mask[i0:i0 + BATCH]).to(DEVICE)
            B = xb.shape[0]
            x = xb.reshape(B * n_win, seq_len, n_feat)
            _, (h_n, _) = model.bilstm(x)
            lstm_out = torch.cat([h_n[0], h_n[1]], dim=-1).view(B, n_win, -1)
            mask_3d = mb.unsqueeze(-1)
            lstm_masked = lstm_out * mask_3d
            attn_out = model.mha(lstm_masked)
            attn_masked = attn_out * mask_3d

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
                    for r in (r_pre, r_post):
                        r['idx'] = i0 + j
                        r['y'] = int(y_all[i0 + j])
                    filas_pre.append(r_pre)
                    filas_post.append(r_post)
    return filas_pre, filas_post, ocupacion, ids_b, bilstm_pt


# ============ BLOQUE B: CNN-2D, enmascarada ============

def analizar_cnn(n_samples, n_match):
    from modelo_cnn2d_v4_pytorch import CNN2DModelV4

    import checkpoint_utils as ck
    model_pt = ck.cnn_pt()              # fijado por CNN2D_TS
    # Muestra B (train+test) en el orden de calib_index
    datos, y_all, ids_c = cargar_muestra_b(ck.datos_cnn(), ['X_events'], n_samples)
    X_events = datos['X_events']
    print("  datos  : muestra B, %d pacientes" % len(y_all), flush=True)
    n = X_events.shape[0]

    # Ocupacion real: una posicion de evento cuenta si su vector no es todo cero.
    occ = (np.abs(X_events) > 0).any(axis=1)          # (n, 100)
    n_ev = occ.sum(axis=1).astype(int)
    print("  n=%d  positivos en el lote=%d  eventos reales: mediana %.1f, P95 %.1f"
          % (n, int(y_all.sum()), np.median(n_ev), np.percentile(n_ev, 95)), flush=True)

    model = CNN2DModelV4(n_features=1541, n_eventos=100, n_demograficos=15).to(DEVICE)
    model.load_state_dict(torch.load(model_pt, map_location=DEVICE, weights_only=True))
    model.eval()

    rng = np.random.default_rng(SEED)
    filas_todas, filas_mask, filas_match = [], [], []
    filas_mask_v2 = []                  # criterio aproximado [8c, 8c+22), sensibilidad
    n_cols = None
    with torch.no_grad():
        for i0 in range(0, n, BATCH):
            xb = torch.from_numpy(X_events[i0:i0 + BATCH]).to(DEVICE)
            fm = model.forward_conv(xb)                        # (B, C, H', W')
            cols = fm.mean(dim=2).permute(0, 2, 1).cpu().numpy()   # (B, W', C)
            n_cols = cols.shape[1]
            for j in range(cols.shape[0]):
                gi = i0 + j
                # (a) sin enmascarar, para contrastar
                r = metricas_colinealidad(cols[j])
                if r:
                    r['idx'] = gi; r['y'] = int(y_all[gi])
                    filas_todas.append(r)
                # (b) enmascarada: la columna c es real si su campo receptivo
                #     toca algun evento real del paciente
                ev = n_ev[gi]
                # campo receptivo real [8c-15, 8c+22]: toca un evento real si 8c-15 < ev
                valid = np.array([(CNN_STRIDE * c + CNN_RF_LO) < ev for c in range(n_cols)])
                valid_v2 = np.array([(CNN_STRIDE * c) < ev for c in range(n_cols)])
                if valid_v2.sum() >= 2:
                    r2_ = metricas_colinealidad(cols[j][valid_v2])
                    if r2_:
                        r2_['idx'] = gi; r2_['y'] = int(y_all[gi]); r2_['n_eventos'] = int(ev)
                        filas_mask_v2.append(r2_)
                if valid.sum() >= 2:
                    rm = metricas_colinealidad(cols[j][valid])
                    if rm:
                        rm['idx'] = gi; rm['y'] = int(y_all[gi])
                        rm['n_eventos'] = int(ev)
                        filas_mask.append(rm)
                    # (c) igualada en numero de posiciones al BiLSTM
                    if valid.sum() >= n_match:
                        sel = rng.choice(np.flatnonzero(valid), size=n_match, replace=False)
                        rr = metricas_colinealidad(cols[j][sel])
                        if rr:
                            rr['idx'] = gi; rr['y'] = int(y_all[gi])
                            filas_match.append(rr)
    return filas_todas, filas_mask, filas_match, int(n_cols or 0), filas_mask_v2, ids_c, model_pt



# ==================== FIGURA ====================

def figura(filas_pre, filas_post, filas_match, n_match, path_png):
    """Panel (a): control interno pareado. Panel (b): coseno comparable."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print("  (sin figura, matplotlib no disponible: %s)" % e)
        return

    pre = np.array([f['cos_mean'] for f in filas_pre])
    post = np.array([f['cos_mean'] for f in filas_post])

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # (a) lineas pareadas paciente a paciente, que es el disenio real
    for a_, b_ in zip(pre, post):
        ax[0].plot([0, 1], [a_, b_], color='0.65', lw=0.7, alpha=0.55, zorder=1)
    ax[0].plot([0, 1], [pre.mean(), post.mean()], color='#c1121f', lw=2.4,
               marker='o', ms=7, zorder=3, label='mean')
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels(['before\nattention', 'after\nattention'])
    ax[0].set_ylabel('Mean pairwise cosine similarity')
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].set_title('(a) Paired within-model comparison (n=%d)' % len(pre))
    ax[0].legend(frameon=False, loc='lower right')

    # (b) coseno comparable entre arquitecturas, a igual numero de posiciones
    datos = [list(pre), list(post)]
    etiquetas = ['BiLSTM\nbefore', 'BiLSTM\nafter']
    if filas_match:
        datos.append([f['cos_mean'] for f in filas_match])
        etiquetas.append('CNN-2D\nmasked, n=%d' % n_match)
    bp = ax[1].boxplot(datos, tick_labels=etiquetas, showfliers=False, patch_artist=True)
    for patch, c in zip(bp['boxes'], ['#8ecae6', '#e07a5f', '#81b29a']):
        patch.set_facecolor(c)
    ax[1].set_ylabel('Mean pairwise cosine similarity')
    ax[1].set_ylim(-0.05, 1.05)
    ax[1].set_title('(b) Comparable positions, cosine only')

    fig.tight_layout()
    fig.savefig(path_png, dpi=300)
    fig.savefig(str(path_png).replace('.png', '.pdf'))
    print("  figura -> %s" % path_png.name, flush=True)


# ==================== MAIN ====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=N_SAMPLES_DEFAULT)
    ap.add_argument('--skip-cnn', action='store_true')
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()
    print("=" * 74)
    print("COLINEALIDAD DE LAS REPRESENTACIONES ATENDIDAS")
    print("inicio: %s   device: %s   n=%d"
          % (t0.strftime('%Y-%m-%d %H:%M:%S'), DEVICE, args.n))
    print("=" * 74)

    print("\n[1/3] BiLSTM-Attention: antes y despues de la atencion", flush=True)
    filas_pre, filas_post, ocupacion, ids_b, bilstm_pt = analizar_bilstm(args.n)
    print("  pacientes con al menos 2 ventanas validas: %d de %d"
          % (len(filas_pre), ocupacion['n_cargados']), flush=True)
    n_match = int(round(np.median([f['n_pos'] for f in filas_pre]))) if filas_pre else 3
    print("  n de posiciones para igualar la CNN: %d" % n_match, flush=True)

    todas = maskd = match = maskd_v2 = []
    n_cols = 0
    ids_c, cnn_pt = None, None
    if not args.skip_cnn:
        print("\n[2/3] CNN-2D: columnas temporales pre-pooling", flush=True)
        try:
            todas, maskd, match, n_cols, maskd_v2, ids_c, cnn_pt = analizar_cnn(args.n, n_match)
            # Los dos modelos deben ver los mismos pacientes en el mismo orden
            if not np.array_equal(ids_b, ids_c):
                raise RuntimeError("IDs de la muestra B distintos entre BiLSTM y CNN")
            print("  IDs BiLSTM == IDs CNN en la muestra B: OK", flush=True)
            print("  sin enmascarar: %d   enmascarada: %d   igualada a n=%d: %d"
                  % (len(todas), len(maskd), n_match, len(match)), flush=True)
        except RuntimeError:
            raise
        except Exception as e:
            print("  AVISO: bloque CNN omitido (%s)" % e, flush=True)

    print("\n[3/3] Resumen y guardado", flush=True)
    resumen = {
        'version': 2,
        'fecha': t0.strftime('%Y-%m-%d %H:%M:%S'),
        'n_solicitado': args.n,
        'device': str(DEVICE),
        'ocupacion_representacion_ventaneada': ocupacion,
        # Muestra y modelos usados
        'muestra': 'B', 'calib_index': str(CALIB_B), 'n_ids': int(len(ids_b)),
        'ids_iguales_bilstm_cnn': bool(ids_c is not None and np.array_equal(ids_b, ids_c)),
        'modelo_bilstm': __import__('checkpoint_utils').registro(bilstm_pt),
        'modelo_cnn': __import__('checkpoint_utils').registro(cnn_pt) if cnn_pt else None,
        'bloques': [
            resumir('bilstm_pre_attention', filas_pre,
                    'Salida del BiLSTM por ventana, antes de la atencion. Solo ventanas validas.'),
            resumir('bilstm_post_attention', filas_post,
                    'La misma tras el bloque de atencion. Control interno pareado: mismo modelo, '
                    'mismos pacientes, misma dimension 512 y mismo n en ambas ramas.'),
        ],
    }
    if todas:
        resumen['bloques'].append(resumir(
            'cnn2d_todas_las_columnas', todas,
            'Las 12 columnas sin enmascarar. SOLO para mostrar el sesgo: '
            'la mayoria de las columnas corresponden a relleno y no a eventos reales.'))
    if maskd:
        resumen['bloques'].append(resumir(
            'cnn2d_enmascarada', maskd,
            'Solo columnas cuyo campo receptivo real (paso 8, campo receptivo 38: [8c-15, 8c+22] '
            'recortado a 0-99) toca algun evento real.'))
    if maskd_v2:
        resumen['bloques'].append(resumir(
            'cnn2d_enmascarada_criterio_aproximado', maskd_v2,
            'SENSIBILIDAD: criterio [8c, 8c+22) (campo aproximado; no es el real). '
            'Columna valida si 8c < n_eventos.'))
    if match:
        resumen['bloques'].append(resumir(
            'cnn2d_igualada_n%d' % n_match, match,
            'Columnas validas submuestreadas a n=%d, la mediana del BiLSTM. Para la comparacion '
            'entre arquitecturas usar SOLO el coseno: la dimension sigue difiriendo (512 vs %d).'
            % (n_match, n_cols)))

    a, b = resumen['bloques'][0], resumen['bloques'][1]
    resumen['control_interno'] = {
        'n_pacientes': a['n_pacientes'],
        'n_positivos': a.get('n_positivos'),
        'cos_mean_antes': a['cos_mean_mean'],
        'cos_mean_despues': b['cos_mean_mean'],
        'delta_cos': b['cos_mean_mean'] - a['cos_mean_mean'],
        'ang_mean_deg_antes': a['ang_mean_deg_mean'],
        'ang_mean_deg_despues': b['ang_mean_deg_mean'],
        'eff_rank_frac_antes': a['eff_rank_frac_mean'],
        'eff_rank_frac_despues': b['eff_rank_frac_mean'],
        'cond_median_antes': a.get('cond_median'),
        'cond_median_despues': b.get('cond_median'),
    }

    with open(OUTPUT_DIR / 'resumen_colinealidad.json', 'w', encoding='utf-8') as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    import csv as _csv
    for nombre, filas in [('bilstm_pre', filas_pre), ('bilstm_post', filas_post),
                          ('cnn2d_todas', todas), ('cnn2d_enmascarada', maskd),
                          ('cnn2d_igualada', match)]:
        if not filas:
            continue
        cols = ['idx', 'y', 'n_pos', 'n_eventos', 'cos_mean', 'cos_median', 'cos_p95',
                'cos_abs_mean', 'ang_mean_deg', 'ang_min_deg', 'cond', 'eff_rank',
                'eff_rank_frac']
        with open(OUTPUT_DIR / ('por_paciente_%s.csv' % nombre), 'w',
                  newline='', encoding='utf-8') as f:
            w = _csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in filas:
                w.writerow({k: r.get(k, '') for k in cols})

    figura(filas_pre, filas_post, match, n_match,
           OUTPUT_DIR / "colinealidad_v2.png")

    print("\n" + "=" * 74)
    print("CONTROL INTERNO del BiLSTM")
    print("=" * 74)
    ci = resumen['control_interno']
    print("  pacientes %d, de los cuales positivos %s"
          % (ci['n_pacientes'], ci['n_positivos']))
    print("  coseno medio      antes %+.4f    despues %+.4f   (delta %+.4f)"
          % (ci['cos_mean_antes'], ci['cos_mean_despues'], ci['delta_cos']))
    print("  angulo medio      antes %6.2f deg despues %6.2f deg"
          % (ci['ang_mean_deg_antes'], ci['ang_mean_deg_despues']))
    print("  rango efectivo    antes %.3f      despues %.3f"
          % (ci['eff_rank_frac_antes'], ci['eff_rank_frac_despues']))

    print("\n" + "=" * 74)
    print("OCUPACION de la representacion ventaneada (para la Seccion 4.6)")
    print("=" * 74)
    o = ocupacion
    print("  mediana de ventanas validas: %.1f de 10" % o['mediana_ventanas_validas'])
    print("  con 1 ventana o menos: %.1f%%   con 2 o mas: %.1f%%"
          % (100 * o['frac_con_1_o_menos'], 100 * o['frac_con_2_o_mas']))
    print("  maximo observado: %d ventanas" % o['max_ventanas_validas'])
    print("  mediana en positivos: %s   en negativos: %s"
          % (o['mediana_ventanas_positivos'], o['mediana_ventanas_negativos']))
    print("  distribucion: %s" % o['distribucion_ventanas_validas'])

    if maskd:
        print("\n" + "=" * 74)
        print("CNN-2D: el efecto de enmascarar el relleno")
        print("=" * 74)
        for nom in ('cnn2d_todas_las_columnas', 'cnn2d_enmascarada',
                    'cnn2d_igualada_n%d' % n_match):
            for bl in resumen['bloques']:
                if bl['bloque'] == nom:
                    print("  %-26s n=%-4d  posiciones %s  coseno %+.4f  eff_rank_frac %.3f"
                          % (nom, bl['n_pacientes'], bl.get('n_pos_median'),
                             bl['cos_mean_mean'], bl['eff_rank_frac_mean']))
        print("\n  Para comparar con el BiLSTM usar SOLO el bloque igualado y SOLO el coseno.")
        print("  BiLSTM post-atencion, coseno %+.4f" % ci['cos_mean_despues'])

    print("\nsalidas en: %s" % OUTPUT_DIR)
    print("duracion: %s" % (datetime.now() - t0))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
