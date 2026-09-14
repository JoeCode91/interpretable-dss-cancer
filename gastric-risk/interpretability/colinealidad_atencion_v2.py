"""
Colinealidad de las representaciones atendidas -- VERSION 2.

REPRODUCES IN THE THESIS
    Collinearity of the attention representation, paired control. Section 4.3.

PROVENANCE
    Copied verbatim from the working tree of the thesis. The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md at the
    repository root.

--------------------------------------------------------------------------

QUE CAMBIA RESPECTO A colinealidad_atencion.py (v1, corrida del 26 Ago)
-----------------------------------------------------------------------
La v1 dejo el control interno del BiLSTM bien medido, pero el bloque de la
CNN estaba contaminado y la comparacion entre arquitecturas no era
interpretable. Tres correcciones:

  (1) LA CNN NO ENMASCARABA EL RELLENO. El tensor de la CNN es fijo,
      1541 x 100, con los eventos empaquetados a la izquierda y ceros a la
      derecha. Sus 12 columnas tienen paso 8 y campo receptivo 22, luego la
      columna j cubre los eventos [8j, 8j+22). Con una mediana de ~22 eventos
      reales, solo las columnas 0 a 2 tienen contenido; las otras nueve son
      relleno. Y el relleno tras convolucion, normalizacion y ReLU NO es cero:
      los sesgos producen activaciones constantes, de modo que esas columnas
      son casi identicas entre si e inflan la colinealidad de forma artificial.
      Eso explica el eff_rank_frac de 0,222 de la v1.
      AHORA: se calcula la ocupacion real por paciente y se descartan las
      columnas cuyo campo receptivo no toca ningun evento real.

  (2) EL NUMERO DE POSICIONES NO ERA COMPARABLE. El BiLSTM aporta una mediana
      de 3 ventanas validas y la CNN aportaba siempre 12. El rango efectivo va
      normalizado por n, asi que compararlos no significaba nada.
      AHORA: ademas del bloque enmascarado se calcula un bloque de la CNN
      SUBMUESTREADO al mismo numero de posiciones que la mediana del BiLSTM,
      y para la comparacion entre arquitecturas se reporta SOLO la similitud
      coseno, que es la metrica robusta a la dimension.

  (3) NO SE GUARDABA LA ETIQUETA. La v1 informaba los positivos del lote
      cargado, no los de los pacientes que sobrevivian al filtro, lo que se
      presto a confusion al redactar.
      AHORA: se guarda y y se informa cuantos positivos hay en cada bloque.

ADEMAS se reporta la DISTRIBUCION DE OCUPACION de la representacion ventaneada
(cuantos pacientes tienen k ventanas validas), que es evidencia directa para la
seccion del confusor de longitud de historia.

El control interno del BiLSTM es identico a la v1 y debe reproducir sus cifras:
coseno 0,725 -> 0,970; angulo 42,75 -> 12,27 grados; rango efectivo 0,831 ->
0,542. Sirve de comprobacion de que la v2 no rompio nada.

SALIDAS
-------
`XAI_Model/colinealidad_results_v2/`. No sobrescribe las de la v1.

USO
---
    cd /d "G:\\Mi unidad\\Doctorado\\Proyecto_CC\\Test\\Book\\XAI_Model"
    D:\\venv_torch_cu121\\Scripts\\python.exe colinealidad_atencion_v2.py

Autor: Johan Pena-Campos. Cap. 4, articulo AIM.
Fecha: 2026-08-27

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
import collections
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

BILSTM_PT = Path(r"D:\Proyecto_CC_Batches\Modelos_V32_PyTorch"
                 r"\modelo_v32_pytorch_20260510_151626_best.pt")
BILSTM_DATA_DIR = Path(r"D:\Proyecto_CC_Batches\Datos_Modelo")
BILSTM_DATA_GLOB = "datos_modelo_test_*.h5"

CNN_MODEL_DIR = Path(r"D:\Proyecto_CC_Batches\Modelos_CNN2D_V4_PyTorch")
CNN_MODEL_GLOB = "modelo_cnn2d_v4_pytorch_*_best.pt"
CNN_DATA_DIR = Path(r"D:\Proyecto_CC_Batches\Datos_Modelo_V4")
CNN_DATA_GLOB = "datos_modelo_test_v4_*.h5"

OUTPUT_DIR = Path(__file__).parent / "colinealidad_results_v2"

N_SAMPLES_DEFAULT = 500
BATCH = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Geometria del backbone convolucional, de la tabla de campo receptivo:
# cada columna del mapa pre-pooling avanza 8 eventos e integra 22.
CNN_STRIDE = 8
CNN_RECEPTIVE = 22

SEED = 20260827


# ==================== METRICAS (identicas a la v1) ====================

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


# ============ BLOQUE A: BiLSTM antes y despues de la atencion ============

def analizar_bilstm(n_samples):
    from Modelo_Predictivo_V3_PyTorch import BiLSTMAttentionModelV3

    data_pt = _resolver(BILSTM_DATA_DIR, BILSTM_DATA_GLOB, "datos test del BiLSTM")
    print("  datos  : %s" % data_pt.name, flush=True)

    with h5py.File(data_pt, 'r') as f:
        X_vent = f['X_ventanas'][:n_samples].astype(np.float32)
        X_mask = f['X_mascaras'][:n_samples].astype(np.float32)
        y_all = np.asarray(f['y'][:n_samples]).ravel()
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
    model.load_state_dict(torch.load(BILSTM_PT, map_location=DEVICE, weights_only=True))
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
    return filas_pre, filas_post, ocupacion


# ============ BLOQUE B: CNN-2D, ahora enmascarada ============

def analizar_cnn(n_samples, n_match):
    from modelo_cnn2d_v4_pytorch import CNN2DModelV4

    model_pt = _resolver(CNN_MODEL_DIR, CNN_MODEL_GLOB, "modelo CNN-2D")
    data_pt = _resolver(CNN_DATA_DIR, CNN_DATA_GLOB, "datos test V4")
    print("  datos  : %s" % data_pt.name, flush=True)

    with h5py.File(data_pt, 'r') as f:
        X_events = f['X_events'][:n_samples].astype(np.float32)
        y_all = np.asarray(f['y'][:n_samples]).ravel()
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
    n_cols = None
    with torch.no_grad():
        for i0 in range(0, n, BATCH):
            xb = torch.from_numpy(X_events[i0:i0 + BATCH]).to(DEVICE)
            fm = model.forward_conv(xb)                        # (B, C, H', W')
            cols = fm.mean(dim=2).permute(0, 2, 1).cpu().numpy()   # (B, W', C)
            n_cols = cols.shape[1]
            for j in range(cols.shape[0]):
                gi = i0 + j
                # (a) sin enmascarar, para reproducir la v1 y poder contrastar
                r = metricas_colinealidad(cols[j])
                if r:
                    r['idx'] = gi; r['y'] = int(y_all[gi])
                    filas_todas.append(r)
                # (b) enmascarada: la columna c es real si su campo receptivo
                #     toca algun evento real del paciente
                ev = n_ev[gi]
                valid = np.array([(CNN_STRIDE * c) < ev for c in range(n_cols)])
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
    return filas_todas, filas_mask, filas_match, int(n_cols or 0)



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
    print("COLINEALIDAD DE LAS REPRESENTACIONES ATENDIDAS -- v2")
    print("inicio: %s   device: %s   n=%d"
          % (t0.strftime('%Y-%m-%d %H:%M:%S'), DEVICE, args.n))
    print("=" * 74)

    print("\n[1/3] BiLSTM-Attention: antes y despues de la atencion", flush=True)
    filas_pre, filas_post, ocupacion = analizar_bilstm(args.n)
    print("  pacientes con al menos 2 ventanas validas: %d de %d"
          % (len(filas_pre), args.n), flush=True)
    n_match = int(round(np.median([f['n_pos'] for f in filas_pre]))) if filas_pre else 3
    print("  n de posiciones para igualar la CNN: %d" % n_match, flush=True)

    todas = maskd = match = []
    n_cols = 0
    if not args.skip_cnn:
        print("\n[2/3] CNN-2D: columnas temporales pre-pooling", flush=True)
        try:
            todas, maskd, match, n_cols = analizar_cnn(args.n, n_match)
            print("  sin enmascarar: %d   enmascarada: %d   igualada a n=%d: %d"
                  % (len(todas), len(maskd), n_match, len(match)), flush=True)
        except Exception as e:
            print("  AVISO: bloque CNN omitido (%s)" % e, flush=True)

    print("\n[3/3] Resumen y guardado", flush=True)
    resumen = {
        'version': 2,
        'fecha': t0.strftime('%Y-%m-%d %H:%M:%S'),
        'n_solicitado': args.n,
        'device': str(DEVICE),
        'ocupacion_representacion_ventaneada': ocupacion,
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
            'Las 12 columnas sin enmascarar. SOLO para reproducir la v1 y mostrar el sesgo: '
            'la mayoria de las columnas corresponden a relleno y no a eventos reales.'))
    if maskd:
        resumen['bloques'].append(resumir(
            'cnn2d_enmascarada', maskd,
            'Solo columnas cuyo campo receptivo (paso %d, extension %d) toca eventos reales.'
            % (CNN_STRIDE, CNN_RECEPTIVE)))
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

    with open(OUTPUT_DIR / 'resumen_colinealidad_v2.json', 'w', encoding='utf-8') as f:
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
    print("CONTROL INTERNO del BiLSTM (debe reproducir la v1)")
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
