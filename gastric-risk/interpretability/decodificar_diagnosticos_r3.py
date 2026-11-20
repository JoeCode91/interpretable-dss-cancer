"""
R3 -- Ingenieria inversa de diagnosticos (CIE-10) y procedimientos (CUPS) en los
eventos de los pacientes con cancer.

REPRODUCES IN THE THESIS
    Reverse-engineers the CIE-10 and CUPS codes behind the temporally salient events,
    Section 4.9.

PROVENANCE
    Taken from the working tree that produced the reported results. The computational
    content is unchanged; absolute paths were replaced by relative defaults and
    environment variables, notification calls were removed, and comments were edited.
    See CHANGES-FROM-THESIS.md at the repository root.

--------------------------------------------------------------------------

Cada evento del tensor codifica [emb_diag(768), emb_proc(768), temporal(5)].
Los bloques de embedding son copia byte-a-byte de los embeddings cacheados
(L2-normalizados), por lo que el codigo se recupera de forma EXACTA via argmax
del producto punto contra el diccionario de codigos.

La localizacion temporal de las atribuciones esta confundida con la longitud de
historia / padding. R3 separa CONTENIDO (que diagnosticos) de POSICION:
  - decodifica todos los eventos reales de los pacientes de la muestra B,
  - enfoca los eventos RECIENTES (frontera dato/padding = ultimos K reales),
  - mide enriquecimiento de cada codigo en cancer vs no-cancer.

Muestra B: todos los cancer de train (100:1) y test, y el mismo numero de no cancer
de train+test (RandomState(42)), en el orden de calib_index.npz.

Salidas en NOBSP_RUNB_DIR (por defecto ./batches/Resultados/nobsp_runB_traintest):
  - r3_diagnosticos_todos.{csv,json}  tabla completa de diagnosticos (pacientes por
                                      clase, ocurrencias, lift sin suavizado)
  - r3_diagnosticos_cancer.csv     codigos diag enriquecidos en cancer (+ descripcion)
  - r3_procedimientos_cancer.csv   codigos proc enriquecidos en cancer
  - r3_diagnosticos_recientes_cancer.csv  diag en los ultimos K eventos
  - r3_eventos_recientes_cancer.csv  ultimos K eventos por paciente cancer (decodificados)
  - r3_resumen.txt                 resumen legible

Necesita los embeddings de los codigos (EMB_DIR, por defecto ./batches/Embeddings/BERT)
y los mapas codigo -> descripcion mapa_cie10.json / mapa_cups.json (EMB_MAP_DIR).

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
import sys as _sys
from pathlib import Path as _P
# Modulos hermanos del repositorio
_HERE = _P(__file__).resolve().parent
for _d in (_HERE, _HERE.parent / "training"):
    _sys.path.insert(0, str(_d))


import csv
import glob
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import h5py

EMB_DIR = Path(os.environ.get('EMB_DIR', "./batches/Embeddings/BERT"))
V4_DATA_DIR = Path(os.environ.get('CNN_DATA_DIR', "./batches/Datos_Modelo_CNN"))
OUT = Path(os.environ.get('NOBSP_RUNB_DIR', "./batches/Resultados/nobsp_runB_traintest"))
MAP_DIR = Path(os.environ.get('EMB_MAP_DIR', "./embeddings/embeddings_v32_setup"))
N_SAMPLES = 500
BERT_DIM = 768
SIM_THRESHOLD = 0.99   # bloques son copia exacta -> sim real ~1.0; rechaza no-matches
K_RECENT = 5           # ultimos K eventos reales = frontera (eventos mas recientes)
# Nota: CUPS (proc) tiene ~473/2879 filas de embedding duplicadas (codigos con
# descripcion identica/fallback) -> el codigo proc recuperado puede ser ambiguo.
# CIE-10 (diag) solo 6/8101 colisiones -> recuperacion practicamente exacta.


def latest(pattern):
    c = sorted(glob.glob(str(pattern)))
    if not c:
        raise FileNotFoundError(pattern)
    return c[-1]


def load_vocab(kind):
    f = latest(EMB_DIR / f"bert_embeddings_{kind}_completos_v32_*.npz")
    d = np.load(f, allow_pickle=True)
    emb = d['embeddings'].astype(np.float32)         # (V, 768) L2-normalizado
    terms = np.array([str(t) for t in d['terms']])
    return emb, terms


def decode_block(block, emb, terms):
    """block (768,) -> (codigo, sim) por argmax de producto punto. None si nulo."""
    n = np.linalg.norm(block)
    if n < 1e-6:
        return None, 0.0
    sims = emb @ (block / n)                          # emb ya normalizado
    j = int(np.argmax(sims))
    return (terms[j], float(sims[j])) if sims[j] > SIM_THRESHOLD else (None, float(sims[j]))


def main():
    print("Cargando vocabularios y descripciones...", flush=True)
    diag_emb, diag_terms = load_vocab("diag")
    proc_emb, proc_terms = load_vocab("proc")
    cie = json.load(open(MAP_DIR / "mapa_cie10.json", encoding="utf-8"))
    cups = json.load(open(MAP_DIR / "mapa_cups.json", encoding="utf-8"))
    # Codigos proc con embedding NO unico (descripcion compartida/fallback) -> ambiguos
    from collections import Counter as _C
    proc_keys = [tuple(np.round(proc_emb[i], 4)) for i in range(len(proc_emb))]
    dup_keys = {k for k, c in _C(proc_keys).items() if c > 1}
    proc_ambiguous = {proc_terms[i] for i, k in enumerate(proc_keys) if k in dup_keys}
    desc_diag = lambda c: cie.get(c, "(sin descripcion)")
    desc_proc = lambda c: ("[AMBIGUO] " if c in proc_ambiguous else "") + cups.get(c, "(sin descripcion)")
    print(f"  diag={diag_emb.shape}, proc={proc_emb.shape}, proc ambiguos={len(proc_ambiguous)}", flush=True)

    global OUT
    R2_DIR = os.environ.get('NOBSP_R2_DIR', str(OUT)).strip()   # por defecto, la muestra B
    if R2_DIR:
        import r2_calib as r2_calib
        X, _, y = r2_calib.load_r2_calib(R2_DIR, str(V4_DATA_DIR), need_events=True)
        X = X.astype(np.float32)
        OUT = Path(R2_DIR)
        print(f"  [R2 train+test] n={len(y)}, cancer={int(y.sum())} -> salida {OUT}", flush=True)
    else:
        test_pt = latest(V4_DATA_DIR / "datos_modelo_test_cnn_*.h5")
        with h5py.File(test_pt, 'r') as f:
            X = f['X_events'][:N_SAMPLES].astype(np.float32)   # (500,1541,100)
            y = f['y'][:N_SAMPLES].astype(int)
        print(f"  test (500), n={N_SAMPLES}, cancer={int(y.sum())}", flush=True)

    # Decodificacion por paciente
    diag_all = {0: Counter(), 1: Counter()}     # conteo de codigos diag por clase
    proc_all = {0: Counter(), 1: Counter()}
    diag_recent = {0: Counter(), 1: Counter()}  # solo ultimos K eventos (frontera)
    pac_diag = {0: Counter(), 1: Counter()}     # pacientes con el codigo (no ocurrencias)
    n_pat = {0: 0, 1: 0}
    sims_check = []
    recientes_rows = []   # detalle por paciente cancer

    for i in range(len(y)):
        cls = int(y[i])
        n_pat[cls] += 1
        # eventos reales: columna con bloque diag o proc no nulo
        col_norm = np.linalg.norm(X[i, :2 * BERT_DIM, :], axis=0)   # (100,)
        real_cols = np.where(col_norm > 1e-6)[0]
        if len(real_cols) == 0:
            continue
        decoded = []   # (col, dcode, pcode)
        for c in real_cols:
            dcode, ds = decode_block(X[i, :BERT_DIM, c], diag_emb, diag_terms)
            pcode, ps = decode_block(X[i, BERT_DIM:2 * BERT_DIM, c], proc_emb, proc_terms)
            if dcode is not None:
                diag_all[cls][dcode] += 1
                sims_check.append(ds)
            if pcode is not None:
                proc_all[cls][pcode] += 1
            decoded.append((int(c), dcode, pcode))
        for dcode_ in {d for _, d, _ in decoded if d is not None}:
            pac_diag[cls][dcode_] += 1
        # frontera: ultimos K eventos reales (columnas mas altas con dato = mas recientes)
        recent = decoded[-K_RECENT:]
        for (c, dcode, pcode) in recent:
            if dcode is not None:
                diag_recent[cls][dcode] += 1
        if cls == 1:
            for (c, dcode, pcode) in recent:
                recientes_rows.append([i, c, dcode or "", desc_diag(dcode) if dcode else "",
                                       pcode or "", desc_proc(pcode) if pcode else ""])

    print(f"\nAuto-check match exacto: sim media={np.mean(sims_check):.4f} "
          f"min={np.min(sims_check):.4f} (esperado ~1.0)", flush=True)

    # Enriquecimiento: tasa por paciente en cancer vs no-cancer
    def enrichment(counter_c, counter_n, n_c, n_n, descfn, top=25):
        codes = set(counter_c) | set(counter_n)
        rows = []
        for code in codes:
            rate_c = counter_c[code] / max(n_c, 1)
            rate_n = counter_n[code] / max(n_n, 1)
            # Lift sin suavizado: inf si el denominador es 0
            lift = rate_c / rate_n if rate_n > 0 else (float('inf') if rate_c > 0 else float('nan'))
            rows.append((code, counter_c[code], rate_c, rate_n, lift, descfn(code)))
        # ordenar por presencia en cancer y luego lift
        rows.sort(key=lambda r: (r[1], r[4]), reverse=True)
        return rows[:top]

    n_c, n_n = n_pat[1], n_pat[0]
    diag_rows = enrichment(diag_all[1], diag_all[0], n_c, n_n, desc_diag)
    proc_rows = enrichment(proc_all[1], proc_all[0], n_c, n_n, desc_proc)
    diag_recent_rows = enrichment(diag_recent[1], diag_recent[0], n_c, n_n, desc_diag, top=20)

    def write_csv(path, rows, head):
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['codigo', 'n_cancer', 'tasa_cancer', 'tasa_nocancer', 'lift', head])
            for r in rows:
                w.writerow([r[0], r[1], f"{r[2]:.3f}", f"{r[3]:.4f}", f"{r[4]:.1f}", r[5]])
        print(f"  guardado {path.name}", flush=True)

    write_csv(OUT / "r3_diagnosticos_cancer.csv", diag_rows, 'descripcion_CIE10')

    # Tabla COMPLETA de diagnosticos, ordenada por lift con el filtro >= MIN_PAC_CANCER
    # pacientes con cancer primero. "Todos los eventos" = los <= 100
    # que conserva el tensor de la CNN (no la historia completa: ~2,7 % de la cohorte supera 100).
    MIN_PAC_CANCER = int(os.environ.get('R3_MIN_PAC_CANCER', '20'))
    filas_t = []
    for code in set(diag_all[1]) | set(diag_all[0]):
        mc = diag_all[1][code] / max(n_c, 1)
        mn = diag_all[0][code] / max(n_n, 1)
        lift = mc / mn if mn > 0 else (float('inf') if mc > 0 else float('nan'))
        filas_t.append(dict(codigo=code, descripcion=desc_diag(code),
                            n_pac_cancer=pac_diag[1][code], n_pac_nocancer=pac_diag[0][code],
                            ocurr_cancer=diag_all[1][code], ocurr_nocancer=diag_all[0][code],
                            media_ocurr_cancer=mc, media_ocurr_nocancer=mn, lift=lift,
                            cumple_filtro=pac_diag[1][code] >= MIN_PAC_CANCER))
    filas_t.sort(key=lambda r: (not r['cumple_filtro'],
                                -(r['lift'] if np.isfinite(r['lift']) else 1e300) if not np.isnan(r['lift']) else 0))
    with open(OUT / "r3_diagnosticos_todos.csv", 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas_t[0].keys()) if filas_t else ['codigo'])
        w.writeheader()
        w.writerows(filas_t)
    with open(OUT / "r3_diagnosticos_todos.json", 'w', encoding='utf-8') as fh:
        json.dump({"muestra": "B",
                   "n_pacientes_cancer": n_c, "n_pacientes_nocancer": n_n,
                   "lift": "media de ocurrencias por paciente en cancer / en no cancer; inf si no aparece en no cancer",
                   "eventos": "todos los eventos reales del tensor CNN (los <= 100 mas recientes de cada paciente)",
                   "orden": f"primero los codigos con >= {MIN_PAC_CANCER} pacientes con cancer, por lift descendente"},
                  fh, indent=2, ensure_ascii=False)
    print(f"  guardado r3_diagnosticos_todos.csv ({len(filas_t)} codigos; "
          f"{sum(r['cumple_filtro'] for r in filas_t)} con >= {MIN_PAC_CANCER} pacientes con cancer)", flush=True)
    write_csv(OUT / "r3_procedimientos_cancer.csv", proc_rows, 'descripcion_CUPS')
    write_csv(OUT / "r3_diagnosticos_recientes_cancer.csv", diag_recent_rows, 'descripcion_CIE10')

    with open(OUT / "r3_eventos_recientes_cancer.csv", 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['paciente', 'columna', 'cie10', 'desc_cie10', 'cups', 'desc_cups'])
        w.writerows(recientes_rows)
    print(f"  guardado r3_eventos_recientes_cancer.csv ({len(recientes_rows)} filas)", flush=True)

    # Resumen legible
    with open(OUT / "r3_resumen.txt", 'w', encoding='utf-8') as fh:
        fh.write(f"R3 -- Diagnosticos en pacientes cancer (n={n_c}) vs no-cancer (n={n_n})\n")
        fh.write(f"Auto-check sim media={np.mean(sims_check):.4f} (match exacto si ~1.0)\n\n")
        fh.write("== TOP diagnosticos (CIE-10) presentes en pacientes cancer ==\n")
        fh.write(f"{'codigo':8s} {'n':>3s} {'tasa_c':>7s} {'tasa_n':>7s} {'lift':>6s}  descripcion\n")
        for r in diag_rows:
            fh.write(f"{r[0]:8s} {r[1]:3d} {r[2]:7.3f} {r[3]:7.4f} {r[4]:6.1f}  {r[5][:55]}\n")
        fh.write("\n== TOP diagnosticos en eventos RECIENTES (frontera, ultimos %d) ==\n" % K_RECENT)
        for r in diag_recent_rows:
            fh.write(f"{r[0]:8s} {r[1]:3d} lift={r[4]:6.1f}  {r[5][:55]}\n")
        fh.write("\n== TOP procedimientos (CUPS) en pacientes cancer ==\n")
        for r in proc_rows:
            fh.write(f"{r[0]:8s} {r[1]:3d} {r[2]:7.3f} {r[3]:7.4f} {r[4]:6.1f}  {r[5][:55]}\n")
    print(f"  guardado r3_resumen.txt", flush=True)
    print("\n=== TOP 12 diagnosticos cancer (consola) ===", flush=True)
    for r in diag_rows[:12]:
        print(f"  {r[0]:7s} n={r[1]:2d} lift={r[4]:5.1f}  {r[5][:50]}", flush=True)


if __name__ == "__main__":
    main()
