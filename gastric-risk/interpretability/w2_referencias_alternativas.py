"""
W2 con REFERENCIAS ALTERNATIVAS: por que la atribucion temporal sale igual para

REPRODUCES IN THE THESIS
    Why the temporal attribution is nearly identical across patients, and whether an
    alternative reference changes it. Section 4.9 and the limitations.

PROVENANCE
    Copied verbatim from the working tree of the thesis. The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md at the
    repository root.

--------------------------------------------------------------------------
todos los pacientes, y si eso se puede arreglar.

EL PROBLEMA QUE MOTIVA ESTO
---------------------------
Medido el 27 Ago sobre `nobsp_results_v4/input_event_explainer.npz`, que tiene
las contribuciones de 500 pacientes sobre las 100 posiciones de evento:

  - similitud coseno entre los perfiles de pacientes distintos: **0,9999**
  - desviacion tipica entre pacientes de la fraccion asignada a cada posicion:
    como maximo 2,7e-04
  - magnitud total por paciente: razon P95/P5 = 1,00
  - la posicion e098 es la dominante en **500 de 500** pacientes
  - el perfil medio se concentra en e092 a e099, que para un paciente con la
    mediana de ~22 encuentros caen DENTRO del relleno

Es decir, esa atribucion devuelve practicamente una sola respuesta y la devuelve
sea cual sea el paciente. Eso esta ya reportado en el articulo, en la Seccion
"What the attribution supports for an individual patient".

LA HIPOTESIS QUE ESTE SCRIPT PONE A PRUEBA
------------------------------------------
En `aplicar_nobsp_v4.py` la referencia es la MEDIA de la cohorte:

    x_events_ref = torch.from_numpy(X_events.mean(axis=0))

Como los eventos van empaquetados a la izquierda y la mayoria de pacientes tiene
pocos, esa media decae con la posicion: las columnas finales estan casi vacias.
Perturbar una columna casi vacia puede producir una respuesta relativa grande por
razones puramente numericas, sin relacion con ningun paciente. Si es asi, el pico
en el relleno y la uniformidad entre pacientes serian artefactos de la
REFERENCIA, no propiedades del modelo.

QUE HACE
--------
Repite la atribucion por posicion de evento con cuatro referencias distintas y
compara la variabilidad entre pacientes que produce cada una:

  (1) media de la cohorte        -- la actual, para reproducir el resultado
  (2) mediana de la cohorte      -- menos sensible a las colas
  (3) el propio paciente         -- referencia individual, que es lo que
                                    conceptualmente pide una explicacion local
  (4) media restringida a las posiciones con ocupacion real, dejando el relleno
      exactamente a cero

Para cada referencia reporta: similitud coseno media entre perfiles de pacientes
distintos, desviacion tipica entre pacientes de la fraccion por posicion, y en
que posicion cae el pico.

COMO SE LEE EL RESULTADO
------------------------
  - Si con (3) o (4) la similitud coseno BAJA de forma apreciable, el problema
    era la referencia y la atribucion temporal SI puede tener contenido
    individual. Eso seria un resultado publicable por si mismo, y obligaria a
    matizar la subseccion correspondiente del articulo.
  - Si sigue en torno a 0,999 con todas las referencias, el limite es del
    modelo o del wrapper y la redaccion actual del articulo se queda como esta.

En cualquiera de los dos casos el resultado se reporta como salga.

DONDE SE EJECUTA
----------------
PC de casa. Necesita el checkpoint y los datos V4 de D:. No modifica nada:
escribe en `XAI_Model/w2_referencias_results/`.

USO
---
    cd /d "G:\\Mi unidad\\Doctorado\\Proyecto_CC\\Test\\Book\\XAI_Model"
    D:\\venv_torch_cu121\\Scripts\\python.exe w2_referencias_alternativas.py

    # mas rapido, con menos pacientes:
    D:\\venv_torch_cu121\\Scripts\\python.exe w2_referencias_alternativas.py --n 100

AVISO DE COSTE
--------------
Este analisis perturba 100 columnas por paciente y por referencia. Con la
referencia (3), que es individual, hay que rehacerlo para cada paciente. Con
--n 100 son unos pocos minutos en GPU; con 500 puede pasar de media hora.
Empezar con --n 100.

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
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import h5py
import torch

sys.path.insert(0, str(Path(__file__).parent))

CNN_MODEL_DIR = Path(r"D:\Proyecto_CC_Batches\Modelos_CNN2D_V4_PyTorch")
CNN_MODEL_GLOB = "modelo_cnn2d_v4_pytorch_*_best.pt"
CNN_DATA_DIR = Path(r"D:\Proyecto_CC_Batches\Datos_Modelo_V4")
CNN_DATA_GLOB = "datos_modelo_test_v4_*.h5"
OUTPUT_DIR = Path(__file__).parent / "w2_referencias_results"

N_DEFAULT = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DELTA = 0.0          # valor al que se lleva la columna perturbada (ablacion a cero)


def _resolver(d, patron, etiqueta):
    c = sorted(d.glob(patron))
    if not c:
        raise FileNotFoundError("No se encontro %s en %s (%s)" % (etiqueta, d, patron))
    return c[-1]


def perfil_por_ablacion(model, x_ref, x_demo, n_pos=100):
    """Contribucion de cada posicion: cambio en la salida al anularla.

    x_ref: (1541, 100) tensor de referencia sobre el que se perturba.
    Devuelve un vector (100,) con |salida_base - salida_sin_la_posicion|.
    """
    with torch.no_grad():
        base = model(x_ref.unsqueeze(0), x_demo.unsqueeze(0)).item()
        out = np.empty(n_pos, dtype=np.float64)
        for c in range(n_pos):
            xp = x_ref.clone()
            xp[:, c] = DELTA
            out[c] = abs(base - model(xp.unsqueeze(0), x_demo.unsqueeze(0)).item())
    return out


def resumen_perfiles(M):
    """M: (n_pacientes, 100). Devuelve las metricas de variabilidad entre pacientes."""
    A = np.abs(M)
    P = A / np.maximum(A.sum(axis=1, keepdims=True), 1e-12)
    Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    S = Pn @ Pn.T
    iu = np.triu_indices(S.shape[0], k=1)
    cos = S[iu]
    tot = A.sum(axis=1)
    return {
        'n': int(M.shape[0]),
        'cos_medio_entre_pacientes': float(cos.mean()),
        'cos_mediano': float(np.median(cos)),
        'cos_p5': float(np.percentile(cos, 5)),
        'sd_max_fraccion_por_posicion': float(P.std(axis=0).max()),
        'sd_media_fraccion_por_posicion': float(P.std(axis=0).mean()),
        'magnitud_ratio_p95_p5': float(np.percentile(tot, 95) /
                                       max(np.percentile(tot, 5), 1e-12)),
        'posicion_pico_del_perfil_medio': int(np.argmax(A.mean(axis=0))),
        'n_pacientes_con_pico_en_la_moda': int(
            np.bincount(np.argmax(A, axis=1), minlength=M.shape[1]).max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=N_DEFAULT)
    args = ap.parse_args()

    from modelo_cnn2d_v4_pytorch import CNN2DModelV4

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()
    print("=" * 74)
    print("W2 CON REFERENCIAS ALTERNATIVAS")
    print("inicio %s   device %s   n=%d" % (t0.strftime('%H:%M:%S'), DEVICE, args.n))
    print("=" * 74)

    model_pt = _resolver(CNN_MODEL_DIR, CNN_MODEL_GLOB, "modelo CNN-2D")
    data_pt = _resolver(CNN_DATA_DIR, CNN_DATA_GLOB, "datos test V4")
    with h5py.File(data_pt, 'r') as f:
        X = f['X_events'][:args.n].astype(np.float32)
        D = f['X_demograficos'][:args.n].astype(np.float32)
        y = np.asarray(f['y'][:args.n]).ravel()
    print("  n=%d  positivos=%d" % (len(y), int(y.sum())), flush=True)

    occ = (np.abs(X) > 0).any(axis=1)
    n_ev = occ.sum(axis=1)
    print("  eventos reales: mediana %.1f, P95 %.1f" % (np.median(n_ev),
                                                        np.percentile(n_ev, 95)))

    model = CNN2DModelV4(n_features=1541, n_eventos=100, n_demograficos=15).to(DEVICE)
    model.load_state_dict(torch.load(model_pt, map_location=DEVICE, weights_only=True))
    model.eval()

    Xt = torch.from_numpy(X).to(DEVICE)
    Dt = torch.from_numpy(D).to(DEVICE)

    ref_media = Xt.mean(dim=0)
    ref_mediana = torch.from_numpy(np.median(X, axis=0)).to(DEVICE)
    ocup_global = torch.from_numpy(
        (occ.mean(axis=0) > 0.5).astype(np.float32)).to(DEVICE)
    ref_media_ocupada = ref_media * ocup_global.unsqueeze(0)

    referencias = {
        '1_media_cohorte': lambda i: ref_media,
        '2_mediana_cohorte': lambda i: ref_mediana,
        '3_paciente_propio': lambda i: Xt[i],
        '4_media_solo_ocupadas': lambda i: ref_media_ocupada,
    }

    resultados = {}
    for nombre, f_ref in referencias.items():
        print("\n  referencia %s ..." % nombre, flush=True)
        M = np.empty((len(y), 100), dtype=np.float64)
        for i in range(len(y)):
            M[i] = perfil_por_ablacion(model, f_ref(i), Dt[i])
            if (i + 1) % 20 == 0:
                print("    %d/%d" % (i + 1, len(y)), end='\r', flush=True)
        print(" " * 24, end='\r')
        r = resumen_perfiles(M)
        resultados[nombre] = r
        np.save(OUTPUT_DIR / ("perfiles_%s.npy" % nombre), M)
        print("    coseno entre pacientes %.4f | sd max de la fraccion %.2e | "
              "pico en e%03d | mismo pico en %d de %d"
              % (r['cos_medio_entre_pacientes'], r['sd_max_fraccion_por_posicion'],
                 r['posicion_pico_del_perfil_medio'],
                 r['n_pacientes_con_pico_en_la_moda'], r['n']), flush=True)

    salida = {
        'fecha': t0.strftime('%Y-%m-%d %H:%M:%S'),
        'n': int(len(y)),
        'positivos': int(y.sum()),
        'mediana_eventos_reales': float(np.median(n_ev)),
        'referencia_actual_del_articulo': '1_media_cohorte',
        'valores_medidos_el_27_ago_sobre_el_explainer_guardado': {
            'cos_medio_entre_pacientes': 0.9999,
            'sd_max_fraccion_por_posicion': 2.7e-4,
            'magnitud_ratio_p95_p5': 1.00,
            'posicion_dominante': 98,
            'n_pacientes_con_ese_pico': 500,
        },
        'resultados': resultados,
    }
    with open(OUTPUT_DIR / 'resumen_w2_referencias.json', 'w', encoding='utf-8') as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 74)
    print("LECTURA")
    print("=" * 74)
    base = resultados['1_media_cohorte']['cos_medio_entre_pacientes']
    print("  La referencia actual da un coseno entre pacientes de %.4f." % base)
    for nombre in ('2_mediana_cohorte', '3_paciente_propio', '4_media_solo_ocupadas'):
        r = resultados[nombre]
        d = r['cos_medio_entre_pacientes'] - base
        print("  %-22s %.4f  (%+.4f)" % (nombre, r['cos_medio_entre_pacientes'], d))
    print()
    print("  Si alguna baja de forma apreciable, el limite era la REFERENCIA y la")
    print("  atribucion temporal si puede tener contenido individual. Eso obligaria")
    print("  a matizar la subseccion 'What the attribution supports for an")
    print("  individual patient' del articulo, y seria un resultado por si mismo.")
    print("  Si todas siguen en torno a 0,999, la redaccion actual se queda igual.")
    print("\nsalidas en %s" % OUTPUT_DIR)
    print("duracion %s" % (datetime.now() - t0))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
