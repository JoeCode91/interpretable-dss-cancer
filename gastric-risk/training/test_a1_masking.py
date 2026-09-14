"""
Test empirico A1: X_mascaras NO afecta las predicciones del modelo V2?

REPRODUCES IN THE THESIS
    Empirical test A1: shows that the masking tensor of Model V2 does not affect its
    predictions. This is what motivates the explicit-masking correction of V3.1,
    Section 4.4.

PROVENANCE
    Copied verbatim from the working tree of the thesis. The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md at the
    repository root.

--------------------------------------------------------------------------

Si las predicciones con mascara real, mascara cero y mascara uno son
IDENTICAS hasta la precision de float32, queda probado empiricamente que
la mascara esta aislada del grafo computacional (hallazgo A1 del
TODO_Cap4_Correcciones.md).

Esto valida que:
  - El gradiente de la perdida no fluye hacia X_mascaras
  - El modelo no "ve" la mascara para producir el output
  - El padding entra al grafo como datos reales

Si el test FALLA (las predicciones cambian con la mascara), entonces
la mascara SI esta conectada y A1 era un falso positivo.

Salida esperada bajo A1 verdadero:
  - max |diff| < 1e-5  (idealmente 0.0)
  - mean |diff| < 1e-5

Autor: Johan S. Pena-Campos
Fecha: 2026-05-05
Sesion: 23 (auditoria pre V3.1)

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


import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import h5py
import tensorflow as tf

# ==================== CONFIGURACION ====================

MODEL_H5 = "./batches/Modelos_V2_HDF5/modelo_v2_hdf5_20251205_083444_best.h5"
TEST_DATA = "./batches/Datos_Modelo/datos_modelo_test_20251124_142346.h5"
N_SAMPLES = 8  # suficiente para validar; mas seria redundante

# ==================== MAIN ====================

def main():
    print("=" * 72)
    print("TEST EMPIRICO A1 -- X_mascaras NO afecta predicciones del modelo V2?")
    print("=" * 72)

    # --- 1. Cargar modelo ---
    print("\n[1/4] Cargando modelo V2...")
    model = tf.keras.models.load_model(MODEL_H5, compile=False)
    input_names = [i.name.split(':')[0] for i in model.inputs]
    print(f"   Modelo: {model.name}")
    print(f"   Parametros: {model.count_params():,}")
    print(f"   Inputs: {input_names}")

    # --- 2. Cargar muestras de test ---
    print(f"\n[2/4] Cargando {N_SAMPLES} muestras del test set...")
    with h5py.File(TEST_DATA, 'r') as f:
        X_v = f['X_ventanas'][:N_SAMPLES].astype(np.float32)
        X_m = f['X_mascaras'][:N_SAMPLES].astype(np.float32)
        X_d = f['X_demograficos'][:N_SAMPLES].astype(np.float32)
        y_true = f['y'][:N_SAMPLES]

    print(f"   X_ventanas:     {X_v.shape}")
    print(f"   X_mascaras:     {X_m.shape}  (real, suma promedio {X_m.mean():.4f})")
    print(f"   X_demograficos: {X_d.shape}")
    print(f"   y_true:         {y_true.tolist()}")

    # --- 3. Generar tres predicciones con mascaras distintas ---
    print("\n[3/4] Prediciendo con tres mascaras distintas...")

    X_m_zeros = np.zeros_like(X_m, dtype=np.float32)
    X_m_ones  = np.ones_like(X_m, dtype=np.float32)

    pred_real  = model.predict([X_v, X_m,       X_d], verbose=0).flatten()
    pred_zeros = model.predict([X_v, X_m_zeros, X_d], verbose=0).flatten()
    pred_ones  = model.predict([X_v, X_m_ones,  X_d], verbose=0).flatten()

    # --- 4. Comparar y reportar ---
    print("\n[4/4] Comparando predicciones:\n")

    header = f"{'#':<4}{'y':<4}{'pred(mask_real)':<20}{'pred(mask_zeros)':<20}{'pred(mask_ones)':<20}{'|d_real-zeros|':<16}{'|d_real-ones|':<16}"
    print(header)
    print("-" * len(header))
    for i in range(N_SAMPLES):
        d_zr = abs(pred_real[i] - pred_zeros[i])
        d_or = abs(pred_real[i] - pred_ones[i])
        print(f"{i:<4}{int(y_true[i]):<4}{pred_real[i]:<20.12f}{pred_zeros[i]:<20.12f}{pred_ones[i]:<20.12f}{d_zr:<16.2e}{d_or:<16.2e}")

    max_diff_zr = float(np.max(np.abs(pred_real - pred_zeros)))
    max_diff_or = float(np.max(np.abs(pred_real - pred_ones)))
    mean_diff_zr = float(np.mean(np.abs(pred_real - pred_zeros)))
    mean_diff_or = float(np.mean(np.abs(pred_real - pred_ones)))

    print("\n" + "=" * 72)
    print(f"Max  |diff| (real vs zeros): {max_diff_zr:.2e}")
    print(f"Max  |diff| (real vs ones):  {max_diff_or:.2e}")
    print(f"Mean |diff| (real vs zeros): {mean_diff_zr:.2e}")
    print(f"Mean |diff| (real vs ones):  {mean_diff_or:.2e}")
    print("=" * 72)

    threshold = 1e-5
    if max_diff_zr < threshold and max_diff_or < threshold:
        print("\n>>> CONFIRMADO: X_mascaras NO afecta las predicciones del modelo V2.")
        print(">>> Hallazgo A1 es REAL: la mascara esta AISLADA del grafo computacional.")
        print(">>> Procedente: re-entrenar V3.1 con masking explicito (Nivel A).")
        return 0
    else:
        print("\n>>> INESPERADO: las predicciones SI cambian con la mascara.")
        print(">>> A1 podria ser falso positivo. Revisar el modelo en detalle.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
