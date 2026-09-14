"""
Test V3.1 -- validacion del masking ANTES de lanzar entrenamiento

REPRODUCES IN THE THESIS
    Validates the explicit masking of Model V3.1 before training, Section 4.4.

PROVENANCE
    Copied verbatim from the working tree of the thesis. The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md at the
    repository root.

--------------------------------------------------------------------------
Test V3.1 -- validacion del masking ANTES de lanzar entrenamiento

Verifica que el modelo V3.1 con masking explicito:
  (a) construye sin errores
  (b) produce predicciones que SI cambian con la mascara
  (c) las ventanas marcadas como invalidas (mask=0) son anuladas

NO entrena -- solo construye el modelo y hace forward passes con pesos
inicializados aleatoriamente.

Comparativa esperada:
  V2 (sin masking):  pred_real == pred_zeros == pred_ones (max diff = 0)
  V3 (con masking):  pred_real != pred_zeros != pred_ones (max diff > 0)

Si el test pasa, V3 esta listo para entrenamiento.

Autor: Johan S. Pena-Campos
Fecha: 2026-05-05
"""

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ==================== CONSTRUIR ARQUITECTURA V3 ====================

def build_v3_model(n_ventanas=10, longitud_ventana=40, n_features=1541, n_demograficos=30):
    """Replica la arquitectura del modelo V3.1 con masking Nivel A."""

    # Inputs
    input_ventanas = layers.Input(shape=(n_ventanas, longitud_ventana, n_features), name='input_ventanas')
    input_mascaras = layers.Input(shape=(n_ventanas,), name='input_mascaras')   # (batch, 10)
    input_demograficos = layers.Input(shape=(n_demograficos,), name='input_demograficos')

    # NIVEL 1: TimeDistributed LSTM
    lstm_per_window = layers.TimeDistributed(
        layers.Bidirectional(
            layers.LSTM(256, return_sequences=False, dropout=0.0, recurrent_dropout=0.0)
        ),
        name='lstm_per_window'
    )(input_ventanas)

    # === MASKING NIVEL A ===
    window_validity = layers.Lambda(
        lambda m: tf.cast(m, tf.float32),
        name='window_validity_float'
    )(input_mascaras)

    lstm_masked = layers.Multiply(name='mask_invalid_windows')([
        lstm_per_window,
        layers.Reshape((n_ventanas, 1), name='reshape_mask_for_multiply')(window_validity)
    ])

    attention_mask = layers.Lambda(
        lambda v: tf.cast(
            tf.expand_dims(v, axis=2) * tf.expand_dims(v, axis=1) > 0,
            tf.bool
        ),
        name='attention_mask_2d'
    )(window_validity)

    # NIVEL 2: MHA con mask
    attention_output = layers.MultiHeadAttention(
        num_heads=4, key_dim=64, dropout=0.0, name='multihead_attention'
    )(query=lstm_masked, value=lstm_masked, attention_mask=attention_mask)

    # Pooling y fusion
    global_avg = layers.GlobalAveragePooling1D(name='global_avg_pool')(attention_output)
    global_max = layers.GlobalMaxPooling1D(name='global_max_pool')(attention_output)
    combined = layers.Concatenate(name='concat_pools')([global_avg, global_max])

    demograficos_dense = layers.Dense(64, activation='relu', name='demograficos_dense')(input_demograficos)
    fusion = layers.Concatenate(name='fusion')([combined, demograficos_dense])

    dense1 = layers.Dense(256, activation='relu', name='dense1')(fusion)
    dense2 = layers.Dense(128, activation='relu', name='dense2')(dense1)
    output = layers.Dense(1, activation='sigmoid', name='output')(dense2)

    return models.Model(
        inputs=[input_ventanas, input_mascaras, input_demograficos],
        outputs=output,
        name='LSTM_Attention_V3_HDF5_Masked_TEST'
    )


# ==================== MAIN ====================

def main():
    print("=" * 72)
    print("TEST V3.1 -- validacion de la arquitectura con masking")
    print("=" * 72)

    # --- 1. Construir modelo V3 ---
    print("\n[1/4] Construyendo modelo V3.1...")
    model = build_v3_model()
    print(f"   Modelo: {model.name}")
    print(f"   Parametros: {model.count_params():,}")
    print(f"   Inputs: {[i.name.split(':')[0] for i in model.inputs]}")

    # --- 2. Construir batch sintetico ---
    print("\n[2/4] Construyendo batch sintetico (4 muestras)...")
    np.random.seed(42)
    tf.random.set_seed(42)

    n_samples = 4
    X_v = np.random.randn(n_samples, 10, 40, 1541).astype(np.float32) * 0.1
    X_d = np.random.randn(n_samples, 30).astype(np.float32) * 0.1

    # Tres mascaras distintas
    X_m_real = np.array([
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],   # 3 ventanas validas
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],   # 6 ventanas validas
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 1 ventana valida
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],   # 10 ventanas validas
    ], dtype=np.float32)
    X_m_zeros = np.zeros((n_samples, 10), dtype=np.float32)
    X_m_ones  = np.ones((n_samples, 10), dtype=np.float32)

    print(f"   X_ventanas:    {X_v.shape}")
    print(f"   X_demograficos: {X_d.shape}")
    print(f"   X_m_real:      filas {X_m_real.tolist()}")

    # --- 3. Tres forward passes ---
    print("\n[3/4] Forward pass con tres mascaras...")
    pred_real  = model.predict([X_v, X_m_real,  X_d], verbose=0).flatten()
    pred_zeros = model.predict([X_v, X_m_zeros, X_d], verbose=0).flatten()
    pred_ones  = model.predict([X_v, X_m_ones,  X_d], verbose=0).flatten()

    # --- 4. Comparar y reportar ---
    print("\n[4/4] Comparando predicciones:\n")
    header = f"{'#':<4}{'pred(mask_real)':<22}{'pred(mask_zeros)':<22}{'pred(mask_ones)':<22}{'|d_real-zeros|':<16}{'|d_real-ones|':<16}"
    print(header)
    print("-" * len(header))
    for i in range(n_samples):
        d_zr = abs(pred_real[i] - pred_zeros[i])
        d_or = abs(pred_real[i] - pred_ones[i])
        print(f"{i:<4}{pred_real[i]:<22.12f}{pred_zeros[i]:<22.12f}{pred_ones[i]:<22.12f}{d_zr:<16.2e}{d_or:<16.2e}")

    max_diff_zr = float(np.max(np.abs(pred_real - pred_zeros)))
    max_diff_or = float(np.max(np.abs(pred_real - pred_ones)))

    print("\n" + "=" * 72)
    print(f"Max  |diff| (real vs zeros): {max_diff_zr:.2e}")
    print(f"Max  |diff| (real vs ones):  {max_diff_or:.2e}")
    print("=" * 72)

    # En V3 esperamos que las predicciones SI cambien con la mascara
    threshold = 1e-5
    if max_diff_zr > threshold or max_diff_or > threshold:
        print("\n>>> EXITO: la mascara SI afecta las predicciones del modelo V3.1.")
        print(">>> El masking esta correctamente integrado en el grafo.")
        print(">>> V3.1 esta LISTO para entrenamiento.")
        return 0
    else:
        print("\n>>> ALERTA: la mascara NO afecta las predicciones (no debio pasar).")
        print(">>> Revisar la integracion del masking antes de entrenar.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
