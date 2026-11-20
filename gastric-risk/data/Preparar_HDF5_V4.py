#!/usr/bin/env python3
"""
Preparar HDF5 -- CNN 2D sobre tensor (1541, 100) por paciente, sin overlap, sin masking

REPRODUCES IN THE THESIS
    Builds the HDF5 cohort of the CNN-2D model: (1541, 100) per patient, 100 events
    without overlap, 15 demographic dimensions, temporal features z-scored with
    training-partition statistics. Section 4.2 of the thesis.

PROVENANCE
    Taken from the working tree that produced the reported results. The computational
    content is unchanged; absolute paths were replaced by relative defaults and
    environment variables, notification calls were removed, and comments were edited.
    See CHANGES-FROM-THESIS.md at the repository root.

--------------------------------------------------------------------------
Estructura de eventos:
    Secuencia plana de 100 eventos por paciente, SIN overlap.
    Pacientes con >100 eventos: se conservan los ULTIMOS 100 (mas recientes).
    Pacientes con <100 eventos: zero-padding al final.
    SIN variable de masking (la CNN aprende a ignorar los ceros).

Vector de cada evento (1541):
    0-767      embedding BSC-Bio-EHR-ES del diagnostico
    768-1535   embedding BSC-Bio-EHR-ES del procedimiento
    1536-1540  5 variables temporales, en este orden:
       A dias desde el primer evento registrado del paciente
       B dias desde el evento anterior (0 en el primero)
       C indice del evento (1..n), D indice/n, E n total
    Las fechas se parsean con formato explicito (parseo_fechas.py); una fecha no
    parseable toma la ultima fecha valida del paciente (si no hay previa, la
    siguiente valida); los casos se cuentan y se reportan (fechas_calidad_cnn_*.json).
    Las 5 temporales se estandarizan (z-score) con media/desv. calculadas SOLO en la
    particion de entrenamiento (100:1), sobre eventos reales (no relleno), y se
    aplican a train/val/test. El relleno queda en 0 exacto (columna toda cero =
    relleno). Los estadisticos van en los attrs del HDF5 y en un JSON.

Demograficos (15 dims):
    - EPS one-hot              (10 dims)
    - Sexo one-hot             (2  dims)
    - Latitud   (StandardScaler, 1 dim)
    - Longitud  (StandardScaler, 1 dim)
    - Edad      (StandardScaler, 1 dim)
    Los parametros del StandardScaler (mu, sigma) se guardan en la metadata
    para invertir la transformacion en las graficas de NObSP.

Geolocalizacion:
    Mapa DIVIPOLA -> (lat, lon) desde DANE (Datos Abiertos Colombia), construido con
    geo/construir_mapa_municipio_latlon.py (ruta por GEOMAP_JSON).

Flujo:
    1. y de toda la poblacion (merge de IDs por chunks) -> undersampling 100:1
       (seed 42) y split estratificado 80/10/10 (seleccion.py). Ambos dependen
       solo de y.
    2. Eventos solo para los pacientes elegidos (carga_filtrada.py), en un HDF5
       intermedio de seleccion; se borra al final.
    3. Split train/val/test + z-score temporal.
    Cada split guarda ademas el ID del paciente, su indice de fila en la poblacion
    (gidx) y las temporales SIN escalar (X_temporal_raw). Los splits se escriben
    como *.h5.partial y se renombran al completarse; el metadata_cnn_<ts>.json
    (con "build_completo") se escribe al final.

Shape final por paciente:
    X_events:        (1541, 100)  -- 768 diag + 768 proc + 5 temporal x 100 eventos
    X_demograficos:  (15,)
    y:               () -- 0 o 1

Rutas: BATCHES_DIR (por defecto ./batches) con Embeddings/BERT, MatChar y la salida
Datos_Modelo_CNN; GEOMAP_JSON para el mapa DIVIPOLA.

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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import gc
import json
import warnings
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import h5py
import psutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

# Modulos auxiliares del builder (misma carpeta)
sys.path.insert(0, str(Path(__file__).parent))
import parseo_fechas               # noqa: E402
import carga_filtrada              # noqa: E402
import seleccion                   # noqa: E402


# ==============================================================================
# CONFIGURACION GLOBAL
# ==============================================================================

class Config:
    BASE_PATH = Path(os.environ.get("BATCHES_DIR", "./batches"))
    EMBEDDINGS_PATH_BERT = BASE_PATH / "Embeddings" / "BERT"
    MATRICES_PATH = BASE_PATH / "MatChar"
    OUTPUT_PATH = BASE_PATH / "Datos_Modelo_CNN"
    GEOCODE_MAP = Path(os.environ.get("GEOMAP_JSON", "./geo/mapa_municipio_latlon.json"))
    # Embeddings fijados por nombre, para que ningun fichero nuevo en la carpeta cambie el build
    EMB_DIAG_FILE = "bert_embeddings_diag_completos_v32_20260506_064131.npz"
    EMB_PROC_FILE = "bert_embeddings_proc_completos_v32_20260506_064131.npz"
    # Formatos de fecha (explicitos). El primero es el de MatChar_Fecha_*.csv.
    FECHA_FORMATOS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d")
    TEMPORAL_NOMBRES = ["dias_desde_primer_evento", "dias_desde_evento_anterior",
                        "indice_evento", "indice_sobre_n", "n_eventos_total"]
    # Columnas temporales dentro del vector de 1541
    TEMPORAL_FILA0 = 2 * 768   # 1536

    # Eventos
    MAX_EVENTOS = 100
    BERT_DIM = 768
    TEMPORAL_DIM = 5
    EVENTO_DIM = BERT_DIM * 2 + TEMPORAL_DIM   # 1541

    # Demograficos (15 dims)
    CODIGO_ADMIN_CATS = 10                       # EPS one-hot
    SEXO_CATS = 2                                # Sexo one-hot
    LATLON_DIMS = 2                              # lat + lon
    EDAD_DIMS = 1                                # edad continua
    DEMOGRAFICOS_DIM = CODIGO_ADMIN_CATS + SEXO_CATS + LATLON_DIMS + EDAD_DIMS  # 15

    # Pipeline
    UNDERSAMPLING_RATIO = 100
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    RANDOM_STATE = 42

    SUB_BATCH_SIZE = 2000
    CHECKPOINT_EVERY = 50

    # Controles para pruebas pequenas (en un build completo: None / 0)
    SOLO_BATCHES = None        # p.ej. {1}
    SMOKE_MAX_CHUNKS = 0       # p.ej. 1 -> solo los primeros 150.000 de cada batch leido
    SMOKE_MAX_GIDX = None      # p.ej. 60000 -> solo pacientes elegidos con gidx < 60000

    def __init__(self, output_path=None):
        if output_path is not None:            # salida alternativa (pruebas)
            self.OUTPUT_PATH = Path(output_path)
        self.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# UTILIDADES
# ==============================================================================

def print_memory_usage():
    process = psutil.Process()
    mem_gb = process.memory_info().rss / 1024**3
    available_gb = psutil.virtual_memory().available / 1024**3
    print(f"Memoria usada: {mem_gb:.2f} GB | Disponible: {available_gb:.2f} GB")


def log_step(step_name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"\n{'='*80}\nPASO: {step_name}")
            print(f"Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*80}")
            print_memory_usage()
            result = func(*args, **kwargs)
            print(f"[OK] {step_name} completado")
            return result
        return wrapper
    return decorator


# ==============================================================================
# CARGA DE EMBEDDINGS BSC-Bio-EHR-ES
# ==============================================================================

@log_step("1. Cargar embeddings BSC-Bio-EHR-ES")
def cargar_embeddings_bert(config: Config) -> Dict[str, Dict]:
    """
    Carga embeddings BERT pre-generados (BSC-Bio-EHR-ES, 768 dims) de los
    ficheros fijados en Config (sin glob ni fallback a BioBERT).
    """
    diag_file = config.EMBEDDINGS_PATH_BERT / config.EMB_DIAG_FILE
    proc_file = config.EMBEDDINGS_PATH_BERT / config.EMB_PROC_FILE
    if not diag_file.exists() or not proc_file.exists():
        raise FileNotFoundError(f"Faltan embeddings fijados: {diag_file} / {proc_file}")

    print(f"  Diagnosticos:   {diag_file.name}")
    print(f"  Procedimientos: {proc_file.name}")

    diag_data = np.load(diag_file, allow_pickle=True)
    proc_data = np.load(proc_file, allow_pickle=True)

    diag_terms = diag_data['terms']
    proc_terms = proc_data['terms']

    embeddings = {
        'diagnosticos': {
            'embeddings': diag_data['embeddings'],
            'code_to_idx': {str(t): i for i, t in enumerate(diag_terms)},
            'codes': diag_terms
        },
        'procedimientos': {
            'embeddings': proc_data['embeddings'],
            'code_to_idx': {str(t): i for i, t in enumerate(proc_terms)},
            'codes': proc_terms
        }
    }

    print(f"  Diag: {embeddings['diagnosticos']['embeddings'].shape}")
    print(f"  Proc: {embeddings['procedimientos']['embeddings'].shape}")

    return embeddings


# ==============================================================================
# CARGA DE MATRICES (generador por batches; el build usa carga_filtrada.py)
# ==============================================================================

@log_step("2. Cargar matrices de caracteristicas")
def cargar_matrices(config: Config):
    """
    Generador que yielde 1 batch a la vez (memoria controlada).
    Cada batch combina los 4 CSVs (Diag, Proc, Fecha, Gen) via merge inner por ID.
    """
    diag_batches = sorted(config.MATRICES_PATH.glob("MatChar_Diag_batch_*.csv"))
    n_batches = len(diag_batches)

    if n_batches == 0:
        raise FileNotFoundError(f"No se encontraron matrices en {config.MATRICES_PATH}")

    print(f"Batches detectados: {n_batches}")

    def _gen():
        CHUNKSIZE = 150000
        for batch_idx, diag_file in enumerate(diag_batches, 1):
            # pruebas: solo los batches indicados
            if config.SOLO_BATCHES and batch_idx not in config.SOLO_BATCHES:
                continue
            print(f"\n[Batch {batch_idx}/{n_batches}]")
            batch_suffix = diag_file.name.replace("MatChar_Diag_batch_", "").replace(".csv", "")

            paths = {
                'diag':  config.MATRICES_PATH / f"MatChar_Diag_batch_{batch_suffix}.csv",
                'proc':  config.MATRICES_PATH / f"MatChar_Proc_batch_{batch_suffix}.csv",
                'fecha': config.MATRICES_PATH / f"MatChar_Fecha_batch_{batch_suffix}.csv",
                'gen':   config.MATRICES_PATH / f"MatChar_Gen_batch_{batch_suffix}.csv",
            }

            batch_chunks = []
            readers = {k: pd.read_csv(p, chunksize=CHUNKSIZE, low_memory=False) for k, p in paths.items()}

            chunk_num = 0
            for diag_chunk, proc_chunk, fecha_chunk, gen_chunk in zip(
                readers['diag'], readers['proc'], readers['fecha'], readers['gen']
            ):
                chunk_num += 1
                diag_chunk = diag_chunk.rename(columns={diag_chunk.columns[0]: 'ID'})
                proc_chunk = proc_chunk.rename(columns={proc_chunk.columns[0]: 'ID'})
                fecha_chunk = fecha_chunk.rename(columns={fecha_chunk.columns[0]: 'ID'})
                if gen_chunk.columns[0] != 'ID':
                    gen_chunk = gen_chunk.rename(columns={gen_chunk.columns[0]: 'ID'})

                merged = diag_chunk.merge(proc_chunk, on='ID', how='inner', suffixes=('_diag', '_proc'))
                merged = merged.merge(fecha_chunk, on='ID', how='inner')
                merged = merged.merge(gen_chunk, on='ID', how='inner')

                batch_chunks.append(merged)
                memoria_gb = psutil.Process().memory_info().rss / 1024**3
                print(f"  Chunk {chunk_num}: {len(merged):,} filas | mem {memoria_gb:.1f} GB")

                del diag_chunk, proc_chunk, fecha_chunk, gen_chunk, merged
                gc.collect()
                # pruebas: cortar tras SMOKE_MAX_CHUNKS chunks
                if config.SMOKE_MAX_CHUNKS and chunk_num >= config.SMOKE_MAX_CHUNKS:
                    break

            # Compactar y consolidar
            for i, chunk in enumerate(batch_chunks):
                for col in chunk.columns:
                    if chunk[col].dtype == 'object':
                        try:
                            batch_chunks[i][col] = chunk[col].astype('category')
                        except Exception:
                            pass
            for i, chunk in enumerate(batch_chunks):
                empty_cols = chunk.columns[chunk.isna().all()]
                if len(empty_cols) > 0:
                    batch_chunks[i] = chunk.drop(columns=empty_cols)

            batch_data = pd.concat(batch_chunks, ignore_index=True, copy=False)
            del batch_chunks
            gc.collect()

            print(f"  Batch {batch_idx} consolidado: {len(batch_data):,} filas")

            yield batch_data
            del batch_data
            gc.collect()

    return _gen(), n_batches


# ==============================================================================
# GEOLOCALIZACION + SCALERS
# ==============================================================================

def cargar_mapa_geolocalizacion(config: Config) -> Dict[int, Tuple[float, float]]:
    """
    Lee mapa_municipio_latlon.json y devuelve dict {DIVIPOLA_int: (lat, lon)}.
    Fuente: DANE Datos Abiertos Colombia (vafm-j2df).
    """
    with open(config.GEOCODE_MAP, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): (v["lat"], v["lon"]) for k, v in raw.items()}


def fit_scalers_one_pass(config: Config, mapa_geo: Dict[int, Tuple[float, float]]):
    """
    Pasa una sola vez por los CSVs Gen para fittear StandardScaler de lat, lon, edad.
    Costo: lectura de ~140 MB de Gen CSVs (rapido, < 1 min).

    Returns:
        scalers: dict con 'lat', 'lon', 'edad' -> StandardScaler fitted
        n_seen: filas vistas (sanity check)
        n_muni_missing: filas con municipio que no esta en el mapa geo
    """
    print(f"\n{'='*80}\nPASO: Fit StandardScalers (lat/lon/edad)\n{'='*80}")

    gen_files = sorted(config.MATRICES_PATH.glob("MatChar_Gen_batch_*.csv"))
    if not gen_files:
        raise FileNotFoundError(f"No se encontraron Gen CSVs en {config.MATRICES_PATH}")

    lat_all, lon_all, edad_all = [], [], []
    n_muni_missing = 0
    n_total = 0

    for gf in gen_files:
        df = pd.read_csv(gf, usecols=['ID', 'Municipio', 'Edad'], low_memory=False)
        df = df.dropna(subset=['Municipio', 'Edad'])
        df['Municipio'] = df['Municipio'].astype(int)
        df['Edad']      = df['Edad'].astype(float)

        # Lookup lat/lon
        for muni_code, edad in zip(df['Municipio'].values, df['Edad'].values):
            n_total += 1
            if muni_code in mapa_geo:
                lat, lon = mapa_geo[muni_code]
                lat_all.append(lat); lon_all.append(lon); edad_all.append(edad)
            else:
                n_muni_missing += 1

        del df
        gc.collect()

    lat_arr  = np.array(lat_all,  dtype=np.float32).reshape(-1, 1)
    lon_arr  = np.array(lon_all,  dtype=np.float32).reshape(-1, 1)
    edad_arr = np.array(edad_all, dtype=np.float32).reshape(-1, 1)

    scalers = {
        'lat':  StandardScaler().fit(lat_arr),
        'lon':  StandardScaler().fit(lon_arr),
        'edad': StandardScaler().fit(edad_arr),
    }
    print(f"  lat scaler:  mu={scalers['lat'].mean_[0]:.6f}, sigma={scalers['lat'].scale_[0]:.6f}")
    print(f"  lon scaler:  mu={scalers['lon'].mean_[0]:.6f}, sigma={scalers['lon'].scale_[0]:.6f}")
    print(f"  edad scaler: mu={scalers['edad'].mean_[0]:.6f}, sigma={scalers['edad'].scale_[0]:.6f}")
    print(f"  Filas Gen procesadas: {n_total:,} (faltantes muni: {n_muni_missing:,})")

    return scalers, n_total, n_muni_missing


def serializar_scalers(scalers: Dict[str, StandardScaler]) -> Dict:
    """Convierte scalers a dict serializable JSON, con mu y sigma."""
    return {
        k: {"mean": float(s.mean_[0]), "scale": float(s.scale_[0]),
            "var":  float(s.var_[0]),  "n_samples_seen": int(s.n_samples_seen_)}
        for k, s in scalers.items()
    }


# ==============================================================================
# CREAR EVENTOS PLANOS
# ==============================================================================

def crear_eventos_paciente_flat(
    diagnosticos: List[str],
    procedimientos: List[str],
    fechas: List[float],          # dias (float) desde 1970-01-01, ya parseados e imputados
    embeddings: Dict,
    config: Config
) -> Tuple[np.ndarray, int, int, int]:
    """
    Construye matriz de eventos planos (config.EVENTO_DIM, config.MAX_EVENTOS) por paciente.

    Comportamiento:
      - Pacientes con >MAX_EVENTOS: se quedan con los ULTIMOS MAX_EVENTOS (mas recientes).
      - Pacientes con <MAX_EVENTOS: zero-padding al final (a la derecha).
      - Sin variable de masking.

    Devuelve:
      eventos      (EVENTO_DIM, MAX_EVENTOS) float32
      n_eventos    int -- conteo total real (antes de truncar)
      diag_hits    int -- cuantos diag se encontraron en el vocabulario BERT
      proc_hits    int
    """
    n_eventos_full = len(diagnosticos)
    diag_hits = 0
    proc_hits = 0

    eventos = np.zeros((config.EVENTO_DIM, config.MAX_EVENTOS), dtype=np.float32)

    if n_eventos_full == 0:
        return eventos, 0, 0, 0

    diag_emb = embeddings['diagnosticos']['embeddings']
    proc_emb = embeddings['procedimientos']['embeddings']
    diag_idx = embeddings['diagnosticos']['code_to_idx']
    proc_idx = embeddings['procedimientos']['code_to_idx']
    default_emb = np.zeros(config.BERT_DIM, dtype=np.float32)

    # Truncar: ultimos MAX_EVENTOS (mas recientes asumiendo orden cronologico ascendente)
    if n_eventos_full > config.MAX_EVENTOS:
        start = n_eventos_full - config.MAX_EVENTOS
    else:
        start = 0
    end = n_eventos_full

    n_in_window = end - start  # numero de eventos que entran en la matriz

    for local_idx, global_idx in enumerate(range(start, end)):
        d_code = str(diagnosticos[global_idx]).strip()
        if d_code in diag_idx:
            emb_d = diag_emb[diag_idx[d_code]]
            diag_hits += 1
        else:
            emb_d = default_emb

        p_code = str(procedimientos[global_idx]).strip()
        if p_code in proc_idx:
            emb_p = proc_emb[proc_idx[p_code]]
            proc_hits += 1
        else:
            emb_p = default_emb

        # A y B en dias. A se mide desde el PRIMER evento registrado (fechas[0]),
        # no desde el primero de los 100 conservados. B = 0 en el primer evento.
        fecha_actual   = fechas[global_idx]
        fecha_anterior = fechas[global_idx - 1] if global_idx > 0 else fecha_actual

        # 5 temporales (A, B, C, D, E); escala cruda aqui, el z-score se aplica
        # despues con estadisticos de train
        temporal = np.array([
            fecha_actual - fechas[0],
            fecha_actual - fecha_anterior,
            global_idx + 1,
            (global_idx + 1) / max(n_eventos_full, 1),
            n_eventos_full
        ], dtype=np.float32)

        # Columna local_idx de la matriz (filas = embedding dims, columnas = eventos)
        eventos[:config.BERT_DIM,                        local_idx] = emb_d
        eventos[config.BERT_DIM:2*config.BERT_DIM,       local_idx] = emb_p
        eventos[2*config.BERT_DIM:,                      local_idx] = temporal

    return eventos, n_eventos_full, diag_hits, proc_hits


def _detectar_cols(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Devuelve (diag_cols, proc_cols, fecha_cols) ordenadas."""
    all_cols = df.columns.tolist()
    diag_cols  = sorted([c for c in all_cols if c.endswith('_diag')],
                        key=lambda x: int(x.replace('_diag', '')) if x.replace('_diag', '').isdigit() else 0)
    proc_cols  = sorted([c for c in all_cols if c.endswith('_proc')],
                        key=lambda x: int(x.replace('_proc', '')) if x.replace('_proc', '').isdigit() else 0)
    fecha_cols = sorted([c for c in all_cols if c.isdigit()], key=lambda x: int(x))
    return diag_cols, proc_cols, fecha_cols


def crear_eventos_subbatch(
    sub_df: pd.DataFrame,
    diag_cols: List[str],
    proc_cols: List[str],
    fecha_cols: List[str],
    embeddings: Dict,
    config: Config,
    calidad: Dict = None          # acumulador de calidad de fechas
) -> Tuple[np.ndarray, int, int, int]:
    """
    Procesa un sub-batch de hasta SUB_BATCH_SIZE pacientes en RAM (sin memmap).
    Devuelve (X_events, total_eventos, diag_hits, proc_hits).

    Size en RAM por sub-batch: SUB_BATCH_SIZE x EVENTO_DIM x MAX_EVENTOS x 4 bytes.
    Con 2000 x 1541 x 100 x 4 = ~1.2 GB.
    """
    n = len(sub_df)
    X_events = np.zeros((n, config.EVENTO_DIM, config.MAX_EVENTOS), dtype=np.float32)
    total_eventos = 0
    diag_hits = 0
    proc_hits = 0

    diag_vals  = sub_df[diag_cols].values
    proc_vals  = sub_df[proc_cols].values
    fecha_vals = sub_df[fecha_cols].values

    for idx in range(n):
        diags, procs, fechas = [], [], []
        for d, p, f in zip(diag_vals[idx], proc_vals[idx], fecha_vals[idx]):
            if pd.notna(d) and pd.notna(p) and pd.notna(f):
                d_s = str(d).strip(); p_s = str(p).strip()
                if d_s and d_s != 'nan' and p_s and p_s != 'nan':
                    diags.append(d_s); procs.append(p_s)
                    # se guarda el valor crudo; se parsea por paciente mas abajo
                    fechas.append(f)
            elif pd.notna(d) and pd.notna(p) and calidad is not None:
                # evento con fecha nula: se descarta y solo se cuenta
                d_s = str(d).strip(); p_s = str(p).strip()
                if d_s and d_s != 'nan' and p_s and p_s != 'nan':
                    calidad["n_eventos_descartados_fecha_nula"] += 1

        if not diags:
            continue

        # parseo con formato explicito + imputacion documentada (parseo_fechas.py)
        fechas = parseo_fechas.fechas_paciente(fechas, calidad if calidad is not None else parseo_fechas.nueva_calidad())

        eventos, n_e, d_h, p_h = crear_eventos_paciente_flat(diags, procs, fechas, embeddings, config)
        X_events[idx] = eventos
        total_eventos += n_e
        diag_hits += d_h
        proc_hits += p_h

    return X_events, total_eventos, diag_hits, proc_hits


# ==============================================================================
# DEMOGRAFICOS (15 dims)
# ==============================================================================

@log_step("4. Codificar demograficos (15 dims)")
def codificar_demograficos_v4(
    df: pd.DataFrame,
    config: Config,
    mapa_geo: Dict[int, Tuple[float, float]],
    scalers: Dict[str, StandardScaler],
    encoders: Dict = None
) -> Tuple[np.ndarray, Dict]:
    """
    Codifica demograficos en 15 dims:
      0-9   EPS one-hot (LabelEncoder + np.eye)
      10-11 Sexo one-hot
      12    Latitud (StandardScaler)
      13    Longitud (StandardScaler)
      14    Edad (StandardScaler)
    """
    n = len(df)
    X = np.zeros((n, config.DEMOGRAFICOS_DIM), dtype=np.float32)

    if encoders is None:
        encoders = {}
        is_first = True
    else:
        is_first = False

    col = 0

    # ----- EPS -----
    if 'CodigoAdmin' in df.columns:
        if df['CodigoAdmin'].dtype.name == 'category':
            df['CodigoAdmin'] = df['CodigoAdmin'].astype('object')
        if is_first:
            le = LabelEncoder()
            enc = le.fit_transform(df['CodigoAdmin'].fillna(0))
            encoders['CodigoAdmin'] = le
        else:
            le = encoders['CodigoAdmin']
            vals = df['CodigoAdmin'].fillna(0)
            enc = np.zeros(len(vals), dtype=int)
            for i, v in enumerate(vals):
                enc[i] = le.transform([v])[0] if v in le.classes_ else 0
        X[:, col:col + config.CODIGO_ADMIN_CATS] = np.eye(config.CODIGO_ADMIN_CATS)[enc]
        col += config.CODIGO_ADMIN_CATS

    # ----- Sexo -----
    if 'Sexo' in df.columns:
        if df['Sexo'].dtype.name == 'category':
            df['Sexo'] = df['Sexo'].astype('object')
        if is_first:
            le = LabelEncoder()
            enc = le.fit_transform(df['Sexo'].fillna('U'))
            encoders['Sexo'] = le
        else:
            le = encoders['Sexo']
            vals = df['Sexo'].fillna('U')
            enc = np.zeros(len(vals), dtype=int)
            for i, v in enumerate(vals):
                enc[i] = le.transform([v])[0] if v in le.classes_ else 0
        X[:, col:col + config.SEXO_CATS] = np.eye(config.SEXO_CATS)[enc]
        col += config.SEXO_CATS

    # ----- Lat / Lon (StandardScaler) -----
    if 'Municipio' in df.columns:
        if df['Municipio'].dtype.name == 'category':
            df['Municipio'] = df['Municipio'].astype('object')
        muni_vals = df['Municipio'].fillna(-1).astype(int).values

        lat_raw = np.zeros(n, dtype=np.float32)
        lon_raw = np.zeros(n, dtype=np.float32)
        n_missing = 0
        for i, m in enumerate(muni_vals):
            if m in mapa_geo:
                lat_raw[i], lon_raw[i] = mapa_geo[m]
            else:
                # fallback: imputar con mean de los scalers ya fitteados
                lat_raw[i] = scalers['lat'].mean_[0]
                lon_raw[i] = scalers['lon'].mean_[0]
                n_missing += 1

        X[:, col] = scalers['lat'].transform(lat_raw.reshape(-1, 1)).ravel()
        col += 1
        X[:, col] = scalers['lon'].transform(lon_raw.reshape(-1, 1)).ravel()
        col += 1

        if n_missing > 0 and is_first:
            print(f"    [warn] {n_missing}/{n} pacientes sin lat/lon (imputados con mean)")

    # ----- Edad continua (StandardScaler) -----
    if 'Edad' in df.columns:
        if df['Edad'].dtype.name == 'category':
            df['Edad'] = df['Edad'].astype('float64')
        edad_raw = df['Edad'].fillna(scalers['edad'].mean_[0]).astype(float).values.reshape(-1, 1)
        X[:, col] = scalers['edad'].transform(edad_raw).ravel()
        col += 1

    if is_first:
        print(f"    Demograficos shape: {X.shape} (dims usadas {col}/{config.DEMOGRAFICOS_DIM})")

    return X, encoders


# ==============================================================================
# HDF5 DE SELECCION (solo los pacientes elegidos)
# ==============================================================================

def crear_hdf5_seleccion(config: Config, timestamp: str) -> Tuple[h5py.File, Path]:
    """HDF5 intermedio solo con los pacientes elegidos, en orden de poblacion:
    X_events, X_demograficos, y, gidx, ID y X_temporal_raw (5 x 100)."""
    output_file = config.OUTPUT_PATH / f"datos_seleccion_{timestamp}.h5"
    f = h5py.File(output_file, 'w')
    f.create_dataset('X_events',
        shape=(0, config.EVENTO_DIM, config.MAX_EVENTOS),
        maxshape=(None, config.EVENTO_DIM, config.MAX_EVENTOS),
        dtype='float32', chunks=(100, config.EVENTO_DIM, config.MAX_EVENTOS),
        compression='gzip', compression_opts=1)
    f.create_dataset('X_demograficos',
        shape=(0, config.DEMOGRAFICOS_DIM), maxshape=(None, config.DEMOGRAFICOS_DIM),
        dtype='float32', chunks=(1000, config.DEMOGRAFICOS_DIM),
        compression='gzip', compression_opts=1)
    f.create_dataset('y', shape=(0,), maxshape=(None,), dtype='int32',
        chunks=(1000,), compression='gzip', compression_opts=1)
    f.create_dataset('gidx', shape=(0,), maxshape=(None,), dtype='int64', chunks=(1000,))
    f.create_dataset('ID', shape=(0,), maxshape=(None,), dtype='int64', chunks=(1000,))
    f.create_dataset('X_temporal_raw',
        shape=(0, config.TEMPORAL_DIM, config.MAX_EVENTOS),
        maxshape=(None, config.TEMPORAL_DIM, config.MAX_EVENTOS),
        dtype='float32', chunks=(1000, config.TEMPORAL_DIM, config.MAX_EVENTOS),
        compression='gzip', compression_opts=4)
    print(f"  Archivo HDF5 de seleccion: {output_file}")
    return f, output_file


def write_subbatch(f: h5py.File, X_events: np.ndarray, X_demo: np.ndarray, y: np.ndarray,
                   gidx: np.ndarray, ids: np.ndarray) -> int:
    n_new = len(y)
    cur = f['X_events'].shape[0]
    new = cur + n_new
    for k in ('X_events', 'X_demograficos', 'y', 'gidx', 'ID', 'X_temporal_raw'):
        f[k].resize((new, *f[k].shape[1:]))
    f['X_events'][cur:new] = X_events
    f['X_demograficos'][cur:new] = X_demo
    f['y'][cur:new] = y
    f['gidx'][cur:new] = gidx
    f['ID'][cur:new] = ids
    f["X_temporal_raw"][cur:new] = X_events[:, Config.TEMPORAL_FILA0:Config.TEMPORAL_FILA0 + 5, :]
    f.flush()
    return n_new


def split_train_val_test(sel_path: Path, out_dir: Path, sel: Dict, config: Config,
                            timestamp: str) -> Tuple[Dict[str, Path], Dict]:
    """Escribe train/val/test leyendo del HDF5 de seleccion en el orden de filas de la
    seleccion. Antes de escribir, calcula media/desv. de las 5 temporales SOLO en train
    (eventos reales) y las aplica a los tres splits; el relleno queda en 0."""
    print(f"\n{'='*80}\nSplit train/val/test (+ z-score temporal con stats de train)\n{'='*80}")
    with h5py.File(sel_path, 'r') as f:
        g_sel = f['gidx'][:]
    if np.any(np.diff(g_sel) <= 0):
        raise RuntimeError("gidx del HDF5 de seleccion no es estrictamente creciente")

    def filas_de(gidx):
        r = np.searchsorted(g_sel, gidx)
        if np.any(r >= len(g_sel)) or np.any(g_sel[np.minimum(r, len(g_sel) - 1)] != gidx):
            raise RuntimeError("hay pacientes elegidos que no estan en el HDF5 de seleccion")
        return r

    # 1. Estadisticos de TRAIN
    rows_tr = filas_de(sel['train']['gidx'])
    with h5py.File(sel_path, 'r') as f:
        T_tr = f['X_temporal_raw'][:][np.sort(rows_tr)]   # lectura contigua y luego indexado
    st = seleccion.stats_temporales(T_tr, eje_temporal=1)
    del T_tr
    print("  Stats temporales (train, eventos reales n=%d):" % st['n_eventos_reales_train'])
    for nm, m, s in zip(config.TEMPORAL_NOMBRES, st['mean'], st['std']):
        print(f"    {nm:28s} media {m:12.4f}  desv {s:12.4f}")

    output_files = {}
    for name in ('train', 'val', 'test'):
        rows = filas_de(sel[name]['gidx'])
        out_final = out_dir / f"datos_modelo_{name}_cnn_{timestamp}.h5"
        # se escribe como *.h5.partial (no casa con los glob "*_cnn_*.h5") y se
        # renombra solo cuando el split esta completo y su y verificada
        out = out_final.with_name(out_final.name + '.partial')
        n_split = len(rows)
        print(f"  {name}: {n_split:,} pacientes -> {out.name}")
        with h5py.File(sel_path, 'r') as f_in, h5py.File(out, 'w') as f_out:
            f_out.create_dataset('X_events', shape=(n_split, *f_in['X_events'].shape[1:]),
                dtype='float32', chunks=(min(100, n_split), *f_in['X_events'].shape[1:]),
                compression='gzip', compression_opts=4)
            f_out.create_dataset('X_demograficos', shape=(n_split, f_in['X_demograficos'].shape[1]),
                dtype='float32', chunks=(min(1000, n_split), f_in['X_demograficos'].shape[1]),
                compression='gzip', compression_opts=4)
            f_out.create_dataset('y', shape=(n_split,), dtype='int32',
                chunks=(min(1000, n_split),), compression='gzip', compression_opts=4)
            f_out.create_dataset('X_temporal_raw', shape=(n_split, *f_in['X_temporal_raw'].shape[1:]),
                dtype='float32', chunks=(min(1000, n_split), *f_in['X_temporal_raw'].shape[1:]),
                compression='gzip', compression_opts=4)
            f_out.create_dataset('gidx', data=sel[name]['gidx'].astype(np.int64))
            f_out.create_dataset('ID', data=f_in['ID'][:][rows])
            CHUNK = 5000
            for s in range(0, n_split, CHUNK):
                e = min(s + CHUNK, n_split)
                idx_slice = rows[s:e]
                so = np.argsort(idx_slice); idx_sorted = idx_slice[so]; uso = np.argsort(so)
                X = f_in['X_events'][idx_sorted][uso]
                T = f_in['X_temporal_raw'][idx_sorted][uso]
                X = seleccion.escalar_cnn(X, T, st, fila0=config.TEMPORAL_FILA0)
                f_out['X_events'][s:e] = X
                f_out['X_temporal_raw'][s:e] = T
                f_out['X_demograficos'][s:e] = f_in['X_demograficos'][idx_sorted][uso]
                f_out['y'][s:e] = f_in['y'][idx_sorted][uso]
                del X, T
            # control: y del split == y esperada por la seleccion
            if not np.array_equal(f_out['y'][:], sel[name]['y']):
                raise RuntimeError(f"y de {name} no coincide con la seleccion")
            for k, v in (('temporal_mean', st['mean']), ('temporal_std', st['std'])):
                f_out.attrs[k] = np.asarray(v, np.float64)
            f_out.attrs['temporal_nombres'] = json.dumps(config.TEMPORAL_NOMBRES)
            f_out.attrs['temporal_filas'] = np.arange(config.TEMPORAL_FILA0, config.TEMPORAL_FILA0 + 5)
            f_out.attrs['politica_relleno'] = 'relleno = 0 exacto (columna toda cero)'
            f_out.attrs['version'] = 'CNN-2D (1541 x 100), temporales z-score'
        os.replace(out, out_final)
        output_files[name] = out_final
    return output_files, st


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def main(output_path=None, smoke: Dict = None, reusar_seleccion: str = None):
    """Flujo preseleccion. smoke = {'solo_batches': {1}, 'max_chunks': 1,
    'max_gidx': 60000, 'y_desde_gen': True} para pruebas pequenas."""
    config = Config(output_path)
    y_desde_gen = False
    if smoke:
        config.SOLO_BATCHES = smoke.get('solo_batches')
        config.SMOKE_MAX_CHUNKS = smoke.get('max_chunks', 0)
        config.SMOKE_MAX_GIDX = smoke.get('max_gidx')
        y_desde_gen = smoke.get('y_desde_gen', False)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'#'*80}\n# PREPARAR HDF5 -- timestamp {timestamp}\n{'#'*80}")
    t0 = datetime.now()

    # 0. Geolocalizacion + scalers (todas las filas Gen)
    mapa_geo = cargar_mapa_geolocalizacion(config)
    print(f"Geolocalizacion: {len(mapa_geo)} DIVIPOLA cargados")
    scalers, n_seen, n_miss = fit_scalers_one_pass(config, mapa_geo)

    # Paso 1: y de toda la poblacion y seleccion
    print(f"\n{'='*80}\nPASO 1: y de la poblacion (merge de IDs por chunks)\n{'='*80}")
    y_all, ids_all, filas_batch = seleccion.y_poblacion(config.MATRICES_PATH, solo_gen=y_desde_gen)
    sel = seleccion.seleccionar(y_all, ratio=config.UNDERSAMPLING_RATIO, seed=config.RANDOM_STATE,
                                   train_ratio=config.TRAIN_RATIO, val_ratio=config.VAL_RATIO,
                                   test_ratio=config.TEST_RATIO)
    print(f"  Poblacion {len(y_all):,} | cancer {int(y_all.sum()):,} | elegidos {len(sel['idx_final']):,}")
    for s in ('train', 'val', 'test'):
        print(f"  {s}: {len(sel[s]['gidx']):,} (cancer {int(sel[s]['y'].sum())})")
    sel_npz = config.OUTPUT_PATH / f"seleccion_indices_cnn_{timestamp}.npz"
    seleccion.guardar_seleccion(sel_npz, sel, ids_all, y_all)

    if config.SMOKE_MAX_GIDX is not None:     # smoke: solo elegidos con gidx pequeno
        for s in ('train', 'val', 'test'):
            keep = sel[s]['gidx'] < config.SMOKE_MAX_GIDX
            for k in ('gidx', 'y', 'pos_balanced'):
                sel[s][k] = sel[s][k][keep]
        print(f"  [smoke] gidx < {config.SMOKE_MAX_GIDX}: "
              + ", ".join(f"{s} {len(sel[s]['gidx'])}" for s in ('train', 'val', 'test')))
    elegidos = np.sort(np.concatenate([sel[s]['gidx'] for s in ('train', 'val', 'test')]))

    # 1. Embeddings BERT
    embeddings = cargar_embeddings_bert(config)

    # Paso 2: eventos solo para los elegidos
    # --reusar-seleccion <h5> salta el paso 2 si ya existe un HDF5 de seleccion completo
    if reusar_seleccion:
        sel_path = Path(reusar_seleccion)
        with h5py.File(sel_path, 'r') as fh:
            g_prev = fh['gidx'][:]
        if not np.array_equal(np.sort(g_prev), elegidos):
            raise RuntimeError("el HDF5 de seleccion reutilizado no contiene exactamente los elegidos")
        ts_prev = sel_path.stem.replace('datos_seleccion_', '')
        cal_prev = sel_path.parent / f"fechas_calidad_cnn_{ts_prev}.json"
        calidad = json.load(open(cal_prev, encoding='utf-8')) if cal_prev.exists() else parseo_fechas.nueva_calidad()
        enc_meta = None
        print(f"  [reusar] seleccion {sel_path.name} ({len(g_prev):,} pacientes); calidad de {cal_prev.name}")
        # encoders (solo para la metadata): ajuste con el Gen del batch 1 completo
        gen1 = sorted(config.MATRICES_PATH.glob("MatChar_Gen_batch_*.csv"))[0]
        g1 = pd.read_csv(gen1, usecols=['CodigoAdmin', 'Sexo'], low_memory=False)
        encoders = {'CodigoAdmin': LabelEncoder().fit(g1['CodigoAdmin'].fillna(0)),
                    'Sexo': LabelEncoder().fit(g1['Sexo'].fillna('U'))}
        batch_gen, n_batches, batch_ids_pob = [], 0, None
    else:
        f_sel, sel_path = crear_hdf5_seleccion(config, timestamp)
        encoders = None
        calidad = parseo_fechas.nueva_calidad()
        # solo las filas elegidas de cada chunk (+ ID/y del batch entero para los controles)
        batch_gen = carga_filtrada.cargar_batches_filtrados(
            config.MATRICES_PATH, elegidos, filas_batch, solo_batches=config.SOLO_BATCHES,
            max_chunks=config.SMOKE_MAX_CHUNKS)
        n_batches = len(filas_batch)
    offset = 0
    for batch_idx, offset, sub_all, ids_full, y_full in batch_gen:
        n_in_batch = len(ids_full)
        print(f"\n{'-'*80}\n[Procesando batch {batch_idx}/{n_batches}] {n_in_batch:,} filas "
              f"(gidx {offset:,}..{offset + n_in_batch - 1:,})\n{'-'*80}")
        # control de alineacion con el paso 1 (mismos IDs en el mismo orden)
        if not np.array_equal(np.asarray(ids_full, dtype=ids_all.dtype),
                              ids_all[offset:offset + n_in_batch]):
            raise RuntimeError(f"batch {batch_idx}: IDs no alineados con el paso 1")
        # y de TODO el batch igual a la del paso 1
        if not np.array_equal(pd.Series(y_full).fillna(0).astype(np.int32).values,
                              y_all[offset:offset + n_in_batch]):
            raise RuntimeError(f"batch {batch_idx}: y no coincide con el paso 1")

        # 3a. Encoders: batch 1 COMPLETO, ajustados con el Gen del batch 1 (mismas filas y valores
        # que el batch 1 fusionado), porque el batch entero no se retiene en memoria
        if encoders is None:
            gen1 = sorted(config.MATRICES_PATH.glob("MatChar_Gen_batch_*.csv"))[0]
            _, encoders = codificar_demograficos_v4(pd.read_csv(gen1, low_memory=False), config,
                                                    mapa_geo, scalers, None)

        loc = elegidos[(elegidos >= offset) & (elegidos < offset + n_in_batch)] - offset
        print(f"  elegidos en este batch: {len(loc):,}")
        if len(loc) == 0:
            del sub_all; gc.collect(); continue
        if len(sub_all) != len(loc) or not np.array_equal(sub_all['ID'].values, ids_full[loc]):
            raise RuntimeError(f"batch {batch_idx}: filas filtradas no coinciden con los elegidos")
        ids_b = ids_full
        X_demo, _ = codificar_demograficos_v4(sub_all.copy(), config, mapa_geo, scalers, encoders)
        y = sub_all['Estomago_x'].fillna(0).astype(np.int32).values
        diag_cols, proc_cols, fecha_cols = _detectar_cols(sub_all)
        # el emparejamiento evento-fecha es posicional; exigir las mismas columnas
        kd = [c[:-5] for c in diag_cols]; kp = [c[:-5] for c in proc_cols]
        if not (kd == kp == fecha_cols):
            raise RuntimeError(f"columnas diag/proc/fecha desalineadas: {len(kd)}/{len(kp)}/{len(fecha_cols)}")

        for s in range(0, len(loc), config.SUB_BATCH_SIZE):
            e = min(s + config.SUB_BATCH_SIZE, len(loc))
            X_events_sub, te, dh, ph = crear_eventos_subbatch(
                sub_all.iloc[s:e], diag_cols, proc_cols, fecha_cols, embeddings, config, calidad)
            write_subbatch(f_sel, X_events_sub, X_demo[s:e], y[s:e],
                           (loc[s:e] + offset).astype(np.int64),
                           np.asarray(ids_b[loc[s:e]], dtype=np.int64))
            del X_events_sub
        print(f"  escritos {len(loc):,} | calidad acumulada: invalidas {calidad['n_fecha_invalida']:,} "
              f"de {calidad['n_eventos']:,} eventos")
        del sub_all, X_demo, y, ids_full, y_full, ids_b
        gc.collect()
    if not reusar_seleccion:
        f_sel.close()

    # Calidad de fechas
    cal = parseo_fechas.resumen_calidad(calidad)
    cal_path = config.OUTPUT_PATH / f"fechas_calidad_cnn_{timestamp}.json"
    with open(cal_path, 'w', encoding='utf-8') as fh:
        json.dump(cal, fh, indent=2, default=str)
    print(f"\nCalidad de fechas -> {cal_path}")
    for k in ('n_eventos', 'n_fecha_invalida', 'frac_eventos_fecha_invalida', 'n_fecha_formato_solo_dia',
              'n_eventos_descartados_fecha_nula', 'frac_intervalos_negativos',
              'frac_pacientes_no_monotonos', 'fecha_min_iso', 'fecha_max_iso'):
        print(f"  {k}: {cal.get(k)}")

    # fechas no monotonas / invalidas / fuera de rango -> aviso fuerte; revisar antes de entrenar
    aviso_fechas = {k: calidad[k] for k in ('n_intervalos_negativos', 'n_pacientes_no_monotonos',
                                             'n_fecha_invalida', 'n_fecha_fuera_rango_2015_07_2016_12')}
    if any(aviso_fechas.values()):
        print("\n" + "!" * 80 + "\nAVISO CALIDAD FECHAS: " + json.dumps(aviso_fechas)
              + "\nRevisar antes de entrenar: reordenar o documentar."
              + "\n" + "!" * 80, flush=True)

    # 5. Split + z-score
    splits, st = split_train_val_test(sel_path, config.OUTPUT_PATH, sel, config, timestamp)
    st_path = config.OUTPUT_PATH / f"temporal_scaler_cnn_{timestamp}.json"
    seleccion.guardar_stats_json(st_path, st, config.TEMPORAL_NOMBRES,
                                    extra={'version': 'CNN', 'timestamp': timestamp},
                                    n_train=len(sel['train']['gidx']))

    if not smoke and not reusar_seleccion:
        try:
            size_gb = sel_path.stat().st_size / (1024 ** 3)
            sel_path.unlink()
            print(f"  Seleccion {sel_path.name} eliminada ({size_gb:.1f} GB)")
        except Exception as e:
            print(f"  WARNING: no se pudo borrar la seleccion: {e}")

    # 6. Metadata
    metadata = {
        "timestamp": timestamp,
        "version": "CNN-2D",
        "config": {
            "embedding_model": "PlanTL-GOB-ES/bsc-bio-ehr-es (768)",
            "embedding_files": [config.EMB_DIAG_FILE, config.EMB_PROC_FILE],
            "evento_dim": config.EVENTO_DIM,
            "max_eventos": config.MAX_EVENTOS,
            "demograficos_dim": config.DEMOGRAFICOS_DIM,
            "demograficos_layout": [
                "EPS_0..9 (one-hot, 10)",
                "Sexo_F, Sexo_M (one-hot, 2)",
                "lat_z (StandardScaler, 1)",
                "lon_z (StandardScaler, 1)",
                "edad_z (StandardScaler, 1)"
            ],
            "temporal_layout": config.TEMPORAL_NOMBRES,
            "temporal_escalado": "z-score con stats de train (eventos reales); relleno = 0",
            "undersampling_ratio": config.UNDERSAMPLING_RATIO,
            "train_ratio": config.TRAIN_RATIO,
            "val_ratio": config.VAL_RATIO,
            "test_ratio": config.TEST_RATIO,
            "random_state": config.RANDOM_STATE,
            "masking": False,
            "truncation": "last MAX_EVENTOS (most recent)",
            "flujo": "preseleccion (y poblacion -> seleccion 100:1 -> eventos solo de los elegidos)",
            "smoke": smoke or None,
        },
        "geocode_source": {
            "url": "https://www.datos.gov.co/Mapas-Nacionales/DIVIPOLA-C-digos-municipios-geolocalizados/vafm-j2df",
            "n_municipios": len(mapa_geo),
            "n_filas_gen_seen_for_scaler_fit": int(n_seen),
            "n_filas_municipio_missing": int(n_miss)
        },
        "scalers": serializar_scalers(scalers),
        "temporal_scaler": st,
        "fechas_calidad": cal,
        "aviso_calidad_fechas": aviso_fechas,
        "build_completo": True,          # este JSON se escribe al final = marca de build completo
        "reusar_seleccion": reusar_seleccion,
        "encoders": {
            "CodigoAdmin": {"classes": list(encoders['CodigoAdmin'].classes_.astype(str))},
            "Sexo":        {"classes": list(encoders['Sexo'].classes_.astype(str))}
        },
        "archivos": {
            "seleccion_indices": str(sel_npz),
            "train":  str(splits['train']),
            "val":    str(splits['val']),
            "test":   str(splits['test']),
            "temporal_scaler_json": str(st_path),
            "fechas_calidad_json": str(cal_path),
        },
        "undersampling_stats": {
            'n_total': int(len(sel['idx_final'])),
            'n_cancer': int(sel['y_bal'].sum()),
            'n_no_cancer': int((sel['y_bal'] == 0).sum())
        },
        "total_poblacion": int(len(y_all)),
        "duracion_s": (datetime.now() - t0).total_seconds(),
    }
    meta_path = config.OUTPUT_PATH / f"metadata_cnn_{timestamp}.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"\nMetadata: {meta_path}")
    print(f"\n{'#'*80}\n# BUILD COMPLETADO\n{'#'*80}")
    return metadata


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument('--reusar-seleccion', default=None,
                     help='HDF5 datos_seleccion_<ts>.h5 ya completo: salta la construccion de eventos')
    _a = _ap.parse_args()
    main(reusar_seleccion=_a.reusar_seleccion)
