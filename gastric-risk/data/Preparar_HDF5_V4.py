"""
Preparar_HDF5_V4.py

REPRODUCES IN THE THESIS
    Builds the V4 HDF5 cohort: (1541, 100) per patient, 100 events without overlap, 15
    demographic dimensions. Section 4.2 of the thesis.

PROVENANCE
    Copied verbatim from the working tree of the thesis. The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md at the
    repository root.
"""

# Rutas configurables por variable de entorno. En el arbol de trabajo de la tesis
# apuntaban a directorios locales del autor; aqui se parametrizan para que el
# script sea ejecutable fuera de esa maquina.
#!/usr/bin/env python3
"""
Preparar HDF5 V4 -- CNN 2D sobre tensor (1541, 100) por paciente, sin overlap, sin masking
==========================================================================================

Cambios principales respecto a Preparar_Datos_Modelo_HDF5_Incremental_V2.py:

  Estructura de eventos (Alex, reunion 6 May 2026):
    - V3.2: ventanas deslizantes 10 x 40 con 50% overlap = 220 eventos unicos por paciente
    - V4:   secuencia plana de 100 eventos por paciente, SIN overlap.
            Pacientes con >100 eventos: se conservan los ULTIMOS 100 (mas recientes).
            Pacientes con <100 eventos: zero-padding al final.
            SIN variable de masking (la CNN aprende a ignorar los ceros).

  Demograficos (15 dims en vez de 30):
    - EPS one-hot              (10 dims) -- igual que V2
    - Sexo one-hot             (2  dims) -- igual que V2
    - Latitud   (StandardScaler, 1 dim)  -- NUEVO, antes municipio en 6 dims
    - Longitud  (StandardScaler, 1 dim)
    - Edad      (StandardScaler, 1 dim)  -- NUEVO, antes edad bins 12 dims
    Los parametros del StandardScaler (mu, sigma) se guardan en metadata.json
    para invertir la transformacion en las graficas de NObSP.

  Geolocalizacion:
    Mapa DIVIPOLA -> (lat, lon) desde DANE (Datos Abiertos Colombia).
    Construido en ./geo/mapa_municipio_latlon.json.

  Shape final por paciente:
    X_events:        (1541, 100)  -- 768 diag + 768 proc + 5 temporal x 100 eventos
    X_demograficos:  (15,)
    y:               () -- 0 o 1

Reuso de V2:
    - cargar_embeddings_bert_raw (mismo formato .npz)
    - cargar_matrices (mismo generador de batches)
    - undersampling 100:1 + train/val/test split (mismas semillas)

Autor: Johan S. Pena-Campos (con asistencia Claude)
Fecha: 2026-05-18 (sesion 29)
"""

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

# Telegram notifications (best-effort import)
sys.path.insert(0, str(Path(os.environ.get("DBMODEL_UTILS", "./utils"))))
try:
    from notification_manager import NotificationManager
    NOTIFICATIONS_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: notification_manager no disponible: {e}")
    NOTIFICATIONS_AVAILABLE = False
    NotificationManager = None


# ==============================================================================
# CONFIGURACION GLOBAL V4
# ==============================================================================

class Config:
    BASE_PATH = Path(os.environ.get("BATCHES_DIR", "./batches"))
    EMBEDDINGS_PATH_BERT = BASE_PATH / "Embeddings" / "BERT"
    MATRICES_PATH = BASE_PATH / "MatChar"
    OUTPUT_PATH = BASE_PATH / "Datos_Modelo_V4"
    GEOCODE_MAP = Path(os.environ.get("GEOMAP_JSON", "./geo/mapa_municipio_latlon.json"))

    # Eventos -- nueva estructura V4
    MAX_EVENTOS = 100
    BERT_DIM = 768
    TEMPORAL_DIM = 5
    EVENTO_DIM = BERT_DIM * 2 + TEMPORAL_DIM   # 1541

    # Demograficos V4 (15 dims)
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

    def __init__(self):
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
# CARGA DE EMBEDDINGS BSC-Bio-EHR-ES (reusa formato V3.2)
# ==============================================================================

@log_step("1. Cargar embeddings BSC-Bio-EHR-ES")
def cargar_embeddings_bert(config: Config) -> Dict[str, Dict]:
    """
    Carga embeddings BERT pre-generados V3.2 (BSC-Bio-EHR-ES, 768 dims).
    Toma el archivo *_completos_v32_*.npz mas reciente (sort alfabetico).
    """
    diag_files = sorted(config.EMBEDDINGS_PATH_BERT.glob("bert_embeddings_diag_completos_v32_*.npz"))
    proc_files = sorted(config.EMBEDDINGS_PATH_BERT.glob("bert_embeddings_proc_completos_v32_*.npz"))

    if not diag_files or not proc_files:
        # fallback a embeddings V2 (BioBERT)
        diag_files = sorted(config.EMBEDDINGS_PATH_BERT.glob("bert_embeddings_diag_completos_*.npz"))
        proc_files = sorted(config.EMBEDDINGS_PATH_BERT.glob("bert_embeddings_proc_completos_*.npz"))

    if not diag_files or not proc_files:
        raise FileNotFoundError(
            f"No se encontraron embeddings BERT en {config.EMBEDDINGS_PATH_BERT}"
        )

    diag_file = diag_files[-1]
    proc_file = proc_files[-1]

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
# CARGA DE MATRICES (generador por batches -- reusa estructura V2)
# ==============================================================================

@log_step("2. Cargar matrices de caracteristicas")
def cargar_matrices(config: Config, notifier=None):
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
            if notifier:
                notifier.notify_custom(
                    f"V4 carga: batch {batch_idx}/{n_batches} consolidado ({len(batch_data):,} filas)",
                    emoji=""
                )

            yield batch_data
            del batch_data
            gc.collect()

    return _gen(), n_batches


# ==============================================================================
# GEOLOCALIZACION + SCALERS (PASO NUEVO V4)
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
# CREAR EVENTOS PLANOS (NUEVA -- reemplaza crear_ventanas_paciente)
# ==============================================================================

def crear_eventos_paciente_flat(
    diagnosticos: List[str],
    procedimientos: List[str],
    fechas: List[int],
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

        fecha_actual   = fechas[global_idx]
        fecha_anterior = fechas[global_idx - 1] if global_idx > 0 else 0

        # 5 temporales: igual que V2 para preservar interpretacion
        temporal = np.array([
            fecha_actual,
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
    config: Config
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
                    try:
                        fechas.append(int(float(f)))
                    except (ValueError, TypeError):
                        fechas.append(0)

        if not diags:
            continue

        eventos, n_e, d_h, p_h = crear_eventos_paciente_flat(diags, procs, fechas, embeddings, config)
        X_events[idx] = eventos
        total_eventos += n_e
        diag_hits += d_h
        proc_hits += p_h

    return X_events, total_eventos, diag_hits, proc_hits


# ==============================================================================
# DEMOGRAFICOS V4 (15 dims, NUEVA)
# ==============================================================================

@log_step("4. Codificar demograficos V4 (15 dims)")
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
        print(f"    Demograficos V4 shape: {X.shape} (dims usadas {col}/{config.DEMOGRAFICOS_DIM})")

    return X, encoders


# ==============================================================================
# HDF5 INCREMENTAL (sin X_mascaras)
# ==============================================================================

def crear_hdf5_temporal(config: Config, timestamp: str) -> Tuple[h5py.File, Path]:
    """Crea HDF5 temp con datasets X_events (EVENTO_DIM, MAX_EVENTOS), X_demograficos, y."""
    output_file = config.OUTPUT_PATH / f"datos_preparados_temp_v4_{timestamp}.h5"
    f = h5py.File(output_file, 'w')

    f.create_dataset('X_events',
        shape=(0, config.EVENTO_DIM, config.MAX_EVENTOS),
        maxshape=(None, config.EVENTO_DIM, config.MAX_EVENTOS),
        dtype='float32',
        chunks=(100, config.EVENTO_DIM, config.MAX_EVENTOS),
        compression='gzip', compression_opts=1)

    f.create_dataset('X_demograficos',
        shape=(0, config.DEMOGRAFICOS_DIM),
        maxshape=(None, config.DEMOGRAFICOS_DIM),
        dtype='float32',
        chunks=(1000, config.DEMOGRAFICOS_DIM),
        compression='gzip', compression_opts=1)

    f.create_dataset('y',
        shape=(0,), maxshape=(None,), dtype='int32',
        chunks=(1000,), compression='gzip', compression_opts=1)

    print(f"  Archivo HDF5 temporal V4: {output_file}")
    return f, output_file


def write_subbatch(f: h5py.File, X_events: np.ndarray, X_demo: np.ndarray, y: np.ndarray) -> int:
    n_new = len(y)
    cur = f['X_events'].shape[0]
    new = cur + n_new

    f['X_events'].resize((new, *f['X_events'].shape[1:]))
    f['X_demograficos'].resize((new, *f['X_demograficos'].shape[1:]))
    f['y'].resize((new,))

    f['X_events'][cur:new] = X_events
    f['X_demograficos'][cur:new] = X_demo
    f['y'][cur:new] = y
    f.flush()
    return n_new


def undersampling_v4(input_path: Path, output_path: Path, config: Config, notifier=None) -> Dict:
    """Undersampling 100:1 deterministico operando sobre HDF5 (sin RAM full)."""
    print(f"\n{'='*80}\nUndersampling V4\n{'='*80}")

    with h5py.File(input_path, 'r') as f_in:
        y_all = f_in['y'][:]
        n_total = len(y_all)
        idx_c  = np.where(y_all == 1)[0]
        idx_nc = np.where(y_all == 0)[0]
        n_c, n_nc = len(idx_c), len(idx_nc)
        print(f"  Total {n_total:,} | Cancer {n_c:,} | NoCancer {n_nc:,} (ratio {n_nc/max(n_c,1):.1f}:1)")

        n_keep = n_c * config.UNDERSAMPLING_RATIO
        if n_keep > n_nc:
            print(f"  WARNING: no hay suficientes para {config.UNDERSAMPLING_RATIO}:1")
            n_keep = n_nc

        np.random.seed(config.RANDOM_STATE)
        idx_nc_keep = np.random.choice(idx_nc, size=n_keep, replace=False)
        idx_final = np.concatenate([idx_c, idx_nc_keep])
        np.random.shuffle(idx_final)
        n_final = len(idx_final)
        print(f"  Pacientes a copiar: {n_final:,}")

        CHUNK = 5000
        n_chunks = (n_final + CHUNK - 1) // CHUNK

        with h5py.File(output_path, 'w') as f_out:
            f_out.create_dataset('X_events',
                shape=(n_final, *f_in['X_events'].shape[1:]),
                dtype='float32',
                chunks=(100, *f_in['X_events'].shape[1:]),
                compression='gzip', compression_opts=4)
            f_out.create_dataset('X_demograficos',
                shape=(n_final, f_in['X_demograficos'].shape[1]),
                dtype='float32',
                chunks=(1000, f_in['X_demograficos'].shape[1]),
                compression='gzip', compression_opts=4)
            f_out.create_dataset('y',
                shape=(n_final,), dtype='int32',
                chunks=(1000,), compression='gzip', compression_opts=4)

            for k in range(n_chunks):
                s = k * CHUNK
                e = min(s + CHUNK, n_final)
                idxs = idx_final[s:e]
                so = np.argsort(idxs)
                idx_sorted = idxs[so]
                uso = np.argsort(so)

                ev = f_in['X_events'][idx_sorted][uso]
                de = f_in['X_demograficos'][idx_sorted][uso]
                yc = f_in['y'][idx_sorted][uso]

                f_out['X_events'][s:e] = ev
                f_out['X_demograficos'][s:e] = de
                f_out['y'][s:e] = yc

                if (k + 1) % 20 == 0:
                    pct = 100 * e / n_final
                    print(f"  Chunk {k+1}/{n_chunks}: {e:,}/{n_final:,} ({pct:.1f}%)")
                    if notifier:
                        notifier.notify_custom(f"V4 undersampling: {pct:.1f}%", emoji="")

                del ev, de, yc
                gc.collect()

    with h5py.File(output_path, 'r') as f:
        yf = f['y'][:]
    return {
        'n_total': n_final,
        'n_cancer': int((yf == 1).sum()),
        'n_no_cancer': int((yf == 0).sum())
    }


def split_train_val_test_v4(input_path: Path, out_dir: Path, config: Config, timestamp: str) -> Dict[str, Path]:
    """Stratified split + escritura de 3 HDF5 separados."""
    print(f"\n{'='*80}\nSplit train/val/test V4\n{'='*80}")
    with h5py.File(input_path, 'r') as f:
        y = f['y'][:]
        n_total = len(y)

    indices = np.arange(n_total)
    idx_trainval, idx_test = train_test_split(
        indices, test_size=config.TEST_RATIO, stratify=y, random_state=config.RANDOM_STATE)
    val_ratio_adj = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_ratio_adj, stratify=y[idx_trainval], random_state=config.RANDOM_STATE)

    splits = {'train': idx_train, 'val': idx_val, 'test': idx_test}
    output_files = {}

    for name, idxs in splits.items():
        out = out_dir / f"datos_modelo_{name}_v4_{timestamp}.h5"
        n_split = len(idxs)
        print(f"  {name}: {n_split:,} pacientes -> {out.name}")

        with h5py.File(input_path, 'r') as f_in, h5py.File(out, 'w') as f_out:
            f_out.create_dataset('X_events',
                shape=(n_split, *f_in['X_events'].shape[1:]),
                dtype='float32',
                chunks=(100, *f_in['X_events'].shape[1:]),
                compression='gzip', compression_opts=4)
            f_out.create_dataset('X_demograficos',
                shape=(n_split, f_in['X_demograficos'].shape[1]),
                dtype='float32',
                chunks=(1000, f_in['X_demograficos'].shape[1]),
                compression='gzip', compression_opts=4)
            f_out.create_dataset('y',
                shape=(n_split,), dtype='int32',
                chunks=(1000,), compression='gzip', compression_opts=4)

            CHUNK = 5000
            n_chunks = (n_split + CHUNK - 1) // CHUNK
            for k in range(n_chunks):
                s = k * CHUNK
                e = min(s + CHUNK, n_split)
                idx_slice = idxs[s:e]
                so = np.argsort(idx_slice)
                idx_sorted = idx_slice[so]
                uso = np.argsort(so)

                f_out['X_events'][s:e]       = f_in['X_events'][idx_sorted][uso]
                f_out['X_demograficos'][s:e] = f_in['X_demograficos'][idx_sorted][uso]
                f_out['y'][s:e]              = f_in['y'][idx_sorted][uso]

        output_files[name] = out

    return output_files


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def main():
    config = Config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'#'*80}\n# PREPARAR HDF5 V4 -- timestamp {timestamp}\n{'#'*80}")

    notifier = NotificationManager() if NOTIFICATIONS_AVAILABLE else None
    if notifier:
        notifier.notify_custom(f"V4 build START {timestamp}", emoji="")

    # 0. Geolocalizacion + scalers
    mapa_geo = cargar_mapa_geolocalizacion(config)
    print(f"Geolocalizacion: {len(mapa_geo)} DIVIPOLA cargados")
    scalers, n_seen, n_miss = fit_scalers_one_pass(config, mapa_geo)

    # 1. Embeddings BERT
    embeddings = cargar_embeddings_bert(config)

    # 2. Crear HDF5 temporal
    f_temp, temp_path = crear_hdf5_temporal(config, timestamp)
    encoders = None
    total_written = 0

    # 3. Procesar batches y escribir incremental
    batch_gen, n_batches = cargar_matrices(config, notifier=notifier)

    for batch_idx, batch_df in enumerate(batch_gen, 1):
        print(f"\n{'-'*80}\n[Procesando batch {batch_idx}/{n_batches}] {len(batch_df):,} filas\n{'-'*80}")

        # 3a. Demograficos V4 (toda la batch a la vez, es chico: n x 15)
        X_demo, encoders = codificar_demograficos_v4(batch_df, config, mapa_geo, scalers, encoders)

        # 3b. y
        y = batch_df['Estomago_x'].fillna(0).astype(np.int32).values

        # 3c. Eventos: procesar en sub-batches de SUB_BATCH_SIZE (RAM controlada)
        diag_cols, proc_cols, fecha_cols = _detectar_cols(batch_df)
        n_in_batch = len(batch_df)
        total_eventos_b = 0
        diag_hits_b = 0
        proc_hits_b = 0
        n_subbatches = (n_in_batch + config.SUB_BATCH_SIZE - 1) // config.SUB_BATCH_SIZE
        print(f"  Procesando {n_subbatches:,} sub-batches de {config.SUB_BATCH_SIZE:,} pacientes ...")

        for sb_idx, s in enumerate(range(0, n_in_batch, config.SUB_BATCH_SIZE), 1):
            e = min(s + config.SUB_BATCH_SIZE, n_in_batch)
            sub_df = batch_df.iloc[s:e]
            X_events_sub, te, dh, ph = crear_eventos_subbatch(
                sub_df, diag_cols, proc_cols, fecha_cols, embeddings, config
            )
            write_subbatch(f_temp, X_events_sub, X_demo[s:e], y[s:e])
            total_eventos_b += te; diag_hits_b += dh; proc_hits_b += ph
            del X_events_sub, sub_df

            if sb_idx % 25 == 0 or sb_idx == n_subbatches:
                pct = 100 * e / n_in_batch
                mem_gb = psutil.Process().memory_info().rss / 1024**3
                print(f"    SubBatch {sb_idx}/{n_subbatches} ({pct:.1f}%) | mem {mem_gb:.1f} GB | "
                      f"escritos {e:,}/{n_in_batch:,}")
                if notifier and sb_idx % 100 == 0:
                    notifier.notify_custom(
                        f"V4 batch {batch_idx}/{n_batches} sub {sb_idx}/{n_subbatches} "
                        f"({pct:.1f}%) mem {mem_gb:.1f}GB", emoji=""
                    )

        if total_eventos_b:
            print(f"  Cobertura batch {batch_idx}: diag {100*diag_hits_b/total_eventos_b:.1f}% | "
                  f"proc {100*proc_hits_b/total_eventos_b:.1f}%")

        total_written += n_in_batch
        print(f"  Total escrito hasta ahora: {total_written:,}")
        if notifier:
            notifier.notify_custom(
                f"V4 build: batch {batch_idx}/{n_batches} OK, total {total_written:,}",
                emoji=""
            )

        del batch_df, X_demo, y
        gc.collect()

    f_temp.close()
    print(f"\nHDF5 temporal V4 cerrado: {total_written:,} pacientes")

    # 4. Undersampling 100:1
    bal_path = config.OUTPUT_PATH / f"datos_balanceados_v4_{timestamp}.h5"
    stats_und = undersampling_v4(temp_path, bal_path, config, notifier=notifier)

    # 5. Split train/val/test
    splits = split_train_val_test_v4(bal_path, config.OUTPUT_PATH, config, timestamp)

    # 5b. Cleanup del temp (libera ~15 GB en D:)
    try:
        if temp_path.exists():
            size_gb = temp_path.stat().st_size / (1024 ** 3)
            temp_path.unlink()
            print(f"  Temp {temp_path.name} eliminado ({size_gb:.1f} GB liberados)")
        if bal_path.exists():
            size_gb = bal_path.stat().st_size / (1024 ** 3)
            bal_path.unlink()
            print(f"  Balanced {bal_path.name} eliminado ({size_gb:.1f} GB liberados)")
    except Exception as e:
        print(f"  WARNING: cleanup parcial fallido: {e}")

    # 6. Metadata
    metadata = {
        "timestamp": timestamp,
        "version": "V4",
        "config": {
            "embedding_model": "PlanTL-GOB-ES/bsc-bio-ehr-es (768)",
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
            "undersampling_ratio": config.UNDERSAMPLING_RATIO,
            "train_ratio": config.TRAIN_RATIO,
            "val_ratio": config.VAL_RATIO,
            "test_ratio": config.TEST_RATIO,
            "random_state": config.RANDOM_STATE,
            "masking": False,
            "truncation": "last MAX_EVENTOS (most recent)"
        },
        "geocode_source": {
            "url": "https://www.datos.gov.co/Mapas-Nacionales/DIVIPOLA-C-digos-municipios-geolocalizados/vafm-j2df",
            "n_municipios": len(mapa_geo),
            "n_filas_gen_seen_for_scaler_fit": int(n_seen),
            "n_filas_municipio_missing": int(n_miss)
        },
        "scalers": serializar_scalers(scalers),
        "encoders": {
            "CodigoAdmin": {"classes": list(encoders['CodigoAdmin'].classes_.astype(str))},
            "Sexo":        {"classes": list(encoders['Sexo'].classes_.astype(str))}
        },
        "archivos": {
            "temp":   str(temp_path),
            "balanced": str(bal_path),
            "train":  str(splits['train']),
            "val":    str(splits['val']),
            "test":   str(splits['test'])
        },
        "undersampling_stats": stats_und,
        "total_written_pre_undersample": total_written
    }

    meta_path = config.OUTPUT_PATH / f"metadata_v4_{timestamp}.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"\nMetadata V4: {meta_path}")

    if notifier:
        notifier.notify_custom(
            f"V4 build COMPLETADO\n"
            f"train: {stats_und['n_total']:,} pacientes\n"
            f"cancer: {stats_und['n_cancer']:,} | nocancer: {stats_und['n_no_cancer']:,}",
            emoji=""
        )

    print(f"\n{'#'*80}\n# V4 BUILD COMPLETADO\n{'#'*80}")
    return metadata


if __name__ == "__main__":
    main()
