#!/usr/bin/env python3
"""
Synthetic cohort generator for the gastric cancer risk pipeline
===============================================================

WHY THIS EXISTS
    The clinical records behind Chapter 4 of the thesis cannot be published. They are
    longitudinal diagnosis and procedure sequences for 4.9 million patients of the
    Colombian health system, together with municipality of residence, and they are not the
    author's data to distribute: they were accessed under ethics approval FM-CIE-030920 and
    remain under the custody of the issuing institutions.

    Without them the pipeline cannot be run at all, which would make the published code
    unverifiable. This generator closes that gap. It emits HDF5 files with the same
    schema, the same array shapes and the same class ratio as the real cohort, so that
    every downstream script runs end to end.

WHAT IT DOES AND DOES NOT REPRODUCE
    It reproduces the MECHANICS of the pipeline: shapes, dtypes, dataset names, the
    ordering convention inside each event vector, the padding rule, the scaling of the
    temporal features, and the class ratio.

    It does NOT reproduce the STATISTICS of the Colombian cohort. Embeddings are rows of a
    random vocabulary of unit vectors, not of the BSC-Bio-EHR-ES encoder; codes are
    synthetic labels (SYN-D0001, SYN-P0001, ...), not ICD-10 or CUPS codes; ages,
    coordinates and dates are drawn over plausible ranges; there is no correlation between
    events and the label beyond a deliberately weak, documented signal.

    A model trained on this data will not reach the metrics reported in the thesis, and
    should not be expected to. Any result obtained here is a statement about the code, not
    about gastric cancer.

    No record derives from any real patient. Nothing here encodes any property of the
    source cohort other than its array dimensions and its class ratio, both of which are
    already published in the thesis.

SCHEMA (from Preparar_HDF5_V4.py)
    X_events        (n, 1541, 100)  float32
                    1541 = 768 diagnosis embedding
                         + 768 procedure embedding
                         +   5 temporal features, rows 1536-1540:
                           A days since the first recorded event, B days since the
                           previous event (0 for the first), C event index (1..n),
                           D index / n, E n; z-scored with the mean and standard
                           deviation of the real events of the training partition.
                    100  = events per patient: patients with more than 100 keep the
                           LAST 100; patients with fewer are zero-padded on the right.
                           Padding columns are exactly zero, temporal rows included.
    X_temporal_raw  (n, 5, 100)     float32, the five temporal features before scaling
    X_demograficos  (n, 15)         float32
                    10 EPS one-hot + 2 sex one-hot + latitude + longitude + age,
                    the last three standardised.
    y               (n,)            int32, 1 = gastric cancer
    ID, gidx        (n,)            int64, patient identifier and row in the population
    attrs           temporal_mean, temporal_std, temporal_nombres, temporal_filas

USAGE
    Single file (one partition, temporal z-score with the statistics of the file itself):
        python generate_synthetic_cohort.py --out synthetic_cohort.h5 --n 2000

    Pipeline layout (what the training and interpretability scripts read):
        python generate_synthetic_cohort.py --pipeline-dir batches --n 6000

    writes, under <dir>:
        Datos_Modelo_CNN/datos_modelo_{train,val,test}_cnn_<ts>.h5   stratified 80/10/10,
                         temporal z-score with training statistics
        Datos_Modelo_CNN/metadata_cnn_<ts>.json   scalers, temporal scaler, file paths
        Embeddings/BERT/bert_embeddings_{diag,proc}_completos_v32_synthetic.npz
                         the synthetic vocabularies (embeddings, terms)
        embeddings_map/mapa_cie10.json, mapa_cups.json   code -> synthetic description
    With --pipeline-dir batches, the scripts' default ./batches paths find the data;
    otherwise point CNN_DATA_DIR, EMB_DIR and EMB_MAP_DIR at those folders.

    --n N         number of patients (default 2000)
    --ratio R     negatives per positive (default 100, the undersampled ratio used for
                  training and evaluation in the thesis; pass 5103 for the population ratio)
    --events-max  events per patient (default 100)
    --seed S      random seed (default 0)
    --signal S    how strongly the label shifts the code distribution (default 0.15): a
                  positive patient draws each diagnosis from a small "risk" subset of the
                  vocabulary with probability 2*S instead of uniformly. Zero gives pure
                  noise and any model will be at chance; larger values make the task
                  artificially easy. The default leaves it learnable but unimpressive,
                  which is the honest choice for a schema-level surrogate.

LICENCE
    MIT, with the rest of the repository; the synthetic data it produces is CC BY 4.0.

REPRODUCES IN THE THESIS
    Nothing directly. This script exists so that the pipeline of Chapter 4 can be run
    without the restricted clinical data, which are not distributed.

PROVENANCE
    Written for this repository; it has no counterpart in the working tree of the thesis.
    See CHANGES-FROM-THESIS.md at the repository root.

AUTHORSHIP
    Johan S. Pena-Campos
    PhD thesis, "Interpretable Decision Support Systems for Early Cancer Diagnosis and
    Personalized Treatment", Universitat Politecnica de Catalunya and Pontificia
    Universidad Javeriana, 2026.

USE OF AI ASSISTANCE
    This file was written with the assistance of generative AI (Claude, Anthropic),
    to the author's specification, and reviewed, corrected and validated by the
    author, who takes full responsibility for it. The
    research question this script answers, and the interpretation of what it returns,
    are the author's.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("h5py is required: pip install h5py")

# --- dimensions, from Preparar_HDF5_V4.py ---
BERT_DIM = 768
TEMPORAL_DIM = 5
TEMPORAL_FILA0 = 2 * BERT_DIM                    # 1536
EVENTO_DIM = BERT_DIM * 2 + TEMPORAL_DIM        # 1541
EPS_CATS, SEXO_CATS = 10, 2
DEMOGRAFICOS_DIM = EPS_CATS + SEXO_CATS + 2 + 1  # 15
TEMPORAL_NOMBRES = ["dias_desde_primer_evento", "dias_desde_evento_anterior",
                    "indice_evento", "indice_sobre_n", "n_eventos_total"]
COL_INDICE = 2                                   # C, the event index: > 0 on real events

# Synthetic vocabularies. The sizes are arbitrary.
N_DIAG, N_PROC, N_RISK = 400, 250, 20

# Colombia, roughly. Only used to keep the geographic features in a plausible range;
# no municipality of the real cohort is reproduced.
LAT_RANGE = (-4.2, 12.5)
LON_RANGE = (-79.0, -66.8)
WINDOW_DAYS = 549.0                              # length of the observation window


def build_vocab(n, prefix, rng):
    """n unit vectors of dimension BERT_DIM and their synthetic code labels."""
    emb = rng.normal(0.0, 1.0, size=(n, BERT_DIM)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    terms = np.array([f"{prefix}{i:04d}" for i in range(1, n + 1)])
    return emb, terms


def build_events(n, max_eventos, y, signal, vocab_d, vocab_p, rng):
    """
    One (EVENTO_DIM, max_eventos) tensor per patient, with RAW temporal rows.

    Event counts are drawn from a lognormal, which gives the long right tail that real
    clinical histories have, and are then clipped at max_eventos. Patients below the cap
    are zero-padded on the right.
    """
    X = np.zeros((n, EVENTO_DIM, max_eventos), dtype=np.float32)
    counts = np.clip(rng.lognormal(mean=3.0, sigma=0.9, size=n).astype(int), 1, max_eventos)
    p_risk = min(1.0, 2.0 * signal)

    for i in range(n):
        k = counts[i]
        # Diagnosis and procedure codes are rows of the synthetic vocabularies, as the real
        # embeddings are rows of the cached encoder output. The label signal is a shift in
        # how often positives draw from a small subset of diagnosis codes.
        d_idx = rng.integers(0, N_DIAG, k)
        if y[i] == 1 and p_risk > 0:
            risk = rng.random(k) < p_risk
            d_idx[risk] = rng.integers(0, N_RISK, int(risk.sum()))
        p_idx = rng.integers(0, N_PROC, k)

        # Dates: sorted days inside the observation window, several events per day allowed.
        days = np.sort(rng.integers(0, int(WINDOW_DAYS) + 1, k)).astype(np.float32)
        idx = np.arange(1, k + 1, dtype=np.float32)
        temporal = np.vstack([
            days - days[0],                               # A
            np.diff(days, prepend=days[0]),               # B
            idx,                                          # C
            idx / float(k),                               # D
            np.full(k, float(k), dtype=np.float32),       # E
        ]).astype(np.float32)

        X[i, :BERT_DIM, :k] = vocab_d[d_idx].T
        X[i, BERT_DIM:2 * BERT_DIM, :k] = vocab_p[p_idx].T
        X[i, TEMPORAL_FILA0:, :k] = temporal

    return X, counts


def temporal_stats(T):
    """Mean and standard deviation (ddof=0) of the five temporal features over real events."""
    Tm = np.moveaxis(T, 1, -1).reshape(-1, TEMPORAL_DIM).astype(np.float64)
    R = Tm[Tm[:, COL_INDICE] > 0]
    return {"mean": R.mean(axis=0).tolist(), "std": R.std(axis=0).tolist(),
            "n_eventos_reales_train": int(len(R))}


def zscore_events(X, T, st):
    """z-score of rows 1536-1540 on real events; padding stays exactly zero."""
    mu = np.asarray(st["mean"], np.float64)[None, :, None]
    sd = np.asarray(st["std"], np.float64)[None, :, None]
    real = (T[:, COL_INDICE, :] > 0)[:, None, :]
    X[:, TEMPORAL_FILA0:, :] = np.where(real, (T - mu) / sd, 0.0).astype(np.float32)
    return X


def build_demographics(n, y, rng):
    """15 dims: 10 EPS one-hot, 2 sex one-hot, then standardised lat, lon and age."""
    X = np.zeros((n, DEMOGRAFICOS_DIM), dtype=np.float32)

    X[np.arange(n), rng.integers(0, EPS_CATS, n)] = 1.0
    X[np.arange(n), EPS_CATS + rng.integers(0, SEXO_CATS, n)] = 1.0

    lat = rng.uniform(*LAT_RANGE, n).astype(np.float32)
    lon = rng.uniform(*LON_RANGE, n).astype(np.float32)
    # Gastric cancer incidence in Colombia is higher in the Andean south, so age is drawn
    # older for positives. This is a coarse gesture towards realism, not a model of it.
    age = rng.normal(np.where(y == 1, 63.0, 47.0), 14.0).clip(18, 95).astype(np.float32)

    stats = {}
    for j, (name, v) in enumerate([("lat", lat), ("lon", lon), ("edad", age)]):
        mu, sd = float(v.mean()), float(v.std() or 1.0)
        X[:, EPS_CATS + SEXO_CATS + j] = (v - mu) / sd
        stats[name] = {"mean": mu, "scale": sd, "var": sd ** 2, "n_samples_seen": int(n),
                       "mu": mu, "sigma": sd}

    return X, stats


def write_h5(path, X, T, D, y, ids, gidx, st, args):
    with h5py.File(path, "w") as f:
        f.create_dataset("X_events", data=X, compression="gzip", compression_opts=4,
                         chunks=(1, EVENTO_DIM, X.shape[2]))
        f.create_dataset("X_temporal_raw", data=T, compression="gzip", compression_opts=4)
        f.create_dataset("X_demograficos", data=D, compression="gzip")
        f.create_dataset("y", data=y.astype(np.int32))
        f.create_dataset("ID", data=ids.astype(np.int64))
        f.create_dataset("gidx", data=gidx.astype(np.int64))
        f.attrs["temporal_mean"] = np.asarray(st["mean"], np.float64)
        f.attrs["temporal_std"] = np.asarray(st["std"], np.float64)
        f.attrs["temporal_nombres"] = json.dumps(TEMPORAL_NOMBRES)
        f.attrs["temporal_filas"] = np.arange(TEMPORAL_FILA0, TEMPORAL_FILA0 + TEMPORAL_DIM)
        f.attrs["politica_relleno"] = "relleno = 0 exacto (columna toda cero)"
        f.attrs["synthetic"] = True
        f.attrs["warning"] = ("Synthetic data. Reproduces the schema of the thesis cohort, "
                              "not its statistics. Results obtained on it are not comparable "
                              "with those reported in the thesis.")
        f.attrs["generator_seed"] = args.seed
        f.attrs["class_ratio"] = args.ratio


def split_stratified(y, rng, train=0.8, val=0.1):
    """Stratified 80/10/10 split of row indices; each partition sorted by row."""
    parts = {"train": [], "val": [], "test": []}
    for c in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == c))
        n_tr = int(round(train * len(idx)))
        n_va = int(round(val * len(idx)))
        parts["train"].append(idx[:n_tr])
        parts["val"].append(idx[n_tr:n_tr + n_va])
        parts["test"].append(idx[n_tr + n_va:])
    return {k: np.sort(np.concatenate(v)) for k, v in parts.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="synthetic_cohort.h5")
    ap.add_argument("--pipeline-dir", default=None,
                    help="write the train/val/test layout the pipeline scripts read")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--ratio", type=float, default=100.0,
                    help="negatives per positive (100 = undersampled, 5103 = population)")
    ap.add_argument("--events-max", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--signal", type=float, default=0.15)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    n_pos = max(1, int(round(args.n / (args.ratio + 1.0))))
    y = np.zeros(args.n, dtype=np.int32)
    y[rng.choice(args.n, n_pos, replace=False)] = 1

    print("Generating %d patients, %d positive (1:%.0f)" % (args.n, n_pos, args.ratio))
    print("  tensor per patient: (%d, %d)" % (EVENTO_DIM, args.events_max))

    vocab_d, terms_d = build_vocab(N_DIAG, "SYN-D", rng)
    vocab_p, terms_p = build_vocab(N_PROC, "SYN-P", rng)
    X_events, counts = build_events(args.n, args.events_max, y, args.signal, vocab_d, vocab_p, rng)
    T_raw = X_events[:, TEMPORAL_FILA0:, :].copy()
    X_demo, stats = build_demographics(args.n, y, rng)
    gidx = np.arange(args.n, dtype=np.int64)
    ids = 10_000_000 + gidx

    if args.pipeline_dir is None:
        st = temporal_stats(T_raw)
        X_events = zscore_events(X_events, T_raw, st)
        write_h5(args.out, X_events, T_raw, X_demo, y, ids, gidx, st, args)
        with open(args.out.replace(".h5", "_metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"synthetic": True, "n": args.n, "n_positive": int(n_pos),
                       "ratio": args.ratio, "seed": args.seed, "signal": args.signal,
                       "evento_dim": EVENTO_DIM, "max_eventos": args.events_max,
                       "demograficos_dim": DEMOGRAFICOS_DIM,
                       "scalers": stats, "temporal_scaler": st,
                       "median_events_per_patient": int(np.median(counts))},
                      f, indent=2)
        print("Written %s and its metadata." % args.out)
    else:
        base = args.pipeline_dir
        ddir = os.path.join(base, "Datos_Modelo_CNN")
        edir = os.path.join(base, "Embeddings", "BERT")
        mdir = os.path.join(base, "embeddings_map")
        for d in (ddir, edir, mdir):
            os.makedirs(d, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts = split_stratified(y, rng)
        st = temporal_stats(T_raw[parts["train"]])
        files = {}
        for name, rows in parts.items():
            Xs = zscore_events(X_events[rows].copy(), T_raw[rows], st)
            path = os.path.join(ddir, f"datos_modelo_{name}_cnn_{ts}.h5")
            write_h5(path, Xs, T_raw[rows], X_demo[rows], y[rows], ids[rows], gidx[rows], st, args)
            files[name] = path
            print("  %-5s n=%5d  pos=%3d  -> %s" % (name, len(rows), int(y[rows].sum()), path))
        meta = {"timestamp": ts, "synthetic": True, "version": "synthetic",
                "config": {"evento_dim": EVENTO_DIM, "max_eventos": args.events_max,
                           "demograficos_dim": DEMOGRAFICOS_DIM,
                           "temporal_layout": TEMPORAL_NOMBRES,
                           "undersampling_ratio": args.ratio, "random_state": args.seed},
                "scalers": {k: {kk: v[kk] for kk in ("mean", "scale", "var", "n_samples_seen")}
                            for k, v in stats.items()},
                "temporal_scaler": st, "build_completo": True,
                "archivos": files, "signal": args.signal,
                "median_events_per_patient": int(np.median(counts))}
        with open(os.path.join(ddir, f"metadata_cnn_{ts}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        for kind, emb, terms in (("diag", vocab_d, terms_d), ("proc", vocab_p, terms_p)):
            np.savez(os.path.join(edir, f"bert_embeddings_{kind}_completos_v32_synthetic.npz"),
                     embeddings=emb, terms=terms)
        with open(os.path.join(mdir, "mapa_cie10.json"), "w", encoding="utf-8") as f:
            json.dump({t: f"synthetic diagnosis {t}" for t in terms_d}, f, indent=1)
        with open(os.path.join(mdir, "mapa_cups.json"), "w", encoding="utf-8") as f:
            json.dump({t: f"synthetic procedure {t}" for t in terms_p}, f, indent=1)
        print("Written the pipeline layout under %s (timestamp %s)." % (base, ts))

    print("Median events per patient: %d" % np.median(counts))
    print("\nReminder: this reproduces the mechanics of the pipeline, not the cohort.")


if __name__ == "__main__":
    main()
