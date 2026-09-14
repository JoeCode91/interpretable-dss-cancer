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
    unverifiable. This generator closes that gap. It emits an HDF5 file with the same
    schema, the same array shapes and the same class ratio as the real cohort, so that
    every downstream script runs end to end.

WHAT IT DOES AND DOES NOT REPRODUCE
    It reproduces the MECHANICS of the pipeline: shapes, dtypes, dataset names, the
    ordering convention inside each event vector, the padding rule, and the class ratio.

    It does NOT reproduce the STATISTICS of the Colombian cohort. Embeddings are drawn
    from a normal distribution, not from the BSC-Bio-EHR-ES encoder; ages and coordinates
    are drawn uniformly over plausible ranges; there is no correlation between events and
    the label beyond a deliberately weak, documented signal.

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
                         +   5 temporal features
                    100  = events per patient, most recent first-padded:
                           patients with more than 100 keep the LAST 100;
                           patients with fewer are zero-padded on the right.
    X_demograficos  (n, 15)         float32
                    10 EPS one-hot + 2 sex one-hot + latitude + longitude + age,
                    the last three standardised.
    y               (n,)            int8, 1 = gastric cancer

USAGE
    python generate_synthetic_cohort.py --out synthetic_cohort.h5 --n 2000

    --n N         number of patients (default 2000)
    --ratio R     negatives per positive (default 100, the undersampled ratio used for
                  training and evaluation in the thesis; pass 5103 for the population ratio)
    --events-max  events per patient (default 100, the V4 value)
    --seed S      random seed (default 0)
    --signal S    how strongly the label shifts the event embeddings (default 0.15).
                  Zero gives pure noise and any model will be at chance; larger values
                  make the task artificially easy. The default leaves it learnable but
                  unimpressive, which is the honest choice for a schema-level surrogate.

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
    This file was drafted with generative AI (Claude, Anthropic) and then reviewed,
    corrected and validated by the author, who takes full responsibility for it. This is
    a stronger statement than the one carried by most files in this repository, which
    were written by the author and only reviewed with AI assistance; it is recorded
    separately so that the distinction is not lost. The research question this script
    answers, and the interpretation of what it returns, are the author's.
"""

import argparse
import json
import sys

import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("h5py is required: pip install h5py")

# --- dimensions, from Preparar_HDF5_V4.py ---
BERT_DIM = 768
TEMPORAL_DIM = 5
EVENTO_DIM = BERT_DIM * 2 + TEMPORAL_DIM        # 1541
EPS_CATS, SEXO_CATS = 10, 2
DEMOGRAFICOS_DIM = EPS_CATS + SEXO_CATS + 2 + 1  # 15

# Colombia, roughly. Only used to keep the geographic features in a plausible range;
# no municipality of the real cohort is reproduced.
LAT_RANGE = (-4.2, 12.5)
LON_RANGE = (-79.0, -66.8)


def build_events(n, max_eventos, y, signal, rng):
    """
    One (EVENTO_DIM, max_eventos) tensor per patient.

    Event counts are drawn from a lognormal, which gives the long right tail that real
    clinical histories have, and are then clipped at max_eventos. Patients below the cap
    are zero-padded on the right, matching the V4 convention.
    """
    X = np.zeros((n, EVENTO_DIM, max_eventos), dtype=np.float32)
    counts = np.clip(rng.lognormal(mean=3.0, sigma=0.9, size=n).astype(int), 1, max_eventos)

    for i in range(n):
        k = counts[i]
        # Two embedding blocks, diagnosis and procedure. Normal, unit scale: the real ones
        # come from a sentence encoder and are not normal, but nothing downstream assumes
        # a distribution.
        emb = rng.normal(0.0, 1.0, size=(BERT_DIM * 2, k)).astype(np.float32)

        # A weak label-dependent shift, so that a model has something to find and the
        # training scripts can be exercised. Deliberately small; see --signal.
        if y[i] == 1 and signal > 0:
            emb[: BERT_DIM // 4, :] += signal

        # Temporal block: days since previous event, cumulative time, and three
        # normalised positional features. Scaled to resemble the real ranges.
        gaps = rng.exponential(45.0, size=k).astype(np.float32)
        temporal = np.vstack([
            gaps,
            np.cumsum(gaps),
            np.linspace(0.0, 1.0, k, dtype=np.float32),
            np.full(k, k / float(max_eventos), dtype=np.float32),
            rng.normal(0.0, 1.0, size=k).astype(np.float32),
        ]).astype(np.float32)

        X[i, :BERT_DIM * 2, :k] = emb
        X[i, BERT_DIM * 2:, :k] = temporal

    return X, counts


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
        stats[name] = {"mu": mu, "sigma": sd}

    return X, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="synthetic_cohort.h5")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--ratio", type=float, default=100.0,
                    help="negatives per positive (100 = undersampled, 5103 = population)")
    ap.add_argument("--events-max", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--signal", type=float, default=0.15)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    n_pos = max(1, int(round(args.n / (args.ratio + 1.0))))
    y = np.zeros(args.n, dtype=np.int8)
    y[rng.choice(args.n, n_pos, replace=False)] = 1

    print("Generating %d patients, %d positive (1:%.0f)" % (args.n, n_pos, args.ratio))
    print("  tensor per patient: (%d, %d)" % (EVENTO_DIM, args.events_max))

    X_events, counts = build_events(args.n, args.events_max, y, args.signal, rng)
    X_demo, stats = build_demographics(args.n, y, rng)

    with h5py.File(args.out, "w") as f:
        f.create_dataset("X_events", data=X_events, compression="gzip", compression_opts=4,
                         chunks=(1, EVENTO_DIM, args.events_max))
        f.create_dataset("X_demograficos", data=X_demo, compression="gzip")
        f.create_dataset("y", data=y)
        f.attrs["synthetic"] = True
        f.attrs["warning"] = ("Synthetic data. Reproduces the schema of the thesis cohort, "
                              "not its statistics. Results obtained on it are not comparable "
                              "with those reported in the thesis.")
        f.attrs["generator_seed"] = args.seed
        f.attrs["class_ratio"] = args.ratio

    with open(args.out.replace(".h5", "_metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"synthetic": True, "n": args.n, "n_positive": int(n_pos),
                   "ratio": args.ratio, "seed": args.seed, "signal": args.signal,
                   "evento_dim": EVENTO_DIM, "max_eventos": args.events_max,
                   "demograficos_dim": DEMOGRAFICOS_DIM,
                   "scaler": stats,
                   "median_events_per_patient": int(np.median(counts))},
                  f, indent=2)

    mb = (X_events.nbytes + X_demo.nbytes) / 1e6
    print("Written %s (%.0f MB uncompressed) and its metadata." % (args.out, mb))
    print("Median events per patient: %d" % np.median(counts))
    print("\nReminder: this reproduces the mechanics of the pipeline, not the cohort.")


if __name__ == "__main__":
    main()
