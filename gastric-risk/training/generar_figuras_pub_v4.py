"""
generar_figuras_pub_v4.py -- Publication-quality NObSP V4 figures for Cap. 4 + Alex presentation.

REPRODUCES IN THE THESIS
    Regenerates the publication figures of Chapter 4 from saved results.

PROVENANCE
    Copied verbatim from the working tree of the thesis. The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md at the
    repository root.

--------------------------------------------------------------------------

Reads existing NObSP results (CSV + NPZ) and generates 6 improved figures.
Output: nobsp_results_v4/pub_*.{png,pdf}

Date: 2026-05-25 (sesion 34)

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

import csv
import glob
import json
import os
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---- Paths ----
OUTPUT_DIR = Path(r"G:\Mi unidad\Doctorado\Proyecto_CC\Test\Book\XAI_Model\nobsp_results_v4")
V4_DATA_DIR = Path(r"D:\Proyecto_CC_Batches\Datos_Modelo_V4")
N_SAMPLES = 500
# Modo R2 (train+test): si NOBSP_R2_DIR esta seteado, lee de ese dir y carga el
# conjunto de calibracion balanceado en vez de las 500 de test.
_R2_DIR = os.environ.get('NOBSP_R2_DIR', '').strip()
if _R2_DIR:
    OUTPUT_DIR = Path(_R2_DIR)
    print(f"[R2] figuras desde {OUTPUT_DIR}")

# ---- Scalers (from metadata_v4_20260518_165534.json) ----
SCALERS = {
    'lat':  {'mean': 5.6452, 'scale': 2.2558},
    'lon':  {'mean': -74.9218, 'scale': 1.0864},
    'edad': {'mean': 52.8734, 'scale': 14.9610},
}
DEMO_CONTINUOUS = {'lat': 12, 'lon': 13, 'edad': 14}

# ---- Feature type colors ----
C_EPS = '#2166ac'
C_SEX = '#4dac26'
C_GEO = '#e08214'
C_AGE = '#b2182b'
FCOLORS = {
    **{f"EPS_{i}": C_EPS for i in range(10)},
    "Sexo_F": C_SEX, "Sexo_M": C_SEX,
    "lat": C_GEO, "lon": C_GEO,
    "edad": C_AGE,
}

# ---- Style ----
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 17,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})


def inverse_scaler(z, key):
    s = SCALERS[key]
    return z * s['scale'] + s['mean']


def load_importance_csv(path):
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row['feature']] = float(row['importance'])
    return out


def save_fig(fig, stem):
    for ext in ['png', 'pdf']:
        fig.savefig(OUTPUT_DIR / f"{stem}.{ext}")
    plt.close(fig)


# ==================================================================
# Figure 1: W1 Demographic Feature Importance (normalized bar chart)
# ==================================================================
def fig1_demo_bar():
    imp = load_importance_csv(OUTPUT_DIR / "demo_importance.csv")
    total = sum(imp.values())

    items = sorted(imp.items(), key=lambda x: x[1])
    features = [f for f, _ in items]
    values = [v / total for _, v in items]
    colors = [FCOLORS[f] for f in features]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    y = np.arange(len(features))
    ax.barh(y, values, color=colors, edgecolor='white', linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=13)
    ax.set_xlabel("Normalized NObSP importance")
    ax.set_title("CNN-2D model: demographic feature importance (W1)")

    for i, v in enumerate(values):
        if v >= 0.015:
            ax.text(v + 0.004, i, f"{v:.1%}", va='center', fontsize=12)

    legend_elements = [
        Patch(facecolor=C_EPS, label='Health insurer (EPS)'),
        Patch(facecolor=C_SEX, label='Sex'),
        Patch(facecolor=C_GEO, label='Geographic (DIVIPOLA)'),
        Patch(facecolor=C_AGE, label='Age'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
    ax.set_xlim(0, max(values) * 1.18)
    ax.grid(alpha=0.2, axis='x', linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    save_fig(fig, "pub_w1_demo_importance_bar")
    print("  [1] pub_w1_demo_importance_bar")


# ==================================================================
# Figure 2: W1 Feature curves for continuous variables (lat, lon, edad)
# ==================================================================
def fig2_continuous_curves():
    data = np.load(OUTPUT_DIR / "demo_explainer.npz", allow_pickle=True)
    contributions = data['contributions']

    if _R2_DIR:
        import r2_calib
        _, X_demo, y_labels = r2_calib.load_r2_calib(_R2_DIR, str(V4_DATA_DIR), need_events=False)
        X_demo = X_demo.astype(np.float32); y_labels = y_labels.astype(int)
    else:
        test_files = sorted(glob.glob(str(V4_DATA_DIR / "datos_modelo_test_v4_*.h5")))
        if not test_files:
            print("  [2] SKIP -- HDF5 test file not found")
            return
        with h5py.File(test_files[-1], 'r') as f:
            X_demo = f['X_demograficos'][:N_SAMPLES].astype(np.float32)
            y_labels = f['y'][:N_SAMPLES].astype(int)

    X_real = X_demo.copy()
    for key, col in DEMO_CONTINUOUS.items():
        X_real[:, col] = inverse_scaler(X_demo[:, col], key)

    vars_info = [
        (12, 'lat', "Latitude (degrees N)", "(a) Geographic latitude"),
        (13, 'lon', "Longitude (degrees W)", "(b) Geographic longitude"),
        (14, 'edad', "Age (years)", "(c) Patient age"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (col, key, xlabel, title) in zip(axes, vars_info):
        x = X_real[:, col]
        c = contributions[:, col, 1]

        order = np.argsort(x)
        x_s, c_s = x[order], c[order]
        pos_mask = y_labels[order] == 1

        ax.scatter(x_s[~pos_mask], c_s[~pos_mask], alpha=0.15, s=10,
                   color='#92c5de', label='No cancer', zorder=2, rasterized=True)
        ax.scatter(x_s[pos_mask], c_s[pos_mask], alpha=0.9, s=30,
                   color='#ca0020', marker='x', linewidths=1.2,
                   label='Cancer', zorder=3)

        w = max(len(x_s) // 8, 30)
        c_sm = np.convolve(c_s, np.ones(w) / w, mode='valid')
        x_sm = x_s[(w - 1) // 2:(w - 1) // 2 + len(c_sm)]
        ax.plot(x_sm, c_sm, color='black', linewidth=2.2, zorder=4, label='Trend')

        ax.axhline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel("NObSP contribution\n(Cancer class)", fontsize=18)
        ax.set_title(title, fontsize=19)
        ax.tick_params(labelsize=15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=15, loc='best', framealpha=0.85)

    fig.suptitle("CNN-2D model: NObSP feature curves for continuous variables (W1)",
                 fontsize=21, y=1.02)
    fig.tight_layout()
    save_fig(fig, "pub_w1_continuous_curves")
    print("  [2] pub_w1_continuous_curves")


# ==================================================================
# Figure 3: Comparison BiLSTM-Attention vs CNN-2D
# ==================================================================
def fig3_comparison():
    rows = []
    with open(OUTPUT_DIR / "compare_demo_v4_vs_prior.csv") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    shared = [r for r in rows if r['imp_v32_bilstm']]
    v4_only = [r for r in rows if not r['imp_v32_bilstm']]

    total_bilstm = sum(float(r['imp_v32_bilstm']) for r in shared)
    total_cnn2d = sum(float(r['imp_v4']) for r in rows)

    all_features, bilstm_vals, cnn2d_vals = [], [], []
    for r in shared:
        all_features.append(r['feature'])
        bilstm_vals.append(float(r['imp_v32_bilstm']) / total_bilstm)
        cnn2d_vals.append(float(r['imp_v4']) / total_cnn2d)
    for r in v4_only:
        all_features.append(r['feature'])
        bilstm_vals.append(0.0)
        cnn2d_vals.append(float(r['imp_v4']) / total_cnn2d)

    order = np.argsort(cnn2d_vals)
    all_features = [all_features[i] for i in order]
    bilstm_vals = [bilstm_vals[i] for i in order]
    cnn2d_vals = [cnn2d_vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    y = np.arange(len(all_features))
    h = 0.35
    ax.barh(y - h / 2, bilstm_vals, h, label='BiLSTM-Attention (12 features)',
            color='#91bfdb', edgecolor='white', linewidth=0.5)
    ax.barh(y + h / 2, cnn2d_vals, h, label='CNN-2D (15 features)',
            color='#fc8d59', edgecolor='white', linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(all_features, fontsize=10)
    ax.set_xlabel("Normalized NObSP importance (each model sums to 1)")
    ax.set_title("NObSP demographic importance: BiLSTM-Attention vs CNN-2D")
    ax.legend(loc='lower right', framealpha=0.9)

    n_only = len(v4_only)
    if n_only > 0:
        sep_y = len(all_features) - n_only - 0.5
        ax.axhline(sep_y, color='gray', linestyle=':', linewidth=0.8)
        ax.annotate("CNN-2D only\n(new continuous features)",
                     xy=(max(cnn2d_vals) * 0.55, sep_y + n_only / 2 + 0.05),
                     fontsize=9, fontstyle='italic', ha='center', va='center',
                     color='#555555')

    ax.grid(alpha=0.2, axis='x', linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, max(max(bilstm_vals), max(cnn2d_vals)) * 1.12)

    fig.tight_layout()
    save_fig(fig, "pub_comparison_bilstm_vs_cnn2d")
    print("  [3] pub_comparison_bilstm_vs_cnn2d")


# ==================================================================
# Figure 4: W2 Top-20 Event Position Importance (bar chart)
# ==================================================================
def fig4_event_bar():
    imp = load_importance_csv(OUTPUT_DIR / "input_event_importance.csv")
    total = sum(imp.values())

    top20 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:20]
    features = [f for f, _ in top20][::-1]
    values = [v / total for _, v in top20][::-1]

    def event_color(name):
        idx = int(name[1:])
        if idx < 30:
            return '#abd9e9'
        elif idx < 70:
            return '#d7191c'
        else:
            return '#fdae61'

    colors = [event_color(f) for f in features]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    y = np.arange(len(features))
    ax.barh(y, values, color=colors, edgecolor='white', linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=13, family='monospace')
    ax.set_xlabel("Normalized NObSP importance")
    ax.set_title("CNN-2D model: top-20 event position importance (W2)")

    legend_elements = [
        Patch(facecolor='#abd9e9', label='Early (e000--e029)'),
        Patch(facecolor='#d7191c', label='Central (e030--e069)'),
        Patch(facecolor='#fdae61', label='Late (e070--e099)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)

    ax.grid(alpha=0.2, axis='x', linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    save_fig(fig, "pub_w2_event_importance_bar")
    print("  [4] pub_w2_event_importance_bar")


# ==================================================================
# Figure 5: W2 Temporal Contribution Heatmap (improved)
# ==================================================================
def fig5_temporal_heatmap():
    data = np.load(OUTPUT_DIR / "input_event_explainer.npz", allow_pickle=True)
    contributions = data['contributions']

    contrib = contributions[:, :, 1]
    contrib = contrib - contrib.mean(axis=0, keepdims=True)

    n_total = contrib.shape[0]
    n_show = 80
    intensity = np.abs(contrib).sum(axis=1)
    show_idx = np.argsort(intensity)[::-1][:n_show]
    contrib_show = contrib[show_idx]
    cmax = float(np.percentile(np.abs(contrib_show), 99))
    print(f"  [5] W2 heatmap: contrib shape={contrib.shape} -> showing top {n_show} of {n_total} probe samples")

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(contrib_show, aspect='auto', cmap='RdBu_r',
                   vmin=-cmax, vmax=cmax, interpolation='nearest')
    ax.set_xlabel("Event position (0 = oldest kept event, 99 = most recent)", fontsize=19)
    ax.set_ylabel(f"Probe samples\n(top {n_show} of {n_total}, by total $|$contribution$|$)", fontsize=19)
    ax.set_title("CNN-2D: per-event NObSP contribution to the cancer class (W2)", fontsize=19)
    ax.set_xticks(np.arange(0, 100, 10))
    ax.tick_params(axis='both', labelsize=16)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("Centered contribution to cancer class", fontsize=18)
    cb.ax.tick_params(labelsize=15)

    fig.tight_layout()
    save_fig(fig, "pub_w2_temporal_heatmap")
    print("  [5] pub_w2_temporal_heatmap")


# ==================================================================
# Figure 6: W3 Temporal Bucket Importance (improved bar)
# ==================================================================
def fig6_bucket_bar():
    imp = load_importance_csv(OUTPUT_DIR / "pre_pool_importance.csv")
    total = sum(imp.values())

    n_buckets = 12
    events_per_bucket = 100 // n_buckets

    temporal_features = [f"tb{i:02d}" for i in range(n_buckets)]
    values = [imp.get(f, 0.0) / total for f in temporal_features]

    labels = []
    for i, f in enumerate(temporal_features):
        e_start = i * events_per_bucket
        e_end = min((i + 1) * events_per_bucket - 1, 99)
        labels.append(f"{f}  (e{e_start:03d}--e{e_end:03d})")

    cmap = plt.cm.YlOrRd
    colors = [cmap(0.15 + 0.75 * i / (n_buckets - 1)) for i in range(n_buckets)]

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(n_buckets)
    ax.barh(y, values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12, family='monospace')
    ax.invert_yaxis()
    ax.set_xlabel("Normalized NObSP importance")
    ax.set_title("CNN-2D model: temporal bucket importance (W3, pre-pool)")

    for i, v in enumerate(values):
        if v >= 0.025:
            ax.text(v + 0.003, i, f"{v:.1%}", va='center', fontsize=12)

    ax.grid(alpha=0.2, axis='x', linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    save_fig(fig, "pub_w3_bucket_importance_bar")
    print("  [6] pub_w3_bucket_importance_bar")


# ==================================================================
if __name__ == "__main__":
    print("Generating publication-quality NObSP V4 figures...")
    print(f"Output: {OUTPUT_DIR}\n")
    fig1_demo_bar()
    fig2_continuous_curves()
    fig3_comparison()
    fig4_event_bar()
    fig5_temporal_heatmap()
    fig6_bucket_bar()
    print("\nDone! 6 figures x 2 formats (PNG 300dpi + PDF vector).")
