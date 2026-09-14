"""
NObSP V4 (Camino 2 -- arquitectura CNN 2D Alex 18 May 2026) -- pipeline en 4 puntos.

REPRODUCES IN THE THESIS
    The four NObSP wrappers applied to the CNN-2D model: W1 demographics, W2 temporal
    positions, W3 event groups, W4 learned channels. Section 4.9. These wrappers are the
    adaptation of the method to this architecture, and are thesis work; the NObSP
    library itself is a third-party dependency, see the repository NOTICE.

PROVENANCE
    Copied verbatim from the working tree of the thesis. The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md at the
    repository root.

--------------------------------------------------------------------------

Cuatro wrappers aplicados al modelo CNN2DModelV4 entrenado sobre HDF5 V4
(tensor (1541, 100) por paciente, sin overlap, sin masking, demograficos 15 dims).

  W1 Demograficos          (15 features: EPS_0..9 + Sx_F,M + lat + lon + edad)
  W2 Input por evento      (100 slots = columnas del tensor V4) -- COSTOSO
  W3 Post-conv pre-pool    (12 buckets temporales = agregacion del feat-map W=12)
  W4 Post-pool             (64 dims abstractas del vector pooled)

Mejoras de plotting respecto a Camino 1:

  (a) Top-N filtering para wrappers con muchas features -- antes los curves de
      W2/W3 se saltaban; ahora plot del top-N por importancia.
  (b) Inverse StandardScaler para lat/lon/edad en W1 -- antes los curves del
      eje X estaban en z-scores (sin interpretacion clinica); ahora se muestran
      en grados / anios reales usando mu, sigma serializados en metadata_v4.
  (c) Heatmap temporal (n_samples x 100) para W2 -- complementa los curves
      mostrando que eventos cronologicos contribuyen positivamente / negativamente
      en muestras individuales.

Costos esperados (RTX 3060 Ti, N_SAMPLES=500):
  W1 (demo 15)          ~ pocos segundos
  W2 (input 100)        ~ minutos      (conv corre 500 veces por feature)
  W3 (temporal 12)      ~ segundos     (solo pool + FC; mucho mas barato que en Camino 1 con 400 slots)
  W4 (post-pool 64)     ~ trivial

Autor: Johan S. Pena-Campos, Cap. 4 V4 Camino 2
Fecha: 2026-05-18 (sesion 29)

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
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from modelo_cnn2d_v4_pytorch import CNN2DModelV4
from nobsp_lib import (
    TabularExplainer,
    plot_importance_heatmap,
    plot_tabular_feature_curves,
)

try:
    from telegram_notifier import MPCNotifier
except Exception:
    class MPCNotifier:
        def __init__(self, **kw): self.enabled = False
        def notify_start(self, *a, **kw): return False
        def notify_progress(self, *a, **kw): return False
        def notify_warning(self, *a, **kw): return False
        def notify_error(self, *a, **kw): return False
        def notify_complete(self, *a, **kw): return False


# ==================== CONFIGURACION ====================

MODEL_DIR = Path(r"D:\Proyecto_CC_Batches\Modelos_CNN2D_V4_PyTorch")
V4_DATA_DIR = Path(r"D:\Proyecto_CC_Batches\Datos_Modelo_V4")
OUTPUT_DIR  = Path(r"G:\Mi unidad\Doctorado\Proyecto_CC\Test\Book\XAI_Model\nobsp_results_v4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Comparativas con versiones previas si existen
CMP_CSV = {
    'v32_demo':  Path(r"G:\Mi unidad\Doctorado\Proyecto_CC\Test\Book\XAI_Model\nobsp_results_v32\demo_importance.csv"),
    'cnn2d_demo': Path(r"G:\Mi unidad\Doctorado\Proyecto_CC\Test\Book\XAI_Model\nobsp_results_cnn2d\demo_importance.csv"),
}

N_SAMPLES = 500
TOP_N_W2 = 20   # top features para curves de input por evento
TOP_N_W4 = 16   # top features para curves de post-pool
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Nombres de features V4
FEATURE_NAMES_DEMO = (
    [f"EPS_{i}" for i in range(10)]
    + ["Sexo_F", "Sexo_M"]
    + ["lat", "lon", "edad"]
)   # 15 dims
EVENT_NAMES = [f"e{i:03d}" for i in range(100)]                # 100 eventos
TEMPORAL_BUCKET_NAMES = [f"tb{i:02d}" for i in range(12)]      # 12 buckets temporales W3
POOL_NAMES = [f"pool_d{i:02d}" for i in range(64)]             # 64 dims abstractas W4

# Indices de columnas continuas en X_demograficos (StandardScaler aplicado)
DEMO_CONTINUOUS_COLS = {'lat': 12, 'lon': 13, 'edad': 14}


# ==================== WRAPPERS V4 ====================

class V4DemoWrapper(nn.Module):
    """W1 -- Demograficos (15 features reales). Pre-computa pool del referente."""
    def __init__(self, full_model: CNN2DModelV4, pooled_ref: torch.Tensor):
        super().__init__()
        self.demo_dense = full_model.demo_dense
        self.demo_dropout = full_model.demo_dropout
        self.dense1 = full_model.dense1
        self.dropout1 = full_model.dropout1
        self.output_layer = full_model.output_layer
        self.register_buffer('pooled_ref', pooled_ref.unsqueeze(0))    # (1, c_pool)

    def forward(self, x_demograficos: torch.Tensor) -> torch.Tensor:
        B = x_demograficos.shape[0]
        pooled = self.pooled_ref.expand(B, -1)
        demo = F.relu(self.demo_dense(x_demograficos))
        demo = self.demo_dropout(demo)
        fused = torch.cat([pooled, demo], dim=-1)
        h = F.relu(self.dense1(fused)); h = self.dropout1(h)
        return torch.sigmoid(self.output_layer(h))


class V4InputEventWrapper(nn.Module):
    """W2 -- Por evento puro (100 slots = columnas del tensor V4).

    Pesos perturbativos en [0.5, 1.5]^100 escalan cada columna del input
    ANTES del primer conv. Re-corre toda la CNN en cada forward (costoso).
    """
    def __init__(self, full_model: CNN2DModelV4,
                 x_events_ref: torch.Tensor, x_demo_ref: torch.Tensor):
        super().__init__()
        self.model = full_model
        # x_events_ref shape: (1541, 100); reshape a (1, 1541, 100)
        self.register_buffer('x_events_ref', x_events_ref.unsqueeze(0))
        self.register_buffer('x_demo_ref',   x_demo_ref.unsqueeze(0))   # (1, 15)

    def forward(self, event_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            event_weights: (B, 100) en [0.5, 1.5]
        Returns:
            (B, 1) probabilidad
        """
        B = event_weights.shape[0]
        # Pesos como (B, 1, 100) para broadcast con (B, 1541, 100)
        w = event_weights.view(B, 1, 100)
        x_events = self.x_events_ref.expand(B, -1, -1) * w              # (B, 1541, 100)
        demo = self.x_demo_ref.expand(B, -1)                            # (B, 15)
        return self.model(x_events, demo)


class V4PreToolWrapper(nn.Module):
    """W3 -- Post-conv pre-pool agregado por bucket temporal.

    El feature map post-conv4 tiene shape (c_pool=64, H'=192, W'=12). En vez
    de perturbar las 192*12=2304 posiciones individuales (intratable), agregamos
    por columna temporal: 12 buckets que aproximadamente corresponden a
    grupos de ~8 eventos cronologicos.

    Pesos en [0.5, 1.5]^12 escalan cada columna del feat map. Solo el pool
    global + cabeza FC corren en cada forward (rapido).
    """
    def __init__(self, full_model: CNN2DModelV4,
                 feat_map_ref: torch.Tensor, x_demo_ref: torch.Tensor):
        super().__init__()
        self.model = full_model
        # feat_map_ref shape: (c_pool, H', W'); reshape a (1, c_pool, H', W')
        self.register_buffer('feat_map_ref', feat_map_ref.unsqueeze(0))
        self.register_buffer('x_demo_ref',   x_demo_ref.unsqueeze(0))   # (1, 15)
        self.W = feat_map_ref.shape[-1]                                 # 12

    def forward(self, bucket_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bucket_weights: (B, 12) en [0.5, 1.5]
        Returns:
            (B, 1) probabilidad
        """
        B = bucket_weights.shape[0]
        # Pesos como (B, 1, 1, 12) para broadcast con (B, c_pool, H', W')
        w = bucket_weights.view(B, 1, 1, self.W)
        feat = self.feat_map_ref.expand(B, -1, -1, -1) * w
        pooled = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1) # (B, c_pool)
        return self.model.forward_head(pooled, self.x_demo_ref.expand(B, -1))


class V4PostPoolWrapper(nn.Module):
    """W4 -- Post-pool (64 dims abstractas).

    Pesos en [0.5, 1.5]^64 escalan cada dimension del vector pooled.
    Solo cabeza FC corre.
    """
    def __init__(self, full_model: CNN2DModelV4,
                 pooled_ref: torch.Tensor, x_demo_ref: torch.Tensor):
        super().__init__()
        self.model = full_model
        self.register_buffer('pooled_ref', pooled_ref.unsqueeze(0))     # (1, c_pool)
        self.register_buffer('x_demo_ref', x_demo_ref.unsqueeze(0))     # (1, 15)

    def forward(self, pool_weights: torch.Tensor) -> torch.Tensor:
        B = pool_weights.shape[0]
        pooled = self.pooled_ref.expand(B, -1) * pool_weights
        return self.model.forward_head(pooled, self.x_demo_ref.expand(B, -1))


# ==================== HELPERS ====================

def find_latest_v4_checkpoint(model_dir: Path) -> Path:
    candidates = sorted(model_dir.glob("modelo_cnn2d_v4_pytorch_*_best.pt"))
    if not candidates:
        raise FileNotFoundError(f"No se encontro checkpoint V4 en {model_dir}")
    return candidates[-1]


def find_latest_v4_metadata(data_dir: Path) -> Path:
    candidates = sorted(data_dir.glob("metadata_v4_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No se encontro metadata_v4 en {data_dir}")
    return candidates[-1]


def find_latest_v4_test(data_dir: Path) -> Path:
    candidates = sorted(data_dir.glob("datos_modelo_test_v4_*.h5"))
    if not candidates:
        raise FileNotFoundError(f"No se encontro test HDF5 V4 en {data_dir}")
    return candidates[-1]


def load_scalers_from_metadata(meta_path: Path) -> dict:
    """Reconstruye StandardScalers de lat/lon/edad desde metadata_v4.json."""
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    scalers = {}
    for key in ['lat', 'lon', 'edad']:
        s = StandardScaler()
        params = meta['scalers'][key]
        s.mean_  = np.array([params['mean']],  dtype=np.float64)
        s.scale_ = np.array([params['scale']], dtype=np.float64)
        s.var_   = np.array([params['var']],   dtype=np.float64)
        s.n_samples_seen_ = int(params['n_samples_seen'])
        s.n_features_in_ = 1
        scalers[key] = s
    return scalers, meta


def denormalize_demo_continuous(X_demo: np.ndarray, scalers: dict) -> np.ndarray:
    """Invierte StandardScaler de lat/lon/edad para mostrar valores reales en plots."""
    X_real = X_demo.copy()
    for key, col_idx in DEMO_CONTINUOUS_COLS.items():
        X_real[:, col_idx] = scalers[key].inverse_transform(
            X_demo[:, col_idx].reshape(-1, 1)
        ).ravel()
    return X_real


def plot_top_n_feature_curves(X, contributions, mean_imp, feature_names,
                              top_n, save_path):
    """Filtra top-N por importancia antes de plot_tabular_feature_curves."""
    top_idx = np.argsort(mean_imp)[::-1][:top_n]
    plot_tabular_feature_curves(
        X[:, top_idx],
        contributions[:, top_idx, :],
        feature_names=[feature_names[i] for i in top_idx],
        max_features=top_n,
        save_path=save_path,
    )
    return top_idx


def plot_temporal_contribution_heatmap(contributions_w2, save_path,
                                       output_index=1, n_show=80):
    """Heatmap (n_samples x 100) de contribuciones por evento para la clase target.

    contributions_w2 shape: (n_samples, 100, n_outputs).
    output_index=1 -> clase Cancer.
    Solo muestra n_show muestras (ordenadas por suma absoluta de contrib).
    """
    contrib = contributions_w2[:, :, output_index]                       # (n, 100)
    # Centra contribuciones (NObSP devuelve contribuciones relativas)
    contrib = contrib - contrib.mean(axis=0, keepdims=True)
    # Ordena por intensidad
    intensity = np.abs(contrib).sum(axis=1)
    show_idx = np.argsort(intensity)[::-1][:n_show]
    contrib_show = contrib[show_idx]
    cmax = float(np.percentile(np.abs(contrib_show), 99))
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(contrib_show, aspect='auto', cmap='RdBu_r',
                   vmin=-cmax, vmax=cmax, interpolation='nearest')
    ax.set_xlabel("Evento (indice cronologico, 0=mas antiguo, 99=mas reciente)")
    ax.set_ylabel(f"Muestra (top {n_show} por intensidad)")
    ax.set_title(f"NObSP W2: contribucion por evento (clase Cancer, V4)")
    fig.colorbar(im, ax=ax, label="contribucion centrada", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_bucket_bar(mean_imp, names, save_path, title):
    """Bar plot horizontal de importancia por bucket (W3)."""
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(names) + 1))
    y = np.arange(len(names))
    ax.barh(y, mean_imp, color='#1f77b4')
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Importancia NObSP (media)")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis='x', linestyle=':')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def load_csv_importance(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row['feature']] = float(row['importance'])
    return out


def run_nobsp(name, wrapper, X_input, feature_names, output_dir, notifier,
              method='alpha', reg=1e-4, top_n_curves=None,
              x_for_plot=None, extra_plot_fn=None):
    """Ejecuta NObSP, guarda plots + CSV + explainer.

    Args:
        x_for_plot: si se da, se usa este array (de-normalizado) para el eje X
                    de los curves en vez de X_input. Util para W1 demograficos
                    con StandardScaler invertido.
        top_n_curves: si es None, plot all (max_features=len). Si entero, plot
                      solo top-N features.
        extra_plot_fn(contributions, save_dir, name): callback opcional para
                      plots adicionales (e.g. heatmap temporal en W2).
    """
    t0 = time.time()
    print(f"\n{'='*64}\n  NObSP V4 -- {name}\n{'='*64}", flush=True)
    print(f"  features: {len(feature_names)}, samples: {len(X_input)}", flush=True)

    explainer = TabularExplainer(
        task="classification", method=method, regularization=reg,
        device=str(DEVICE),
    )
    explainer.fit(
        wrapper, X_input,
        feature_names=feature_names,
        class_names=["No Cancer", "Cancer"],
        output_kind="probabilities",
    )

    contributions = explainer.explain()                                 # (n, d, c)
    importance = explainer.feature_importance()                         # (d, c)
    mean_imp = importance.mean(axis=1) if importance.ndim > 1 else importance
    sorted_idx = np.argsort(mean_imp)[::-1]

    print(f"  Top-10:", flush=True)
    for r, idx in enumerate(sorted_idx[:10]):
        print(f"    {r+1:2d}. {feature_names[idx]:<18s} = {mean_imp[idx]:.4f}", flush=True)

    plot_importance_heatmap(
        importance, feature_names=feature_names,
        output_names=["No Cancer", "Cancer"],
        save_path=output_dir / f"{name}_importance_heatmap.png",
    )

    X_plot = x_for_plot if x_for_plot is not None else X_input
    if top_n_curves is None:
        plot_tabular_feature_curves(
            X_plot, contributions, feature_names=feature_names,
            max_features=len(feature_names),
            save_path=output_dir / f"{name}_feature_curves.png",
        )
    else:
        plot_top_n_feature_curves(
            X_plot, contributions, mean_imp, feature_names,
            top_n=min(top_n_curves, len(feature_names)),
            save_path=output_dir / f"{name}_feature_curves_top{top_n_curves}.png",
        )

    if extra_plot_fn is not None:
        extra_plot_fn(contributions, output_dir, name)

    explainer.save(output_dir / f"{name}_explainer.npz")

    with open(output_dir / f"{name}_importance.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["rank", "feature", "importance"])
        for r, idx in enumerate(sorted_idx):
            w.writerow([r+1, feature_names[idx], f"{mean_imp[idx]:.6f}"])

    elapsed = time.time() - t0
    print(f"  -> {name} listo en {elapsed:.1f}s", flush=True)
    notifier.notify_warning(f"NObSP V4 {name} listo en {elapsed:.1f}s")
    return mean_imp, sorted_idx


# ==================== MAIN ====================

def main():
    notifier = MPCNotifier(enabled=True)

    # Encontrar artefactos
    model_pt = find_latest_v4_checkpoint(MODEL_DIR)
    meta_pt  = find_latest_v4_metadata(V4_DATA_DIR)
    test_pt  = find_latest_v4_test(V4_DATA_DIR)
    print(f"Checkpoint: {model_pt}", flush=True)
    print(f"Metadata:   {meta_pt}", flush=True)
    print(f"Test data:  {test_pt}", flush=True)

    notifier.notify_start("NObSP V4 (4 puntos)", {
        'model': model_pt.name,
        'data': test_pt.name,
        'n_samples': N_SAMPLES,
        'device': str(DEVICE),
        'output_dir': str(OUTPUT_DIR),
    })
    t_global = time.time()

    # --- 1. Datos + scalers ---
    print("\n[1/8] Cargando datos test V4 y scalers de metadata...", flush=True)
    scalers, meta = load_scalers_from_metadata(meta_pt)
    print(f"  scalers: lat mu={scalers['lat'].mean_[0]:.4f} sigma={scalers['lat'].scale_[0]:.4f}")
    print(f"           lon mu={scalers['lon'].mean_[0]:.4f} sigma={scalers['lon'].scale_[0]:.4f}")
    print(f"           edad mu={scalers['edad'].mean_[0]:.4f} sigma={scalers['edad'].scale_[0]:.4f}")

    with h5py.File(test_pt, 'r') as f:
        X_events = f['X_events'][:N_SAMPLES].astype(np.float32)         # (N, 1541, 100)
        X_demo   = f['X_demograficos'][:N_SAMPLES].astype(np.float32)   # (N, 15)
        y_all    = f['y'][:N_SAMPLES]
    print(f"  n={N_SAMPLES}, pos={int(y_all.sum())}, neg={N_SAMPLES - int(y_all.sum())}", flush=True)

    # De-normalizar demograficos para plots (lat/lon/edad)
    X_demo_real = denormalize_demo_continuous(X_demo, scalers)
    print(f"  X_demo real ranges: "
          f"lat=[{X_demo_real[:,12].min():.2f}, {X_demo_real[:,12].max():.2f}] "
          f"lon=[{X_demo_real[:,13].min():.2f}, {X_demo_real[:,13].max():.2f}] "
          f"edad=[{X_demo_real[:,14].min():.1f}, {X_demo_real[:,14].max():.1f}]", flush=True)

    # Referencias en GPU
    x_events_ref = torch.from_numpy(X_events.mean(axis=0)).to(DEVICE)   # (1541, 100)
    x_demo_ref   = torch.from_numpy(X_demo.mean(axis=0)).to(DEVICE)     # (15,)

    # --- 2. Modelo ---
    print("\n[2/8] Cargando modelo V4...", flush=True)
    model = CNN2DModelV4(n_features=1541, n_eventos=100, n_demograficos=15).to(DEVICE)
    model.load_state_dict(torch.load(model_pt, map_location=DEVICE, weights_only=True))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  pesos cargados ({n_params:,} params)", flush=True)

    # --- 3. Pre-computos referencia ---
    print("\n[3/8] Pre-computando feature map y pooled del referente...", flush=True)
    with torch.no_grad():
        feat_map_ref = model.forward_conv(x_events_ref.unsqueeze(0)).squeeze(0)   # (64, H', W')
        pooled_ref   = F.adaptive_avg_pool2d(
            feat_map_ref.unsqueeze(0), 1
        ).squeeze(-1).squeeze(-1).squeeze(0)                                       # (64,)
    print(f"  feat_map_ref: {tuple(feat_map_ref.shape)} norm={feat_map_ref.norm():.2f}", flush=True)
    print(f"  pooled_ref:   {tuple(pooled_ref.shape)} norm={pooled_ref.norm():.2f}", flush=True)
    H_pool, W_pool = feat_map_ref.shape[-2], feat_map_ref.shape[-1]
    # Actualizar nombres W3 si W_pool != 12
    global TEMPORAL_BUCKET_NAMES
    if W_pool != len(TEMPORAL_BUCKET_NAMES):
        TEMPORAL_BUCKET_NAMES = [f"tb{i:02d}" for i in range(W_pool)]
        print(f"  W3 buckets ajustados a W_pool={W_pool}", flush=True)

    rng = np.random.RandomState(42)

    # --- 4. W1 Demograficos (15 features, full curves + heatmap, inverse scaler) ---
    wrapper_demo = V4DemoWrapper(model, pooled_ref).to(DEVICE).eval()
    imp_demo, idx_demo = run_nobsp(
        "demo", wrapper_demo, X_demo, FEATURE_NAMES_DEMO, OUTPUT_DIR, notifier,
        top_n_curves=None,            # plot all 15
        x_for_plot=X_demo_real,       # eje X en grados/anios reales
    )

    # --- 5. W2 Input por evento (100 slots, top-N curves + heatmap temporal) ---
    wrapper_input = V4InputEventWrapper(model, x_events_ref, x_demo_ref).to(DEVICE).eval()
    X_input_weights = (1.0 + rng.uniform(-0.5, 0.5, size=(N_SAMPLES, 100))).astype(np.float32)
    imp_input, idx_input = run_nobsp(
        "input_event", wrapper_input, X_input_weights, EVENT_NAMES,
        OUTPUT_DIR, notifier,
        top_n_curves=TOP_N_W2,
        extra_plot_fn=lambda c, d, n: plot_temporal_contribution_heatmap(
            c, d / f"{n}_temporal_heatmap.png"
        ),
    )

    # --- 6. W3 Post-conv pre-pool (W_pool buckets, full curves + bar) ---
    wrapper_pre = V4PreToolWrapper(model, feat_map_ref, x_demo_ref).to(DEVICE).eval()
    X_pre_weights = (1.0 + rng.uniform(-0.5, 0.5, size=(N_SAMPLES, W_pool))).astype(np.float32)
    imp_pre, idx_pre = run_nobsp(
        "pre_pool", wrapper_pre, X_pre_weights, TEMPORAL_BUCKET_NAMES,
        OUTPUT_DIR, notifier,
        top_n_curves=None,            # plot all W_pool
        extra_plot_fn=lambda c, d, n: plot_bucket_bar(
            np.abs(c).mean(axis=(0,2)), TEMPORAL_BUCKET_NAMES,
            d / f"{n}_bucket_bar.png",
            f"NObSP W3 V4: importancia por bucket temporal (W_pool={W_pool})"
        ),
    )

    # --- 7. W4 Post-pool (64 dims abstractas, top-N curves) ---
    wrapper_pool = V4PostPoolWrapper(model, pooled_ref, x_demo_ref).to(DEVICE).eval()
    X_pool_weights = (1.0 + rng.uniform(-0.5, 0.5, size=(N_SAMPLES, 64))).astype(np.float32)
    imp_pool, idx_pool = run_nobsp(
        "post_pool", wrapper_pool, X_pool_weights, POOL_NAMES,
        OUTPUT_DIR, notifier,
        top_n_curves=TOP_N_W4,
    )

    # --- 8. Comparativa demograficos V4 vs V3.2 y vs CNN 2D Camino 1 ---
    print("\n[8/8] Comparativa demograficos V4 vs versiones previas...", flush=True)
    cmp_path = OUTPUT_DIR / "compare_demo_v4_vs_prior.csv"
    prev = {k: load_csv_importance(p) for k, p in CMP_CSV.items()}
    v4_imp_dict = {FEATURE_NAMES_DEMO[i]: float(imp_demo[i]) for i in range(len(FEATURE_NAMES_DEMO))}
    with open(cmp_path, 'w', newline='') as f:
        w = csv.writer(f)
        cols = ['feature', 'imp_v32_bilstm', 'imp_cnn2d_camino1', 'imp_v4']
        w.writerow(cols + ['delta_v32', 'delta_camino1'])
        for feat in FEATURE_NAMES_DEMO:
            v32 = prev['v32_demo'].get(feat, np.nan)
            cn1 = prev['cnn2d_demo'].get(feat, np.nan)
            v4  = v4_imp_dict[feat]
            dv32 = v4 - v32 if not np.isnan(v32) else np.nan
            dcn1 = v4 - cn1 if not np.isnan(cn1) else np.nan
            w.writerow([
                feat,
                f"{v32:.6f}" if not np.isnan(v32) else "",
                f"{cn1:.6f}" if not np.isnan(cn1) else "",
                f"{v4:.6f}",
                f"{dv32:+.6f}" if not np.isnan(dv32) else "",
                f"{dcn1:+.6f}" if not np.isnan(dcn1) else "",
            ])
    print(f"  comparativa: {cmp_path}", flush=True)

    # --- Resumen ---
    elapsed = time.time() - t_global
    print(f"\nDONE en {elapsed/60:.1f} min", flush=True)

    notifier.notify_complete(
        "NObSP V4 (4 puntos)", elapsed,
        kpis={
            'demo_top1':       f"{FEATURE_NAMES_DEMO[idx_demo[0]]} ({imp_demo[idx_demo[0]]:.4f})",
            'demo_top2':       f"{FEATURE_NAMES_DEMO[idx_demo[1]]} ({imp_demo[idx_demo[1]]:.4f})",
            'input_top1':      f"{EVENT_NAMES[idx_input[0]]} ({imp_input[idx_input[0]]:.4f})",
            'input_top2':      f"{EVENT_NAMES[idx_input[1]]} ({imp_input[idx_input[1]]:.4f})",
            'pre_pool_top1':   f"{TEMPORAL_BUCKET_NAMES[idx_pre[0]]} ({imp_pre[idx_pre[0]]:.4f})",
            'post_pool_top1':  f"{POOL_NAMES[idx_pool[0]]} ({imp_pool[idx_pool[0]]:.4f})",
        },
        config={'output_dir': str(OUTPUT_DIR), 'samples': N_SAMPLES},
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        tb = traceback.format_exc()
        try:
            n = MPCNotifier(enabled=True)
            n.notify_error(type(e).__name__, str(e), traceback_text=tb)
        except Exception:
            pass
        print(tb, flush=True)
        raise
