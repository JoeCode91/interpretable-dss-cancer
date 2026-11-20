"""
Validacion de los beta de NObSP: convergencia vs numero de probe points.

REPRODUCES IN THE THESIS
    Convergence of the NObSP beta coefficients, Section 4.9.

PROVENANCE
    Taken from the working tree that produced the reported results. The computational
    content is unchanged; absolute paths were replaced by relative defaults and
    environment variables, notification calls were removed, and comments were edited.
    See CHANGES-FROM-THESIS.md at the repository root.

--------------------------------------------------------------------------

Para cada wrapper (W4=64 dims, W3=12, W2=100) corre NObSP con n_probe creciente
y mide si los beta (importancia por feature, clase cancer) se ESTABILIZAN:
correlacion de Pearson beta(n) vs beta(n_max). Tambien la COMPLETITUD
(reconstruccion): corr(sum_d contribuciones, salida del modelo centrada) -- debe
ser ~1 (descomposicion aditiva del metodo alpha).

Si beta(500) ~ beta(n_max) (r alto) y la completitud ~1, los beta a n=500 validados.

Referencia: la media de la muestra B (calib_index.npz en NOBSP_RUNB_DIR, leida con
r2_calib). El checkpoint se fija con CNN2D_TS (training/checkpoint_utils.py).

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
import sys
import glob
from pathlib import Path

# Modulos hermanos del repositorio
_HERE = Path(__file__).resolve().parent
for _d in (_HERE, _HERE.parent / "models", _HERE.parent / "training"):
    sys.path.insert(0, str(_d))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nobsp_lib import TabularExplainer
from modelo_cnn2d_v4_pytorch import CNN2DModelV4
import r2_calib as r2_calib
import checkpoint_utils as ck

R2_DIR = os.environ.get('NOBSP_RUNB_DIR', "./batches/Resultados/nobsp_runB_traintest")
V4_DATA = os.environ.get('CNN_DATA_DIR', "./batches/Datos_Modelo_CNN")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---- Wrappers (inline, copiados de aplicar_nobsp_v4.py) ----
class V4InputEventWrapper(nn.Module):
    def __init__(self, m, x_events_ref, x_demo_ref):
        super().__init__(); self.model = m
        self.register_buffer('x_events_ref', x_events_ref.unsqueeze(0))
        self.register_buffer('x_demo_ref', x_demo_ref.unsqueeze(0))

    def forward(self, w):
        B = w.shape[0]
        x = self.x_events_ref.expand(B, -1, -1) * w.view(B, 1, 100)
        return self.model(x, self.x_demo_ref.expand(B, -1))


class V4PreToolWrapper(nn.Module):
    def __init__(self, m, feat_map_ref, x_demo_ref):
        super().__init__(); self.model = m
        self.register_buffer('feat_map_ref', feat_map_ref.unsqueeze(0))
        self.register_buffer('x_demo_ref', x_demo_ref.unsqueeze(0))
        self.W = feat_map_ref.shape[-1]

    def forward(self, w):
        B = w.shape[0]
        feat = self.feat_map_ref.expand(B, -1, -1, -1) * w.view(B, 1, 1, self.W)
        pooled = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
        return self.model.forward_head(pooled, self.x_demo_ref.expand(B, -1))


class V4PostPoolWrapper(nn.Module):
    def __init__(self, m, pooled_ref, x_demo_ref):
        super().__init__(); self.model = m
        self.register_buffer('pooled_ref', pooled_ref.unsqueeze(0))
        self.register_buffer('x_demo_ref', x_demo_ref.unsqueeze(0))

    def forward(self, w):
        B = w.shape[0]
        return self.model.forward_head(self.pooled_ref.expand(B, -1) * w,
                                       self.x_demo_ref.expand(B, -1))


def beta_and_completeness(wrapper, d, n):
    rng = np.random.default_rng(0)
    X = (1.0 + rng.uniform(-0.5, 0.5, size=(n, d))).astype(np.float32)
    expl = TabularExplainer(task="classification", method="alpha",
                            regularization=1e-4, device=str(DEVICE))
    expl.fit(wrapper, X, feature_names=[f"x{i}" for i in range(d)],
             class_names=["No Cancer", "Cancer"], output_kind="probabilities")
    imp = expl.feature_importance()
    imp = imp[:, 1] if imp.ndim > 1 else imp
    contr = expl.explain()
    recon = contr[:, :, 1].sum(axis=1)
    with torch.no_grad():
        out = wrapper(torch.from_numpy(X).to(DEVICE)).cpu().numpy().ravel()
    oc = out - out.mean(); rc = recon - recon.mean()
    comp = float(np.corrcoef(oc, rc)[0, 1]) if oc.std() > 1e-9 and rc.std() > 1e-9 else float('nan')
    return imp, comp


def run_wrapper(tag, wrapper, d, ns):
    print(f"\n{'='*60}\n  {tag} (d={d})\n{'='*60}", flush=True)
    betas = {}
    for n in ns:
        imp, comp = beta_and_completeness(wrapper, d, n)
        betas[n] = imp
        print(f"  n_probe={n:5d}  completeness(recon r)={comp:.4f}", flush=True)
    nmax = ns[-1]
    print(f"  --- correlacion beta(n) vs beta(n_max={nmax}) ---", flush=True)
    for n in ns:
        r = float(np.corrcoef(betas[n], betas[nmax])[0, 1])
        print(f"    beta({n:5d}) vs beta({nmax}): r = {r:.4f}", flush=True)


def main():
    print(f"Device: {DEVICE}", flush=True)
    X_events, X_demo, y = r2_calib.load_r2_calib(R2_DIR, V4_DATA, need_events=True)
    X_events = X_events.astype(np.float32); X_demo = X_demo.astype(np.float32)
    model = CNN2DModelV4(n_features=1541, n_eventos=100, n_demograficos=15).to(DEVICE)
    ckpt = ck.cnn_pt()   # checkpoint fijado por CNN2D_TS
    print("checkpoint", ckpt, flush=True)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()
    x_events_ref = torch.from_numpy(X_events.mean(0)).to(DEVICE)
    x_demo_ref = torch.from_numpy(X_demo.mean(0)).to(DEVICE)
    with torch.no_grad():
        feat_map_ref = model.forward_conv(x_events_ref.unsqueeze(0)).squeeze(0)
        pooled_ref = F.adaptive_avg_pool2d(feat_map_ref.unsqueeze(0), 1).squeeze(-1).squeeze(-1).squeeze(0)
    W_pool = feat_map_ref.shape[-1]
    print(f"refs OK: feat_map {tuple(feat_map_ref.shape)}, W_pool={W_pool}", flush=True)

    w4 = V4PostPoolWrapper(model, pooled_ref, x_demo_ref).to(DEVICE).eval()
    w3 = V4PreToolWrapper(model, feat_map_ref, x_demo_ref).to(DEVICE).eval()
    w2 = V4InputEventWrapper(model, x_events_ref, x_demo_ref).to(DEVICE).eval()

    run_wrapper("W4 post-pool", w4, 64, [100, 250, 500, 1000, 2000])
    run_wrapper("W3 pre-pool", w3, W_pool, [50, 100, 250, 500, 1000, 2000])
    run_wrapper("W2 input-event (lento)", w2, 100, [250, 500, 1000])
    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
