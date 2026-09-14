"""
CAM por paciente sobre base balanceada COMPLETA (todos los canceres + no-cancer aleatorios).

REPRODUCES IN THE THESIS
    NObSP-CAM per patient over the balanced set, Section 4.9.

PROVENANCE
    Copied verbatim from the working tree of the thesis. The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md at the
    repository root.

--------------------------------------------------------------------------

Pregunta del usuario: en vez de las 500 perturbaciones de W2, usar pacientes REALES.
Metodo: NObSP-CAM por paciente (reusa los beta_c de W4 ya obtenidos en
post_pool_explainer.npz; solo se calcula el feature-map de cada paciente con un
forward pass, sin re-correr NObSP). Cada fila del heatmap es un paciente real.

Base: TODOS los pacientes de cancer de train+val+test (~963) + igual numero de
no-cancer aleatorios (mismo conteo por split), balanceada.

Salidas en nobsp_results_v4_traintest/:
  - cam_per_patient_heatmap.{png,pdf}        heatmap por paciente (cancer arriba / no-cancer abajo)
  - cam_per_patient_mean_profiles.{png,pdf}  perfiles medios cancer vs no-cancer + ocupacion
  - cam_per_patient.npz                       datos crudos

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
import glob
from pathlib import Path
import numpy as np
import h5py
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modelo_cnn2d_v4_pytorch import CNN2DModelV4

MODEL_DIR = Path(r"D:\Proyecto_CC_Batches\Modelos_CNN2D_V4_PyTorch")
V4 = Path(r"D:\Proyecto_CC_Batches\Datos_Modelo_V4")
OUT = Path(r"G:\Mi unidad\Doctorado\Proyecto_CC\Test\Book\XAI_Model\nobsp_results_v4_traintest")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_FEAT, N_EV = 1541, 100
rng = np.random.default_rng(42)


def latest(pattern):
    c = sorted(glob.glob(str(pattern)))
    if not c:
        raise FileNotFoundError(pattern)
    return c[-1]


def cancer_idx(npz):
    names = [str(x).lower().strip() for x in npz['class_names']]
    for i, nm in enumerate(names):
        if not (nm.startswith('no') or nm.startswith('sin')) and \
           ('cancer' in nm or nm in ('1', 'pos', 'positive')):
            return i
    return len(names) - 1


def norm01(a):
    a = np.asarray(a, float); lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def main():
    print(f"Device: {DEVICE}", flush=True)
    # --- beta_c de W4 (resultado NObSP ya obtenido) ---
    w4 = np.load(OUT / "post_pool_explainer.npz", allow_pickle=True)
    ci = cancer_idx(w4)
    beta = np.clip(w4['contributions'][:, :, ci].astype(np.float32).mean(0), 0, None)  # (64,)
    print(f"W4 beta_c (64) cargado, clase cancer idx={ci}", flush=True)

    # --- cargar TODOS los canceres + igual no-cancer por split ---
    Xc_list, Xn_list = [], []
    for split in ['train', 'val', 'test']:
        f = latest(V4 / f"datos_modelo_{split}_v4_*.h5")
        with h5py.File(f, 'r') as h:
            y = h['y'][:]
            cidx = np.sort(np.where(y == 1)[0])
            nall = np.where(y == 0)[0]
            nidx = np.sort(rng.choice(nall, size=len(cidx), replace=False))
            Xc_list.append(h['X_events'][cidx].astype(np.float32))
            Xn_list.append(h['X_events'][nidx].astype(np.float32))
        print(f"  {split}: {len(cidx)} cancer + {len(cidx)} no-cancer", flush=True)
    Xc = np.concatenate(Xc_list); Xn = np.concatenate(Xn_list)
    print(f"Total: {len(Xc)} cancer + {len(Xn)} no-cancer", flush=True)

    # --- modelo ---
    model = CNN2DModelV4(n_features=N_FEAT, n_eventos=N_EV, n_demograficos=15).to(DEVICE)
    model.load_state_dict(torch.load(latest(MODEL_DIR / "*best*.pt"),
                                     map_location=DEVICE, weights_only=True))
    model.eval()

    def cam_profiles(X):
        """CAM por paciente -> perfil temporal (N, 100)."""
        profs = []
        with torch.no_grad():
            for i in range(0, len(X), 64):
                xb = torch.from_numpy(X[i:i + 64]).to(DEVICE)
                fm = model.forward_conv(xb).cpu().numpy()        # (b,64,Hp,Wp)
                cam = np.einsum('c,bchw->bhw', beta, fm)         # (b,Hp,Wp)
                cam = np.clip(cam, 0, None)
                prof = cam.mean(axis=1)                          # (b,Wp) colapsa eje feature
                profs.append(prof)
        profs = np.concatenate(profs)                            # (N,Wp)
        t = torch.from_numpy(profs).float().unsqueeze(1)         # (N,1,Wp)
        up = F.interpolate(t, size=N_EV, mode='linear', align_corners=False)
        return up.squeeze(1).numpy()                             # (N,100)

    prof_c = cam_profiles(Xc)
    prof_n = cam_profiles(Xn)
    print(f"perfiles: cancer {prof_c.shape}, no-cancer {prof_n.shape}", flush=True)

    # --- ocupacion (fraccion de pacientes con evento real en cada posicion) ---
    occ_c = (np.abs(Xc).sum(axis=1) > 0).mean(axis=0)  # (100,)
    occ_n = (np.abs(Xn).sum(axis=1) > 0).mean(axis=0)

    # === FIG 1: heatmap por paciente (cancer arriba, no-cancer abajo) ===
    # normalizacion global comun para comparar magnitudes
    allp = np.concatenate([prof_c, prof_n])
    cmax = float(np.percentile(allp, 99))
    # ordenar dentro de cada grupo por intensidad total (mas legible)
    oc = np.argsort(prof_c.sum(1))[::-1]
    on = np.argsort(prof_n.sum(1))[::-1]
    stack = np.vstack([prof_c[oc], prof_n[on]])
    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(stack, aspect='auto', cmap='hot_r', vmin=0, vmax=cmax,
                   interpolation='nearest')
    ax.axhline(len(prof_c) - 0.5, color='cyan', lw=2)
    ax.text(2, len(prof_c) * 0.5, f'CANCER\n(n={len(prof_c)})', color='blue',
            fontsize=15, fontweight='bold', va='center')
    ax.text(2, len(prof_c) + len(prof_n) * 0.5, f'NO CANCER\n(n={len(prof_n)})',
            color='black', fontsize=15, fontweight='bold', va='center')
    ax.set_xlabel("Event position (0 = oldest kept event, 99 = most recent)", fontsize=18)
    ax.set_ylabel("Patients (all cancer + balanced non-cancer)", fontsize=18)
    ax.set_title("Per-patient NObSP-CAM contribution to the cancer class",
                 fontsize=19, pad=12)
    ax.set_xticks(np.arange(0, 100, 10)); ax.tick_params(labelsize=15)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("CAM contribution\n(positive evidence)", fontsize=17, labelpad=10)
    cb.ax.tick_params(labelsize=14)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f"cam_per_patient_heatmap.{ext}", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # === FIG 2: perfiles medios cancer vs no-cancer + ocupacion ===
    ev = np.arange(1, N_EV + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ev, norm01(prof_c.mean(0)), color='C3', lw=2.4, label=f'Cancer mean (n={len(prof_c)})')
    ax.plot(ev, norm01(prof_n.mean(0)), color='C0', lw=2.4, label=f'Non-cancer mean (n={len(prof_n)})')
    ax.plot(ev, occ_c, color='C3', lw=1.2, ls=':', alpha=0.7, label='Cancer occupancy')
    ax.plot(ev, occ_n, color='C0', lw=1.2, ls=':', alpha=0.7, label='Non-cancer occupancy')
    ax.set_xlabel("Event position (0 = oldest kept, 99 = most recent)", fontsize=14)
    ax.set_ylabel("Normalized CAM / occupancy", fontsize=14)
    ax.set_title("Mean per-patient CAM profile vs occupancy, by class", fontsize=14)
    ax.legend(fontsize=10); ax.grid(alpha=0.3); ax.tick_params(labelsize=12)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f"cam_per_patient_mean_profiles.{ext}", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- stats ---
    r = float(np.corrcoef(norm01(prof_c.mean(0)), norm01(prof_n.mean(0)))[0, 1])
    rc = float(np.corrcoef(norm01(prof_c.mean(0)), occ_c)[0, 1])
    print(f"\nPearson r (perfil medio cancer vs no-cancer) = {r:.3f}", flush=True)
    print(f"Pearson r (perfil cancer vs ocupacion cancer) = {rc:.3f}", flush=True)
    print(f"masa CAM media: cancer={prof_c.mean():.4f}, no-cancer={prof_n.mean():.4f}", flush=True)
    print(f"ocupacion media (pos 30-50): cancer={occ_c[30:50].mean():.3f}, no-cancer={occ_n[30:50].mean():.3f}", flush=True)
    np.savez(OUT / "cam_per_patient.npz", prof_c=prof_c, prof_n=prof_n,
             occ_c=occ_c, occ_n=occ_n, pearson_cn=r, pearson_occ=rc)
    print("guardado cam_per_patient_heatmap/_mean_profiles + .npz", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
