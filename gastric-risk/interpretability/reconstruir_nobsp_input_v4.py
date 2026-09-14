"""
R4 -- Reconstruccion NObSP al espacio de entrada (solicitud S4 de Alex, 26 May).

REPRODUCES IN THE THESIS
    Input-space reconstruction of the W4 attributions via NObSP-CAM, Section 4.9.

PROVENANCE
    Copied verbatim from the working tree of the thesis. The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md at the
    repository root.

--------------------------------------------------------------------------

Toma las contribuciones por canal de W4 (post-pool, 64 canales del ultimo bloque
conv) calculadas por aplicar_nobsp_v4.py y las proyecta de vuelta sobre el
feature-map espacial (64, H', W'), produciendo un NObSP-CAM que se sube por
interpolacion bilineal al tamano de la matriz de entrada (1541 x 100). Esto
responde literalmente a S4: "te manda la informacion al mismo tamano de la matriz
de entrada... haces overlap e identificas que zonas de la matriz de entrada te
estan dando esa decision."

Reusa los mismos 500 primeros pacientes del test V4 que uso aplicar_nobsp_v4.py,
en el mismo orden, por lo que las contribuciones W4 (500, 64, 2) ya alineadas.

Salidas en nobsp_results_v4/:
  - recon_input_cam_mean_cancer.{png,pdf}     CAM medio sobre pacientes cancer
  - recon_input_cam_diff.{png,pdf}            (cancer - no_cancer) discriminante
  - recon_input_cam_ej{1,2}.{png,pdf}         ejemplos individuales cancer
  - recon_input_cam.npz                        CAMs crudos (192x12) + upsample medios

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
import json
import os
from pathlib import Path

import numpy as np
import h5py
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modelo_cnn2d_v4_pytorch import CNN2DModelV4

# ---- Constantes alineadas con aplicar_nobsp_v4.py ----
MODEL_DIR  = Path(r"D:\Proyecto_CC_Batches\Modelos_CNN2D_V4_PyTorch")
V4_DATA_DIR = Path(r"D:\Proyecto_CC_Batches\Datos_Modelo_V4")
OUTPUT_DIR = Path(r"G:\Mi unidad\Doctorado\Proyecto_CC\Test\Book\XAI_Model\nobsp_results_v4")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_SAMPLES = 500
N_EVENTS  = 100
N_FEATURES = 1541
HOT_ZONE = (40, 76)   # zona caliente W2 (slots e040-e076 sobre el eje de 100 eventos)


def latest(pattern):
    c = sorted(glob.glob(str(pattern)))
    if not c:
        raise FileNotFoundError(pattern)
    return c[-1]


def cancer_class_index(npz):
    """Indice de la clase cancer-positiva desde class_names del explainer.
    Cuidado: 'No Cancer' contiene el substring 'cancer'; hay que excluir los
    nombres con 'no'/'sin' antes de matchear."""
    names = [str(x).lower().strip() for x in npz['class_names']]
    for i, nm in enumerate(names):
        is_neg = nm.startswith('no') or nm.startswith('sin') or nm in ('0', 'neg', 'negative')
        if not is_neg and ('cancer' in nm or nm in ('1', 'pos', 'positive')):
            return i
    return len(names) - 1  # fallback: ultima clase (convencion: positiva al final)


def normalize01(a):
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def upsample_to_input(cam_hw):
    """(H',W') -> (1541,100) por interpolacion bilineal (igual que NObSPCAM_CNN)."""
    t = torch.from_numpy(cam_hw).float().unsqueeze(0).unsqueeze(0)
    up = F.interpolate(t, size=(N_FEATURES, N_EVENTS), mode='bilinear', align_corners=False)
    return up.squeeze().cpu().numpy()


def plot_cam(cam_input, title, path, cmap='jet'):
    """Heatmap del CAM (1541 x 100) con eje de eventos y zona caliente marcada."""
    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(cam_input, aspect='auto', cmap=cmap, origin='lower',
                   extent=[1, N_EVENTS, 1, N_FEATURES])
    # Marcar zona caliente W2 (columnas 40-76 del eje de eventos)
    ax.axvspan(HOT_ZONE[0], HOT_ZONE[1], color='white', alpha=0.0)
    ax.axvline(HOT_ZONE[0], color='w', ls='--', lw=1.2)
    ax.axvline(HOT_ZONE[1], color='w', ls='--', lw=1.2)
    ax.text((HOT_ZONE[0] + HOT_ZONE[1]) / 2, N_FEATURES * 0.97,
            'zona caliente W2\n(e040-e076)', color='w', ha='center', va='top',
            fontsize=11, fontweight='bold')
    ax.set_xlabel('Posicion de evento (0=mas antiguo conservado; padding a la derecha)', fontsize=12)
    ax.set_ylabel('Dimension de codigo de evento (1..1541)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Importancia NObSP-CAM (normalizada)', fontsize=11)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f"{path}.{ext}", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  guardado {path}.png/.pdf", flush=True)


def main():
    print(f"Device: {DEVICE}", flush=True)
    global OUTPUT_DIR
    model_pt = latest(MODEL_DIR / "*best*.pt")
    R2_DIR = os.environ.get('NOBSP_R2_DIR', '').strip()
    if R2_DIR:
        import r2_calib
        OUTPUT_DIR = Path(R2_DIR)
        X_events, _, y_all = r2_calib.load_r2_calib(R2_DIR, str(V4_DATA_DIR), need_events=True)
        X_events = X_events.astype(np.float32); y_all = y_all.astype(np.int64)
        print(f"[R2 train+test] n={len(y_all)} -> salida {OUTPUT_DIR}", flush=True)
    else:
        test_pt = latest(V4_DATA_DIR / "datos_modelo_test_v4_*.h5")
        with h5py.File(test_pt, 'r') as f:
            X_events = f['X_events'][:N_SAMPLES].astype(np.float32)   # (500,1541,100)
            y_all    = f['y'][:N_SAMPLES].astype(np.int64)            # (500,)
    w4_pt = OUTPUT_DIR / "post_pool_explainer.npz"
    print(f"Checkpoint: {model_pt}\nW4: {w4_pt}", flush=True)
    N = len(y_all)
    n_pos = int(y_all.sum())
    print(f"n={N}, cancer={n_pos}, no_cancer={N - n_pos}", flush=True)

    # 2. Contribuciones W4 -> importancia GLOBAL de canal (media sobre las
    #    muestras de perturbacion). W4 usa pesos aleatorios, no pacientes, asi que
    #    la media es la importancia de canal correcta para ponderar los feat-maps.
    w4 = np.load(w4_pt, allow_pickle=True)
    c_idx = cancer_class_index(w4)
    contrib_mean = w4['contributions'][:, :, c_idx].astype(np.float32).mean(axis=0)  # (64,)
    print(f"W4 contributions: {w4['contributions'].shape} (M perturbaciones), "
          f"clase cancer idx={c_idx} ({w4['class_names'][c_idx]})", flush=True)

    # 3. Modelo + feature-maps por lotes
    model = CNN2DModelV4(n_features=N_FEATURES, n_eventos=N_EVENTS, n_demograficos=15).to(DEVICE)
    model.load_state_dict(torch.load(model_pt, map_location=DEVICE, weights_only=True))
    model.eval()

    feat_maps = []
    with torch.no_grad():
        for i in range(0, N, 50):
            xb = torch.from_numpy(X_events[i:i+50]).to(DEVICE)
            fm = model.forward_conv(xb)                 # (b,64,H',W')
            feat_maps.append(fm.cpu())
    feat_maps = torch.cat(feat_maps, 0).numpy()         # (500,64,H',W')
    _, C, Hp, Wp = feat_maps.shape
    print(f"feat_map: (500, {C}, {Hp}, {Wp})", flush=True)

    # 4. NObSP-CAM: pondera el feat-map de cada paciente por la importancia GLOBAL
    #    de canal de W4 (positive-evidence). cam[n,h,w] = sum_c w_c * featmap[n,c,h,w]
    contrib_pos = np.clip(contrib_mean, 0, None)        # (64,)
    cam = np.einsum('c,nchw->nhw', contrib_pos, feat_maps)   # (N,H',W')
    cam = np.clip(cam, 0, None)

    # 5. Agregados
    mask_pos = y_all == 1
    mask_neg = y_all == 0
    cam_mean_cancer = cam[mask_pos].mean(0)             # (H',W')
    cam_mean_nocanc = cam[mask_neg].mean(0)
    cam_diff = cam_mean_cancer - cam_mean_nocanc        # discriminante

    # 6. Upsample a (1541,100)
    up_cancer = normalize01(upsample_to_input(cam_mean_cancer))
    up_diff   = normalize01(upsample_to_input(cam_diff))

    plot_cam(up_cancer,
             f"NObSP-CAM reconstruido al espacio de entrada\nMedia sobre {n_pos} pacientes con cancer (W4 -> input 1541x100)",
             str(OUTPUT_DIR / "recon_input_cam_mean_cancer"))
    plot_cam(up_diff,
             "NObSP-CAM discriminante (cancer - no cancer)\nReconstruccion al espacio de entrada",
             str(OUTPUT_DIR / "recon_input_cam_diff"), cmap='RdBu_r')

    # 7. Dos ejemplos individuales de cancer (los de mayor masa de CAM)
    pos_idx = np.where(mask_pos)[0]
    mass = cam[pos_idx].reshape(len(pos_idx), -1).sum(1)
    top2 = pos_idx[np.argsort(mass)[::-1][:2]]
    for rank, idx in enumerate(top2, 1):
        up_ej = normalize01(upsample_to_input(cam[idx]))
        plot_cam(up_ej,
                 f"NObSP-CAM individual -- paciente cancer #{idx}\nReconstruccion al espacio de entrada",
                 str(OUTPUT_DIR / f"recon_input_cam_ej{rank}"))

    # 7b. Perfil temporal + ocupacion real por clase (HONESTO: revela el confound
    #     de longitud de historia / padding). Eventos van cronologicos a la
    #     izquierda (pos 0 = mas antiguo conservado) y padding a la DERECHA
    #     (Preparar_HDF5_V4.py L370-371), por lo que la frontera dato/padding ~
    #     longitud de historia del paciente.
    prof_cancer = upsample_to_input(cam_mean_cancer).sum(axis=0)   # (100,)
    prof_nocanc = upsample_to_input(cam_mean_nocanc).sum(axis=0)   # (100,)
    prof_cancer /= prof_cancer.sum()
    prof_nocanc /= prof_nocanc.sum()
    # Ocupacion real (fraccion de pacientes con evento no-nulo en cada columna)
    occ = (np.linalg.norm(X_events, axis=1) > 1e-6)               # (500,100)
    occ_cancer = occ[mask_pos].mean(0)                            # (100,)
    occ_nocanc = occ[mask_neg].mean(0)

    fig, ax = plt.subplots(figsize=(9, 5))
    xx = np.arange(N_EVENTS)
    ax.plot(xx, prof_cancer, color='crimson', lw=2.2, label=f'CAM mass -- Cancer (n={n_pos})')
    ax.plot(xx, prof_nocanc, color='steelblue', lw=2.2, label=f'CAM mass -- No Cancer (n={N - n_pos})')
    ax.set_xlabel('Event position (0 = oldest kept event; events fill left, zero-padding to the right)', fontsize=15)
    ax.set_ylabel('NObSP-CAM mass (normalized)', fontsize=15)
    ax.set_title('Input-space NObSP-CAM temporal profile vs. real-data occupancy',
                 fontsize=16, fontweight='bold', pad=12)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=13)
    # Eje gemelo: ocupacion real (frontera dato/padding)
    ax2 = ax.twinx()
    ax2.plot(xx, occ_cancer, color='crimson', lw=1.4, ls='--', alpha=0.7,
             label='Real-data occupancy -- Cancer')
    ax2.plot(xx, occ_nocanc, color='steelblue', lw=1.4, ls='--', alpha=0.7,
             label='Real-data occupancy -- No Cancer')
    ax2.set_ylabel('Fraction of patients with\nreal event (occupancy)', fontsize=15)
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(labelsize=13)
    # Leyendas combinadas
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=12, loc='upper right')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(str(OUTPUT_DIR / f"recon_input_temporal_profile.{ext}"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  guardado recon_input_temporal_profile.png/.pdf", flush=True)
    # Cuantificar el confound: correlacion del pico CAM con la frontera dato/padding
    print(f"  [confound] eventos reales por paciente cancer: {occ[mask_pos].sum(1)}", flush=True)
    print(f"  [confound] ocupacion media cols 30-50: cancer={occ_cancer[30:50].mean():.2f} "
          f"nocancer={occ_nocanc[30:50].mean():.2f}", flush=True)

    # 8. Guardar crudos
    np.savez_compressed(
        OUTPUT_DIR / "recon_input_cam.npz",
        cam_mean_cancer=cam_mean_cancer, cam_mean_nocanc=cam_mean_nocanc,
        cam_diff=cam_diff, up_cancer=up_cancer, up_diff=up_diff,
        hot_zone=np.array(HOT_ZONE), feat_shape=np.array([C, Hp, Wp]),
        n_cancer=n_pos, cancer_class_idx=c_idx,
        prof_cancer=prof_cancer, prof_nocanc=prof_nocanc,
    )
    print(f"\nOK. CAM crudos en {OUTPUT_DIR / 'recon_input_cam.npz'}", flush=True)

    # Resumen cuantitativo de la zona caliente para la redaccion
    col_lo, col_hi = HOT_ZONE
    mass_hot = up_cancer[:, col_lo:col_hi].sum()
    mass_tot = up_cancer.sum()
    print(f"Fraccion de masa CAM en zona caliente (cols {col_lo}-{col_hi}): "
          f"{100*mass_hot/mass_tot:.1f}% (ancho zona = {100*(col_hi-col_lo)/N_EVENTS:.0f}% del eje)",
          flush=True)


if __name__ == "__main__":
    main()
