"""
W3 input-space reconstruction and comparison with W4.

REPRODUCES IN THE THESIS
    Input-space reconstruction of W3 and its comparison with W4, Section 4.9.

PROVENANCE
    Taken from the working tree that produced the reported results. The computational
    content is unchanged; absolute paths were replaced by relative defaults and
    environment variables, notification calls were removed, and comments were edited.
    See CHANGES-FROM-THESIS.md at the repository root.

--------------------------------------------------------------------------

A diferencia de W4 (cuyos 64 canales post-pool son no-espaciales y requieren el
NObSP-CAM para localizarse en el espacio de entrada), W3 ya atribuye DIRECTAMENTE
sobre el eje temporal: sus 12 buckets son columnas del feature-map pre-pool. Por
tanto, reconstruir W3 al espacio de entrada es simplemente mapear las 12
importancias de bucket al eje de 100 eventos (upsampling), usando la misma malla
que el upsample bilineal de W4 (12 -> 100 en el eje temporal).

Esto permite comparar la localizacion temporal recuperada por pre-pool (W3) y
post-pool (W4): si ambas siguen el mismo perfil (y este sigue la ocupacion),
refuerza que la localizacion temporal es un confound de longitud de historia.

Muestra B: todos los cancer de train (100:1) y test, y el mismo numero de no cancer
de train+test (RandomState(42)); W2-W4 con 500 sondas multiplicativas alrededor de
la media de esa muestra.

Ligero: trabaja sobre los .npz ya generados (sin GPU, sin modelo).
Salidas en NOBSP_RUNB_DIR (por defecto ./batches/Resultados/nobsp_runB_traintest):
  - recon_w3_input_profile.npz          perfil W3 (12 -> 100) + W4
  - recon_w3_vs_w4_profile.{png,pdf}    overlay comparativo

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
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(os.environ.get('NOBSP_RUNB_DIR', "./batches/Resultados/nobsp_runB_traintest"))
N_EVENTS = 100


def cancer_idx(class_names):
    names = [str(x).lower().strip() for x in class_names]
    for i, nm in enumerate(names):
        if not (nm.startswith('no') or nm.startswith('sin')) and \
           ('cancer' in nm or nm in ('1', 'pos', 'positive')):
            return i
    return len(names) - 1


# El reescalado 12 -> 100 usa centros de celda (8,33c + 4,2), mientras que el centro del
# campo receptivo de la columna c de conv4 es 8c + 3,5: desfase de hasta 3,7 eventos en
# c = 11. Es menor, pero conviene tenerlo presente si se citan posiciones finas.
def upsample_1d(v, n):
    """Lineal de len(v) -> n con centros de celda (analogo a bilinear align_corners=False)."""
    x_old = (np.arange(len(v)) + 0.5) / len(v)
    x_new = (np.arange(n) + 0.5) / n
    return np.interp(x_new, x_old, v)


def norm01(a):
    a = np.asarray(a, float)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def main():
    # --- W3: importancia por bucket (12) para la clase cancer ---
    w3 = np.load(D / "pre_pool_explainer.npz", allow_pickle=True)
    ci = cancer_idx(w3['class_names'])
    w3_imp = np.abs(w3['contributions'][:, :, ci]).mean(0)   # (12,)
    w3_prof = upsample_1d(w3_imp, N_EVENTS)                   # (100,)
    print(f"clase cancer idx={ci} ({w3['class_names'][ci]})")
    print("W3 importancia por bucket (12):", np.round(w3_imp, 4))

    # --- W4: perfil temporal reconstruido (100) ya guardado ---
    rec = np.load(D / "recon_input_cam.npz", allow_pickle=True)
    w4_prof = rec['prof_cancer'].astype(float)               # (100,)

    w3n, w4n = norm01(w3_prof), norm01(w4_prof)
    r = float(np.corrcoef(w3n, w4n)[0, 1])
    print(f"Pearson r (W3 vs W4, perfil normalizado) = {r:.3f}")

    # --- plot comparativo ---
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ev = np.arange(1, N_EVENTS + 1)
    ax.plot(ev, w4n, color='C3', lw=2.2, label='W4 reconstruction (post-pool, NObSP-CAM)')
    ax.plot(ev, w3n, color='C0', lw=2.2, ls='--',
            label='W3 reconstruction (pre-pool buckets, upsampled)')
    ax.set_xlabel('Event position (0 = oldest kept; padding to the right)', fontsize=15)
    ax.set_ylabel('Normalized contribution', fontsize=15)
    ax.set_title(f'Input-space reconstruction: W3 vs W4 (cancer class)  |  Pearson r = {r:.2f}',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=13, loc='upper right')
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=13)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(D / f"recon_w3_vs_w4_profile.{ext}", dpi=150, bbox_inches='tight')
    np.savez(D / "recon_w3_input_profile.npz",
             w3_bucket_imp=w3_imp, w3_prof=w3_prof, w4_prof=w4_prof, pearson_r=r)
    print("guardado recon_w3_vs_w4_profile.png/pdf y recon_w3_input_profile.npz")


if __name__ == "__main__":
    main()
