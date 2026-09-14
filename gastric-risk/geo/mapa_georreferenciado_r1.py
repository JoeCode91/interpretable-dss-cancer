"""
R1 -- Mapa georreferenciado de la contribucion NObSP geografica (solicitud S1 de

REPRODUCES IN THE THESIS
    Georeferenced NObSP geographic contribution. Produces the per-department cache that
    the choropleth below consumes. Section 4.9.

PROVENANCE
    Copied verbatim from the working tree of the thesis. The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md at the
    repository root.

--------------------------------------------------------------------------
Alex, 26 May): "mostrar en un mapa de georeferencia donde son las zonas".

Combina la contribucion NObSP de latitud + longitud (W1, clase cancer) y la
proyecta sobre un mapa de Colombia. Fondo = 1014 municipios DIVIPOLA (silueta del
pais, sin necesidad de geopandas). Frente = los 500 pacientes de test ubicados en
sus coordenadas reales, coloreados por su contribucion geografica; casos cancer
resaltados.

Marco (post-R4/R3): la contribucion geografica puede estar confundida con la
distribucion de casos (confound b: Bogota concentra ~53% de los canceres). Por eso
el script tambien agrega por departamento (contribucion media + n_pacientes +
n_cancer) para que el confound sea visible y auditable.

NOTA: NObSP W1 es de primer orden (g_lat y g_lon por separado); el mapa muestra la
contribucion ADITIVA g_lat+g_lon. Una interaccion lat x lon de segundo orden
(mencionada por Alex) requeriria un wrapper 2D distinto -> fuera de alcance, se
declara como limitacion.

Ejecutar (en este PC, venv con numpy/h5py/matplotlib):
    python mapa_georreferenciado_r1.py

Salidas en nobsp_results_v4/:
    pub_w1_geo_map.{png,pdf}        mapa de pacientes coloreados por contribucion geo
    pub_w1_geo_dept.{png,pdf}       mapa por departamento (centroides)
    r1_geo_por_departamento.csv     tabla dpto: contrib media, n_pac, n_cancer

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

# ---- Constantes (alineadas con generar_figuras_pub_v4.py) ----
OUTPUT_DIR = Path(r".\nobsp_results_v4")
V4_DATA_DIR = Path(r"./batches\Datos_Modelo_V4")
GEOCODE = Path(r".\geocode\mapa_municipio_latlon.json")
N_SAMPLES = 500
SCALERS = {'lat': {'mean': 5.6452, 'scale': 2.2558},
           'lon': {'mean': -74.9218, 'scale': 1.0864}}
LAT_COL, LON_COL = 12, 13
CANCER = 1


def inverse_scaler(z, key):
    return z * SCALERS[key]['scale'] + SCALERS[key]['mean']


def main():
    global OUTPUT_DIR
    R2_DIR = os.environ.get('NOBSP_R2_DIR', '').strip()
    if R2_DIR:
        OUTPUT_DIR = Path(R2_DIR)
        print(f"  [R2 train+test] leyendo de {OUTPUT_DIR}", flush=True)

    # 1. Contribuciones NObSP W1 (demo)
    data = np.load(OUTPUT_DIR / "demo_explainer.npz", allow_pickle=True)
    contributions = data['contributions']                 # (N, 15, 2)
    # verificacion defensiva de orden de features
    if 'feature_names' in data.files:
        fn = [str(x) for x in data['feature_names']]
        if fn[LAT_COL] != 'lat' or fn[LON_COL] != 'lon':
            print(f"  AVISO: feature_names[{LAT_COL}]={fn[LAT_COL]}, [{LON_COL}]={fn[LON_COL]} "
                  f"(esperado lat/lon) -- revisar orden")
    print(f"  contributions: {contributions.shape}")

    # 2. X_demo + etiquetas (500 test, o conjunto R2 train+test)
    if R2_DIR:
        import r2_calib
        _, X_demo, y = r2_calib.load_r2_calib(R2_DIR, str(V4_DATA_DIR), need_events=False)
        X_demo = X_demo.astype(np.float32)
    else:
        test_files = sorted(glob.glob(str(V4_DATA_DIR / "datos_modelo_test_v4_*.h5")))
        with h5py.File(test_files[-1], 'r') as f:
            X_demo = f['X_demograficos'][:N_SAMPLES].astype(np.float32)
            y = f['y'][:N_SAMPLES].astype(int)
    lat = inverse_scaler(X_demo[:, LAT_COL], 'lat')       # grados N
    lon = inverse_scaler(X_demo[:, LON_COL], 'lon')       # grados W (negativo)
    n_pos = int(y.sum())
    print(f"  n={len(y)}, cancer={n_pos}; lat[{lat.min():.2f},{lat.max():.2f}] "
          f"lon[{lon.min():.2f},{lon.max():.2f}]")

    # 3. Contribuciones geograficas por separado (clase cancer)
    glat = contributions[:, LAT_COL, CANCER]
    glon = contributions[:, LON_COL, CANCER]
    W_LAT, W_LON = 0.9, 0.1                       # ponderacion: lat domina, lon residual
    gw = W_LAT * glat + W_LON * glon              # combinada pesada
    print(f"  g_lat: %pos={100*(glat>0).mean():.0f}%  g_lon: %pos={100*(glon>0).mean():.0f}%  "
          f"gw(0.9/0.1): %pos={100*(gw>0).mean():.0f}%", flush=True)

    # 4. DIVIPOLA: municipios (fondo) + reverse-map paciente -> departamento
    divipola = json.load(open(GEOCODE, encoding='utf-8'))
    m_lat = np.array([v['lat'] for v in divipola.values()])
    m_lon = np.array([v['lon'] for v in divipola.values()])
    m_dpto = np.array([v.get('nom_dpto', '?') for v in divipola.values()])
    # nearest municipio para cada paciente (match ~exacto: V4 uso centroides)
    # distancia^2 (500 x 1014) -- barato
    d2 = (lat[:, None] - m_lat[None, :]) ** 2 + (lon[:, None] - m_lon[None, :]) ** 2
    nn = d2.argmin(axis=1)
    pat_dpto = m_dpto[nn]
    max_err = np.sqrt(d2[np.arange(len(nn)), nn]).max()
    print(f"  reverse-map dpto: error max {max_err:.4f} grados (deberia ~0)")

    # 5. TRES mapas: latitud sola, longitud sola, y combinada pesada (0.9/0.1).
    #    Colormap divergente centrado en 0 -> positivo (hacia cancer) en ROJO.
    def geo_map(val, title, cblabel, fname):
        cmax = float(np.abs(val).max()) or 1.0
        fig, ax = plt.subplots(figsize=(8, 9))
        ax.scatter(m_lon, m_lat, s=3, c='#dddddd', zorder=1, rasterized=True)  # fondo pais
        sc = ax.scatter(lon[y == 0], lat[y == 0], c=val[y == 0], cmap='RdBu_r',
                        vmin=-cmax, vmax=cmax, s=22, alpha=0.8, zorder=2,
                        edgecolors='none', rasterized=True, label='No cancer')
        ax.scatter(lon[y == 1], lat[y == 1], c=val[y == 1], cmap='RdBu_r',
                   vmin=-cmax, vmax=cmax, s=95, zorder=3, edgecolors='black',
                   linewidths=1.3, marker='o', label=f'Cancer (n={n_pos})')
        cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.03)
        cb.set_label(cblabel)
        ax.set_xlabel('Longitude (degrees W)')
        ax.set_ylabel('Latitude (degrees N)')
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='lower left', fontsize=9)
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(OUTPUT_DIR / f"{fname}.{ext}", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  guardado {fname}.png/.pdf")

    geo_map(glat, 'Latitude NObSP contribution over Colombia (W1, cancer class)',
            'g_lat (cancer class)', 'pub_w1_geo_map_lat')
    geo_map(glon, 'Longitude NObSP contribution over Colombia (W1, cancer class)',
            'g_lon (cancer class)', 'pub_w1_geo_map_lon')
    geo_map(gw, f'Weighted geographic NObSP contribution ({W_LAT:.0%} lat + {W_LON:.0%} lon)',
            f'{W_LAT:.0%} g_lat + {W_LON:.0%} g_lon', 'pub_w1_geo_map_weighted')

    # 6. Agregacion por departamento + FIGURA 2 (centroides)
    rows = []
    dptos = sorted(set(pat_dpto))
    for d in dptos:
        mask = pat_dpto == d
        rows.append({
            'departamento': d, 'n_pac': int(mask.sum()),
            'n_cancer': int(y[mask].sum()),
            'contrib_media': float(gw[mask].mean()),
            'lat_c': float(lat[mask].mean()), 'lon_c': float(lon[mask].mean()),
        })
    rows.sort(key=lambda r: abs(r['contrib_media']), reverse=True)
    with open(OUTPUT_DIR / "r1_geo_por_departamento.csv", 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['departamento', 'n_pac', 'n_cancer',
                                           'contrib_media', 'lat_c', 'lon_c'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("  guardado r1_geo_por_departamento.csv")

    cmax2 = max(abs(r['contrib_media']) for r in rows) or 1.0
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.scatter(m_lon, m_lat, s=3, c='#eeeeee', zorder=1, rasterized=True)
    for r in rows:
        ax.scatter(r['lon_c'], r['lat_c'], s=30 + 18 * r['n_pac'] ** 0.5,
                   c=[r['contrib_media']], cmap='RdBu_r', vmin=-cmax2, vmax=cmax2,
                   edgecolors='black', linewidths=0.8, zorder=2)
        if r['n_cancer'] > 0:
            ax.text(r['lon_c'], r['lat_c'] + 0.18, f"{r['departamento'][:10]}\n({r['n_cancer']} ca)",
                    fontsize=6.5, ha='center', va='bottom', zorder=4)
    ax.set_xlabel('Longitude (degrees W)')
    ax.set_ylabel('Latitude (degrees N)')
    ax.set_title('Mean geographic NObSP contribution by department\n(size = n patients; labeled = has cancer cases)')
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUTPUT_DIR / f"pub_w1_geo_dept.{ext}", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  guardado pub_w1_geo_dept.png/.pdf")

    # 7. Resumen consola (para auditar el confound b)
    print("\n=== TOP departamentos por |contribucion media| ===")
    print(f"{'depto':22s} {'n_pac':>5s} {'n_ca':>4s} {'contrib':>9s}")
    for r in rows[:12]:
        print(f"  {r['departamento'][:20]:20s} {r['n_pac']:5d} {r['n_cancer']:4d} {r['contrib_media']:9.4f}")
    n_dpto_cancer = sum(1 for r in rows if r['n_cancer'] > 0)
    print(f"\nDepartamentos con >=1 cancer en test: {n_dpto_cancer} de {len(rows)}")
    print("(Confound b: si los canceres se concentran en pocos departamentos, la")
    print(" contribucion geografica refleja esa distribucion, no causalidad espacial.)")


if __name__ == "__main__":
    main()
