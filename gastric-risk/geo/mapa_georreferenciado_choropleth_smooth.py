"""
Mapa coropletico NObSP W1 -- version GRADIENTE SUAVE full-pais (#13 Alex).

REPRODUCES IN THE THESIS
    The smooth municipal choropleth: this is the map reproduced in the thesis as
    pub_w1_geo_choropleth_smooth, Section 4.9.

PROVENANCE
    Copied verbatim from the working tree of the thesis. The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md at the
    repository root.

--------------------------------------------------------------------------
Mapa coropletico NObSP W1 -- version GRADIENTE SUAVE full-pais (#13 Alex).

Como g_lat (contribucion W1 de la latitud) es esencialmente una funcion de la
latitud, se deriva g_lat(latitud) a partir de los puntos por departamento
cacheados (r1_geo_por_departamento.csv: lat_c, contrib_media; monotono
sur(+)/norte(-)) via PCHIP, y se colorea CADA uno de los 1122 municipios por la
latitud de su centroide. Coropleta completa, suave, sin huecos grises.

NO depende de la unidad D: (usa solo lo cacheado en G:).
Salida: nobsp_results_v4_traintest/pub_w1_geo_choropleth_smooth.{png,pdf}
"""
import os
import csv
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.interpolate import PchipInterpolator

from mapa_georreferenciado_choropleth import (read_shp_records, read_dbf_codes,
                                              SHP, DBF, GEOJSON, OUTPUT_DIR)

DEPT_CSV = OUTPUT_DIR / "r1_geo_por_departamento.csv"


def main():
    print("Leyendo shapefile + dbf...", flush=True)
    records = read_shp_records(SHP)
    codes = read_dbf_codes(DBF, 'ID')
    assert len(records) == len(codes)
    geo = json.loads(GEOJSON.read_text(encoding='utf-8'))

    # --- g_lat(latitud) via PCHIP sobre puntos por departamento ---
    lat_c, contrib = [], []
    with open(DEPT_CSV, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            lat_c.append(float(row['lat_c']))
            contrib.append(float(row['contrib_media']))
    lat_c, contrib = np.array(lat_c), np.array(contrib)
    o = np.argsort(lat_c)
    lat_s, c_s = lat_c[o], contrib[o]
    # dedupe latitudes (PCHIP requiere x estrictamente creciente)
    uniq, inv = np.unique(lat_s, return_inverse=True)
    c_u = np.array([c_s[inv == i].mean() for i in range(len(uniq))])
    pchip = PchipInterpolator(uniq, c_u)
    lat_min, lat_max = float(uniq.min()), float(uniq.max())
    print(f"  PCHIP sobre {len(uniq)} latitudes [{lat_min:.2f}, {lat_max:.2f}]", flush=True)

    def g_of_lat(L):
        return float(pchip(np.clip(L, lat_min, lat_max)))

    # --- latitud por municipio (centroide JSON; fallback geometria) ---
    vals = []
    for rings, code in zip(records, codes):
        if not rings:
            vals.append(None)
            continue
        key = str(code)
        if key in geo:
            L = geo[key]['lat']
        else:  # municipio fuera de la cohorte -> centroide geometrico
            allpts = np.vstack(rings)
            L = float(allpts[:, 1].mean())
        vals.append(g_of_lat(L))

    finite = [v for v in vals if v is not None]
    cmax = float(np.max(np.abs(finite))) or 1.0
    norm = Normalize(vmin=-cmax, vmax=cmax)
    cmap = plt.get_cmap('RdBu_r')

    fig, ax = plt.subplots(figsize=(8.5, 10.5))
    patches, facecolors = [], []
    for rings, v in zip(records, vals):
        if not rings or v is None:
            continue
        verts, codes_p = [], []
        for ring in rings:
            verts.extend(ring.tolist())
            verts.append(ring[0].tolist())
            codes_p.extend([MPath.MOVETO] + [MPath.LINETO] * (len(ring) - 1) + [MPath.CLOSEPOLY])
        patches.append(MPath(verts, codes_p))
        facecolors.append(cmap(norm(v)))

    pc = PatchCollection([PathPatch(p) for p in patches], match_original=False,
                         facecolors=facecolors, edgecolors=(0.5, 0.5, 0.5, 0.18),
                         linewidths=0.2, zorder=1, rasterized=True)
    ax.add_collection(pc)

    ax.set_xlim(-79.5, -66.8)
    ax.set_ylim(-4.5, 13.2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Longitude ($^\\circ$W)', fontsize=18)
    ax.set_ylabel('Latitude ($^\\circ$N)', fontsize=18)
    ax.set_title('Latitude NObSP contribution $g_{\\mathrm{lat}}$ (smooth gradient)',
                 fontsize=19, fontweight='bold', pad=12)
    ax.tick_params(labelsize=14)

    fig.subplots_adjust(left=0.11, right=0.97, top=0.94, bottom=0.15)
    cax = fig.add_axes([0.30, 0.038, 0.42, 0.018])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation='horizontal')
    cb.set_label('NObSP contribution to cancer class', fontsize=15)
    cb.ax.tick_params(labelsize=12)

    for ext in ('png', 'pdf'):
        fig.savefig(OUTPUT_DIR / f"pub_w1_geo_choropleth_smooth.{ext}", dpi=200,
                    bbox_inches='tight')
    print("guardado pub_w1_geo_choropleth_smooth.png/pdf", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
