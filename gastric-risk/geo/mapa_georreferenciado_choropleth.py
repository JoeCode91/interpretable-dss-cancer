"""
Mapa coropletico NObSP W1 por municipio.

REPRODUCES IN THE THESIS
    The municipal choropleth of the NObSP geographic contribution; also provides the
    shapefile readers used by mapa_georreferenciado_choropleth_smooth.py. Section 4.9.

PROVENANCE
    Taken from the working tree that produced the reported results. The computational
    content is unchanged; absolute paths were replaced by relative defaults and
    environment variables, notification calls were removed, and comments were edited.
    See CHANGES-FROM-THESIS.md at the repository root.

--------------------------------------------------------------------------

COROPLETA municipal de la contribucion NObSP de la latitud:
  - Cada municipio se rellena con la contribucion NObSP media (g_lat) de sus
    pacientes (clase cancer), escala divergente RdBu_r.
  - Un solo panel (latitud).
  - Lineas de division municipal muy transparentes.
  - Municipios sin pacientes en el conjunto -> gris claro (sin dato).

Pipeline de matching:
  patient (lat,lon) inverse-scaled == centroide DANE del JSON DIVIPOLA -> se
  asigna cada paciente al codigo de municipio por centroide mas cercano; se
  promedia g_lat por codigo; el codigo del poligono se lee del DBF (campo ID).

Pacientes de la muestra B (calib_index.npz, leida con r2_calib). Necesita el
shapefile municipal COL_Municipio.{shp,dbf} (GEO_SHP_DIR, por defecto ./geo/col_shp)
y el JSON DIVIPOLA de centroides (GEOMAP_JSON, ver construir_mapa_municipio_latlon.py).

Salida: NOBSP_R2_DIR o NOBSP_RUNB_DIR (por defecto
./batches/Resultados/nobsp_runB_traintest)/pub_w1_geo_choropleth.{png,pdf}

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
import sys as _sys
from pathlib import Path as _P
# Modulos hermanos del repositorio
_HERE = _P(__file__).resolve().parent
for _d in (_HERE, _HERE.parent / "training"):
    _sys.path.insert(0, str(_d))

import os
import struct
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

SHP_DIR = Path(os.environ.get("GEO_SHP_DIR", "./geo/col_shp"))
SHP = SHP_DIR / "COL_Municipio.shp"
DBF = SHP_DIR / "COL_Municipio.dbf"
GEOJSON = Path(os.environ.get("GEOMAP_JSON", "./geo/mapa_municipio_latlon.json"))
OUTPUT_DIR = Path(os.environ.get('NOBSP_R2_DIR',
                  os.environ.get('NOBSP_RUNB_DIR', "./batches/Resultados/nobsp_runB_traintest")))
V4_DATA_DIR = Path(os.environ.get('CNN_DATA_DIR', "./batches/Datos_Modelo_CNN"))
SCALERS = {'lat': {'mean': 5.6452, 'scale': 2.2558},
           'lon': {'mean': -74.9218, 'scale': 1.0864}}
LAT_COL, LON_COL, CANCER = 12, 13, 1
R_MERC = 6378137.0


def merc_inv(x, y):
    lon = np.degrees(x / R_MERC)
    lat = np.degrees(2.0 * np.arctan(np.exp(y / R_MERC)) - np.pi / 2.0)
    return lon, lat


def read_shp_records(path):
    """Lee el .shp -> lista (alineada al DBF) de records; cada record es una
    lista de anillos (arrays (n,2) en grados lon/lat). Record no-poligono -> []."""
    records = []
    with open(path, 'rb') as f:
        f.seek(100)
        while True:
            rh = f.read(8)
            if len(rh) < 8:
                break
            _, clen = struct.unpack('>ii', rh)  # recnum, content length (16-bit words)
            content = f.read(clen * 2)
            shp_type = struct.unpack('<i', content[:4])[0]
            if shp_type == 5:  # polygon
                nparts, npts = struct.unpack('<ii', content[36:44])
                parts = struct.unpack('<%di' % nparts, content[44:44 + 4 * nparts])
                pts_off = 44 + 4 * nparts
                pts = np.array(struct.unpack('<%dd' % (2 * npts),
                               content[pts_off:pts_off + 16 * npts])).reshape(-1, 2)
                lon, lat = merc_inv(pts[:, 0], pts[:, 1])
                xy = np.column_stack([lon, lat])
                idx = list(parts) + [npts]
                rings = [xy[idx[i]:idx[i + 1]] for i in range(nparts)]
                records.append(rings)
            else:
                records.append([])
    return records


def read_dbf_codes(path, field='ID'):
    """Lee el campo `field` del DBF -> lista de codigos DIVIPOLA (int) por record."""
    with open(path, 'rb') as f:
        hdr = f.read(32)
        nrec, hlen, rlen = struct.unpack('<Ihh', hdr[4:12])
        fields, pos = [], 32
        while True:
            fd = f.read(32)
            if fd[0:1] == b'\x0d':
                break
            name = fd[0:11].split(b'\x00')[0].decode('latin1')
            flen = fd[16]
            fields.append((name, flen))
            pos += 32
            if pos > hlen:
                break
        f.seek(hlen)
        codes = []
        for _ in range(nrec):
            rec = f.read(rlen)
            off = 1
            code = None
            for name, flen in fields:
                val = rec[off:off + flen].decode('latin1').strip()
                if name == field:
                    code = int(val) if val.isdigit() else None
                off += flen
            codes.append(code)
    return codes


def inverse_scaler(z, key):
    return z * SCALERS[key]['scale'] + SCALERS[key]['mean']


def main():
    print("Leyendo shapefile + dbf...", flush=True)
    records = read_shp_records(SHP)
    codes = read_dbf_codes(DBF, 'ID')
    print(f"  {len(records)} records shp, {len(codes)} codes dbf", flush=True)
    assert len(records) == len(codes), "shp/dbf desalineados"

    # --- centroides oficiales por codigo (JSON DIVIPOLA) ---
    geo = json.loads(GEOJSON.read_text(encoding='utf-8'))
    j_codes = np.array([int(k) for k in geo.keys()])
    j_lat = np.array([geo[k]['lat'] for k in geo.keys()])
    j_lon = np.array([geo[k]['lon'] for k in geo.keys()])

    # --- datos NObSP W1 (lat/lon/glat/y por paciente) ---
    # Cache local para no depender de los HDF5 de la cohorte en cada corrida.
    cache = OUTPUT_DIR / "geo_patient_cache.npz"
    data = np.load(OUTPUT_DIR / "demo_explainer.npz", allow_pickle=True)
    glat = data['contributions'][:, LAT_COL, CANCER]
    try:
        import r2_calib as r2_calib
        _, X_demo, y = r2_calib.load_r2_calib(str(OUTPUT_DIR), str(V4_DATA_DIR), need_events=False)
        X_demo = X_demo.astype(np.float32)
        lat = inverse_scaler(X_demo[:, LAT_COL], 'lat')
        lon = inverse_scaler(X_demo[:, LON_COL], 'lon')
        y = y.astype(int)
        np.savez(cache, lat=lat, lon=lon, glat=glat, y=y)
        print(f"  cargado de los HDF5 + cacheado en {cache.name}", flush=True)
    except (IndexError, FileNotFoundError, OSError):
        if not cache.exists():
            raise SystemExit(
                "ERROR: los HDF5 de la cohorte no estan disponibles y no hay cache "
                f"({cache}). Corre una vez con los HDF5 accesibles para generar el cache.")
        c = np.load(cache)
        lat, lon, glat, y = c['lat'], c['lon'], c['glat'], c['y'].astype(int)
        print(f"  HDF5 no disponibles -> usando cache {cache.name}", flush=True)
    n_pos = int(y.sum())
    print(f"  n={len(y)}, cancer={n_pos}", flush=True)

    # --- asignar cada paciente al codigo por centroide mas cercano ---
    d2 = (lat[:, None] - j_lat[None, :]) ** 2 + (lon[:, None] - j_lon[None, :]) ** 2
    nn = np.argmin(d2, axis=1)
    pat_code = j_codes[nn]

    # media de g_lat por codigo + flag de municipio con cancer
    agg = {}
    has_cancer = {}
    for c in np.unique(pat_code):
        m = pat_code == c
        agg[int(c)] = float(glat[m].mean())
        has_cancer[int(c)] = bool(y[m].sum() > 0)
    print(f"  municipios con dato: {len(agg)}", flush=True)

    cmax = float(np.max(np.abs(list(agg.values())))) or 1.0
    norm = Normalize(vmin=-cmax, vmax=cmax)
    cmap = plt.get_cmap('RdBu_r')

    # --- construir patches por record ---
    fig, ax = plt.subplots(figsize=(8.5, 10.5))
    patches, facecolors = [], []
    cancer_rings = []
    for rings, code in zip(records, codes):
        if not rings:
            continue
        verts, codes_p = [], []
        for ring in rings:
            verts.extend(ring.tolist())
            verts.append(ring[0].tolist())
            codes_p.extend([MPath.MOVETO] + [MPath.LINETO] * (len(ring) - 1) + [MPath.CLOSEPOLY])
        path = MPath(verts, codes_p)
        val = agg.get(code, None)
        if val is None:
            facecolors.append((0.93, 0.93, 0.93, 1.0))  # gris claro: sin dato
        else:
            facecolors.append(cmap(norm(val)))
            if has_cancer.get(code):
                cancer_rings.extend(rings)
        patches.append(path)

    pc = PatchCollection([PathPatch(p) for p in patches], match_original=False,
                         facecolors=facecolors, edgecolors=(0.5, 0.5, 0.5, 0.25),
                         linewidths=0.25, zorder=1, rasterized=True)
    ax.add_collection(pc)

    # contorno sutil de municipios con casos de cancer (sin punticos)
    if cancer_rings:
        lc = LineCollection([r for r in cancer_rings], colors=(0, 0, 0, 0.55),
                            linewidths=0.7, zorder=2)
        ax.add_collection(lc)

    ax.set_xlim(-79.5, -66.8)
    ax.set_ylim(-4.5, 13.2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Longitude ($^\\circ$W)', fontsize=18)
    ax.set_ylabel('Latitude ($^\\circ$N)', fontsize=18)
    ax.set_title('Latitude NObSP contribution $g_{\\mathrm{lat}}$ by municipality',
                 fontsize=19, fontweight='bold', pad=12)
    ax.tick_params(labelsize=14)

    fig.subplots_adjust(left=0.11, right=0.97, top=0.94, bottom=0.15)
    cax = fig.add_axes([0.30, 0.038, 0.42, 0.018])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation='horizontal')
    cb.set_label('Mean NObSP contribution to cancer class', fontsize=15)
    cb.ax.tick_params(labelsize=12)

    for ext in ('png', 'pdf'):
        fig.savefig(OUTPUT_DIR / f"pub_w1_geo_choropleth.{ext}", dpi=200, bbox_inches='tight')
    print("guardado pub_w1_geo_choropleth.png/pdf", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
