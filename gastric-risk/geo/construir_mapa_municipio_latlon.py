"""
Construye el mapa DIVIPOLA -> (lat, lon) para los municipios del dataset SGSSS.

REPRODUCES IN THE THESIS
    Builds the DIVIPOLA municipality to latitude/longitude map from DANE open data.

PROVENANCE
    Copied verbatim from the working tree of the thesis. The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md at the
    repository root.

--------------------------------------------------------------------------

Fuente bibliografica:
    DANE (Departamento Administrativo Nacional de Estadistica).
    "DIVIPOLA: Codigos municipios geolocalizados." Datos Abiertos Colombia.
    URL: https://www.datos.gov.co/Mapas-Nacionales/DIVIPOLA-C-digos-municipios-geolocalizados/vafm-j2df
    Endpoint Socrata JSON: https://www.datos.gov.co/resource/vafm-j2df.json

Salidas:
    mapa_municipio_latlon.json  -- dict {divipola_str: {lat, lon, nom_mpio, nom_dpto}}
    cobertura_report.txt        -- diagnostico de matching contra metadata V3.2
    divipola_oficial_raw.json   -- snapshot bruto del DANE (para trazabilidad)

Uso:
    python construir_mapa_municipio_latlon.py

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

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

OUT_DIR = Path(__file__).parent
SOCRATA_URL = "https://www.datos.gov.co/resource/vafm-j2df.json"
METADATA_V32 = Path(r"./batches/Datos_Modelo/metadata_20260506_120925.json")


def descargar_divipola_oficial() -> list[dict]:
    """Descarga el dataset completo del DANE via Socrata API."""
    print(f"[1/4] Descargando DIVIPOLA oficial desde {SOCRATA_URL} ...")
    url = f"{SOCRATA_URL}?$limit=5000"
    req = urllib.request.Request(url, headers={"User-Agent": "PhD-Research-Bot/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode("utf-8"))
    print(f"      -> {len(data)} municipios oficiales descargados")
    return data


def cargar_codigos_dataset() -> list[int]:
    """Lee los DIVIPOLA codes presentes en el dataset V3.2."""
    print(f"[2/4] Leyendo {METADATA_V32.name} ...")
    with open(METADATA_V32, "r", encoding="utf-8") as f:
        meta = json.load(f)
    codigos = meta["encoders"]["Municipio"]["classes"]
    print(f"      -> {len(codigos)} municipios en el dataset SGSSS")
    return codigos


def construir_mapa(oficial: list[dict], codigos_ds: list[int]) -> tuple[dict, dict]:
    """
    Match por cod_mpio. Devuelve (mapa, faltantes_info).
    """
    print(f"[3/4] Matching {len(codigos_ds)} codigos dataset vs {len(oficial)} oficiales ...")
    # Indexar oficial por cod_mpio (entero)
    idx_oficial = {int(r["cod_mpio"]): r for r in oficial if "cod_mpio" in r}

    mapa = {}
    faltantes = []
    for cod in codigos_ds:
        cod_int = int(cod)
        if cod_int in idx_oficial:
            r = idx_oficial[cod_int]
            mapa[str(cod_int)] = {
                "lat": float(r["latitud"]),
                "lon": float(r["longitud"]),
                "nom_mpio": r.get("nom_mpio", ""),
                "nom_dpto": r.get("nom_dpto", ""),
                "cod_dpto": int(r.get("cod_dpto", -1)),
            }
        else:
            faltantes.append(cod_int)

    info = {
        "n_dataset": len(codigos_ds),
        "n_oficial": len(idx_oficial),
        "n_match": len(mapa),
        "n_faltantes": len(faltantes),
        "cobertura_pct": 100.0 * len(mapa) / len(codigos_ds),
        "faltantes": faltantes,
    }
    print(f"      -> match {info['n_match']}/{info['n_dataset']} "
          f"({info['cobertura_pct']:.2f}% cobertura), {info['n_faltantes']} faltantes")
    return mapa, info


def guardar_outputs(oficial, mapa, info) -> None:
    print(f"[4/4] Guardando outputs en {OUT_DIR} ...")
    with open(OUT_DIR / "divipola_oficial_raw.json", "w", encoding="utf-8") as f:
        json.dump(oficial, f, ensure_ascii=False, indent=2)

    with open(OUT_DIR / "mapa_municipio_latlon.json", "w", encoding="utf-8") as f:
        json.dump(mapa, f, ensure_ascii=False, indent=2)

    lat_lons = [(v["lat"], v["lon"]) for v in mapa.values()]
    lats = [x[0] for x in lat_lons]
    lons = [x[1] for x in lat_lons]
    report_lines = [
        "Reporte de cobertura DIVIPOLA -> (lat, lon)",
        "===========================================",
        f"Fuente: DANE Datos Abiertos Colombia",
        f"Endpoint: {SOCRATA_URL}",
        "",
        f"Municipios en dataset SGSSS:           {info['n_dataset']}",
        f"Municipios oficiales DANE (vigentes):  {info['n_oficial']}",
        f"Match exitoso:                         {info['n_match']} ({info['cobertura_pct']:.2f}%)",
        f"Faltantes (sin coordenada):            {info['n_faltantes']}",
        "",
        f"Rango lat: [{min(lats):.4f}, {max(lats):.4f}]",
        f"Rango lon: [{min(lons):.4f}, {max(lons):.4f}]",
        "",
    ]
    if info["faltantes"]:
        report_lines.append("Codigos DIVIPOLA faltantes en el oficial:")
        report_lines.extend(f"  - {c}" for c in info["faltantes"])
    with open(OUT_DIR / "cobertura_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"      -> mapa_municipio_latlon.json ({len(mapa)} entradas)")
    print(f"      -> cobertura_report.txt")
    print(f"      -> divipola_oficial_raw.json")


def main():
    oficial = descargar_divipola_oficial()
    codigos_ds = cargar_codigos_dataset()
    mapa, info = construir_mapa(oficial, codigos_ds)
    guardar_outputs(oficial, mapa, info)
    print()
    print(f"Listo. Cobertura {info['cobertura_pct']:.2f}%, "
          f"{info['n_faltantes']} faltantes.")
    return 0 if info["cobertura_pct"] > 95.0 else 1


if __name__ == "__main__":
    sys.exit(main())
