"""
parseo_fechas.py -- parseo de las fechas de evento.

REPRODUCES IN THE THESIS
    Nothing directly. Shared by the cohort builder: converts the date of each event into
    the temporal features "days since the first recorded event" and "days since the
    previous event" of the event vector, Section 4.2.

PROVENANCE
    Taken from the working tree that produced the reported results. The computational
    content is unchanged; comments were edited. See CHANGES-FROM-THESIS.md at the
    repository root.

--------------------------------------------------------------------------
Regla de parseo:
  - Parseo con pd.to_datetime(format=explicito, errors='coerce'): primero
    "%Y-%m-%d %H:%M:%S" (formato de MatChar_Fecha_*.csv), luego "%Y-%m-%d".
    Nada de inferencia de formato.
  - La fecha se convierte a dias (float) desde 1970-01-01.
  - Fecha no parseable (cadena no nula que no cumple ningun formato): el evento
    SE CONSERVA (no cambia que eventos entran ni su orden) y su fecha se imputa
    con la ultima fecha valida anterior del paciente (arrastre hacia delante);
    si no hay ninguna anterior, con la primera valida posterior. Si el paciente
    no tiene ninguna fecha valida, todas sus fechas valen 0 (A = B = 0) y se
    cuenta aparte.
  - Fecha nula (celda vacia) con diagnostico y procedimiento presentes: el
    evento se DESCARTA (pd.notna(f) es condicion de entrada). Se cuenta para el
    informe.
  - El orden de los eventos NO se toca (es el de la fuente). Si las fechas no
    son monotonas, el intervalo B puede salir negativo: se cuenta y se reporta.

AUTHORSHIP
    Johan S. Pena-Campos
    PhD thesis, "Interpretable Decision Support Systems for Early Cancer Diagnosis and
    Personalized Treatment", Universitat Politecnica de Catalunya and Pontificia
    Universidad Javeriana, 2026.

USE OF AI ASSISTANCE
    This file was written with the assistance of generative AI (Claude, Anthropic),
    to the author's specification, and reviewed, corrected and validated by the
    author, who takes full responsibility for it.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd

FECHA_FORMATOS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d")
_EPOCH = pd.Timestamp("1970-01-01")
_RANGO_ESPERADO = (pd.Timestamp("2015-07-01"), pd.Timestamp("2016-12-31 23:59:59"))

_CACHE: Dict[str, tuple] = {}


def nueva_calidad() -> Dict:
    return {
        "n_pacientes": 0,
        "n_eventos": 0,                         # eventos que entran al tensor (historia completa)
        "n_fecha_formato_principal": 0,
        "n_fecha_formato_solo_dia": 0,
        "n_fecha_invalida": 0,                  # no nula pero no parseable -> imputada
        "n_fecha_con_hora_no_cero": 0,
        "n_fecha_fuera_rango_2015_07_2016_12": 0,
        "n_pacientes_con_alguna_invalida": 0,
        "n_pacientes_sin_fecha_valida": 0,
        "n_eventos_descartados_fecha_nula": 0,  # no entran al tensor
        "n_intervalos_negativos": 0,
        "n_pacientes_no_monotonos": 0,
        "n_intervalos_cero": 0,
        "fecha_min": None,
        "fecha_max": None,
        "ejemplos_invalidas": [],
    }


def parsear_fecha(valor) -> tuple:
    """valor (cadena/obj no nulo) -> (dias float o nan, codigo)."""
    s = str(valor).strip()
    hit = _CACHE.get(s)
    if hit is not None:
        return hit
    res = (float("nan"), "invalida")
    for k, fmt in enumerate(FECHA_FORMATOS):
        ts = pd.to_datetime(s, format=fmt, errors="coerce")
        if not pd.isna(ts):
            res = ((ts - _EPOCH) / pd.Timedelta(days=1),
                   "principal" if k == 0 else "solo_dia")
            break
    _CACHE[s] = res
    return res


def fechas_paciente(raw: List, calidad: Dict) -> List[float]:
    """Lista de valores crudos (no nulos) de un paciente -> dias imputados."""
    n = len(raw)
    calidad["n_pacientes"] += 1
    calidad["n_eventos"] += n
    dias = np.full(n, np.nan, dtype=np.float64)
    hubo_invalida = False
    for i, v in enumerate(raw):
        d, cod = parsear_fecha(v)
        if cod == "principal":
            calidad["n_fecha_formato_principal"] += 1
        elif cod == "solo_dia":
            calidad["n_fecha_formato_solo_dia"] += 1
        else:
            calidad["n_fecha_invalida"] += 1
            hubo_invalida = True
            if len(calidad["ejemplos_invalidas"]) < 20:
                calidad["ejemplos_invalidas"].append(str(v)[:40])
            continue
        dias[i] = d
        if d != math.floor(d):
            calidad["n_fecha_con_hora_no_cero"] += 1
        ts = _EPOCH + pd.Timedelta(days=d)
        if ts < _RANGO_ESPERADO[0] or ts > _RANGO_ESPERADO[1]:
            calidad["n_fecha_fuera_rango_2015_07_2016_12"] += 1
        if calidad["fecha_min"] is None or d < calidad["fecha_min"]:
            calidad["fecha_min"] = d
        if calidad["fecha_max"] is None or d > calidad["fecha_max"]:
            calidad["fecha_max"] = d

    if hubo_invalida:
        calidad["n_pacientes_con_alguna_invalida"] += 1
        validos = np.flatnonzero(~np.isnan(dias))
        if len(validos) == 0:
            calidad["n_pacientes_sin_fecha_valida"] += 1
            dias[:] = 0.0
        else:
            # arrastre hacia delante; los iniciales toman la primera valida
            ultimo = dias[validos[0]]
            for i in range(n):
                if np.isnan(dias[i]):
                    dias[i] = ultimo
                else:
                    ultimo = dias[i]

    if n > 1:
        dif = np.diff(dias)
        nneg = int((dif < 0).sum())
        calidad["n_intervalos_negativos"] += nneg
        calidad["n_intervalos_cero"] += int((dif == 0).sum())
        if nneg:
            calidad["n_pacientes_no_monotonos"] += 1
    return dias.tolist()


def fusionar_calidad(a: Dict, b: Dict) -> Dict:
    out = dict(a)
    for k, v in b.items():
        if k in ("fecha_min",):
            out[k] = v if a[k] is None else (a[k] if v is None else min(a[k], v))
        elif k in ("fecha_max",):
            out[k] = v if a[k] is None else (a[k] if v is None else max(a[k], v))
        elif k == "ejemplos_invalidas":
            out[k] = (a[k] + v)[:20]
        else:
            out[k] = a[k] + v
    return out


def resumen_calidad(c: Dict) -> Dict:
    """Agrega fracciones legibles y convierte dias a fechas ISO."""
    r = dict(c)
    ne = max(c["n_eventos"], 1)
    np_ = max(c["n_pacientes"], 1)
    r["frac_eventos_fecha_invalida"] = c["n_fecha_invalida"] / ne
    r["frac_eventos_solo_dia"] = c["n_fecha_formato_solo_dia"] / ne
    r["frac_eventos_fuera_rango"] = c["n_fecha_fuera_rango_2015_07_2016_12"] / ne
    r["frac_pacientes_con_alguna_invalida"] = c["n_pacientes_con_alguna_invalida"] / np_
    r["frac_pacientes_no_monotonos"] = c["n_pacientes_no_monotonos"] / np_
    n_int = max(c["n_eventos"] - c["n_pacientes"], 1)
    r["frac_intervalos_negativos"] = c["n_intervalos_negativos"] / n_int
    r["frac_intervalos_cero"] = c["n_intervalos_cero"] / n_int
    tot_cand = c["n_eventos"] + c["n_eventos_descartados_fecha_nula"]
    r["frac_eventos_descartados_fecha_nula"] = c["n_eventos_descartados_fecha_nula"] / max(tot_cand, 1)
    for k in ("fecha_min", "fecha_max"):
        if c[k] is not None:
            r[k + "_iso"] = str((_EPOCH + pd.Timedelta(days=c[k])).date())
    r["regla_imputacion"] = ("fecha no parseable -> ultima fecha valida anterior del paciente "
                             "(si no hay, primera valida posterior; si ninguna, 0); "
                             "fecha nula -> evento descartado")
    return r
