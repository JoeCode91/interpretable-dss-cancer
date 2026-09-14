"""
Verificacion del insight socioeconomico de EPS (NObSP).

REPRODUCES IN THE THESIS
    Tests whether the high cancer rate of one insurer reflects a genuine access effect
    or is a proxy for age and region. Supports the detection-artifact discussion of
    Section 4.9.

PROVENANCE
    Copied verbatim from the working tree of the thesis. The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md at the
    repository root.

--------------------------------------------------------------------------
Verificacion del insight socioeconomico de EPS (NObSP).

Pregunta: la tasa alta de cancer de EPS017 (Famisanar, 58% de casos) refleja un
efecto socioeconomico/acceso GENUINO del asegurador, o es un PROXY de edad +
region (Bogota/Cundinamarca, alta incidencia andina)?

Cruza por EPS: tasa cancer, edad media, %>=60, % Bogota (11001), % Cundinamarca
(dept 25), y la tasa ESTRATIFICADA (EPS017 vs resto dentro de >=60 y dentro de
Bogota). Si el exceso de EPS017 sobrevive la estratificacion -> efecto genuino;
si desaparece -> confound de edad/region.

Solo datos crudos del cohorte (MatChar_Gen). No nombra aseguradoras en salidas.
"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd

MATCHAR = Path(r"D:\Proyecto_CC_Batches\MatChar")


def load_gen():
    files = sorted(MATCHAR.glob("MatChar_Gen_batch_*_7_2015_to_12_2016.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['Municipio'] = df['Municipio'].astype(str).str.replace(r'\.0$', '', regex=True)
    df['dept'] = df['Municipio'].str.zfill(5).str[:2]
    return df


def main():
    print("Cargando Gen...", flush=True)
    gen = load_gen()
    print(f"  {len(gen):,} pacientes, {int(gen['Estomago_x'].sum())} cancer", flush=True)

    # --- perfil por EPS ---
    g = gen.groupby('CodigoAdmin')
    prof = pd.DataFrame({
        'n': g.size(),
        'cancer': g['Estomago_x'].sum(),
        'rate_pct': g['Estomago_x'].mean() * 100,
        'edad_media': g['Edad'].mean(),
        'pct_ge60': g.apply(lambda x: (x['Edad'] >= 60).mean() * 100, include_groups=False),
        'pct_bogota': g.apply(lambda x: (x['Municipio'].str.zfill(5) == '11001').mean() * 100, include_groups=False),
        'pct_cundi': g.apply(lambda x: (x['dept'] == '25').mean() * 100, include_groups=False),
    }).sort_values('cancer', ascending=False)
    print("\n=== PERFIL POR EPS (edad/region vs tasa) ===", flush=True)
    print(prof.to_string(float_format=lambda v: f"{v:.3f}"), flush=True)

    # correlaciones: tasa vs edad_media, tasa vs %bogota (ponderado por n no, simple por EPS)
    print("\n=== Correlacion (entre EPS) rate_pct vs ... ===", flush=True)
    for col in ['edad_media', 'pct_ge60', 'pct_bogota', 'pct_cundi']:
        r = prof['rate_pct'].corr(prof[col])
        print(f"  rate_pct vs {col}: r = {r:.3f}", flush=True)

    # --- estratificacion: EPS017 vs resto, dentro de subgrupos ---
    print("\n=== ESTRATIFICACION: tasa cancer EPS017 vs RESTO ===", flush=True)
    gen['is017'] = (gen['CodigoAdmin'] == 'EPS017')
    gen['ge60'] = gen['Edad'] >= 60
    gen['bog'] = gen['Municipio'].str.zfill(5) == '11001'
    gen['cundi_bog'] = gen['dept'].isin(['11', '25'])  # Bogota + Cundinamarca

    def rate(mask):
        sub = gen[mask]
        return len(sub), int(sub['Estomago_x'].sum()), sub['Estomago_x'].mean() * 100

    for label, m in [
        ("GLOBAL", np.ones(len(gen), bool)),
        (">=60 anios", gen['ge60'].values),
        ("<60 anios", (~gen['ge60']).values),
        ("Bogota", gen['bog'].values),
        ("Bogota+Cundinamarca", gen['cundi_bog'].values),
        ("fuera Bog+Cundi", (~gen['cundi_bog']).values),
    ]:
        n17, c17, r17 = rate(m & gen['is017'].values)
        nR, cR, rR = rate(m & (~gen['is017']).values)
        ratio = (r17 / rR) if rR > 0 else float('nan')
        print(f"  [{label:22s}] EPS017: {r17:.4f}% (n={n17:,}, ca={c17})  | "
              f"RESTO: {rR:.4f}% (n={nR:,}, ca={cR})  | ratio={ratio:.1f}x", flush=True)

    print("\nInterpretacion: si el ratio EPS017/resto se MANTIENE alto dentro de cada", flush=True)
    print("estrato (>=60, Bogota), el efecto EPS es independiente de edad/region", flush=True)
    print("(apoya socioeconomico/acceso); si CAE hacia 1x, es confound de edad/region.", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
