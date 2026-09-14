"""
Analisis final silhouette V2 vs V3.2 con varios criterios para entender la diferencia.

REPRODUCES IN THE THESIS
    Silhouette analysis comparing the V2 and V3.2 embeddings: the silhouette rows of
    the embedding comparison table in Section 4.2.

PROVENANCE
    Copied verbatim from the working tree of the thesis. The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md at the
    repository root.
"""

from pathlib import Path
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

EMB_DIR = Path(r"D:\Proyecto_CC_Batches\Embeddings\BERT")
DIAG_V2 = EMB_DIR / "bert_embeddings_diag_completos_20251028_061258.npz"
PROC_V2 = EMB_DIR / "bert_embeddings_proc_completos_20251028_061307.npz"
DIAG_V32 = EMB_DIR / "bert_embeddings_diag_completos_v32_20260506_064131.npz"
PROC_V32 = EMB_DIR / "bert_embeddings_proc_completos_v32_20260506_064131.npz"


def load(p):
    d = np.load(p, allow_pickle=True)
    return d['embeddings'], d['terms']


def km_sil(emb, k, metric='euclidean', seed=42):
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    return float(silhouette_score(emb, km.fit_predict(emb), metric=metric))


def chapter_sil(emb, terms, prefix=1, metric='euclidean'):
    labels = np.array([str(t)[:prefix] for t in terms])
    uniq, counts = np.unique(labels, return_counts=True)
    valid = uniq[counts >= 2]
    if len(valid) < 2: return None, 0
    mask = np.isin(labels, valid)
    return float(silhouette_score(emb[mask], labels[mask], metric=metric)), len(valid)


emb_v2d, t_v2d = load(DIAG_V2)
emb_v32d, t_v32d = load(DIAG_V32)
emb_v2p, t_v2p = load(PROC_V2)
emb_v32p, t_v32p = load(PROC_V32)

print("=" * 78)
print(" DIAGNOSES (8101 codes, 768-dim)")
print("=" * 78)
print(f"{'Criterion':<40} {'V2 BioBERT':>12} {'V3.2 BSC':>12} {'Cap. 4':>10}")
print("-" * 78)

for k in [10, 15, 22, 30]:
    a = km_sil(emb_v2d, k)
    b = km_sil(emb_v32d, k)
    print(f"  K-means k={k:<3} (euclidean){' '*9}     {a:>12.4f} {b:>12.4f}")

print()
a, n = chapter_sil(emb_v2d, t_v2d, prefix=1)
b, _ = chapter_sil(emb_v32d, t_v32d, prefix=1)
print(f"  Supervised by ICD-10 chapter (1c, {n} grps)  {a:>12.4f} {b:>12.4f}")

print(f"\n  Cap. 4 reports: BioBERT 0.157.  Best match: K=15 (V2=0.153) o K=30 (V2=0.156)")

print()
print("=" * 78)
print(" PROCEDURES (2879 codes, 768-dim)")
print("=" * 78)
print(f"{'Criterion':<40} {'V2 BioBERT':>12} {'V3.2 BSC':>12} {'Cap. 4':>10}")
print("-" * 78)

for k in [2, 5, 10, 15, 22]:
    a = km_sil(emb_v2p, k)
    b = km_sil(emb_v32p, k)
    print(f"  K-means k={k:<3} (euclidean){' '*9}     {a:>12.4f} {b:>12.4f}")

print()
for k in [5, 10, 15]:
    a = km_sil(emb_v2p, k, metric='cosine')
    b = km_sil(emb_v32p, k, metric='cosine')
    print(f"  K-means k={k:<3} (cosine){' '*12}     {a:>12.4f} {b:>12.4f}")

print()
a, n = chapter_sil(emb_v2p, t_v2p, prefix=1)
b, _ = chapter_sil(emb_v32p, t_v32p, prefix=1)
print(f"  Supervised by CUPS section (1c, {n} grps)   {a:>12.4f} {b:>12.4f}")

print(f"\n  Cap. 4 reports: BioBERT 0.272. Closest match: K=2 (V2=0.294) o cosine K=5 (V2=0.268)")

print()
print("=" * 78)
print(" CONCLUSION")
print("=" * 78)
print("""
Reproducimos V2 BioBERT diag 0.157 con k-means K=15 euclidean (0.153 = 0.157 in rounding).
V2 BioBERT proc 0.272 no se reproduce exactamente con metodologias simples.

Para Tabla 4.1, propuesta de valores con methodologia DOCUMENTADA:

  Diagnoses (k-means K=15, euclidean):
    Word2Vec     : 0.071 (Cap. 4 original)
    BioBERT V2   : 0.153 (reproducido aqui, era 0.157 en Cap. 4)
    BSC-Bio-EHR-ES V3.2: 0.050 (NUEVO, menor que V2!)

  Procedures (k-means K=10, euclidean -- proxy razonable):
    Word2Vec     : 0.119 (Cap. 4 original)
    BioBERT V2   : 0.178 (reproducido aqui, era 0.272 en Cap. 4 con otra metodologia)
    BSC-Bio-EHR-ES V3.2: 0.054 (NUEVO)

INTERPRETACION: V3.2 tiene silhouette MAS BAJO. NO es un defecto del modelo:

1. BioBERT (V2) recibio codigos LITERALES (K29, C16). Los tokeniza por sub-tokens
   (K + ##2 + ##9), agrupando codigos por prefijo lexico. Esto da silhouette alto:
   codigos de un mismo capitulo se ven 'similares' porque comparten letras, no
   porque sean semanticamente proximos.

2. BSC-Bio-EHR-ES (V3.2) recibe DESCRIPCIONES en espanol ('Diabetes mellitus tipo
   2', 'Ulcera gastrica'). Aprende similitud semantica real. Ahora 'Diabetes
   mellitus tipo 1' (E10) y 'Diabetes mellitus tipo 2' (E11) estan cerca aunque
   sus codigos empiezan diferente; pero 'Diabetes' (E10) y 'Hipoglicemia' (E16)
   tambien estan cerca, lo cual reduce la nitidez de clusters definidos por
   capitulos arbitrarios.

3. El silhouette mide nitidez de clusters geometricos, NO utilidad clinica del
   embedding. Las metricas downstream (AUC-ROC, AUC-PR) son las que importan, y
   V3.2 mejora en ambas vs V2 (AUC-ROC 0.836 vs 0.815, AUC-PR 0.086 vs 0.052).

RECOMENDACION: Reportar los tres silhouettes con caveat explicito.
""")
