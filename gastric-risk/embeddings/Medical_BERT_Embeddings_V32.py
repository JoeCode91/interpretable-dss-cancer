"""
Medical BERT Embeddings V3.2 -- A2 + A3 + A4 + cambio de modelo a espanol clinico

REPRODUCES IN THE THESIS
    Computes the BSC-Bio-EHR-ES embeddings of the CIE-10 and CUPS code descriptions.
    Section 4.2, and the third column of the embedding comparison table.

PROVENANCE
    Copied verbatim from the working tree of the thesis. The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md at the
    repository root.

--------------------------------------------------------------------------
Medical BERT Embeddings V3.2 -- A2 + A3 + A4 + cambio de modelo a espanol clinico

Cambios respecto a V2:
  A2: usar DESCRIPCIONES reales (no codigos literales) -> mapa_*.json
  A3: media-pooling sobre tokens validos (attention_mask) de la media de
      las ultimas 4 capas. Estrategia estandar SimCSE/Sentence-BERT: el
      [CLS] de RoBERTa sin fine-tuning concentra todas las representaciones
      en una zona estrecha del espacio (sims off-diag > 0.94, no
      discriminativo). Mean-pool da sims off-diag 0.7-0.85 (discrimina
      sinonimos cl\'inicos vs no-relacionados; verificado en sesion 24).
  A4: L2 normalize de cada embedding antes de guardar
  Modelo: PlanTL-GOB-ES/bsc-bio-ehr-es (RoBERTa biomedico clinico en espanol)
          en vez de dmis-lab/biobert-base-cased-v1.2

Salida (formato compatible con V2):
  bert_embeddings_diag_v32_<TS>.npz  con keys:
    embeddings (N, 768) float32 L2-normalizado
    terms (N,) -- codigos originales (con .0 si aplica)

Autor: Johan S. Pena-Campos
Fecha: 2026-05-06 (sesion 24)
"""

# Rutas configurables por variable de entorno. En el arbol de trabajo de la tesis
# apuntaban a directorios locales del autor; aqui se parametrizan para que el
# script sea ejecutable fuera de esa maquina.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# ==================== CONFIG ====================

MODEL_NAME = "PlanTL-GOB-ES/bsc-bio-ehr-es"   # RoBERTa biomedico clinico en espanol
MAX_LENGTH = 64                                # descripciones reales son cortas
BATCH_SIZE = 32

CODES_DIR = Path(os.environ.get("EMBEDDINGS_DIR", "./embeddings"))
MAP_DIR   = Path(os.environ.get("SETUP_DIR", "."))
OUT_DIR   = Path(os.environ.get("EMBEDDINGS_DIR", "./embeddings"))  # mismo lugar que V2


# ==================== UTILS ====================

def clean_code(c):
    """Quita .0 trailing (artefacto de pandas float)."""
    c = str(c).strip()
    return c[:-2] if c.endswith('.0') else c


def load_codes_and_descriptions(codes_file, map_file):
    """
    Lee la lista de codigos del .txt y construye la lista de descripciones
    (o codigos crudos si no hay descripcion).
    Devuelve: (raw_codes, descriptions, fallback_count)
    """
    raw = [l.strip() for l in open(codes_file, encoding='utf-8').read().split() if l.strip()]
    mapa = json.load(open(map_file, encoding='utf-8'))
    descriptions = []
    fallback = 0
    for c in raw:
        cleaned = clean_code(c)
        if cleaned in mapa and mapa[cleaned] != cleaned:
            descriptions.append(mapa[cleaned])
        else:
            descriptions.append(cleaned)  # fallback al codigo literal
            fallback += 1
    return raw, descriptions, fallback


def load_model(device):
    print(f"\nCargando modelo: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parametros: {n_params:,}", flush=True)
    print(f"  Capas:      {model.config.num_hidden_layers}", flush=True)
    print(f"  Dim hidden: {model.config.hidden_size}", flush=True)
    return tokenizer, model


def embed_batch(texts, tokenizer, model, device):
    """
    A3: mean-pooling sobre tokens validos (attention_mask), de la media
        de las ultimas 4 capas.
    A4: L2 normalize.
    Retorna: (B, 768) numpy float32 L2-normalized.
    """
    inputs = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(device)
    attn_mask = inputs['attention_mask'].unsqueeze(-1).float()  # (B, T, 1)

    with torch.no_grad():
        out = model(**inputs)

    # out.hidden_states: tupla (n_layers+1) tensores (B, T, 768)
    last4 = torch.stack(out.hidden_states[-4:], dim=0)   # (4, B, T, 768)
    mean_layers = last4.mean(dim=0)                      # (B, T, 768)
    masked_sum = (mean_layers * attn_mask).sum(dim=1)    # (B, 768)
    counts = attn_mask.sum(dim=1).clamp(min=1)           # (B, 1)
    pooled = masked_sum / counts                         # (B, 768)
    normalized = F.normalize(pooled, p=2, dim=1)
    return normalized.cpu().numpy().astype(np.float32)


def generate_embeddings(raw_codes, descriptions, tokenizer, model, device, tipo):
    print(f"\n[{tipo}] Generando embeddings para {len(raw_codes):,} codigos", flush=True)
    n_batches = (len(descriptions) + BATCH_SIZE - 1) // BATCH_SIZE
    out = np.zeros((len(descriptions), model.config.hidden_size), dtype=np.float32)
    t0 = time.time()
    for bi in range(n_batches):
        s = bi * BATCH_SIZE
        e = min(s + BATCH_SIZE, len(descriptions))
        out[s:e] = embed_batch(descriptions[s:e], tokenizer, model, device)
        if (bi + 1) % 20 == 0 or (bi + 1) == n_batches:
            elapsed = time.time() - t0
            rate = e / elapsed if elapsed > 0 else 0
            eta = (len(descriptions) - e) / rate if rate > 0 else 0
            print(f"  Batch {bi+1:>4}/{n_batches} | {e:>6}/{len(descriptions):,} | "
                  f"rate {rate:>5.1f} cod/s | ETA {eta/60:>5.1f} min", flush=True)
    elapsed = time.time() - t0
    print(f"[{tipo}] OK -- shape {out.shape} | {elapsed/60:.1f} min "
          f"({len(descriptions)/elapsed:.1f} cod/s)", flush=True)
    return out


def save_npz(emb, terms, tipo, ts):
    out_file = OUT_DIR / f"bert_embeddings_{tipo}_v32_{ts}.npz"
    np.savez_compressed(
        out_file,
        embeddings=emb.astype(np.float32),
        terms=np.array(terms),
        timestamp=ts,
        modelo=MODEL_NAME,
        version="V3.2 (A2+A3+A4 + BSC-Bio-EHR-ES)",
    )
    size_mb = out_file.stat().st_size / 1e6
    print(f"  Guardado: {out_file.name} ({size_mb:.1f} MB)", flush=True)
    return out_file


def verify(emb, raw, descs, tipo):
    norms = np.linalg.norm(emb, axis=1)
    print(f"\n[{tipo}] Verificacion:")
    print(f"  Shape          : {emb.shape}")
    print(f"  Norma media    : {norms.mean():.4f} (debe ser ~1.0 por L2)")
    print(f"  Norma std      : {norms.std():.4e}")
    print(f"  Min/Max norma  : {norms.min():.4f} / {norms.max():.4f}")
    n_unique = len(set(map(tuple, emb)))
    print(f"  Embeddings unicos: {n_unique}/{len(emb)}")
    print(f"  Ejemplos:")
    for i in range(min(3, len(raw))):
        prev = ', '.join(f"{v:+.3f}" for v in emb[i, :3])
        print(f"    {raw[i]:>10} ({descs[i][:50]:<50}) -> [{prev}, ...]")


# ==================== MAIN ====================

def main():
    print("=" * 72)
    print("EMBEDDINGS V3.2 -- BSC-Bio-EHR-ES con descripciones + A3 + A4")
    print("=" * 72)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU   : {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")

    tokenizer, model = load_model(device)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Diagnosticos ---
    raw_d, desc_d, fb_d = load_codes_and_descriptions(
        CODES_DIR / "codigos_diagnosticos_completos.txt",
        MAP_DIR / "mapa_cie10.json",
    )
    print(f"\nDiagnosticos: {len(raw_d)} codigos | "
          f"con descripcion: {len(raw_d) - fb_d} ({100*(len(raw_d)-fb_d)/len(raw_d):.1f}%)")

    emb_d = generate_embeddings(raw_d, desc_d, tokenizer, model, device, "DIAG")
    verify(emb_d, raw_d, desc_d, "DIAG")
    save_npz(emb_d, raw_d, "diag", ts)

    # --- Procedimientos ---
    raw_p, desc_p, fb_p = load_codes_and_descriptions(
        CODES_DIR / "codigos_procedimientos_completos.txt",
        MAP_DIR / "mapa_cups.json",
    )
    print(f"\nProcedimientos: {len(raw_p)} codigos | "
          f"con descripcion: {len(raw_p) - fb_p} ({100*(len(raw_p)-fb_p)/len(raw_p):.1f}%)")

    emb_p = generate_embeddings(raw_p, desc_p, tokenizer, model, device, "PROC")
    verify(emb_p, raw_p, desc_p, "PROC")
    save_npz(emb_p, raw_p, "proc", ts)

    print("\nDONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
