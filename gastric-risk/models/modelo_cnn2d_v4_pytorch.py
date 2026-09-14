"""
CNN2DModelV4 -- Camino 2 (refinamiento propuesta Alex, reunion 18 May 2026).

REPRODUCES IN THE THESIS
    The CNN-2D architecture adopted as the final model. Section 4.4.

PROVENANCE
    Copied verbatim from the working tree of the thesis. The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md at the
    repository root.

--------------------------------------------------------------------------
CNN2DModelV4 -- Camino 2 (refinamiento propuesta Alex, reunion 18 May 2026).

Diferencias respecto a `modelo_cnn2d_pytorch.py` (Camino 1):

  Input shape:
    Camino 1: $X_{vent} \\in \\mathbb{R}^{B\\times 10\\times 40\\times 1541}$ con
              mascara $X_{masc} \\in \\{0,1\\}^{B\\times 10}$.
    V4:       $X_{events} \\in \\mathbb{R}^{B\\times 1541\\times 100}$ tratado como
              "imagen" de un solo canal: $(B, 1, 1541, 100)$. Sin masking
              (los pacientes con menos eventos tienen zero-padding al final
              y la CNN aprende a ignorarlo).

  Arquitectura:
    - Conv 2D 3x3 entre bloques (no Conv 1x1 de reduccion como Camino 1)
    - MaxPool 2x2 entre bloques para controlar memoria (dims iniciales
      muy grandes: 1541x100). Sin pool, las activaciones intermedias
      excederian 8 GB de VRAM en RTX 3060 Ti.
    - Global Average Pool al final (sin masking).
    - Demograficos pasan por una rama densa pequena y se concatenan.

  Demograficos V4 (15 dims vs 30 en V3.2):
    - EPS one-hot (10) + Sexo one-hot (2) + lat StandardScaler (1) +
      lon StandardScaler (1) + edad StandardScaler (1) = 15.

  NObSP compatibility:
    - `forward_conv(x_events)` devuelve el feature map pre-pool
      $(B, c_{pool}, H', W')$ para perturbar en wrappers W3 (post-conv).
    - `forward_head(pooled, x_demograficos)` desde el vector post-pool
      $(B, c_{pool})$ para wrappers W1 (demo) y W4 (post-pool).

Autor: Johan S. Pena-Campos, Cap. 4 Camino 2 V4
Fecha: 2026-05-18 (sesion 29)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN2DModelV4(nn.Module):
    """CNN 2D sobre tensor (1541, 100) por paciente, sin masking.

    Forward signature simplificado (sin mascara):
        forward(x_events, x_demograficos) -> torch.Tensor (B, 1)
    """

    def __init__(
        self,
        n_features: int = 1541,
        n_eventos: int = 100,
        n_demograficos: int = 15,
        c1: int = 32,
        c2: int = 64,
        c3: int = 128,
        c_pool: int = 64,
        n_dense: int = 64,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_eventos = n_eventos
        self.n_demograficos = n_demograficos
        self.c_pool = c_pool

        # Bloque 1: (B, 1, 1541, 100) -> (B, c1, 770, 50)
        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(c1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bloque 2: -> (B, c2, 385, 25)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(c2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bloque 3: -> (B, c3, 192, 12)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(c3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bloque 4: -> (B, c_pool, 192, 12)  (sin pool, sin BN: va directo al pool global)
        self.conv4 = nn.Conv2d(c3, c_pool, kernel_size=3, padding=1, bias=True)

        # Rama demograficos
        self.demo_dense = nn.Linear(n_demograficos, n_dense)
        self.demo_dropout = nn.Dropout(dropout)

        # Cabeza densa: concat(pool c_pool, demo n_dense) -> n_dense -> 1
        self.dense1 = nn.Linear(c_pool + n_dense, n_dense)
        self.dropout1 = nn.Dropout(dropout)
        self.output_layer = nn.Linear(n_dense, 1)

    def forward_conv(self, x_events: torch.Tensor) -> torch.Tensor:
        """Hasta la ultima conv: $(B, 1541, 100) \\to (B, c_{pool}, H', W')$.

        Util para NObSP donde queremos perturbar el activation map
        post-conv pre-pool (wrapper W3).
        """
        # Agregar canal: (B, 1541, 100) -> (B, 1, 1541, 100)
        x = x_events.unsqueeze(1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = F.relu(self.conv4(x))
        return x  # (B, c_pool, ~192, ~12)

    def forward_head(
        self, pooled: torch.Tensor, x_demograficos: torch.Tensor
    ) -> torch.Tensor:
        """Desde el vector post-pool hasta la salida (sigmoide).

        Args:
            pooled:         (B, c_pool)
            x_demograficos: (B, 15)
        """
        demo = F.relu(self.demo_dense(x_demograficos))
        demo = self.demo_dropout(demo)

        fused = torch.cat([pooled, demo], dim=-1)  # (B, c_pool + n_dense)
        h = F.relu(self.dense1(fused))
        h = self.dropout1(h)
        return torch.sigmoid(self.output_layer(h))

    def forward(
        self,
        x_events: torch.Tensor,
        x_demograficos: torch.Tensor,
    ) -> torch.Tensor:
        feat_map = self.forward_conv(x_events)
        # Global Average Pool 2D (sin masking)
        pooled = F.adaptive_avg_pool2d(feat_map, 1).squeeze(-1).squeeze(-1)
        return self.forward_head(pooled, x_demograficos)


if __name__ == "__main__":
    # Sanity check: forward dummy + conteo de parametros + verificacion VRAM
    import sys

    model = CNN2DModelV4()
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CNN2DModelV4: {total:,} parametros ({train:,} entrenables)")

    print("\nDesglose por modulo:")
    for name, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters())
        if n:
            print(f"  {name:<20s} {n:>10,}")

    print("\nReferencia comparativa:")
    print("  V3.2 BiLSTM+MHA:  4,521,665 params (4.5M)")
    print("  CNN 2D Camino 1:  1,365,249 params (1.36M)")
    print(f"  CNN 2D V4 (este): {total:>10,} params")

    # Dummy forward CPU
    B = 4
    x_e = torch.randn(B, 1541, 100)
    x_d = torch.randn(B, 15)

    model.eval()
    with torch.no_grad():
        p = model(x_e, x_d)
        feat = model.forward_conv(x_e)

    print(f"\nDummy forward (CPU):")
    print(f"  X_events:       {tuple(x_e.shape)}")
    print(f"  X_demograficos: {tuple(x_d.shape)}")
    print(f"  feat_map (pre-pool): {tuple(feat.shape)}")
    print(f"  output:         {tuple(p.shape)}  (rango {p.min():.4f}, {p.max():.4f})")
    assert p.shape == (B, 1)
    assert not torch.isnan(p).any()
    print("\n  OK: shapes y rango correctos, sin NaN")

    # Sanity de memoria GPU si esta disponible
    if torch.cuda.is_available():
        print("\nSanity check GPU (batch 128 que es el de training):")
        model_gpu = CNN2DModelV4().cuda()
        x_e_gpu = torch.randn(128, 1541, 100, device='cuda')
        x_d_gpu = torch.randn(128, 15, device='cuda')
        torch.cuda.reset_peak_memory_stats()
        model_gpu.eval()
        with torch.no_grad():
            out = model_gpu(x_e_gpu, x_d_gpu)
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"  forward batch 128: peak VRAM {peak_mb:.1f} MB")
        print(f"  output shape: {tuple(out.shape)}")
        # Backward
        torch.cuda.reset_peak_memory_stats()
        model_gpu.train()
        out = model_gpu(x_e_gpu, x_d_gpu)
        loss = out.mean()
        loss.backward()
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"  forward+backward batch 128: peak VRAM {peak_mb:.1f} MB")
    else:
        print("\nGPU no disponible, skip sanity check GPU")
