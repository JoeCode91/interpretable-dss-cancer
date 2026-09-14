# nobsp

Nonlinear Oblique Subspace Projections — the interpretability method of **Chapter 3** of
the thesis, and the pruning experiments that extend it.

---

## What is here, and what is not

A distinction matters in this directory, and it is worth stating before anything else.

**The method is the thesis contribution.** NObSP decomposes a trained network's prediction
into functional components, one per input feature, by projecting obliquely in the space of
the penultimate representation. Orthogonal projection would not do, because the subspaces
that a network learns for different features overlap; the oblique projection isolates the
part attributable to one feature while accounting for the others. Chapter 3 develops this,
and three publications report it:

- IFAC World Congress 2023 — out-of-sample extension for SVM regression
- *Frontiers in Artificial Intelligence* **8**, 1706566 (2025) — extension to dynamic system identification
- ACDSA 2026 — the pruning application, whose experiments are in this directory

**The implementations are not.** Two separate pieces of software implement the method, and
neither is published here:

| Implementation | Authors | Where | Licence |
|---|---|---|---|
| `nobsp_lib` | Santiago Alférez, Alexander Caicedo | [santialferez/nobsp_lib](https://github.com/santialferez/nobsp_lib) | Apache-2.0 |
| `NObSP_Decomposition.py` | Alexander Caicedo | reproduced inline in the notebooks below | see `NOTICE` |

The interpretability analysis of Chapter 4 used `nobsp_lib` at version 0.1.0, accessed
2026-04-17. It is a dependency, not a contribution of this thesis, and it is not
redistributed here. Install it from its own repository.

---

## `pruning/`

The ACDSA 2026 experiments. The question they test is whether **functional contribution**,
measured by NObSP, identifies a better set of neurons to remove than **weight magnitude**
does. The same network is pruned by both criteria and accuracy is tracked against the
number of neurons removed.

```
pruning/
├── pruning_MNIST.ipynb       multilayer perceptron on MNIST
├── pruning_CIFAR10.ipynb     the same on CIFAR-10
└── checkpoints/              the trained networks the notebooks load
```

**The checkpoints are publishable.** They were trained on MNIST and CIFAR-10, which are
public datasets. This is the opposite of the situation in `gastric-risk/`, where the
weights are withheld because the data behind them are patient records.

### Running them

Set `DATA_ROOT` in the first cell to the directory holding the checkpoints — `./checkpoints/`
if you run from `pruning/`. The datasets download on first run through `torchvision`.

The notebooks were developed on Google Colab, and the hard-coded Drive paths have been
replaced by that constant. Their outputs have been stripped: the originals carried embedded
figures that made them 4 MB and 8.5 MB.

### Attribution inside the notebooks

The cells defining `ObSP`, `NObSP_NN_single_MultiOutput_L1` and
`NObSP_NN_single_MultiOutput_L2` are **© Alexander Caicedo, 2023**, reproduced with his
permission. Everything else is thesis work. See `NOTICE` at the repository root.
