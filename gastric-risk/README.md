# gastric-risk

Population-scale gastric cancer risk prediction from administrative health records, and
the NObSP interpretability analysis applied to it.

Corresponds to **Chapter 4** of the thesis.

---

## Read this first

**The data are not here and will not be.** The models were trained on longitudinal
diagnosis and procedure sequences for 4.9 million patients of the Colombian SGSSS,
together with municipality of residence and demographics. Removing direct identifiers does
not make such records anonymous — a clinical trajectory plus a geographic attribute is a
standard re-identification vector — and they are not the authors' to distribute: they were
accessed under ethics approval FM-CIE-030920, extended to cover this work, and remain
under the custody of the issuing institutions. Requests for access go to that custodian, not to the authors.

**The trained weights are not here either.** They were fitted to the restricted records
above, and the ethics approval under which those records were accessed does not contemplate
the distribution of models derived from them. Releasing them would require a decision by the
data custodian rather than by the authors; researchers with a specific interest may contact
the authors, who will refer the request to the custodian.

**So the code here cannot reproduce the numbers in the thesis.** It can be read, and it can
be run end to end on synthetic data. That is the honest extent of it.

## Running it on synthetic data

```bash
python data/synthetic/generate_synthetic_cohort.py --out cohort.h5 --n 2000
```

This emits an HDF5 with the same schema as the real cohort — `X_events` of shape
`(n, 1541, 100)`, `X_demograficos` of `(n, 15)`, `y` of `(n,)` — the same padding rule and
the same class ratio. Use `--ratio 5103` for the population prevalence instead of the
1:100 undersampled ratio used for training.

It reproduces the **mechanics** of the pipeline and not the **statistics** of the cohort.
Embeddings are drawn from a normal distribution rather than from the BSC-Bio-EHR-ES
encoder, and no record derives from any real patient. A model trained on it will not
approach the reported metrics, and should not.

Paths in these scripts were absolute in the author's working tree and have been replaced
by relative defaults. Adjust them before running.

## Layout

```
data/            cohort construction, and the synthetic generator
embeddings/      BSC-Bio-EHR-ES embeddings of the code descriptions, and the silhouette
                 comparison against the earlier BioBERT and Word2Vec pilots
models/          the CNN-2D architecture adopted as the final model
training/        training, operating thresholds, metrics, masking validation
interpretability/  the four NObSP wrappers and everything derived from them
geo/             DIVIPOLA municipality mapping and the choropleth of Section 4.9
```

## On NObSP

The four wrappers in `interpretability/aplicar_nobsp_v4.py` — demographics, temporal
positions, event groups, learned channels — are the adaptation of the method to this
architecture, and are thesis work.

The NObSP library itself is **not**. It is
[`nobsp_lib`](https://github.com/santialferez/nobsp_lib) by Santiago Alférez and Alexander
Caicedo, version 0.1.0, Apache-2.0, and it is a dependency rather than a contribution of
this thesis. Install it from its own repository. See `NOTICE` at the repository root.

## What was left out

Superseded versions of these scripts — the V2, V3, V3.2 and pre-V4 generations — are not
published; only the version that produced the reported result is. Two scripts that
characterise the raw source database are also absent: they generate the statistical
appendix of the thesis, but they operate on the restricted raw records, which the synthetic
generator does not emulate, so nobody outside could run them. `CHANGES-FROM-THESIS.md` at
the repository root has the full account.

## A caveat the thesis itself makes

The reported precision figures are measured on a test partition held at the 1:100
undersampled ratio, not at the 1:5103 prevalence of the source population. Sensitivity,
specificity and AUC-ROC transfer; precision and the count of false-positive referrals do
not, and fall by roughly a factor of fifty at the true prevalence. Do not read any
precision produced by this code as a deployment estimate.
