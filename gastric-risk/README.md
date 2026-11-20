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
python data/synthetic/generate_synthetic_cohort.py --pipeline-dir batches --n 6000
```

This writes, under `batches/`, the files the rest of the pipeline reads: train, validation
and test partitions (`Datos_Modelo_CNN/datos_modelo_{train,val,test}_cnn_<ts>.h5`,
stratified 80/10/10), the build metadata with the demographic scalers, and a synthetic code
vocabulary with its descriptions. Each partition has the schema of the real cohort:
`X_events` of shape `(n, 1541, 100)`, `X_demograficos` of `(n, 15)`, `y` of `(n,)`, plus
`X_temporal_raw`, `ID` and `gidx`; the same padding rule, the same scaling of the temporal
features and the same class ratio. `--out cohort.h5` writes a single file instead, and
`--ratio 5103` gives the population prevalence instead of the 1:100 undersampled ratio.

It reproduces the **mechanics** of the pipeline and not the **statistics** of the cohort.
Embeddings are rows of a random vocabulary of unit vectors rather than of the
BSC-Bio-EHR-ES encoder, codes are synthetic labels, and no record derives from any real
patient. A model trained on it will not approach the reported metrics, and should not.

Paths in these scripts were absolute in the author's working tree and have been replaced
by relative defaults under `./batches`, overridable by environment variables. Run the
scripts from the folder that contains `batches/`, or adjust the variables.

## The pipeline

**Cohort** (`data/Preparar_HDF5_V4.py`, on the restricted records). The label of every
patient of the source population is read first; the selection (all cancer patients and
100 non-cancer patients per cancer patient, seed 42) and the stratified 80/10/10 split
depend only on it. Event tensors are then built for the selected patients only. Each event
is a 1541-vector: the 768-dimensional embeddings of its diagnosis and of its procedure,
followed by five temporal features — days since the first recorded event, days since the
previous event, event index, index over history length, history length. Dates are parsed
with an explicit format; an unparseable date takes the last valid date of the patient
(`data/parseo_fechas.py`). The five temporal features are z-scored with the mean and
standard deviation of the real events of the training partition; padding columns stay
exactly zero. Each partition also stores the unscaled features (`X_temporal_raw`), the
patient identifier and the row in the population.

**Training.** `data/generar_balanced_20to1.py` subsamples the 1:100 training partition to
1:20 (all positives, 20 negatives per positive, seed 42). The CNN-2D model is trained on
that 1:20 partition and selected and evaluated on the 1:100 validation and test partitions
(`training/Modelo_Predictivo_CNN2D_V4.py`; hybrid focal + weighted BCE loss, checkpoint
criterion AUC-PR + 0.1 Youden's J on validation, seed 42).
`training/threshold_sweep_v4.py` sets the Youden-optimal threshold on validation and saves
the test probabilities, and `training/recalculo_aucpr_final.py` computes AUC-PR as average
precision with paired bootstrap intervals against the BiLSTM-Attention comparison model.
Every script after training fixes its checkpoint by timestamp (`CNN2D_TS`, and
`BILSTM_TS` where the comparison model is used) and records its SHA-256 in its outputs
(`training/checkpoint_utils.py`).

**Interpretability.** NObSP is computed on *sample B*: all cancer patients of the 1:100
training partition and of the test partition, and as many non-cancer patients drawn from
both (seed 42); the temporal, event-group and channel wrappers use 500 multiplicative
probes around the mean of that sample (`interpretability/aplicar_nobsp_v4_r2.py`). The
input-space reconstructions, the figures, the georeferenced map and its department table,
the decoded diagnoses and the per-patient attribution are derived from it. The per-patient
CAM figure uses *sample C*: all cancer patients of the three partitions (963 in the real
cohort) and the same number of non-cancer patients per partition.

On synthetic data, the sequence is:

```bash
python data/synthetic/generate_synthetic_cohort.py --pipeline-dir batches --n 6000
python data/generar_balanced_20to1.py --src batches/Datos_Modelo_CNN/datos_modelo_train_cnn_<ts>.h5 \
       --dst-dir batches/Datos_Modelo_CNN
python training/Modelo_Predictivo_CNN2D_V4.py          # CNN2D_MAX_EPOCHS / CNN2D_MAX_STEPS shorten it
CNN2D_TS=<training timestamp> python training/threshold_sweep_v4.py
CNN2D_TS=<training timestamp> python interpretability/aplicar_nobsp_v4_r2.py   # R2_N_PERTURB to shorten
CNN2D_TS=<training timestamp> python interpretability/reconstruir_nobsp_input_v4.py
python interpretability/reconstruir_w3_input_v4.py
python training/generar_figuras_pub_v4.py
EMB_MAP_DIR=batches/embeddings_map python interpretability/decodificar_diagnosticos_r3.py
python interpretability/analisis_atribucion_individual.py
CNN2D_TS=<training timestamp> python interpretability/cam_por_paciente_balanceado.py
```

The geographic scripts additionally need the DIVIPOLA map (`geo/construir_mapa_municipio_latlon.py`,
from DANE open data) and the municipal shapefile. The comparison with the BiLSTM-Attention
model (`training/recalculo_aucpr_final.py`, `interpretability/colinealidad_atencion_v2.py`)
needs that model and its windowed HDF5, which are not part of this repository.

## Layout

```
data/            cohort construction (builder, date parsing, selection and scaling,
                 filtered loading), the 1:20 training subsample, and the synthetic generator
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
published; only the version that produced the reported result is. The builder and the
training script of the BiLSTM-Attention comparison model are not published either; the
scripts that compare against it read its saved outputs and its checkpoint, and the loss and
checkpoint criterion shared by both models are in `training/perdida_hibrida.py`. Two scripts that
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
