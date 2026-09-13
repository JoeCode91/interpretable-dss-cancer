# Differences from the code that produced the thesis results

The code in this repository is **not byte-for-byte** the code that generated the figures
and tables of the thesis. This file records every difference, so that any discrepancy can
be traced rather than guessed at.

The working tree that produced the results is archived separately and unchanged.

---

## 1. Presentation

- **Provenance headers.** Every published file carries a `REPRODUCES IN THE THESIS` block
  naming the section, table or figure it produces, and a `PROVENANCE` block stating that
  the computational content is unchanged. Nothing else in those files was touched.
- **Directory layout.** The working tree was flat (`Modelos_Replicator/Codigos/`,
  `XAI_Model/`). Here it is organized by contribution. **Scripts that write results to
  relative paths will create those directories wherever they are run from.**

## 2. Selection

Not everything was carried over. Three categories were left out:

- **Superseded versions.** Where several versions of a procedure exist, only the one that
  produced the reported result is published. For `adaptive-therapy` that means the `_V3`
  controllers and sweeps; the earlier `MPC_Replicator_SA.py`, `_V2`, `MPC_Replicator_DA.py`,
  `_V2`, `Sensitivity_SA.py`, `_V1`, `_V2`, `Sensitivity_DA_T4.py`, `_V1`,
  `Sensitivity_DA_T5A.py` and `Sensitivity_DA_T5B.py` are not included.
- **Environment checks and exploratory scripts.** They produce no result reported in the
  thesis.
- **Credentials.** `telegram_notifier.py` carried a bot token and is excluded. Scripts that
  imported it degrade gracefully when it is absent.

## 3. One correction

**The dual-agent competition matrix.** Two entries were inconsistent between the two
implementations:

| Entry | Chapter 5 code | Chapter 6 code | Published here |
|---|---|---|---|
| `alpha[0][3]` | `8 - 2*delta` = 7.9 | `0.8 - 2*delta` = 0.7 | **0.7** |
| `alpha[1][3]` | `5 - 2*delta` = 4.9 | `0.5 - 2*delta` = 0.4 | **0.4** |

Every other entry of that matrix follows the rule `alpha[i][3] = alpha[i][2] - 2*delta`.
Only these two break it, which points to a lost decimal point in the Chapter 5 code.
This repository adopts the Chapter 6 form, which is the one documented in the thesis
parameter table.

**Verified not to affect any reported result.** Those coefficients multiply the
doubly-resistant phenotype, which starts at `3.97e-12` and stays at zero throughout every
scenario simulated. The stable equilibria computed under either matrix are identical to
the last reported digit:

```
alpha[.][3] = 0.7 / 0.4   ->   T+ = 2167.88   TP = 6819.46   T-/+ = 1809.88
alpha[.][3] = 7.9 / 4.9   ->   T+ = 2167.88   TP = 6819.46   T-/+ = 1809.88
```

The defect is **latent**: it would become active only in a scenario where the
doubly-resistant population grows appreciably, which is the scenario the dual-agent
therapy is designed to prevent. It is corrected here for correctness, not because it moves
a number.

> **Note.** The uncorrected form remains in the archived working tree, in the files that
> supported the conference and journal papers derived from Chapter 5. Whether to correct
> it there is a separate decision.

---

## What this means for reproducing the thesis

Running this code reproduces the reported results. The differences above are
presentational, selective, or verified inert. If a number you obtain differs from the
thesis, it is not explained by anything on this page, and it is worth reporting.
