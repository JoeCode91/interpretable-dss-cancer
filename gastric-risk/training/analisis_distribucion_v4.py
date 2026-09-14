"""
analisis_distribucion_v4.py -- Distribution analysis for V4 test data.

REPRODUCES IN THE THESIS
    Distribution of predicted scores, Section 4.7.

PROVENANCE
    Copied verbatim from the working tree of the thesis. The computational content is
    unchanged; only this provenance block was added. See CHANGES-FROM-THESIS.md at the
    repository root.

--------------------------------------------------------------------------
analisis_distribucion_v4.py -- Distribution analysis for V4 test data.

Analyzes: event count distribution, cancer rate by latitude, cancer rate by EPS.
"""
import glob
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT = Path(r"G:\Mi unidad\Doctorado\Proyecto_CC\Test\Book\XAI_Model\nobsp_results_v4")
V4_DIR = Path(r"D:\Proyecto_CC_Batches\Datos_Modelo_V4")
SCALERS = {'lat': (5.6452, 2.2558), 'lon': (-74.9218, 1.0864), 'edad': (52.8734, 14.9610)}

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

test_f = sorted(glob.glob(str(V4_DIR / "datos_modelo_test_v4_*.h5")))[-1]
print(f"Test file: {test_f}")

with h5py.File(test_f, 'r') as f:
    X_demo = f['X_demograficos'][:].astype(np.float32)
    y = f['y'][:].astype(int)
    n_total = len(y)
    print(f"Total: {n_total}, Cancer: {y.sum()}, Rate: {y.mean()*100:.2f}%")

    # Event count: process in batches
    print("\nCounting events per patient (batches of 200)...")
    event_counts = np.zeros(n_total, dtype=int)
    batch_sz = 200
    for start in range(0, n_total, batch_sz):
        end = min(start + batch_sz, n_total)
        X_batch = f['X_events'][start:end]  # (batch, 1541, 100)
        col_sums = np.abs(X_batch).sum(axis=1)  # (batch, 100)
        event_counts[start:end] = (col_sums > 0).sum(axis=1)
        if start % 2000 == 0:
            print(f"  {start}/{n_total}", flush=True)
    print(f"  Done. Median events: {np.median(event_counts):.0f}, "
          f"P25: {np.percentile(event_counts,25):.0f}, "
          f"P75: {np.percentile(event_counts,75):.0f}, "
          f"P95: {np.percentile(event_counts,95):.0f}")

# --- 1. Event count distribution ---
fig, ax = plt.subplots(figsize=(10, 4))
bins = np.arange(0, 102, 2)
ax.hist(event_counts[y==0], bins=bins, alpha=0.6, color='#91bfdb', label='No cancer', density=True)
ax.hist(event_counts[y==1], bins=bins, alpha=0.8, color='#d7191c', label='Cancer', density=True)
ax.axvline(np.median(event_counts), color='black', linestyle='--', linewidth=1, label=f'Median={np.median(event_counts):.0f}')
ax.set_xlabel("Number of non-zero events (out of 100 positions)")
ax.set_ylabel("Density")
ax.set_title("Distribution of event count per patient (test set)")
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add annotation about zero-padding
ax.annotate(f"Positions < (100 - n_events)\nare zero-padded",
            xy=(15, ax.get_ylim()[1]*0.7), fontsize=9, fontstyle='italic', color='gray')
for ext in ['png', 'pdf']:
    fig.savefig(OUTPUT / f"pub_event_count_distribution.{ext}")
plt.close(fig)
print("\n[1] pub_event_count_distribution saved")

# --- 2. Cancer rate by latitude ---
lat_real = X_demo[:, 12] * SCALERS['lat'][1] + SCALERS['lat'][0]

lat_bands = [
    (0, 2, "0-2 N (Amazonas)"),
    (2, 3, "2-3 N (Narino/Cauca)"),
    (3, 5, "3-5 N (Valle/Tolima)"),
    (5, 6, "5-6 N (Bogota/Boyaca S)"),
    (6, 7, "6-7 N (Boyaca N/Santander)"),
    (7, 8, "7-8 N (Santander/NdS)"),
    (8, 15, "8+ N (Costa Caribe)"),
]

print("\n--- Cancer rate by latitude ---")
print(f"{'Band':<30s} {'Total':>7s} {'Cancer':>7s} {'Rate':>8s}")
print("-" * 55)
band_names, band_totals, band_cancer, band_rates = [], [], [], []
for lo, hi, name in lat_bands:
    mask = (lat_real >= lo) & (lat_real < hi)
    n = mask.sum()
    c = y[mask].sum()
    rate = c / n * 100 if n > 0 else 0
    print(f"{name:<30s} {n:>7d} {c:>7d} {rate:>7.2f}%")
    band_names.append(name.split("(")[1].rstrip(")") if "(" in name else name)
    band_totals.append(n)
    band_cancer.append(c)
    band_rates.append(rate)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
y_pos = np.arange(len(band_names))

ax1.barh(y_pos, band_totals, color='#91bfdb', edgecolor='white')
for i, (t, c) in enumerate(zip(band_totals, band_cancer)):
    ax1.text(t + 50, i, f"{c} cancer", va='center', fontsize=9, color='#d7191c')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(band_names, fontsize=9)
ax1.invert_yaxis()
ax1.set_xlabel("Number of patients")
ax1.set_title("(a) Patients per latitude band")
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

colors = ['#d7191c' if r > np.mean(band_rates) else '#91bfdb' for r in band_rates]
ax2.barh(y_pos, band_rates, color=colors, edgecolor='white')
for i, r in enumerate(band_rates):
    ax2.text(r + 0.05, i, f"{r:.2f}%", va='center', fontsize=9)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(band_names, fontsize=9)
ax2.invert_yaxis()
ax2.set_xlabel("Cancer rate (%)")
ax2.set_title("(b) Cancer rate per latitude band")
ax2.axvline(y.mean()*100, color='gray', linestyle='--', linewidth=0.8, label=f'Overall={y.mean()*100:.2f}%')
ax2.legend(fontsize=8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.suptitle("Geographic distribution of patients and cancer rate (test set, by latitude)", fontsize=13)
fig.tight_layout()
for ext in ['png', 'pdf']:
    fig.savefig(OUTPUT / f"pub_cancer_rate_by_latitude.{ext}")
plt.close(fig)
print("\n[2] pub_cancer_rate_by_latitude saved")

# --- 3. Cancer rate by EPS ---
eps_idx = np.argmax(X_demo[:, :10], axis=1)

print("\n--- Cancer rate by EPS ---")
print(f"{'EPS':<10s} {'Total':>7s} {'Cancer':>7s} {'Rate':>8s}")
print("-" * 35)
eps_names, eps_totals, eps_cancer, eps_rates = [], [], [], []
for i in range(10):
    mask = eps_idx == i
    n = mask.sum()
    c = y[mask].sum()
    rate = c / n * 100 if n > 0 else 0
    print(f"EPS_{i:<5d} {n:>7d} {c:>7d} {rate:>7.2f}%")
    eps_names.append(f"EPS_{i}")
    eps_totals.append(n)
    eps_cancer.append(c)
    eps_rates.append(rate)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
y_pos = np.arange(10)

ax1.barh(y_pos, eps_totals, color='#2166ac', edgecolor='white')
for i, (t, c) in enumerate(zip(eps_totals, eps_cancer)):
    ax1.text(t + 20, i, f"{c} cancer", va='center', fontsize=9, color='#d7191c')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(eps_names, fontsize=10)
ax1.invert_yaxis()
ax1.set_xlabel("Number of patients")
ax1.set_title("(a) Patients per EPS")
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

colors = ['#d7191c' if r > y.mean()*100 else '#2166ac' for r in eps_rates]
ax2.barh(y_pos, eps_rates, color=colors, edgecolor='white')
for i, r in enumerate(eps_rates):
    ax2.text(r + 0.03, i, f"{r:.2f}%", va='center', fontsize=9)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(eps_names, fontsize=10)
ax2.invert_yaxis()
ax2.set_xlabel("Cancer rate (%)")
ax2.set_title("(b) Cancer rate per EPS")
ax2.axvline(y.mean()*100, color='gray', linestyle='--', linewidth=0.8, label=f'Overall={y.mean()*100:.2f}%')
ax2.legend(fontsize=8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.suptitle("EPS distribution of patients and cancer rate (test set)", fontsize=13)
fig.tight_layout()
for ext in ['png', 'pdf']:
    fig.savefig(OUTPUT / f"pub_cancer_rate_by_eps.{ext}")
plt.close(fig)
print("\n[3] pub_cancer_rate_by_eps saved")

print("\nDone!")
