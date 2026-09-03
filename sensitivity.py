"""
sensitivity.py
--------------
Hyperparameter sensitivity for the two experts whose settings a reviewer can
reasonably question, run on the same screened pool and the same injection
protocol as the main benchmark.

  (a) Sequential Expert: HMM n_components in {2, 3, 4, 5, 6, 8}, evaluated on
      the sequential benchmark. Replaces the previous Figure 4b, which was
      flat because the score did not depend on the model.
  (b) GPS Expert: Isolation Forest n_estimators in {50, 100, 200, 400},
      evaluated on the GPS benchmark.

Run after AETHER.py, with the same environment variables:
  AETHER_DATA=/path/to/data.csv AETHER_OUT=/path/to/results python -u sensitivity.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

import AETHER as A

SEEDS = [A.RANDOM_STATE + 1000, A.RANDOM_STATE + 1101]   # disjoint from selection and reporting
HMM_GRID = [2, 3, 4, 5, 6, 8]
IF_GRID = [50, 100, 200, 400]

gdf = A.preprocess_movement_data(A.DATA_FILE_PATH)
fk = ["speed_ms", "acceleration_ms2", "turning_angle"]
fg = ["satellite_count", "dop", "time_to_fix_s"]
fs = fk + (["altitude_m"] if "altitude_m" in gdf.columns else [])

gdf, models, scalers, fsets = A.train_experts(gdf, fk, fg, fs)
pool, _ = A.build_clean_pool(gdf, models, scalers, fsets)
poor = gdf.loc[gdf["dop"] >= gdf["dop"].quantile(0.99),
               ["satellite_count", "dop", "time_to_fix_s"]].copy()
del gdf

seq_scaler = StandardScaler().fit(pool[fs].to_numpy(dtype=float))
X_seq = seq_scaler.transform(pool[fs].to_numpy(dtype=float))
L = A.sequence_lengths(pool)
gps_scaler = StandardScaler().fit(pool[fg].to_numpy(dtype=float))
X_gps = gps_scaler.transform(pool[fg].to_numpy(dtype=float))

rng = np.random.RandomState(A.RANDOM_STATE)
n = len(pool)
fit_idx = (rng.choice(n, size=min(n, A.FIT_SUBSAMPLE), replace=False)
           if n > A.FIT_SUBSAMPLE else np.arange(n))

rows = []

print("\n--- Sweep (a): HMM n_components on the sequential benchmark ---")
for k in HMM_GRID:
    hmm = A.make_hmm(k)
    hmm.fit(X_seq, L)
    base = A.hmm_pointwise_surprisal(hmm, X_seq, L)
    for sd in SEEDS:
        inj, y, ch = A.inject_anomalies(pool, "sequential", A.N_ANOMALIES, sd, poor)
        sc = A.score_benchmark(hmm, seq_scaler, fs, inj, ch, base, is_sequential=True)
        r = A.evaluate_scores(f"HMM k={k}", y, sc)
        r.update({"sweep": "hmm_n_components", "param": k, "seed": sd})
        rows.append(r)
    print(f"  n_components={k} done "
          f"(AUC-ROC {np.mean([r['AUC-ROC'] for r in rows if r.get('param')==k and r['sweep']=='hmm_n_components']):.3f})")

print("\n--- Sweep (b): IF n_estimators on the GPS benchmark ---")
for ne in IF_GRID:
    m = IsolationForest(n_estimators=ne, contamination="auto",
                        random_state=A.RANDOM_STATE, n_jobs=-1).fit(X_gps[fit_idx])
    base = A.score_point_model(m, X_gps)
    for sd in SEEDS:
        inj, y, ch = A.inject_anomalies(pool, "gps", A.N_ANOMALIES, sd, poor)
        sc = A.score_benchmark(m, gps_scaler, fg, inj, ch, base)
        r = A.evaluate_scores(f"IF n_estimators={ne}", y, sc)
        r.update({"sweep": "if_n_estimators", "param": ne, "seed": sd})
        rows.append(r)
    print(f"  n_estimators={ne} done")

df = pd.DataFrame(rows)
out = A.OUTPUT_DIR
df.to_csv(f"{out}/sensitivity_results.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, sweep, xlab, title in [
    (axes[0], "if_n_estimators", "Number of trees (n_estimators)",
     "(a) GPS Expert (Isolation Forest), GPS benchmark"),
    (axes[1], "hmm_n_components", "Number of hidden states (n_components)",
     "(b) Sequential Expert (HMM), sequential benchmark"),
]:
    sub = df[df["sweep"] == sweep].groupby("param").agg(
        auc=("AUC-ROC", "mean"), aucsd=("AUC-ROC", "std"),
        pr=("PR-AUC", "mean"), prsd=("PR-AUC", "std")).reset_index()
    ax.errorbar(sub["param"], sub["auc"], yerr=sub["aucsd"], marker="o",
                capsize=3, label="AUC-ROC")
    ax.set_xlabel(xlab)
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.errorbar(sub["param"], sub["pr"], yerr=sub["prsd"], marker="s",
                 color="firebrick", capsize=3, label="PR-AUC")
    ax2.set_ylabel("PR-AUC", color="firebrick")
    ax.set_title(title)
plt.tight_layout()
plt.savefig(f"{out}/figure_sensitivity.png", dpi=200)
print(f"\nWrote {out}/sensitivity_results.csv and {out}/figure_sensitivity.png")
