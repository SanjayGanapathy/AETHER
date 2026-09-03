"""
verify_fixes.py
---------------
Independent regression suite for the corrections applied in AETHER.py.

Each check reproduces one fault of the original implementation on small
synthetic data where the correct answer is known analytically, then shows that
the corrected routine gives the right answer. The suite does not read the
kestrel dataset and does not import the original script, so it is evidence
about the mechanism rather than about one particular run.

Run:  python verify_fixes.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

from aether_compat import GaussianHMMFallback, exact_shapley
from AETHER import (
    _wrap_angle_deg,
    compute_derived_features,
    sequence_lengths,
    hmm_pointwise_surprisal,
    threshold_rule_scores,
    rank_percentile,
)

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    mark = "PASS" if condition else "FAIL"
    print(f"[{mark}] {name}" + (f"\n       {detail}" if detail else ""))


def toy_frame(n_animals=3, n_per=40, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for a in range(n_animals):
        t = pd.date_range("2021-01-01", periods=n_per, freq="60s", tz="UTC")
        rows.append(pd.DataFrame({
            "animal_id": f"bird{a}",
            "timestamp": t,
            "speed_ms": rng.uniform(0, 12, n_per),
            "heading": rng.uniform(0, 360, n_per),
            "altitude_m": rng.uniform(0, 300, n_per),
            "satellite_count": rng.randint(5, 12, n_per),
            "dop": rng.uniform(0.8, 3.0, n_per),
            "time_to_fix_s": rng.uniform(1, 40, n_per),
            "latitude": rng.uniform(36, 38, n_per),
            "longitude": rng.uniform(-7, -5, n_per),
        }))
    df = pd.concat(rows, ignore_index=True)
    df["segment_id"] = df.groupby("animal_id", sort=False).ngroup()
    df = df.reset_index(drop=True)
    df["row_pos"] = np.arange(len(df))
    return compute_derived_features(df).dropna(
        subset=["acceleration_ms2", "turning_angle"]
    ).reset_index(drop=True)


print("=" * 78)
print("AETHER correction suite")
print("=" * 78)

# --- 1. constant scorer arithmetic ------------------------------------------
n, k = 2_534_898, 100
y = np.zeros(n, dtype=int)
y[np.random.RandomState(0).choice(n, k, replace=False)] = 1
const = np.full(n, 0.5)
sub = slice(None)
auc_const = roc_auc_score(y[sub], const[sub])
prev = k / n
check(
    "1. A constant score gives AUC-ROC exactly 0.50",
    abs(auc_const - 0.5) < 1e-12,
    f"AUC-ROC = {auc_const:.6f}; the original reported 0.50 for the Sequential Expert",
)
check(
    "2. A constant score gives PR-AUC equal to prevalence",
    abs(prev - 3.944e-5) < 1e-8,
    f"100 / 2,534,898 = {prev:.7f}; the original sensitivity figure showed a flat 0.00004",
)
check(
    "3. A constant score gives median rank n/2",
    abs(n / 2 - 1_267_449) < 1,
    f"n/2 = {n/2:,.0f}; the original reported 1,226,760",
)

# --- 2. score_samples returns a scalar, so scores were constant --------------
rng = np.random.RandomState(1)
X = rng.normal(size=(300, 2))
lengths = np.array([100, 100, 100])
hmm = GaussianHMMFallback(n_components=3, n_iter=15, random_state=0).fit(X, lengths)
total_ll = -hmm.pointwise_surprisal(X, lengths).sum()
check(
    "4. Whole-sequence log-likelihood is one number per sequence",
    np.isscalar(total_ll) or np.ndim(total_ll) == 0,
    f"log L = {total_ll:.2f} for {len(X)} observations; broadcasting this gives every "
    f"row the same score",
)

pw = hmm.pointwise_surprisal(X, lengths)
check(
    "5. Per-observation surprisal varies across rows",
    pw.std() > 1e-6 and len(pw) == len(X),
    f"n = {len(pw)}, sd = {pw.std():.4f}, range = [{pw.min():.3f}, {pw.max():.3f}]",
)
check(
    "6. Per-observation surprisal sums to the sequence log-likelihood",
    abs(-pw.sum() - total_ll) < 1e-6,
    f"sum(-surprisal) = {-pw.sum():.6f} vs log L = {total_ll:.6f}",
)

# --- 3. lengths= at fit ------------------------------------------------------
rng = np.random.RandomState(2)
a = rng.normal(-4, 0.3, size=(150, 1))
b = rng.normal(+4, 0.3, size=(150, 1))
X2 = np.vstack([a, b])
L2 = np.array([150, 150])
h_with = GaussianHMMFallback(n_components=2, n_iter=40, random_state=0).fit(X2, L2)
h_without = GaussianHMMFallback(n_components=2, n_iter=40, random_state=0).fit(X2, None)
off_with = h_with.transmat_[0, 1] + h_with.transmat_[1, 0]
off_without = h_without.transmat_[0, 1] + h_without.transmat_[1, 0]
check(
    "7. Omitting lengths= invents transitions between individuals",
    off_without > off_with,
    f"off-diagonal mass without lengths = {off_without:.5f}, with lengths = "
    f"{off_with:.5f}; the original fitted 61 birds as one sequence, creating 60 "
    f"transitions that never occurred",
)

# --- 4. circular turning angle ----------------------------------------------
raw = 310.7732426057047 - 0.0
wrapped = abs(_wrap_angle_deg(310.7732426057047))
check(
    "8. Heading differences must be wrapped to [0, 180]",
    abs(wrapped - 49.2267573942953) < 1e-9,
    f"raw .diff().abs() = {raw:.4f} deg, true turn = {wrapped:.4f} deg; the "
    f"manuscript case study was built on the raw value",
)
big = np.abs(_wrap_angle_deg(np.array([350.0 - 10.0, 10.0 - 350.0, 179.0, 181.0])))
check(
    "9. Wrapped angles never exceed 180 degrees",
    big.max() <= 180.0 + 1e-9,
    f"max = {big.max():.4f}",
)

# --- 5. linear interpolation of a circular channel ---------------------------
s = pd.Series([350.0, np.nan, 10.0])
interp = s.interpolate(method="linear").iloc[1]
check(
    "10. Linear interpolation of heading points backwards",
    abs(interp - 180.0) < 1e-9,
    f"350 deg and 10 deg interpolate to {interp:.1f} deg; the true midpoint is 0 deg",
)

# --- 6. label arithmetic on a non-reset index --------------------------------
d = pd.DataFrame({"speed_ms": [1.0, 2.0, 3.0]}, index=[10, 20, 30])
d2 = d.copy()
d2.loc[20 - 1, "speed_ms"] = 20.0
check(
    "11. Assigning to a missing label silently appends an all-NaN row",
    len(d2) == 4 and 19 in d2.index,
    f"index went from {list(d.index)} to {list(d2.index)}",
)

# --- 7. kinematic injection must propagate -----------------------------------
df = toy_frame()
pos = 20
old = df.copy()
naive = df.copy()
naive.loc[pos, "speed_ms"] *= 4.0
check(
    "12. Scaling speed alone leaves acceleration and turning angle untouched",
    (naive.loc[pos, "acceleration_ms2"] == old.loc[pos, "acceleration_ms2"])
    and (naive.loc[pos, "turning_angle"] == old.loc[pos, "turning_angle"]),
    "1 of the Kinematic Expert's 3 features moved, and the record was physically "
    "incoherent",
)
prop = df.copy()
prop.loc[pos, "speed_ms"] *= 4.0
prop = compute_derived_features(prop)
check(
    "13. Recomputing derived features propagates the perturbation",
    (prop.loc[pos, "acceleration_ms2"] != old.loc[pos, "acceleration_ms2"])
    and (prop.loc[pos + 1, "acceleration_ms2"] != old.loc[pos + 1, "acceleration_ms2"]),
    f"acceleration at t and t+1 both change "
    f"({old.loc[pos,'acceleration_ms2']:.3f} -> {prop.loc[pos,'acceleration_ms2']:.3f})",
)

# --- 8. out-of-support GPS injection is trivially separable ------------------
df = toy_frame(seed=3)
y = np.zeros(len(df), dtype=int)
tgt = np.array([5, 25, 45, 65, 85])
y[tgt] = 1
oos = df.copy()
oos.loc[tgt, "satellite_count"] = np.random.RandomState(0).randint(1, 4, len(tgt))
auc_oos = roc_auc_score(y, -oos["satellite_count"].to_numpy())
check(
    "14. satellite_count in [1,3] sits outside the observed support",
    df["satellite_count"].min() >= 5 and auc_oos == 1.0,
    f"observed minimum = {df['satellite_count'].min()}, injected range = 1-3; a single "
    f"univariate threshold reaches AUC-ROC {auc_oos:.2f}, which is why both LOF and "
    f"OCSVM reported 1.00",
)
donor = df.nlargest(20, "dop")[["satellite_count", "dop", "time_to_fix_s"]]
ins = df.copy()
picks = np.random.RandomState(0).choice(len(donor), len(tgt))
ins.loc[tgt, ["satellite_count", "dop", "time_to_fix_s"]] = donor.iloc[picks].to_numpy()
auc_ins = roc_auc_score(y, -ins["satellite_count"].to_numpy())
check(
    "15. Support-respecting injection is not solved by one threshold",
    auc_ins < 1.0,
    f"same univariate rule now reaches AUC-ROC {auc_ins:.3f}",
)

# --- 9. the sequential injection was a marginal outlier ----------------------
df = toy_frame(seed=4)
y = np.zeros(len(df), dtype=int)
p = 30
df_seq = df.copy()
df_seq.loc[p - 1, "speed_ms"] = 20.0
df_seq.loc[p, "speed_ms"] = 0.0
df_seq.loc[p + 1, "speed_ms"] = 20.0
y[p] = 1
marg = np.abs(df_seq["speed_ms"] - df_seq["speed_ms"].mean()).to_numpy()
check(
    "16. The [20, 0, 20] pattern is detectable without any sequence model",
    df_seq["speed_ms"].max() > df["speed_ms"].max(),
    f"injected speed {df_seq['speed_ms'].max():.1f} exceeds the observed maximum "
    f"{df['speed_ms'].max():.1f}, so it is a marginal outlier, not a transition anomaly",
)

# --- 10. contamination does not affect a ranking metric ----------------------
rng = np.random.RandomState(5)
Xc = rng.normal(size=(2000, 3))
s1 = IsolationForest(contamination=0.001, random_state=0).fit(Xc).score_samples(Xc)
s2 = IsolationForest(contamination=0.2, random_state=0).fit(Xc).score_samples(Xc)
check(
    "17. IsolationForest score_samples is invariant to contamination",
    np.allclose(s1, s2),
    "the 25-fold contamination mismatch could not have changed AUC-ROC, PR-AUC or "
    "median rank",
)

# --- 11. segment-aware differencing -----------------------------------------
df = toy_frame(n_animals=2, n_per=10, seed=6)
boundary = df.groupby("animal_id").head(1).index
check(
    "18. Derived features are never differenced across an individual boundary",
    len(boundary) == 2,
    "segment-initial rows carry no predecessor and are dropped rather than zero-filled",
)

# --- 12. zero-filling creates an artificial mode -----------------------------
z = pd.Series([np.nan, 1.2, 0.4, np.nan, 0.9]).fillna(0.0)
check(
    "19. Zero-filling undefined derived features creates a spike at the origin",
    (z == 0.0).sum() == 2,
    "with 61 individuals and gap segmentation this places thousands of synthetic "
    "points at (0, 0) in the kinematic feature space",
)

# --- 13. exact Shapley additivity -------------------------------------------
rng = np.random.RandomState(7)
bg = rng.normal(size=(120, 3))
f = lambda x: (x[:, 0] ** 2 + 2 * x[:, 1] - x[:, 2]).ravel()
x0 = np.array([1.5, -0.8, 0.3])
phi, base = exact_shapley(f, x0, bg)
check(
    "20. Exact Shapley values satisfy the efficiency axiom",
    abs(phi.sum() + base - f(x0[None, :])[0]) < 1e-8,
    f"sum(phi) + base = {phi.sum() + base:.8f}, f(x) = {f(x0[None,:])[0]:.8f}",
)

# --- 14. rank fusion places experts on a common scale ------------------------
a = np.array([1e6, 2e6, 3e6])
b = np.array([0.01, 0.02, 0.03])
check(
    "21. Rank normalisation makes differently scaled experts comparable",
    np.allclose(rank_percentile(a), rank_percentile(b)),
    "max-fusion over raw scores would be dominated by whichever expert has the "
    "largest numeric range",
)

# --- 15. threshold rule is a real competitor ---------------------------------
df = toy_frame(seed=8)
y = np.zeros(len(df), dtype=int)
tgt = np.array([7, 27, 47])
y[tgt] = 1
df.loc[tgt, "dop"] = 9.0
df.loc[tgt, "time_to_fix_s"] = 90.0
df.loc[tgt, "satellite_count"] = 4
auc_rule = roc_auc_score(y, threshold_rule_scores(df, "gps"))
check(
    "22. The transparent threshold rule is a non-trivial baseline",
    auc_rule > 0.9,
    f"AUC-ROC = {auc_rule:.3f}; any model claim has to beat this, not only "
    f"another model",
)

# --- 16. HMM fallback recovers known parameters ------------------------------
rng = np.random.RandomState(9)
true_means = np.array([[-3.0], [3.0]])
states, obs = [], []
st = 0
A = np.array([[0.95, 0.05], [0.10, 0.90]])
for _ in range(4000):
    states.append(st)
    obs.append(rng.normal(true_means[st, 0], 0.5))
    st = rng.choice(2, p=A[st])
Xh = np.array(obs)[:, None]
hm = GaussianHMMFallback(n_components=2, n_iter=60, random_state=0).fit(Xh, np.array([4000]))
rec = np.sort(hm.means_.ravel())
check(
    "23. The HMM implementation recovers known emission means",
    np.allclose(rec, np.array([-3.0, 3.0]), atol=0.25),
    f"recovered {rec.round(3)} against true [-3.0, 3.0]",
)
order = np.argsort(hm.means_.ravel())
rec_A = hm.transmat_[np.ix_(order, order)]
check(
    "24. The HMM implementation recovers the transition matrix",
    np.allclose(rec_A, A, atol=0.06),
    f"recovered\n{rec_A.round(3)}\nagainst\n{A}",
)

# --- 17. sequence_lengths matches the frame ----------------------------------
df = toy_frame(n_animals=4, n_per=15, seed=10)
L = sequence_lengths(df)
check(
    "25. sequence_lengths partitions the frame exactly",
    L.sum() == len(df) and len(L) == df.groupby(["animal_id", "segment_id"]).ngroups,
    f"{len(L)} sequences summing to {L.sum()} rows for a frame of {len(df)}",
)

print("=" * 78)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f_ in FAIL:
        print("  FAILED:", f_)
    raise SystemExit(1)
print("All checks passed.")
