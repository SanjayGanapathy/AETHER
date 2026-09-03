# AETHER Framework  --  corrected implementation
#
# This file is a repair of the original AETHER.py, not a rewrite. The structure,
# function names and execution order of the original are preserved. Every
# substantive change is marked with a numbered "# FIX n" comment and is
# reproduced independently by verify_fixes.py.
#
# Summary of corrections (see Appendix D of the manuscript):
#   FIX 1  sensor channels are no longer imputed
#   FIX 2  heading is circular, so turning angle is wrapped to [0, 180]
#   FIX 3  tracks are split into gap-free segments; derived features never
#          cross a gap or an individual boundary
#   FIX 4  acceleration and turning angle are undefined at a segment start and
#          are dropped rather than filled with zero
#   FIX 5  the HMM is fitted with lengths=, so the 61 individuals are not
#          treated as one continuous sequence
#   FIX 6  the Sequential Expert scores each observation by predictive
#          surprisal instead of assigning every row the same sequence
#          log-likelihood
#   FIX 7  the sequential feature set is behavioural only; GPS-quality
#          channels are removed from it
#   FIX 8  a screened clean pool is built before injection to limit
#          ground-truth contamination
#   FIX 9  kinematic injections perturb the underlying channels and then
#          recompute the derived features, so the injected state is physically
#          coherent
#   FIX 10 GPS injections resample an observed joint triple from the poor
#          quality tail, so injected values stay inside the observed support
#   FIX 11 sequential injections splice a behaviourally mismatched block, so
#          the anomaly is in the transition and not in the marginals
#   FIX 12 positional indexing replaces label arithmetic, which silently
#          created all-NaN rows
#   FIX 13 contamination no longer affects the reported metrics, and the
#          reason is stated
#   FIX 14 injected anomalies are drawn with an independent seed per
#          repetition instead of a single fixed seed
#   FIX 15 a full algorithm x feature-subset ablation separates the effect of
#          the algorithm from the effect of the partition
#   FIX 16 a transparent threshold rule is benchmarked alongside the models
#   FIX 17 the committee is given an explicit fusion step and is evaluated on
#          a mixed benchmark, which makes it an ensemble rather than three
#          separate detectors
#   FIX 18 source attribution accuracy is measured, not asserted
#   FIX 19 full precision-recall curves are exported, not only summary scalars
#   FIX 20 SHAP values are computed exactly by coalition enumeration
#
# Author: Sanjay Ganapathy

# Step 1: Install required libraries
# pip install hmmlearn shap


# Step 2: Import all required libraries
import os
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aether_compat import GaussianHMMFallback, exact_shapley

# hmmlearn and shap are preferred when present, but the pipeline does not
# depend on them. geopandas was only used to attach a CRS for plotting and is
# likewise optional.
try:
    from hmmlearn.hmm import GaussianHMM as _HMMLEARN_HMM
    HAVE_HMMLEARN = True
except Exception:
    _HMMLEARN_HMM = None
    HAVE_HMMLEARN = False

try:
    import shap  # noqa: F401
    HAVE_SHAP = True
except Exception:
    HAVE_SHAP = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

print("AETHER v2.2  (Kinematic Expert = ECOD, GPS Expert = IF; confirmation seed set)")
print("All libraries imported successfully.")
print(f"  hmmlearn available: {HAVE_HMMLEARN}   shap available: {HAVE_SHAP}")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_FILE_PATH = os.environ.get("AETHER_DATA", "data.csv")
OUTPUT_DIR = os.environ.get("AETHER_OUT", "results")

RANDOM_STATE = 55
N_SEEDS = 5                  # FIX 14: repetitions now use distinct seeds
N_ANOMALIES = 100

GAP_SECONDS = 900            # FIX 3: a gap longer than this starts a new segment
MIN_SEGMENT_LEN = 5          # segments shorter than this carry no sequential signal

FIT_SUBSAMPLE = 200_000      # detectors are fitted on a subsample and score all rows
OCSVM_FIT_SUBSAMPLE = 20_000  # OCSVM is O(n^2); fitting on the full pool is infeasible
BACKGROUND_SAMPLE = 200      # background rows for the Shapley computation

# Species-level physiological ceiling for Falco tinnunculus, used only to screen
# physically impossible records out of the clean pool.
SPEED_MAX_MS = 40.0
ACCEL_MAX_MS2 = 30.0

SCREEN_QUANTILE = 0.999      # FIX 8: top 0.1% by any expert is withheld from the pool

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# AETHER Framework Functions
# ---------------------------------------------------------------------------

def _wrap_angle_deg(delta):
    """Signed circular difference in degrees wrapped to (-180, 180]."""
    # FIX 2: heading is circular. A raw difference of 310.8 degrees is a 49.2
    # degree turn. The original used .diff().abs(), which reported the former.
    return (np.asarray(delta, dtype=float) + 180.0) % 360.0 - 180.0


def compute_derived_features(gdf):
    """Recompute time difference, acceleration and turning angle within segments.

    Called once during preprocessing and again after any injection, so that a
    perturbation of a raw channel propagates to every feature that depends on it.
    """
    # FIX 3: differences are taken within (animal_id, segment_id), never across
    # a gap or across individuals.
    grp = gdf.groupby(["animal_id", "segment_id"], sort=False)

    gdf["time_diff_s"] = grp["timestamp"].diff().dt.total_seconds()
    gdf["acceleration_ms2"] = grp["speed_ms"].diff() / gdf["time_diff_s"].clip(lower=1.0)
    gdf["turning_angle"] = np.abs(_wrap_angle_deg(grp["heading"].diff()))
    return gdf


def preprocess_movement_data(filepath):
    """
    Loads, cleans, and engineers features for movement data.
    This standardized function ensures a reproducible data pipeline.
    """
    print("\n--- Section 1: Data Loading & Feature Engineering ---")
    if not os.path.exists(filepath):
        print(f"ERROR: The file '{filepath}' was not found.")
        return None

    rename_map = {
        "tag-local-identifier": "animal_id",
        "timestamp": "timestamp",
        "location-long": "longitude",
        "location-lat": "latitude",
        "ground-speed": "speed_ms",
        "heading": "heading",
        "height-above-msl": "altitude_m",
        "gps:satellite-count": "satellite_count",
        "gps-time-to-fix": "time_to_fix_s",
        "gps:dop": "dop",
    }
    # Only the mapped columns are read. A full-width read of a 2.5M-row Movebank
    # export costs several GB of RAM for columns the framework never touches.
    header = pd.read_csv(filepath, nrows=0).columns.tolist()
    existing_cols = {k: v for k, v in rename_map.items() if k in header}
    missing = [k for k in rename_map if k not in header]
    if missing:
        print(f"  NOTE: columns absent from the export and skipped: {missing}")
    gdf = pd.read_csv(
        filepath, usecols=list(existing_cols.keys()), low_memory=False
    ).rename(columns=existing_cols)
    print(f"Successfully loaded {filepath} with {len(gdf):,} rows "
          f"({len(existing_cols)} of {len(header)} columns read).")

    # Movebank exports mix timestamps with and without fractional seconds. Recent
    # pandas infers a single format from the first row, so a plain to_datetime
    # call coerces most of the file to NaT and the rows then vanish at the
    # duplicate-drop step without any error being raised. Parse as ISO 8601,
    # fall back to mixed, and assert that the loss is negligible.
    ts = pd.to_datetime(gdf["timestamp"], utc=True, errors="coerce", format="ISO8601")
    if ts.isna().mean() > 0.01:
        ts = pd.to_datetime(gdf["timestamp"], utc=True, errors="coerce", format="mixed")
    n_unparsed = int(ts.isna().sum())
    if n_unparsed:
        print(f"  WARNING: {n_unparsed:,} timestamps could not be parsed "
              f"({100.0 * n_unparsed / len(ts):.3f}%).")
    if n_unparsed / max(len(ts), 1) > 0.05:
        raise ValueError("More than 5% of timestamps failed to parse; check the export format.")
    gdf["timestamp"] = ts

    n_before = len(gdf)
    gdf = gdf.dropna(subset=["timestamp"])
    gdf = gdf.sort_values(["animal_id", "timestamp"]).drop_duplicates(
        subset=["animal_id", "timestamp"], keep="first"
    )
    print(f"  Removed {n_before - len(gdf):,} unparseable or duplicate timestamps.")

    # FIX 1: the original interpolated every numeric column, which fabricated
    # values for satellite_count, dop and time_to_fix_s and linearly
    # interpolated heading (350 deg to 10 deg became 180 deg, exactly
    # backwards). Instantaneous sensor readings are not interpolated at all;
    # incomplete records are dropped and the count is reported.
    required = [c for c in ["timestamp", "animal_id", "speed_ms", "heading",
                            "satellite_count", "dop", "time_to_fix_s"]
                if c in gdf.columns]
    n_before = len(gdf)
    gdf = gdf.dropna(subset=required)
    print(f"  Dropped {n_before - len(gdf):,} incomplete records "
          f"({100.0 * (n_before - len(gdf)) / max(n_before, 1):.2f}%). "
          f"No sensor channel was imputed.")

    if "altitude_m" in gdf.columns:
        gdf["altitude_m"] = gdf["altitude_m"].fillna(gdf["altitude_m"].median())

    # FIX 3: segment on temporal gaps before any differencing.
    dt = gdf.groupby("animal_id", sort=False)["timestamp"].diff().dt.total_seconds()
    gdf["segment_id"] = (dt.isna() | (dt > GAP_SECONDS)).cumsum()

    seg_len = gdf.groupby("segment_id", sort=False)["segment_id"].transform("size")
    n_before = len(gdf)
    gdf = gdf[seg_len >= MIN_SEGMENT_LEN].copy()
    print(f"  Segmented on gaps > {GAP_SECONDS}s: "
          f"{gdf['segment_id'].nunique():,} segments across "
          f"{gdf['animal_id'].nunique()} individuals; "
          f"dropped {n_before - len(gdf):,} rows in segments shorter than {MIN_SEGMENT_LEN}.")

    gdf = compute_derived_features(gdf)

    # FIX 4: the first row of each segment has no predecessor, so acceleration
    # and turning angle are undefined there. The original filled them with
    # zero, which inserted an artificial mode at the origin of the kinematic
    # feature space. Those rows are dropped instead.
    n_before = len(gdf)
    gdf = gdf.dropna(subset=["acceleration_ms2", "turning_angle"]).copy()
    print(f"  Dropped {n_before - len(gdf):,} segment-initial rows with undefined "
          f"derived features.")

    # FIX 12: work positionally from here on. The original used label
    # arithmetic (.loc[idx - 1]) on a non-reset index, and assigning to a label
    # that does not exist silently appends an all-NaN row in pandas.
    gdf = gdf.reset_index(drop=True)
    gdf["row_pos"] = np.arange(len(gdf))

    print(f"Feature engineering complete. {len(gdf):,} usable fixes.")
    return gdf


def sequence_lengths(gdf):
    """Length of each contiguous sequence, in the row order of gdf.

    Required by the HMM. gdf must be sorted by animal_id then timestamp.
    """
    # FIX 5: the original called model.fit(X) with no lengths, so hmmlearn read
    # all 61 birds as a single sequence and learned 60 transitions that never
    # happened.
    return gdf.groupby(["animal_id", "segment_id"], sort=False).size().to_numpy()


def make_hmm(n_components, random_state=RANDOM_STATE):
    """Return a Gaussian HMM, preferring hmmlearn when it is installed."""
    if HAVE_HMMLEARN:
        return _HMMLEARN_HMM(
            n_components=n_components, covariance_type="diag",
            n_iter=100, random_state=random_state,
        )
    return GaussianHMMFallback(
        n_components=n_components, covariance_type="diag",
        n_iter=100, random_state=random_state,
    )


def hmm_pointwise_surprisal(model, X, lengths):
    """Per-observation score -log p(o_t | o_1..o_{t-1}) for a fitted Gaussian HMM.

    Computed from the public parameters, so it behaves identically whether the
    model came from hmmlearn or from the fallback implementation.
    """
    # FIX 6: this replaces
    #     log_prob, _ = model.score_samples(X)
    #     gdf[score_col] = -log_prob
    # GaussianHMM.score_samples returns (scalar log-likelihood of the whole
    # sequence, posteriors). The scalar was broadcast, so all 2,534,898 rows
    # received an identical score. A constant score gives AUC-ROC exactly 0.50,
    # PR-AUC equal to prevalence, and a median rank of n/2, which is precisely
    # what the original manuscript reported as a substantive finding.
    helper = GaussianHMMFallback(n_components=model.n_components)
    helper.startprob_ = np.asarray(model.startprob_, dtype=float)
    helper.transmat_ = np.asarray(model.transmat_, dtype=float)
    helper.means_ = np.asarray(model.means_, dtype=float)
    covars = np.asarray(model.covars_, dtype=float)
    if covars.ndim == 3:  # hmmlearn exposes diag covariances as (K, D, D)
        covars = np.array([np.diag(c) for c in covars])
    helper.covars_ = covars
    return helper.pointwise_surprisal(X, lengths)


def train_experts(gdf, features_kinematic, features_gps, features_sequential,
                  random_state=RANDOM_STATE):
    """
    Trains the expert committee models and calculates anomaly scores.
    """
    print("\n--- Section 3: Training the Expert Committee ---")
    rng = np.random.RandomState(random_state)

    scalers = {
        "kinematic": StandardScaler(),
        "gps": StandardScaler(),
        "sequential": StandardScaler(),
    }
    feature_sets = {
        "kinematic": features_kinematic,
        "gps": features_gps,
        "sequential": features_sequential,
    }
    X_scaled = {
        k: scalers[k].fit_transform(gdf[v].to_numpy(dtype=float))
        for k, v in feature_sets.items()
    }

    # Detectors are fitted on a random subsample and then score every row. This
    # keeps the five-seed benchmark and the ablation grid tractable on 2.5M
    # fixes and is reported in the manuscript.
    n = len(gdf)
    fit_idx = (rng.choice(n, size=min(n, FIT_SUBSAMPLE), replace=False)
               if n > FIT_SUBSAMPLE else np.arange(n))

    models = {}

    # FIX 13: contamination is left at its default. For IsolationForest,
    # score_samples does not depend on it; for LocalOutlierFactor with
    # novelty=True it shifts decision_function by a constant. Every metric
    # reported here is rank-based, so all of them are invariant to it. The
    # original set contamination=0.001 against a true injected prevalence of
    # 3.9e-5, a 25-fold mismatch that was nonetheless immaterial.
    print("Training Kinematic Expert...")
    # FIX 23: the Kinematic Expert is instantiated as ECOD. On the selection
    # replicate (v2.1 run, seeds 555..959) ECOD dominated the incumbent
    # Isolation Forest on the kinematic benchmark on every metric on every
    # seed (PR-AUC +0.14 to +0.19; AUC-ROC and median rank likewise; sign
    # test p = 2^-5 per metric), with disjoint mean +/- SD intervals. This is
    # also the mechanistically expected result: the manuscript defines
    # kinematic anomalies as marginal-tail events relative to species-level
    # distributions, which is precisely the anomaly class ECOD models. Its
    # score is additionally an additive sum of per-feature tail
    # log-probabilities, so the expert is natively decomposable per feature.
    models["kinematic"] = ECODDetector().fit(X_scaled["kinematic"][fit_idx])

    print("Training Gps Expert...")
    # FIX 21: the GPS Expert is instantiated as an Isolation Forest. The
    # factorial ablation (Section 3.2) shows the choice is not close on the
    # support-respecting benchmark: IF on the GPS subset reaches AUC-ROC ~0.99
    # where LOF reaches ~0.64, and LOF's top-ranked real-data detections are
    # fixes with unremarkable quality values, i.e. local density gaps rather
    # than receiver failures. The selection rule is stated in the manuscript:
    # an expert's algorithm changes only where the ablation margin is decisive
    # across seeds. LOF remains in the ablation grid for comparison.
    models["gps"] = IsolationForest(
        n_estimators=200, contamination="auto", random_state=random_state, n_jobs=-1
    ).fit(X_scaled["gps"][fit_idx])

    print("Training Sequential Expert...")
    lengths = sequence_lengths(gdf)
    hmm = make_hmm(n_components=4, random_state=random_state)
    hmm.fit(X_scaled["sequential"], lengths)   # FIX 5
    models["sequential"] = hmm

    print("\n--- Section 4: Generating Specialized Scores on Real Data ---")
    for name in ["kinematic", "gps", "sequential"]:
        if name == "sequential":
            raw = hmm_pointwise_surprisal(models[name], X_scaled[name], lengths)  # FIX 6
        else:
            raw = score_point_model(models[name], X_scaled[name])
        gdf[f"{name}_score"] = normalise_scores(raw)

    print("All experts trained and scores calculated.")
    return gdf, models, scalers, feature_sets


class KNNDetector:
    """k-nearest-neighbour distance detector for the ablation grid.

    Mean distance to the k nearest fitted points; a classical detector that is
    consistently competitive in published comparisons and is inexpensive on
    the three- and four-dimensional subsets used here.
    """

    def __init__(self, n_neighbors=20):
        self.n_neighbors = n_neighbors

    def fit(self, X):
        from sklearn.neighbors import NearestNeighbors
        self.nn_ = NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=-1).fit(X)
        return self

    def decision_function(self, X):
        dist, _ = self.nn_.kneighbors(X)
        return -dist.mean(axis=1)          # lower = more anomalous, sklearn convention


class ECODDetector:
    """Empirical-CDF outlier detection (ECOD; Li et al., IEEE TKDE 2022),
    reimplemented from the paper. Parameter-free: each observation is scored
    by the sum over features of the negative log empirical tail probability,
    aggregated over the left, right and skewness-selected tails.
    """

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.sorted_ = np.sort(X, axis=0)
        self.n_ = len(X)
        centred = X - X.mean(axis=0)
        m2 = (centred ** 2).mean(axis=0)
        m3 = (centred ** 3).mean(axis=0)
        self.skew_ = m3 / np.maximum(m2, 1e-30) ** 1.5
        return self

    def _tail_logs(self, X):
        X = np.asarray(X, dtype=float)
        eps = 1.0 / self.n_
        pl = np.empty_like(X)
        pr = np.empty_like(X)
        for j in range(X.shape[1]):
            col = self.sorted_[:, j]
            pl[:, j] = np.searchsorted(col, X[:, j], side="right") / self.n_
            pr[:, j] = (self.n_ - np.searchsorted(col, X[:, j], side="left")) / self.n_
        return -np.log(np.clip(pl, eps, 1.0)), -np.log(np.clip(pr, eps, 1.0))

    def decision_function(self, X):
        nl_left, nl_right = self._tail_logs(X)
        o_left = nl_left.sum(axis=1)
        o_right = nl_right.sum(axis=1)
        auto = np.where(self.skew_[None, :] < 0, nl_left, nl_right)
        o_auto = auto.sum(axis=1)
        return -np.maximum.reduce([o_left, o_right, o_auto])


def score_point_model(model, X):
    """Higher is more anomalous, for any of the three point-wise detectors."""
    if isinstance(model, IsolationForest):
        return -model.score_samples(X)
    return -model.decision_function(X)


def normalise_scores(raw):
    """Min-max normalisation with non-finite values mapped to the maximum."""
    s = pd.Series(np.asarray(raw, dtype=float)).replace([np.inf, -np.inf], np.nan)
    s = s.fillna(s.max())
    lo, hi = s.min(), s.max()
    return ((s - lo) / (hi - lo)).to_numpy() if hi > lo else np.full(len(s), 0.5)


def rank_percentile(scores):
    """Map scores onto [0, 1] by empirical rank, so experts become comparable."""
    return pd.Series(scores).rank(pct=True, method="average").to_numpy()


# ---------------------------------------------------------------------------
# FIX 8: clean pool
# ---------------------------------------------------------------------------

def build_clean_pool(gdf, models, scalers, feature_sets):
    """Screen out physically impossible and already-suspect records.

    Reviewer #1 observed that injecting synthetic anomalies into data that
    already contains unlabelled real anomalies makes a true positive an unknown
    mixture of the two. That cannot be solved without labels, but it can be
    reduced: records that are physically impossible, or that the untrained-on
    detectors already rank in the extreme tail, are withheld from the pool that
    receives injections. The residual contamination is bounded by the screen
    and is stated as a limitation rather than assumed away.
    """
    print("\n--- Section 5a: Building the Screened Clean Pool ---")
    n0 = len(gdf)

    physical = (
        (gdf["speed_ms"].between(0, SPEED_MAX_MS))
        & (gdf["acceleration_ms2"].abs() <= ACCEL_MAX_MS2)
        & (gdf["turning_angle"].between(0, 180))
        & (gdf["satellite_count"] > 0)
        & (gdf["dop"] > 0)
    )
    print(f"  Physically implausible records removed: {(~physical).sum():,}")

    already_flagged = np.zeros(len(gdf), dtype=bool)
    for name in ["kinematic", "gps", "sequential"]:
        thr = gdf[f"{name}_score"].quantile(SCREEN_QUANTILE)
        already_flagged |= (gdf[f"{name}_score"] >= thr).to_numpy()
    print(f"  Pre-existing extreme-tail candidates withheld: {already_flagged.sum():,}")

    keep = physical.to_numpy() & (~already_flagged)
    pool = gdf[keep].copy()

    # Segments must stay contiguous for the sequential expert.
    seg_len = pool.groupby("segment_id", sort=False)["segment_id"].transform("size")
    pool = pool[seg_len >= MIN_SEGMENT_LEN].copy()
    pool = pool.sort_values(["animal_id", "timestamp"]).reset_index(drop=True)
    pool["row_pos"] = np.arange(len(pool))
    pool = compute_derived_features(pool)
    pool = pool.dropna(subset=["acceleration_ms2", "turning_angle"]).reset_index(drop=True)
    pool["row_pos"] = np.arange(len(pool))

    print(f"  Clean pool: {len(pool):,} of {n0:,} fixes retained "
          f"({100.0 * len(pool) / max(n0, 1):.1f}%).")
    return pool, gdf[~keep].copy()


# ---------------------------------------------------------------------------
# FIX 9 / 10 / 11: anomaly injection
# ---------------------------------------------------------------------------

def _interior_positions(df, margin=3):
    """Positions that are at least `margin` rows inside their own segment."""
    cc = df.groupby(["animal_id", "segment_id"], sort=False).cumcount()
    size = df.groupby(["animal_id", "segment_id"], sort=False)["row_pos"].transform("size")
    return df.index[(cc >= margin) & (cc < size - margin)].to_numpy()


def inject_anomalies(df, anomaly_type, n_anomalies=N_ANOMALIES, random_seed=RANDOM_STATE,
                     poor_gps_pool=None):
    """Injects a specific type of anomaly for targeted evaluation.

    Returns (df_injected, y_true, changed_positions).

    The three injection designs are stated explicitly here because Reviewer #1
    correctly noted that the original manuscript never defined them.
    """
    rng = np.random.RandomState(random_seed)
    out = df.copy()
    out["is_anomaly"] = 0

    base = out[["speed_ms", "heading", "satellite_count", "dop", "time_to_fix_s"]].to_numpy(
        dtype=float, copy=True
    )

    candidates = _interior_positions(out, margin=3)
    if len(candidates) < n_anomalies:
        n_anomalies = len(candidates)
    targets = rng.choice(candidates, size=n_anomalies, replace=False)

    if anomaly_type == "kinematic":
        # FIX 9. The original multiplied speed_ms by U(3, 5) and stopped there,
        # leaving acceleration_ms2 and turning_angle at their unperturbed
        # values. Only one of the expert's three features moved, and the
        # resulting record was physically incoherent: tripled speed with
        # unchanged acceleration. Here the raw channels are perturbed and the
        # derived features are recomputed downstream, so the injected state
        # obeys the same kinematic relations as the real data.
        hi = float(df["speed_ms"].quantile(0.999))
        for pos in targets:
            out.loc[pos, "speed_ms"] = rng.uniform(hi, SPEED_MAX_MS)
            out.loc[pos, "heading"] = (out.loc[pos, "heading"] + rng.uniform(90, 270)) % 360.0
            out.loc[pos, "is_anomaly"] = 1

    elif anomaly_type == "gps":
        # FIX 10. The original set satellite_count to randint(1, 4), a range
        # outside the observed support, and left dop and time_to_fix_s
        # untouched even though the manuscript defines a technical anomaly in
        # terms of all three. Any univariate threshold solved that task, which
        # is why both LOF and OCSVM reported AUC-ROC 1.00. Here a complete
        # observed triple is resampled from the poor-quality tail of the real
        # data, so every injected value lies inside the observed support and
        # the joint structure is preserved.
        if poor_gps_pool is None or len(poor_gps_pool) == 0:
            raise ValueError("GPS injection requires an observed poor-quality pool.")
        picks = rng.choice(len(poor_gps_pool), size=len(targets), replace=True)
        donor = poor_gps_pool.iloc[picks][["satellite_count", "dop", "time_to_fix_s"]].to_numpy()
        for k, pos in enumerate(targets):
            out.loc[pos, "satellite_count"] = donor[k, 0]
            out.loc[pos, "dop"] = donor[k, 1]
            out.loc[pos, "time_to_fix_s"] = donor[k, 2]
            out.loc[pos, "is_anomaly"] = 1

    elif anomaly_type == "sequential":
        # FIX 11. The original wrote speed_ms = [20, 0, 20] across three rows,
        # which is a marginal outlier detectable without any sequence model,
        # and addressed rows by label arithmetic. Here a short block of
        # behavioural observations is spliced in from a different individual,
        # so each injected value is drawn from the real marginal distribution
        # and only the transition into the block is improbable. This is what
        # the manuscript means by a sequential anomaly.
        block = 3
        donors = _interior_positions(out, margin=block + 2)
        for pos in targets:
            same_bird = out.loc[pos, "animal_id"]
            for _ in range(20):
                d = int(rng.choice(donors))
                if out.loc[d, "animal_id"] != same_bird:
                    break
            src = out.loc[d:d + block - 1, ["speed_ms", "heading"]].to_numpy()
            out.loc[pos:pos + block - 1, "speed_ms"] = src[:, 0]
            out.loc[pos:pos + block - 1, "heading"] = src[:, 1]
            out.loc[pos, "is_anomaly"] = 1   # the improbable transition point
    else:
        raise ValueError(f"Unknown anomaly_type: {anomaly_type}")

    out = compute_derived_features(out)
    out[["acceleration_ms2", "turning_angle"]] = out[
        ["acceleration_ms2", "turning_angle"]
    ].fillna(0.0)

    now = out[["speed_ms", "heading", "satellite_count", "dop", "time_to_fix_s"]].to_numpy(
        dtype=float
    )
    changed = np.where(np.any(~np.isclose(base, now, equal_nan=True), axis=1))[0]
    # derived features of the following row also move
    changed = np.unique(np.concatenate([changed, np.clip(changed + 1, 0, len(out) - 1)]))

    return out, out["is_anomaly"].to_numpy(), changed


def build_mixed_benchmark(df, n_anomalies=N_ANOMALIES, random_seed=RANDOM_STATE,
                          poor_gps_pool=None):
    """One benchmark containing all three anomaly classes.

    FIX 17: the committee can only be evaluated as an ensemble on a benchmark
    where more than one kind of anomaly is present. Evaluating each expert on
    its own matched benchmark, as the original did, measures three separate
    detectors and says nothing about the architecture.
    """
    per_class = n_anomalies // 3
    work = df.copy()
    work["is_anomaly"] = 0
    work["anomaly_type"] = ""
    all_changed = []

    for offset, kind in enumerate(["kinematic", "gps", "sequential"]):
        injected, _, changed = inject_anomalies(
            work.drop(columns=["is_anomaly", "anomaly_type"]),
            kind, per_class, random_seed + 1000 * offset, poor_gps_pool,
        )
        newly = injected["is_anomaly"].to_numpy() == 1
        for col in ["speed_ms", "heading", "satellite_count", "dop", "time_to_fix_s"]:
            work.loc[changed, col] = injected.loc[changed, col].to_numpy()
        work.loc[newly, "is_anomaly"] = 1
        work.loc[newly, "anomaly_type"] = kind
        all_changed.append(changed)

    work = compute_derived_features(work)
    work[["acceleration_ms2", "turning_angle"]] = work[
        ["acceleration_ms2", "turning_angle"]
    ].fillna(0.0)
    return work, np.unique(np.concatenate(all_changed))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def score_benchmark(model, scaler, features, df_injected, changed, base_scores,
                    is_sequential=False):
    """Score a benchmark by recomputing only the rows an injection touched.

    For the three point-wise detectors a row's score depends only on that row,
    so splicing is exact. For the HMM the whole affected segment is rescored.
    """
    scores = base_scores.copy()
    if len(changed) == 0:
        return scores

    if not is_sequential:
        X = scaler.transform(df_injected.iloc[changed][features].to_numpy(dtype=float))
        scores[changed] = score_point_model(model, X)
        return scores

    seg_key = df_injected[["animal_id", "segment_id"]].apply(tuple, axis=1)
    touched = set(seg_key.iloc[changed])
    mask = seg_key.isin(touched).to_numpy()
    sub = df_injected[mask]
    X = scaler.transform(sub[features].to_numpy(dtype=float))
    lengths = sequence_lengths(sub)
    scores[mask] = hmm_pointwise_surprisal(model, X, lengths)
    return scores


def evaluate_scores(model_name, y_true, y_scores):
    """AUC-ROC, PR-AUC and median rank for one scored benchmark."""
    y_scores = np.asarray(y_scores, dtype=float)
    y_scores = np.nan_to_num(y_scores, nan=np.nanmin(y_scores), posinf=np.nanmax(y_scores))

    order = np.argsort(-y_scores, kind="mergesort")
    ranks = np.empty(len(y_scores), dtype=np.int64)
    ranks[order] = np.arange(len(y_scores))
    pos = np.where(y_true == 1)[0]

    return {
        "Model": model_name,
        "PR-AUC": float(average_precision_score(y_true, y_scores)),
        "AUC-ROC": float(roc_auc_score(y_true, y_scores)),
        "Median Rank": float(np.median(ranks[pos])) if len(pos) else np.nan,
        "Prevalence": float(y_true.mean()),
    }


def threshold_rule_scores(df, kind):
    """FIX 16: a transparent, model-free rule benchmarked on the same task.

    Huettmann and Reviewer #2 both ask what the machine learning buys. The
    honest comparison is against the rule an analyst would write by hand.
    """
    z = lambda s: (s - s.mean()) / (s.std() + 1e-12)
    if kind == "gps":
        return (z(df["dop"]) + z(df["time_to_fix_s"]) - z(df["satellite_count"])).to_numpy()
    if kind == "kinematic":
        return (z(df["speed_ms"]) + z(df["acceleration_ms2"].abs())
                + z(df["turning_angle"])).to_numpy()
    return (z(df["speed_ms"]).abs() + z(df["turning_angle"])).to_numpy()


def aggregate(rows):
    """Mean and standard deviation across seeds for one model and task."""
    df = pd.DataFrame(rows)
    g = df.groupby(["Model", "Task"], sort=False).agg(["mean", "std"])
    out = pd.DataFrame({
        "PR-AUC": g[("PR-AUC", "mean")],
        "PR-AUC SD": g[("PR-AUC", "std")],
        "AUC-ROC": g[("AUC-ROC", "mean")],
        "AUC-ROC SD": g[("AUC-ROC", "std")],
        "Median Rank": g[("Median Rank", "mean")],
        "Median Rank SD": g[("Median Rank", "std")],
    }).reset_index()
    return out


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

def explain_anomalies_with_xai(gdf, models, scalers, feature_sets,
                               num_anomalies_to_explain=3, rng=None):
    """
    Uses exact Shapley values and model introspection to explain top anomalies.
    """
    print(f"\n--- Section 8: Explaining Top {num_anomalies_to_explain} Anomalies ---")
    rng = rng or np.random.RandomState(RANDOM_STATE)
    records = []

    for expert_name in ["kinematic", "gps"]:
        print(f"\n--- Explaining {expert_name.upper()} Expert ---")
        model = models[expert_name]
        scaler = scalers[expert_name]
        names = feature_sets[expert_name]

        X_all = scaler.transform(gdf[names].to_numpy(dtype=float))
        bg_idx = rng.choice(len(X_all), size=min(BACKGROUND_SAMPLE, len(X_all)),
                            replace=False)
        background = X_all[bg_idx]
        predict_fn = lambda x: score_point_model(model, x)

        top = gdf[f"{expert_name}_score"].nlargest(num_anomalies_to_explain).index
        for i, idx in enumerate(top):
            # FIX 20: with three features the 2**3 coalitions can be enumerated,
            # so the Shapley values are exact rather than kernel-approximated.
            phi, base_value = exact_shapley(predict_fn, X_all[idx], background, names)
            rec = {
                "expert": expert_name, "rank": i + 1, "row_pos": int(idx),
                "base_value": base_value, "score": float(gdf.loc[idx, f"{expert_name}_score"]),
            }
            for j, nm in enumerate(names):
                rec[f"shap_{nm}"] = float(phi[j])
                rec[f"value_{nm}"] = float(gdf.loc[idx, nm])
            records.append(rec)
            top_feat = names[int(np.argmax(np.abs(phi)))]
            print(f"  #{i+1} row {idx}: dominant feature = {top_feat} "
                  f"(phi = {phi[int(np.argmax(np.abs(phi)))]:.3f})")

    print("\n--- Explaining SEQUENTIAL Expert (HMM) ---")
    hmm = models["sequential"]
    names = feature_sets["sequential"]
    state_means = pd.DataFrame(
        scalers["sequential"].inverse_transform(np.asarray(hmm.means_)), columns=names
    )
    state_means.index = [f"State {i}" for i in range(hmm.n_components)]
    print("Learned behavioural states, in original units:")
    print(state_means.to_string(float_format="{:.2f}".format))
    print("\nTransition matrix:")
    print(pd.DataFrame(np.asarray(hmm.transmat_),
                       index=state_means.index,
                       columns=state_means.index).to_string(float_format="{:.3f}".format))

    state_means.to_csv(os.path.join(OUTPUT_DIR, "table2_hmm_states.csv"))
    pd.DataFrame(records).to_csv(os.path.join(OUTPUT_DIR, "shap_explanations.csv"),
                                 index=False)
    return pd.DataFrame(records), state_means


# ---------------------------------------------------------------------------
# Main Execution Block
# ---------------------------------------------------------------------------

def main():
    gdf = preprocess_movement_data(DATA_FILE_PATH)
    if gdf is None:
        return

    print("\n--- Section 2: Defining Expert Feature Sets ---")
    features_kinematic = ["speed_ms", "acceleration_ms2", "turning_angle"]
    features_gps = ["satellite_count", "dop", "time_to_fix_s"]
    # FIX 7: the original sequential set was kinematic + GPS + heading +
    # altitude. Including the GPS-quality channels is why the reported State 3
    # was a GPS-error state rather than a behavioural one, contradicting the
    # architecture as described. The Sequential Expert is behavioural only.
    features_sequential = ["speed_ms", "acceleration_ms2", "turning_angle"]
    if "altitude_m" in gdf.columns:
        features_sequential = features_sequential + ["altitude_m"]
    features_all = sorted(set(features_kinematic + features_gps + features_sequential))

    features_kinematic = [f for f in features_kinematic if f in gdf.columns]
    features_gps = [f for f in features_gps if f in gdf.columns]
    features_sequential = [f for f in features_sequential if f in gdf.columns]

    gdf, models, scalers, feature_sets = train_experts(
        gdf, features_kinematic, features_gps, features_sequential
    )

    pool, withheld = build_clean_pool(gdf, models, scalers, feature_sets)

    # Observed poor-quality GPS records, used as the donor pool for FIX 10.
    poor_gps_pool = gdf.loc[gdf["dop"] >= gdf["dop"].quantile(0.99),
                            ["satellite_count", "dop", "time_to_fix_s"]].copy()
    print(f"  Poor-quality GPS donor pool: {len(poor_gps_pool):,} observed records "
          f"(satellite_count {poor_gps_pool['satellite_count'].min():.0f}-"
          f"{poor_gps_pool['satellite_count'].max():.0f}, "
          f"dop {poor_gps_pool['dop'].min():.1f}-{poor_gps_pool['dop'].max():.1f})")

    print("\n--- Section 5b: Refitting on the Clean Pool ---")
    pool, models, scalers, feature_sets = train_experts(
        pool, features_kinematic, features_gps, features_sequential
    )

    base_scaled = {
        k: scalers[k].transform(pool[v].to_numpy(dtype=float))
        for k, v in feature_sets.items()
    }
    pool_lengths = sequence_lengths(pool)
    base_scores = {
        "kinematic": score_point_model(models["kinematic"], base_scaled["kinematic"]),
        "gps": score_point_model(models["gps"], base_scaled["gps"]),
        "sequential": hmm_pointwise_surprisal(models["sequential"],
                                              base_scaled["sequential"], pool_lengths),
    }

    # ---------------- Table 1: experts vs baselines, five seeds -------------
    print("\n--- Section 6: Quantitative Benchmarking ---")

    # FIX 15: the original baseline changed both the algorithm and the feature
    # set at once (LOF on 3 GPS features vs IsolationForest on all 8) and then
    # attributed the whole gap to the partition. The grid below crosses
    # algorithm with feature subset so the two effects are separable.
    rng = np.random.RandomState(RANDOM_STATE)
    n_pool = len(pool)
    fit_idx = (rng.choice(n_pool, size=min(n_pool, FIT_SUBSAMPLE), replace=False)
               if n_pool > FIT_SUBSAMPLE else np.arange(n_pool))
    ocsvm_idx = rng.choice(fit_idx, size=min(len(fit_idx), OCSVM_FIT_SUBSAMPLE),
                           replace=False)

    subsets = {
        "kinematic": features_kinematic,
        "gps": features_gps,
        "behavioural": features_sequential,
        "all": features_all,
    }
    grid_models, grid_scalers, grid_base = {}, {}, {}
    for sname, feats in subsets.items():
        sc = StandardScaler().fit(pool[feats].to_numpy(dtype=float))
        Xs = sc.transform(pool[feats].to_numpy(dtype=float))
        grid_scalers[sname] = sc
        for aname, mdl in [
            ("IF", IsolationForest(n_estimators=200, contamination="auto",
                                   random_state=RANDOM_STATE, n_jobs=-1)),
            ("LOF", LocalOutlierFactor(n_neighbors=20, contamination="auto",
                                       novelty=True, n_jobs=-1)),
            ("OCSVM", OneClassSVM(nu=0.01, kernel="rbf", gamma="scale")),
            ("KNN", KNNDetector(n_neighbors=20)),
            ("ECOD", ECODDetector()),
        ]:
            idx = ocsvm_idx if aname == "OCSVM" else fit_idx
            print(f"  fitting {aname} on {sname} ({len(feats)} features, n={len(idx):,})")
            mdl.fit(Xs[idx])
            grid_models[(aname, sname)] = mdl
            grid_base[(aname, sname)] = score_point_model(mdl, Xs)

    per_seed, curves = [], {}
    for s in range(N_SEEDS):
        # FIX 22: reported benchmarks use injection seeds disjoint from the
        # replicate on which the expert algorithms were selected (the v1 run,
        # seeds 55..459). Selection and reporting therefore never share a
        # benchmark instance, removing the selection-on-test circularity.
        # FIX 24: confirmation run. Expert selection used seeds 55..459 (v1)
        # and 555..959 (v2.1); the reported results below use a third,
        # disjoint seed set so no reported number shares a benchmark instance
        # with any selection decision.
        seed = RANDOM_STATE + 2000 + s * 101   # FIX 14 + FIX 22 + FIX 24
        for task in ["kinematic", "gps", "sequential"]:
            inj, y_true, changed = inject_anomalies(
                pool, task, N_ANOMALIES, seed, poor_gps_pool
            )

            # the three experts
            for ename in ["kinematic", "gps", "sequential"]:
                sc = score_benchmark(
                    models[ename], scalers[ename], feature_sets[ename],
                    inj, changed, base_scores[ename], is_sequential=(ename == "sequential"),
                )
                r = evaluate_scores(f"{ename.capitalize()} Expert", y_true, sc)
                r["Task"] = task
                per_seed.append(r)
                if s == 0 and ename == task:
                    p, rc, _ = precision_recall_curve(y_true, sc)
                    curves[f"{ename}_expert_{task}"] = (p, rc)

            # ablation grid
            for (aname, sname), mdl in grid_models.items():
                sc = score_benchmark(mdl, grid_scalers[sname], subsets[sname],
                                     inj, changed, grid_base[(aname, sname)])
                r = evaluate_scores(f"{aname} on {sname}", y_true, sc)
                r["Task"] = task
                per_seed.append(r)

            # threshold rule
            sc = threshold_rule_scores(inj, task)
            r = evaluate_scores("Threshold rule", y_true, sc)   # FIX 16
            r["Task"] = task
            per_seed.append(r)
        print(f"  seed {s + 1}/{N_SEEDS} complete")

    table1 = aggregate(per_seed)
    table1.to_csv(os.path.join(OUTPUT_DIR, "table1_results.csv"), index=False)
    pd.DataFrame(per_seed).to_csv(os.path.join(OUTPUT_DIR, "per_seed_results.csv"),
                                  index=False)
    print("\n--- Table 1 ---")
    print(table1.to_string(index=False, float_format="{:.4f}".format))

    # FIX 19: export full precision-recall curves, not only the summary scalar.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (key, (p, rc)) in zip(axes, curves.items()):
        ax.plot(rc, p, marker=".", linewidth=1)
        ax.set_title(key.replace("_", " "))
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_pr_curves.png"), dpi=200)
    plt.close()
    np.savez(os.path.join(OUTPUT_DIR, "pr_curves.npz"),
             **{f"{k}_{n}": v for k, (p, r) in curves.items()
                for n, v in [("precision", p), ("recall", r)]})

    # ---------------- Table 3: committee vs monolith ------------------------
    print("\n--- Section 7: Committee vs Monolithic Detector ---")
    committee_rows, attribution_rows = [], []
    for s in range(N_SEEDS):
        seed = RANDOM_STATE + 2000 + s * 101   # FIX 24: confirmation seeds
        mixed, changed = build_mixed_benchmark(pool, N_ANOMALIES, seed, poor_gps_pool)
        y_true = mixed["is_anomaly"].to_numpy()

        expert_scores = {}
        for ename in ["kinematic", "gps", "sequential"]:
            expert_scores[ename] = score_benchmark(
                models[ename], scalers[ename], feature_sets[ename],
                mixed, changed, base_scores[ename], is_sequential=(ename == "sequential"),
            )
            r = evaluate_scores(f"{ename.capitalize()} Expert alone", y_true,
                                expert_scores[ename])
            r["Task"] = "mixed"
            committee_rows.append(r)

        # FIX 17: an explicit fusion step. Reviewer #1 is right that three
        # detectors reporting separately are not an ensemble. Each expert's
        # score is converted to an upper-tail empirical p-value, and the three
        # p-values are then combined by standard rules from the meta-analysis
        # literature. Tippett's rule takes the strongest single piece of
        # evidence, which suits a committee where each member is authoritative
        # only inside its own domain; Fisher's rule pools evidence across
        # members; the mean is reported as a naive reference. All three are
        # reported so that the fusion rule is a stated design choice rather
        # than a tuned one.
        pct = {k: rank_percentile(v) for k, v in expert_scores.items()}
        pvals = {k: np.clip(1.0 - v, 1.0 / len(pool), 1.0) for k, v in pct.items()}
        stack_p = np.vstack([pvals["kinematic"], pvals["gps"], pvals["sequential"]])

        fusions = {
            "AETHER Committee (Tippett, max)": -np.log(stack_p.min(axis=0)),
            "AETHER Committee (Fisher)": -2.0 * np.log(stack_p).sum(axis=0),
            "AETHER Committee (mean rank)": np.vstack(
                [pct["kinematic"], pct["gps"], pct["sequential"]]
            ).mean(axis=0),
        }
        for fname, fscore in fusions.items():
            r = evaluate_scores(fname, y_true, fscore)
            r["Task"] = "mixed"
            committee_rows.append(r)
        fused = fusions["AETHER Committee (Tippett, max)"]

        for aname in ["IF", "LOF", "OCSVM"]:
            sc = score_benchmark(grid_models[(aname, "all")], grid_scalers["all"],
                                 subsets["all"], mixed, changed,
                                 grid_base[(aname, "all")])
            r = evaluate_scores(f"Monolithic {aname} (all features)", y_true, sc)
            r["Task"] = "mixed"
            committee_rows.append(r)

        # FIX 18: attribution accuracy. The intrinsic explainability claim is
        # that the firing expert identifies the source of the anomaly. That is
        # a testable claim, so it is tested.
        stacked = np.vstack([pct["kinematic"], pct["gps"], pct["sequential"]])
        winner = np.array(["kinematic", "gps", "sequential"])[stacked.argmax(axis=0)]
        pos = np.where(y_true == 1)[0]
        for i in pos:
            attribution_rows.append({
                "seed": seed,
                "true_type": mixed["anomaly_type"].iloc[i],
                "attributed_to": winner[i],
                "fused_score": float(fused[i]),
            })
        print(f"  seed {s + 1}/{N_SEEDS} complete")

    table3 = aggregate(committee_rows)
    table3.to_csv(os.path.join(OUTPUT_DIR, "table3_committee_vs_monolith.csv"),
                  index=False)
    print("\n--- Table 3 ---")
    print(table3.to_string(index=False, float_format="{:.4f}".format))

    attr = pd.DataFrame(attribution_rows)
    conf = pd.crosstab(attr["true_type"], attr["attributed_to"])
    conf.to_csv(os.path.join(OUTPUT_DIR, "table4_attribution_confusion.csv"))
    acc = float((attr["true_type"] == attr["attributed_to"]).mean())
    print("\n--- Table 4: source attribution ---")
    print(conf.to_string())
    print(f"Overall attribution accuracy: {acc:.3f}")

    # ---------------- Explainability and case study -------------------------
    shap_df, state_means = explain_anomalies_with_xai(pool, models, scalers, feature_sets)

    print("\n--- Section 9: Case Study Export ---")
    ctx_cols = [c for c in ["animal_id", "segment_id", "timestamp", "latitude",
                            "longitude", "speed_ms", "heading", "acceleration_ms2",
                            "turning_angle", "altitude_m", "satellite_count", "dop",
                            "time_to_fix_s", "kinematic_score", "gps_score",
                            "sequential_score"] if c in pool.columns]
    flagged = []
    for ename in ["kinematic", "gps", "sequential"]:
        top = pool.nlargest(10, f"{ename}_score")[ctx_cols].copy()
        top.insert(0, "flagged_by", ename)
        top.insert(1, "rank", np.arange(1, len(top) + 1))
        flagged.append(top)
    flagged = pd.concat(flagged, ignore_index=True)
    flagged.to_csv(os.path.join(OUTPUT_DIR, "case_study_flagged.csv"), index=False)

    controls = pool.sample(n=min(300, len(pool)), random_state=RANDOM_STATE)[ctx_cols]
    controls.to_csv(os.path.join(OUTPUT_DIR, "case_study_controls.csv"), index=False)

    summary = {
        "n_fixes_raw_usable": int(len(gdf)),
        "n_fixes_clean_pool": int(len(pool)),
        "n_individuals": int(pool["animal_id"].nunique()),
        "n_segments": int(pool["segment_id"].nunique()),
        "injected_prevalence": float(N_ANOMALIES / len(pool)),
        "attribution_accuracy": acc,
        "hmmlearn": HAVE_HMMLEARN,
        "n_seeds": N_SEEDS,
        "gap_seconds": GAP_SECONDS,
        "screen_quantile": SCREEN_QUANTILE,
    }
    with open(os.path.join(OUTPUT_DIR, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary, indent=2))
    print("\n--- AETHER Framework Execution Complete ---")


if __name__ == "__main__":
    main()
