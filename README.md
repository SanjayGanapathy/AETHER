# AETHER

**Anomaly Ensemble framework for Trajectory Heuristics and Explainable Reasoning**

An explainable, unsupervised anomaly-detection framework for animal movement data. AETHER organises detection as an "Expert Committee": three unsupervised detectors, each trained on a disjoint, ecologically meaningful feature subset, fused through classical p-value combination rules, with source attribution (which expert fired) and exact Shapley feature attribution for every detection.

Manuscript: *An Explainable Unsupervised Machine Learning Anomaly Detection Framework for Ecological Investigation in Anomalous Animal Movement Data*, under review at Ecological Informatics (ECOINF-D-25-02580).

## Repository contents

| File | Purpose |
|---|---|
| `AETHER.py` | The full pipeline: preprocessing, expert training, benchmark construction, evaluation, fusion, attribution, explainability, case-study export. |
| `aether_compat.py` | Fallback implementations used when `hmmlearn` or `shap` are not installed: a batched Gaussian HMM (Baum–Welch, per-observation surprisal scoring) and exact Shapley values by coalition enumeration. |
| `verify_fixes.py` | Regression suite (25 checks) that validates the implementation on synthetic data where the correct answer is known analytically. Run this first. |
| `sensitivity.py` | Hyperparameter sweeps: Isolation Forest `n_estimators` (GPS Expert) and HMM `n_components` (Sequential Expert). Produces Figure 4. |
| `appendixD_full_grid.csv` | Complete ablation grid (5 algorithms × 4 feature subsets × 3 benchmarks), mean ± SD over five confirmation seeds. |
| `AETHER.ipynb` | Notebook version of the original pipeline (superseded by `AETHER.py`; retained for history). |

## The committee

| Expert | Features | Algorithm | Selected by |
|---|---|---|---|
| Kinematic | speed, acceleration, turning angle | ECOD | ablation protocol (Section 2.3 of the manuscript) |
| GPS | satellite count, DOP, time-to-fix | Isolation Forest | ablation protocol |
| Sequential | speed, acceleration, turning angle, altitude | Gaussian HMM (per-observation predictive surprisal) | only sequence model considered |

Each expert's score is converted to an upper-tail empirical p-value by rank; the three are fused by Tippett's rule (primary) and Fisher's rule. Algorithm selection used two benchmark replicates; all reported numbers come from a third, disjoint seed set.

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install numpy pandas scipy scikit-learn matplotlib
pip install hmmlearn shap        # optional; fallbacks are used if absent

python verify_fixes.py           # expect: 25 passed, 0 failed

AETHER_DATA=/path/to/data.csv AETHER_OUT=results python -u AETHER.py
AETHER_DATA=/path/to/data.csv AETHER_OUT=results python -u sensitivity.py
```

`data.csv` is the Movebank export (see Data below). Runtime on an Apple M-series laptop is roughly 15 minutes for the main run.

## Outputs (`results/`)

| File | Manuscript |
|---|---|
| `table1_results.csv` | Table 1 and Appendix D grid (experts, threshold rule, baselines, full ablation) |
| `table2_hmm_states.csv` | Table 5 (HMM state means, original units) |
| `table3_committee_vs_monolith.csv` | Table 3 (fused committee vs monolithic baselines, mixed benchmark) |
| `table4_attribution_confusion.csv` | Table 4 (source-attribution confusion matrix) |
| `figure_pr_curves.png` | Figure 7 |
| `shap_explanations.csv` | Figures 5–6 (exact Shapley values for top-ranked anomalies) |
| `case_study_flagged.csv`, `case_study_controls.csv` | Section 3.7 |
| `sensitivity_results.csv`, `figure_sensitivity.png` | Figure 4 (from `sensitivity.py`) |
| `run_summary.json` | Record counts, prevalence, attribution accuracy, configuration |

## Data

- **Source:** Movebank Data Repository study *(EBD) Common Kestrel (Falco tinnunculus) Spain, MERCURIO-SUMHAL* (Movebank Study ID 2970193504; Bustamante, 2025).
- **Archived version used in the manuscript:** Zenodo, https://doi.org/10.5281/zenodo.16990288
- **Scale:** 2,534,898 GPS fixes from 61 individuals (2020–2023). After dropping records with incomplete GPS-quality channels and segmenting on gaps > 900 s, 2,254,308 fixes from 40 individuals are usable.

The pipeline reads only the following Movebank columns:

| Movebank column | Internal name | Description | Unit |
|---|---|---|---|
| `tag-local-identifier` | `animal_id` | Individual identifier | – |
| `timestamp` | `timestamp` | Time of fix | UTC |
| `location-lat`, `location-long` | `latitude`, `longitude` | Position (WGS 84) | decimal degrees |
| `height-above-msl` | `altitude_m` | Altitude above mean sea level | m |
| `ground-speed` | `speed_ms` | Ground speed reported by the tag | m/s |
| `heading` | `heading` | Direction of travel from true north | degrees (0–360) |
| `gps:satellite-count` | `satellite_count` | Satellites used for the fix | integer |
| `gps:dop` | `dop` | Dilution of precision (lower is better) | unitless |
| `gps-time-to-fix` | `time_to_fix_s` | Time to acquire the fix | s |

Derived features, computed only within gap-free segments of one individual's track:

- `acceleration_ms2` = Δ`speed_ms` / Δt between consecutive fixes (m/s²)
- `turning_angle` = |wrapped Δ`heading`| between consecutive fixes, bounded on [0°, 180°]

Instantaneous sensor channels are never imputed; the first fix of each segment (undefined derived features) is dropped.

## Citation

**Software:** Ganapathy, S. (2026). AETHER: Anomaly Ensemble framework for Trajectory Heuristics and Explainable Reasoning [software]. https://github.com/SanjayGanapathy/AETHER

**Data:** Bustamante, J. (2025). Data from: (EBD) Common Kestrel (Falco tinnunculus) Spain, MERCURIO-SUMHAL. Movebank Data Repository, Study ID 2970193504. Archived at Zenodo: https://doi.org/10.5281/zenodo.16990288

The dataset was curated with funding from the Spanish Ministry of Science and Innovation (MICINN) through the European Regional Development Fund (SUMHAL, LIFEWATCH-2019-09-CSIC-4, POPE 2014-2020) and project MERCURIO (ref: PID2020-115793GB).
