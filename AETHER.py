# AETHER

# Import all required libraries
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import roc_curve, auc, average_precision_score
import matplotlib.pyplot as plt
import shap

print("All libraries imported successfully.")


def preprocess_movement_data(filepath):
    """
    Loads, cleans, and engineers features for movement data.
    This standardized function ensures a reproducible data pipeline.
    """
    print("\n Section 1: Data Loading & Feature Engineering ")
    try:
        dtype_spec = {15: str, 18: str, 22: str}
        df = pd.read_csv(filepath, dtype=dtype_spec, low_memory=False)
        print(f"Successfully loaded {filepath} with {len(df)} rows.")

        # Define a mapping for renaming columns for consistency
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
        existing_cols = {k: v for k, v in rename_map.items() if k in df.columns}
        gdf = df[list(existing_cols.keys())].rename(columns=existing_cols)

        # Convert types and handle missing data
        gdf["timestamp"] = pd.to_datetime(gdf["timestamp"])
        gdf = gdf.sort_values(by=["animal_id", "timestamp"]).drop_duplicates(
            subset=["animal_id", "timestamp"], keep="first"
        )
        numeric_cols = gdf.select_dtypes(include=np.number).columns
        gdf[numeric_cols] = gdf.groupby("animal_id")[numeric_cols].transform(
            lambda x: x.interpolate(method="linear").bfill().ffill()
        )
        gdf.dropna(inplace=True)

        #  Advanced Feature Engineering
        gdf["time_diff_s"] = (
            gdf.groupby("animal_id")["timestamp"]
            .diff()
            .dt.total_seconds()
            .clip(lower=1e-6)
        )
        gdf["acceleration_ms2"] = (
            gdf.groupby("animal_id")["speed_ms"].diff() / gdf["time_diff_s"]
        )
        gdf["turning_angle"] = gdf.groupby("animal_id")["heading"].diff().abs()
        gdf.fillna(0, inplace=True)
        print("Feature engineering complete.")
        return gdf

    except Exception as e:
        print(f"Data loading or feature engineering failed: {e}")
        return None


# Run the preprocessing
gdf = preprocess_movement_data("data.csv")


if gdf is not None:
    print("\n Section 2: Defining Expert Feature Sets ")

    features_kinematic = ["speed_ms", "acceleration_ms2", "turning_angle"]
    features_gps = ["satellite_count", "dop", "time_to_fix_s"]
    features_sequential = features_kinematic + features_gps + ["heading", "altitude_m"]

    features_kinematic = [f for f in features_kinematic if f in gdf.columns]
    features_gps = [f for f in features_gps if f in gdf.columns]
    features_sequential = [f for f in features_sequential if f in gdf.columns]

    models = {
        "kinematic": IsolationForest(contamination=0.001, random_state=55, n_jobs=-1),
        "gps": LocalOutlierFactor(
            n_neighbors=20, contamination=0.001, novelty=True, n_jobs=-1
        ),
        "sequential": GaussianHMM(
            n_components=4, covariance_type="diag", n_iter=100, random_state=55
        ),
    }


# Standardized Model Training
def train_experts(gdf, models, features_kinematic, features_gps, features_sequential):
    """
    Trains the expert committee models and calculates anomaly scores.
    """
    print("\n Section 3: Training the Expert Committee ")

    scalers = {
        "kinematic": StandardScaler(),
        "gps": StandardScaler(),
        "sequential": StandardScaler(),
    }
    X_data = {
        "kinematic": gdf[features_kinematic],
        "gps": gdf[features_gps],
        "sequential": gdf[features_sequential],
    }
    X_scaled = {key: scalers[key].fit_transform(X_data[key]) for key in X_data}

    for name, model in models.items():
        print(f"Training {name.capitalize()} Expert...")
        model.fit(X_scaled[name])

    print("\n Section 4: Generating Specialized Scores on Real Data ")
    for name, model in models.items():
        score_col = f"{name}_score"
        if name != "sequential":
            gdf[score_col] = (
                -model.score_samples(X_scaled[name])
                if isinstance(model, IsolationForest)
                else -model.decision_function(X_scaled[name])
            )
        else:
            gdf[score_col] = -model.score_samples(X_scaled[name])[0]

        cleaned_scores = (
            gdf[score_col]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(gdf[score_col].max())
        )
        min_val, max_val = cleaned_scores.min(), cleaned_scores.max()
        gdf[score_col] = (
            (cleaned_scores - min_val) / (max_val - min_val)
            if max_val > min_val
            else 0.5
        )

    print("All experts trained and scores calculated.")
    return gdf, models, scalers


if "models" in locals() and gdf is not None:
    gdf, models, scalers = train_experts(
        gdf, models, features_kinematic, features_gps, features_sequential
    )


# Quantitative Evaluation Framework
def inject_anomalies(df, anomaly_type, n_anomalies=100, random_seed=55):
    """Injects a specific type of anomaly for targeted evaluation."""
    np.random.seed(random_seed)
    df_injected = df.copy()
    df_injected["is_anomaly"] = 0
    valid_indices = df.index[df.groupby("animal_id").cumcount().between(2, len(df) - 3)]
    anomaly_indices = np.random.choice(valid_indices, size=n_anomalies, replace=False)

    for idx in anomaly_indices:
        if anomaly_type == "kinematic":
            df_injected.loc[idx, "speed_ms"] *= np.random.uniform(3.0, 5.0)
            df_injected.loc[idx, "is_anomaly"] = 1
        elif anomaly_type == "gps":
            df_injected.loc[idx, "satellite_count"] = np.random.randint(1, 4)
            df_injected.loc[idx, "is_anomaly"] = 1
        elif anomaly_type == "sequential":
            df_injected.loc[idx - 1, "speed_ms"] = 20
            df_injected.loc[idx, "speed_ms"] = 0
            df_injected.loc[idx + 1, "speed_ms"] = 20
            df_injected.loc[idx, "is_anomaly"] = 1

    return df_injected


def evaluate_model(
    model_name, model, scaler, features, gdf_benchmark, anomaly_type_to_inject
):
    """Runs a full benchmark evaluation for a single model and returns metrics."""
    # gdf_benchmark = inject_anomalies(gdf, anomaly_type=anomaly_type_to_inject, n_anomalies=100) # Inject outside this function
    X_benchmark = gdf_benchmark[features]
    y_true = gdf_benchmark["is_anomaly"]
    X_benchmark_scaled = scaler.transform(X_benchmark)

    if model_name != "Sequential Expert":
        y_scores_array = (
            -model.score_samples(X_benchmark_scaled)
            if isinstance(model, IsolationForest)
            else -model.decision_function(X_benchmark_scaled)
        )
        y_scores = pd.Series(y_scores_array, index=gdf_benchmark.index)
    else:
        y_scores = pd.Series(np.nan, index=gdf_benchmark.index)
        for animal_id, group in gdf_benchmark.groupby("animal_id"):
            if group["is_anomaly"].sum() > 0:  # Only score sequences with anomalies
                group_scaled = scaler.transform(group[features])
                log_likelihood = model.score(group_scaled)
                # Assign the sequence anomaly score to the marked anomaly point
                y_scores.loc[group[group["is_anomaly"] == 1].index] = -log_likelihood

    y_scores.fillna(
        y_scores.min() - 1, inplace=True
    )  # Fill non-anomaly sequences with a low score
    ranks = (
        pd.DataFrame({"score": y_scores, "is_anomaly": y_true})
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
        .query("is_anomaly == 1")
        .index
    )

    return {
        "Model": model_name,
        "PR-AUC": average_precision_score(y_true, y_scores),
        "AUC-ROC": auc(*roc_curve(y_true, y_scores)[:2]),
        "Median Rank": np.median(ranks) if len(ranks) > 0 else np.nan,
    }


if "models" in locals() and gdf is not None:
    print("\n Section 5: Rigorous Quantitative Benchmarking ")

    baseline_model = IsolationForest(contamination=0.001, random_state=55, n_jobs=-1)
    baseline_scaler = StandardScaler()
    baseline_features = features_sequential
    baseline_model.fit(baseline_scaler.fit_transform(gdf[baseline_features]))
    print("Baseline model trained.")

    tasks = [
        (
            "Kinematic Expert",
            models["kinematic"],
            scalers["kinematic"],
            features_kinematic,
            inject_anomalies(gdf.copy(), anomaly_type="kinematic", n_anomalies=100),
            "kinematic",
        ),
        (
            "GPS Expert",
            models["gps"],
            scalers["gps"],
            features_gps,
            inject_anomalies(gdf.copy(), anomaly_type="gps", n_anomalies=100),
            "gps",
        ),
        (
            "Sequential Expert",
            models["sequential"],
            scalers["sequential"],
            features_sequential,
            inject_anomalies(gdf.copy(), anomaly_type="sequential", n_anomalies=100),
            "sequential",
        ),
        (
            "Baseline (vs Kinematic)",
            baseline_model,
            baseline_scaler,
            baseline_features,
            inject_anomalies(gdf.copy(), anomaly_type="kinematic", n_anomalies=100),
            "kinematic",
        ),
        (
            "Baseline (vs GPS)",
            baseline_model,
            baseline_scaler,
            baseline_features,
            inject_anomalies(gdf.copy(), anomaly_type="gps", n_anomalies=100),
            "gps",
        ),
    ]

    results = [evaluate_model(*task) for task in tasks]
    results_table = pd.DataFrame(results).set_index("Model")
    print("\n Final Benchmark Performance Summary ")
    print(
        results_table.to_string(
            formatters={
                "PR-AUC": "{:.4f}".format,
                "AUC-ROC": "{:.4f}".format,
                "Median Rank": "{:,.0f}".format,
            }
        )
    )


# Hyperparameter Sensitivity Analysis
def run_sensitivity_analysis(
    model_name, param_name, param_values, gdf, features, anomaly_type
):
    """Performs sensitivity analysis for a given hyperparameter."""
    print(f"\n Sensitivity Analysis for {model_name}: {param_name} ")
    results = []

    for value in param_values:
        print(f"Testing {param_name} = {value}...")
        if model_name == "GPS Expert":
            model = LocalOutlierFactor(
                n_neighbors=value, contamination=0.001, novelty=True, n_jobs=-1
            )
        elif model_name == "Sequential Expert":
            model = GaussianHMM(
                n_components=value, covariance_type="diag", n_iter=100, random_state=55
            )

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(gdf[features])
        model.fit(X_scaled)
        # Inject anomalies for each sensitivity run to ensure fresh data
        gdf_benchmark = inject_anomalies(
            gdf.copy(), anomaly_type=anomaly_type, n_anomalies=100
        )
        metrics = evaluate_model(
            model_name, model, scaler, features, gdf_benchmark, anomaly_type
        )
        results.append({"param_value": value, "PR-AUC": metrics["PR-AUC"]})

    return pd.DataFrame(results)


if "models" in locals() and gdf is not None:
    print("\n\n Section 6: Hyperparameter Sensitivity Analysis ")

    lof_sensitivity = run_sensitivity_analysis(
        "GPS Expert", "n_neighbors", [10, 20, 30, 40, 50], gdf, features_gps, "gps"
    )
    hmm_sensitivity = run_sensitivity_analysis(
        "Sequential Expert",
        "n_components",
        [2, 3, 4, 5, 6, 8],
        gdf,
        features_sequential,
        "sequential",
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    lof_sensitivity.plot(
        x="param_value",
        y="PR-AUC",
        ax=axes[0],
        marker="o",
        title="GPS Expert (LOF) Sensitivity",
    )
    axes[0].set_xlabel("Number of Neighbors")
    axes[0].set_ylabel("PR-AUC")
    axes[0].grid(True)

    hmm_sensitivity.plot(
        x="param_value",
        y="PR-AUC",
        ax=axes[1],
        marker="o",
        title="Sequential Expert (HMM) Sensitivity",
    )
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("PR-AUC")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

# Qualitative Case Study Analysis
if "models" in locals() and gdf is not None:
    print("\n Section 7: Qualitative Case Study Analysis ")
    for expert_name in ["kinematic", "gps", "sequential"]:
        top_anomaly_idx = gdf[f"{expert_name}_score"].idxmax()
        print(
            f"Top anomaly for {expert_name.upper()} expert found at index {top_anomaly_idx}."
        )


# Adding Model Explainability using SHAP
def explain_anomalies_with_xai(
    gdf, models, scalers, features, num_anomalies_to_explain=3
):
    """
    Uses SHAP and model introspection to explain the top anomalies for each expert.

    Args:
        gdf (pd.DataFrame): The main dataframe with anomaly scores.
        models (dict): The trained expert models.
        scalers (dict): The trained scalers.
        features (dict): Dictionary of feature lists for each expert.
        num_anomalies_to_explain (int): The number of top anomalies to explain for each expert.
    """
    print(
        f"\n\n Section 8: Explaining Top {num_anomalies_to_explain} Anomalies with XAI "
    )

    # 1. Explain Kinematic and GPS Experts using SHAP
    for expert_name in ["kinematic", "gps"]:
        print(f"\n--- Explaining {expert_name.upper()} Expert ---")

        # Get the data and model for the current expert
        model = models[expert_name]
        scaler = scalers[expert_name]
        feature_names = features[expert_name]
        X_data_scaled = scaler.transform(gdf[feature_names])

        # Find the indices of the top N anomalies
        top_anomaly_indices = (
            gdf[f"{expert_name}_score"].nlargest(num_anomalies_to_explain).index
        )
        X_anomalies_scaled = scaler.transform(
            gdf.loc[top_anomaly_indices, feature_names]
        )

        # We need a prediction function for SHAP. It's slightly different for each model.
        if isinstance(model, IsolationForest):
            # For Isolation Forest, SHAP works with score_samples
            predict_fn = lambda x: -model.score_samples(x)
        else:  # LocalOutlierFactor
            # For LOF, SHAP works with decision_function
            predict_fn = lambda x: -model.decision_function(x)

        # Initialize the SHAP explainer
        # KernelExplainer is model-agnostic and works well here. We use a sample of the data for the background distribution.
        explainer = shap.KernelExplainer(predict_fn, shap.sample(X_data_scaled, 50))

        # Calculate SHAP values for the anomalous instances
        shap_values = explainer.shap_values(X_anomalies_scaled)

        # Display a SHAP force plot for each top anomaly
        for i in range(num_anomalies_to_explain):
            print(
                f"Explanation for {expert_name} anomaly #{i+1} (Index: {top_anomaly_indices[i]}):"
            )
            shap.force_plot(
                explainer.expected_value,
                shap_values[i],
                gdf.loc[top_anomaly_indices[i], feature_names],
                matplotlib=True,
                show=False,  # We will show it manually
            )
            plt.title(f"SHAP Explanation for {expert_name.upper()} Anomaly #{i+1}")
            plt.show()

    # 2. Explain Sequential Expert by Interpreting Hidden States
    print("\n--- Explaining SEQUENTIAL Expert (HMM) ---")
    hmm_model = models["sequential"]
    feature_names = features["sequential"]

    # The "explanation" for an HMM is understanding what its hidden states represent.
    # We can do this by looking at the mean feature values for each state.
    state_means = pd.DataFrame(hmm_model.means_, columns=feature_names)
    state_means["state_label"] = [f"State {i}" for i in range(hmm_model.n_components)]
    state_means = state_means.set_index("state_label")

    print("The Sequential (HMM) model learned the following behavioral states:")
    print(
        "These are the average (scaled) feature values that characterize each hidden state."
    )
    print(state_means.to_string(float_format="{:.2f}".format))
    print(
        "\nAnomalies in the HMM are sequences with a low probability, often involving unlikely transitions between these states (e.g., State 1 to State 3)."
    )


# This should be the last block of code to run.
if __name__ == "__main__":
    print("\n AETHER Framework Execution Complete ")

    # Check if models were trained before trying to explain
    if "models" in locals() and gdf is not None:
        # We create a dictionary of all feature sets to pass to the function
        all_features = {
            "kinematic": features_kinematic,
            "gps": features_gps,
            "sequential": features_sequential,
        }
        explain_anomalies_with_xai(gdf, models, scalers, all_features)
