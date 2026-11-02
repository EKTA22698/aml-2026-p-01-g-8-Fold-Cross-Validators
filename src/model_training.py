
"""Train and evaluate Ridge and Lasso models for hourly bike demand."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from feature_engineering import (
    prepare_features,
    build_preprocessor,
)


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "hour.csv"
MODEL_DIR = PROJECT_ROOT / "models"

TEST_SIZE = 0.20
N_SPLITS = 5


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

def load_data():

    df = pd.read_csv(DATA_PATH)

    if "dteday" not in df.columns or "hr" not in df.columns:
        raise ValueError(
            "Dataset must contain dteday and hr columns."
        )

    # Create timestamp
    df["timestamp"] = (
        pd.to_datetime(df["dteday"])
        + pd.to_timedelta(df["hr"], unit="h")
    )

    # Sort chronologically
    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)

    print(
        "Date range:",
        df["timestamp"].min(),
        "to",
        df["timestamp"].max()
    )

    return df


# --------------------------------------------------
# 3. CREATE MODEL PIPELINE
# --------------------------------------------------

def make_pipeline(model, X_train):

    preprocessor = build_preprocessor(
        X_train
    )

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ])

    return pipeline


# --------------------------------------------------
# 4. CALCULATE METRICS
# --------------------------------------------------

def calculate_metrics(y_true, predictions):

    mae = mean_absolute_error(
        y_true,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions
        )
    )

    r2 = r2_score(
        y_true,
        predictions
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# --------------------------------------------------
# 5. TIME-SERIES CROSS-VALIDATION
# --------------------------------------------------

def run_time_series_cv(
    model_name,
    model,
    X_train,
    y_train
):

    tscv = TimeSeriesSplit(
        n_splits=N_SPLITS
    )

    fold_results = []

    print("\n" + "=" * 50)
    print(model_name, "- TIME SERIES CROSS VALIDATION")
    print("=" * 50)

    for fold, (train_idx, val_idx) in enumerate(
        tscv.split(X_train),
        start=1
    ):

        # Split fold data chronologically
        X_fold_train = X_train.iloc[
            train_idx
        ]

        X_fold_val = X_train.iloc[
            val_idx
        ]

        y_fold_train = y_train.iloc[
            train_idx
        ]

        y_fold_val = y_train.iloc[
            val_idx
        ]

        # Build fresh pipeline for each fold
        pipeline = make_pipeline(
            model,
            X_fold_train
        )

        # Train
        pipeline.fit(
            X_fold_train,
            y_fold_train
        )

        # Predict
        predictions = pipeline.predict(
            X_fold_val
        )

        # Evaluate
        metrics = calculate_metrics(
            y_fold_val,
            predictions
        )

        metrics["Model"] = model_name
        metrics["Fold"] = fold

        fold_results.append(
            metrics
        )

        print(f"\nFold {fold}")

        print(
            f"MAE  : {metrics['MAE']:.2f}"
        )

        print(
            f"RMSE : {metrics['RMSE']:.2f}"
        )

        print(
            f"R2   : {metrics['R2']:.4f}"
        )

    return fold_results


# --------------------------------------------------
# 6. FINAL TEST EVALUATION
# --------------------------------------------------

def evaluate_on_test(
    model_name,
    model,
    X_train,
    y_train,
    X_test,
    y_test
):

    print("\n" + "=" * 50)
    print(model_name, "- FINAL TEST EVALUATION")
    print("=" * 50)

    # Build model
    pipeline = make_pipeline(
        model,
        X_train
    )

    # Train on training data
    pipeline.fit(
        X_train,
        y_train
    )

    # Predict unseen test data
    predictions = pipeline.predict(
        X_test
    )

    # Evaluate
    metrics = calculate_metrics(
        y_test,
        predictions
    )

    metrics["Model"] = model_name

    print(
        f"\nMAE  : {metrics['MAE']:.2f} bikes"
    )

    print(
        f"RMSE : {metrics['RMSE']:.2f} bikes"
    )

    print(
        f"R2   : {metrics['R2']:.4f}"
    )

    return pipeline, metrics, predictions


# --------------------------------------------------
# 7. VISUALIZATION
# --------------------------------------------------

def save_plots(
    timestamps,
    y_test,
    predictions,
    model_name
):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Plot first 7 days of test data
    plot_count = min(
        7 * 24,
        len(y_test)
    )

    plot_dates = timestamps.iloc[
        :plot_count
    ]

    actual = y_test.iloc[
        :plot_count
    ]

    predicted = predictions[
        :plot_count
    ]

    # Actual vs Predicted
    plt.figure(figsize=(13, 5))

    plt.plot(
        plot_dates,
        actual,
        label="Actual Demand"
    )

    plt.plot(
        plot_dates,
        predicted,
        label="Predicted Demand"
    )

    plt.title(
        f"Actual vs Predicted Bike Demand - {model_name}"
    )

    plt.xlabel("Time")
    plt.ylabel("Bike Rentals")

    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        MODEL_DIR / "actual_vs_predicted.png",
        dpi=150
    )

    plt.close()

    # Residual analysis
    residuals = (
        y_test.to_numpy() - predictions
    )

    plt.figure(figsize=(8, 5))

    plt.scatter(
        predictions,
        residuals,
        alpha=0.35
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.title(
        f"Residual Analysis - {model_name}"
    )

    plt.xlabel("Predicted Demand")
    plt.ylabel("Residuals")

    plt.tight_layout()

    plt.savefig(
        MODEL_DIR / "residuals.png",
        dpi=150
    )

    plt.close()


# --------------------------------------------------
# 8. MAIN FUNCTION
# --------------------------------------------------

def main():

    # Load dataset
    df = load_data()

    # Keep timestamp for ordering and visualization
    timestamps = df["timestamp"].copy()

    # Timestamp is not a predictor
    modeling_df = df.drop(
        columns=["timestamp"]
    )

    # Feature engineering
    X, y = prepare_features(
        modeling_df
    )

    print("\nFeatures shape:", X.shape)
    print("Target shape:", y.shape)

    # Chronological train-test split
    split_index = int(
        len(X) * (1 - TEST_SIZE)
    )

    X_train = X.iloc[
        :split_index
    ].copy()

    X_test = X.iloc[
        split_index:
    ].copy()

    y_train = y.iloc[
        :split_index
    ].copy()

    y_test = y.iloc[
        split_index:
    ].copy()

    test_timestamps = timestamps.iloc[
        split_index:
    ].reset_index(drop=True)

    print(
        "\nTraining samples:",
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
    )

    # Define models
    models = {

        "Ridge": Ridge(
            alpha=1.0
        ),

        "Lasso": Lasso(
            alpha=0.001,
            max_iter=10000
        )

    }

    cv_results = []
    test_results = []

    trained_models = {}
    test_predictions = {}

    # Train and evaluate models
    for model_name, model in models.items():

        # Time-series cross-validation
        cv_result = run_time_series_cv(
            model_name,
            model,
            X_train,
            y_train
        )

        cv_results.extend(
            cv_result
        )

        # Final test evaluation
        pipeline, test_metrics, predictions = (
            evaluate_on_test(
                model_name,
                model,
                X_train,
                y_train,
                X_test,
                y_test
            )
        )

        trained_models[model_name] = pipeline

        test_predictions[model_name] = predictions

        test_results.append(
            test_metrics
        )

    # --------------------------------------------------
    # 9. MODEL COMPARISON
    # --------------------------------------------------

    cv_df = pd.DataFrame(
        cv_results
    )

    test_df = pd.DataFrame(
        test_results
    )

    cv_summary = (
        cv_df.groupby("Model")[[
            "MAE",
            "RMSE",
            "R2"
        ]]
        .mean()
        .reset_index()
        .sort_values("RMSE")
    )

    print("\nMEAN CROSS-VALIDATION RESULTS")

    print(
        cv_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    print("\nFINAL TEST RESULTS")

    print(
        test_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    # Select best model using CV RMSE
    best_model_name = cv_summary.iloc[0]["Model"]

    best_model = trained_models[
        best_model_name
    ]

    best_predictions = test_predictions[
        best_model_name
    ]

    print(
        "\nBest model:",
        best_model_name
    )

    # --------------------------------------------------
    # 10. SAVE MODELS AND RESULTS
    # --------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save Ridge and Lasso
    for model_name, pipeline in trained_models.items():

        filename = (
            model_name.lower()
            + "_regression.joblib"
        )

        joblib.dump(
            pipeline,
            MODEL_DIR / filename
        )

    # Save best model
    joblib.dump(
        best_model,
        MODEL_DIR / "best_model.joblib"
    )

    # Save evaluation reports
    cv_df.to_csv(
        MODEL_DIR / "cv_fold_results.csv",
        index=False
    )

    cv_summary.to_csv(
        MODEL_DIR / "cv_summary.csv",
        index=False
    )

    test_df.to_csv(
        MODEL_DIR / "test_results.csv",
        index=False
    )

    # Save graphs
    save_plots(
        test_timestamps,
        y_test.reset_index(drop=True),
        best_predictions,
        best_model_name
    )

    print("\nModels and reports saved successfully!")

    print(
        "Output folder:",
        MODEL_DIR
    )


# --------------------------------------------------
# 11. EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    main()
