import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/hawk-dove-ml-matplotlib")

import matplotlib
import pandas as pd
from joblib import load
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROCESSED_DATA_DIR = Path("data/processed")
FIGURES_DIR = Path("results/figures")
METRICS_DIR = Path("results/metrics")
BEST_MODEL_PATH = Path("models/best_model.joblib")


def load_test_data():
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test_scaled.csv")
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").squeeze()

    return X_test, y_test


def calculate_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def save_actual_vs_predicted_plot(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred)
    plt.xlabel("Stvarne vrednosti")
    plt.ylabel("Predikovane vrednosti")
    plt.title("Stvarne vs Predikovane vrednosti")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "actual_vs_predicted.png")
    plt.close()


def save_residual_plot(y_test, y_pred):
    residuals = y_test - y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0)
    plt.xlabel("Predikovane vrednosti")
    plt.ylabel("Reziduali")
    plt.title("Residual plot")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "residual_plot.png")
    plt.close()


def get_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        return pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        })

    if hasattr(model, "coef_"):
        return pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.coef_,
        })

    return None


def save_feature_importance(feature_importance):
    if feature_importance is None:
        print("\nModel ne podržava feature importance ili koeficijente.")
        return

    print("\nUticaj atributa:")
    print(feature_importance)

    feature_importance.to_csv(
        METRICS_DIR / "feature_importance.csv",
        index=False,
    )

    plt.figure(figsize=(8, 6))
    plt.bar(
        feature_importance["Feature"],
        feature_importance["Importance"],
    )
    plt.xticks(rotation=45)
    plt.ylabel("Vrednost")
    plt.title("Uticaj atributa na predikciju")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png")
    plt.close()


def save_evaluation_metrics(metrics):
    with open(METRICS_DIR / "evaluation_summary.txt", "w") as file:
        file.write("Evaluacija najboljeg modela\n")
        file.write("===========================\n\n")
        file.write(f"Model path: {BEST_MODEL_PATH}\n")
        file.write(f"Test MAE: {metrics['MAE']}\n")
        file.write(f"Test MSE: {metrics['MSE']}\n")
        file.write(f"Test RMSE: {metrics['RMSE']}\n")
        file.write(f"Test R2: {metrics['R2']}\n")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    model = load(BEST_MODEL_PATH)
    X_test, y_test = load_test_data()

    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)

    save_actual_vs_predicted_plot(y_test, y_pred)
    save_residual_plot(y_test, y_pred)
    save_feature_importance(get_feature_importance(model, X_test.columns))
    save_evaluation_metrics(metrics)

    print("\nEvaluacija najboljeg modela:")
    print("Test MAE:", metrics["MAE"])
    print("Test MSE:", metrics["MSE"])
    print("Test RMSE:", metrics["RMSE"])
    print("Test R2:", metrics["R2"])
    print("\nGrafikoni i metrika su uspešno sačuvani.")


if __name__ == "__main__":
    main()
