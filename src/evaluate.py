import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/hawk-dove-ml-matplotlib")

import matplotlib
import numpy as np
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

PLOT_COLOR = "#1f77b4"
REFERENCE_COLOR = "#c43c39"
GRID_COLOR = "#d9dee7"


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


def save_actual_vs_predicted_plot(y_test, y_pred, metrics):
    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())

    plt.figure(figsize=(8, 7))
    plt.scatter(
        y_test,
        y_pred,
        s=28,
        alpha=0.45,
        color=PLOT_COLOR,
        edgecolors="none",
        label="Test instance",
    )
    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        color=REFERENCE_COLOR,
        linewidth=2,
        label="Idealna predikcija",
    )
    plt.xlabel("Stvarni final_hawk")
    plt.ylabel("Predikovani final_hawk")
    plt.title("Stvarne i predikovane vrednosti")
    plt.grid(True, color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    plt.legend()
    plt.text(
        0.04,
        0.96,
        f"R2 = {metrics['R2']:.3f}\nRMSE = {metrics['RMSE']:.3f}\nMAE = {metrics['MAE']:.3f}",
        transform=plt.gca().transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": GRID_COLOR},
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "actual_vs_predicted.png")
    plt.close()


def save_residual_plot(y_test, y_pred):
    residuals = y_test - y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(
        y_pred,
        residuals,
        s=28,
        alpha=0.45,
        color=PLOT_COLOR,
        edgecolors="none",
    )
    plt.axhline(y=0, color=REFERENCE_COLOR, linewidth=2)
    plt.xlabel("Predikovani final_hawk")
    plt.ylabel("Rezidual (stvarno - predikovano)")
    plt.title("Reziduali po predikovanoj vrednosti")
    plt.grid(True, color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "residual_plot.png")
    plt.close()


def save_residual_distribution_plot(y_test, y_pred):
    residuals = y_test - y_pred

    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=35, color=PLOT_COLOR, alpha=0.78, edgecolor="white")
    plt.axvline(0, color=REFERENCE_COLOR, linewidth=2)
    plt.axvline(residuals.mean(), color="#2f855a", linewidth=2, linestyle="--")
    plt.xlabel("Rezidual (stvarno - predikovano)")
    plt.ylabel("Broj instanci")
    plt.title("Raspodela gresaka modela")
    plt.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    plt.legend(["Nulta greska", "Prosecna greska"])
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "residual_distribution.png")
    plt.close()


def save_prediction_distribution_plot(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    bins = np.linspace(0, 1, 31)
    plt.hist(y_test, bins=bins, alpha=0.58, label="Stvarno", color="#2f855a")
    plt.hist(y_pred, bins=bins, alpha=0.58, label="Predikovano", color=PLOT_COLOR)
    plt.xlabel("final_hawk")
    plt.ylabel("Broj instanci")
    plt.title("Raspodela stvarnih i predikovanih vrednosti")
    plt.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "prediction_distribution.png")
    plt.close()


def get_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        return pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False)

    if hasattr(model, "coef_"):
        return pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.coef_,
        }).sort_values("Importance", ascending=False)

    return None


def save_feature_importance(feature_importance):
    if feature_importance is None:
        (METRICS_DIR / "feature_importance.csv").unlink(missing_ok=True)
        (FIGURES_DIR / "feature_importance.png").unlink(missing_ok=True)
        print("\nModel ne podržava feature importance ili koeficijente.")
        return

    print("\nUticaj atributa:")
    print(feature_importance)

    feature_importance.to_csv(
        METRICS_DIR / "feature_importance.csv",
        index=False,
    )

    sorted_importance = feature_importance.sort_values("Importance")

    plt.figure(figsize=(9, 6))
    plt.barh(
        sorted_importance["Feature"],
        sorted_importance["Importance"],
        color=PLOT_COLOR,
    )
    plt.xlabel("Vaznost atributa")
    plt.title("Uticaj atributa na predikciju")
    plt.grid(True, axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
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

    save_actual_vs_predicted_plot(y_test, y_pred, metrics)
    save_residual_plot(y_test, y_pred)
    save_residual_distribution_plot(y_test, y_pred)
    save_prediction_distribution_plot(y_test, y_pred)
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
