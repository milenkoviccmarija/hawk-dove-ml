import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/hawk-dove-ml-matplotlib")

import matplotlib
import numpy as np
import pandas as pd
from joblib import load
from pathlib import Path
import scipy.stats as stats

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROCESSED_DATA_DIR = Path("data/processed")
FIGURES_DIR = Path("results/figures/evaluation")
METRICS_DIR = Path("results/metrics")
BEST_PIPELINE_PATH = Path("models/best_pipeline.joblib")

# Estetska podešavanja za grafikone
PLOT_COLOR = "#1f77b4"
REFERENCE_COLOR = "#c43c39"
GRID_COLOR = "#d9dee7"


def load_test_data():

    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").squeeze()

    return X_test, y_test


def calculate_metrics(y_true, y_pred):

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def save_actual_vs_predicted_plot(y_true, y_pred, metrics):

    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, color=PLOT_COLOR, alpha=0.4, edgecolors="none")
    
    # Idealna linija savršene predikcije (y = x)
    ideal_line = np.linspace(0, 1, 100)
    plt.plot(ideal_line, ideal_line, color=REFERENCE_COLOR, linestyle="--", linewidth=2, label="Savršena predikcija")
    
    plt.xlabel("Stvarni udeo Hawk strategije (Test skup)")
    plt.ylabel("Predikovani udeo Hawk strategije")
    plt.title(f"Stvarne vs. Predikovane Vrednosti\n(R² = {metrics['R2']:.4f})")
    plt.grid(True, linestyle=":", alpha=0.6, color=GRID_COLOR)
    plt.legend(loc="upper left")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()


def save_residual_plot(y_true, y_pred):

    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # 1. Grafikon Reziduali vs Predikovane vrednosti
    ax1.scatter(y_pred, residuals, color=PLOT_COLOR, alpha=0.4, edgecolors="none")
    ax1.axhline(0, color=REFERENCE_COLOR, linestyle="-", linewidth=1.5)
    ax1.set_xlabel("Predikovane vrednosti")
    ax1.set_ylabel("Reziduali (Stvarno - Predikovano)")
    ax1.set_title("Grafik Reziduala (Provera varijanse grešaka)")
    ax1.grid(True, linestyle=":", alpha=0.6, color=GRID_COLOR)
    
    # 2. Histogram raspodele grešaka
    ax2.hist(residuals, bins=30, color=PLOT_COLOR, edgecolor="#ffffff", alpha=0.8)
    ax2.axvline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Vrednost greške")
    ax2.set_ylabel("Učestalost (Frekvencija)")
    ax2.set_title("Raspodela grešaka predikcije (Reziduala)")
    ax2.grid(True, linestyle=":", alpha=0.6, color=GRID_COLOR)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "residual_analysis.png", dpi=150)
    plt.close()


def save_qq_plot(y_true, y_pred):

    residuals = y_true - y_pred
    
    plt.figure(figsize=(6, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    
    # Prilagođavanje izgleda matplotlib elemenata koje stats.probplot kreira generički
    plt.title("Q-Q Grafik Reziduala\n(Provera normalnosti grešaka)")
    plt.xlabel("Teorijski kvantili")
    plt.ylabel("Vrednosti reziduala iz modela")
    plt.grid(True, linestyle=":", alpha=0.6, color=GRID_COLOR)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "qq_plot_residuals.png", dpi=150)
    plt.close()


def save_evaluation_metrics(metrics):

    with open(METRICS_DIR / "evaluation_summary.txt", "w", encoding="utf-8") as file:
        file.write("==============================================\n")
        file.write("Evaluacija najboljeg modela (preko Pipeline-a)\n")
        file.write("==============================================\n\n")
        file.write(f"Putanja do modela: {BEST_PIPELINE_PATH}\n")
        file.write(f"Test MAE (Srednja apsolutna greška):  {metrics['MAE']:.5f}\n")
        file.write(f"Test MSE (Srednja kvadratna greška):   {metrics['MSE']:.5f}\n")
        file.write(f"Test RMSE (Koren srednje kv. greške): {metrics['RMSE']:.5f}\n")
        file.write(f"Test R2 score (Koeficijent determinacije): {metrics['R2']:.5f}\n")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Validacija postojanja istreniranog modela
    if not BEST_PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Nedostaje fajl {BEST_PIPELINE_PATH}. Pokreni prvo train.py da istreniraš i sačuvaš model!"
        )

    print("Učitavanje sačuvanog unifikovanog Pipeline-a...")
    pipeline = load(BEST_PIPELINE_PATH)
    
    print("Učitavanje skrivenih test podataka...")
    X_test, y_test = load_test_data()

    print("Pokretanje predikcija na test podacima...")
    # Pipeline automatski skalira X_test kroz unutrašnji StandardScaler i prosleđuje ga modelu
    y_pred = pipeline.predict(X_test)
    
    print("Izračunavanje finalnih metričkih pokazatelja...")
    metrics = calculate_metrics(y_test, y_pred)

    print("Generisanje i čuvanje grafičkih analiza vizuelnih performansi...")
    save_actual_vs_predicted_plot(y_test, y_pred, metrics)
    save_residual_plot(y_test, y_pred)
    save_qq_plot(y_test, y_pred)

    print("Zapisivanje rezultata u tekstualni izveštaj...")
    save_evaluation_metrics(metrics)
    
    print("\n" + "="*50)
    print(" KONAČNI REZULTATI NA TEST SKUPU:")
    print(f" R² (Koeficijent determinacije): {metrics['R2']:.4f}")
    print(f" RMSE (Koren srednje kv. greške): {metrics['RMSE']:.4f}")
    print(f" MAE (Srednja apsolutna greška):  {metrics['MAE']:.4f}")
    print("="*50)
    print("Evaluacija na test skupu uspešno završena. Izveštaj i grafikoni su sačuvani u 'results/'.")


if __name__ == "__main__":
    main()