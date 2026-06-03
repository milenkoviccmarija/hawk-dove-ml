import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/hawk-dove-ml-matplotlib")

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


RAW_DATA_PATH = Path("data/raw/hawk_dove_dataset.csv")
EDA_FIGURES_DIR = Path("results/figures/eda")
METRICS_DIR = Path("results/metrics")

NUMERIC_COLUMNS = [
    "V",
    "C",
    "cost_value_ratio",
    "value_cost_ratio",
    "conflict_severity",
    "theoretical_hawk",
    "initial_hawk",
    "iterations",
    "learning_rate",
    "mutation_rate",
    "environment_volatility",
    "population_size",
    "final_hawk",
    "final_dove",
]


def load_dataset(path=RAW_DATA_PATH):
    return pd.read_csv(path)


def save_dataset_report(dataset):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = METRICS_DIR / "eda_summary.txt"
    numeric_dataset = dataset[NUMERIC_COLUMNS]

    with open(report_path, "w") as file:
        file.write("EDA izvestaj za Hawk-Dove dataset\n")
        file.write("=================================\n\n")
        file.write(f"Broj redova: {dataset.shape[0]}\n")
        file.write(f"Broj kolona: {dataset.shape[1]}\n\n")

        file.write("Tipovi podataka\n")
        file.write("---------------\n")
        file.write(dataset.dtypes.to_string())
        file.write("\n\n")

        file.write("Nedostajuce vrednosti\n")
        file.write("---------------------\n")
        file.write(dataset.isnull().sum().to_string())
        file.write("\n\n")

        file.write("Broj duplikata\n")
        file.write("--------------\n")
        file.write(str(dataset.duplicated().sum()))
        file.write("\n\n")

        file.write("Deskriptivna statistika numerickih atributa\n")
        file.write("-------------------------------------------\n")
        file.write(numeric_dataset.describe().T.to_string())
        file.write("\n\n")

        file.write("Raspodela dominantne strategije\n")
        file.write("-------------------------------\n")
        file.write(dataset["dominant_strategy"].value_counts().to_string())
        file.write("\n\n")

        file.write("Korelacija atributa sa ciljnom promenljivom final_hawk\n")
        file.write("------------------------------------------------------\n")
        correlations = numeric_dataset.corr(numeric_only=True)["final_hawk"]
        correlations = correlations.sort_values(key=lambda values: values.abs(), ascending=False)
        file.write(correlations.to_string())
        file.write("\n")

    print(f"EDA tekstualni izvestaj je sacuvan u {report_path}.")


def save_target_distribution(dataset):
    plt.figure(figsize=(8, 5))
    plt.hist(dataset["final_hawk"], bins=30, edgecolor="black")
    plt.xlabel("Finalni udeo hawk strategije")
    plt.ylabel("Broj instanci")
    plt.title("Raspodela ciljne promenljive final_hawk")
    plt.tight_layout()
    plt.savefig(EDA_FIGURES_DIR / "target_distribution.png")
    plt.close()


def save_numeric_distributions(dataset):
    selected_columns = [
        "V",
        "C",
        "initial_hawk",
        "iterations",
        "learning_rate",
        "mutation_rate",
        "environment_volatility",
        "population_size",
        "final_hawk",
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.ravel()

    for index, column in enumerate(selected_columns):
        axes[index].hist(dataset[column], bins=30, edgecolor="black")
        axes[index].set_title(column)
        axes[index].set_xlabel(column)
        axes[index].set_ylabel("Broj instanci")

    plt.tight_layout()
    plt.savefig(EDA_FIGURES_DIR / "numeric_distributions.png")
    plt.close()


def save_boxplots(dataset):
    selected_columns = [
        "V",
        "C",
        "cost_value_ratio",
        "value_cost_ratio",
        "iterations",
        "learning_rate",
        "mutation_rate",
        "environment_volatility",
        "population_size",
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.ravel()

    for index, column in enumerate(selected_columns):
        axes[index].boxplot(dataset[column], vert=True)
        axes[index].set_title(column)
        axes[index].set_ylabel("Vrednost")

    plt.tight_layout()
    plt.savefig(EDA_FIGURES_DIR / "boxplots.png")
    plt.close()


def save_correlation_heatmap(dataset):
    correlation_matrix = dataset[NUMERIC_COLUMNS].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(correlation_matrix.columns)))
    ax.set_yticks(np.arange(len(correlation_matrix.index)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(correlation_matrix.index)
    ax.set_title("Korelaciona matrica numerickih atributa")

    plt.tight_layout()
    plt.savefig(EDA_FIGURES_DIR / "correlation_matrix.png")
    plt.close()


def save_relationship_plots(dataset):
    plot_specs = [
        ("value_cost_ratio", "final_hawk"),
        ("cost_value_ratio", "final_hawk"),
        ("conflict_severity", "final_hawk"),
        ("theoretical_hawk", "final_hawk"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    for index, (x_column, y_column) in enumerate(plot_specs):
        axes[index].scatter(dataset[x_column], dataset[y_column], alpha=0.35)
        axes[index].set_xlabel(x_column)
        axes[index].set_ylabel(y_column)
        axes[index].set_title(f"{x_column} vs {y_column}")

    plt.tight_layout()
    plt.savefig(EDA_FIGURES_DIR / "feature_target_relationships.png")
    plt.close()


def save_strategy_balance(dataset):
    counts = dataset["dominant_strategy"].value_counts()

    plt.figure(figsize=(6, 5))
    plt.bar(counts.index, counts.values)
    plt.xlabel("Dominantna strategija")
    plt.ylabel("Broj instanci")
    plt.title("Raspodela dominantne strategije")
    plt.tight_layout()
    plt.savefig(EDA_FIGURES_DIR / "strategy_balance.png")
    plt.close()


def main():
    EDA_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()

    print("Prvih 5 redova dataset-a:")
    print(dataset.head())
    print("\nDimenzije dataset-a:", dataset.shape)
    print("\nNedostajuce vrednosti:")
    print(dataset.isnull().sum())
    print("\nBroj duplikata:", dataset.duplicated().sum())

    save_dataset_report(dataset)
    save_target_distribution(dataset)
    save_numeric_distributions(dataset)
    save_boxplots(dataset)
    save_correlation_heatmap(dataset)
    save_relationship_plots(dataset)
    save_strategy_balance(dataset)

    print(f"EDA grafikoni su sacuvani u {EDA_FIGURES_DIR}.")


if __name__ == "__main__":
    main()
