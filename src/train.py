import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/hawk-dove-ml-matplotlib")

import matplotlib
import pandas as pd
import numpy as np
from joblib import dump
from itertools import product
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROCESSED_DATA_DIR = Path("data/processed")
METRICS_DIR = Path("results/metrics")
FIGURES_DIR = Path("results/figures")
BEST_MODEL_PATH = Path("models/best_model.joblib")
RANDOM_SEED = 42


def load_processed_data():
    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train_scaled.csv")
    y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").squeeze()

    X_val = pd.read_csv(PROCESSED_DATA_DIR / "X_val_scaled.csv")
    y_val = pd.read_csv(PROCESSED_DATA_DIR / "y_val.csv").squeeze()

    return (
        X_train,
        X_val,
        y_train,
        y_val,
    )


MODEL_EXPLANATIONS = {
    "Baseline Dummy Regressor": (
        "Uzima se kao polazna tacka jer uvek predvidja srednju vrednost "
        "ciljne promenljive. Ako drugi modeli nisu bolji od njega, onda "
        "nisu naucili koristan obrazac iz atributa."
    ),
    "Ridge Regression": (
        "Uzima se kao glavni linearni model jer je jednostavan za tumacenje, "
        "a L2 regularizacija smanjuje preprilagodjavanje i stabilizuje "
        "koeficijente kada su atributi povezani."
    ),
    "KNN Regression": (
        "Uzima se jer koristi slicnost izmedju scenarija. Za ovaj problem je "
        "prirodno proveriti da li slicne kombinacije V, C, pocetnog udela i "
        "parametara simulacije daju slican final_hawk."
    ),
    "Random Forest Regression": (
        "Uzima se kao nelinearni model jer simulacija hawk-dove dinamike ima "
        "interakcije izmedju atributa, na primer V/C, learning_rate, "
        "mutation_rate i volatilnost. Ansambl stabala dobro hvata takve "
        "nelinearne odnose bez potrebe da ih rucno zadamo."
    ),
}


def build_models(best_ridge_alpha, knn_neighbors, random_forest_params):
    return {
        "Baseline Dummy Regressor": DummyRegressor(strategy="mean"),
        f"Ridge Regression (alpha={best_ridge_alpha})": Ridge(
            alpha=best_ridge_alpha,
        ),
        f"KNN Regression (k={knn_neighbors})": KNeighborsRegressor(
            n_neighbors=knn_neighbors,
        ),
        "Random Forest Regression": RandomForestRegressor(
            **random_forest_params,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }


def calculate_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def calculate_validation_metrics(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    return calculate_metrics(y_val, y_val_pred)


def find_elbow_index(x_values, y_values):
    points = np.column_stack([x_values, y_values]).astype(float)

    if len(points) <= 2:
        return int(np.argmin(y_values))

    value_range = points.max(axis=0) - points.min(axis=0)
    value_range[value_range == 0] = 1
    normalized_points = (points - points.min(axis=0)) / value_range

    first_point = normalized_points[0]
    last_point = normalized_points[-1]
    line = last_point - first_point
    line_length = np.linalg.norm(line)

    if line_length == 0:
        return int(np.argmin(y_values))

    vectors = first_point - normalized_points
    distances = np.abs(
        (line[0] * vectors[:, 1] - line[1] * vectors[:, 0]) / line_length
    )
    return int(np.argmax(distances))


def mark_elbow(results_df, sort_column):
    sorted_results = results_df.sort_values(sort_column).reset_index(drop=True)
    elbow_index = find_elbow_index(
        sorted_results[sort_column].to_numpy(),
        sorted_results["Val_RMSE"].to_numpy(),
    )
    sorted_results["is_elbow"] = False
    sorted_results.loc[elbow_index, "is_elbow"] = True
    return sorted_results


def mark_best_validation_rmse(results_df, sort_column):
    sorted_results = results_df.sort_values(sort_column).reset_index(drop=True)
    best_index = sorted_results["Val_RMSE"].idxmin()
    sorted_results["is_best"] = False
    sorted_results.loc[best_index, "is_best"] = True
    return sorted_results


def evaluate_knn_neighbors(X_train, y_train, X_val, y_val):
    neighbor_candidates = [1, 3, 5, 7, 9, 11, 15, 21, 31]
    results = []

    for neighbors in neighbor_candidates:
        model = KNeighborsRegressor(n_neighbors=neighbors)
        val_metrics = calculate_validation_metrics(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
        )
        results.append({
            "neighbors": neighbors,
            "Val_MAE": val_metrics["MAE"],
            "Val_MSE": val_metrics["MSE"],
            "Val_RMSE": val_metrics["RMSE"],
            "Val_R2": val_metrics["R2"],
        })

    results_df = mark_elbow(pd.DataFrame(results), "neighbors")
    best_neighbors = int(
        results_df.loc[results_df["is_elbow"]].iloc[0]["neighbors"]
    )

    return results_df, best_neighbors


def evaluate_ridge_regularization_strengths(X_train, y_train, X_val, y_val):
    alpha_candidates = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    results = []

    for alpha in alpha_candidates:
        model = Ridge(alpha=alpha)
        val_metrics = calculate_validation_metrics(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
        )
        results.append({
            "model": "Ridge Regression",
            "alpha": alpha,
            "log10_alpha": np.log10(alpha),
            "Val_MAE": val_metrics["MAE"],
            "Val_MSE": val_metrics["MSE"],
            "Val_RMSE": val_metrics["RMSE"],
            "Val_R2": val_metrics["R2"],
        })

    results_df = mark_best_validation_rmse(pd.DataFrame(results), "log10_alpha")
    best_ridge_alpha = float(
        results_df.loc[results_df["is_best"]].iloc[0]["alpha"]
    )

    return results_df, best_ridge_alpha


def evaluate_random_forest_hyperparameters(X_train, y_train, X_val, y_val):
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [8, 12, 16],
        "min_samples_leaf": [1, 3, 5],
    }
    results = []

    for n_estimators, max_depth, min_samples_leaf in product(
        param_grid["n_estimators"],
        param_grid["max_depth"],
        param_grid["min_samples_leaf"],
    ):
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
        }
        model = RandomForestRegressor(
            **params,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        val_metrics = calculate_validation_metrics(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
        )
        results.append({
            **params,
            "complexity": n_estimators * max_depth / min_samples_leaf,
            "Val_MAE": val_metrics["MAE"],
            "Val_MSE": val_metrics["MSE"],
            "Val_RMSE": val_metrics["RMSE"],
            "Val_R2": val_metrics["R2"],
        })

    results_df = pd.DataFrame(results).sort_values("complexity").reset_index(drop=True)
    results_df["candidate_rank"] = np.arange(1, len(results_df) + 1)
    results_df = mark_best_validation_rmse(results_df, "candidate_rank")
    best_result = results_df.loc[results_df["is_best"]].iloc[0]
    best_params = {
        "n_estimators": int(best_result["n_estimators"]),
        "max_depth": int(best_result["max_depth"]),
        "min_samples_leaf": int(best_result["min_samples_leaf"]),
    }

    return results_df, best_params


def save_hyperparameter_elbow_plot(
    knn_results_df,
    regularization_results_df,
    random_forest_results_df,
):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    plot_specs = [
        (
            axes[0],
            knn_results_df,
            "neighbors",
            "KNN: broj suseda",
            "k",
            "is_elbow",
            "koleno",
        ),
        (
            axes[1],
            regularization_results_df,
            "alpha",
            "Ridge: regularizacija",
            "alpha",
            "is_best",
            "najbolji RMSE",
        ),
        (
            axes[2],
            random_forest_results_df,
            "candidate_rank",
            "Random Forest: slozenost",
            "kandidat sortiran po slozenosti",
            "is_best",
            "najbolji RMSE",
        ),
    ]

    for ax, data, x_column, title, x_label, marker_column, marker_label in plot_specs:
        data = data.sort_values(x_column)
        selected = data.loc[data[marker_column]].iloc[0]

        ax.plot(data[x_column], data["Val_RMSE"], marker="o")
        ax.scatter(
            selected[x_column],
            selected["Val_RMSE"],
            color="red",
            zorder=3,
            label=marker_label,
        )
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Validation RMSE")
        ax.legend()

        if x_column == "alpha":
            ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hyperparameter_elbow.png")
    plt.close()


def select_important_features(X_train, y_train, random_forest_params, top_n=4):
    model = RandomForestRegressor(
        **random_forest_params,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    feature_importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    selected_features = feature_importance_df.head(top_n)["Feature"].tolist()

    return feature_importance_df, selected_features


def evaluate_models(models, X_train, y_train, X_val, y_val, verbose=True):
    results = []
    fitted_models = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        val_metrics = calculate_metrics(y_val, y_val_pred)

        results.append({
            "model": model_name,
            "Val_MAE": val_metrics["MAE"],
            "Val_MSE": val_metrics["MSE"],
            "Val_RMSE": val_metrics["RMSE"],
            "Val_R2": val_metrics["R2"],
        })

        fitted_models[model_name] = model

        if verbose:
            print(f"\n===== {model_name} =====")
            print("Validation MAE:", val_metrics["MAE"])
            print("Validation MSE:", val_metrics["MSE"])
            print("Validation RMSE:", val_metrics["RMSE"])
            print("Validation R2:", val_metrics["R2"])

    results_df = pd.DataFrame(results)
    best_model_name = results_df.sort_values("Val_R2", ascending=False).iloc[0]["model"]

    return results_df, fitted_models[best_model_name], best_model_name


def build_feature_selection_comparison(
    all_features_results_df,
    selected_features_results_df,
):
    all_features_results_df = all_features_results_df.copy()
    selected_features_results_df = selected_features_results_df.copy()

    all_features_results_df.insert(0, "feature_set", "all_features")
    selected_features_results_df.insert(0, "feature_set", "selected_features")

    return pd.concat(
        [all_features_results_df, selected_features_results_df],
        ignore_index=True,
    )


def save_results(
    results_df,
    selected_features_results_df,
    feature_selection_results_df,
    knn_results_df,
    regularization_results_df,
    random_forest_results_df,
    feature_importance_df,
    best_ridge_alpha,
    best_knn_neighbors,
    best_random_forest_params,
    selected_features,
    best_model_name,
    selected_features_best_model_name,
):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(METRICS_DIR / "model_results.csv", index=False)
    selected_features_results_df.to_csv(
        METRICS_DIR / "model_results_selected_features.csv",
        index=False,
    )
    feature_selection_results_df.to_csv(
        METRICS_DIR / "feature_selection_results.csv",
        index=False,
    )
    knn_results_df.to_csv(METRICS_DIR / "knn_neighbor_results.csv", index=False)
    regularization_results_df.to_csv(
        METRICS_DIR / "regularization_results.csv",
        index=False,
    )
    random_forest_results_df.to_csv(
        METRICS_DIR / "random_forest_hyperparameter_results.csv",
        index=False,
    )
    feature_importance_df.to_csv(METRICS_DIR / "feature_importance.csv", index=False)

    with open(METRICS_DIR / "results_summary.txt", "w") as file:
        file.write("Rezultati evaluacije regresionih modela\n")
        file.write("======================================\n\n")

        file.write("Izabrani modeli i razlog izbora\n")
        file.write("--------------------------------\n")
        for model_name, explanation in MODEL_EXPLANATIONS.items():
            file.write(f"{model_name}: {explanation}\n\n")

        file.write("Izbor broja suseda za KNN regresiju\n")
        file.write("-----------------------------------\n")
        file.write(
            "Pretraga je radjena tako sto se svaki kandidat fituje na train skupu, "
            "a ocenjuje na validation skupu. Koleno je tacka validation RMSE "
            "krive najudaljenija od prave izmedju prve i poslednje tacke.\n"
        )
        file.write(f"Izabrani broj suseda na kolenu: {best_knn_neighbors}\n\n")

        for result in knn_results_df.to_dict("records"):
            file.write(f"k = {int(result['neighbors'])}\n")
            file.write(f"Validation RMSE: {result['Val_RMSE']}\n")
            file.write(f"Validation R2: {result['Val_R2']}\n")
            file.write(f"Koleno: {bool(result['is_elbow'])}\n")
            file.write("\n")

        file.write("Izbor lambda/alpha parametra za Ridge regresiju\n")
        file.write("-----------------------------------------------\n")
        file.write(f"Izabrani alpha prema najmanjem validation RMSE: {best_ridge_alpha}\n\n")

        for result in regularization_results_df.to_dict("records"):
            file.write(f"Model: {result['model']}\n")
            file.write(f"alpha = {result['alpha']}\n")
            file.write(f"Validation RMSE: {result['Val_RMSE']}\n")
            file.write(f"Validation R2: {result['Val_R2']}\n")
            file.write(f"Najbolji validation RMSE: {bool(result['is_best'])}\n")
            file.write("\n")

        file.write("Izbor hiperparametara za Random Forest regresiju\n")
        file.write("-----------------------------------------------\n")
        file.write(
            "Parametri se biraju kao kombinacija sa najmanjim validation RMSE.\n"
        )
        file.write(f"Izabrani parametri: {best_random_forest_params}\n\n")

        for result in random_forest_results_df.to_dict("records"):
            file.write(
                "n_estimators = "
                f"{int(result['n_estimators'])}, "
                f"max_depth = {int(result['max_depth'])}, "
                f"min_samples_leaf = {int(result['min_samples_leaf'])}\n"
            )
            file.write(f"Slozenost: {result['complexity']}\n")
            file.write(f"Validation RMSE: {result['Val_RMSE']}\n")
            file.write(f"Validation R2: {result['Val_R2']}\n")
            file.write(f"Najbolji validation RMSE: {bool(result['is_best'])}\n")
            file.write("\n")

        file.write("Odabir najznacajnijih atributa\n")
        file.write("------------------------------\n")
        file.write(
            "Atributi su rangirani pomocu feature_importances_ vrednosti "
            "iz tuniranog Random Forest modela.\n"
        )
        file.write(f"Izabrani atributi: {', '.join(selected_features)}\n\n")

        for result in feature_importance_df.to_dict("records"):
            file.write(f"{result['Feature']}: {result['Importance']}\n")
        file.write("\n")

        file.write("Poredjenje modela sa svim i izabranim atributima\n")
        file.write("-----------------------------------------------\n")
        for result in feature_selection_results_df.to_dict("records"):
            file.write(f"Skup atributa: {result['feature_set']}\n")
            file.write(f"Model: {result['model']}\n")
            file.write(f"Validation MAE: {result['Val_MAE']}\n")
            file.write(f"Validation MSE: {result['Val_MSE']}\n")
            file.write(f"Validation RMSE: {result['Val_RMSE']}\n")
            file.write(f"Validation R2: {result['Val_R2']}\n")
            file.write("\n")

        for result in results_df.to_dict("records"):
            file.write(f"Model: {result['model']}\n")
            file.write(f"Validation MAE: {result['Val_MAE']}\n")
            file.write(f"Validation MSE: {result['Val_MSE']}\n")
            file.write(f"Validation RMSE: {result['Val_RMSE']}\n")
            file.write(f"Validation R2: {result['Val_R2']}\n")
            file.write("\n")

        best_model_result = results_df.loc[
            results_df["model"] == best_model_name
        ].iloc[0]
        selected_features_best_model_result = selected_features_results_df.loc[
            selected_features_results_df["model"] == selected_features_best_model_name
        ].iloc[0]

        file.write("Najbolji model prema validation skupu\n")
        file.write("------------------------------------\n")
        file.write(f"Model: {best_model_name}\n")
        file.write(f"Validation MAE: {best_model_result['Val_MAE']}\n")
        file.write(f"Validation MSE: {best_model_result['Val_MSE']}\n")
        file.write(f"Validation RMSE: {best_model_result['Val_RMSE']}\n")
        file.write(f"Validation R2: {best_model_result['Val_R2']}\n")
        file.write("\n")

        file.write("Najbolji model sa izabranim atributima prema validation skupu\n")
        file.write("-------------------------------------------------------------\n")
        file.write(f"Model: {selected_features_best_model_name}\n")
        file.write(f"Atributi: {', '.join(selected_features)}\n")
        file.write(f"Validation MAE: {selected_features_best_model_result['Val_MAE']}\n")
        file.write(f"Validation MSE: {selected_features_best_model_result['Val_MSE']}\n")
        file.write(f"Validation RMSE: {selected_features_best_model_result['Val_RMSE']}\n")
        file.write(f"Validation R2: {selected_features_best_model_result['Val_R2']}\n")
        file.write("\n")


def main():
    (
        X_train,
        X_val,
        y_train,
        y_val,
    ) = load_processed_data()
    knn_results_df, best_knn_neighbors = evaluate_knn_neighbors(
        X_train,
        y_train,
        X_val,
        y_val,
    )
    (
        regularization_results_df,
        best_ridge_alpha,
    ) = evaluate_ridge_regularization_strengths(
        X_train,
        y_train,
        X_val,
        y_val,
    )
    (
        random_forest_results_df,
        best_random_forest_params,
    ) = evaluate_random_forest_hyperparameters(
        X_train,
        y_train,
        X_val,
        y_val,
    )
    save_hyperparameter_elbow_plot(
        knn_results_df,
        regularization_results_df,
        random_forest_results_df,
    )
    feature_importance_df, selected_features = select_important_features(
        X_train,
        y_train,
        best_random_forest_params,
        top_n=4,
    )

    models = build_models(
        best_ridge_alpha,
        best_knn_neighbors,
        best_random_forest_params,
    )

    results_df, best_model, best_model_name = evaluate_models(
        models,
        X_train,
        y_train,
        X_val,
        y_val,
    )
    selected_features_models = build_models(
        best_ridge_alpha,
        best_knn_neighbors,
        best_random_forest_params,
    )
    (
        selected_features_results_df,
        _,
        selected_features_best_model_name,
    ) = evaluate_models(
        selected_features_models,
        X_train[selected_features],
        y_train,
        X_val[selected_features],
        y_val,
        verbose=False,
    )
    feature_selection_results_df = build_feature_selection_comparison(
        results_df,
        selected_features_results_df,
    )

    save_results(
        results_df,
        selected_features_results_df,
        feature_selection_results_df,
        knn_results_df,
        regularization_results_df,
        random_forest_results_df,
        feature_importance_df,
        best_ridge_alpha,
        best_knn_neighbors,
        best_random_forest_params,
        selected_features,
        best_model_name,
        selected_features_best_model_name,
    )
    dump(best_model, BEST_MODEL_PATH)

    print(f"\nNajbolji alpha za Ridge regresiju: {best_ridge_alpha}")
    print(f"\nNajbolji broj suseda za KNN regresiju: {best_knn_neighbors}")
    print(f"\nNajbolji parametri za Random Forest: {best_random_forest_params}")
    print(f"\nIzabrani najznacajniji atributi: {', '.join(selected_features)}")
    print(f"\nNajbolji model prema validation skupu: {best_model_name}")
    print("\nNajbolji model sa izabranim atributima prema validation skupu:", selected_features_best_model_name)
    print(f"\nRezultati su sačuvani u {METRICS_DIR}.")
    print(f"Najbolji model je sačuvan u {BEST_MODEL_PATH}.")


if __name__ == "__main__":
    main()
