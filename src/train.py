import os
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/hawk-dove-ml-matplotlib")

import matplotlib
import pandas as pd
from joblib import dump
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
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
BEST_MODEL_PATH = Path("models/best_pipeline.joblib")
RANDOM_SEED = 42


def load_processed_data():
    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").squeeze()
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def get_pipelines_and_param_grids():
    
    pipelines = {
        "Baseline Dummy Regressor": Pipeline([
            ("scaler", StandardScaler()),
            ("model", DummyRegressor(strategy="mean"))
        ]),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge())
        ]),
        "KNN Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor())
        ]),
        "Random Forest Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1))
        ])
    }

    param_grids = {
        "Baseline Dummy Regressor": {},
        "Ridge Regression": {
            "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        },
        "KNN Regression": {
            "model__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 31]
        },
        "Random Forest Regression": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [8, 12, 16],
            "model__min_samples_leaf": [1, 3, 5]
        }
    }
    
    return pipelines, param_grids


def calculate_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def select_important_features(X_train, y_train, best_rf_pipeline, top_n=4):
    rf_model = best_rf_pipeline.named_steps["model"]
    
    feature_importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": rf_model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    selected_features = feature_importance_df.head(top_n)["Feature"].tolist()
    return feature_importance_df, selected_features


def save_tuning_plots(cv_results_dict):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ridge_df = pd.DataFrame(cv_results_dict["Ridge Regression"])
    if not ridge_df.empty:
        axes[0].plot(ridge_df["param_model__alpha"], ridge_df["mean_test_score"], marker="o")
        axes[0].set_xscale("log")
        axes[0].set_title("Ridge: Alpha vs CV Score")
        axes[0].set_xlabel("alpha")
        axes[0].set_ylabel("Neg Mean Squared Error")

    knn_df = pd.DataFrame(cv_results_dict["KNN Regression"])
    if not knn_df.empty:
        axes[1].plot(knn_df["param_model__n_neighbors"], knn_df["mean_test_score"], marker="o")
        axes[1].set_title("KNN: Broj suseda vs CV Score")
        axes[1].set_xlabel("n_neighbors")
        axes[1].set_ylabel("Neg Mean Squared Error")

    rf_df = pd.DataFrame(cv_results_dict["Random Forest Regression"]).sort_values("mean_test_score")
    if not rf_df.empty:
        axes[2].plot(range(len(rf_df)), rf_df["mean_test_score"], marker="o")
        axes[2].set_title("Random Forest: Kombinacije vs CV Score")
        axes[2].set_xlabel("Kandidati (indeks)")
        axes[2].set_ylabel("Neg Mean Squared Error")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hyperparameter_cv_scores.png")
    plt.close()


def main():
    X_train, X_test, y_train, y_test = load_processed_data()
    pipelines, param_grids = get_pipelines_and_param_grids()
    
    cv_results_history = {}
    best_pipelines = {}
    validation_metrics = {}

    print("Započinjemo unakrsnu validaciju i pretragu hiperparametara...")
    
    for name, pipeline in pipelines.items():
        print(f"Tuniranje modela: {name}")
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[name],
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        best_pipelines[name] = grid_search.best_estimator_
        cv_results_history[name] = grid_search.cv_results_

        best_cv_rmse = -grid_search.best_score_
        print(f"Najbolji CV RMSE za {name}: {best_cv_rmse:.4f} sa parametrima {grid_search.best_params_}")
        
        validation_metrics[name] = {
            "Best_CV_RMSE": best_cv_rmse,
            "Best_Params": grid_search.best_params_
        }

    save_tuning_plots(cv_results_history)

    feature_importance_df, selected_features = select_important_features(
        X_train, y_train, best_pipelines["Random Forest Regression"]
    )
    
    print("\nEvaluacija modela sa smanjenim brojem atributa (Selected Features) kroz 5-Fold CV...")
    selected_features_metrics = {}
    best_selected_pipelines = {}
    
    for name, pipeline in pipelines.items():
        grid_search_sel = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[name],
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )
        grid_search_sel.fit(X_train[selected_features], y_train)
        best_selected_pipelines[name] = grid_search_sel.best_estimator_
        selected_features_metrics[name] = {
            "Best_CV_RMSE": -grid_search_sel.best_score_,
            "Best_Params": grid_search_sel.best_params_
        }

    best_model_name = min(validation_metrics, key=lambda k: validation_metrics[k]["Best_CV_RMSE"])
    final_best_pipeline = best_pipelines[best_model_name]
    
    y_test_pred = final_best_pipeline.predict(X_test)
    test_scores = calculate_metrics(y_test, y_test_pred)
    
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    feature_importance_df.to_csv(METRICS_DIR / "feature_importance.csv", index=False)
    
    with open(METRICS_DIR / "results_summary.txt", "w") as file:
        file.write("Rezultati evaluacije regresionih modela preko Cross-Validation i Pipeline-a\n")
        file.write("========================================================================\n\n")
        
        file.write("Performanse modela na 5-Fold Cross-Validaciji (Svi Atributi):\n")
        for name, metrics in validation_metrics.items():
            file.write(f"- {name}:\n")
            file.write(f"  Najbolji CV RMSE: {metrics['Best_CV_RMSE']:.5f}\n")
            file.write(f"  Izabrani parametri: {metrics['Best_Params']}\n")
            
        file.write("\nPerformanse modela na 5-Fold Cross-Validaciji (Izabrani Atributi):\n")
        for name, metrics in selected_features_metrics.items():
            file.write(f"- {name} (Selected Features):\n")
            file.write(f"  Najbolji CV RMSE: {metrics['Best_CV_RMSE']:.5f}\n")
            file.write(f"  Izabrani parametri: {metrics['Best_Params']}\n")

        file.write(f"\nIzabrani najvažniji atributi: {', '.join(selected_features)}\n")
        
        file.write("\n=========================================\n")
        file.write(f"FINALNI REZULTATI NA SKRIVENOM TEST SKUPU\n")
        file.write(f"Izabrani model: {best_model_name}\n")
        file.write("=========================================\n")
        file.write(f"Test MAE:  {test_scores['MAE']:.5f}\n")
        file.write(f"Test MSE:  {test_scores['MSE']:.5f}\n")
        file.write(f"Test RMSE: {test_scores['RMSE']:.5f}\n")
        file.write(f"Test R2:   {test_scores['R2']:.5f}\n")

    dump(final_best_pipeline, BEST_MODEL_PATH)
    
    print(f"\nUspješno završeno! Najbolji model na CV-u je: {best_model_name}")
    print(f"Finalni R2 na test skupu: {test_scores['R2']:.4f}")
    print(f"Kompletan pipeline (skaler + model) je sačuvan na lokaciji: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()