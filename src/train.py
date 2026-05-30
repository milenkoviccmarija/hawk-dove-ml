import pandas as pd
import numpy as np
from joblib import dump
from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)


PROCESSED_DATA_DIR = Path("data/processed")
METRICS_DIR = Path("results/metrics")
BEST_MODEL_PATH = Path("models/best_model.joblib")
RANDOM_SEED = 42


def load_processed_data():
    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train_scaled.csv")
    y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").squeeze()

    X_val = pd.read_csv(PROCESSED_DATA_DIR / "X_val_scaled.csv")
    y_val = pd.read_csv(PROCESSED_DATA_DIR / "y_val.csv").squeeze()

    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test_scaled.csv")
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").squeeze()

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_models():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.001),
        "KNN Regression": KNeighborsRegressor(n_neighbors=5),
        "Random Forest Regression": RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_SEED,
        ),
        "Extra Trees Regression": ExtraTreesRegressor(
            n_estimators=100,
            random_state=RANDOM_SEED,
        ),
        "Gradient Boosting Regression": GradientBoostingRegressor(
            random_state=RANDOM_SEED,
        ),
        "Hist Gradient Boosting Regression": HistGradientBoostingRegressor(
            random_state=RANDOM_SEED,
        ),
    }


def calculate_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def evaluate_models(models, X_train, y_train, X_val, y_val):
    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    results = []
    fitted_models = {}

    for model_name, model in models.items():
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=kf,
            scoring="r2",
        )

        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        val_metrics = calculate_metrics(y_val, y_val_pred)

        results.append({
            "model": model_name,
            "CV_R2_mean": np.mean(cv_scores),
            "CV_R2_std": np.std(cv_scores),
            "Val_MAE": val_metrics["MAE"],
            "Val_MSE": val_metrics["MSE"],
            "Val_RMSE": val_metrics["RMSE"],
            "Val_R2": val_metrics["R2"],
        })

        fitted_models[model_name] = model

        print(f"\n===== {model_name} =====")
        print("Cross-validation R2 scores:", cv_scores)
        print("Mean CV R2:", np.mean(cv_scores))
        print("Std CV R2:", np.std(cv_scores))
        print("Validation MAE:", val_metrics["MAE"])
        print("Validation MSE:", val_metrics["MSE"])
        print("Validation RMSE:", val_metrics["RMSE"])
        print("Validation R2:", val_metrics["R2"])

    results_df = pd.DataFrame(results)
    best_model_name = results_df.sort_values("Val_R2", ascending=False).iloc[0]["model"]

    return results_df, fitted_models[best_model_name], best_model_name


def evaluate_best_model(best_model, X_test, y_test):
    y_test_pred = best_model.predict(X_test)
    return calculate_metrics(y_test, y_test_pred)


def save_results(results_df, best_model_name, test_metrics):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(METRICS_DIR / "model_results.csv", index=False)

    with open(METRICS_DIR / "results_summary.txt", "w") as file:
        file.write("Rezultati evaluacije regresionih modela\n")
        file.write("======================================\n\n")

        for result in results_df.to_dict("records"):
            file.write(f"Model: {result['model']}\n")
            file.write(f"Mean CV R2: {result['CV_R2_mean']}\n")
            file.write(f"Std CV R2: {result['CV_R2_std']}\n")
            file.write(f"Validation MAE: {result['Val_MAE']}\n")
            file.write(f"Validation MSE: {result['Val_MSE']}\n")
            file.write(f"Validation RMSE: {result['Val_RMSE']}\n")
            file.write(f"Validation R2: {result['Val_R2']}\n")
            file.write("\n")

        file.write("Najbolji model\n")
        file.write("--------------\n")
        file.write(f"Model: {best_model_name}\n")
        file.write(f"Test MAE: {test_metrics['MAE']}\n")
        file.write(f"Test MSE: {test_metrics['MSE']}\n")
        file.write(f"Test RMSE: {test_metrics['RMSE']}\n")
        file.write(f"Test R2: {test_metrics['R2']}\n")


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()
    models = build_models()

    results_df, best_model, best_model_name = evaluate_models(
        models,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    test_metrics = evaluate_best_model(best_model, X_test, y_test)
    save_results(results_df, best_model_name, test_metrics)
    dump(best_model, BEST_MODEL_PATH)

    print(f"\nNajbolji model: {best_model_name}")
    print("Test MAE:", test_metrics["MAE"])
    print("Test MSE:", test_metrics["MSE"])
    print("Test RMSE:", test_metrics["RMSE"])
    print("Test R2:", test_metrics["R2"])
    print(f"\nRezultati su sačuvani u {METRICS_DIR}.")
    print(f"Najbolji model je sačuvan u {BEST_MODEL_PATH}.")


if __name__ == "__main__":
    main()
