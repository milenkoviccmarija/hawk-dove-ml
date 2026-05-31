import pandas as pd
import numpy as np
from joblib import dump
from pathlib import Path

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
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
    y_class_train = pd.read_csv(PROCESSED_DATA_DIR / "y_class_train.csv").squeeze()

    X_val = pd.read_csv(PROCESSED_DATA_DIR / "X_val_scaled.csv")
    y_val = pd.read_csv(PROCESSED_DATA_DIR / "y_val.csv").squeeze()
    y_class_val = pd.read_csv(PROCESSED_DATA_DIR / "y_class_val.csv").squeeze()

    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test_scaled.csv")
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").squeeze()
    y_class_test = pd.read_csv(PROCESSED_DATA_DIR / "y_class_test.csv").squeeze()

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        y_class_train,
        y_class_val,
        y_class_test,
    )


def build_models(best_ridge_alpha, best_lasso_alpha, knn_neighbors):
    return {
        "Linear Regression": LinearRegression(),
        f"Ridge Regression (alpha={best_ridge_alpha})": Ridge(
            alpha=best_ridge_alpha,
        ),
        f"Lasso Regression (alpha={best_lasso_alpha})": Lasso(
            alpha=best_lasso_alpha,
        ),
        f"KNN Regression (k={knn_neighbors})": KNeighborsRegressor(
            n_neighbors=knn_neighbors,
        ),
    }


def calculate_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def evaluate_knn_neighbors(X_train, y_train):
    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    neighbor_candidates = [1, 3, 5, 7, 9, 11, 15, 21, 31]
    results = []

    for neighbors in neighbor_candidates:
        model = KNeighborsRegressor(n_neighbors=neighbors)
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=kf,
            scoring="r2",
        )
        results.append({
            "neighbors": neighbors,
            "CV_R2_mean": np.mean(cv_scores),
            "CV_R2_std": np.std(cv_scores),
        })

    results_df = pd.DataFrame(results)
    best_neighbors = int(
        results_df.sort_values("CV_R2_mean", ascending=False).iloc[0]["neighbors"]
    )

    return results_df, best_neighbors


def evaluate_regularization_strengths(X_train, y_train):
    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    alpha_candidates = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    results = []

    for model_name, model_class in [
        ("Ridge Regression", Ridge),
        ("Lasso Regression", Lasso),
    ]:
        for alpha in alpha_candidates:
            model = model_class(alpha=alpha)
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=kf,
                scoring="r2",
            )
            results.append({
                "model": model_name,
                "alpha": alpha,
                "CV_R2_mean": np.mean(cv_scores),
                "CV_R2_std": np.std(cv_scores),
            })

    results_df = pd.DataFrame(results)
    best_ridge_alpha = float(
        results_df[results_df["model"] == "Ridge Regression"]
        .sort_values("CV_R2_mean", ascending=False)
        .iloc[0]["alpha"]
    )
    best_lasso_alpha = float(
        results_df[results_df["model"] == "Lasso Regression"]
        .sort_values("CV_R2_mean", ascending=False)
        .iloc[0]["alpha"]
    )

    return results_df, best_ridge_alpha, best_lasso_alpha


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


def evaluate_classification_example(X_train, y_train, X_test, y_test):
    classifier = LogisticRegression(random_state=RANDOM_SEED)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    return {
        "model": "Logistic Regression",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
    }


def save_results(
    results_df,
    knn_results_df,
    regularization_results_df,
    best_ridge_alpha,
    best_lasso_alpha,
    best_knn_neighbors,
    best_model_name,
    test_metrics,
    classification_metrics,
):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(METRICS_DIR / "model_results.csv", index=False)
    knn_results_df.to_csv(METRICS_DIR / "knn_neighbor_results.csv", index=False)
    regularization_results_df.to_csv(
        METRICS_DIR / "regularization_results.csv",
        index=False,
    )

    with open(METRICS_DIR / "results_summary.txt", "w") as file:
        file.write("Rezultati evaluacije regresionih modela\n")
        file.write("======================================\n\n")

        file.write("Izbor broja suseda za KNN regresiju\n")
        file.write("-----------------------------------\n")
        file.write(f"Najbolji broj suseda prema CV R2: {best_knn_neighbors}\n\n")

        for result in knn_results_df.to_dict("records"):
            file.write(f"k = {int(result['neighbors'])}\n")
            file.write(f"Mean CV R2: {result['CV_R2_mean']}\n")
            file.write(f"Std CV R2: {result['CV_R2_std']}\n")
            file.write("\n")

        file.write("Izbor lambda/alpha parametra za Ridge i Lasso\n")
        file.write("---------------------------------------------\n")
        file.write(f"Najbolji alpha za Ridge: {best_ridge_alpha}\n")
        file.write(f"Najbolji alpha za Lasso: {best_lasso_alpha}\n\n")

        for result in regularization_results_df.to_dict("records"):
            file.write(f"Model: {result['model']}\n")
            file.write(f"alpha = {result['alpha']}\n")
            file.write(f"Mean CV R2: {result['CV_R2_mean']}\n")
            file.write(f"Std CV R2: {result['CV_R2_std']}\n")
            file.write("\n")

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
        file.write("\n")

        file.write("Primer klasifikacije\n")
        file.write("--------------------\n")
        file.write("Cilj: hawk_dominant, odnosno da li je final_hawk > 0.5\n")
        file.write(f"Model: {classification_metrics['model']}\n")
        file.write(f"Test accuracy: {classification_metrics['accuracy']}\n")
        file.write(f"Test precision: {classification_metrics['precision']}\n")
        file.write(f"Test recall: {classification_metrics['recall']}\n")
        file.write("Confusion matrix:\n")
        file.write(str(classification_metrics["confusion_matrix"]))
        file.write("\n\n")
        file.write("Classification report:\n")
        file.write(classification_metrics["classification_report"])
        file.write("\n")
        file.write("Napomena: klasifikacija daje samo klasu hawk/dove, dok regresija daje tacan udeo final_hawk.\n")


def main():
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        y_class_train,
        y_class_val,
        y_class_test,
    ) = load_processed_data()
    knn_results_df, best_knn_neighbors = evaluate_knn_neighbors(X_train, y_train)
    (
        regularization_results_df,
        best_ridge_alpha,
        best_lasso_alpha,
    ) = evaluate_regularization_strengths(X_train, y_train)
    models = build_models(best_ridge_alpha, best_lasso_alpha, best_knn_neighbors)

    results_df, best_model, best_model_name = evaluate_models(
        models,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    test_metrics = evaluate_best_model(best_model, X_test, y_test)
    classification_metrics = evaluate_classification_example(
        X_train,
        y_class_train,
        X_test,
        y_class_test,
    )
    save_results(
        results_df,
        knn_results_df,
        regularization_results_df,
        best_ridge_alpha,
        best_lasso_alpha,
        best_knn_neighbors,
        best_model_name,
        test_metrics,
        classification_metrics,
    )
    dump(best_model, BEST_MODEL_PATH)

    print(f"\nNajbolji alpha za Ridge regresiju: {best_ridge_alpha}")
    print(f"Najbolji alpha za Lasso regresiju: {best_lasso_alpha}")
    print(f"\nNajbolji broj suseda za KNN regresiju: {best_knn_neighbors}")
    print(f"\nNajbolji model: {best_model_name}")
    print("Test MAE:", test_metrics["MAE"])
    print("Test MSE:", test_metrics["MSE"])
    print("Test RMSE:", test_metrics["RMSE"])
    print("Test R2:", test_metrics["R2"])
    print("\nPrimer klasifikacije za hawk_dominant:")
    print("Model:", classification_metrics["model"])
    print("Test accuracy:", classification_metrics["accuracy"])
    print("Klasifikacija predvidja samo klasu, a regresija procenjuje tacan final_hawk.")
    print(f"\nRezultati su sačuvani u {METRICS_DIR}.")
    print(f"Najbolji model je sačuvan u {BEST_MODEL_PATH}.")


if __name__ == "__main__":
    main()
