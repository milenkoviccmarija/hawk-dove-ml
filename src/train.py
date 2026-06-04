import pandas as pd
import numpy as np
from joblib import dump
from itertools import product
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor

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


def evaluate_ridge_regularization_strengths(X_train, y_train):
    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    alpha_candidates = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    results = []

    for alpha in alpha_candidates:
        model = Ridge(alpha=alpha)
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=kf,
            scoring="r2",
        )
        results.append({
            "model": "Ridge Regression",
            "alpha": alpha,
            "CV_R2_mean": np.mean(cv_scores),
            "CV_R2_std": np.std(cv_scores),
        })

    results_df = pd.DataFrame(results)
    best_ridge_alpha = float(
        results_df.sort_values("CV_R2_mean", ascending=False).iloc[0]["alpha"]
    )

    return results_df, best_ridge_alpha


def evaluate_random_forest_hyperparameters(X_train, y_train):
    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
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
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=kf,
            scoring="r2",
        )
        results.append({
            **params,
            "CV_R2_mean": np.mean(cv_scores),
            "CV_R2_std": np.std(cv_scores),
        })

    results_df = pd.DataFrame(results)
    best_result = results_df.sort_values("CV_R2_mean", ascending=False).iloc[0]
    best_params = {
        "n_estimators": int(best_result["n_estimators"]),
        "max_depth": int(best_result["max_depth"]),
        "min_samples_leaf": int(best_result["min_samples_leaf"]),
    }

    return results_df, best_params


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
    test_metrics,
    selected_features_test_metrics,
    classification_metrics,
):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
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
        file.write(f"Najbolji broj suseda prema CV R2: {best_knn_neighbors}\n\n")

        for result in knn_results_df.to_dict("records"):
            file.write(f"k = {int(result['neighbors'])}\n")
            file.write(f"Mean CV R2: {result['CV_R2_mean']}\n")
            file.write(f"Std CV R2: {result['CV_R2_std']}\n")
            file.write("\n")

        file.write("Izbor lambda/alpha parametra za Ridge regresiju\n")
        file.write("-----------------------------------------------\n")
        file.write(f"Najbolji alpha za Ridge: {best_ridge_alpha}\n\n")

        for result in regularization_results_df.to_dict("records"):
            file.write(f"Model: {result['model']}\n")
            file.write(f"alpha = {result['alpha']}\n")
            file.write(f"Mean CV R2: {result['CV_R2_mean']}\n")
            file.write(f"Std CV R2: {result['CV_R2_std']}\n")
            file.write("\n")

        file.write("Izbor hiperparametara za Random Forest regresiju\n")
        file.write("-----------------------------------------------\n")
        file.write(f"Najbolji parametri: {best_random_forest_params}\n\n")

        for result in random_forest_results_df.to_dict("records"):
            file.write(
                "n_estimators = "
                f"{int(result['n_estimators'])}, "
                f"max_depth = {int(result['max_depth'])}, "
                f"min_samples_leaf = {int(result['min_samples_leaf'])}\n"
            )
            file.write(f"Mean CV R2: {result['CV_R2_mean']}\n")
            file.write(f"Std CV R2: {result['CV_R2_std']}\n")
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
            file.write(f"Mean CV R2: {result['CV_R2_mean']}\n")
            file.write(f"Std CV R2: {result['CV_R2_std']}\n")
            file.write(f"Validation MAE: {result['Val_MAE']}\n")
            file.write(f"Validation MSE: {result['Val_MSE']}\n")
            file.write(f"Validation RMSE: {result['Val_RMSE']}\n")
            file.write(f"Validation R2: {result['Val_R2']}\n")
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

        file.write("Najbolji model sa izabranim atributima\n")
        file.write("--------------------------------------\n")
        file.write(f"Model: {selected_features_best_model_name}\n")
        file.write(f"Atributi: {', '.join(selected_features)}\n")
        file.write(f"Test MAE: {selected_features_test_metrics['MAE']}\n")
        file.write(f"Test MSE: {selected_features_test_metrics['MSE']}\n")
        file.write(f"Test RMSE: {selected_features_test_metrics['RMSE']}\n")
        file.write(f"Test R2: {selected_features_test_metrics['R2']}\n")
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
    ) = evaluate_ridge_regularization_strengths(X_train, y_train)
    (
        random_forest_results_df,
        best_random_forest_params,
    ) = evaluate_random_forest_hyperparameters(X_train, y_train)
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
        selected_features_best_model,
        selected_features_best_model_name,
    ) = evaluate_models(
        selected_features_models,
        X_train[selected_features],
        y_train,
        X_val[selected_features],
        y_val,
    )
    feature_selection_results_df = build_feature_selection_comparison(
        results_df,
        selected_features_results_df,
    )

    test_metrics = evaluate_best_model(best_model, X_test, y_test)
    selected_features_test_metrics = evaluate_best_model(
        selected_features_best_model,
        X_test[selected_features],
        y_test,
    )
    classification_metrics = evaluate_classification_example(
        X_train,
        y_class_train,
        X_test,
        y_class_test,
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
        test_metrics,
        selected_features_test_metrics,
        classification_metrics,
    )
    dump(best_model, BEST_MODEL_PATH)

    print(f"\nNajbolji alpha za Ridge regresiju: {best_ridge_alpha}")
    print(f"\nNajbolji broj suseda za KNN regresiju: {best_knn_neighbors}")
    print(f"\nNajbolji parametri za Random Forest: {best_random_forest_params}")
    print(f"\nIzabrani najznacajniji atributi: {', '.join(selected_features)}")
    print(f"\nNajbolji model: {best_model_name}")
    print("Test MAE:", test_metrics["MAE"])
    print("Test MSE:", test_metrics["MSE"])
    print("Test RMSE:", test_metrics["RMSE"])
    print("Test R2:", test_metrics["R2"])
    print("\nNajbolji model sa izabranim atributima:", selected_features_best_model_name)
    print("Selected features Test MAE:", selected_features_test_metrics["MAE"])
    print("Selected features Test MSE:", selected_features_test_metrics["MSE"])
    print("Selected features Test RMSE:", selected_features_test_metrics["RMSE"])
    print("Selected features Test R2:", selected_features_test_metrics["R2"])
    print("\nPrimer klasifikacije za hawk_dominant:")
    print("Model:", classification_metrics["model"])
    print("Test accuracy:", classification_metrics["accuracy"])
    print("Klasifikacija predvidja samo klasu, a regresija procenjuje tacan final_hawk.")
    print(f"\nRezultati su sačuvani u {METRICS_DIR}.")
    print(f"Najbolji model je sačuvan u {BEST_MODEL_PATH}.")


if __name__ == "__main__":
    main()
