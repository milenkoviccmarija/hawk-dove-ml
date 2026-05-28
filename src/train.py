import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)


X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()

X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()


models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.001),
    "KNN Regression": KNeighborsRegressor(n_neighbors=5),
    "Random Forest Regression": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
}


kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


results = []


for model_name, model in models.items():

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=kf,
        scoring="r2"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "model": model_name,
        "CV_R2_mean": np.mean(cv_scores),
        "CV_R2_std": np.std(cv_scores),
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2_test": r2
    })

    print(f"\n===== {model_name} =====")
    print("Cross-validation R2 scores:", cv_scores)
    print("Mean CV R2:", np.mean(cv_scores))
    print("Std CV R2:", np.std(cv_scores))
    print("Test MAE:", mae)
    print("Test MSE:", mse)
    print("Test RMSE:", rmse)
    print("Test R2:", r2)


results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/metrics/model_results.csv",
    index=False
)

with open("results/metrics/results_summary.txt", "w") as file:
    file.write("Rezultati evaluacije regresionih modela\n")
    file.write("======================================\n\n")

    for result in results:
        file.write(f"Model: {result['model']}\n")
        file.write(f"Mean CV R2: {result['CV_R2_mean']}\n")
        file.write(f"Std CV R2: {result['CV_R2_std']}\n")
        file.write(f"Test MAE: {result['MAE']}\n")
        file.write(f"Test MSE: {result['MSE']}\n")
        file.write(f"Test RMSE: {result['RMSE']}\n")
        file.write(f"Test R2: {result['R2_test']}\n")
        file.write("\n")


print("\nRezultati su sačuvani u results/metrics.")