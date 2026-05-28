import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge


# Učitavanje podataka
X_train = pd.read_csv("data/processed/X_train_scaled.csv")

y_train = pd.read_csv(
    "data/processed/y_train.csv"
).squeeze()

X_test = pd.read_csv("data/processed/X_test_scaled.csv")

y_test = pd.read_csv(
    "data/processed/y_test.csv"
).squeeze()


# Treniranje najboljeg modela
model = Ridge(alpha=1.0)

model.fit(X_train, y_train)


# Predikcije
y_pred = model.predict(X_test)


# ==============================
# Actual vs Predicted plot
# ==============================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Stvarne vrednosti")

plt.ylabel("Predikovane vrednosti")

plt.title("Stvarne vs Predikovane vrednosti")

plt.savefig(
    "results/figures/actual_vs_predicted.png"
)

plt.close()


# ==============================
# Residual plot
# ==============================

residuals = y_test - y_pred

plt.figure(figsize=(8, 6))

plt.scatter(y_pred, residuals)

plt.axhline(y=0)

plt.xlabel("Predikovane vrednosti")

plt.ylabel("Reziduali")

plt.title("Residual plot")

plt.savefig(
    "results/figures/residual_plot.png"
)

plt.close()


# ==============================
# Feature importance
# ==============================

coefficients = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": model.coef_
})

print("\nKoeficijenti modela:")
print(coefficients)

coefficients.to_csv(
    "results/metrics/feature_coefficients.csv",
    index=False
)


# ==============================
# Plot koeficijenata
# ==============================

plt.figure(figsize=(8, 6))

plt.bar(
    coefficients["Feature"],
    coefficients["Coefficient"]
)

plt.xticks(rotation=45)

plt.ylabel("Vrednost koeficijenta")

plt.title("Uticaj atributa na predikciju")

plt.tight_layout()

plt.savefig(
    "results/figures/feature_importance.png"
)

plt.close()


print("\nGrafikoni i koeficijenti su uspešno sačuvani.")