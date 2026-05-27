import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Učitavanje skupa podataka
dataset = pd.read_csv("hawk_dove_dataset.csv")

print("Prvih 5 redova dataset-a:")
print(dataset.head())

print("\nInformacije o dataset-u:")
print(dataset.info())

print("\nProvera nedostajućih vrednosti:")
print(dataset.isnull().sum())

print("\nProvera duplikata:")
print("Broj duplikata:", dataset.duplicated().sum())


# Uklanjanje duplikata ako postoje
dataset = dataset.drop_duplicates()


# Ulazni atributi
X = dataset[
    [
        "V",
        "C",
        "initial_hawk",
        "initial_dove",
        "iterations",
        "population_size",
    ]
]

# Target promenljiva
y = dataset["final_hawk"]


# Podela: 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
)

print("\nVeličina skupova:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# Skaliranje atributa
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# Pretvaranje nazad u DataFrame zbog preglednosti
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)


# Čuvanje pretprocesiranih podataka
X_train_scaled.to_csv("X_train_scaled.csv", index=False)
X_val_scaled.to_csv("X_val_scaled.csv", index=False)
X_test_scaled.to_csv("X_test_scaled.csv", index=False)

y_train.to_csv("y_train.csv", index=False)
y_val.to_csv("y_val.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\nPretprocesiranje uspešno završeno.")
print("Sačuvani su train, validation i test skupovi.")