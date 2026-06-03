import pandas as pd
from joblib import dump
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RAW_DATA_PATH = Path("data/raw/hawk_dove_dataset.csv")
PROCESSED_DATA_DIR = Path("data/processed")
SCALER_PATH = Path("models/standard_scaler.joblib")
RANDOM_SEED = 42

FEATURE_COLUMNS = [
    "V",
    "C",
    "initial_hawk",
    "iterations",
    "learning_rate",
    "mutation_rate",
    "environment_volatility",
    "population_size",
]
TARGET_COLUMN = "final_hawk"
CLASSIFICATION_TARGET_COLUMN = "hawk_dominant"


def load_dataset(path=RAW_DATA_PATH):
    return pd.read_csv(path)


def print_dataset_report(dataset):
    print("Prvih 5 redova dataset-a:")
    print(dataset.head())

    print("\nInformacije o dataset-u:")
    print(dataset.info())

    print("\nProvera nedostajućih vrednosti:")
    print(dataset.isnull().sum())

    print("\nProvera duplikata:")
    print("Broj duplikata:", dataset.duplicated().sum())


def split_dataset(dataset):
    dataset = dataset.drop_duplicates()

    X = dataset[FEATURE_COLUMNS]
    y_regression = dataset[TARGET_COLUMN]
    y_classification = dataset[CLASSIFICATION_TARGET_COLUMN]

    X_train, X_temp, y_train, y_temp, y_class_train, y_class_temp = train_test_split(
        X,
        y_regression,
        y_classification,
        test_size=0.30,
        random_state=RANDOM_SEED,
    )

    X_val, X_test, y_val, y_test, y_class_val, y_class_test = train_test_split(
        X_temp,
        y_temp,
        y_class_temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
    )

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


def scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def save_processed_data(
    X_train_scaled,
    X_val_scaled,
    X_test_scaled,
    y_train,
    y_val,
    y_test,
    y_class_train,
    y_class_val,
    y_class_test,
    scaler,
):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)

    X_train_scaled.to_csv(PROCESSED_DATA_DIR / "X_train_scaled.csv", index=False)
    X_val_scaled.to_csv(PROCESSED_DATA_DIR / "X_val_scaled.csv", index=False)
    X_test_scaled.to_csv(PROCESSED_DATA_DIR / "X_test_scaled.csv", index=False)

    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_val.to_csv(PROCESSED_DATA_DIR / "y_val.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)

    y_class_train.to_csv(PROCESSED_DATA_DIR / "y_class_train.csv", index=False)
    y_class_val.to_csv(PROCESSED_DATA_DIR / "y_class_val.csv", index=False)
    y_class_test.to_csv(PROCESSED_DATA_DIR / "y_class_test.csv", index=False)

    dump(scaler, SCALER_PATH)


def main():
    dataset = load_dataset()
    print_dataset_report(dataset)

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
    ) = split_dataset(dataset)

    print("\nVeličina skupova:")
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train,
        X_val,
        X_test,
    )

    save_processed_data(
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        y_class_train,
        y_class_val,
        y_class_test,
        scaler,
    )

    print("\nPretprocesiranje uspešno završeno.")
    print(f"Sačuvani su train, validation i test skupovi u {PROCESSED_DATA_DIR}.")
    print(f"Scaler je sačuvan u {SCALER_PATH}.")


if __name__ == "__main__":
    main()
