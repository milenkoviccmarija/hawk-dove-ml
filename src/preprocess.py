import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = Path("data/raw/hawk_dove_dataset.csv")
PROCESSED_DATA_DIR = Path("data/processed")
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

def load_dataset(path=RAW_DATA_PATH):
    return pd.read_csv(path)


def split_dataset(dataset):
    dataset = dataset.drop_duplicates()

    X = dataset[FEATURE_COLUMNS]
    y_regression = dataset[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_regression,
        test_size=0.20,
        random_state=RANDOM_SEED,
    )

    return X_train, X_test, y_train, y_test


def save_processed_data(X_train, X_test, y_train, y_test):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)

    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)


def main():
    dataset = load_dataset()

    X_train, X_test, y_train, y_test = split_dataset(dataset)

    print("\nVeličina skupova:")
    print("Train:", X_train.shape)
    print("Test:", X_test.shape)

    save_processed_data(X_train, X_test, y_train, y_test)

    print("\nPretprocesiranje uspešno završeno.")
    print(f"Sačuvani su ne-skalirani train i test skupovi u {PROCESSED_DATA_DIR}.")
    print("Skaliranje će se vršiti unutar Pipeline-a tokom treninga radi sprečavanja curenja podataka.")


if __name__ == "__main__":
    main()