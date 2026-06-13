import pandas as pd
from joblib import load
from pathlib import Path

from preprocess import FEATURE_COLUMNS


BEST_MODEL_PATH = Path("models/best_model.joblib")
SCALER_PATH = Path("models/standard_scaler.joblib")


EXAMPLE_SCENARIO = {
    "V": 12.0,
    "C": 18.0,
    "initial_hawk": 0.45,
    "iterations": 120,
    "learning_rate": 0.08,
    "mutation_rate": 0.01,
    "environment_volatility": 0.05,
    "population_size": 500,
}


def predict_final_hawk(scenario):
    model = load(BEST_MODEL_PATH)
    scaler = load(SCALER_PATH)

    scenario_df = pd.DataFrame([scenario], columns=FEATURE_COLUMNS)
    scenario_scaled = scaler.transform(scenario_df)
    scenario_scaled = pd.DataFrame(scenario_scaled, columns=FEATURE_COLUMNS)

    prediction = model.predict(scenario_scaled)[0]
    prediction = max(0, min(1, prediction))

    return prediction


def main():
    prediction = predict_final_hawk(EXAMPLE_SCENARIO)
    dominant_strategy = "hawk" if prediction > 0.5 else "dove"

    print("Primer ulaznog scenarija:")
    for column, value in EXAMPLE_SCENARIO.items():
        print(f"{column}: {value}")

    print("\nPredikcija:")
    print(f"final_hawk: {prediction:.4f}")
    print(f"final_dove: {1 - prediction:.4f}")
    print(f"dominant_strategy: {dominant_strategy}")


if __name__ == "__main__":
    main()