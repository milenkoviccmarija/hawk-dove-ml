import pandas as pd
from joblib import load
from pathlib import Path

from preprocess import FEATURE_COLUMNS

BEST_PIPELINE_PATH = Path("models/best_pipeline.joblib")

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
    if not BEST_PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Nedostaje trenirani pipeline na lokaciji {BEST_PIPELINE_PATH}. Pokreni prvo train.py."
        )
        
    pipeline = load(BEST_PIPELINE_PATH)

    scenario_df = pd.DataFrame([scenario], columns=FEATURE_COLUMNS)
    
    prediction = pipeline.predict(scenario_df)[0]
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
    print(f"Dominantna strategija: {dominant_strategy}")


if __name__ == "__main__":
    main()