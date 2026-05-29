import numpy as np
import pandas as pd
from pathlib import Path


RAW_DATA_PATH = Path("data/raw/hawk_dove_dataset.csv")
RANDOM_SEED = 42


def simulate_hawk_dove(V, C, x0, iterations, learning_rate=0.1):
    x = x0

    for _ in range(iterations):
        hawk_fitness = x * ((V - C) / 2) + (1 - x) * V
        dove_fitness = (1 - x) * (V / 2)

        average_fitness = x * hawk_fitness + (1 - x) * dove_fitness

        x = x + learning_rate * x * (hawk_fitness - average_fitness)
        x = max(0, min(1, x))

    return x


def generate_dataset(num_samples, random_seed=RANDOM_SEED):
    rng = np.random.default_rng(random_seed)
    data = []

    for _ in range(num_samples):
        V = rng.uniform(1, 20)
        C = rng.uniform(1, 30)
        initial_hawk = rng.uniform(0, 1)
        iterations = rng.integers(50, 200)
        population_size = rng.integers(100, 1000)
        final_hawk = simulate_hawk_dove(V, C, initial_hawk, iterations)
        final_dove = 1 - final_hawk

        data.append({
            "V": V,
            "C": C,
            "initial_hawk": initial_hawk,
            "initial_dove": 1 - initial_hawk,
            "iterations": iterations,
            "final_hawk": final_hawk,
            "final_dove": final_dove,
            "population_size": population_size
        })

    return pd.DataFrame(data)


def main():
    dataset = generate_dataset(10000)

    print(dataset.head())

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(RAW_DATA_PATH, index=False)

    print(f"Dataset uspešno sačuvan kao {RAW_DATA_PATH}")


if __name__ == "__main__":
    main()
