import numpy as np
import pandas as pd
from pathlib import Path


RAW_DATA_PATH = Path("data/raw/hawk_dove_dataset.csv")
RANDOM_SEED = 42


def simulate_hawk_dove(
    V,
    C,
    x0,
    iterations,
    population_size,
    rng,
    learning_rate,
    mutation_rate,
    environment_volatility,
    noise_std=0.05,
):
    x = x0

    for _ in range(iterations):
        current_V = max(0.1, rng.normal(V, environment_volatility * V))
        current_C = max(0.1, rng.normal(C, environment_volatility * C))

        hawk_fitness = x * ((current_V - current_C) / 2) + (1 - x) * current_V
        dove_fitness = (1 - x) * (current_V / 2)

        average_fitness = x * hawk_fitness + (1 - x) * dove_fitness

        noise = rng.normal(0, noise_std)
        x = x + learning_rate * x * (hawk_fitness - average_fitness) + noise
        x = x + mutation_rate * (1 - 2 * x)
        x = max(0, min(1, x))

        hawk_count = rng.binomial(population_size, x)
        x = hawk_count / population_size

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
        learning_rate = rng.uniform(0.03, 0.15)
        mutation_rate = rng.uniform(0.001, 0.03)
        environment_volatility = rng.uniform(0.01, 0.12)
        theoretical_hawk = min(1, V / C)
        final_hawk = simulate_hawk_dove(
            V,
            C,
            initial_hawk,
            iterations,
            population_size,
            rng,
            learning_rate,
            mutation_rate,
            environment_volatility,
        )
        final_dove = 1 - final_hawk

        data.append({
            "V": V,
            "C": C,
            "cost_value_ratio": C / V,
            "value_cost_ratio": V / C,
            "conflict_severity": C - V,
            "theoretical_hawk": theoretical_hawk,
            "initial_hawk": initial_hawk,
            "iterations": iterations,
            "learning_rate": learning_rate,
            "mutation_rate": mutation_rate,
            "environment_volatility": environment_volatility,
            "final_hawk": final_hawk,
            "final_dove": final_dove,
            "hawk_dominant": int(final_hawk > 0.5),
            "dominant_strategy": "hawk" if final_hawk > 0.5 else "dove",
            "population_size": population_size,
        })

    return pd.DataFrame(data)


def main():
    dataset = generate_dataset(10000)

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(RAW_DATA_PATH, index=False)

    print(f"Dataset uspešno sačuvan kao {RAW_DATA_PATH}")


if __name__ == "__main__":
    main()
