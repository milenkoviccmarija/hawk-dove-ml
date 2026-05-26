import numpy as np
import pandas as pd


def simulate_hawk_dove(V, C, x0, iterations, learning_rate=0.1):
    x = x0

    for _ in range(iterations):
        hawk_fitness = x * ((V - C) / 2) + (1 - x) * V
        dove_fitness = (1 - x) * (V / 2)

        average_fitness = x * hawk_fitness + (1 - x) * dove_fitness

        x = x + learning_rate * x * (hawk_fitness - average_fitness)
        x = max(0, min(1, x))

    return x


def generate_dataset(num_samples):
    data = []

    for _ in range(num_samples):
        V = np.random.uniform(1, 20)
        C = np.random.uniform(1, 30)
        initial_hawk = np.random.uniform(0, 1)
        iterations = np.random.randint(50, 200)

        final_hawk = simulate_hawk_dove(V, C, initial_hawk, iterations)
        final_dove = 1 - final_hawk

        data.append({
            "V": V,
            "C": C,
            "initial_hawk": initial_hawk,
            "initial_dove": 1 - initial_hawk,
            "iterations": iterations,
            "final_hawk": final_hawk,
            "final_dove": final_dove
        })

    return pd.DataFrame(data)


dataset = generate_dataset(1000)

print(dataset.head())

dataset.to_csv("hawk_dove_dataset.csv", index=False)

print("Dataset uspešno sačuvan kao hawk_dove_dataset.csv")