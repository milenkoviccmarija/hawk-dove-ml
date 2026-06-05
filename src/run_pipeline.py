import subprocess
import sys
from pathlib import Path


PIPELINE_STEPS = [
    ("Generisanje dataset-a", Path("src/main.py")),
    ("EDA analiza", Path("src/eda.py")),
    ("Pretprocesiranje i splitovanje", Path("src/preprocess.py")),
    ("Treniranje i izbor modela na validation skupu", Path("src/train.py")),
    ("Finalna evaluacija generalizacije na test skupu", Path("src/evaluate.py")),
    ("Primer predikcije", Path("src/predict.py")),
]

EXPECTED_OUTPUTS = [
    Path("data/raw/hawk_dove_dataset.csv"),
    Path("data/processed/X_train_scaled.csv"),
    Path("data/processed/X_val_scaled.csv"),
    Path("data/processed/X_test_scaled.csv"),
    Path("models/standard_scaler.joblib"),
    Path("models/best_model.joblib"),
    Path("results/metrics/results_summary.txt"),
    Path("results/metrics/evaluation_summary.txt"),
    Path("results/figures/actual_vs_predicted.png"),
    Path("results/figures/residual_plot.png"),
]


def print_data_flow_check():
    print("\nProvera podele podataka:", flush=True)
    print("- train: koristi se za fitovanje modela i cross-validation tuning.", flush=True)
    print("- validation: koristi se za poredjenje modela i izbor najboljeg modela.", flush=True)
    print("- test: koristi se samo u evaluate.py za finalnu proveru generalizacije.", flush=True)
    print("- predict: koristi sacuvani model i scaler za novi scenario.", flush=True)


def run_step(step_name, script_path):
    print(f"\n=== {step_name} ===", flush=True)
    subprocess.run(
        [sys.executable, str(script_path)],
        check=True,
    )


def verify_outputs():
    missing_outputs = [path for path in EXPECTED_OUTPUTS if not path.exists()]

    if missing_outputs:
        print("\nNedostaju ocekivani izlazi:", flush=True)
        for path in missing_outputs:
            print(f"- {path}", flush=True)
        raise SystemExit(1)

    print("\nSvi kljucni izlazi su napravljeni.", flush=True)


def main():
    print_data_flow_check()

    for step_name, script_path in PIPELINE_STEPS:
        run_step(step_name, script_path)

    verify_outputs()
    print("\nPipeline je uspesno zavrsen.", flush=True)


if __name__ == "__main__":
    main()
