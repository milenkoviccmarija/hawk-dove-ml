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
    ("Eksport modela", Path("src/export_model.py")),
]


EXPECTED_OUTPUTS = [
    Path("data/raw/hawk_dove_dataset.csv"),
    Path("data/processed/X_train.csv"),
    Path("data/processed/X_test.csv"),
    Path("models/best_pipeline.joblib"),  
    Path("results/metrics/results_summary.txt"),
    Path("results/metrics/evaluation_summary.txt"),
    Path("results/figures/hyperparameter_cv_scores.png"), 
    Path("deployment/model_metadata.json"),
    Path("deployment/hawk_dove_model_bundle.zip"),
]


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
    for step_name, script_path in PIPELINE_STEPS:
        run_step(step_name, script_path)

    verify_outputs()
    print("\nPipeline je uspesno zavrsen.", flush=True)


if __name__ == "__main__":
    main()
