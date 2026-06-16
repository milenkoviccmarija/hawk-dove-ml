import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from preprocess import FEATURE_COLUMNS, TARGET_COLUMN

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results/metrics")
DEPLOYMENT_DIR = Path("deployment")

BEST_PIPELINE_PATH = MODELS_DIR / "best_pipeline.joblib"
DEPLOYED_PIPELINE_PATH = DEPLOYMENT_DIR / "best_pipeline.joblib"
METADATA_PATH = DEPLOYMENT_DIR / "model_metadata.json"
BUNDLE_PATH = DEPLOYMENT_DIR / "hawk_dove_model_bundle.zip"


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Nedostaje fajl {path}. Pokreni prvo preprocess.py i train.py."
        )


def load_metric_summary():
    summary_path = RESULTS_DIR / "evaluation_summary.txt"
    if summary_path.exists():
        return summary_path.read_text(encoding="utf-8")
    return "Evaluacija jos nije pokrenuta."


def build_metadata():
    return {
        "project": "Hawk-Dove ML Pipeline",
        "model_type": "regression_pipeline",
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "pipeline_file": DEPLOYED_PIPELINE_PATH.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_output": {
            "final_hawk": "Predikovani finalni udeo hawk strategije (automatski skalirano unutar pipeline-a).",
            "final_dove": "1 - final_hawk.",
        },
        "metrics_summary": load_metric_summary(),
    }


def create_bundle():
    require_file(BEST_PIPELINE_PATH)

    DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(BEST_PIPELINE_PATH, DEPLOYED_PIPELINE_PATH)

    metadata = build_metadata()
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with ZipFile(BUNDLE_PATH, "w", compression=ZIP_DEFLATED) as archive:
        archive.write(DEPLOYED_PIPELINE_PATH, arcname=DEPLOYED_PIPELINE_PATH.name)
        archive.write(METADATA_PATH, arcname=METADATA_PATH.name)

    return metadata


def main():
    print("Započinjemo pakovanje modela za produkciju...")
    create_bundle()
    print(f"Uspešno kreiran paket: {BUNDLE_PATH}")
    print("Metapodaci uspešno generisani.")


if __name__ == "__main__":
    main()