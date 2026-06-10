import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from preprocess import FEATURE_COLUMNS, TARGET_COLUMN


MODELS_DIR = Path("models")
RESULTS_DIR = Path("results/metrics")
DEPLOYMENT_DIR = Path("deployment")

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
SCALER_PATH = MODELS_DIR / "standard_scaler.joblib"
DEPLOYED_MODEL_PATH = DEPLOYMENT_DIR / "best_model.joblib"
DEPLOYED_SCALER_PATH = DEPLOYMENT_DIR / "standard_scaler.joblib"
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
        "project": "Hawk-Dove ML",
        "model_type": "regression",
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "model_file": DEPLOYED_MODEL_PATH.name,
        "scaler_file": DEPLOYED_SCALER_PATH.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_output": {
            "final_hawk": "Predikovani finalni udeo hawk strategije.",
            "final_dove": "1 - final_hawk.",
            "dominant_strategy": "hawk ako je final_hawk > 0.5, inace dove.",
        },
        "metrics_summary": load_metric_summary(),
    }


def create_bundle():
    require_file(BEST_MODEL_PATH)
    require_file(SCALER_PATH)

    DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(BEST_MODEL_PATH, DEPLOYED_MODEL_PATH)
    shutil.copy2(SCALER_PATH, DEPLOYED_SCALER_PATH)

    metadata = build_metadata()
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with ZipFile(BUNDLE_PATH, "w", compression=ZIP_DEFLATED) as archive:
        archive.write(DEPLOYED_MODEL_PATH, arcname=DEPLOYED_MODEL_PATH.name)
        archive.write(DEPLOYED_SCALER_PATH, arcname=DEPLOYED_SCALER_PATH.name)
        archive.write(METADATA_PATH, arcname=METADATA_PATH.name)

    return metadata


def main():
    metadata = create_bundle()

    print("\nEksport modela je zavrsen.")
    print(f"Model: {DEPLOYED_MODEL_PATH}")
    print(f"Scaler: {DEPLOYED_SCALER_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"ZIP paket: {BUNDLE_PATH}")
    print(f"Atributi: {', '.join(metadata['feature_columns'])}")


if __name__ == "__main__":
    main()
