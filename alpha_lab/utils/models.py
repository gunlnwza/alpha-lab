from pathlib import Path

import joblib

_MODEL_ARTIFACTS = Path(__file__).parents[2] / ".model_artifacts"


def save_model(model, name: str, *, dir: str = _MODEL_ARTIFACTS, overwrite: bool = False, compress: int = 3):
    """
    Save sklearn model using joblib.

    Args:
        model: Trained sklearn model
        filepath: Destination path (.pkl)
        overwrite: Prevent accidental overwrite
        compress: 0-9 compression level
    """
    path = Path(dir, f"{name}.joblib")
    if path.exists() and not overwrite:
        raise FileExistsError(f"'{path.name}' already exists. Set overwrite=True to replace.")

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=compress)
    print(f"Model saved to {path.name}")


def load_model(name: str, *, dir: str = _MODEL_ARTIFACTS):
    """
    Load sklearn model saved with joblib.
    """
    path = Path(dir, f"{name}.joblib")
    if not path.exists():
        raise FileNotFoundError(f"'{path.name}' not found.")

    return joblib.load(path)
