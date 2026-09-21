"""Model persistence for GridPulse probabilistic forecasting (Phase 6A).

Provides robust, secure, and reproducible saving and loading of fitted
QuantileRegressionModel instances using LightGBM native text boosters and
human-readable JSON metadata sidecars.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import lightgbm as lgb

from gridpulse.forecast.models.quantile import QuantileRegressionModel


def save_model(
    model: QuantileRegressionModel,
    path: Path | str,
    *,
    data_status: str | None = None,
) -> Path:
    """Save a fitted QuantileRegressionModel to an explicit directory bundle.

    The bundle contains:
    - metadata.json: Hyperparameters, feature columns, fit stats, and provenance.
    - booster_p10.txt, booster_p50.txt, booster_p90.txt: Native LightGBM booster strings.

    Parameters
    ----------
    model : QuantileRegressionModel
        The fitted model instance to save.
    path : Path or str
        Target directory path for the model bundle.
    data_status : str, optional
        Optional data status label to record in metadata.

    Returns
    -------
    Path
        The absolute path to the saved model directory.
    """
    if not getattr(model, "_fitted", False) or not getattr(model, "_boosters", None):
        raise ValueError("Cannot save an unfitted model.")

    target_dir = Path(path)

    # Prepare metadata dictionary
    meta = model.metadata()
    if data_status is not None:
        meta["data_status"] = data_status

    booster_files = {}
    for alpha, booster in model._boosters.items():
        pct = int(round(float(alpha) * 100))
        filename = f"booster_p{pct:02d}.txt"
        booster_files[str(alpha)] = filename

    payload = {
        "metadata": meta,
        "booster_files": booster_files,
    }

    # Atomic write pattern using a temporary directory in the same parent
    parent = target_dir.parent
    parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=parent) as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Write metadata.json
        meta_file = tmp_path / "metadata.json"
        with open(meta_file, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, ensure_ascii=True, indent=2, sort_keys=True)
            fh.write("\n")

        # Write booster files
        for alpha, booster in model._boosters.items():
            pct = int(round(float(alpha) * 100))
            filename = f"booster_p{pct:02d}.txt"
            booster_path = tmp_path / filename
            model_str = booster.model_to_string()
            with open(booster_path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(model_str)

        # Atomic replacement
        if target_dir.exists():
            import shutil

            shutil.rmtree(target_dir, ignore_errors=True)
        tmp_path.replace(target_dir)

    return target_dir.resolve()


def load_model(path: Path | str) -> QuantileRegressionModel:
    """Load a fitted QuantileRegressionModel from a directory bundle.

    Parameters
    ----------
    path : Path or str
        Directory path containing metadata.json and booster text files.

    Returns
    -------
    QuantileRegressionModel
        The fully reconstructed and fitted model instance.
    """
    target_dir = Path(path)
    if not target_dir.exists() or not target_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {target_dir}")

    meta_file = target_dir / "metadata.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"Model metadata not found at {meta_file}")

    try:
        with open(meta_file, encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as e:
        raise ValueError(f"Invalid model metadata JSON at {meta_file}: {e}") from e

    if "metadata" not in payload or "booster_files" not in payload:
        raise ValueError(
            f"Malformed model bundle at {target_dir}: missing 'metadata' or 'booster_files'."
        )

    m = payload["metadata"]
    booster_files = payload["booster_files"]

    # Re-instantiate model with stored hyperparameters
    feature_columns = m.get("feature_columns", [])
    quantiles = tuple(float(q) for q in m.get("quantiles", [0.1, 0.5, 0.9]))
    n_estimators = int(m.get("n_estimators", 120))
    learning_rate = float(m.get("learning_rate", 0.05))
    num_leaves = int(m.get("num_leaves", 15))
    min_child_samples = int(m.get("min_child_samples", 3))
    random_state = int(m.get("random_state", 0))
    boost_from_average = bool(m.get("boost_from_average", True))
    min_train_rows = int(m.get("min_train_rows", 5))

    model = QuantileRegressionModel(
        feature_columns=feature_columns,
        quantiles=quantiles,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        random_state=random_state,
        boost_from_average=boost_from_average,
        min_train_rows=min_train_rows,
    )

    # Restore fit state and boosters
    model._fitted = bool(m.get("fitted", True))
    model._n_train_rows = int(m.get("n_train_rows", 0))
    model._n_dropped = int(m.get("n_rows_dropped_missing_features", 0))
    model._data_status = m.get("data_status")
    model._boosters = {}

    for alpha_str, filename in booster_files.items():
        alpha = float(alpha_str)
        booster_path = target_dir / filename
        if not booster_path.exists():
            raise FileNotFoundError(
                f"Missing booster file for quantile {alpha}: {booster_path}"
            )
        try:
            with open(booster_path, encoding="utf-8") as fh:
                model_str = fh.read()
            model._boosters[alpha] = lgb.Booster(model_str=model_str)
        except Exception as e:
            raise ValueError(
                f"Failed to load LightGBM booster from {booster_path}: {e}"
            ) from e

    model._crossing = {"n_detected": 0, "n_corrected": 0, "n_triples": 0}
    return model


__all__ = ["save_model", "load_model"]
