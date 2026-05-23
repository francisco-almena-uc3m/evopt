"""Shared runner used by the dataset example scripts."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as experiments


def run_dataset(dataset_key: str, opt_kwargs: Dict[str, Any] | None = None) -> Dict[str, Any]:
    data = experiments._prepare_experiment_data(dataset_key)
    summary = experiments.run_experiment(
        label=dataset_key,
        opt_kwargs=opt_kwargs or {},
        task_type=data["task_type"],
        metric=data["metric"],
        feature_names=data["feature_names"],
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_test=data["X_test"],
        y_test=data["y_test"],
        eval_model_names=data["eval_model_names"],
        standardize_models=True,
        random_state=experiments.RANDOM_STATE,
    )
    summary["dataset"] = dataset_key
    return summary
