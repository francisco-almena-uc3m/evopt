import argparse
import io
import math
import urllib.request
import zipfile
from datetime import datetime
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_california_housing,
    fetch_openml,
    load_breast_cancer,
    load_iris,
    load_wine,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from evopt import EvolutionaryOptimizer
from evopt.model_utils import (
    ModelConfig,
    TaskMetric,
    get_supported_models,
    get_task_metric,
    normalize_task_type,
    tune_models,
)

RANDOM_STATE = 1
SHOW_PLOTS = False

CLASSIFICATION_DATASETS = [
    "adult",
    "breast_cancer",
    "churn",
    "credit_default",
    "iris",
    "wine",
    "letter",
    "magic_telescope",
    "online_shoppers",
    "spam",
    "titanic",
    "airlines",
    "creditcard",
    "poker-hand-training-true",
]
REGRESSION_DATASETS = [
    "airfoil_self_noise",
    "bike_sharing_day",
    "bike_sharing_hour",
    "california",
    "concrete_compressive_strength",
    "energy_efficiency_cooling",
    "energy_efficiency_heating",
    "online_news_popularity",
    "student_performance_math",
    "student_performance_portuguese",
    "superconduct",
]
DATASETS = CLASSIFICATION_DATASETS + REGRESSION_DATASETS
DEFAULT_DATASETS = ("california",)


MAX_CATEGORICAL_UNIQUE = 25

CALIFORNIA_NUMERIC_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
IRIS_NUMERIC_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
WINE_NUMERIC_COLUMNS = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280/od315_of_diluted_wines",
    "proline",
]

TITANIC_NUMERIC_COLUMNS = ["age", "sibsp", "parch", "fare", "body"]
TITANIC_CATEGORICAL_COLUMNS = ["pclass", "sex", "embarked", "boat"]
CHURN_NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CHURN_CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
ONLINE_SHOPPERS_CATEGORICAL_COLUMNS = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]
ADULT_NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
ADULT_CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
STUDENT_NUMERIC_COLUMNS = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
    "G1",
    "G2",
]
STUDENT_CATEGORICAL_COLUMNS = [
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
]
BIKE_NUMERIC_COLUMNS = ["temp", "atemp", "hum", "windspeed"]
BIKE_CATEGORICAL_COLUMNS = ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]
BIKE_HOUR_CATEGORICAL_COLUMNS = BIKE_CATEGORICAL_COLUMNS + ["hr"]
ENERGY_EFFICIENCY_NUMERIC_COLUMNS = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]
CONCRETE_NUMERIC_COLUMNS = [
    "cement",
    "blast_furnace_slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
]
AIRFOIL_NUMERIC_COLUMNS = ["frequency", "angle", "length", "velocity", "thickness"]
AIRLINES_NUMERIC_COLUMNS = ["Flight", "DayOfWeek", "Time", "Length"]
AIRLINES_CATEGORICAL_COLUMNS = ["Airline", "AirportFrom", "AirportTo"]

DatasetLoader = Callable[[], pd.DataFrame]

COMMON_EVAL_MODELS = {
    "regression": (
        "linear_regression",
        "ridge",
        "lasso",
        "elastic_net",
        "knn_regressor",
        "decision_tree_regressor",
        "random_forest_regressor",
        "xgb_regressor",
        "svr",
        "mlp_regressor",
    ),
    "classification": (
        "logistic_regression",
        "gaussian_nb",
        "lda",
        "qda",
        "knn_classifier",
        "decision_tree_classifier",
        "random_forest_classifier",
        "xgb_classifier",
        "svc",
        "mlp_classifier",
    ),
}


def log(msg: str) -> None:
    """Imprime un mensaje con timestamp HH:MM:SS."""
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")


def _format_experiment_results(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    label = summary.get("label", "")
    dataset = summary.get("dataset", "")
    task_type = summary.get("task_type", "")
    metric_label = summary.get("metric_label", "")
    eval_model_names = summary.get("eval_model_names", []) or []

    lines.append(f"=== Experimento: {label} ===")
    if dataset:
        lines.append(f"dataset: {dataset}")
    if task_type:
        lines.append(f"task_type: {task_type}")
    if metric_label:
        lines.append(f"metric: {metric_label}")
    if eval_model_names:
        lines.append(f"models: {', '.join(eval_model_names)}")

    per_metric = summary.get("per_metric", {}) or {}
    metric_order = summary.get("metric_order") or list(per_metric.keys())
    for metric_name in metric_order:
        metric_info = per_metric.get(metric_name)
        if not metric_info:
            continue
        lines.append(f"metric: {metric_name}")
        per_model = metric_info.get("per_model", {}) or {}
        model_names = eval_model_names or list(per_model.keys())
        for name in model_names:
            vals = per_model.get(name)
            if not vals:
                continue
            base = vals.get("base", float("nan"))
            opt = vals.get("opt", float("nan"))
            delta = vals.get("delta", float("nan"))
            delta_pct = vals.get("delta_pct", float("nan"))
            lines.append(
                f"  {name}: base={base:.6f} opt={opt:.6f} delta={delta:+.6f} delta_pct={delta_pct:+.2f}%"
            )
        mean_base = metric_info.get("mean_base", float("nan"))
        mean_opt = metric_info.get("mean_opt", float("nan"))
        mean_delta = metric_info.get("mean_delta", float("nan"))
        mean_pct = metric_info.get("mean_pct", float("nan"))
        lines.append(
            f"  MEDIA: base={mean_base:.6f} opt={mean_opt:.6f} "
            f"delta={mean_delta:+.6f} delta_pct={mean_pct:+.2f}%"
        )

    kept = summary.get("features_kept", []) or []
    removed = summary.get("features_removed", []) or []
    created = summary.get("features_created", []) or []
    if kept or removed or created:
        lines.append(f"features_kept: {', '.join(kept) if kept else '(none)'}")
        lines.append(f"features_removed: {', '.join(removed) if removed else '(none)'}")
        lines.append(f"features_created: {', '.join(created) if created else '(none)'}")

    return "\n".join(lines)


def _load_california_frame() -> pd.DataFrame:
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy() if getattr(data, "frame", None) is not None else data.data.copy()
    target_col = "MedHouseVal"
    if target_col not in df.columns:
        df[target_col] = data.target
    return df


def _load_iris_frame() -> pd.DataFrame:
    X, y = load_iris(return_X_y=True, as_frame=True)
    df = X.copy()
    df["target"] = y
    return df


def _load_wine_frame() -> pd.DataFrame:
    X, y = load_wine(return_X_y=True, as_frame=True)
    df = X.copy()
    df["target"] = y
    return df


def _load_titanic_frame() -> pd.DataFrame:
    data = fetch_openml("titanic", version=1, as_frame=True)
    df = data.frame.copy()
    df = df.drop(columns=["name", "ticket"], errors="ignore")
    target_col = data.target.name or "survived"
    if target_col not in df.columns:
        df[target_col] = data.target
    return df


def _load_spam_frame() -> pd.DataFrame:
    data = fetch_openml("spambase", version=1, as_frame=True)
    df = data.frame.copy()
    target_col = data.target.name or "class"
    if target_col not in df.columns:
        df[target_col] = data.target
    return df


def _load_letter_frame() -> pd.DataFrame:
    data = fetch_openml("letter", version=1, as_frame=True)
    df = data.frame.copy()
    target_col = data.target.name or "class"
    if target_col not in df.columns:
        df[target_col] = data.target
    return df


def _load_magic_telescope_frame() -> pd.DataFrame:
    data = fetch_openml("MagicTelescope", version=1, as_frame=True)
    df = data.frame.copy()
    df = df.rename(columns=lambda c: c.rstrip(":") if isinstance(c, str) else c)
    target_col = (data.target.name or "class").rstrip(":")
    if target_col not in df.columns:
        df[target_col] = data.target
    return df


def _load_churn_frame() -> pd.DataFrame:
    data = fetch_openml("telco-customer-churn", version=1, as_frame=True)
    df = data.frame.copy()
    target_col = data.target.name or "Churn"
    if target_col not in df.columns:
        df[target_col] = data.target
    return df


def _load_online_shoppers_frame() -> pd.DataFrame:
    data = fetch_openml("online_shoppers_intention", version=1, as_frame=True)
    df = data.frame.copy()
    target_col = data.target.name or "Revenue"
    if target_col not in df.columns:
        df[target_col] = data.target
    return df


def _load_credit_default_frame() -> pd.DataFrame:
    data = fetch_openml("default-of-credit-card-clients", version=1, as_frame=True)
    df = data.frame.copy()
    target_col = data.target.name or "y"
    if target_col not in df.columns:
        df[target_col] = data.target
    return df


def _load_adult_frame() -> pd.DataFrame:
    data = fetch_openml("adult", version=2, as_frame=True)
    df = data.frame.copy()
    target_col = data.target.name or "class"
    if target_col not in df.columns:
        df[target_col] = data.target
    return df


def _load_creditcard_frame() -> pd.DataFrame:
    data = fetch_openml("creditcard", version=1, as_frame=True)
    df = data.frame.copy()
    original_target = data.target.name if data.target is not None else None
    target_col = "target"
    if data.target is not None:
        df[target_col] = data.target
    if original_target and original_target != target_col:
        df = df.drop(columns=[original_target], errors="ignore")
    return df


def _load_airlines_frame() -> pd.DataFrame:
    data = fetch_openml("airlines", version=1, as_frame=True)
    df = data.frame.copy()
    original_target = data.target.name if data.target is not None else None
    normalize = {
        "airline": "Airline",
        "flight": "Flight",
        "airportfrom": "AirportFrom",
        "airportto": "AirportTo",
        "dayofweek": "DayOfWeek",
        "time": "Time",
        "length": "Length",
        "delay": "Delay",
    }
    rename_map = {}
    for col in df.columns:
        if isinstance(col, str):
            desired = normalize.get(col.lower())
            if desired and col != desired:
                rename_map[col] = desired
    if rename_map:
        df = df.rename(columns=rename_map)
        if original_target in rename_map:
            original_target = rename_map[original_target]
    target_col = "target"
    if data.target is not None:
        df[target_col] = data.target
    if original_target and original_target != target_col:
        df = df.drop(columns=[original_target], errors="ignore")
    return df


def _load_poker_hand_training_true_frame() -> pd.DataFrame:
    data = fetch_openml("poker-hand-training-true", version=1, as_frame=True)
    df = data.frame.copy()
    original_target = data.target.name if data.target is not None else None
    target_col = "target"
    if data.target is not None:
        df[target_col] = data.target
    if original_target and original_target != target_col:
        df = df.drop(columns=[original_target], errors="ignore")
    return df


def _load_cancer_frame() -> pd.DataFrame:
    data = load_breast_cancer(as_frame=True)
    return data.frame.copy()


def _load_student_performance_frame(file_name: str) -> pd.DataFrame:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        with zf.open(file_name) as f:
            df = pd.read_csv(f, sep=";")
    return df


BIKE_SHARING_ZIP_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
)
_BIKE_SHARING_ZIP_BYTES: Optional[bytes] = None


def _load_bike_sharing_frame(file_name: str) -> pd.DataFrame:
    global _BIKE_SHARING_ZIP_BYTES
    if _BIKE_SHARING_ZIP_BYTES is None:
        with urllib.request.urlopen(BIKE_SHARING_ZIP_URL) as resp:
            _BIKE_SHARING_ZIP_BYTES = resp.read()
    with zipfile.ZipFile(io.BytesIO(_BIKE_SHARING_ZIP_BYTES)) as zf:
        with zf.open(file_name) as f:
            df = pd.read_csv(f)
    return df.drop(columns=["instant", "dteday"], errors="ignore")


def _load_bike_sharing_day_frame() -> pd.DataFrame:
    return _load_bike_sharing_frame("day.csv")


def _load_bike_sharing_hour_frame() -> pd.DataFrame:
    return _load_bike_sharing_frame("hour.csv")


def _load_energy_efficiency_frame() -> pd.DataFrame:
    data = fetch_openml("energy-efficiency", version=1, as_frame=True)
    return data.frame.copy()


def _load_concrete_frame() -> pd.DataFrame:
    data = fetch_openml(data_id=44959, as_frame=True)
    return data.frame.copy()


def _load_airfoil_frame() -> pd.DataFrame:
    data = fetch_openml("airfoil_self_noise", version=1, as_frame=True)
    return data.frame.copy()


def _load_online_news_popularity_frame() -> pd.DataFrame:
    data = fetch_openml("OnlineNewsPopularity", version=1, as_frame=True)
    df = data.frame.copy()
    target_col = data.target.name or "shares"
    if target_col not in df.columns:
        df[target_col] = data.target
    return df


def _load_superconduct_frame() -> pd.DataFrame:
    data = fetch_openml("superconduct", version=1, as_frame=True)
    df = data.frame.copy()
    target_col = data.target.name or "critical_temp"
    if target_col not in df.columns:
        df[target_col] = data.target
    return df


def _prepare_dataset(dataset_key: str):
    key = dataset_key.lower()
    if key not in DATASET_CONFIGS:
        raise ValueError(f"Dataset '{dataset_key}' no soportado. Opciones: {list(DATASET_CONFIGS)}")
    cfg = DATASET_CONFIGS[key]
    df = cfg["loader"]()
    task_type = normalize_task_type(cfg["task_type"])
    target_col = cfg["target_col"]
    numeric_cols = list(cfg.get("numeric_cols") or [])
    categorical_cols = list(cfg.get("categorical_cols") or [])

    if target_col not in df.columns:
        raise ValueError(f"No se encuentra la columna objetivo '{target_col}' en el dataset '{dataset_key}'.")

    return df, task_type, target_col, numeric_cols, categorical_cols, cfg.get("source")


def _prepare_feature_frames(
    df: pd.DataFrame,
    *,
    target_col: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    missing_numeric = [c for c in numeric_cols if c not in df.columns]
    missing_categorical = [c for c in categorical_cols if c not in df.columns]
    if missing_numeric:
        raise ValueError(f"Columnas numericas no encontradas: {missing_numeric}")
    if missing_categorical:
        raise ValueError(f"Columnas categoricas no encontradas: {missing_categorical}")

    if categorical_cols:
        high_cardinality = [
            col for col in categorical_cols if df[col].nunique(dropna=True) >= MAX_CATEGORICAL_UNIQUE
        ]
        if high_cardinality:
            log(f"Eliminando categoricas con alta cardinalidad: {high_cardinality}")
            categorical_cols = [col for col in categorical_cols if col not in high_cardinality]

    numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce").astype(float)
    if categorical_cols:
        categorical_df = df[categorical_cols].copy()
        for col in categorical_df.columns:
            series = categorical_df[col]
            if isinstance(series.dtype, pd.CategoricalDtype):
                series = series.cat.add_categories(["missing"]).fillna("missing")
            else:
                series = series.fillna("missing")
            categorical_df[col] = series
    else:
        categorical_df = pd.DataFrame(index=df.index)

    # Rellenar NaN en numéricas si aparecen (ej. titanic)
    if numeric_df.isna().any().any():
        numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))

    return numeric_df, categorical_df, numeric_cols, categorical_cols


DATASET_CONFIGS: Dict[str, Dict] = {
    "california": {
        "task_type": "regression",
        "target_col": "MedHouseVal",
        "numeric_cols": CALIFORNIA_NUMERIC_COLUMNS,
        "categorical_cols": [],
        "loader": lambda: _load_california_frame(),
        "source": "sklearn.datasets.fetch_california_housing",
    },
    "student_performance_math": {
        "task_type": "regression",
        "target_col": "G3",
        "numeric_cols": STUDENT_NUMERIC_COLUMNS,
        "categorical_cols": STUDENT_CATEGORICAL_COLUMNS,
        "loader": lambda: _load_student_performance_frame("student-mat.csv"),
        "source": "uci student.zip (student-mat.csv)",
    },
    "student_performance_portuguese": {
        "task_type": "regression",
        "target_col": "G3",
        "numeric_cols": STUDENT_NUMERIC_COLUMNS,
        "categorical_cols": STUDENT_CATEGORICAL_COLUMNS,
        "loader": lambda: _load_student_performance_frame("student-por.csv"),
        "source": "uci student.zip (student-por.csv)",
    },
    "iris": {
        "task_type": "classification",
        "target_col": "target",
        "numeric_cols": IRIS_NUMERIC_COLUMNS,
        "categorical_cols": [],
        "loader": lambda: _load_iris_frame(),
        "source": "sklearn.datasets.load_iris",
    },
    "wine": {
        "task_type": "classification",
        "target_col": "target",
        "numeric_cols": WINE_NUMERIC_COLUMNS,
        "categorical_cols": [],
        "loader": lambda: _load_wine_frame(),
        "source": "sklearn.datasets.load_wine",
    },
    "titanic": {
        "task_type": "classification",
        "target_col": "survived",
        "numeric_cols": TITANIC_NUMERIC_COLUMNS,
        "categorical_cols": TITANIC_CATEGORICAL_COLUMNS,
        "loader": lambda: _load_titanic_frame(),
        "source": "sklearn.datasets.fetch_openml('titanic')",
    },
    "spam": {
        "task_type": "classification",
        "target_col": "class",
        "numeric_cols": [],
        "categorical_cols": [],
        "loader": lambda: _load_spam_frame(),
        "source": "sklearn.datasets.fetch_openml('spambase')",
    },
    "letter": {
        "task_type": "classification",
        "target_col": "class",
        "numeric_cols": [],
        "categorical_cols": [],
        "loader": lambda: _load_letter_frame(),
        "source": "sklearn.datasets.fetch_openml('letter')",
    },
    "magic_telescope": {
        "task_type": "classification",
        "target_col": "class",
        "numeric_cols": [],
        "categorical_cols": [],
        "loader": lambda: _load_magic_telescope_frame(),
        "source": "sklearn.datasets.fetch_openml('MagicTelescope')",
    },
    "online_shoppers": {
        "task_type": "classification",
        "target_col": "Revenue",
        "numeric_cols": [],
        "categorical_cols": ONLINE_SHOPPERS_CATEGORICAL_COLUMNS,
        "loader": lambda: _load_online_shoppers_frame(),
        "source": "sklearn.datasets.fetch_openml('online_shoppers_intention')",
    },
    "churn": {
        "task_type": "classification",
        "target_col": "Churn",
        "numeric_cols": CHURN_NUMERIC_COLUMNS,
        "categorical_cols": CHURN_CATEGORICAL_COLUMNS,
        "loader": lambda: _load_churn_frame(),
        "source": "sklearn.datasets.fetch_openml('telco-customer-churn')",
    },
    "credit_default": {
        "task_type": "classification",
        "target_col": "y",
        "numeric_cols": [],
        "categorical_cols": [],
        "loader": lambda: _load_credit_default_frame(),
        "source": "sklearn.datasets.fetch_openml('default-of-credit-card-clients')",
    },
    "adult": {
        "task_type": "classification",
        "target_col": "class",
        "numeric_cols": ADULT_NUMERIC_COLUMNS,
        "categorical_cols": ADULT_CATEGORICAL_COLUMNS,
        "loader": lambda: _load_adult_frame(),
        "source": "sklearn.datasets.fetch_openml('adult')",
    },
    "breast_cancer": {
        "task_type": "classification",
        "target_col": "target",
        "numeric_cols": [],
        "categorical_cols": [],
        "loader": lambda: _load_cancer_frame(),
        "source": "sklearn.datasets.load_breast_cancer",
    },
    "creditcard": {
        "task_type": "classification",
        "target_col": "target",
        "numeric_cols": [],
        "categorical_cols": [],
        "loader": lambda: _load_creditcard_frame(),
        "source": "sklearn.datasets.fetch_openml('creditcard')",
    },
    "airlines": {
        "task_type": "classification",
        "target_col": "target",
        "numeric_cols": AIRLINES_NUMERIC_COLUMNS,
        "categorical_cols": AIRLINES_CATEGORICAL_COLUMNS,
        "loader": lambda: _load_airlines_frame(),
        "source": "sklearn.datasets.fetch_openml('airlines')",
    },
    "poker-hand-training-true": {
        "task_type": "classification",
        "target_col": "target",
        "numeric_cols": [],
        "categorical_cols": [],
        "loader": lambda: _load_poker_hand_training_true_frame(),
        "source": "sklearn.datasets.fetch_openml('poker-hand-training-true')",
    },
    "bike_sharing_day": {
        "task_type": "regression",
        "target_col": "cnt",
        "numeric_cols": BIKE_NUMERIC_COLUMNS,
        "categorical_cols": BIKE_CATEGORICAL_COLUMNS,
        "loader": lambda: _load_bike_sharing_day_frame(),
        "source": "uci Bike-Sharing-Dataset.zip (day.csv)",
    },
    "bike_sharing_hour": {
        "task_type": "regression",
        "target_col": "cnt",
        "numeric_cols": BIKE_NUMERIC_COLUMNS,
        "categorical_cols": BIKE_HOUR_CATEGORICAL_COLUMNS,
        "loader": lambda: _load_bike_sharing_hour_frame(),
        "source": "uci Bike-Sharing-Dataset.zip (hour.csv)",
    },
    "energy_efficiency_heating": {
        "task_type": "regression",
        "target_col": "y1",
        "numeric_cols": ENERGY_EFFICIENCY_NUMERIC_COLUMNS,
        "categorical_cols": [],
        "loader": lambda: _load_energy_efficiency_frame(),
        "source": "sklearn.datasets.fetch_openml('energy-efficiency')",
    },
    "energy_efficiency_cooling": {
        "task_type": "regression",
        "target_col": "y2",
        "numeric_cols": ENERGY_EFFICIENCY_NUMERIC_COLUMNS,
        "categorical_cols": [],
        "loader": lambda: _load_energy_efficiency_frame(),
        "source": "sklearn.datasets.fetch_openml('energy-efficiency')",
    },
    "concrete_compressive_strength": {
        "task_type": "regression",
        "target_col": "strength",
        "numeric_cols": CONCRETE_NUMERIC_COLUMNS,
        "categorical_cols": [],
        "loader": lambda: _load_concrete_frame(),
        "source": "sklearn.datasets.fetch_openml(data_id=44959)",
    },
    "airfoil_self_noise": {
        "task_type": "regression",
        "target_col": "pressure",
        "numeric_cols": AIRFOIL_NUMERIC_COLUMNS,
        "categorical_cols": [],
        "loader": lambda: _load_airfoil_frame(),
        "source": "sklearn.datasets.fetch_openml('airfoil_self_noise')",
    },
    "online_news_popularity": {
        "task_type": "regression",
        "target_col": "shares",
        "numeric_cols": [],
        "categorical_cols": [],
        "loader": lambda: _load_online_news_popularity_frame(),
        "source": "sklearn.datasets.fetch_openml('OnlineNewsPopularity')",
    },
    "superconduct": {
        "task_type": "regression",
        "target_col": "critical_temp",
        "numeric_cols": [],
        "categorical_cols": [],
        "loader": lambda: _load_superconduct_frame(),
        "source": "sklearn.datasets.fetch_openml('superconduct')",
    },
}


def compute_metric(metric: TaskMetric, model_config: ModelConfig, X_train, y_train, X_test, y_test) -> float:
    """Entrena el estimator contenido en el ModelConfig y devuelve la métrica en test."""
    estimator = model_config.make_estimator()
    estimator.fit(X_train, y_train)
    return metric.evaluate_estimator(estimator, X_test, y_test)


def _metric_order(task_type: str) -> List[str]:
    if normalize_task_type(task_type) == "classification":
        return ["f1", "accuracy", "auc"]
    return ["mse"]


def _compute_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
    }


def _get_classification_scores(estimator, X):
    if hasattr(estimator, "predict_proba"):
        try:
            return estimator.predict_proba(X)
        except Exception:
            pass
    if hasattr(estimator, "decision_function"):
        try:
            return estimator.decision_function(X)
        except Exception:
            pass
    return None


def _compute_classification_metrics(y_true, y_pred, y_score) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    auc_val = float("nan")
    if y_score is not None:
        try:
            score_arr = np.asarray(y_score)
            classes = np.unique(y_true)
            if score_arr.ndim == 1 or score_arr.shape[1] == 1 or len(classes) == 2:
                if score_arr.ndim == 2 and score_arr.shape[1] >= 2:
                    score_use = score_arr[:, 1]
                else:
                    score_use = score_arr.reshape(-1)
                auc_val = float(roc_auc_score(y_true, score_use))
            else:
                auc_val = float(roc_auc_score(y_true, score_arr, multi_class="ovr", average="macro"))
        except Exception:
            auc_val = float("nan")
    metrics["auc"] = auc_val
    return metrics


def _extract_feature_importances(estimator) -> Optional[np.ndarray]:
    if hasattr(estimator, "feature_importances_"):
        return getattr(estimator, "feature_importances_", None)
    if hasattr(estimator, "named_steps"):
        model = estimator.named_steps.get("model")
        if model is not None and hasattr(model, "feature_importances_"):
            return getattr(model, "feature_importances_", None)
    return None


def _top_feature_importances(
    importances: np.ndarray,
    feature_names: Optional[List[str]],
    *,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    arr = np.asarray(importances, dtype=float).ravel()
    if arr.size == 0:
        return []
    if feature_names and len(feature_names) == arr.size:
        names = feature_names
    else:
        names = [f"x{i}" for i in range(arr.size)]
    order = np.argsort(arr)[::-1]
    top_n = min(top_k, arr.size)
    order = order[:top_n]
    return [(names[idx], float(arr[idx])) for idx in order]


def _print_rf_feature_importances(
    label: str,
    model_name: str,
    top_pairs: List[Tuple[str, float]],
) -> None:
    if not top_pairs:
        return
    header = f"{label} | {model_name}" if label else model_name
    print(f"\n=== Importancia de variables [{header}] ===")
    for name, val in top_pairs:
        print(f"{name}: {val:.6f}")


def evaluate_models_metrics(
    X_train,
    y_train,
    X_test,
    y_test,
    model_configs: List[ModelConfig],
    task_type: str,
    *,
    feature_names: Optional[List[str]] = None,
    importance_label: str = "",
    print_rf_importances: bool = False,
    rf_importances: Optional[Dict[str, List[Tuple[str, float]]]] = None,
) -> tuple[Dict[str, Dict[str, float]], Dict[str, float], List[str]]:
    per_model: Dict[str, Dict[str, float]] = {}
    metric_order = _metric_order(task_type)
    for cfg in model_configs:
        estimator = cfg.make_estimator()
        estimator.fit(X_train, y_train)
        if print_rf_importances and cfg.name.startswith("random_forest"):
            importances = _extract_feature_importances(estimator)
            if importances is not None:
                top_pairs = _top_feature_importances(importances, feature_names)
                _print_rf_feature_importances(
                    importance_label,
                    cfg.name,
                    top_pairs,
                )
                if rf_importances is not None:
                    rf_importances[cfg.name] = top_pairs
        if normalize_task_type(task_type) == "classification":
            y_pred = estimator.predict(X_test)
            y_score = _get_classification_scores(estimator, X_test)
            metrics = _compute_classification_metrics(y_test, y_pred, y_score)
        else:
            y_pred = estimator.predict(X_test)
            metrics = _compute_regression_metrics(y_test, y_pred)
        per_model[cfg.name] = metrics

    mean_metrics = {
        name: safe_mean([vals.get(name, float("nan")) for vals in per_model.values()])
        for name in metric_order
    }
    return per_model, mean_metrics, metric_order


def print_metrics_report(
    label: str,
    model_names: List[str],
    metric_order: List[str],
    task_type: str,
    base_metrics: Dict[str, Dict[str, float]],
    opt_metrics: Dict[str, Dict[str, float]],
    base_means: Dict[str, float],
    opt_means: Dict[str, float],
) -> None:
    task = normalize_task_type(task_type)
    better_high = task == "classification"

    def delta_val(base_val: float, opt_val: float) -> float:
        return opt_val - base_val if better_high else base_val - opt_val

    print(f"\n=== Metricas en test [{label}] ===")
    for name in model_names:
        base_vals = base_metrics.get(name, {})
        opt_vals = opt_metrics.get(name, {})
        base_txt = " | ".join(f"{m}: {base_vals.get(m, float('nan')):.6f}" for m in metric_order)
        opt_txt = " | ".join(f"{m}: {opt_vals.get(m, float('nan')):.6f}" for m in metric_order)
        delta_txt = " | ".join(
            f"{m}: {delta_val(base_vals.get(m, float('nan')), opt_vals.get(m, float('nan'))):+.6f}"
            for m in metric_order
        )
        print(f"{name:>15} | base: {base_txt} | optimizado: {opt_txt} | delta: {delta_txt}")
    base_mean_txt = " | ".join(f"{m}: {base_means.get(m, float('nan')):.6f}" for m in metric_order)
    opt_mean_txt = " | ".join(f"{m}: {opt_means.get(m, float('nan')):.6f}" for m in metric_order)
    delta_mean_txt = " | ".join(
        f"{m}: {delta_val(base_means.get(m, float('nan')), opt_means.get(m, float('nan'))):+.6f}"
        for m in metric_order
    )
    print(f"{'MEDIA':>15} | base: {base_mean_txt} | optimizado: {opt_mean_txt} | delta: {delta_mean_txt}")


def grid_search_and_score(
    X_train,
    y_train,
    X_test,
    y_test,
    model_names: List[str],
    metric: TaskMetric,
    random_state: int,
    task_type: str,
    *,
    standardize: bool = True,
    cv_folds: int = 3,
    label: str = "",
    verbose: bool = False,
) -> tuple[List[ModelConfig], Dict[str, float]]:
    print(f"=== Grid search {label} ({cv_folds}-fold) ===")
    configs = tune_models(
        X_train,
        y_train,
        model_names,
        standardize=standardize,
        cv_folds=cv_folds,
        random_state=random_state,
        task_type=task_type,
        metric=metric,
        verbose=verbose,
    )
    for cfg in configs:
        print(f"  Mejor params {cfg.name}: {cfg.best_params}")

    scores: Dict[str, float] = {}
    for cfg in configs:
        scores[cfg.name] = compute_metric(metric, cfg, X_train, y_train, X_test, y_test)
    return configs, scores


def safe_mean(values: List[float]) -> float:
    valid = [v for v in values if math.isfinite(v)]
    return mean(valid) if valid else float("nan")


def print_opt_report(
    label: str,
    opt: EvolutionaryOptimizer,
    model_names: List[str],
    metric: TaskMetric,
    feature_names: List[str],
    X_test,
    y_test,
) -> Dict[str, Any]:
    report = opt.report_final_selection(feature_names, X_test=X_test, y_test=y_test)
    metric_label = getattr(metric, "label", metric.name)
    better_high = bool(getattr(metric, "higher_is_better", False))
    direction_txt = "alto=mejor" if better_high else "bajo=mejor"
    best_score_display = (
        metric.display_value(report["best_score"]) if hasattr(metric, "display_value") else report["best_score"]
    )
    print(f"\n=== Reporte final [{label}] (ultima iteracion completada) ===")
    print("Iteracion:", report["best_iteration"])
    print(f"Score final (GA) [{metric_label} | {direction_txt}]: {best_score_display:.6f}")
    pop_estimate = getattr(opt, "_ga_pop_estimate", None)
    if getattr(opt, "_ga_pop_estimated", False) and pop_estimate is not None:
        pop_detail = f"{pop_estimate}"
        if getattr(opt, "run_final_ga", False):
            pop_detail = f"{pop_detail} | GA final estimada x4: {max(2, int(pop_estimate) * 4)}"
        print(f"Poblacion GA estimada: {pop_detail}")

    print("\nOriginal -> numero de transformaciones que la usan:")
    for name, exprs in report["original_to_transformations"].items():
        if exprs:
            print(f"  {name}: {len(exprs)} expr(s)")

    if "test_results" in report:
        print("\n=== Resultados en test (transformadas finales) ===")
        per_model_raw = report["test_results"]["per_model"]
        per_model = {
            name: (metric.display_value(val) if hasattr(metric, "display_value") else val)
            for name, val in per_model_raw.items()
        }
        for name in model_names:
            score_val = per_model.get(name, float("nan"))
            print(f"{name:>15} | {metric_label}: {score_val:.6f}")
        mean_score = safe_mean(list(per_model.values()))
        print(f"{metric_label} medio: {mean_score:.6f}")
    return report


def print_feature_selection_summary(
    label: str,
    feature_names: List[str],
    report: Dict[str, Any],
) -> None:
    kept = list(report.get("selected_originals", []))
    kept_set = set(kept)
    removed = [name for name in feature_names if name not in kept_set]
    created = list(report.get("selected_transformations", []))

    print(f"\n=== Resumen final de features [{label}] ===")
    print("Originales mantenidas:", kept)
    print("Originales eliminadas:", removed)
    print("Features creadas (expr):")
    if created:
        for expr in created:
            print("  -", expr)
    else:
        print("  (ninguna)")


def compare_scores(
    label: str,
    eval_model_names: List[str],
    task_type: str,
    metric_order: List[str],
    base_metrics: Dict[str, Dict[str, float]],
    opt_metrics: Dict[str, Dict[str, float]],
) -> Dict:
    task = normalize_task_type(task_type)
    better_high = task == "classification"
    direction_txt = "alto=mejor" if better_high else "bajo=mejor"
    metrics = metric_order or _metric_order(task)

    summary: Dict[str, Any] = {
        "label": label,
        "task_type": task,
        "metric_order": metrics,
        "per_metric": {},
    }

    for metric_name in metrics:
        print(f"\n=== Comparativa por modelo [{label}] ({metric_name} en test, {direction_txt}) ===")
        base_scores_list: List[float] = []
        opt_scores_list: List[float] = []
        per_model: Dict[str, Dict[str, float]] = {}
        for name in eval_model_names:
            base_score = base_metrics.get(name, {}).get(metric_name, float("nan"))
            opt_score = opt_metrics.get(name, {}).get(metric_name, float("nan"))
            delta = opt_score - base_score if better_high else base_score - opt_score
            pct = (
                (delta / base_score * 100.0)
                if (math.isfinite(base_score) and abs(base_score) > 1e-12 and math.isfinite(delta))
                else float("nan")
            )
            base_scores_list.append(base_score)
            opt_scores_list.append(opt_score)
            per_model[name] = {
                "base": base_score,
                "opt": opt_score,
                "delta": delta,
                "delta_pct": pct,
            }
            print(
                f"{name:>15} | base: {base_score:.6f} | optimizado: {opt_score:.6f} "
                f"| delta: {delta:+.6f} | delta%: {pct:+.2f}%"
            )

        mean_base = safe_mean(base_scores_list)
        mean_opt = safe_mean(opt_scores_list)
        if math.isfinite(mean_base) and math.isfinite(mean_opt):
            mean_delta = mean_opt - mean_base if better_high else mean_base - mean_opt
        else:
            mean_delta = float("nan")
        mean_pct = (
            (mean_delta / mean_base * 100.0)
            if (math.isfinite(mean_base) and abs(mean_base) > 1e-12 and math.isfinite(mean_delta))
            else float("nan")
        )
        print(
            f"{'MEDIA':>15} | base: {mean_base:.6f} | optimizado: {mean_opt:.6f} "
            f"| delta: {mean_delta:+.6f} | delta%: {mean_pct:+.2f}%"
        )
        summary["per_metric"][metric_name] = {
            "per_model": per_model,
            "mean_base": mean_base,
            "mean_opt": mean_opt,
            "mean_delta": mean_delta,
            "mean_pct": mean_pct,
        }
    return summary


def _prepare_experiment_data(dataset_key: str) -> Dict[str, Any]:
    df, task_type, target_col, numeric_cols, categorical_cols, data_source = _prepare_dataset(dataset_key)
    metric = get_task_metric(task_type)
    metric_label = getattr(metric, "label", metric.name)

    numeric_df, categorical_df, numeric_cols, categorical_cols = _prepare_feature_frames(
        df,
        target_col=target_col,
        numeric_cols=numeric_cols or [c for c in df.select_dtypes(include=["number"]).columns if c != target_col],
        categorical_cols=categorical_cols,
    )

    encoded_cats = (
        pd.get_dummies(categorical_df, drop_first=False, dummy_na=False) if not categorical_df.empty else categorical_df
    )

    feature_df = pd.concat([numeric_df, encoded_cats], axis=1)
    feature_names = feature_df.columns.tolist()
    if not feature_names:
        raise ValueError("No hay columnas de entrada definidas.")

    X = feature_df.to_numpy(dtype=float)
    y_series = df[target_col]
    if task_type == "classification":
        if not pd.api.types.is_numeric_dtype(y_series):
            y, class_labels = pd.factorize(y_series)
            print(f"Clases detectadas ({len(class_labels)}): {list(class_labels)}")
        else:
            y = y_series.to_numpy()
    else:
        y = y_series.to_numpy(dtype=float)

    log("=== Configuracion de entrada ===")
    log(f"Dataset: {dataset_key} ({data_source})")
    log(f"Tarea: {task_type}")
    log(f"Objetivo: {target_col}")
    log(f"Features numericas: {numeric_cols}")
    log(f"Features categoricas: {categorical_cols}")

    stratify = y if task_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    log(f"Tarea: {task_type} | metrica base: {metric_label}")
    eval_candidates = COMMON_EVAL_MODELS.get(task_type, ())
    eval_model_names = [name for name in eval_candidates if name in get_supported_models(task_type)]
    if not eval_model_names:
        raise ValueError(f"No hay modelos comunes definidos para '{task_type}'.")
    log(f"Modelos evaluados externamente: {eval_model_names}")

    return {
        "dataset": dataset_key,
        "task_type": task_type,
        "metric": metric,
        "metric_label": metric_label,
        "feature_names": feature_names,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "eval_model_names": eval_model_names,
    }


def run_experiment(
    label: str,
    opt_kwargs: Dict,
    *,
    task_type: str,
    metric: TaskMetric,
    feature_names: List[str],
    X_train,
    y_train,
    X_test,
    y_test,
    eval_model_names: List[str],
    standardize_models: bool = True,
    random_state: int = 0,
) -> Dict:
    log(f"[{label}] === Inicio de prueba ===")
    opt_params = {"task_type": task_type, "random_state": random_state, "metric_name": metric.name}
    opt_params.update(opt_kwargs or {})
    opt = EvolutionaryOptimizer(**opt_params)
    model_names = list(opt.model_names)
    log(f"[{label}] Modelos utilizados por EVOPT: {model_names}")

    log(f"[{label}] Entrenando EvolutionaryOptimizer")
    opt.fit(X_train, y_train, feature_names=feature_names, verbose=True)

    X_train_opt = opt.transform(X_train)
    X_test_opt = opt.transform(X_test)
    log(f"[{label}] Shapes -> X_train_opt: {X_train_opt.shape}, X_test_opt: {X_test_opt.shape}")

    base_model_configs, _ = grid_search_and_score(
        X_train,
        y_train,
        X_test,
        y_test,
        eval_model_names,
        metric,
        random_state,
        task_type,
        label=f"{label} | datos originales",
        standardize=standardize_models,
        verbose=True,
    )

    opt_model_configs, _ = grid_search_and_score(
        X_train_opt,
        y_train,
        X_test_opt,
        y_test,
        eval_model_names,
        metric,
        random_state,
        task_type,
        label=f"{label} | datos transformados",
        standardize=standardize_models,
        verbose=True,
    )

    opt_report = print_opt_report(label, opt, model_names, metric, feature_names, X_test, y_test)
    opt_feature_names = feature_names
    kept_names: List[str] = []
    removed_names: List[str] = []
    created_names: List[str] = []
    if opt_report:
        kept_names = opt_report.get("selected_originals", []) or []
        created_names = opt_report.get("selected_transformations", []) or []
        kept_set = set(kept_names)
        removed_names = [name for name in feature_names if name not in kept_set]
        opt_feature_names = list(kept_names) + list(created_names)

    rf_importances: Dict[str, List[Tuple[str, float]]] = {}
    base_metrics, base_means, metric_order = evaluate_models_metrics(
        X_train,
        y_train,
        X_test,
        y_test,
        base_model_configs,
        task_type,
        feature_names=feature_names,
        importance_label=f"{label} | datos originales",
        print_rf_importances=False,
    )
    opt_metrics, opt_means, _ = evaluate_models_metrics(
        X_train_opt,
        y_train,
        X_test_opt,
        y_test,
        opt_model_configs,
        task_type,
        feature_names=opt_feature_names,
        importance_label=f"{label} | datos transformados",
        print_rf_importances=True,
        rf_importances=rf_importances,
    )

    summary = compare_scores(
        label,
        eval_model_names,
        task_type,
        metric_order,
        base_metrics,
        opt_metrics,
    )

    print_metrics_report(
        label,
        eval_model_names,
        metric_order,
        task_type,
        base_metrics,
        opt_metrics,
        base_means,
        opt_means,
    )

    pop_estimate = getattr(opt, "_ga_pop_estimate", None)
    if getattr(opt, "_ga_pop_estimated", False) and pop_estimate is not None:
        summary["ga_population_estimate"] = int(pop_estimate)
        if getattr(opt, "run_final_ga", False):
            summary["ga_population_final_estimate"] = max(2, int(pop_estimate) * 4)
    summary["features_kept"] = kept_names
    summary["features_removed"] = removed_names
    summary["features_created"] = created_names
    summary["rf_importances"] = rf_importances
    summary["rf_importance_label"] = f"{label} | datos transformados"

    if SHOW_PLOTS:
        try:
            opt.plot_history()
        except Exception as exc:
            print(f"[{label}] No se pudo pintar la evolucion EVOPT:", exc)

        try:
            opt.plot_feature_evolution_history(feature_names)
        except Exception as exc:
            print(f"[{label}] No se pudo pintar la evolucion de features:", exc)

        try:
            opt.plot_feature_dependency_matrix(feature_names)
        except Exception as exc:
            print(f"[{label}] No se pudo pintar la matriz de dependencias de features:", exc)

    print_feature_selection_summary(label, feature_names, opt_report)

    return summary


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EVOPT examples on one or more datasets.")
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset keys to run. Defaults to california.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print available dataset keys and exit.",
    )
    parser.add_argument(
        "--maxtime",
        type=int,
        default=None,
        help="Override EVOPT global time budget in seconds.",
    )
    return parser.parse_args(argv)


def _build_experiments(dataset_keys: List[str], maxtime: Optional[int]) -> List[Dict[str, Any]]:
    opt_kwargs: Dict[str, Any] = {}
    if maxtime is not None:
        opt_kwargs["maxtime"] = maxtime
    return [
        {"label": dataset_key, "dataset": dataset_key, "opt_kwargs": dict(opt_kwargs)}
        for dataset_key in dataset_keys
    ]


def main(argv: Optional[List[str]] = None):
    args = _parse_args(argv)

    if args.list_datasets:
        print("\n".join(DATASETS))
        return

    dataset_keys = [str(key).lower() for key in (args.datasets or DEFAULT_DATASETS)]
    unknown = [key for key in dataset_keys if key not in DATASETS]
    if unknown:
        raise ValueError(f"Datasets desconocidos: {unknown}. Opciones validas: {DATASETS}")

    log("Inicio de ejecucion")
    experiments = _build_experiments(dataset_keys, args.maxtime)
    multi_dataset = len(set(dataset_keys)) > 1
    prepared = {}
    summaries = []
    failures = []
    for exp in experiments:
        dataset_key = str(exp.get("dataset", DATASETS)).lower()
        label = exp["label"]
        if multi_dataset:
            label = f"{label} | {dataset_key}"

        try:
            if dataset_key not in prepared:
                prepared[dataset_key] = _prepare_experiment_data(dataset_key)
            data = prepared[dataset_key]

            summary = run_experiment(
                label=label,
                opt_kwargs=exp.get("opt_kwargs", {}),
                task_type=data["task_type"],
                metric=data["metric"],
                feature_names=data["feature_names"],
                X_train=data["X_train"],
                y_train=data["y_train"],
                X_test=data["X_test"],
                y_test=data["y_test"],
                eval_model_names=data["eval_model_names"],
                standardize_models=exp.get("standardize_models", True),
                random_state=RANDOM_STATE,
            )
            summary["dataset"] = dataset_key
            summary["metric_label"] = data["metric_label"]
            summary["eval_model_names"] = data["eval_model_names"]
            summaries.append(summary)
        except Exception as exc:
            log(f"[{label}] Error en experimento: {exc}")
            failures.append({"label": label, "dataset": dataset_key, "error": str(exc)})

    print("\n=== Resumen global de mejoras (test) ===")
    if not summaries:
        print("No se pudo completar ningun experimento.")
    else:
        for summary in summaries:
            print(f"\n[{summary['label']}]")
            pop_est = summary.get("ga_population_estimate")
            pop_final = summary.get("ga_population_final_estimate")
            if pop_est is not None:
                if pop_final is not None:
                    print(f"Poblacion GA estimada: {pop_est} | GA final estimada x4: {pop_final}")
                else:
                    print(f"Poblacion GA estimada: {pop_est}")

            per_metric = summary.get("per_metric")
            metric_order = summary.get("metric_order", [])
            eval_model_names = summary.get("eval_model_names", [])

            if per_metric:
                for metric_name in metric_order:
                    metric_summary = per_metric.get(metric_name)
                    if not metric_summary:
                        continue
                    print(f"\nMetrica: {metric_name}")
                    per_model = metric_summary.get("per_model", {})
                    for name in eval_model_names:
                        vals = per_model.get(name)
                        if not vals:
                            continue
                        print(
                            f"{name:>15} | base: {vals['base']:.6f} | optimizado: {vals['opt']:.6f} "
                            f"| delta: {vals['delta']:+.6f} | delta%: {vals['delta_pct']:+.2f}%"
                        )
                    print(
                        f"{'MEDIA':>15} | base: {metric_summary.get('mean_base', float('nan')):.6f} "
                        f"| optimizado: {metric_summary.get('mean_opt', float('nan')):.6f} "
                        f"| delta: {metric_summary.get('mean_delta', float('nan')):+.6f} "
                        f"| delta%: {metric_summary.get('mean_pct', float('nan')):+.2f}%"
                    )
            else:
                metric_label = summary.get("metric_label", "metric")
                print(f"\nMetrica: {metric_label}")
                per_model = summary.get("per_model", {})
                for name in eval_model_names:
                    vals = per_model.get(name)
                    if not vals:
                        continue
                    print(
                        f"{name:>15} | base: {vals['base']:.6f} | optimizado: {vals['opt']:.6f} "
                        f"| delta: {vals['delta']:+.6f} | delta%: {vals['delta_pct']:+.2f}%"
                    )
                print(
                    f"{'MEDIA':>15} | base: {summary.get('mean_base', float('nan')):.6f} "
                    f"| optimizado: {summary.get('mean_opt', float('nan')):.6f} "
                    f"| delta: {summary.get('mean_delta', float('nan')):+.6f} "
                    f"| delta%: {summary.get('mean_pct', float('nan')):+.2f}%"
                )

            kept_names = summary.get("features_kept", [])
            removed_names = summary.get("features_removed", [])
            created_names = summary.get("features_created", [])
            print("\nResumen de features finales:")
            print("Originales mantenidas:", kept_names)
            print("Originales eliminadas:", removed_names)
            print("Features creadas (expr):")
            if created_names:
                for expr in created_names:
                    print("  -", expr)
            else:
                print("  (ninguna)")

            rf_importances = summary.get("rf_importances", {})
            if rf_importances:
                importance_label = summary.get("rf_importance_label", summary.get("label", ""))
                for model_name, top_pairs in rf_importances.items():
                    _print_rf_feature_importances(importance_label, model_name, top_pairs)

    if failures:
        print("\n=== Experimentos fallidos ===")
        for fail in failures:
            print(f"{fail['label']} | {fail['dataset']}: {fail['error']}")

if __name__ == "__main__":
    main()
