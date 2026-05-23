from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

try:
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore
except ImportError:  # pragma: no cover
    XGBClassifier = None  # type: ignore
    XGBRegressor = None  # type: ignore

GRIDSEARCH_N_JOBS = -1


# ============================================================
#   Tareas y metricas
# ============================================================
@dataclass(frozen=True)
class TaskMetric:
    name: str
    loss_func: Callable[[np.ndarray, np.ndarray], float]
    grid_scoring: str
    task_type: str
    needs_proba: bool = False
    needs_decision: bool = False
    higher_is_better: bool = False

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(self.loss_func(y_true, y_pred))

    def display_value(self, loss_value: float) -> float:
        """Convierte la pérdida interna a una métrica interpretable."""
        if self.task_type == "classification":
            if not np.isfinite(loss_value):
                return 0.0
            return float(np.clip(1.0 - loss_value, 0.0, 1.0))
        return float(loss_value)

    def predict(self, estimator: BaseEstimator, X: np.ndarray) -> np.ndarray:
        """Obtiene las salidas adecuadas para la métrica (proba/decision/pred)."""
        if self.needs_proba and hasattr(estimator, "predict_proba"):
            try:
                return estimator.predict_proba(X)
            except Exception:
                pass
        if self.needs_decision and hasattr(estimator, "decision_function"):
            try:
                scores = estimator.decision_function(X)
                return scores
            except Exception:
                pass
        return estimator.predict(X)

    def evaluate_estimator(self, estimator: BaseEstimator, X: np.ndarray, y_true: np.ndarray) -> float:
        preds = self.predict(estimator, X)
        return self.loss(y_true, preds)


def normalize_task_type(task_type: str) -> str:
    task = str(task_type).strip().lower()
    if task.startswith("cls") or task.startswith("clas"):
        return "classification"
    return "regression"


DEFAULT_METRIC = {
    "regression": "mse",
    "classification": "f1",
}


def get_task_metric(task_type: str, metric_name: Optional[str] = None) -> TaskMetric:
    task = normalize_task_type(task_type)
    metric_key = (metric_name or DEFAULT_METRIC[task]).lower()

    if task == "regression":
        if metric_key == "mse":
            return TaskMetric(
                name="mse",
                loss_func=lambda y_true, y_pred: float(mean_squared_error(y_true, y_pred)),
                grid_scoring="neg_mean_squared_error",
                task_type=task,
            )
        if metric_key == "mae":
            return TaskMetric(
                name="mae",
                loss_func=lambda y_true, y_pred: float(mean_absolute_error(y_true, y_pred)),
                grid_scoring="neg_mean_absolute_error",
                task_type=task,
            )
        raise ValueError("Métrica de regresión no soportada. Usa 'mse' o 'mae'.")

    if task == "classification":
        if metric_key == "accuracy":
            return TaskMetric(
                name="accuracy",
                loss_func=lambda y_true, y_pred: float(1.0 - accuracy_score(y_true, y_pred)),
                grid_scoring="accuracy",
                task_type=task,
                higher_is_better=True,
            )

        if metric_key in {"f1", "f1_macro"}:
            return TaskMetric(
                name="f1",
                loss_func=lambda y_true, y_pred: float(1.0 - f1_score(y_true, y_pred, average="macro")),
                grid_scoring="f1_macro",
                task_type=task,
                higher_is_better=True,
            )

        if metric_key == "auc":
            def auc_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
                try:
                    score = roc_auc_score(y_true, y_pred, multi_class="ovr", average="macro")
                    return float(1.0 - score)
                except Exception:
                    return 1.0

            return TaskMetric(
                name="auc",
                loss_func=auc_loss,
                grid_scoring="roc_auc_ovr",
                task_type=task,
                needs_proba=True,
                needs_decision=True,
                higher_is_better=True,
            )

        raise ValueError("Métrica de clasificación no soportada. Usa 'f1', 'accuracy' o 'auc'.")

    raise ValueError(f"Tarea desconocida '{task_type}'. Usa 'regression' o 'classification'.")


# ============================================================
#   Modelos soportados
# ============================================================
ORDERED_MODELS_REGRESSION: tuple[str, ...] = (
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
)

ORDERED_MODELS_CLASSIFICATION: tuple[str, ...] = (
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
)

SUPPORTED_MODELS_REGRESSION: Set[str] = set(ORDERED_MODELS_REGRESSION)
SUPPORTED_MODELS_CLASSIFICATION: Set[str] = set(ORDERED_MODELS_CLASSIFICATION)

SUPPORTED_MODELS: Set[str] = SUPPORTED_MODELS_REGRESSION | SUPPORTED_MODELS_CLASSIFICATION

def get_supported_models(task_type: str) -> Set[str]:
    task = normalize_task_type(task_type)
    if task == "regression":
        return set(SUPPORTED_MODELS_REGRESSION)
    if task == "classification":
        return set(SUPPORTED_MODELS_CLASSIFICATION)
    raise ValueError(f"Tarea desconocida '{task_type}'.")


@dataclass
class ModelConfig:
    name: str
    best_estimator: BaseEstimator
    best_params: Dict[str, object]

    def make_estimator(self) -> BaseEstimator:
        return clone(self.best_estimator)


warnings.filterwarnings("ignore", category=ConvergenceWarning)


class SelectiveStandardScaler(BaseEstimator, TransformerMixin):
    """Estandariza solo las columnas no binarias, deja las binarias intactas y preserva el orden."""

    def __init__(self, non_binary_mask):
        # Guardamos el parámetro tal cual para que sklearn pueda clonar sin que se modifique
        self.non_binary_mask = non_binary_mask
        self.scaler = StandardScaler()
        self._mask_array: Optional[np.ndarray] = None

    def _fit_mask(self, n_features: int) -> np.ndarray:
        mask_arr = np.asarray(self.non_binary_mask, dtype=bool)
        if mask_arr.size < n_features:
            pad = np.ones(n_features - mask_arr.size, dtype=bool)
            mask_arr = np.concatenate([mask_arr, pad])
        elif mask_arr.size > n_features:
            mask_arr = mask_arr[:n_features]
        return mask_arr

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SelectiveStandardScaler":
        X_arr = np.asarray(X)
        self._mask_array = self._fit_mask(X_arr.shape[1])
        if self._mask_array.any():
            self.scaler.fit(X_arr[:, self._mask_array], y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._mask_array is None:
            raise RuntimeError("SelectiveStandardScaler no ha sido ajustado.")
        X_arr = np.asarray(X, dtype=float)
        if self._mask_array.size != X_arr.shape[1]:
            mask_adj = self._fit_mask(X_arr.shape[1])
        else:
            mask_adj = self._mask_array
        if not self._mask_array.any():
            return X_arr
        X_out = X_arr.copy()
        X_out[:, mask_adj] = self.scaler.transform(X_arr[:, mask_adj])
        return X_out


def _detect_binary_columns(X: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Devuelve máscara booleana de columnas binarias (0/1 con tolerancia)."""
    mask = []
    for j in range(X.shape[1]):
        col = X[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            mask.append(False)
            continue
        unique_vals = np.unique(col)
        is_bin = True
        for val in unique_vals:
            if not (abs(val) <= tol or abs(val - 1.0) <= tol):
                is_bin = False
                break
        mask.append(is_bin)
    return np.array(mask, dtype=bool)


def _build_estimator_and_grid(
    model_name: str,
    random_state: int,
    task_type: str,
    n_classes: Optional[int] = None,
) -> Tuple[
    BaseEstimator,
    Union[Dict[str, Sequence[object]], List[Dict[str, Sequence[object]]]],
]:
    task = normalize_task_type(task_type)

    if task == "regression":
        if model_name == "linear_regression":
            return LinearRegression(), {}

        if model_name == "ridge":
            return Ridge(), {
                "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
            }

        if model_name == "lasso":
            return Lasso(), {
                "alpha": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1],
            }

        if model_name == "elastic_net":
            return ElasticNet(), {
                "alpha": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1],
                "l1_ratio": [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
            }

        if model_name == "knn_regressor":
            return KNeighborsRegressor(), {
                "n_neighbors": [2, 3, 4, 5, 7, 11, 15, 21, 31, 51],
                "weights": ["uniform", "distance"],
                "p": [1, 2],
            }

        if model_name == "decision_tree_regressor":
            return DecisionTreeRegressor(random_state=random_state), {
                "max_depth": [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20],
                "min_samples_leaf": [1, 2, 3, 5, 7, 10, 15],
            }

        if model_name == "random_forest_regressor":
            return RandomForestRegressor(
                random_state=random_state,
                n_jobs=-1,
                n_estimators=500,
            ), {
            "max_depth": [3, 5, 7, 10, 12, 15, 20],
            "min_samples_leaf": [1, 2, 3, 5, 7, 10],
            "max_features": ["sqrt", 0.5],
        }

        if model_name == "xgb_regressor":
            if XGBRegressor is None:
                raise ImportError(
                    "El modelo 'xgb_regressor' requiere instalar la libreria xgboost (pip install xgboost)."
                )
            return XGBRegressor(
                objective="reg:squarederror",
                random_state=random_state,
                tree_method="hist",
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                eval_metric="rmse",
            ), {
                "max_depth": [2, 3, 4, 5, 6, 7],
                "min_child_weight": [1, 3, 5],
                "learning_rate": [0.05, 0.1],
                "n_estimators": [300, 600],
                "reg_lambda": [1, 10],
            }

        if model_name == "svr":
            return SVR(), [
                {"kernel": ["linear"], "C": [0.01, 0.1, 1], "epsilon": [0.01, 0.05, 0.1, 0.2]},
                {"kernel": ["rbf"], "C": [0.01, 0.1, 1], "gamma": ["scale", 0.1], "epsilon": [0.01, 0.05, 0.1, 0.2]},
            ]

        if model_name == "mlp_regressor":
            return MLPRegressor(
                random_state=random_state,
                max_iter=1000,
                early_stopping=True,
                n_iter_no_change=3,
                learning_rate="adaptive",
            ), {
                "hidden_layer_sizes": [(32, 16), (64, 32), (128, 64), (128, 64, 32)],
                "alpha": [1e-5, 1e-4, 1e-3],
                "learning_rate_init": [1e-3, 1e-2],
                "activation": ["relu", "tanh"],
            }

    if task == "classification":
        if model_name == "logistic_regression":
            solver = "liblinear"
            if n_classes is not None and n_classes > 2:
                solver = "saga"
            return LogisticRegression(solver=solver), {
                "C": [0.01, 0.1, 1.0, 10.0],
                "class_weight": [None, "balanced"],
                "penalty": ["l1", "l2"],
            }

        if model_name == "gaussian_nb":
            return GaussianNB(), {"var_smoothing": [1e-12, 1e-11, 1e-10, 1e-9, 1e-8]}
        
        if model_name == "lda":
            return LinearDiscriminantAnalysis(), {
                "solver": ["lsqr", "eigen"],
                "shrinkage": ["auto", 0.1, 0.3, 0.5, 0.7, 0.9],
            }

        if model_name == "qda":
            return QuadraticDiscriminantAnalysis(), {
                "reg_param": [0.01, 0.1, 0.25, 0.5, 0.75],
            }

        if model_name == "knn_classifier":
            return KNeighborsClassifier(), {
                "n_neighbors": [2, 3, 4, 5, 7, 11, 15, 21, 31, 51],
                "weights": ["uniform", "distance"],
                "p": [1, 2],
            }

        if model_name == "decision_tree_classifier":
            return DecisionTreeClassifier(random_state=random_state), {
                "max_depth": [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20],
                "min_samples_leaf": [1, 2, 3, 5, 7, 10, 15],
            }

        if model_name == "random_forest_classifier":
            return RandomForestClassifier(
                random_state=random_state,
                n_jobs=-1,
            ), {
            "max_depth": [3, 5, 7, 10, 12, 15, 20],
            "min_samples_leaf": [1, 2, 3, 5, 7, 10],
            "max_features": ["sqrt", 0.5],
        }

        if model_name == "xgb_classifier":
            if XGBClassifier is None:
                raise ImportError(
                    "El modelo 'xgb_classifier' requiere instalar la libreria xgboost (pip install xgboost)."
                )
            if n_classes is None:
                raise ValueError("xgb_classifier requiere n_classes para configurar la salida.")
            n_classes = int(n_classes)
            if n_classes > 2:
                objective = "multi:softprob"
                eval_metric = "mlogloss"
                num_class = n_classes
            else:
                objective = "binary:logistic"
                eval_metric = "logloss"
                num_class = None
            return XGBClassifier(
                objective=objective,
                random_state=random_state,
                tree_method="hist",
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                eval_metric=eval_metric,
                **({"num_class": num_class} if num_class is not None else {}),
            ), {
                "max_depth": [2, 3, 4, 5, 6, 7],
                "min_child_weight": [1, 3, 5],
                "learning_rate": [0.05, 0.1],
                "n_estimators": [300, 600],
                "reg_lambda": [1, 10],
            }

        if model_name == "svc":
            use_proba = bool(n_classes is not None and n_classes > 2)
            return SVC(
                probability=use_proba,
                decision_function_shape="ovr",
            ), [
                {"kernel": ["linear"], "C": [0.01, 0.1, 1]},
                {"kernel": ["rbf"], "C": [0.01, 0.1, 1], "gamma": ["scale", 0.1]},
            ]

        if model_name == "mlp_classifier":
            return MLPClassifier(
                random_state=random_state,
                max_iter=1000,
                early_stopping=True,
                n_iter_no_change=3,
                learning_rate="adaptive",
                ), {
                "hidden_layer_sizes": [(32, 16), (64, 32), (128, 64), (128, 64, 32)],
                "alpha": [1e-5, 1e-4, 1e-3],
                "learning_rate_init": [1e-3, 1e-2],
                "activation": ["relu", "tanh"],
            }

    raise ValueError(
        f"Modelo '{model_name}' no soportado para '{task_type}'. "
        f"Opciones validas: {sorted(get_supported_models(task_type))}"
    )


def tune_models(
    X: np.ndarray,
    y: np.ndarray,
    model_names: Iterable[str],
    *,
    standardize: bool,
    cv_folds: int,
    random_state: int,
    task_type: str = "regression",
    metric: Optional[TaskMetric] = None,
    verbose: bool = False,
) -> List[ModelConfig]:
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    configs: List[ModelConfig] = []
    model_list = list(model_names)
    task = normalize_task_type(task_type)
    metric_obj = metric or get_task_metric(task)
    binary_mask = _detect_binary_columns(X)
    non_binary_mask = ~binary_mask
    n_classes: Optional[int] = None
    if task == "classification":
        n_classes = int(np.unique(y).size)
        if n_classes < 2:
            raise ValueError("Se necesitan al menos 2 clases para clasificacion.")

    for idx, model_name in enumerate(model_list, start=1):
        if verbose:
            print(f"[MODELS] GridSearch {idx}/{len(model_list)} -> {model_name}")
            model_start = time.perf_counter()
        estimator, param_grid = _build_estimator_and_grid(
            model_name,
            random_state,
            task,
            n_classes=n_classes,
        )

        scaler = SelectiveStandardScaler(non_binary_mask) if standardize else "passthrough"

        pipeline = Pipeline([
            ("scaler", scaler),
            ("model", estimator),
        ])

        if isinstance(param_grid, list):
            grid = [{f"model__{k}": v for k, v in item.items()} for item in param_grid]
            grid_keys = sorted({key for item in grid for key in item.keys()})
        else:
            grid = {f"model__{k}": v for k, v in param_grid.items()}
            grid_keys = list(grid.keys())

        if verbose:
            print(f"[MODELS] GridSearch '{model_name}' | grid keys: {grid_keys}")

        if grid:
            if task == "classification":
                cv_split = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            else:
                cv_split = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=grid,
                scoring=metric_obj.grid_scoring,
                cv=cv_split,
                n_jobs=GRIDSEARCH_N_JOBS,
                refit=True,
                verbose=0,
            )
            search.fit(X, y)
            best_estimator = search.best_estimator_
            best_params = search.best_params_
            if verbose:
                score_display = float(search.best_score_)
                if metric_obj.grid_scoring.startswith("neg_"):
                    score_display = -score_display
                print(
                    f"[MODELS]   -> Mejor params {model_name}: "
                    f"{best_params} | score={score_display:.6f}"
                )
        else:
            pipeline.fit(X, y)
            best_estimator = pipeline
            best_params = {}
            if verbose:
                print(f"[MODELS]   -> Sin grid: usando configuracion por defecto.")

        configs.append(ModelConfig(
            name=model_name,
            best_estimator=best_estimator,
            best_params=best_params,
        ))
        if verbose:
            elapsed = time.perf_counter() - model_start
            print(f"[MODELS] GridSearch completado -> {model_name} ({elapsed:.2f}s)")

    if verbose:
        print()

    return configs


def evaluate_models_cv(
    X: np.ndarray,
    y: np.ndarray,
    cv: KFold,
    model_configs: Sequence[ModelConfig],
    metric: TaskMetric,
) -> Tuple[float, Dict[str, float]]:
    if not model_configs:
        raise ValueError("No hay modelos configurados para la evaluacion.")

    X = np.asarray(X)
    y = np.asarray(y).ravel()

    splits = list(cv.split(X, y))
    per_model: Dict[str, float] = {}

    for config in model_configs:
        fold_losses: List[float] = []
        for train_idx, test_idx in splits:
            est = config.make_estimator()
            est.fit(X[train_idx], y[train_idx])
            fold_losses.append(metric.evaluate_estimator(est, X[test_idx], y[test_idx]))
        per_model[config.name] = float(np.mean(fold_losses))

    mean_loss = float(np.mean(list(per_model.values())))
    return mean_loss, per_model
