import copy
import math
import os
import re
import time
import warnings
import numpy as np
from typing import Any, Dict, Iterable, List, Optional, Set

from sklearn.model_selection import KFold

from deap import gp as deap_gp
from evopt.gp_feature_generation.gp_feature_generation import GPFeatureGenerator, OPERATOR_CATALOG
from evopt.ga_feature_selection.ga_feature_selection import GAFeatureSelector
from evopt.model_utils import (
    ModelConfig,
    TaskMetric,
    evaluate_models_cv,
    get_supported_models,
    get_task_metric,
    normalize_task_type,
    tune_models,
)

# Silence joblib loky resource_tracker cleanup warnings (Windows temp files).
_WARN_FILTERS = (
    "ignore:resource_tracker:UserWarning:joblib.externals.loky.backend.resource_tracker",
    "ignore:resource_tracker:UserWarning:loky.backend.resource_tracker",
)
_existing_filters = [f for f in os.environ.get("PYTHONWARNINGS", "").split(",") if f]
for _flt in _WARN_FILTERS:
    if _flt not in _existing_filters:
        _existing_filters.append(_flt)
if _existing_filters:
    os.environ["PYTHONWARNINGS"] = ",".join(_existing_filters)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"joblib\.externals\.loky\.backend\.resource_tracker",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"loky\.backend\.resource_tracker",
)


class EvolutionaryOptimizer:
    """
    Orquestador evolutivo:
      - GP genera nuevas features.
      - GA selecciona subconjuntos.
      - Siempre se generan GP a partir de todas las originales (y GP retenidas previas).
      - La salida final es la SELECCIÓN de la ÚLTIMA iteración de GA completada.
    """

    DEFAULT_MODELS_BY_TASK = {
        "regression": ("elastic_net", "decision_tree_regressor"),
        "classification": ("logistic_regression", "decision_tree_classifier"),
    }

    def __init__(
        # ---------- Comunes globales ----------
        self,
        maxtime: int = 3600,
        num_iterations: int = 100,
        random_state: int = 0,
        alter_random_state: bool = True,
        verbose: bool = False,
        task_type: str = "regression",
        metric_name: Optional[str] = None,
        models: Optional[List[str]] = None,
        standardize: bool = True,
        model_tuning_cv_folds: Optional[int] = 5,
        split_data_between_gp_and_ga: bool = False,
        split_data_between_gp_and_ga_and_final_ga: bool = True,
        estimate_pop_size: bool = True,
        estimate_pop_seconds: int = 25,

        # ---------- GP (comunes) ----------
        gp_population_size: Optional[int] = None,
        gp_offspring_size: Optional[float] = 0.95,
        gp_elitism_size: Optional[float] = 0.05,
        gp_tournament_size: int = 3,
        gp_init_population_method: str = "grow",

        gp_num_generations: int = 10,
        gp_maxtime: Optional[int] = 300,
        gp_patience: Optional[int] = 2,
        gp_cv_folds: int = 3,

        # ---------- GP (resto) ----------
        gp_num_new_features: Optional[int] = None,
        gp_min_tree_height: int = 1,
        gp_max_tree_height: int = 2,
        gp_unary_continuous_operators: List[str] = ("sqrt", "square", "inv", "log"),
        gp_unary_binary_operators: List[str] = (),
        gp_binary_continuous_operators: List[str] = ("add", "sub", "mul", "div", "max", "min"),
        gp_binary_binary_operators: List[str] = ("max", "min"),
        gp_binary_mixed_operators: List[str] = ("mul",),
        gp_root_excluded_operator_set: List[str] = ("add", "sub"),
        gp_ephemeral_constants: Optional[List[float]] = [-3, -2, -1.5, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5, 2, 3],
        gp_ephemeral_constants_range: Optional[List[float]] = None,
        gp_crossover_probability: float = 1.0,
        gp_mutation_probability: float = 0.25,
        gp_mutation_weights: tuple = (0.25, 0.25, 0.50),
        gp_evolve_mutation_weights: bool = True,
        gp_mutation_weights_end: tuple = (0.45, 0.45, 0.10),
        gp_evolve_mutation_probability: bool = True,
        gp_mutation_probability_end: float = 0.05,
        gp_duplicate_corr_threshold: float = 0.7,
        gp_penalty_mode: Optional[str] = None,
        gp_penalty_coeff: float = 0.0,

        # ---------- GA (comunes) ----------
        ga_population_size: Optional[int] = None,
        ga_offspring_size: Optional[float] = 0.95,
        ga_elitism_size: Optional[float] = 0.05,
        ga_tournament_size: int = 3,

        ga_num_generations: int = 10,
        ga_maxtime: Optional[int] = 300,
        ga_patience: Optional[int] = 2,
        ga_cv_folds: int = 3,

        # ---------- GA (resto) ----------
        ga_individual_all_features: bool = True,
        ga_initial_bit_prob: float = 0.5,
        ga_mutation_probability: float = 0.25,
        ga_evolve_mutation_probability: bool = True,
        ga_mutation_probability_end: float = 0.05,
        ga_crossover_type: str = "single",

        # ---------- GA final ----------
        run_final_ga: bool = True,
        final_ga_maxtime: Optional[int] = 600,
        final_ga_population_size: Optional[int] = None,
        final_ga_num_generations: int = 10,
        final_ga_patience: int = 5,
        final_ga_retune_models: bool = False,
        sticky_selected_features: bool = False,
    ) -> None:

        # General
        self.maxtime = maxtime
        self.num_iterations = num_iterations
        self.random_state = int(random_state)
        self.alter_random_state = bool(alter_random_state)
        self.verbose = bool(verbose)
        self.task_type = normalize_task_type(task_type)
        self.metric: TaskMetric = get_task_metric(self.task_type, metric_name)

        if models is None:
            defaults = self.DEFAULT_MODELS_BY_TASK.get(self.task_type)
            if defaults is None:
                raise ValueError(f"No hay modelos por defecto definidos para '{self.task_type}'.")
            self.model_names = list(defaults)
        else:
            self.model_names = list(models)

        supported = get_supported_models(self.task_type)
        invalid = [m for m in self.model_names if m not in supported]
        if invalid:
            raise ValueError(f"Modelos no soportados para {self.task_type}: {invalid}. Soportados: {sorted(supported)}")

        self.standardize = bool(standardize)
        self.model_tuning_cv_folds = model_tuning_cv_folds
        self.split_data_between_gp_and_ga = bool(split_data_between_gp_and_ga)
        self.split_data_between_gp_and_ga_and_final_ga = bool(split_data_between_gp_and_ga_and_final_ga)
        self.estimate_pop_size = bool(estimate_pop_size)
        self.estimate_pop_seconds = int(estimate_pop_seconds)

        # GP params
        self.gp_population_size = gp_population_size
        self.gp_offspring_size = gp_offspring_size
        self.gp_elitism_size = gp_elitism_size
        self.gp_tournament_size = gp_tournament_size
        self.gp_init_population_method = gp_init_population_method

        self.gp_num_generations = gp_num_generations
        self.gp_maxtime = gp_maxtime
        self.gp_patience = gp_patience
        self.gp_cv_folds = gp_cv_folds

        self._gp_num_new_features = (
            None if gp_num_new_features is None else max(1, int(gp_num_new_features))
        )
        self.gp_min_tree_height = gp_min_tree_height
        self.gp_max_tree_height = gp_max_tree_height
        self.gp_unary_continuous_operators = list(gp_unary_continuous_operators)
        self.gp_unary_binary_operators = list(gp_unary_binary_operators)
        self.gp_binary_continuous_operators = list(gp_binary_continuous_operators)
        self.gp_binary_binary_operators = list(gp_binary_binary_operators)
        self.gp_binary_mixed_operators = list(gp_binary_mixed_operators)
        self.gp_root_excluded_operator_set = list(gp_root_excluded_operator_set or [])
        self.gp_ephemeral_constants = list(gp_ephemeral_constants) if gp_ephemeral_constants is not None else []
        self.gp_ephemeral_constants_range = gp_ephemeral_constants_range
        self.gp_crossover_probability = gp_crossover_probability
        self.gp_mutation_probability = gp_mutation_probability
        self.gp_mutation_weights = gp_mutation_weights
        self.gp_evolve_mutation_weights = gp_evolve_mutation_weights
        self.gp_mutation_weights_end = gp_mutation_weights_end
        self.gp_evolve_mutation_probability = gp_evolve_mutation_probability
        self.gp_mutation_probability_end = gp_mutation_probability_end
        self.gp_duplicate_corr_threshold = float(gp_duplicate_corr_threshold)
        if not (0.0 <= self.gp_duplicate_corr_threshold <= 1.0):
            raise ValueError("gp_duplicate_corr_threshold debe estar en [0, 1].")
        self.gp_penalty_mode = gp_penalty_mode
        self.gp_penalty_coeff = float(gp_penalty_coeff)

        # GA params
        self.ga_population_size = ga_population_size
        self.ga_offspring_size = ga_offspring_size
        self.ga_elitism_size = ga_elitism_size
        self.ga_tournament_size = ga_tournament_size

        self.ga_num_generations = ga_num_generations
        self.ga_maxtime = ga_maxtime
        self.ga_patience = ga_patience
        self.ga_cv_folds = ga_cv_folds

        self.ga_individual_all_features = ga_individual_all_features
        if not (0.0 <= ga_initial_bit_prob <= 1.0):
            raise ValueError("[GA] ga_initial_bit_prob debe estar en [0,1].")
        self.ga_initial_bit_prob = float(ga_initial_bit_prob)
        self.ga_mutation_probability = float(ga_mutation_probability)
        self.ga_evolve_mutation_probability = bool(ga_evolve_mutation_probability)
        self.ga_mutation_probability_end = float(ga_mutation_probability_end)
        self.ga_crossover_type = str(ga_crossover_type).lower()

        # Final GA params
        self.run_final_ga = bool(run_final_ga)
        self.final_ga_maxtime = final_ga_maxtime
        self.final_ga_population_size = final_ga_population_size
        self.final_ga_num_generations = final_ga_num_generations
        self.final_ga_patience = final_ga_patience
        self.final_ga_retune_models = bool(final_ga_retune_models)
        self.sticky_selected_features = bool(sticky_selected_features)

        # RNG
        self.rng = np.random.default_rng(self.random_state)

        # Estado/resultados
        self.history_: List[Dict[str, Any]] = []
        self.kept_gp_expressions_: List[str] = []
        self.kept_gp_matrix_: Optional[np.ndarray] = None
        self.kept_gp_trees_: List[deap_gp.PrimitiveTree] = []

        # Resultado final
        self.final_X_: Optional[np.ndarray] = None
        self.final_y_: Optional[np.ndarray] = None
        self.final_best_score_: Optional[float] = None
        self.final_selected_original_idx_: List[int] = []
        self.final_prev_context_exprs_: List[str] = []
        self.final_prev_context_trees_: List[deap_gp.PrimitiveTree] = []
        self.final_prev_context_exprs_inline_: List[str] = []
        self.final_gp_expressions_: List[str] = []
        self.final_gp_expression_trees_: List[deap_gp.PrimitiveTree] = []
        self.final_gp_expressions_display_: List[str] = []
        self.final_prev_context_exprs_eval_: List[str] = []
        self.final_gp_expressions_eval_: List[str] = []
        self.final_model_mses_: Dict[str, float] = {}
        self.final_iteration_: Optional[int] = None

        # Metadatos
        self._p_: Optional[int] = None
        self._last_X_shape_: Optional[tuple] = None
        self._feature_names: Optional[List[str]] = None
        self._original_binary_mask: Optional[List[bool]] = None
        self._model_configs: List[ModelConfig] = []
        self._X_fit_: Optional[np.ndarray] = None
        self._y_fit_: Optional[np.ndarray] = None
        self._final_ga_reserved_time: int = 0
        self._final_ga_started: bool = False
        self._ga_pop_estimated: bool = False
        self._ga_pop_estimate: Optional[int] = None

    # --------------------------------------------------------------
    def _prepare_models(self, X: np.ndarray, y: np.ndarray) -> None:
        if not self.model_names:
            raise ValueError("Debes especificar al menos un modelo para la evaluación.")

        cv_folds = self.model_tuning_cv_folds or self.ga_cv_folds
        if cv_folds < 2:
            raise ValueError("model_tuning_cv_folds debe ser >=2.")

        if self.verbose:
            print("[EVOPT] Preparando modelos base y ajuste de hiperparámetros...\n")

        self._model_configs = tune_models(
            X=X,
            y=y,
            model_names=self.model_names,
            standardize=self.standardize,
            cv_folds=cv_folds,
            random_state=self.random_state,
            task_type=self.task_type,
            metric=self.metric,
            verbose=self.verbose,
        )

        if self.verbose:
            print("[EVOPT] Modelos preparados:", [cfg.name for cfg in self._model_configs])
            print()

    def _remaining_time(self, start_time: float) -> float:
        if self.maxtime is None:
            return float("inf")
        return self.maxtime - (time.time() - start_time)

    def _remaining_time_for_iterations(self, start_time: float) -> float:
        rem = self._remaining_time(start_time)
        if self.run_final_ga and not self._final_ga_started and self.maxtime is not None:
            rem -= self._final_ga_reserved_time
        return rem

    @staticmethod
    def _auto_population_size(n_rows: int, n_features: int) -> int:
        denom = max(1, int(n_rows) * max(1, int(n_features)))
        size = int(round(10_000_000 / denom))
        return max(2, size)

    def _estimate_population_size(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int,
        *,
        label: str,
    ) -> int:
        budget = max(1.0, float(self.estimate_pop_seconds))
        if X.size == 0 or y.size == 0:
            return 2
        n_samples = int(X.shape[0])
        if n_samples < 2:
            return 2
        folds = max(2, int(cv_folds))
        if n_samples < folds:
            folds = n_samples
        if folds < 2:
            return 2

        if self.verbose:
            print(f"[EVOPT] Estimando poblacion ({budget:.0f}s)...")

        cv = KFold(n_splits=folds, shuffle=True, random_state=self.random_state)
        start = time.perf_counter()
        runs = 0
        elapsed = 0.0
        while True:
            try:
                evaluate_models_cv(X, y, cv, self._model_configs, self.metric)
            except Exception:
                pass
            runs += 1
            elapsed = time.perf_counter() - start
            if elapsed >= budget:
                break

        if elapsed <= 0:
            return 2
        estimate = int(round(runs * (budget / elapsed)))
        estimate = max(2, estimate)
        if self.verbose:
            print(
                f"[EVOPT] Estimacion poblacion {label}: {estimate} "
                f"(runs={runs}, elapsed={elapsed:.2f}s, budget={budget:.0f}s)"
            )
        return estimate

    def _build_pset_for_nvars(self, n_vars: int) -> deap_gp.PrimitiveSet:
        pset = deap_gp.PrimitiveSet("MAIN", n_vars)
        added: Set[str] = set()

        def add_op(name: str, alias: Optional[str] = None) -> None:
            prim_name = alias or name
            if prim_name in added:
                return
            if name not in OPERATOR_CATALOG:
                return
            arity, func = OPERATOR_CATALOG[name]
            pset.addPrimitive(func, arity, name=prim_name)
            added.add(prim_name)

        def register_all(ops: Iterable[str], suffixes: Iterable[str]) -> None:
            for op in ops:
                add_op(op)
                for suf in suffixes:
                    add_op(op, f"{op}__{suf}")

        register_all(self.gp_unary_continuous_operators, ("uc",))
        register_all(self.gp_unary_binary_operators, ("ub",))
        register_all(self.gp_binary_continuous_operators, ("cc",))
        register_all(self.gp_binary_binary_operators, ("bb",))
        register_all(self.gp_binary_mixed_operators, ("cb", "bc"))
        add_op("id_bin")

        pset.renameArguments(**{f"ARG{i}": f"x{i}" for i in range(n_vars)})
        return pset

    @staticmethod
    def _detect_binary_columns(X: np.ndarray, tol: float = 1e-6) -> List[bool]:
        mask: List[bool] = []
        for j in range(X.shape[1]):
            col = X[:, j]
            col = col[np.isfinite(col)]
            if col.size == 0:
                mask.append(False)
                continue
            unique_vals = np.unique(col)
            is_binary = True
            for val in unique_vals:
                if not (abs(val) <= tol or abs(val - 1.0) <= tol):
                    is_binary = False
                    break
            mask.append(is_binary)
        return mask

    def _resolve_gp_num_new_features(self, num_input_features: int) -> int:
        if self._gp_num_new_features is not None:
            base = int(self._gp_num_new_features)
        elif num_input_features <= 0:
            base = 1
        else:
            base = max(1, int(round(math.sqrt(num_input_features))))

        if self.gp_population_size is not None:
            try:
                pop_size = int(self.gp_population_size)
            except (TypeError, ValueError):
                pop_size = None
            if pop_size is not None and pop_size > 0:
                if base > pop_size and self.verbose:
                    print(
                        "[EVOPT] Ajustando gp_num_new_features "
                        f"{base} -> {pop_size} (por tamaño de población)"
                    )
                base = min(base, pop_size)

        return max(1, base)

    def _build_gp(
        self,
        gp_arg_names: List[str],
        gp_arg_is_binary: List[bool],
        num_new_features: int,
    ) -> GPFeatureGenerator:
        return GPFeatureGenerator(
            population_size=self.gp_population_size,
            offspring_size=self.gp_offspring_size,
            elitism_size=self.gp_elitism_size,
            tournament_size=self.gp_tournament_size,
            init_population_method=self.gp_init_population_method,

            num_generations=self.gp_num_generations,
            maxtime=self.gp_maxtime,
            patience=self.gp_patience,
            cv_folds=self.gp_cv_folds,

            num_new_features=num_new_features,
            min_tree_height=self.gp_min_tree_height,
            max_tree_height=self.gp_max_tree_height,
            unary_continuous_operators=self.gp_unary_continuous_operators,
            unary_binary_operators=self.gp_unary_binary_operators,
            binary_continuous_operators=self.gp_binary_continuous_operators,
            binary_binary_operators=self.gp_binary_binary_operators,
            binary_mixed_operators=self.gp_binary_mixed_operators,
            root_excluded_operator_set=self.gp_root_excluded_operator_set,
            ephemeral_constants=self.gp_ephemeral_constants,
            ephemeral_constants_range=self.gp_ephemeral_constants_range,

            crossover_probability=self.gp_crossover_probability,
            mutation_probability=self.gp_mutation_probability,
            mutation_weights=self.gp_mutation_weights,
            evolve_mutation_weights=self.gp_evolve_mutation_weights,
            mutation_weights_end=self.gp_mutation_weights_end,
            evolve_mutation_probability=self.gp_evolve_mutation_probability,
            mutation_probability_end=self.gp_mutation_probability_end,
            duplicate_correlation_threshold=self.gp_duplicate_corr_threshold,
            penalty_mode=self.gp_penalty_mode,
            penalty_coeff=self.gp_penalty_coeff,

            random_state=self.random_state,
            alter_random_state=self.alter_random_state,
            rng=np.random.default_rng(self.random_state),
            verbose=self.verbose,

            feature_names=gp_arg_names,
            feature_is_binary=gp_arg_is_binary,
            model_configs=self._model_configs,
            metric=self.metric,
        )

    def _build_ga(self, overrides: Optional[Dict[str, Any]] = None) -> GAFeatureSelector:
        params: Dict[str, Any] = {
            "population_size": self.ga_population_size,
            "offspring_size": self.ga_offspring_size,
            "elitism_size": self.ga_elitism_size,
            "tournament_size": self.ga_tournament_size,
            "num_generations": self.ga_num_generations,
            "maxtime": self.ga_maxtime,
            "patience": self.ga_patience,
            "cv_folds": self.ga_cv_folds,
            "individual_all_features": self.ga_individual_all_features,
            "initial_bit_prob": self.ga_initial_bit_prob,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "mutation_probability": self.ga_mutation_probability,
            "evolve_mutation_probability": self.ga_evolve_mutation_probability,
            "mutation_probability_end": self.ga_mutation_probability_end,
            "crossover_type": self.ga_crossover_type,
            "model_configs": self._model_configs,
            "metric": self.metric,
            "alter_random_state": self.alter_random_state,
        }
        if overrides:
            for key, value in overrides.items():
                if key == "model_configs":
                    continue
                if value is not None:
                    params[key] = value
        return GAFeatureSelector(**params)

    # --------------------------------------------------------------

    @staticmethod
    def _strip_id_bin_expr(expr: str) -> str:
        """Elimina id_bin(...) solo para visualizacion."""
        if "id_bin(" not in expr:
            return expr
        out: List[str] = []
        i = 0
        n = len(expr)
        while i < n:
            if expr.startswith("id_bin(", i) and (i == 0 or not (expr[i - 1].isalnum() or expr[i - 1] == "_")):
                i += len("id_bin(")
                depth = 1
                content: List[str] = []
                while i < n:
                    ch = expr[i]
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    content.append(ch)
                    i += 1
                inner = EvolutionaryOptimizer._strip_id_bin_expr("".join(content))
                out.append(inner)
                if i < n and expr[i] == ")":
                    i += 1
                continue
            out.append(expr[i])
            i += 1
        return "".join(out)

    @staticmethod
    def _pretty_expr(expr: str, arg_names: List[str]) -> str:
        def repl(m):
            idx = int(m.group(1))
            if 0 <= idx < len(arg_names):
                return arg_names[idx]
            return f"x{idx}"
        out = re.sub(r"\bx(\d+)\b", repl, expr)
        return EvolutionaryOptimizer._strip_id_bin_expr(out)

    def _disp(self, value: float) -> float:
        if hasattr(self.metric, "display_value"):
            return float(self.metric.display_value(value))
        return float(value)

    def _inline_expression(
        self,
        expr: str,
        base_names: List[str],
        context_inlined: List[str],
        *,
        strip_id_bin: bool = False,
    ) -> str:
        pattern = re.compile(r"\bx(\d+)\b")
        base_count = len(base_names)

        def repl(match):
            idx = int(match.group(1))
            if idx < base_count:
                return base_names[idx]
            ctx_idx = idx - base_count
            if 0 <= ctx_idx < len(context_inlined):
                return context_inlined[ctx_idx]
            return f"x{idx}"

        out = pattern.sub(repl, expr)
        if strip_id_bin:
            return self._strip_id_bin_expr(out)
        return out

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        verbose: Optional[bool] = None,
    ) -> "EvolutionaryOptimizer":
        """
        - Mantiene originales para GP.
        - Usa la ÚLTIMA selección del GA que se complete.
        - Incrementa self.random_state al final de cada iteración si alter_random_state=True.
        - verbose: si se indica, sobreescribe el modo verbose solo para esta ejecución de fit().
        """
        start_time = time.time()

        if verbose is not None:
            self.verbose = bool(verbose)

        X = np.asarray(X); y = np.asarray(y).ravel()
        n, p = X.shape
        self._p_ = p
        self._feature_names = list(feature_names) if feature_names is not None else [f"x{i}" for i in range(p)]
        self._X_fit_ = X
        self._y_fit_ = y
        self._original_binary_mask = self._detect_binary_columns(X)

        X_orig = X.copy()

        base_x_names = [f"x{i}" for i in range(p)]

        self._prepare_models(X_orig, y)
        self._final_ga_started = False
        if self.run_final_ga and self.maxtime is not None:
            if self.final_ga_maxtime is not None:
                reserved_time = max(0.0, float(self.final_ga_maxtime))
            else:
                reserved_time = max(1.0, float(self.maxtime) * 0.1)
            reserved_time = min(reserved_time, float(self.maxtime))
            if reserved_time <= 0.0 and self.maxtime > 0:
                reserved_time = min(float(self.maxtime), 1.0)
            self._final_ga_reserved_time = max(0, int(round(reserved_time)))
        else:
            self._final_ga_reserved_time = 0

        idx_final_ga = np.arange(n)
        if self.split_data_between_gp_and_ga_and_final_ga and n >= 3:
            rng_split = np.random.default_rng(self.random_state)
            indices = np.arange(n)
            rng_split.shuffle(indices)
            cut1 = int(round(n * 0.4))
            cut1 = max(1, min(cut1, n - 2))
            cut2_raw = int(round(n * 0.8))
            cut2 = max(cut1 + 1, min(cut2_raw, n - 1))
            idx_gp = np.sort(indices[:cut1])
            idx_ga = np.sort(indices[cut1:cut2])
            idx_final_ga = np.sort(indices[cut2:]) if cut2 < n else np.empty(0, dtype=int)
            if idx_final_ga.size == 0:
                idx_final_ga = np.sort(indices[-1:])
            if idx_ga.size == 0:
                idx_ga = np.sort(indices[cut1:cut1 + max(1, n - cut1 - idx_final_ga.size)])
            if idx_ga.size == 0:
                idx_ga = np.sort(indices[:max(1, n // 2)])
        elif self.split_data_between_gp_and_ga:
            rng_split = np.random.default_rng(self.random_state)
            indices = np.arange(n)
            rng_split.shuffle(indices)
            split_point = max(1, n // 2)
            idx_gp = np.sort(indices[:split_point])
            idx_ga = np.sort(indices[split_point:])
            if idx_gp.size == 0:
                idx_gp = np.arange(n)
            if idx_ga.size == 0:
                idx_ga = idx_gp
            idx_final_ga = np.arange(n)
        else:
            idx_gp = np.arange(n)
            idx_ga = np.arange(n)
            idx_final_ga = np.arange(n)

        X_gp_base = X_orig[idx_gp]
        y_gp_base = y[idx_gp]
        X_ga_base = X_orig[idx_ga]
        y_ga_base = y[idx_ga]

        Z_kept_gp = np.empty((X_gp_base.shape[0], 0))
        Z_kept_ga = np.empty((X_ga_base.shape[0], 0))
        Z_kept_full = np.empty((n, 0))
        self._ga_pop_estimated = False
        self._ga_pop_estimate = None

        if self.estimate_pop_size:
            if self.verbose and self.gp_population_size is not None:
                print("[EVOPT] estimate_pop_size=True -> ignorando gp_population_size fijada.")
            if self.verbose and self.ga_population_size is not None:
                print("[EVOPT] estimate_pop_size=True -> ignorando ga_population_size fijada.")
            estimated_pop = self._estimate_population_size(
                X_gp_base, y_gp_base, self.gp_cv_folds, label="GP/GA"
            )
            self.gp_population_size = estimated_pop
            self.ga_population_size = estimated_pop
            gp_pop_note = "estimada"
            ga_pop_note = "estimada"
            self._ga_pop_estimated = True
            self._ga_pop_estimate = estimated_pop
        elif self.gp_population_size is None:
            self.gp_population_size = self._auto_population_size(X_gp_base.shape[0], X_gp_base.shape[1])
            gp_pop_note = "auto"
        else:
            self.gp_population_size = int(self.gp_population_size)
            gp_pop_note = "fijada"

        if not self.estimate_pop_size and self.ga_population_size is None:
            self.ga_population_size = self._auto_population_size(X_ga_base.shape[0], X_ga_base.shape[1])
            ga_pop_note = "auto"
        elif not self.estimate_pop_size:
            self.ga_population_size = int(self.ga_population_size)
            ga_pop_note = "fijada"

        orig_feature_count = len(self._feature_names) if self._feature_names is not None else p
        num_new_features = self._resolve_gp_num_new_features(orig_feature_count)

        kept_exprs: List[str] = []
        kept_exprs_inlined: List[str] = []
        kept_exprs_eval: List[str] = []
        kept_expr_trees: List[deap_gp.PrimitiveTree] = []
        sticky_selected_exprs: Set[str] = set()

        if self.verbose:
            print(f"[EVOPT] ===== INICIO =====")
            print(f"[EVOPT] X original: {X_orig.shape} | Iteraciones: {self.num_iterations}")
            if self.split_data_between_gp_and_ga:
                print(f"[EVOPT] Subconjuntos -> GP: {X_gp_base.shape} | GA: {X_ga_base.shape}")
            if self.maxtime is not None:
                print(f"[EVOPT] Límite de tiempo GLOBAL: {self.maxtime} s\n")
            print(
                f"[EVOPT] Población GP: {self.gp_population_size} "
                f"(n={X_gp_base.shape[0]}, p={X_gp_base.shape[1]}, {gp_pop_note})"
            )
            print(
                f"[EVOPT] Población GA: {self.ga_population_size} "
                f"(n={X_ga_base.shape[0]}, p={X_ga_base.shape[1]}, {ga_pop_note})"
            )
            if self.run_final_ga and self.estimate_pop_size and self._ga_pop_estimated and self._ga_pop_estimate is not None:
                final_pop_est = max(2, int(self._ga_pop_estimate) * 4)
                print(f"[EVOPT] Población GA ampliada: {final_pop_est} (estimada x4)")
            print()

        did_any_ga = False

        for it in range(1, self.num_iterations + 1):
            if self.maxtime is not None and self._remaining_time_for_iterations(start_time) <= 0:
                if self.verbose:
                    print(f"[EVOPT] Parada por tiempo global ANTES de iteración {it}.")
                break

            remaining_total = self._remaining_time(start_time)
            expected_gp_time = float(self.gp_maxtime) if self.gp_maxtime is not None else 0.0
            if remaining_total < (self._final_ga_reserved_time + expected_gp_time):
                if self.verbose:
                    print(f"[EVOPT] Tiempo restante ({remaining_total:.2f}s) insuficiente para GP+GA antes del GA final.")
                break

            elapsed_after_gp = None
            elapsed_after_ga = None

            ctx_names = [f"gprev{j}" for j in range(Z_kept_gp.shape[1])]
            gp_arg_names = self._feature_names + ctx_names
            base_mask = list(self._original_binary_mask or [False] * len(self._feature_names))
            if len(base_mask) != len(self._feature_names):
                base_mask = [False] * len(self._feature_names)
            gp_arg_is_binary = base_mask + [False] * len(ctx_names)

            X_gp_input = np.hstack([X_gp_base, Z_kept_gp]) if Z_kept_gp.shape[1] > 0 else X_gp_base
            X_ga_input = np.hstack([X_ga_base, Z_kept_ga]) if Z_kept_ga.shape[1] > 0 else X_ga_base
            X_full_input = np.hstack([X_orig, Z_kept_full]) if Z_kept_full.shape[1] > 0 else X_orig

            if self.verbose:
                print(f"[EVOPT][Iter {it}] Entrenando GP sobre shape={X_gp_input.shape}\n")

            gp_fg = self._build_gp(gp_arg_names, gp_arg_is_binary, num_new_features)
            if self.maxtime is not None and self.gp_maxtime is not None:
                rest = max(0, int(self._remaining_time_for_iterations(start_time)))
                gp_fg.maxtime = min(self.gp_maxtime, max(0, rest - 1))
                if gp_fg.maxtime == 0:
                    if self.verbose:
                        print(f"[EVOPT][Iter {it}] Sin tiempo suficiente para GP. Parando.")
                    break

            gp_fg.fit(X_gp_input, y_gp_base)
            Z_new_gp = gp_fg.transform(X_gp_input)
            expr_new = gp_fg.best_expressions()
            expr_new_trees = gp_fg.best_expression_trees()
            k_new = int(Z_new_gp.shape[1])
            expr_new_inlined = [
                self._inline_expression(
                    expr,
                    self._feature_names,
                    kept_exprs_inlined,
                    strip_id_bin=True,
                )
                for expr in expr_new
            ]
            expr_new_eval = [
                self._inline_expression(expr, base_x_names, kept_exprs_eval)
                for expr in expr_new
            ]

            if self.verbose:
                print(f"\n[EVOPT][Iter {it}] GP generó {k_new} nuevas features.\n")

            elapsed_after_gp = time.time() - start_time
            if self.verbose:
                if self.maxtime is not None:
                    print(f"[EVOPT][Iter {it}] Tiempo transcurrido tras GP: {elapsed_after_gp:.2f}s de {self.maxtime}s\n")
                else:
                    print(f"[EVOPT][Iter {it}] Tiempo transcurrido tras GP: {elapsed_after_gp:.2f}s\n")

            if self.maxtime is not None and self._remaining_time_for_iterations(start_time) <= 0:
                if self.verbose:
                    print(f"[EVOPT][Iter {it}] Parada por tiempo global TRAS GP, antes de GA.")
                break
            if self.maxtime is not None:
                remaining_after_gp = self._remaining_time(start_time)
                if remaining_after_gp <= self._final_ga_reserved_time:
                    if self.verbose:
                        print(f"[EVOPT][Iter {it}] Reservando tiempo para GA final: quedan {remaining_after_gp:.2f}s.")
                    break

            eval_on_start = time.perf_counter()
            if k_new > 0:
                Z_new_ga = gp_fg.evaluate_on(X_ga_input)
                Z_new_full = gp_fg.evaluate_on(X_full_input)
            else:
                Z_new_ga = np.empty((X_ga_input.shape[0], 0))
                Z_new_full = np.empty((X_full_input.shape[0], 0))
            eval_on_elapsed = time.perf_counter() - eval_on_start

            X_ga_for_ga = np.hstack([X_ga_input, Z_new_ga]) if k_new > 0 else X_ga_input
            X_full_for_ga = np.hstack([X_full_input, Z_new_full]) if k_new > 0 else X_full_input

            ga_feature_names = (
                self._feature_names
                + [f"gprev{j}" for j in range(Z_kept_ga.shape[1])]
                + ([f"gnew{j}" for j in range(k_new)] if k_new > 0 else [])
            )

            if self.verbose:
                print(f"[EVOPT][Iter {it}] Ejecutando GA sobre shape={X_ga_for_ga.shape}\n")

            ga = self._build_ga()
            ga_start = time.perf_counter()
            ga.fit(X_ga_for_ga, y_ga_base, feature_names=ga_feature_names)
            ga_elapsed = time.perf_counter() - ga_start
            did_any_ga = True

            mask_best = ga.best_individual_.astype(int)
            score_best = float(ga.best_score_)
            model_mses_best = ga.best_model_mses()

            selected_idx_all = np.where(mask_best == 1)[0]

            n_orig = p
            n_prev = Z_kept_gp.shape[1]
            n_new = k_new
            start_prev = n_orig
            start_new = n_orig + n_prev

            selected_original_idx = [i for i in range(n_orig) if mask_best[i] == 1]
            selected_prev_gp_idx = [j for j in range(n_prev) if mask_best[start_prev + j] == 1]
            selected_new_gp_idx = [j for j in range(n_new) if n_new and mask_best[start_new + j] == 1]

            final_gp_exprs_iter = [kept_exprs[j] for j in selected_prev_gp_idx] + (
                [expr_new[j] for j in selected_new_gp_idx] if n_new else []
            )
            final_gp_exprs_inlined_iter = [kept_exprs_inlined[j] for j in selected_prev_gp_idx] + (
                [expr_new_inlined[j] for j in selected_new_gp_idx] if n_new else []
            )
            final_gp_exprs_eval_iter = [kept_exprs_eval[j] for j in selected_prev_gp_idx] + (
                [expr_new_eval[j] for j in selected_new_gp_idx] if n_new else []
            )
            final_gp_trees_iter = [kept_expr_trees[j] for j in selected_prev_gp_idx] + (
                [expr_new_trees[j] for j in selected_new_gp_idx] if n_new else []
            )
            self.final_X_ = X_full_for_ga[:, selected_idx_all].copy()
            self.final_y_ = y.copy()
            self.final_best_score_ = score_best
            self.final_model_mses_ = dict(model_mses_best)
            self.final_selected_original_idx_ = selected_original_idx
            self.final_gp_expressions_ = list(final_gp_exprs_iter)
            self.final_gp_expressions_display_ = list(final_gp_exprs_inlined_iter)
            self.final_gp_expressions_eval_ = list(final_gp_exprs_eval_iter)
            self.final_gp_expression_trees_ = [copy.deepcopy(tree) for tree in final_gp_trees_iter]
            self.final_iteration_ = it
            self._last_X_shape_ = X_full_for_ga.shape

            if self.verbose:
                pretty_selected = list(final_gp_exprs_inlined_iter)
                detail_lines = [
                    f"[EVOPT][Iter {it}] Selección FINAL iteración",
                    f"   - Score GA             : {self._disp(score_best):.6f}",
                ]
                if model_mses_best:
                    detail_models = ", ".join(
                        f"{k}={self._disp(model_mses_best[k]):.6f}" for k in sorted(model_mses_best)
                    )
                    detail_lines.append(f"   - {self.metric.name} por modelo       : {detail_models}")
                detail_lines.append(f"   - Originales seleccion.: {[self._feature_names[i] for i in selected_original_idx]}")
                detail_lines.append(f"   - GP seleccionadas     : {pretty_selected}")
                print("\n".join(detail_lines) + "\n")

            elapsed_after_ga = time.time() - start_time
            if self.verbose:
                if self.maxtime is not None:
                    print(f"[EVOPT][Iter {it}] Tiempo transcurrido tras GA: {elapsed_after_ga:.2f}s de {self.maxtime}s\n")
                else:
                    print(f"[EVOPT][Iter {it}] Tiempo transcurrido tras GA: {elapsed_after_ga:.2f}s\n")

            if self.sticky_selected_features:
                sticky_selected_exprs.update(kept_exprs[j] for j in selected_prev_gp_idx)
                sticky_selected_exprs.update(expr_new[j] for j in selected_new_gp_idx)
                keep_prev_idx = [idx for idx, expr in enumerate(kept_exprs) if expr in sticky_selected_exprs]
                keep_new_idx = [idx for idx, expr in enumerate(expr_new) if expr in sticky_selected_exprs]
            else:
                keep_prev_idx = selected_prev_gp_idx
                keep_new_idx = selected_new_gp_idx

            Z_prev_kept_next_gp = Z_kept_gp[:, keep_prev_idx] if keep_prev_idx else np.empty((X_gp_base.shape[0], 0))
            Z_new_kept_gp = Z_new_gp[:, keep_new_idx] if keep_new_idx else np.empty((X_gp_base.shape[0], 0))
            Z_kept_gp = (
                np.hstack([Z_prev_kept_next_gp, Z_new_kept_gp])
                if (Z_prev_kept_next_gp.size or Z_new_kept_gp.size)
                else np.empty((X_gp_base.shape[0], 0))
            )

            Z_prev_kept_next_ga = Z_kept_ga[:, keep_prev_idx] if keep_prev_idx else np.empty((X_ga_base.shape[0], 0))
            Z_new_kept_ga = Z_new_ga[:, keep_new_idx] if keep_new_idx else np.empty((X_ga_base.shape[0], 0))
            Z_kept_ga = (
                np.hstack([Z_prev_kept_next_ga, Z_new_kept_ga])
                if (Z_prev_kept_next_ga.size or Z_new_kept_ga.size)
                else np.empty((X_ga_base.shape[0], 0))
            )

            Z_prev_kept_next_full = Z_kept_full[:, keep_prev_idx] if keep_prev_idx else np.empty((n, 0))
            Z_new_kept_full = Z_new_full[:, keep_new_idx] if keep_new_idx else np.empty((n, 0))
            Z_kept_full = (
                np.hstack([Z_prev_kept_next_full, Z_new_kept_full])
                if (Z_prev_kept_next_full.size or Z_new_kept_full.size)
                else np.empty((n, 0))
            )

            kept_exprs = [kept_exprs[j] for j in keep_prev_idx] + (
                [expr_new[j] for j in keep_new_idx] if n_new else []
            )
            kept_exprs_inlined = [kept_exprs_inlined[j] for j in keep_prev_idx] + (
                [expr_new_inlined[j] for j in keep_new_idx] if n_new else []
            )
            kept_exprs_eval = [kept_exprs_eval[j] for j in keep_prev_idx] + (
                [expr_new_eval[j] for j in keep_new_idx] if n_new else []
            )
            kept_expr_trees = [kept_expr_trees[j] for j in keep_prev_idx] + (
                [expr_new_trees[j] for j in keep_new_idx] if n_new else []
            )

            self.final_prev_context_exprs_ = list(kept_exprs)
            self.final_prev_context_trees_ = [copy.deepcopy(tree) for tree in kept_expr_trees]
            self.final_prev_context_exprs_inline_ = list(kept_exprs_inlined)
            self.final_prev_context_exprs_eval_ = list(kept_exprs_eval)

            if self.verbose:
                print(
                    f"[EVOPT][Iter {it}] GP retenidas para siguiente: previas={len(keep_prev_idx)} "
                    f"nuevas={len(keep_new_idx)} | Total contexto siguiente: {Z_kept_gp.shape[1]}\n"
                )

            self.history_.append({
                "iteration": it,
                "ga_best_score": score_best,
                "ga_model_mses": dict(model_mses_best),
                "added_gp_features": k_new,
                "kept_prev_gp_for_next": len(keep_prev_idx),
                "kept_new_gp_for_next": len(keep_new_idx),
                "gp_features_total_for_next": int(Z_kept_gp.shape[1]),
                "gp_expressions_new": list(expr_new),
                "gp_expressions_kept_for_next": list(kept_exprs),
                "gp_expressions_new_inline": list(expr_new_inlined),
                "gp_expressions_kept_for_next_inline": list(kept_exprs_inlined),
                "final_candidate_selected_originals": list(selected_original_idx),
                "final_candidate_selected_gp_exprs": list(final_gp_exprs_iter),
                "final_candidate_selected_gp_exprs_inline": list(final_gp_exprs_inlined_iter),
                "elapsed_after_gp_s": elapsed_after_gp,
                "elapsed_after_ga_s": elapsed_after_ga,
                "elapsed_s": time.time() - start_time,
            })

            if self.maxtime is not None and self._remaining_time_for_iterations(start_time) <= 0:
                if self.verbose:
                    print(f"[EVOPT] Parada por tiempo global al final de iteración {it}.")
                if self.alter_random_state:
                    self.random_state += self.gp_num_generations + self.ga_num_generations
                break

            if self.alter_random_state:
                self.random_state += self.gp_num_generations + self.ga_num_generations

        if not did_any_ga:
            if self.verbose:
                print("[EVOPT] No se ejecutó ningún GA por límite de tiempo. Fallback a todas las originales.\n")
            self.final_X_ = X_orig.copy()
            self.final_y_ = y.copy()
            self.final_best_score_ = float("inf")
            self.final_model_mses_ = {cfg.name: float("inf") for cfg in self._model_configs}
            self.final_selected_original_idx_ = list(range(p))
            self.final_gp_expressions_ = []
            self.final_gp_expression_trees_ = []
            self.final_prev_context_exprs_ = []
            self.final_prev_context_trees_ = []
            self.final_prev_context_exprs_inline_ = []
            self.final_prev_context_exprs_eval_ = []
            self.final_gp_expressions_display_ = []
            self.final_gp_expressions_eval_ = []
            self.final_iteration_ = 0
            self._last_X_shape_ = X_orig.shape

        final_ga_executed = False
        if self.run_final_ga:
            final_ga_executed = self._execute_final_ga(
                X_orig,
                y,
                idx_final_ga,
                Z_kept_full,
                start_time,
            )
            if self.verbose:
                if final_ga_executed:
                    print("[EVOPT] GA final extendido ejecutado.\n")
                else:
                    print("[EVOPT] GA final extendido no se ejecutó.\n")

        self.kept_gp_matrix_ = Z_kept_full
        self.kept_gp_expressions_ = list(kept_exprs)
        self.kept_gp_trees_ = [copy.deepcopy(tree) for tree in kept_expr_trees]

        if self.verbose:
            print(f"[EVOPT] ===== FIN =====")
            print(f"[EVOPT] Última iteración usada : {self.final_iteration_}")
            print(f"[EVOPT] Score GA (última)     : {self._disp(self.final_best_score_):.6f}")
            if self.final_model_mses_:
                detail_models = ", ".join(
                    f"{k}={self._disp(self.final_model_mses_[k]):.6f}"
                    for k in sorted(self.final_model_mses_)
                )
                print(f"[EVOPT] {self.metric.name} por modelo       : {detail_models}")
            print(f"[EVOPT] #orig finales         : {len(self.final_selected_original_idx_)}")
            print(f"[EVOPT] #gp finales           : {len(self.final_gp_expressions_)}")
            print(f"[EVOPT] X final shape         : {None if self.final_X_ is None else self.final_X_.shape}\n")

        return self

    # --------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Aplica la selección FINAL a cualquier X:
          1) Evalúa el CONTEXTO GP previo de la iteración final.
          2) Evalúa las GP seleccionadas en esa iteración.
          3) Concatena [originales seleccionados | GP seleccionadas].
        """
        if self.final_X_ is None:
            raise RuntimeError("Debes ejecutar fit() primero.")

        X = np.asarray(X)
        if self._p_ is None:
            raise RuntimeError("A?n no hay metadatos del n? de originales (?llamaste a fit()?).")
        if X.shape[1] != self._p_:
            raise ValueError(f"X tiene {X.shape[1]} columnas, pero el modelo fue ajustado con {self._p_} originales.")

        n, p = X.shape

        base_args: List[np.ndarray] = [X[:, j] for j in range(p)]
        outputs: List[np.ndarray] = []

        if self.final_gp_expressions_eval_:
            pset_base = self._build_pset_for_nvars(p)
            for idx, expr in enumerate(self.final_gp_expressions_eval_):
                f_final = deap_gp.compile(expr=expr, pset=pset_base)
                try:
                    z = f_final(*base_args)
                except Exception as exc:
                    expr_disp = (
                        self.final_gp_expressions_display_[idx]
                        if idx < len(self.final_gp_expressions_display_)
                        else expr
                    )
                    raise RuntimeError(
                        f"Error evaluando GP final #{idx} ({expr_disp}): {exc}"
                    ) from exc
                z = np.nan_to_num(np.asarray(z, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6).reshape(-1)
                outputs.append(z.reshape(-1, 1))
        else:
            args_list: List[np.ndarray] = list(base_args)

            if self.final_prev_context_exprs_:
                for idx, expr in enumerate(self.final_prev_context_exprs_):
                    pset_ctx = self._build_pset_for_nvars(len(args_list))
                    f_ctx = deap_gp.compile(expr=expr, pset=pset_ctx)
                    try:
                        z_ctx = f_ctx(*args_list)
                    except Exception as exc:
                        expr_disp = expr if not isinstance(expr, str) else self._strip_id_bin_expr(expr)
                        raise RuntimeError(
                            f"Error evaluando GP de contexto #{idx} ({expr_disp}): {exc}"
                        ) from exc
                    z_ctx = np.nan_to_num(
                        np.asarray(z_ctx, dtype=float),
                        nan=0.0,
                        posinf=1e6,
                        neginf=-1e6,
                    ).reshape(-1)
                    args_list.append(z_ctx)
            elif self.final_prev_context_trees_:
                for idx, tree in enumerate(self.final_prev_context_trees_):
                    pset_ctx = self._build_pset_for_nvars(len(args_list))
                    f_ctx = deap_gp.compile(expr=tree, pset=pset_ctx)
                    try:
                        z_ctx = f_ctx(*args_list)
                    except Exception as exc:
                        raise RuntimeError(
                            f"Error evaluando GP de contexto #{idx}: {exc}"
                        ) from exc
                    z_ctx = np.nan_to_num(np.asarray(z_ctx, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6).reshape(-1)
                    args_list.append(z_ctx)

            if self.final_gp_expressions_:
                for idx, expr in enumerate(self.final_gp_expressions_):
                    pset_final = self._build_pset_for_nvars(len(args_list))
                    f_final = deap_gp.compile(expr=expr, pset=pset_final)
                    try:
                        z = f_final(*args_list)
                    except Exception as exc:
                        expr_disp = expr if not isinstance(expr, str) else self._strip_id_bin_expr(expr)
                        raise RuntimeError(
                            f"Error evaluando GP final #{idx} ({expr_disp}): {exc}"
                        ) from exc
                    z = np.nan_to_num(
                        np.asarray(z, dtype=float),
                        nan=0.0,
                        posinf=1e6,
                        neginf=-1e6,
                    ).reshape(-1)
                    outputs.append(z.reshape(-1, 1))
            elif self.final_gp_expression_trees_:
                for idx, tree in enumerate(self.final_gp_expression_trees_):
                    pset_final = self._build_pset_for_nvars(len(args_list))
                    f_final = deap_gp.compile(expr=tree, pset=pset_final)
                    try:
                        z = f_final(*args_list)
                    except Exception as exc:
                        raise RuntimeError(
                            f"Error evaluando GP final #{idx}: {exc}"
                        ) from exc
                    z = np.nan_to_num(np.asarray(z, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6).reshape(-1)
                    outputs.append(z.reshape(-1, 1))

        Z_out = np.hstack(outputs) if outputs else np.empty((n, 0))

        X_sel = X[:, self.final_selected_original_idx_] if self.final_selected_original_idx_ else np.empty((n, 0))
        X_final = np.hstack([X_sel, Z_out]) if (X_sel.size or Z_out.size) else np.empty((n, 0))
        return X_final

    def kept_expressions(self) -> List[str]:
        return list(self.kept_gp_expressions_)

    def history(self) -> List[Dict[str, Any]]:
        return list(self.history_)

    def report_final_selection(
        self,
        original_feature_names: List[str],
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        if self._p_ is None:
            raise RuntimeError("Aún no has ejecutado fit().")
        if len(original_feature_names) != self._p_:
            raise ValueError(f"Se esperaban {self._p_} nombres originales, recibidos {len(original_feature_names)}.")

        # Construye nombres inlined (humanos) y expresiones evaluables en base a originales
        if self.final_prev_context_exprs_inline_:
            context_pretty = list(self.final_prev_context_exprs_inline_)
        else:
            names_list = list(original_feature_names)
            context_pretty = []
            for expr_ctx in self.final_prev_context_exprs_:
                pretty_ctx = self._pretty_expr(expr_ctx, names_list)
                names_list.append(pretty_ctx)
                context_pretty.append(pretty_ctx)

        base_x_names = [f"x{i}" for i in range(self._p_)]
        if self.final_prev_context_exprs_eval_:
            context_eval = list(self.final_prev_context_exprs_eval_)
        else:
            context_eval = []
            for expr_ctx in self.final_prev_context_exprs_:
                eval_ctx = self._inline_expression(expr_ctx, base_x_names, context_eval)
                context_eval.append(eval_ctx)

        if self.final_gp_expressions_display_:
            pretty_final = list(self.final_gp_expressions_display_)
        else:
            names_for_final = list(original_feature_names) + list(context_pretty)
            pretty_final = [self._pretty_expr(e, names_for_final) for e in self.final_gp_expressions_]

        if self.final_gp_expressions_eval_:
            eval_final = list(self.final_gp_expressions_eval_)
        else:
            eval_final = []
            for expr in self.final_gp_expressions_:
                eval_expr = self._inline_expression(expr, base_x_names, context_eval)
                eval_final.append(eval_expr)

        # Dependencias hasta originales (resolviendo contexto)
        pattern = re.compile(r"x(\d+)")

        def deps_to_originals(expr: str) -> set[int]:
            seen = set()
            stack = [expr]
            while stack:
                ex = stack.pop()
                if not isinstance(ex, str):
                    ex = str(ex)
                for m in pattern.finditer(ex):
                    idx = int(m.group(1))
                    if idx < self._p_:
                        seen.add(idx)
                    else:
                        j = idx - self._p_
                        if 0 <= j < len(context_eval):
                            stack.append(context_eval[j])
            return seen

        uses_map: Dict[str, List[str]] = {name: [] for name in original_feature_names}
        for expr_eval, expr_pretty in zip(eval_final, pretty_final):
            for vidx in deps_to_originals(expr_eval):
                uses_map[original_feature_names[vidx]].append(expr_pretty)

        selected_orig_names = [original_feature_names[i] for i in self.final_selected_original_idx_]

        report: Dict[str, Any] = {
            "best_iteration": self.final_iteration_,
            "best_score": self.final_best_score_,
            "selected_originals": selected_orig_names,
            "selected_transformations": pretty_final,                      # bonitas con nombres (inlined)
            "selected_transformations_raw": list(self.final_gp_expressions_),  # crudas xN
            "original_to_transformations": uses_map,
            "metric_name": self.metric.name,
        }

        if X_test is not None and y_test is not None:
            if self.final_X_ is None or self._model_configs is None:
                raise RuntimeError("No hay información final de entrenamiento para evaluar en test.")
            X_test_arr = np.asarray(X_test)
            y_test_arr = np.asarray(y_test).ravel()
            if self._p_ is None:
                raise RuntimeError("Metadatos de número de columnas no disponibles.")
            if X_test_arr.shape[1] != self._p_:
                raise ValueError(
                    f"X_test tiene {X_test_arr.shape[1]} columnas, pero el modelo fue ajustado con {self._p_}."
                )

            X_test_transformed = self.transform(X_test_arr)

            per_model: Dict[str, float] = {}
            for cfg in self._model_configs:
                estimator = cfg.make_estimator()
                if self.final_X_ is None or self.final_X_.shape[1] == 0:
                    score_val = float("inf")
                else:
                    estimator.fit(self.final_X_, self._y_fit_)
                    preds = estimator.predict(X_test_transformed)
                    score_val = float(self.metric.loss(y_test_arr, preds))
                per_model[cfg.name] = score_val

            mean_score = float(np.mean(list(per_model.values()))) if per_model else float("inf")
            report["test_results"] = {
                "per_model": per_model,
                "mean_score": mean_score,
                "metric_name": self.metric.name,
                "mean_mse": mean_score,
            }

        return report

    def _build_feature_timeline(
        self,
        feature_names: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        history = self.history_
        if not history:
            return None
        if feature_names is None:
            if self._feature_names is None:
                raise ValueError(
                    "No hay nombres de features almacenados. Proporciona feature_names de forma explicita."
                )
            feature_names = list(self._feature_names)
        else:
            feature_names = list(feature_names)
        orig_count = len(feature_names)
        iterations = [entry["iteration"] for entry in history]
        num_iters = len(iterations)

        feature_nodes: List[str] = []
        node_metadata: Dict[str, Dict[str, Any]] = {}
        row_index: Dict[str, int] = {}
        status_matrix: Dict[str, List[float]] = {}

        original_node_ids: List[str] = []
        for idx, name in enumerate(feature_names):
            node_id = f"orig_{idx}"
            original_node_ids.append(node_id)
            row_index[node_id] = len(row_index)
            feature_nodes.append(node_id)
            node_metadata[node_id] = {
                "name": name,
                "type": "original",
                "iteration_created": 0,
                "dependencies_idx": {idx},
                "expr": None,
            }
            status_matrix[node_id] = [float("nan")] * num_iters

        expr_to_node: Dict[str, str] = {}
        feature_dependencies: Dict[str, Set[int]] = {}
        pattern = re.compile(r"x(\d+)")

        def dependencies_from_expression(expr: str, context_exprs: List[str]) -> Set[int]:
            deps: Set[int] = set()
            stack = [expr]
            visited: Set[str] = set()
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                for m in pattern.finditer(current):
                    idx = int(m.group(1))
                    if idx < orig_count:
                        deps.add(idx)
                    else:
                        ctx_idx = idx - orig_count
                        if 0 <= ctx_idx < len(context_exprs):
                            ctx_expr = context_exprs[ctx_idx]
                            if ctx_expr in feature_dependencies:
                                deps.update(feature_dependencies[ctx_expr])
                            else:
                                stack.append(ctx_expr)
            return deps

        context_exprs: List[str] = []
        context_exprs_inline: List[str] = []
        for iter_idx, entry in enumerate(history):
            iteration = entry.get("iteration", iter_idx + 1)
            selected_original_idx = set(entry.get("final_candidate_selected_originals", []))
            selected_gp_exprs = set(entry.get("final_candidate_selected_gp_exprs", []))

            new_exprs = entry.get("gp_expressions_new", [])
            new_exprs_inline = entry.get("gp_expressions_new_inline")
            inline_lookup: Dict[str, str] = {}
            for idx_ctx, expr_ctx in enumerate(context_exprs):
                if idx_ctx < len(context_exprs_inline):
                    inline_lookup[expr_ctx] = context_exprs_inline[idx_ctx]
            if not new_exprs_inline or len(new_exprs_inline) != len(new_exprs):
                new_exprs_inline = [
                    self._inline_expression(
                        expr,
                        feature_names,
                        context_exprs_inline,
                        strip_id_bin=True,
                    )
                    for expr in new_exprs
                ]
            for expr_idx, expr in enumerate(new_exprs):
                deps_idx = dependencies_from_expression(expr, context_exprs)
                feature_dependencies[expr] = deps_idx
                pretty_name = (
                    new_exprs_inline[expr_idx]
                    if expr_idx < len(new_exprs_inline)
                    else self._pretty_expr(expr, feature_names + context_exprs_inline)
                )
                inline_lookup[expr] = pretty_name
                node_id = expr_to_node.get(expr)
                if node_id is None:
                    node_id = f"gp_{len(expr_to_node)}"
                    expr_to_node[expr] = node_id
                    row_index[node_id] = len(row_index)
                    feature_nodes.append(node_id)
                    status_matrix[node_id] = [float("nan")] * num_iters
                    node_metadata[node_id] = {
                        "name": pretty_name,
                        "type": "gp",
                        "iteration_created": iteration,
                        "dependencies_idx": deps_idx,
                        "expr": expr,
                    }
                else:
                    pretty_expr = inline_lookup.get(expr)
                    if pretty_expr is None:
                        pretty_expr = self._inline_expression(
                            expr,
                            feature_names,
                            context_exprs_inline,
                            strip_id_bin=True,
                        )
                    node_metadata[node_id].setdefault("iteration_created", iteration)
                    node_metadata[node_id]["name"] = pretty_expr
                    node_metadata[node_id]["dependencies_idx"] = deps_idx

            active_gp_exprs = set(context_exprs) | set(new_exprs)

            for orig_idx, node_id in enumerate(original_node_ids):
                status_matrix[node_id][iter_idx] = 1.0 if orig_idx in selected_original_idx else 0.0

            for expr in active_gp_exprs:
                node_id = expr_to_node.get(expr)
                if node_id is None:
                    node_id = f"gp_ctx_{len(expr_to_node)}"
                    expr_to_node[expr] = node_id
                    row_index[node_id] = len(row_index)
                    feature_nodes.append(node_id)
                    status_matrix[node_id] = [float("nan")] * num_iters
                    pretty_expr = inline_lookup.get(expr)
                    if pretty_expr is None:
                        pretty_expr = self._inline_expression(
                            expr,
                            feature_names,
                            context_exprs_inline,
                            strip_id_bin=True,
                        )
                    node_metadata[node_id] = {
                        "name": pretty_expr,
                        "type": "gp",
                        "iteration_created": iteration,
                        "dependencies_idx": feature_dependencies.get(expr, set()),
                        "expr": expr,
                    }
                selected_value = 1.0 if expr in selected_gp_exprs else 0.0
                status_matrix[node_id][iter_idx] = selected_value

            context_exprs = list(entry.get("gp_expressions_kept_for_next", []))
            context_exprs_inline_next = entry.get("gp_expressions_kept_for_next_inline")
            if not context_exprs_inline_next or len(context_exprs_inline_next) != len(context_exprs):
                context_exprs_inline_next = [
                    self._inline_expression(
                        expr,
                        feature_names,
                        context_exprs_inline,
                        strip_id_bin=True,
                    )
                    for expr in context_exprs
                ]
            context_exprs_inline = list(context_exprs_inline_next)

        status_grid = np.array([status_matrix[node_id] for node_id in feature_nodes], dtype=float)
        return {
            "iterations": iterations,
            "feature_nodes": feature_nodes,
            "status_grid": status_grid,
            "node_metadata": node_metadata,
            "row_index": row_index,
            "original_node_ids": original_node_ids,
        }

    def _execute_final_ga(
        self,
        X_orig: np.ndarray,
        y: np.ndarray,
        idx_final_ga: np.ndarray,
        Z_kept_full: np.ndarray,
        start_time: float,
    ) -> bool:
        if idx_final_ga.size == 0:
            if self.verbose:
                print("[EVOPT] GA final extendido: sin muestras disponibles.\n")
            return False
        if self._feature_names is None:
            raise RuntimeError("No hay nombres de features almacenados para el GA final.")

        reserved_time = max(0, int(self._final_ga_reserved_time))
        self._final_ga_started = True
        self._final_ga_reserved_time = 0

        X_final_base = X_orig[idx_final_ga]
        y_final_base = y[idx_final_ga]
        Z_final = (
            Z_kept_full[idx_final_ga]
            if Z_kept_full.size
            else np.empty((X_final_base.shape[0], 0))
        )
        X_final_ga_input = (
            np.hstack([X_final_base, Z_final]) if Z_final.size else X_final_base
        )

        if self.final_ga_retune_models:
            cv_folds = self.model_tuning_cv_folds or self.ga_cv_folds
            if cv_folds < 2:
                raise ValueError("model_tuning_cv_folds debe ser >=2.")
            if self.verbose:
                print("[EVOPT] Reajustando modelos antes del GA final...\n")
            X_retune_input = np.hstack([X_orig, Z_kept_full]) if Z_kept_full.size else X_orig
            self._model_configs = tune_models(
                X=X_retune_input,
                y=y,
                model_names=self.model_names,
                standardize=self.standardize,
                cv_folds=cv_folds,
                random_state=self.random_state,
                task_type=self.task_type,
                metric=self.metric,
                verbose=self.verbose,
            )
            if self.verbose:
                print("[EVOPT] Modelos reajustados:", [cfg.name for cfg in self._model_configs])
                print()

        feature_names = list(self._feature_names)
        if Z_final.shape[1] > 0:
            feature_names += [f"gctx{j}" for j in range(Z_final.shape[1])]

        # Configuración específica del GA final (población base *4 si no se fija)
        if self.estimate_pop_size and self.final_ga_population_size is not None and self.verbose:
            print("[EVOPT] estimate_pop_size=True -> ignorando final_ga_population_size fijada.")
        if self.estimate_pop_size and self._ga_pop_estimated and self._ga_pop_estimate is not None:
            final_pop = max(2, int(self._ga_pop_estimate) * 4)
            final_pop_note = "estimada x4"
        elif self.final_ga_population_size is None:
            base_pop = self._auto_population_size(X_final_ga_input.shape[0], X_final_ga_input.shape[1])
            final_pop = max(2, base_pop * 4)
            final_pop_note = "auto x4"
        else:
            final_pop = int(self.final_ga_population_size)
            final_pop_note = "fijada"

        overrides: Dict[str, Any] = {
            "population_size": final_pop,
            "num_generations": self.final_ga_num_generations,
            "patience": self.final_ga_patience,
        }
        if self.final_ga_maxtime is not None:
            overrides["maxtime"] = self.final_ga_maxtime

        if self.verbose:
            print(
                f"[EVOPT] Ejecutando GA final extendido sobre shape={X_final_ga_input.shape} | "
                f"población={final_pop} ({final_pop_note})\n"
            )

        final_ga = self._build_ga(overrides=overrides)

        if self.maxtime is not None:
            remaining_total = max(0, int(math.floor(self._remaining_time(start_time))))
            remaining_total = max(1, remaining_total)
            # Allow the final GA to use any leftover time, even if final_ga_maxtime was set.
            final_ga.maxtime = int(remaining_total)

        final_ga.fit(X_final_ga_input, y_final_base, feature_names=feature_names)

        mask_best = final_ga.best_individual_.astype(int)
        score_best = float(final_ga.best_score_)
        model_mses_best = final_ga.best_model_mses()
        selected_idx_all = np.where(mask_best == 1)[0]

        n_orig = self._p_ if self._p_ is not None else X_orig.shape[1]
        n_ctx = Z_final.shape[1]

        selected_original_idx = [i for i in range(min(n_orig, mask_best.size)) if mask_best[i] == 1]
        selected_ctx_idx = [
            j for j in range(n_ctx)
            if (n_orig + j) < mask_best.size and mask_best[n_orig + j] == 1
        ]

        selected_ctx_exprs: List[str] = []
        selected_ctx_exprs_inline: List[str] = []
        selected_ctx_exprs_eval: List[str] = []
        selected_ctx_trees: List[deap_gp.PrimitiveTree] = []
        if selected_ctx_idx:
            if not self.final_prev_context_exprs_ or not self.final_prev_context_trees_:
                raise RuntimeError("No hay contexto GP final disponible para el GA extendido.")
            if not self.final_prev_context_exprs_inline_:
                raise RuntimeError("No hay expresiones GP en formato expandido para el GA extendido.")
            if not self.final_prev_context_exprs_eval_:
                raise RuntimeError("No hay expresiones GP en formato eval para el GA extendido.")
            for j in selected_ctx_idx:
                if j < len(self.final_prev_context_exprs_):
                    selected_ctx_exprs.append(self.final_prev_context_exprs_[j])
                if j < len(self.final_prev_context_exprs_inline_):
                    selected_ctx_exprs_inline.append(self.final_prev_context_exprs_inline_[j])
                if j < len(self.final_prev_context_exprs_eval_):
                    selected_ctx_exprs_eval.append(self.final_prev_context_exprs_eval_[j])
                if j < len(self.final_prev_context_trees_):
                    selected_ctx_trees.append(copy.deepcopy(self.final_prev_context_trees_[j]))

        X_full_for_ga = (
            np.hstack([X_orig, Z_kept_full]) if Z_kept_full.size else X_orig
        )

        if selected_idx_all.size == 0:
            if self.verbose:
                print("[EVOPT] GA final extendido no seleccionó ninguna feature.\n")
            return False

        self.final_X_ = X_full_for_ga[:, selected_idx_all].copy()
        self.final_y_ = y.copy()
        self.final_best_score_ = score_best
        self.final_model_mses_ = dict(model_mses_best)
        self.final_selected_original_idx_ = selected_original_idx
        self.final_gp_expressions_ = selected_ctx_exprs
        self.final_gp_expressions_display_ = selected_ctx_exprs_inline
        self.final_gp_expressions_eval_ = selected_ctx_exprs_eval
        self.final_gp_expression_trees_ = selected_ctx_trees
        self.final_prev_context_exprs_inline_ = selected_ctx_exprs_inline
        self.final_prev_context_exprs_eval_ = selected_ctx_exprs_eval
        if self.final_iteration_ is None:
            self.final_iteration_ = 0
        self._last_X_shape_ = X_full_for_ga.shape
        return True

    def plot_feature_evolution(self, feature_names: Optional[List[str]] = None) -> None:
        """Wrapper legacy: muestra ambos gráficos de evolución."""
        self.plot_feature_evolution_history(feature_names)
        self.plot_feature_dependency_matrix(feature_names)

    def plot_feature_evolution_history(self, feature_names: Optional[List[str]] = None) -> None:
        timeline = self._build_feature_timeline(feature_names)
        if not timeline:
            raise RuntimeError("No hay historia. Llama antes a fit().")

        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap

        iterations: List[int] = timeline["iterations"]  # type: ignore[index]
        feature_nodes: List[str] = timeline["feature_nodes"]  # type: ignore[index]
        status_grid: np.ndarray = timeline["status_grid"]  # type: ignore[index]
        node_metadata: Dict[str, Dict[str, Any]] = timeline["node_metadata"]  # type: ignore[index]

        category_grid = np.zeros_like(status_grid, dtype=int)
        active_mask = ~np.isnan(status_grid)
        category_grid[~active_mask] = 0
        category_grid[active_mask & (status_grid == 0)] = 1
        category_grid[active_mask & (status_grid > 0)] = 2

        cmap = ListedColormap(["#f2f2f2", "#fddbc7", "#2ca25f"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

        fig, ax = plt.subplots(
            figsize=(max(8, len(iterations) * 1.4), max(6, len(feature_nodes) * 0.45)),
            constrained_layout=True,
        )

        im = ax.imshow(category_grid, aspect="auto", cmap=cmap, norm=norm, origin="upper")
        ax.set_xticks(np.arange(len(iterations)))
        ax.set_xticklabels(iterations)
        ax.set_xlabel("Iteracion EVOPT")

        feature_labels = [str(node_metadata[node]["name"]) for node in feature_nodes]
        ax.set_yticks(np.arange(len(feature_nodes)))
        ax.set_yticklabels(feature_labels)
        ax.set_title("Seleccion por iteracion (0 = no, 1 = si)")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(["No disponible", "No seleccionada", "Seleccionada"])
        plt.show()

    def plot_feature_dependency_matrix(self, feature_names: Optional[List[str]] = None) -> None:
        timeline = self._build_feature_timeline(feature_names)
        if not timeline:
            raise RuntimeError("No hay historia. Llama antes a fit().")

        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap

        feature_nodes: List[str] = timeline["feature_nodes"]  # type: ignore[index]
        node_metadata: Dict[str, Dict[str, Any]] = timeline["node_metadata"]  # type: ignore[index]
        original_node_ids: List[str] = timeline["original_node_ids"]  # type: ignore[index]

        final_exprs = set(self.final_gp_expressions_ or [])
        final_gp_node_ids = [
            node_id
            for node_id in feature_nodes
            if node_metadata[node_id].get("type") == "gp"
            and node_metadata[node_id].get("expr") in final_exprs
        ]

        relation_labels = [
            str(node_metadata[node_id].get("name", node_id)) for node_id in final_gp_node_ids
        ]
        original_labels = [
            str(node_metadata[node_id].get("name", node_id)) for node_id in original_node_ids
        ]

        fig_width = max(6, len(original_labels) * 0.6 + 1)
        fig_height = max(4, len(relation_labels) * 0.6 + 1)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

        if final_gp_node_ids:
            relation_matrix = []
            for node_id in final_gp_node_ids:
                deps_idx: Set[int] = node_metadata[node_id].get("dependencies_idx", set())  # type: ignore[assignment]
                row = [1 if orig_idx in deps_idx else 0 for orig_idx in range(len(original_labels))]
                relation_matrix.append(row)

            relation_grid = np.array(relation_matrix, dtype=int)
            rel_cmap = ListedColormap(["#f2f2f2", "#3182bd"])
            rel_norm = BoundaryNorm([-0.5, 0.5, 1.5], rel_cmap.N)
            rel_im = ax.imshow(
                relation_grid,
                aspect="auto",
                cmap=rel_cmap,
                norm=rel_norm,
                origin="upper",
            )
            ax.set_yticks(np.arange(len(relation_labels)))
            ax.set_yticklabels(relation_labels)
            ax.set_xticks(np.arange(len(original_labels)))
            ax.set_xticklabels(original_labels, rotation=45, ha="right")
            ax.set_xlabel("Variables originales")
            ax.set_ylabel("Transformaciones finales")
            ax.set_title("Originales usadas por GP finales")
            ax.tick_params(axis="x", labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            cbar_rel = fig.colorbar(rel_im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1])
            cbar_rel.ax.set_yticklabels(["No utilizada", "Utilizada"])
        else:
            ax.text(
                0.5,
                0.5,
                "Sin transformaciones finales",
                ha="center",
                va="center",
                fontsize=10,
                transform=ax.transAxes,
            )
            ax.axis("off")

        ax.grid(False)
        plt.show()

    def plot_history(self):
        import matplotlib.pyplot as plt
        if not self.history_:
            raise RuntimeError("No hay historia. Llama antes a fit().")

        its = [h["iteration"] for h in self.history_]
        ga_best = [self._disp(h["ga_best_score"]) for h in self.history_]
        kept_next = [h["gp_features_total_for_next"] for h in self.history_]

        fig, ax1 = plt.subplots()
        ax1.plot(its, ga_best, marker="o", label=f"{self.metric.name} GA (iter)")
        ax1.set_xlabel("Iteración")
        ax1.set_ylabel(f"{self.metric.name} (GA)")
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(its, kept_next, marker="s", linestyle="--", label="#GP retenidas (siguiente)", alpha=0.7)
        ax2.set_ylabel("#GP retenidas")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
        plt.title("Evolución por iteración (EVOPT)")
        plt.show()
