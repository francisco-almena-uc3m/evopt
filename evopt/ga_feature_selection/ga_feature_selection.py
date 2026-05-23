# ga_feature_selection.py
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import KFold

from evopt.model_utils import ModelConfig, TaskMetric, evaluate_models_cv

BAD_FITNESS = 1e12


# ============================================================
#                   UTILIDADES INTERNAS
# ============================================================

def init_cv(cv_folds: int, random_state: int) -> KFold:
    """Inicializa un KFold reproducible."""
    return KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)


def evaluate_individual(
    X: np.ndarray,
    y: np.ndarray,
    individual: np.ndarray,
    cv: KFold,
    model_configs: List[ModelConfig],
    metric: TaskMetric,
    cache: Optional[Dict[bytes, Tuple[float, Dict[str, float]]]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Evalúa un individuo (máscara binaria de features) mediante CV.
    Retorna (score medio, dict por modelo).
    """
    cache_key: Optional[bytes] = None
    if cache is not None:
        cache_key = np.asarray(individual, dtype=np.uint8).tobytes()
        cached = cache.get(cache_key)
        if cached is not None:
            return float(cached[0]), dict(cached[1])

    mask = individual.astype(bool)
    if not mask.any():
        mask[0] = True

    X_sel = X[:, mask]
    try:
        mean_mse, per_model = evaluate_models_cv(X_sel, y, cv, model_configs, metric)
    except Exception:
        mean_mse = BAD_FITNESS
        per_model = {cfg.name: BAD_FITNESS for cfg in model_configs}
    if cache is not None and cache_key is not None:
        cache[cache_key] = (float(mean_mse), dict(per_model))
    return float(mean_mse), per_model


def evaluate_population(
    X: np.ndarray,
    y: np.ndarray,
    population: np.ndarray,
    cv: KFold,
    model_configs: List[ModelConfig],
    metric: TaskMetric,
    cache: Optional[Dict[bytes, Tuple[float, Dict[str, float]]]] = None,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Evalúa toda la población."""
    scores = np.zeros(population.shape[0], dtype=float)
    details: List[Dict[str, float]] = []
    for i, ind in enumerate(population):
        mean_mse, per_model = evaluate_individual(X, y, ind, cv, model_configs, metric, cache=cache)
        scores[i] = mean_mse
        details.append(per_model)
    return scores, details


def create_initial_population(
    population_size: int,
    num_variables: int,
    individual_all_features: bool,
    rng: np.random.Generator,
    initial_bit_prob: float = 0.5,
) -> np.ndarray:
    """Crea población inicial binaria."""
    population = []
    if not (0.0 <= initial_bit_prob <= 1.0):
        raise ValueError("[GA] initial_bit_prob debe estar en [0,1].")

    if individual_all_features:
        population.append(np.ones(num_variables, dtype=int))

    while len(population) < population_size:
        ind = (rng.random(num_variables) < initial_bit_prob).astype(np.int32)
        if not np.any(ind):
            ind[rng.integers(0, num_variables)] = 1
        population.append(ind.astype(int))

    return np.vstack(population)


def tournament_selection(
    population: np.ndarray,
    scores: np.ndarray,
    tournament_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Selección por torneo: el mejor (menor MSE) entre un subconjunto aleatorio."""
    n = population.shape[0]
    idxs = rng.integers(0, n, size=tournament_size)  # con reemplazo
    best_idx = idxs[np.argmin(scores[idxs])]
    return population[best_idx].copy()


def select_population(
    population: np.ndarray,
    scores: np.ndarray,
    target_size: int,
    num_variables: int,
    tournament_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Construye la mating pool seleccionando individuos por torneos independientes.
    Devuelve un array (target_size, num_variables).
    """
    selected = np.zeros((target_size, num_variables), dtype=int)
    for i in range(target_size):
        selected[i] = tournament_selection(
            population=population,
            scores=scores,
            tournament_size=tournament_size,
            rng=rng,
        )
    return selected


def single_point_crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    num_variables: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Cruce de un punto entre dos padres."""
    if num_variables < 2:
        return p1.copy(), p2.copy()
    point = rng.integers(1, num_variables)
    c1 = np.concatenate([p1[:point], p2[point:]])
    c2 = np.concatenate([p2[:point], p1[point:]])
    return c1, c2


def multi_point_crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    num_variables: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Cruce de mǭltiples puntos con nǭmero de cortes aleatorio."""
    if num_variables < 2:
        return p1.copy(), p2.copy()
    max_points = num_variables - 1
    if max_points == 1:
        return single_point_crossover(p1, p2, num_variables, rng)
    num_points = int(rng.integers(2, max_points + 1))
    cut_positions = rng.choice(
        np.arange(1, num_variables, dtype=int),
        size=num_points,
        replace=False,
    )
    cut_positions.sort()
    cut_positions = np.concatenate([cut_positions, np.array([num_variables])])

    c1_segments: list[np.ndarray] = []
    c2_segments: list[np.ndarray] = []
    start = 0
    swap = False
    for cut in cut_positions:
        if swap:
            c1_segments.append(p2[start:cut])
            c2_segments.append(p1[start:cut])
        else:
            c1_segments.append(p1[start:cut])
            c2_segments.append(p2[start:cut])
        swap = not swap
        start = cut

    c1 = np.concatenate(c1_segments)
    c2 = np.concatenate(c2_segments)
    return c1, c2


def reproduce_population(
    population: np.ndarray,
    offspring_size: int,
    num_variables: int,
    crossover_mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Genera la descendencia mediante cruces.
    Robusto cuando offspring_size > población (padres con reemplazo).
    """
    n = population.shape[0]
    offspring = np.zeros((offspring_size, num_variables), dtype=int)
    k = 0
    while k < offspring_size:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        p1, p2 = population[i], population[j]
        if crossover_mode == "single":
            c1, c2 = single_point_crossover(p1, p2, num_variables, rng)
        elif crossover_mode == "multi":
            c1, c2 = multi_point_crossover(p1, p2, num_variables, rng)
        else:
            raise ValueError(f"[GA] crossover_mode desconocido: {crossover_mode!r}")
        offspring[k] = c1
        k += 1
        if k < offspring_size:
            offspring[k] = c2
            k += 1
    return offspring


def mutate_individual(
    individual: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mutación por flip de un bit aleatorio, garantizando al menos una característica."""
    out = individual.copy()
    idx = int(rng.integers(0, out.shape[0]))
    out[idx] = 1 - out[idx]
    if not np.any(out):
        out[rng.integers(0, out.shape[0])] = 1
    return out


# ============================================================
#                   CLASE PRINCIPAL GA
# ============================================================

class GAFeatureSelector:
    """
    Algoritmo Genético para selección de características (wrapper con CV).
    Paradas: num_generations | maxtime | patience.
    """

    def __init__(
        self,
        population_size: int,
        num_generations: int,
        tournament_size: int,
        offspring_size: int,   
        elitism_size: int,     
        cv_folds: int,
        individual_all_features: bool,
        random_state: int,
        verbose: bool,
        maxtime: int,
        patience: int,
        mutation_probability: float,
        evolve_mutation_probability: bool,
        mutation_probability_end: float,
        model_configs: Optional[List[ModelConfig]],
        metric: TaskMetric,
        crossover_type: str,
        alter_random_state: bool,
        initial_bit_prob: float = 0.5,
    ) -> None:

        # Parámetros base
        self.population_size = int(population_size)
        self.num_generations = int(num_generations)
        self.tournament_size = int(tournament_size)
        self.cv_folds = int(cv_folds)
        self.individual_all_features = bool(individual_all_features)
        if not (0.0 <= initial_bit_prob <= 1.0):
            raise ValueError("[GA] initial_bit_prob debe estar en [0,1].")
        self.initial_bit_prob = float(initial_bit_prob)
        self.random_state = int(random_state)
        self.alter_random_state = bool(alter_random_state)
        self.verbose = bool(verbose)
        self.model_configs = list(model_configs or [])
        if not self.model_configs:
            raise ValueError("[GA] Debes proporcionar al menos un modelo en model_configs.")
        self.metric = metric
        self.crossover_type = str(crossover_type).lower()
        if self.crossover_type not in {"single", "multi"}:
            raise ValueError(
                "[GA] crossover_type debe ser 'single' o 'multi' "
                f"(recibido: {self.crossover_type!r})."
            )

        if not (0.0 <= mutation_probability <= 1.0):
            raise ValueError("[GA] mutation_probability debe estar en [0,1].")
        if not (0.0 <= mutation_probability_end <= 1.0):
            raise ValueError("[GA] mutation_probability_end debe estar en [0,1].")

        self.mutation_probability = float(mutation_probability)
        self.evolve_mutation_probability = bool(evolve_mutation_probability)
        self.mutation_probability_end = float(mutation_probability_end)
        self._current_mutation_probability = self.mutation_probability

        # Paradas
        self.maxtime = maxtime
        self.patience = patience

        # Resolver relación población/descendencia/elitismo
        self.offspring_size, self.elitism_size, self._sizing_note = self._resolve_sizes(
            population_size=self.population_size,
            offspring_size=offspring_size,
            elitism_size=elitism_size,
            who="GA",
        )

        # Internos
        self._rng = np.random.default_rng(self.random_state)
        self._cv = None

        # Post-fit
        self.num_variables_: int | None = None
        self.population_: np.ndarray | None = None
        self.population_scores_: np.ndarray | None = None
        self.population_model_mses_: List[Dict[str, float]] | None = None
        self.best_individual_: np.ndarray | None = None
        self.best_score_: float | None = None
        self.best_model_mses_: Dict[str, float] | None = None
        self.history_: list[dict] = []

        # Parada
        self.stop_reason_: str | None = None
        self.stop_detail_: str | None = None

        if self.verbose and self._sizing_note:
            print(self._sizing_note)

    def _disp(self, value: float) -> float:
        if hasattr(self.metric, "display_value"):
            return float(self.metric.display_value(value))
        return float(value)

    @staticmethod
    def _resolve_sizes(
        population_size: int,
        offspring_size: float | int | None,
        elitism_size: float | int | None,
        who: str,
    ) -> tuple[int, int, str | None]:
        """
        Reglas:
          - Ambos None: os=ps, es=0
          - Uno None: el otro completa (ps - conocido). Si complemento < 0, saturar a 0 y avisar.
          - Ambos dados: si os+es < ps -> ERROR. Si > ps -> OK (se filtra descendencia sobrante).
        """
        ps = int(population_size)
        note = None

        def _as_count(val: float | int | None) -> float | int | None:
            if val is None:
                return None
            if isinstance(val, (int, float)) and 0.0 < float(val) <= 1.0:
                return int(round(ps * float(val)))
            return int(val)

        os_in = _as_count(offspring_size)
        es_in = _as_count(elitism_size)

        if os_in is None and es_in is None:
            os = ps; es = 0
            note = f"[{who}] Config por defecto: offspring_size={os}, elitism_size={es}."
            return os, es, note

        if os_in is None and es_in is not None:
            es = es_in
            if es < 0:
                raise ValueError(f"[{who}] elitism_size no puede ser negativo (es={es}).")
            os = ps - es
            if os < 0:
                note = (f"[{who}] Aviso: elitism_size ({es}) > population_size ({ps}). "
                        f"Se establece offspring_size=0. La población se formará solo con élites.")
                os = 0
            else:
                note = f"[{who}] offspring_size no especificado, se completa a {os} (ps - es)."
            return os, es, note

        if os_in is not None and es_in is None:
            os = os_in
            if os < 0:
                raise ValueError(f"[{who}] offspring_size no puede ser negativo (os={os}).")
            es = ps - os
            if es < 0:
                note = (f"[{who}] Aviso: offspring_size ({os}) > population_size ({ps}). "
                        f"Se establece elitism_size=0. La supervivencia filtrará descendencia.")
                es = 0
            else:
                note = f"[{who}] elitism_size no especificado, se completa a {es} (ps - os)."
            return os, es, note

        os = os_in; es = es_in
        if os < 0 or es < 0:
            raise ValueError(f"[{who}] offspring/elites no pueden ser negativos (os={os}, es={es}).")

        if os + es < ps:
            raise ValueError(
                f"[{who}] Config inválida: offspring_size ({os}) + elitism_size ({es}) "
                f"< population_size ({ps})."
            )

        if os + es > ps:
            note = (f"[{who}] Nota: os + es ({os + es}) > population_size ({ps}). "
                    f"Se filtrarán {max(0, os - (ps - es))} hijos para mantener tamaño.")
        return os, es, note

    # ============================================================
    #                         FIT
    # ============================================================
    def _schedule_fraction(self, gen_index_zero_based: int) -> float:
        if self.num_generations > 1:
            return float(np.clip(gen_index_zero_based / (self.num_generations - 1), 0.0, 1.0))
        return 0.0

    @staticmethod
    def _blend_scalar(start: float, end: float, frac: float) -> float:
        return float((1.0 - frac) * float(start) + frac * float(end))

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "GAFeatureSelector":
        """Entrena el GA y selecciona el subconjunto de features."""
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        self.num_variables_ = X.shape[1]

        names = feature_names if feature_names is not None else [f"x{i}" for i in range(self.num_variables_)]

        # Chequeo suma (por si modifican atributos después del init)
        if (self.offspring_size + self.elitism_size) < self.population_size:
            raise ValueError(
                "[GA] Config inválida en fit(): offspring_size + elitism_size "
                f"({self.offspring_size + self.elitism_size}) < population_size "
                f"({self.population_size})."
            )

        # Cronómetro
        start_time = time.time()

        # CV y población inicial
        self._cv = init_cv(self.cv_folds, self.random_state)
        self._rng = np.random.default_rng(self.random_state)
        eval_cache: Optional[Dict[bytes, Tuple[float, Dict[str, float]]]] = (
            {} if not self.alter_random_state else None
        )
        self.population_ = create_initial_population(
            population_size=self.population_size,
            num_variables=self.num_variables_,
            individual_all_features=self.individual_all_features,
            rng=self._rng,
            initial_bit_prob=self.initial_bit_prob,
        )

        # Evaluación inicial (Gen 0)
        self.population_scores_, self.population_model_mses_ = evaluate_population(
            X, y, self.population_, self._cv, self.model_configs, self.metric, cache=eval_cache
        )
        best_idx = int(np.argmin(self.population_scores_))
        self.best_individual_ = self.population_[best_idx].copy()
        self.best_score_ = float(self.population_scores_[best_idx])
        self.best_model_mses_ = dict(self.population_model_mses_[best_idx])
        best_global = self.best_score_
        best_global_models = dict(self.population_model_mses_[best_idx])
        no_improve_gens = 0

        # -------- Verbose Gen 0 con mismo formato --------
        if self.verbose:
            print("[GA] ---- Generación 0 ----")
            mask0 = self.population_[best_idx].astype(int)
            sel0 = [names[i] for i, v in enumerate(mask0) if v == 1]
            print(f"[GA] Métricas gen 00")
            best_gen0 = float(np.min(self.population_scores_))
            mean_gen0 = float(np.mean(self.population_scores_))
            print(f"     - Mejor de la generación : {self._disp(best_gen0):.6f}")
            print(f"     - Media                   : {self._disp(mean_gen0):.6f}")
            print(f"     - STD                     : {float(np.std(self.population_scores_)):.6f}")
            print(f"     - Mejor de la generación (bin): {mask0}")
            print(f"     - Mejor de la generación (nombres): {sel0}")
            detail_best = ", ".join(
                f"{k}={self._disp(self.population_model_mses_[best_idx][k]):.6f}"
                for k in sorted(self.population_model_mses_[best_idx])
            )
            if detail_best:
                print(f"     - {self.metric.name} por modelo (gen 00) : {detail_best}")
            print(f"     - Mejor global hasta ahora: {self._disp(best_global):.6f}")
            print(f"     - Mejor global (bin): {self.best_individual_.astype(int)}")
            print(f"     - Mejor global (nombres): {[names[i] for i, v in enumerate(self.best_individual_) if v == 1]}")
            detail_global = ", ".join(
                f"{k}={self._disp(best_global_models[k]):.6f}" for k in sorted(best_global_models)
            )
            if detail_global:
                print(f"     - {self.metric.name} por modelo (global): {detail_global}")
            print()

        # ======================
        # Bucle evolutivo
        # ======================
        for gen in range(1, self.num_generations + 1):
            # Parada por tiempo
            if self.maxtime is not None and (time.time() - start_time) >= float(self.maxtime):
                elapsed = time.time() - start_time
                self.stop_reason_ = "tiempo"
                self.stop_detail_ = f"{elapsed:.2f}s >= {self.maxtime}s"
                if self.verbose:
                    print(f"[GA] Parada por tiempo: {self.stop_detail_}\n")
                break

            # ---------- CV de la generación ----------
            if self.alter_random_state:
                self.random_state += 1
                self._rng = np.random.default_rng(self.random_state)
                self._cv = init_cv(self.cv_folds, self.random_state)

            frac = self._schedule_fraction(gen - 1)
            if self.evolve_mutation_probability:
                self._current_mutation_probability = self._blend_scalar(
                    self.mutation_probability, self.mutation_probability_end, frac
                )
            else:
                self._current_mutation_probability = self.mutation_probability

            if self.verbose:
                print(f"[GA] ---- Generación {gen} ----")

            # ---- Elitismo (según puntuaciones de la generación anterior) ----
            elite_inds, _, _ = self._elitism()

            # ---- Selección (mating pool) ----
            parents = select_population(
                population=self.population_,
                scores=self.population_scores_,
                target_size=self.population_size,
                num_variables=self.num_variables_,
                tournament_size=self.tournament_size,
                rng=self._rng,
            )

            # ---- Reproducción ----
            offspring = reproduce_population(
                population=parents,
                offspring_size=self.offspring_size,
                num_variables=self.num_variables_,
                crossover_mode=self.crossover_type,
                rng=self._rng,
            )

            # ---- Mutación ----
            for i in range(len(offspring)):
                if self._rng.random() < self._current_mutation_probability:
                    offspring[i] = mutate_individual(offspring[i], self._rng)

            # ---- Evaluación de descendencia con CV de esta generación ----
            offspring_scores, offspring_model_mses = evaluate_population(
                X, y, offspring, self._cv, self.model_configs, self.metric, cache=eval_cache
            )

            if self.elitism_size > 0:
                elite_scores, elite_model_mses = evaluate_population(
                    X, y, elite_inds, self._cv, self.model_configs, self.metric, cache=eval_cache
                )
            else:
                elite_scores = np.empty((0,), dtype=float)
                elite_model_mses = []

            # ---- Supervivencia ----
            replace_size = max(0, self.population_size - self.elitism_size)
            if replace_size > 0:
                top_idx = np.argsort(offspring_scores)[:replace_size]
                survivors = offspring[top_idx].copy()
                survivors_scores = offspring_scores[top_idx].copy()
                survivors_model_mses = [offspring_model_mses[idx] for idx in top_idx]
            else:
                survivors = np.empty((0, self.num_variables_), dtype=int)
                survivors_scores = np.empty((0,), dtype=float)
                survivors_model_mses = []

            # Nueva población
            self.population_ = np.vstack([elite_inds, survivors])
            self.population_scores_ = np.concatenate([elite_scores, survivors_scores])
            self.population_model_mses_ = list(elite_model_mses) + list(survivors_model_mses)

            if self.individual_all_features:
                full_ind = np.ones(self.num_variables_, dtype=int)
                has_full = any(np.array_equal(ind, full_ind) for ind in self.population_)
                if not has_full:
                    full_score, full_models = evaluate_individual(
                        X, y, full_ind, self._cv, self.model_configs, self.metric, cache=eval_cache
                    )
                    worst_idx = int(np.argmax(self.population_scores_))
                    self.population_[worst_idx] = full_ind
                    self.population_scores_[worst_idx] = full_score
                    self.population_model_mses_[worst_idx] = full_models

            # Actualiza mejor de la generación / global / paciencia
            gen_best_idx = int(np.argmin(self.population_scores_))
            gen_best_score = float(self.population_scores_[gen_best_idx])
            gen_best_models = self.population_model_mses_[gen_best_idx]

            if gen_best_score < self.best_score_:
                self.best_score_ = gen_best_score
                self.best_individual_ = self.population_[gen_best_idx].copy()
                self.best_model_mses_ = dict(gen_best_models)

            improved = False
            if gen_best_score < best_global:
                best_global = gen_best_score
                best_global_models = dict(gen_best_models)
                no_improve_gens = 0
                improved = True
            else:
                no_improve_gens += 1

            # -------- Verbose de generación con binario y nombres --------
            if self.verbose:
                mask_gen = self.population_[gen_best_idx].astype(int)
                sel_gen = [names[i] for i, v in enumerate(mask_gen) if v == 1]
                mask_glob = self.best_individual_.astype(int)
                sel_glob = [names[i] for i, v in enumerate(mask_glob) if v == 1]
                print(f"[GA] Métricas gen {gen:02d}")
                mean_gen = float(np.mean(self.population_scores_))
                print(f"     - Mejor de la generación : {self._disp(gen_best_score):.6f}")
                print(f"     - Media                   : {self._disp(mean_gen):.6f}")
                print(f"     - STD                     : {float(np.std(self.population_scores_)):.6f}")
                print(f"     - Mejor de la generación (bin): {mask_gen}")
                print(f"     - Mejor de la generación (nombres): {sel_gen}")
                detail_gen = ", ".join(
                    f"{k}={self._disp(gen_best_models[k]):.6f}" for k in sorted(gen_best_models)
                )
                if detail_gen:
                    print(f"     - {self.metric.name} por modelo (gen)    : {detail_gen}")
                print(f"     - Mejor global hasta ahora: {self._disp(best_global):.6f}")
                print(f"     - Mejor global (bin): {mask_glob}")
                print(f"     - Mejor global (nombres): {sel_glob}")
                detail_global = ", ".join(
                    f"{k}={self._disp(best_global_models[k]):.6f}" for k in sorted(best_global_models)
                )
                if detail_global:
                    print(f"     - {self.metric.name} por modelo (global): {detail_global}")
                print()

            # Parada por paciencia
            if self.patience is not None and self.patience > 0 and not improved:
                if no_improve_gens >= int(self.patience):
                    self.stop_reason_ = "paciencia"
                    self.stop_detail_ = f"{no_improve_gens} generaciones sin mejora"
                    if self.verbose:
                        print(f"[GA] Parada por paciencia: {self.stop_detail_}.\n")
                    break
        else:
            self.stop_reason_ = "número de generaciones"
            self.stop_detail_ = f"se alcanzó el máximo de {self.num_generations} generaciones"

        if self.verbose:
            print(f"[GA] ===== FIN =====")
            print(f"[GA] Motivo de parada : {self.stop_reason_} ({self.stop_detail_})")
            print(f"[GA] Mejor score global: {self._disp(self.best_score_):.6f}")
            print(f"[GA] Mejor individuo   : {self.best_individual_.astype(int)}")
            if self.best_model_mses_:
                detail = ", ".join(
                    f"{k}={self._disp(self.best_model_mses_[k]):.6f}" for k in sorted(self.best_model_mses_)
                )
                print(f"[GA] {self.metric.name} por modelo   : {detail}")
            print()

        return self

    # ============================================================
    #                  MÉTODOS AUXILIARES
    # ============================================================

    def _elitism(self) -> tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
        if self.elitism_size <= 0:
            return (
                np.empty((0, self.num_variables_), dtype=int),
                np.empty((0,), dtype=float),
                []
            )
        elite_idx = np.argsort(self.population_scores_)[:self.elitism_size]
        elites = self.population_[elite_idx].copy()
        elite_scores = self.population_scores_[elite_idx].copy()
        elite_models = [dict(self.population_model_mses_[idx]) for idx in elite_idx]
        return elites, elite_scores, elite_models

    def _log_generation(self, gen: int, scores: np.ndarray, best_global: float) -> None:
        # (Mantenido por compatibilidad; no se usa para imprimir nombres)
        best = float(np.min(scores))
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        self.history_.append({"generation": gen, "best_score": best, "mean_score": mean, "std_score": std})
        if self.verbose and gen < 0:  # deshabilitado por defecto
            print(f"[GA] Métricas gen {gen:02d}")
            print(f"     - Mejor de la generación : {best:.6f}")
            print(f"     - Mejor global hasta ahora: {best_global:.6f}")
            print(f"     - Media                   : {mean:.6f}")
            print(f"     - STD                     : {std:.6f}\n")

    # ============================================================
    #                   API PÚBLICA
    # ============================================================

    def best_features(self) -> np.ndarray:
        if self.best_individual_ is None:
            raise RuntimeError("Debes ejecutar fit() primero.")
        return np.where(self.best_individual_ == 1)[0]

    def best_score(self) -> float:
        return float(self.best_score_) if self.best_score_ is not None else np.nan

    def best_model_mses(self) -> Dict[str, float]:
        return dict(self.best_model_mses_) if self.best_model_mses_ is not None else {}

    def history(self) -> list[dict]:
        return list(self.history_) if self.history_ is not None else []
