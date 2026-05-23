import time
import copy
import operator
import sys
import numpy as np
import random
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from deap import base, creator, tools, gp
from sklearn.model_selection import KFold

from evopt.model_utils import ModelConfig, TaskMetric, evaluate_models_cv

# =========================
#  PRIMITIVAS PROTEGIDAS (VECTORIALES) + SANEADO
# =========================

EPS = 1e-12
CLIP_IN = 1e6       # recorte para entradas a primitivas
CLIP_OUT = 1e6      # recorte para salidas de primitivas
BAD_FITNESS = 1e12  # penalización si algo sale mal

def _asarray(x):
    return np.asarray(x, dtype=float)

def _sanitize(a, clip: float = CLIP_OUT):
    """Convierte a float, reemplaza NaN/Inf y recorta magnitudes."""
    a = _asarray(a)
    a = np.nan_to_num(a, nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(a, -clip, clip)

# ---- Identidad binaria (para GP tipado) ----
def f_binary_identity(x):
    """Mantiene entradas binarias en [0, 1] para permitir árboles tipados."""
    x = _asarray(x)
    return np.clip(x, 0.0, 1.0)

# ---- Operadores básicos seguros ----
def f_add(x, y):
    x = np.clip(_asarray(x), -CLIP_IN, CLIP_IN)
    y = np.clip(_asarray(y), -CLIP_IN, CLIP_IN)
    return _sanitize(x + y)

def f_sub(x, y):
    x = np.clip(_asarray(x), -CLIP_IN, CLIP_IN)
    y = np.clip(_asarray(y), -CLIP_IN, CLIP_IN)
    return _sanitize(x - y)

def f_mul(x, y):
    x = np.clip(_asarray(x), -CLIP_IN, CLIP_IN)
    y = np.clip(_asarray(y), -CLIP_IN, CLIP_IN)
    out = x * y
    return _sanitize(out)

# ---- Divisiones/transformaciones protegidas SIN evaluar la rama mala ----
def f_protected_div(x, y):
    x = _asarray(x); y = _asarray(y)
    bshape = np.broadcast(x, y).shape
    out = np.ones(bshape, dtype=float)                     
    mask = np.broadcast_to(np.abs(y) > EPS, bshape)       
    np.divide(x, y, out=out, where=mask)
    return _sanitize(out)

def f_protected_inv(x):
    x = _asarray(x)
    out = np.zeros_like(x, dtype=float) 
    np.divide(1.0, x, out=out, where=(np.abs(x) > EPS))
    return _sanitize(out)

def f_protected_exp(x):
    x = _asarray(x)
    return _sanitize(np.exp(np.clip(x, -50.0, 50.0)))

def f_protected_log(x):
    x = _asarray(x)
    return _sanitize(np.log1p(np.abs(x)))

def f_protected_pow(x, y):
    # Maneja x<0 y exponentes no enteros, 0**neg, y magnitudes grandes
    x = np.clip(_asarray(x), -CLIP_IN, CLIP_IN)
    y = np.clip(_asarray(y), -32.0, 32.0)  # limitar exponente
    mask_zero_neg = (np.abs(x) <= EPS) & (y < 0)
    # Evita complejos: si base<0 y exponente no entero, usar |x|
    y_is_int = np.isfinite(y) & np.isclose(y, np.round(y))
    base = np.where((x < 0) & (~y_is_int), np.abs(x), x)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        val = np.power(base, y)
    val = np.where(mask_zero_neg, 0.0, val)
    return _sanitize(val)

def f_square(x):
    x = np.clip(_asarray(x), -np.sqrt(CLIP_OUT), np.sqrt(CLIP_OUT))
    return _sanitize(x * x)

def f_cube(x):
    x = np.clip(_asarray(x), -CLIP_IN**(1/3), CLIP_IN**(1/3))
    return _sanitize(x * x * x)

def f_protected_sqrt(x):
    x = _asarray(x)
    return _sanitize(np.sqrt(np.abs(x)))

def f_abs(x):
    return _sanitize(np.abs(_asarray(x)))

def f_sin(x):
    return _sanitize(np.sin(_asarray(x)))

def f_cos(x):
    return _sanitize(np.cos(_asarray(x)))

def f_protected_tan(x):
    x = _asarray(x)
    # recorta entrada para evitar verticales de tan
    return _sanitize(np.tan(np.clip(x, -10.0, 10.0)))

def f_relu(x):
    x = _asarray(x)
    return _sanitize(np.maximum(0.0, x))

def f_maximum(x, y):
    x = np.clip(_asarray(x), -CLIP_IN, CLIP_IN)
    y = np.clip(_asarray(y), -CLIP_IN, CLIP_IN)
    return _sanitize(np.maximum(x, y))

def f_minimum(x, y):
    x = np.clip(_asarray(x), -CLIP_IN, CLIP_IN)
    y = np.clip(_asarray(y), -CLIP_IN, CLIP_IN)
    return _sanitize(np.minimum(x, y))


OPERATOR_CATALOG: Dict[str, Tuple[int, callable]] = {
    "add":  (2, f_add),
    "sub":  (2, f_sub),
    "mul":  (2, f_mul),
    "div":  (2, f_protected_div),
    "exp":  (1, f_protected_exp),
    "log":  (1, f_protected_log),
    "pow":  (2, f_protected_pow),
    "square": (1, f_square),
    "cube":   (1, f_cube),
    "sqrt": (1, f_protected_sqrt),
    "inv":  (1, f_protected_inv),
    "sin":  (1, f_sin),
    "cos":  (1, f_cos),
    "tan":  (1, f_protected_tan),
    "relu": (1, f_relu),
    "abs":  (1, f_abs),
    "id_bin": (1, f_binary_identity),
    "max":  (2, f_maximum),
    "min":  (2, f_minimum),
}


class Continuous(float):
    """Marca usada por GP tipado para representar flujos continuos."""
    pass


class Binary(float):
    """Marca usada por GP tipado para representar flujos binarios 0/1."""
    pass


class Const(float):
    """Marca usada por GP tipado para constantes efímeras (solo en ops binarias continuas)."""
    pass

# ========== Helpers Ephemeral Constants (picklables) ==========
def _rand_uniform(low: float, high: float, rng: Optional[random.Random] = None) -> float:
    """Devuelve un float uniforme en [low, high) usando el RNG indicado (o random global)."""
    generator = rng if rng is not None else random
    return float(generator.uniform(low, high))

def _rand_choice(pool: List[float], rng: Optional[random.Random] = None) -> float:
    """Devuelve un elemento aleatorio del pool usando el RNG indicado (o random global)."""
    generator = rng if rng is not None else random
    return float(generator.choice(pool))

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
            inner = _strip_id_bin_expr("".join(content))
            out.append(inner)
            if i < n and expr[i] == ")":
                i += 1
            continue
        out.append(expr[i])
        i += 1
    return "".join(out)

def _generate_const_safe(pset, min_, max_, condition, type_=None):
    """Generate tree, forcing terminals for Const or types without primitives."""
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while stack:
        depth, cur_type = stack.pop()
        no_prims = not pset.primitives.get(cur_type)
        force_terminal = (cur_type is Const) or no_prims
        if force_terminal or condition(height, depth):
            try:
                term = random.choice(pset.terminals[cur_type])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError(
                    "The gp.generate function tried to add a terminal of type '%s', but there is none available."
                    % (cur_type,)
                ).with_traceback(traceback)
            if type(term) is gp.MetaEphemeral:
                term = term()
            expr.append(term)
        else:
            try:
                prim = random.choice(pset.primitives[cur_type])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError(
                    "The gp.generate function tried to add a primitive of type '%s', but there is none available."
                    % (cur_type,)
                ).with_traceback(traceback)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr

def _gen_grow_const_safe(pset, min_, max_, type_=None):
    def condition(height, depth):
        return depth == height or (depth >= min_ and random.random() < pset.terminalRatio)
    return _generate_const_safe(pset, min_, max_, condition, type_)

def _gen_full_const_safe(pset, min_, max_, type_=None):
    def condition(height, depth):
        return depth == height
    return _generate_const_safe(pset, min_, max_, condition, type_)

def _gen_half_and_half_const_safe(pset, min_, max_, type_=None):
    method = random.choice((_gen_grow_const_safe, _gen_full_const_safe))
    return method(pset, min_, max_, type_)


class GPFeatureGenerator:
    def __init__(
        self,
        # ---------- Tamaños de población / selección ----------
        population_size: int,
        offspring_size: Optional[float],
        elitism_size: Optional[float],
        tournament_size: int,

        # ---------- Criterios de parada ----------
        num_generations: Optional[int],
        maxtime: Optional[int],
        patience: Optional[int],

        # ---------- Evaluación ----------
        cv_folds: int,
        penalty_mode: Optional[str],
        penalty_coeff: float,

        # ---------- Salida (cuántas features generar) ----------
        num_new_features: int,

        # ---------- Espacio de búsqueda ----------
        min_tree_height: int,
        max_tree_height: int,
        unary_continuous_operators: List[str],
        unary_binary_operators: List[str],
        binary_continuous_operators: List[str],
        binary_binary_operators: List[str],
        binary_mixed_operators: List[str],
        root_excluded_operator_set: Optional[List[str]],
        ephemeral_constants: Optional[List[float]],
        ephemeral_constants_range: Optional[List[float]],

        # ---------- Operadores genéticos ----------
        crossover_probability: float,
        mutation_probability: float,
        mutation_weights: Tuple[float, float, float],

        # ---------- Schedules de exploración->explotación ----------
        evolve_mutation_weights: bool,
        mutation_weights_end: Tuple[float, float, float],
        evolve_mutation_probability: bool,
        mutation_probability_end: float,

        # ---------- Aleatoriedad / logging ----------
        random_state: int,
        alter_random_state: bool,
        rng: np.random.Generator,
        verbose: bool,

        # ---------- (Opcional) nombres de variables originales para logs bonitos ----------
        feature_names: Optional[List[str]],
        feature_is_binary: Optional[List[bool]],
        model_configs: Optional[List[ModelConfig]],
        duplicate_correlation_threshold: float,
        metric: TaskMetric,
        init_population_method: str = "half_and_half",
    ) -> None:

        # Tamaños / selección
        self.population_size = int(population_size)

        # Resolver relación población / descendencia / elitismo (reglas solicitadas)
        (self.offspring_size,
         self.elitism_size,
         self._sizing_note) = self._resolve_sizes(
            population_size=self.population_size,
            offspring_size=offspring_size,
            elitism_size=elitism_size,
            who="GP"
        )

        self.tournament_size = int(tournament_size)
        # Tamaño que puede ocupar la descendencia tras poner élites
        self._replacement_size = max(0, self.population_size - self.elitism_size)

        # Paradas
        self.num_generations = num_generations
        self.maxtime = maxtime
        self.patience = patience

        # Evaluación
        self.cv_folds = int(cv_folds)
        self.metric = metric
        penalty_mode = (penalty_mode or "").strip().lower()
        if penalty_mode not in {"depth", "size", ""}:
            raise ValueError("penalty_mode debe ser 'depth', 'size' o None.")
        self.penalty_mode: Optional[str] = penalty_mode if penalty_mode else None
        self.penalty_coeff: float = float(penalty_coeff)

        # Salida
        self.num_new_features = int(num_new_features)

        # Espacio de búsqueda
        self.min_tree_height = max(0, int(min_tree_height))
        self.max_tree_height = int(max_tree_height)
        self.unary_continuous_operators = self._validate_operator_list(
            unary_continuous_operators, 1, "gp_unary_continuous_operators"
        )
        self.unary_binary_operators = self._validate_operator_list(
            unary_binary_operators, 1, "gp_unary_binary_operators"
        )
        self.binary_continuous_operators = self._validate_operator_list(
            binary_continuous_operators, 2, "gp_binary_continuous_operators"
        )
        self.binary_binary_operators = self._validate_operator_list(
            binary_binary_operators, 2, "gp_binary_binary_operators"
        )
        self.binary_mixed_operators = self._validate_operator_list(
            binary_mixed_operators, 2, "gp_binary_mixed_operators"
        )
        self.root_excluded_operator_set = set(root_excluded_operator_set or [])
        self.ephemeral_constants = list(ephemeral_constants or [])

        if self.min_tree_height > self.max_tree_height:
            raise ValueError(
                f"min_tree_height ({self.min_tree_height}) no puede ser mayor que "
                f"max_tree_height ({self.max_tree_height})."
            )
        self.ephemeral_constants_range = ephemeral_constants_range

        # Operadores genéticos
        self.crossover_probability = float(crossover_probability)
        self.mutation_probability = float(mutation_probability)
        self.mutation_weights = self._normalize_weights_tuple(mutation_weights)

        # Schedules
        self.evolve_mutation_weights = bool(evolve_mutation_weights)
        self.mutation_weights_end = self._normalize_weights_tuple(mutation_weights_end)
        self.evolve_mutation_probability = bool(evolve_mutation_probability)
        self.mutation_probability_end = float(mutation_probability_end)
        self.duplicate_corr_threshold = float(duplicate_correlation_threshold)
        if not (0.0 <= self.duplicate_corr_threshold <= 1.0):
            raise ValueError("duplicate_correlation_threshold debe estar en [0, 1].")

        # Estado dinámico (se actualiza cada generación)
        self._current_mutation_weights = self.mutation_weights
        self._current_mutation_probability = self.mutation_probability

        # Aleatoriedad / logs
        self.random_state = int(random_state)
        self.alter_random_state = bool(alter_random_state)
        # RNG dedicado para constantes efímeras; evita reutilizar valores al reseedear random global.
        self._const_rng = random.Random(self.random_state)
        self.rng = rng if rng is not None else np.random.default_rng(self.random_state)
        self.verbose = verbose

        # Nombres de variables para logs
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.feature_is_binary = (
            [bool(v) for v in feature_is_binary]
            if feature_is_binary is not None
            else None
        )
        self._pretty_names = self._make_pretty_names(self.feature_names) if self.feature_names else None
        self.model_configs: List[ModelConfig] = list(model_configs or [])
        if not self.model_configs:
            raise ValueError("Debes proporcionar al menos un modelo en model_configs.")
        self._primitive_base_names: Dict[str, str] = {}
        self.init_population_method = str(init_population_method or "half_and_half").strip().lower()
        if self.init_population_method in {"halfandhalf", "half-and-half", "half"}:
            self.init_population_method = "half_and_half"
        if self.init_population_method not in {"half_and_half", "grow", "full"}:
            raise ValueError(
                "init_population_method debe ser 'half_and_half', 'grow' o 'full'."
            )

        # Resultados
        self._best_exprs: List[str] = []
        self._best_mses: List[float] = []
        self._best_vectors_: Optional[np.ndarray] = None
        self._best_trees: List[gp.PrimitiveTree] = []
        self._best_model_mses: List[Dict[str, float]] = []

        # Motivo de parada
        self.stop_reason_: Optional[str] = None  # "tiempo" | "paciencia" | "número de generaciones"
        self.stop_detail_: Optional[str] = None

        if self.verbose and self._sizing_note:
            print(self._sizing_note)

    # ---------------- Utilidades internas ----------------

    @staticmethod
    def _normalize_weights_tuple(w: Tuple[float, float, float]) -> Tuple[float, float, float]:
        a, b, c = w
        a = max(0.0, float(a)); b = max(0.0, float(b)); c = max(0.0, float(c))
        s = a + b + c
        if s <= 0.0:
            return (1/3, 1/3, 1/3)
        return (a/s, b/s, c/s)

    @staticmethod
    def _validate_operator_list(ops: Optional[Iterable[str]], expected_arity: int, param_name: str) -> List[str]:
        clean: List[str] = []
        seen: Set[str] = set()
        if not ops:
            return clean
        for name in ops:
            if name in seen:
                continue
            if name not in OPERATOR_CATALOG:
                raise ValueError(f"{param_name}: operador '{name}' no existe en OPERATOR_CATALOG.")
            arity, _ = OPERATOR_CATALOG[name]
            if arity != expected_arity:
                raise ValueError(
                    f"{param_name}: operador '{name}' tiene aridad {arity} pero se esperaba {expected_arity}."
                )
            seen.add(name)
            clean.append(name)
        return clean

    @staticmethod
    def _make_pretty_names(names: Optional[List[str]]) -> Optional[List[str]]:
        if not names:
            return None
        out = []
        used = set()
        for nm in names:
            s = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in nm.strip())
            if not s or s[0].isdigit():
                s = f"v_{s}"
            base = s
            k = 1
            while s in used:
                s = f"{base}_{k}"
                k += 1
            used.add(s)
            out.append(s)
        return out

    def _expr_pretty(self, expr: str) -> str:
        """Sustituye x{i} por nombre bonito si hay feature_names."""
        if not self._pretty_names:
            import re
            out = re.sub(r"\b([A-Za-z][A-Za-z0-9]*)__\w+\b", r"\1", expr)
            return _strip_id_bin_expr(out)
        # Reemplazo seguro por tokens 'xNN'
        import re
        expr_base = re.sub(r"\b([A-Za-z][A-Za-z0-9]*)__\w+\b", r"\1", expr)

        def repl(match):
            idx = int(match.group(1))
            if 0 <= idx < len(self._pretty_names):
                return self._pretty_names[idx]
            return match.group(0)
        out = re.sub(r"\bx(\d+)\b", repl, expr_base)
        return _strip_id_bin_expr(out)

    def _disp(self, value: float) -> float:
        if hasattr(self.metric, "display_value"):
            return float(self.metric.display_value(value))
        return float(value)

    def _ensure_feature_type_mask(self, n_vars: int) -> List[bool]:
        if self.feature_is_binary is None:
            self.feature_is_binary = [False] * n_vars
        if len(self.feature_is_binary) != n_vars:
            raise ValueError(
                f"feature_is_binary tiene longitud {len(self.feature_is_binary)} y debería ser {n_vars}."
            )
        return self.feature_is_binary

    def _primitive_base_name(self, prim_name: str) -> str:
        if prim_name in self._primitive_base_names:
            return self._primitive_base_names[prim_name]
        if "__" in prim_name:
            return prim_name.split("__", 1)[0]
        return prim_name

    def _add_typed_primitive(
        self,
        pset: gp.PrimitiveSetTyped,
        op_name: str,
        arg_types: Tuple[type, ...],
        suffix: str,
    ) -> None:
        arity, func = OPERATOR_CATALOG[op_name]
        if arity != len(arg_types):
            raise ValueError(
                f"El operador '{op_name}' tiene aridad {arity}, incompatible con arg_types={arg_types}."
            )
        unique_name = f"{op_name}__{suffix}"
        counter = 1
        while unique_name in self._primitive_base_names:
            unique_name = f"{op_name}__{suffix}_{counter}"
            counter += 1
        pset.addPrimitive(func, arg_types, Continuous, name=unique_name)
        self._primitive_base_names[unique_name] = op_name

    def _tree_to_expr(self, tree: gp.PrimitiveTree) -> str:
        def recurse(idx: int) -> Tuple[str, int]:
            node = tree[idx]
            # --- Terminal: puede ser variable (x0, x1, ...) o constante efímera ---
            if isinstance(node, gp.Terminal):
                # Si tiene 'value' numérico, es (casi seguro) una constante efímera -> imprime el número
                if hasattr(node, "value"):
                    val = node.value
                    # Detectamos tipos numéricos (incluye numpy)
                    if isinstance(val, (int, float, np.integer, np.floating)):
                        valf = float(val)
                        if np.isfinite(valf):
                            return (format(valf, ".12g"), idx + 1)
                    # Para variables renombradas (value="x0") devolvemos directamente ese identificador
                    if isinstance(val, str) and val:
                        return (val, idx + 1)
                # Usa el formateador propio de DEAP si está disponible (cubre constantes y x{i})
                try:
                    formatted = node.format()
                except Exception:
                    formatted = None
                if isinstance(formatted, str) and formatted:
                    return (formatted, idx + 1)
                return (str(node), idx + 1)

            # --- Primitive: construimos recursivamente sus argumentos ---
            args = []
            next_idx = idx + 1
            for _ in range(node.arity):
                arg_expr, next_idx = recurse(next_idx)
                args.append(arg_expr)
            prim_name = self._primitive_base_name(node.name)
            return (f"{prim_name}({', '.join(args)})", next_idx)

        expr, _ = recurse(0)
        return expr


    def _schedule_fraction(self, gen_index_zero_based: int) -> float:
        """
        Fracción de progreso del schedule en [0,1].
        - gen_index_zero_based = 0  => fracción 0.0 (primera generación = valores originales)
        - gen_index_zero_based = num_generations-1 => fracción 1.0
        """
        if self.num_generations and self.num_generations > 1:
            return float(np.clip(gen_index_zero_based / (self.num_generations - 1), 0.0, 1.0))
        return 0.0

    def _blend_tuple_by_fraction(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        frac: float
    ) -> Tuple[float, float, float]:
        ws = np.asarray(start, dtype=float)
        we = np.asarray(end, dtype=float)
        w = (1.0 - frac) * ws + frac * we
        return self._normalize_weights_tuple(tuple(w.tolist()))

    def _blend_scalar_by_fraction(self, start: float, end: float, frac: float) -> float:
        val = (1.0 - frac) * float(start) + frac * float(end)
        return float(np.clip(val, 0.0, 1.0))

    @staticmethod
    def _resolve_sizes(
        population_size: int,
        offspring_size: Optional[float],
        elitism_size: Optional[float],
        who: str
    ) -> Tuple[int, int, Optional[str]]:
        """
        Reglas:
          - Si ambas son None:
                offspring_size = population_size
                elitism_size = 0
          - Si exactamente una es None:
                completar con la otra: complemento exacto a population_size
                (os = ps - es, o es = ps - os). Si el complemento fuera negativo,
                se satura a 0 y se continúa (la supervivencia filtrará el exceso).
          - Si ambas están dadas:
                * Si os + es > ps: permitido. Se filtrará la descendencia
                  para que solo sobrevivan los (ps - es) mejores hijos.
                * Si os + es < ps: ERROR -> lanzar ValueError.
        """
        ps = int(population_size)
        note = None

        def _as_count(val: Optional[float], pname: str) -> Optional[int]:
            if val is None:
                return None
            if isinstance(val, (float, int)) and 0.0 < float(val) <= 1.0:
                return int(round(ps * float(val)))
            return int(val)

        os_in = _as_count(offspring_size, "offspring_size")
        es_in = _as_count(elitism_size, "elitism_size")

        if os_in is None and es_in is None:
            os = ps
            es = 0
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

        os = os_in
        es = es_in
        if os < 0 or es < 0:
            raise ValueError(f"[{who}] offspring/elites no pueden ser negativos (os={os}, es={es}).")

        if os + es < ps:
            raise ValueError(
                f"[{who}] Config inválida: offspring_size ({os}) + elitism_size ({es}) "
                f"< population_size ({ps})."
            )

        if os + es > ps:
            note = (f"[{who}] Nota: os + es ({os + es}) > population_size ({ps}). "
                    f"Se filtrarán los peores hijos para que sobrevivan solo "
                    f"{max(0, ps - es)} descendientes.")
        return os, es, note

    def _build_pset(self, n_vars: int) -> gp.PrimitiveSetTyped:
        mask = self._ensure_feature_type_mask(n_vars)
        arg_types = [Binary if is_bin else Continuous for is_bin in mask]
        has_binary_inputs = any(is_bin for is_bin in mask)
        self._primitive_base_names = {}

        pset = gp.PrimitiveSetTyped("MAIN", arg_types, Continuous)

        for name in self.unary_continuous_operators:
            self._add_typed_primitive(pset, name, (Continuous,), "uc")
        has_const = bool(self.ephemeral_constants) or (
            self.ephemeral_constants_range and len(self.ephemeral_constants_range) == 2
        )
        for name in self.binary_continuous_operators:
            self._add_typed_primitive(pset, name, (Continuous, Continuous), "cc")
            if has_const:
                self._add_typed_primitive(pset, name, (Continuous, Const), "ck")
                self._add_typed_primitive(pset, name, (Const, Continuous), "kc")
        # Solo registrar operadores que requieren flujos binarios si hay entradas binarias
        if has_binary_inputs:
            # Primitiva binaria para permitir generación de árboles tipados
            pset.addPrimitive(f_binary_identity, (Binary,), Binary, name="id_bin")
            for name in self.unary_binary_operators:
                self._add_typed_primitive(pset, name, (Binary,), "ub")
            for name in self.binary_binary_operators:
                self._add_typed_primitive(pset, name, (Binary, Binary), "bb")
            for name in self.binary_mixed_operators:
                self._add_typed_primitive(pset, name, (Continuous, Binary), "cb")
                self._add_typed_primitive(pset, name, (Binary, Continuous), "bc")

        # ---- Ephemeral constants SIN lambda (picklable) ----
        if len(self.ephemeral_constants) > 0:
            pool = [float(v) for v in self.ephemeral_constants]
            pset.addEphemeralConstant(
                "const_disc",
                partial(_rand_choice, pool, self._const_rng),
                ret_type=Const,
            )
        elif self.ephemeral_constants_range and len(self.ephemeral_constants_range) == 2:
            low, high = float(self.ephemeral_constants_range[0]), float(self.ephemeral_constants_range[1])
            pset.addEphemeralConstant(
                "const",
                partial(_rand_uniform, low, high, self._const_rng),
                ret_type=Const,
            )

        pset.renameArguments(**{f"ARG{i}": f"x{i}" for i in range(n_vars)})
        return pset

    def _build_toolbox(self, pset: gp.PrimitiveSet):
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("clone", copy.deepcopy)
        if self.init_population_method == "grow":
            expr_basic = _gen_grow_const_safe
        elif self.init_population_method == "full":
            expr_basic = _gen_full_const_safe
        else:
            expr_basic = _gen_half_and_half_const_safe
        toolbox.register(
            "expr_basic",
            expr_basic,
            pset=pset,
            min_=self.min_tree_height,
            max_=self.max_tree_height,
        )
        toolbox.register("individual_basic", tools.initIterate, creator.Individual, toolbox.expr_basic)

        def _is_valid_individual(individual) -> bool:
            return (
                not self._has_unary_on_ephemeral(individual)
                and not self._has_binary_same_variable(individual)
            )

        def individual_root_restricted():
            for _ in range(256):
                ind = toolbox.individual_basic()
                root = ind[0]
                if isinstance(root, gp.Primitive) and self._primitive_base_name(root.name) in self.root_excluded_operator_set:
                    continue
                if not _is_valid_individual(ind):
                    continue
                return ind
            if self.verbose:
                print("[GP][WARN] No se pudo forzar raíz válida tras 256 intentos; usando individuo básico.")
            if getattr(pset, "arguments", None):
                return creator.Individual.from_string(pset.arguments[0], pset)
            return toolbox.individual_basic()

        toolbox.register("individual", individual_root_restricted)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        def mate_safe(ind1, ind2):
            for _ in range(16):
                c1 = toolbox.clone(ind1)
                c2 = toolbox.clone(ind2)
                c1, c2 = gp.cxOnePoint(c1, c2)
                if _is_valid_individual(c1) and _is_valid_individual(c2):
                    return c1, c2
            return ind1, ind2

        toolbox.register("mate", mate_safe)

        # Generadores para mutación
        toolbox.register(
            "expr_mut",
            _gen_full_const_safe,
            min_=self.min_tree_height,
            max_=self.max_tree_height,
            pset=pset,
        )
        toolbox.register("expr_leaf", _gen_full_const_safe, min_=0, max_=0, pset=pset)  # SIEMPRE una hoja

        # --- Mutaciones locales ---
        def mutate_tree_root_safe(individual):
            for _ in range(64):
                ind_clone = toolbox.clone(individual)
                ind_clone, = gp.mutUniform(ind_clone, expr=toolbox.expr_mut, pset=pset)
                root = ind_clone[0]
                if isinstance(root, gp.Primitive) and self._primitive_base_name(root.name) in self.root_excluded_operator_set:
                    continue
                if _is_valid_individual(ind_clone):
                    return (ind_clone,)
            return (individual,)

        def mutate_terminal(individual):
            term_idx = [i for i, node in enumerate(individual) if isinstance(node, gp.Terminal)]
            if not term_idx:
                return (individual,)
            for _ in range(64):
                ind_clone = toolbox.clone(individual)
                i = random.choice(term_idx)
                new_leaf = toolbox.expr_leaf()
                if len(new_leaf) != 1 or not isinstance(new_leaf[0], gp.Terminal):
                    continue
                if str(new_leaf[0]) == str(ind_clone[i]):
                    continue
                sl = ind_clone.searchSubtree(i)
                ind_clone[sl] = new_leaf
                if _is_valid_individual(ind_clone):
                    return (ind_clone,)
            return (individual,)

        def mutate_function(individual):
            prim_idx = [i for i, node in enumerate(individual) if isinstance(node, gp.Primitive)]
            if not prim_idx:
                return (individual,)
            i = random.choice(prim_idx)
            current = individual[i]
            cand = [p for p in pset.primitives[pset.ret] if p.arity == current.arity]
            if i == 0:
                cand = [p for p in cand if self._primitive_base_name(p.name) not in self.root_excluded_operator_set]
            if not cand:
                return (individual,)
            for _ in range(64):
                newp = random.choice(cand)
                if newp is current:
                    continue
                ind_clone = toolbox.clone(individual)
                ind_clone[i] = newp
                if _is_valid_individual(ind_clone):
                    return (ind_clone,)
            return (individual,)

        def mutate_mixed(individual):
            pt, pf, pa = self._current_mutation_weights
            r = random.random()
            if r < pt:
                return mutate_terminal(individual)
            elif r < pt + pf:
                return mutate_function(individual)
            else:
                return mutate_tree_root_safe(individual)

        toolbox.register("mutate_tree", mutate_tree_root_safe)
        toolbox.register("mutate_terminal", mutate_terminal)
        toolbox.register("mutate_function", mutate_function)
        toolbox.register("mutate", mutate_mixed)

        # Anti-bloat: **LÍMITE ESTRICTO DE PROFUNDIDAD** en mate y mutate
        def _height_no_id_bin(individual):
            return self._tree_height_no_id_bin(individual)

        toolbox.decorate("mate", gp.staticLimit(key=_height_no_id_bin,
                                                max_value=self.max_tree_height))
        toolbox.decorate("mutate", gp.staticLimit(key=_height_no_id_bin,
                                                  max_value=self.max_tree_height))

        return toolbox

    def _cv_metrics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_gen: np.ndarray,
        seed_offset: int = 0,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evalúa la métrica vía CV con SEED que cambia por generación si alter_random_state=True.
        Devuelve (score medio, {modelo: score}).
        """
        X_aug = np.column_stack([X, X_gen])
        seed = self.random_state + seed_offset if self.alter_random_state else self.random_state
        kf = KFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=seed,
        )
        mean_mse, per_model = evaluate_models_cv(X_aug, y, kf, self.model_configs, self.metric)
        return mean_mse, per_model

    def _is_duplicate(self, vec: np.ndarray, selected: List[np.ndarray]) -> bool:
        """True si vec es (casi) igual o colineal con alguno en selected."""
        if not selected:
            return False
        threshold = self.duplicate_corr_threshold
        for v in selected:
            if np.allclose(vec, v, rtol=1e-8, atol=1e-10) or np.allclose(vec, -v, rtol=1e-8, atol=1e-10):
                return True
            sv, ss = float(np.std(v)), float(np.std(vec))
            if sv < 1e-12 or ss < 1e-12:
                if np.allclose(vec, v, rtol=1e-8, atol=1e-10):
                    return True
            else:
                corr = np.corrcoef(v, vec)
                if corr.shape == (2, 2):
                        r = float(corr[0, 1])
                        if np.isfinite(r) and abs(r) >= threshold:
                            return True
        return False

    def _has_unary_on_ephemeral(self, tree: gp.PrimitiveTree) -> bool:
        """Devuelve True si algún operador unario aplica sobre una constante efímera numérica."""
        def walk(idx: int) -> tuple[bool, int]:
            node = tree[idx]
            idx += 1
            if isinstance(node, gp.Primitive):
                if node.arity == 1 and idx < len(tree):
                    child = tree[idx]
                    if isinstance(child, gp.Terminal):
                        val = getattr(child, "value", None)
                        if isinstance(val, (int, float, np.integer, np.floating)):
                            return True, idx + 1
                for _ in range(node.arity):
                    found, idx = walk(idx)
                    if found:
                        return True, idx
            return False, idx

        found, _ = walk(0)
        return found

    @staticmethod
    def _terminal_var_index(node: gp.Terminal) -> Optional[int]:
        val = getattr(node, "value", None)
        if isinstance(val, str):
            if val.startswith("x") and val[1:].isdigit():
                return int(val[1:])
            if val.startswith("ARG") and val[3:].isdigit():
                return int(val[3:])
        name = getattr(node, "name", None)
        if isinstance(name, str):
            if name.startswith("x") and name[1:].isdigit():
                return int(name[1:])
            if name.startswith("ARG") and name[3:].isdigit():
                return int(name[3:])
        return None

    def _has_binary_same_variable(self, tree: gp.PrimitiveTree) -> bool:
        """True si algun operador binario tiene la misma variable en ambos argumentos inmediatos."""
        def walk(idx: int) -> tuple[bool, int]:
            node = tree[idx]
            idx += 1
            if isinstance(node, gp.Primitive):
                if node.arity == 2:
                    left_idx = idx
                    found, idx = walk(idx)
                    if found:
                        return True, idx
                    right_idx = idx
                    if right_idx < len(tree):
                        left_node = tree[left_idx]
                        right_node = tree[right_idx]
                        if isinstance(left_node, gp.Terminal) and isinstance(right_node, gp.Terminal):
                            left_var = self._terminal_var_index(left_node)
                            right_var = self._terminal_var_index(right_node)
                            if left_var is not None and right_var is not None and left_var == right_var:
                                return True, idx
                    found, idx = walk(idx)
                    if found:
                        return True, idx
                    return False, idx
                for _ in range(node.arity):
                    found, idx = walk(idx)
                    if found:
                        return True, idx
            return False, idx

        found, _ = walk(0)
        return found

    def _is_id_bin_node(self, node: gp.Primitive) -> bool:
        return isinstance(node, gp.Primitive) and self._primitive_base_name(node.name) == "id_bin"

    def _tree_height_no_id_bin(self, tree: gp.PrimitiveTree) -> int:
        def walk(idx: int) -> tuple[int, int]:
            node = tree[idx]
            idx += 1
            if isinstance(node, gp.Primitive):
                max_child = 0
                for _ in range(node.arity):
                    h, idx = walk(idx)
                    if h > max_child:
                        max_child = h
                if self._is_id_bin_node(node):
                    return max_child, idx
                return 1 + max_child, idx
            return 0, idx

        height, _ = walk(0)
        return int(height)

    def _tree_size_no_id_bin(self, tree: gp.PrimitiveTree) -> int:
        def walk(idx: int) -> tuple[int, int]:
            node = tree[idx]
            idx += 1
            if isinstance(node, gp.Primitive):
                total = 0
                for _ in range(node.arity):
                    s, idx = walk(idx)
                    total += s
                if self._is_id_bin_node(node):
                    return total, idx
                return 1 + total, idx
            return 1, idx

        size, _ = walk(0)
        return int(size)

    # ---------------- Evolución ----------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPFeatureGenerator":
        n_samples, n_vars = X.shape

        if self.verbose:
            print(f"[GP] ===== INICIO =====\n")
            if self.penalty_mode and self.penalty_coeff > 0.0:
                print(f"[GP] Penalización activa -> modo={self.penalty_mode} coef={self.penalty_coeff}")
            else:
                print("[GP] Penalización desactivada")

        # Comprobación extra de tamaños por si modifican atributos tras __init__
        if (self.offspring_size + self.elitism_size) < self.population_size:
            raise ValueError(
                "[GP] Config inválida en fit(): offspring_size + elitism_size "
                f"({self.offspring_size + self.elitism_size}) < population_size "
                f"({self.population_size})."
            )

        # Siembra reproducible para 'random' (afecta a ephemerals y mutaciones)
        random.seed(self.random_state)
        self._const_rng.seed(self.random_state)

        pset = self._build_pset(n_vars)
        toolbox = self._build_toolbox(pset)
        pop = toolbox.population(n=self.population_size)

        archive: Dict[str, Dict[str, Any]] = {}
        eval_cache: Optional[Dict[str, Dict[str, Any]]] = {} if not self.alter_random_state else None

        max_repair_attempts = 10

        def _is_invalid(individual) -> bool:
            return self._has_unary_on_ephemeral(individual) or self._has_binary_same_variable(individual)

        def _repair_individual(individual) -> bool:
            for _ in range(max_repair_attempts):
                new_ind = toolbox.individual()
                if not _is_invalid(new_ind):
                    individual[:] = new_ind
                    return True
            return False

        # === evaluación de un individuo con seed dependiente de generación ===
        def eval_ind(ind, gen_offset: int):
            retries = max_repair_attempts
            while True:
                if _is_invalid(ind):
                    if not _repair_individual(ind):
                        ind.model_mses = {cfg.name: BAD_FITNESS for cfg in self.model_configs}
                        ind._penalty = 0.0
                        return (BAD_FITNESS,)
                expr_key = str(ind)
                if eval_cache is not None and expr_key in eval_cache:
                    cached = eval_cache[expr_key]
                    ind.model_mses = dict(cached.get("models", {}))
                    ind._penalty = float(cached.get("penalty", 0.0))
                    return (float(cached["mean"]),)
                func = toolbox.compile(expr=ind)
                try:
                    X_gen = func(*[X[:, j] for j in range(n_vars)])
                except Exception:
                    if retries <= 0 or not _repair_individual(ind):
                        ind.model_mses = {cfg.name: BAD_FITNESS for cfg in self.model_configs}
                        ind._penalty = 0.0
                        return (BAD_FITNESS,)
                    retries -= 1
                    continue
                X_gen = np.asarray(X_gen, dtype=float)
                if X_gen.shape != (n_samples,):
                    if retries <= 0 or not _repair_individual(ind):
                        ind.model_mses = {cfg.name: BAD_FITNESS for cfg in self.model_configs}
                        ind._penalty = 0.0
                        return (BAD_FITNESS,)
                    retries -= 1
                    continue
                X_gen = np.nan_to_num(X_gen, nan=0.0, posinf=1e6, neginf=-1e6)
                mse, model_mses = self._cv_metrics(X, y, X_gen, seed_offset=gen_offset)
                ind.model_mses = model_mses
                penalty = 0.0
                if self.penalty_mode and self.penalty_coeff > 0.0:
                    if self.penalty_mode == "depth":
                        measure = float(self._tree_height_no_id_bin(ind))
                    else:
                        measure = float(self._tree_size_no_id_bin(ind))
                    penalty = self.penalty_coeff * measure
                    mse += penalty
                ind._penalty = penalty  # solo informativo
                if eval_cache is not None:
                    eval_cache[expr_key] = {
                        "mean": mse,
                        "models": dict(model_mses),
                        "penalty": float(penalty),
                    }
                return (mse,)

        def _safe_vector(ind) -> Optional[np.ndarray]:
            func = toolbox.compile(expr=ind)
            try:
                vec = func(*[X[:, j] for j in range(n_vars)])
            except Exception:
                return None
            vec = np.nan_to_num(np.asarray(vec), nan=0.0, posinf=1e6, neginf=-1e6)
            vec = np.asarray(vec).reshape(-1)
            if vec.shape[0] != n_samples:
                return None
            return vec

        # ---- Evaluación inicial (Gen 0) ----
        if self.verbose:
            print("[GP] ---- Generación 00 ----")
        for ind in pop:
            ind.fitness.values = eval_ind(ind, 0)
            expr = str(ind)
            mse = ind.fitness.values[0]
            if (expr not in archive) or (mse < archive[expr]["mean"]):
                vec = _safe_vector(ind)
                if vec is not None:
                    archive[expr] = {
                        "mean": mse,
                        "vector": vec,
                        "models": dict(getattr(ind, "model_mses", {})),
                        "tree": toolbox.clone(ind),
                        "penalty": float(getattr(ind, "_penalty", 0.0)),
                    }

        # Métricas Gen 0 (mismo formato que el resto)
        scores0 = np.array([ind.fitness.values[0] for ind in pop], dtype=float)
        best0 = float(np.min(scores0))
        mean0 = float(np.mean(scores0))
        std0  = float(np.std(scores0))
        if self.verbose:
            best0_disp = self._disp(best0)
            mean0_disp = self._disp(mean0)
            print(f"[GP] Gen 00 | Mejor={best0_disp:.6f} | Media={mean0_disp:.6f} | STD={std0:.6f}")
            k = max(1, self.num_new_features)
            top_idx = np.argsort(scores0)[:k]
            print("[GP]  Top-k de la generación:")
            for rank, idx in enumerate(top_idx, 1):
                e = str(pop[idx])
                model_detail = getattr(pop[idx], "model_mses", {})
                detail_text = ", ".join(
                    f"{m}={self._disp(model_detail[m]):.6f}" for m in sorted(model_detail)
                )
                pen = float(getattr(pop[idx], "_penalty", 0.0))
                pen_txt = f" | pen={pen:.4f}" if pen else ""
                score_disp = self._disp(scores0[idx])
                print(
                    f"      #{rank}: {self.metric.name}={score_disp:.6f}{pen_txt} | expr={self._expr_pretty(e)}"
                )
                if detail_text:
                    print(f"         modelos -> {detail_text}")
            ranked_global = sorted(archive.items(), key=lambda t: t[1]["mean"])[:k]
            print("[GP]  Top-k global (hasta ahora):")
            for rank, (e, info) in enumerate(ranked_global, 1):
                detail_text = ", ".join(
                    f"{m}={self._disp(info['models'][m]):.6f}" for m in sorted(info["models"])
                )
                pen = float(info.get("penalty", 0.0))
                pen_txt = f" | pen={pen:.4f}" if pen else ""
                score_disp = self._disp(info["mean"])
                print(
                    f"      #{rank}: {self.metric.name}={score_disp:.6f}{pen_txt} | expr={self._expr_pretty(e)}"
                )
                if detail_text:
                    print(f"         modelos -> {detail_text}")
            print()

        best_global = float(np.min(scores0))
        no_improve_gens = 0
        start_time = time.time()
        gen = 0

        # =================== Bucle evolutivo ===================
        while True:
            # --------- Paradas (con motivo) ---------
            if self.num_generations is not None and gen >= int(self.num_generations):
                self.stop_reason_ = "número de generaciones"
                self.stop_detail_ = f"se alcanzó el máximo de {self.num_generations} generaciones"
                if self.verbose:
                    print(f"[GP] Parada por número de generaciones: {self.stop_detail_}.\n")
                break

            if self.maxtime is not None and (time.time() - start_time) >= float(self.maxtime):
                elapsed = time.time() - start_time
                self.stop_reason_ = "tiempo"
                self.stop_detail_ = f"{elapsed:.2f}s >= {self.maxtime}s"
                if self.verbose:
                    print(f"[GP] Parada por tiempo: {self.stop_detail_}.\n")
                break

            gen += 1

            # ======= RNG coherente por generación =======
            if self.alter_random_state:
                self.random_state += 1
                random.seed(self.random_state)
                self._const_rng.seed(self.random_state)

            # ---- SCHEDULES ----
            frac = self._schedule_fraction(gen_index_zero_based=gen - 1)

            self._current_mutation_weights = (
                self._blend_tuple_by_fraction(self.mutation_weights, self.mutation_weights_end, frac)
                if self.evolve_mutation_weights else self.mutation_weights
            )
            self._current_mutation_probability = (
                self._blend_scalar_by_fraction(self.mutation_probability, self.mutation_probability_end, frac)
                if self.evolve_mutation_probability else self.mutation_probability
            )

            if self.verbose:
                pt, pf, pa = self._current_mutation_weights
                print(f"[GP] ---- Generación {gen:02d} ----")

            # --------- ELITISMO ---------
            elites: List[creator.Individual] = []
            if self.elitism_size > 0:
                pop_sorted = sorted(pop, key=lambda ind: ind.fitness.values[0])
                elites = [toolbox.clone(ind) for ind in pop_sorted[:self.elitism_size]]

            # --------- DESCENDENCIA ---------
            parents = toolbox.select(pop, self.offspring_size)
            offspring = list(map(toolbox.clone, parents))

            # Cruce
            for i in range(1, len(offspring), 2):
                if random.random() < self.crossover_probability:
                    offspring[i-1], offspring[i] = toolbox.mate(offspring[i-1], offspring[i])
                    if hasattr(offspring[i-1].fitness, "values"):
                        try: del offspring[i-1].fitness.values
                        except Exception: pass
                    if hasattr(offspring[i].fitness, "values"):
                        try: del offspring[i].fitness.values
                        except Exception: pass

            # Mutación (probabilidad dinámica)
            for i in range(len(offspring)):
                if random.random() < self._current_mutation_probability:
                    offspring[i], = toolbox.mutate(offspring[i])
                    if hasattr(offspring[i].fitness, "values"):
                        try: del offspring[i].fitness.values
                        except Exception: pass

            # Evaluar descendencia con la seed de esta generación
            improved_this_gen = False
            for i in range(len(offspring)):
                offspring[i].fitness.values = eval_ind(offspring[i], gen)
                expr = str(offspring[i])
                mse = offspring[i].fitness.values[0]
                if (expr not in archive) or (mse < archive[expr]["mean"]):
                    vec = _safe_vector(offspring[i])
                    if vec is not None:
                        archive[expr] = {
                            "mean": mse,
                            "vector": vec,
                            "models": dict(getattr(offspring[i], "model_mses", {})),
                            "tree": toolbox.clone(offspring[i]),
                            "penalty": float(getattr(offspring[i], "_penalty", 0.0)),
                        }
                if mse < best_global:
                    best_global = mse
                    improved_this_gen = True

            # Evaluar élites con la CV de esta generación
            for ind in elites:
                ind.fitness.values = eval_ind(ind, gen)
                expr = str(ind)
                mse = ind.fitness.values[0]
                if (expr not in archive) or (mse < archive[expr]["mean"]):
                    vec = _safe_vector(ind)
                    if vec is not None:
                        archive[expr] = {
                            "mean": mse,
                            "vector": vec,
                            "models": dict(getattr(ind, "model_mses", {})),
                            "tree": toolbox.clone(ind),
                            "penalty": float(getattr(ind, "_penalty", 0.0)),
                        }
                if mse < best_global:
                    best_global = mse
                    improved_this_gen = True

            # --------- Supervivencia ---------
            replace_size = self._replacement_size  # = population_size - elitism_size
            offspring_sorted = sorted(offspring, key=lambda ind: ind.fitness.values[0])
            selected_offspring = offspring_sorted[:min(replace_size, len(offspring_sorted))]

            # (Relleno de huecos solo por robustez)
            if len(selected_offspring) < replace_size:
                missing = replace_size - len(selected_offspring)
                prev_non_elites = sorted(pop, key=lambda ind: ind.fitness.values[0])[self.elitism_size:]
                extra = prev_non_elites[:missing]
                selected_offspring += extra
                for ind in extra:
                    ind.fitness.values = eval_ind(ind, gen)
                    expr = str(ind)
                    mse = ind.fitness.values[0]
                    if (expr not in archive) or (mse < archive[expr]["mean"]):
                        vec = _safe_vector(ind)
                        if vec is not None:
                            archive[expr] = {
                                "mean": mse,
                                "vector": vec,
                                "models": dict(getattr(ind, "model_mses", {})),
                                "tree": toolbox.clone(ind),
                                "penalty": float(getattr(ind, "_penalty", 0.0)),
                            }
                    if mse < best_global:
                        best_global = mse
                        improved_this_gen = True

            # Nueva población (élites + mejores hijos que caben)
            pop = elites + selected_offspring

            # ======= Métricas de la generación (población actual) =======
            scores_gen = np.array([ind.fitness.values[0] for ind in pop], dtype=float)
            best_gen = float(np.min(scores_gen))
            mean_gen = float(np.mean(scores_gen))
            std_gen  = float(np.std(scores_gen))

            if self.verbose:
                best_gen_disp = self._disp(best_gen)
                best_global_disp = self._disp(best_global)
                mean_gen_disp = self._disp(mean_gen)
                print(
                    f"[GP] Gen {gen:02d} | Mejor={best_gen_disp:.6f} | Mejor global={best_global_disp:.6f} "
                    f"| Media={mean_gen_disp:.6f} | STD={std_gen:.6f}"
                )

                # Top-k de la generación
                k = max(1, self.num_new_features)
                top_idx = np.argsort(scores_gen)[:k]
                print("[GP]  Top-k de la generación:")
                for rank, idx in enumerate(top_idx, 1):
                    e = str(pop[idx])
                    model_detail = getattr(pop[idx], "model_mses", {})
                    detail_text = ", ".join(
                        f"{m}={self._disp(model_detail[m]):.6f}" for m in sorted(model_detail)
                    )
                    pen = float(getattr(pop[idx], "_penalty", 0.0))
                    pen_txt = f" | pen={pen:.4f}" if pen else ""
                    score_disp = self._disp(scores_gen[idx])
                    print(
                        f"      #{rank}: {self.metric.name}={score_disp:.6f}{pen_txt} | expr={self._expr_pretty(e)}"
                    )
                    if detail_text:
                        print(f"         modelos -> {detail_text}")

                # Top-k global (archivo)
                ranked_global = sorted(archive.items(), key=lambda t: t[1]["mean"])[:k]
                print("[GP]  Top-k global (hasta ahora):")
                for rank, (e, info) in enumerate(ranked_global, 1):
                    detail_text = ", ".join(
                        f"{m}={self._disp(info['models'][m]):.6f}" for m in sorted(info["models"])
                    )
                    pen = float(info.get("penalty", 0.0))
                    pen_txt = f" | pen={pen:.4f}" if pen else ""
                    score_disp = self._disp(info["mean"])
                    print(
                        f"      #{rank}: {self.metric.name}={score_disp:.6f}{pen_txt} | expr={self._expr_pretty(e)}"
                    )
                    if detail_text:
                        print(f"         modelos -> {detail_text}")
                print()

            # --------- Parada por paciencia ---------
            if self.patience and int(self.patience) > 0:
                if improved_this_gen:
                    no_improve_gens = 0
                else:
                    no_improve_gens += 1
                    if no_improve_gens >= int(self.patience):
                        self.stop_reason_ = "paciencia"
                        self.stop_detail_ = f"{no_improve_gens} generaciones sin mejora"
                        if self.verbose:
                            print(f"[GP] Parada por paciencia: {self.stop_detail_}.\n")
                        break

        # --------- Selección final sin duplicados ---------
        if self.verbose:
            print("[GP] Seleccionando top-k del archivo global (sin duplicados)...\n")

        ranked = sorted(archive.items(), key=lambda t: t[1]["mean"])

        selected_exprs: List[str] = []
        selected_mses: List[float] = []
        selected_vecs: List[np.ndarray] = []
        selected_trees: List[gp.PrimitiveTree] = []
        selected_models: List[Dict[str, float]] = []
        selected_penalties: List[float] = []

        for expr, info in ranked:
            if len(selected_exprs) >= self.num_new_features:
                break
            v = np.asarray(info["vector"]).reshape(-1)
            if v.shape[0] != n_samples:
                continue
            if not self._is_duplicate(v, selected_vecs):
                selected_exprs.append(expr)
                selected_mses.append(float(info["mean"]))
                selected_vecs.append(v)
                selected_trees.append(toolbox.clone(info["tree"]))
                selected_models.append(dict(info["models"]))
                selected_penalties.append(float(info.get("penalty", 0.0)))

        k = len(selected_exprs)
        Z = np.column_stack([v.reshape(-1, 1) for v in selected_vecs]) if k > 0 else np.empty((X.shape[0], 0))

        self._best_trees = selected_trees
        self._best_exprs = [self._tree_to_expr(tree) for tree in self._best_trees]
        self._best_mses = selected_mses
        self._best_vectors_ = Z
        self._best_model_mses = selected_models
        self._best_penalties = selected_penalties

        if self.verbose:
            print("[GP] ===== FIN =====")
            if self.stop_reason_ is None:
                self.stop_reason_ = "desconocido"
                self.stop_detail_ = "-"
            print(f"[GP] Motivo de parada: {self.stop_reason_} ({self.stop_detail_})\n")
            print("[GP] Top-k final (archivo, sin duplicados):")
            for i, (expr, m, models, pen) in enumerate(zip(self._best_exprs, self._best_mses, self._best_model_mses, getattr(self, "_best_penalties", [])), 1):
                detail_text = ", ".join(f"{name}={self._disp(models[name]):.6f}" for name in sorted(models))
                pen_txt = f" | pen={pen:.4f}" if pen else ""
                score_disp = self._disp(m)
                print(f"      #{i}: {self.metric.name}={score_disp:.6f}{pen_txt} | expr={self._expr_pretty(expr)}")
                if detail_text:
                    print(f"         modelos -> {detail_text}")
            print()

        return self

    # ---------------- API pública ----------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._best_vectors_ is None:
            raise RuntimeError("Debes llamar a fit() primero.")
        if self.verbose:
            print(f"[GP] transform(): retornando Z con shape={self._best_vectors_.shape}\n")
        return self._best_vectors_

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def best_expressions(self) -> List[str]:
        """Expresiones internas (x0,x1,...) para compatibilidad con EVOPT/transform()."""
        return list(self._best_exprs)

    def best_expression_trees(self) -> List[gp.PrimitiveTree]:
        return [copy.deepcopy(tree) for tree in self._best_trees]

    def best_cv_mse(self) -> List[float]:
        return list(self._best_mses)

    def best_cv_scores(self) -> List[float]:
        return list(self._best_mses)

    def best_model_mses(self) -> List[Dict[str, float]]:
        return [dict(m) for m in self._best_model_mses]

    def evaluate_on(self, X: np.ndarray) -> np.ndarray:
        if not self._best_trees:
            return np.empty((X.shape[0], 0)) if X.size else np.empty((0, 0))
        X = np.asarray(X, dtype=float)
        n_samples, n_vars = X.shape
        outputs: List[np.ndarray] = []
        pset = self._build_pset(n_vars)
        for tree in self._best_trees:
            func = gp.compile(expr=tree, pset=pset)
            vec = func(*[X[:, j] for j in range(n_vars)])
            vec = np.nan_to_num(np.asarray(vec), nan=0.0, posinf=1e6, neginf=-1e6).reshape(-1)
            outputs.append(vec.reshape(-1, 1))
        return np.hstack(outputs) if outputs else np.empty((n_samples, 0))
