# EVOPT

EVOPT is an evolutionary feature optimization library for tabular machine
learning. It combines genetic programming (GP) for feature generation with
genetic algorithms (GA) for feature selection.

The public API is:

```python
from evopt import EvolutionaryOptimizer
```

## Installation

Install the project in editable mode from the repository root:

```bash
pip install -e .
```

Alternatively, install only the dependencies:

```bash
pip install -r requirements.txt
```

## Basic Usage

```python
from evopt import EvolutionaryOptimizer

optimizer = EvolutionaryOptimizer(
    task_type="regression",
    metric_name="mse",
    models=["elastic_net", "decision_tree_regressor"],
    random_state=1,
)

optimizer.fit(X_train, y_train, feature_names=feature_names)

X_train_opt = optimizer.transform(X_train)
X_test_opt = optimizer.transform(X_test)

report = optimizer.report_final_selection(feature_names)
print(report["selected_originals"])
print(report["selected_transformations"])
```

Inputs should be numeric arrays. Encode categorical variables before calling
`fit`. EVOPT is not tied to the datasets in this repository; it can be applied
to any tabular dataset once you have `X_train`, `y_train`, `X_test`, `y_test`,
and optional `feature_names`.

## Default Configuration

EVOPT is designed to be useful without tuning every evolutionary parameter. The
default configuration is intentionally conservative: it uses simple, robust
model families, bounded GP trees, cross-validation inside GP and GA, automatic
population-size estimation, separate data splits for GP/GA/final GA, and a final
GA pass to consolidate the selected feature set.

In many cases, you only need to specify the task, metric, time budget, random
seed, and optionally the model list. The lower-level GP and GA parameters can be
left at their defaults until you have a specific reason to change them.

The default optimizer is equivalent to:

```python
EvolutionaryOptimizer(
    # Core
    maxtime=3600,
    num_iterations=100,
    random_state=0,
    alter_random_state=True,
    verbose=False,
    task_type="regression",
    metric_name=None,
    models=None,
    standardize=True,
    model_tuning_cv_folds=5,

    # Data splitting and final selection
    split_data_between_gp_and_ga=False,
    split_data_between_gp_and_ga_and_final_ga=True,
    run_final_ga=True,
    final_ga_maxtime=600,
    final_ga_population_size=None,
    final_ga_num_generations=10,
    final_ga_patience=5,
    final_ga_retune_models=False,
    sticky_selected_features=False,

    # Automatic population sizing
    estimate_pop_size=True,
    estimate_pop_seconds=25,

    # GP
    gp_population_size=None,
    gp_offspring_size=0.95,
    gp_elitism_size=0.05,
    gp_tournament_size=3,
    gp_init_population_method="grow",
    gp_num_generations=10,
    gp_maxtime=300,
    gp_patience=2,
    gp_cv_folds=3,
    gp_num_new_features=None,
    gp_min_tree_height=1,
    gp_max_tree_height=2,
    gp_unary_continuous_operators=("sqrt", "square", "inv", "log"),
    gp_unary_binary_operators=(),
    gp_binary_continuous_operators=("add", "sub", "mul", "div", "max", "min"),
    gp_binary_binary_operators=("max", "min"),
    gp_binary_mixed_operators=("mul",),
    gp_root_excluded_operator_set=("add", "sub"),
    gp_ephemeral_constants=[-3, -2, -1.5, -1, -0.75, -0.5, -0.25,
                            0, 0.25, 0.5, 1, 1.5, 2, 3],
    gp_ephemeral_constants_range=None,
    gp_crossover_probability=1.0,
    gp_mutation_probability=0.25,
    gp_mutation_weights=(0.25, 0.25, 0.50),
    gp_evolve_mutation_weights=True,
    gp_mutation_weights_end=(0.45, 0.45, 0.10),
    gp_evolve_mutation_probability=True,
    gp_mutation_probability_end=0.05,
    gp_duplicate_corr_threshold=0.7,
    gp_penalty_mode=None,
    gp_penalty_coeff=0.0,

    # GA
    ga_population_size=None,
    ga_offspring_size=0.95,
    ga_elitism_size=0.05,
    ga_tournament_size=3,
    ga_num_generations=10,
    ga_maxtime=300,
    ga_patience=2,
    ga_cv_folds=3,
    ga_individual_all_features=True,
    ga_initial_bit_prob=0.5,
    ga_mutation_probability=0.25,
    ga_evolve_mutation_probability=True,
    ga_mutation_probability_end=0.05,
    ga_crossover_type="single",
)
```

When `models=None`, EVOPT chooses:

```python
{
    "regression": ["elastic_net", "decision_tree_regressor"],
    "classification": ["logistic_regression", "decision_tree_classifier"],
}
```

## Examples

The files in `examples/` are runnable examples, not required entry points. They
show how to prepare different public datasets and call `EvolutionaryOptimizer`.
Use them as templates for your own datasets.

Run one example at a time:

```bash
python examples/california.py
python examples/iris.py
python examples/titanic.py
```

Available dataset scripts:

```text
adult
airlines
airfoil_self_noise
bike_sharing_day
bike_sharing_hour
breast_cancer
california
churn
concrete_compressive_strength
credit_default
creditcard
energy_efficiency_cooling
energy_efficiency_heating
iris
letter
magic_telescope
online_news_popularity
online_shoppers
poker_hand_training_true
spam
student_performance_math
student_performance_portuguese
superconduct
titanic
wine
```

The example files only choose the dataset. The actual optimization keeps the
default GP/GA configuration unless you pass `opt_kwargs` to `run_dataset`.
Place snippets like the following inside `examples/`, or adapt the import path
in your own project.

For example, this is enough to run the default one-hour configuration on a
regression dataset:

```python
from run_dataset import run_dataset

run_dataset(
    "california",
    opt_kwargs={
        "maxtime": 3600,
        "random_state": 1,
        "verbose": True,
    },
)
```

For a classification dataset:

```python
from run_dataset import run_dataset

run_dataset(
    "iris",
    opt_kwargs={
        "maxtime": 3600,
        "metric_name": "f1",
        "random_state": 1,
        "verbose": True,
    },
)
```

The same pattern applies to your own data: prepare train/test arrays, choose the
task and metric, and leave the GP/GA internals at their defaults unless you are
running an ablation or a very constrained benchmark.

## Optimizer Parameters

### Core

- `task_type`: `"regression"` or `"classification"`.
- `metric_name`: metric optimized internally. Regression: `"mse"`, `"mae"`.
  Classification: `"f1"`, `"accuracy"`, `"auc"`.
- `models`: model names used to evaluate candidate feature sets. If omitted,
  defaults are `["elastic_net", "decision_tree_regressor"]` for regression and
  `["logistic_regression", "decision_tree_classifier"]` for classification.
- `maxtime`: global time budget in seconds.
- `num_iterations`: number of GP + GA cycles.
- `random_state`: seed for reproducibility.
- `alter_random_state`: updates the seed between iterations when `True`.
- `verbose`: prints progress.
- `standardize`: standardizes non-binary columns inside model pipelines.
- `model_tuning_cv_folds`: CV folds for model tuning. Use `None` to skip tuning
  where supported by the current code path.

### Data Splitting

- `split_data_between_gp_and_ga`: uses separate data splits for GP and GA.
- `split_data_between_gp_and_ga_and_final_ga`: uses separate splits for GP, GA,
  and final GA. Enabled by default.
- `run_final_ga`: runs an extended final GA over the retained features.
- `final_ga_maxtime`: reserved time for final GA.
- `final_ga_population_size`: population size for final GA. If `None`, EVOPT
  estimates or derives it.
- `final_ga_num_generations`: final GA generation limit.
- `final_ga_patience`: final GA early-stopping patience.
- `final_ga_retune_models`: retunes models before final GA.
- `sticky_selected_features`: keeps selected GP features across iterations.

### Population Estimation

- `estimate_pop_size`: estimates GP/GA population sizes automatically.
- `estimate_pop_seconds`: time budget used for population-size estimation.
- `gp_population_size`: manual GP population size.
- `ga_population_size`: manual GA population size.

If `estimate_pop_size=True`, manual GP/GA population sizes can be ignored.

### Genetic Programming

- `gp_num_generations`: GP generation limit.
- `gp_maxtime`: GP time budget per iteration.
- `gp_patience`: GP early-stopping patience.
- `gp_cv_folds`: CV folds used when scoring GP individuals.
- `gp_num_new_features`: number of generated features to keep per GP run. If
  `None`, EVOPT derives it from the feature count.
- `gp_min_tree_height`, `gp_max_tree_height`: expression tree depth bounds.
- `gp_population_size`: GP population size.
- `gp_offspring_size`: offspring proportion or count.
- `gp_elitism_size`: elite proportion or count.
- `gp_tournament_size`: tournament size.
- `gp_init_population_method`: initial population method, for example `"grow"`.
- `gp_crossover_probability`: crossover probability.
- `gp_mutation_probability`: initial mutation probability.
- `gp_evolve_mutation_probability`: interpolates mutation probability across
  generations.
- `gp_mutation_probability_end`: final mutation probability.
- `gp_mutation_weights`: mutation operator weights at the start.
- `gp_evolve_mutation_weights`: interpolates mutation weights.
- `gp_mutation_weights_end`: mutation operator weights at the end.
- `gp_duplicate_corr_threshold`: correlation threshold for removing duplicate
  generated features.
- `gp_penalty_mode`: optional complexity penalty mode.
- `gp_penalty_coeff`: complexity penalty coefficient.

### GP Operators

Available operators are:

```text
add, sub, mul, div, exp, log, pow, square, cube, sqrt, inv,
sin, cos, tan, relu, abs, id_bin, max, min
```

You can control which operators are available by type:

- `gp_unary_continuous_operators`
- `gp_unary_binary_operators`
- `gp_binary_continuous_operators`
- `gp_binary_binary_operators`
- `gp_binary_mixed_operators`
- `gp_root_excluded_operator_set`

Constants can be controlled with:

- `gp_ephemeral_constants`
- `gp_ephemeral_constants_range`

### Genetic Algorithm

- `ga_num_generations`: GA generation limit.
- `ga_maxtime`: GA time budget per iteration.
- `ga_patience`: GA early-stopping patience.
- `ga_cv_folds`: CV folds used when scoring GA individuals.
- `ga_population_size`: GA population size.
- `ga_offspring_size`: offspring proportion or count.
- `ga_elitism_size`: elite proportion or count.
- `ga_tournament_size`: tournament size.
- `ga_individual_all_features`: includes an all-features individual in the
  initial population.
- `ga_initial_bit_prob`: probability that each feature starts selected.
- `ga_mutation_probability`: initial mutation probability.
- `ga_evolve_mutation_probability`: interpolates mutation probability across
  generations.
- `ga_mutation_probability_end`: final mutation probability.
- `ga_crossover_type`: crossover strategy, for example `"single"`.

## Supported Models

Regression:

```text
linear_regression
ridge
lasso
elastic_net
knn_regressor
decision_tree_regressor
random_forest_regressor
xgb_regressor
svr
mlp_regressor
```

Classification:

```text
logistic_regression
gaussian_nb
lda
qda
knn_classifier
decision_tree_classifier
random_forest_classifier
xgb_classifier
svc
mlp_classifier
```

## Outputs

After `fit`, use:

- `transform(X)`: returns the selected original features plus generated GP
  features.
- `report_final_selection(feature_names)`: returns selected original feature
  names, generated transformations, raw expressions, dependencies, metric name,
  and final score.
- `history()`: returns iteration history.
- `kept_expressions()`: returns retained GP expressions.

Optional plots:

- `plot_history()`
- `plot_feature_evolution_history(feature_names)`
- `plot_feature_dependency_matrix(feature_names)`

## Recommended Starting Points

The defaults are the recommended starting point for real one-hour experiments:

```python
EvolutionaryOptimizer(
    task_type="regression",
    metric_name="mse",
    random_state=1,
    verbose=True,
)
```

For classification, switch only the task and metric:

```python
EvolutionaryOptimizer(
    task_type="classification",
    metric_name="f1",
    random_state=1,
    verbose=True,
)
```

For quick smoke checks, reduce the time budget and population sizes:

```python
EvolutionaryOptimizer(
    task_type="regression",
    maxtime=120,
    num_iterations=3,
    estimate_pop_size=False,
    gp_population_size=20,
    ga_population_size=20,
    gp_num_generations=5,
    ga_num_generations=5,
)
```

Only start changing GP/GA internals when you need a faster benchmark, an
ablation, or a deliberately different search behavior.

## Project Layout

- `evopt/`: installable Python package.
- `evopt/evopt.py`: high-level evolutionary optimizer.
- `evopt/model_utils.py`: model configuration, tuning, metrics, and evaluation.
- `evopt/ga_feature_selection/`: GA feature selection internals.
- `evopt/gp_feature_generation/`: GP feature generation internals.
- `examples/`: dataset preparation and runnable examples.
- `pyproject.toml`: package metadata.
- `requirements.txt`: dependency list for simple installation.

## Notes

- Some example datasets are downloaded from OpenML or external URLs.
- Large datasets and long configurations can take a long time.
- XGBoost is included because the experiment runner evaluates XGBoost models.
- Generated caches, local environments, and experiment outputs are ignored by
  Git through `.gitignore`.
