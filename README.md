# EVOPT

[![CI](https://github.com/francisco-almena-uc3m/evopt/actions/workflows/ci.yml/badge.svg)](https://github.com/francisco-almena-uc3m/evopt/actions/workflows/ci.yml)

EVOPT is an evolutionary feature optimization library for tabular machine
learning. Its purpose is to improve the input feature space before training a
supervised model.

Many tabular problems contain useful relationships that are not present as
explicit columns: ratios, products, protected logarithms, nonlinear rescalings,
interactions between variables, or compact subsets of the original feature set.
EVOPT searches for these improvements automatically. It uses genetic programming
(GP) to generate candidate mathematical transformations and genetic algorithms
(GA) to select which original and generated features should remain.

The optimizer is evaluated through supervised models and cross-validation. That
means generated features are not kept simply because they look statistically
interesting; they are kept when they help predictive performance under the
chosen task, metric, and model family. The final result is a transformed matrix
that can be passed to downstream machine learning models, plus a report showing
which original variables were kept and which generated expressions were selected.

EVOPT supports regression and classification and follows a scikit-learn-style
workflow: instantiate `EvolutionaryOptimizer`, call `fit` on training data, use
`transform` to obtain the optimized feature matrix, and inspect the final report
when you want to understand which features were selected.

The default configuration is intended to work directly in many tabular
problems. It provides a one-hour evolutionary search budget and a balanced GP +
GA setup. The parameters remain fully configurable when you need a faster run,
a stricter benchmark, or a different search behavior.

The public API is:

```python
from evopt import EvolutionaryOptimizer
```

## Installation

Install EVOPT directly from GitHub:

```bash
pip install "evopt @ git+https://github.com/francisco-almena-uc3m/evopt.git"
```

Or install it from a local clone:

```bash
pip install -e .
```

If you prefer installing dependencies explicitly instead of installing the
package, use the single requirements file:

```bash
pip install -r requirements.txt
```

The examples use XGBoost as one of the evaluation models for comparing results.
If you want to run the examples, install `requirements.txt` so that XGBoost is
available.

After installation, use EVOPT from your own scripts or notebooks with:

```python
from evopt import EvolutionaryOptimizer
```

For development and tests:

```bash
pip install -e ".[dev]"
pytest -q
```

## Applying EVOPT

EVOPT works on numeric matrices. If your dataset has categorical variables,
encode them before calling `fit` (for example with one-hot encoding). A typical
workflow is:

1. Prepare `X_train`, `y_train`, `X_test`, `y_test`.
2. Keep a list of `feature_names` if you want readable reports.
3. Choose `task_type` and optionally `metric_name` and `models`.
4. Fit EVOPT on the training data.
5. Transform train/test matrices.
6. Train or evaluate downstream models on the transformed matrices.

Minimal example:

```python
from evopt import EvolutionaryOptimizer

optimizer = EvolutionaryOptimizer(
    task_type="regression",
    metric_name="mse",
)

optimizer.fit(X_train, y_train, feature_names=feature_names)

X_train_opt = optimizer.transform(X_train)
X_test_opt = optimizer.transform(X_test)

report = optimizer.report_final_selection(feature_names)
print(report["selected_originals"])
print(report["selected_transformations"])
```

For classification:

```python
optimizer = EvolutionaryOptimizer(
    task_type="classification",
    metric_name="f1",
)
```

EVOPT is not tied to the datasets in this repository. The examples are only
templates showing how different public datasets can be loaded and prepared.

## Default Configuration

The default configuration is intended to be a practical one-hour experiment.
It was established by testing EVOPT across the datasets included in `examples/`,
with the goal of improving predictive performance while keeping the search
bounded and usable across different tabular settings.

The example suite covers both regression and classification, including binary
classification, multiclass classification, small datasets, medium datasets and
larger tabular datasets. The documented benchmark set ranges approximately from
150 rows and 4 variables (`iris`) to tens of thousands of rows and up to around
82 prepared variables (`online_shoppers`). It includes datasets such as:

- Small datasets: `iris`, `wine`, `breast_cancer`, `titanic`,
  `airfoil_self_noise`, `bike_sharing_day`, `energy_efficiency_cooling`,
  `energy_efficiency_heating`, `student_performance_math`,
  `student_performance_portuguese`.
- Medium datasets: `churn`, `spam`, `magic_telescope`, `online_shoppers`,
  `bike_sharing_hour`, `california`, `concrete_compressive_strength`.
- Larger datasets: `adult`, `credit_default`, `letter`,
  `online_news_popularity`, `superconduct`, `airlines`, `creditcard`, and
  `poker-hand-training-true`.

The defaults are deliberately conservative:

- One-hour global time budget: `maxtime=3600`.
- Bounded GP trees: default depth from 1 to 2.
- Cross-validation inside GP and GA.
- Automatic GP/GA population-size estimation.
- Separate data usage for GP, GA and final GA by default.
- A final GA stage to consolidate the retained original and generated features.
- Simple default model families to avoid overfitting the search to a single
  complex estimator.

In many cases, you only need to specify the task and metric. Set `random_state`
only when you need strict reproducibility. Change lower-level GP, GA and final
GA parameters only when you need a faster benchmark, an ablation, or a different
search behavior.

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

    # Data splitting and automatic population sizing
    split_data_between_gp_and_ga=False,
    split_data_between_gp_and_ga_and_final_ga=True,
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

    # Final GA
    run_final_ga=True,
    final_ga_maxtime=600,
    final_ga_population_size=None,
    final_ga_num_generations=10,
    final_ga_patience=5,
    final_ga_retune_models=False,
    sticky_selected_features=False,
)
```

For a five-minute smoke check, use:

```python
EvolutionaryOptimizer(
    task_type="regression",
    maxtime=300,
    num_iterations=3,
    estimate_pop_size=False,
    gp_population_size=20,
    ga_population_size=20,
    gp_num_generations=5,
    ga_num_generations=5,
)
```

## Models

When `models=None`, EVOPT uses two default model families during the
evolutionary search:

```python
{
    "regression": ["elastic_net", "decision_tree_regressor"],
    "classification": ["logistic_regression", "decision_tree_classifier"],
}
```

These defaults were chosen to combine a simple linear model with a simple
nonlinear tree model. The intent is to reward generated/selected features that
are useful under different inductive biases without making the search too slow.

The experiment runner evaluates the final result with a broader set of models.
Those evaluation models are not the default search models; they are used to
compare original data versus transformed data more broadly.

Regression evaluation models:

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

Classification evaluation models:

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

The `xgb_regressor` and `xgb_classifier` options require XGBoost.

## Examples

The files in `examples/` are runnable examples, not required entry points. They
show how to prepare different public datasets and call `EvolutionaryOptimizer`.
Use them as templates for your own datasets.

Run one example at a time:

```bash
python examples/quick_smoke.py
python examples/california.py
python examples/iris.py
python examples/titanic.py
```

The shared runner can also execute any supported dataset key without editing
the file:

```bash
python examples/experiment_runner.py california
python examples/experiment_runner.py california iris --maxtime 300
python examples/experiment_runner.py --list-datasets
```

Available dataset scripts:

```text
quick_smoke
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
default GP/GA/final-GA configuration unless you pass `opt_kwargs` to
`run_dataset`. Place snippets like the following inside `examples/`, or adapt
the import path in your own project:

```python
from run_dataset import run_dataset

run_dataset(
    "california",
    opt_kwargs={
        "verbose": True,
    },
)
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

## Optimizer Parameters

### Core

- `task_type`: `"regression"` or `"classification"`.
- `metric_name`: metric optimized internally. Regression: `"mse"`, `"mae"`.
  Classification: `"f1"`, `"accuracy"`, `"auc"`.
- `models`: model names used to evaluate candidate feature sets. If omitted,
  EVOPT uses the default search models described above.
- `maxtime`: global time budget in seconds.
- `num_iterations`: number of GP + GA cycles.
- `random_state`: seed for reproducibility.
- `alter_random_state`: updates the seed between iterations when `True`.
- `verbose`: prints progress.
- `standardize`: standardizes non-binary columns inside model pipelines.
- `model_tuning_cv_folds`: CV folds for model tuning. Use `None` to skip tuning
  where supported by the current code path.

### Data Splitting and Population Estimation

- `split_data_between_gp_and_ga`: uses separate data splits for GP and GA.
- `split_data_between_gp_and_ga_and_final_ga`: uses separate splits for GP, GA,
  and final GA. Enabled by default.
- `estimate_pop_size`: estimates GP/GA population sizes automatically.
- `estimate_pop_seconds`: time budget used for population-size estimation.

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

### Final Genetic Algorithm

- `run_final_ga`: runs an extended final GA over the retained features.
- `final_ga_maxtime`: reserved time for final GA.
- `final_ga_population_size`: population size for final GA. If `None`, EVOPT
  estimates or derives it.
- `final_ga_num_generations`: final GA generation limit.
- `final_ga_patience`: final GA early-stopping patience.
- `final_ga_retune_models`: retunes models before final GA.
- `sticky_selected_features`: keeps selected GP features across iterations
  before the final selection stage.

## Project Layout

- `evopt/`: installable Python package.
- `evopt/evopt.py`: high-level evolutionary optimizer.
- `evopt/model_utils.py`: model configuration, tuning, metrics, and evaluation.
- `evopt/ga_feature_selection/`: GA feature selection internals.
- `evopt/gp_feature_generation/`: GP feature generation internals.
- `examples/`: dataset preparation and runnable examples.
- `pyproject.toml`: package metadata.
- `requirements.txt`: dependency list for the library and example scripts.
- `tests/`: minimal regression smoke tests for the public API.
- `.github/workflows/ci.yml`: GitHub Actions test workflow.

## License

This project is distributed under the MIT License. See `LICENSE`.

## Notes

- Some example datasets are downloaded from OpenML or external URLs.
- Large datasets and long configurations can take a long time.
- Generated caches, local environments, and experiment outputs are ignored by
  Git through `.gitignore`.
