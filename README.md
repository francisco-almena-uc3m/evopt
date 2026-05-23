# EVOPT

EVOPT is an evolutionary feature optimization library for tabular machine
learning. It combines genetic programming for feature generation with genetic
algorithms for feature selection.

## Installation

Clone the repository and install it in editable mode:

```bash
pip install -e .
```

Alternatively, install the dependencies directly:

```bash
pip install -r requirements.txt
```

## Running Experiments

Configure `CLASSIFICATION_EXPERIMENTS` and `REGRESSION_EXPERIMENTS` in
`examples/experiment_runner.py`, then run:

```bash
python examples/experiment_runner.py
```

`examples/experiment_runner.py` includes the dataset configurations used in the
experiments.

Each dataset also has its own script in `examples/`:

```bash
python examples/california.py
python examples/iris.py
python examples/titanic.py
```

## Basic Usage

```python
from evopt import EvolutionaryOptimizer

optimizer = EvolutionaryOptimizer(
    task_type="regression",
    metric_name="mse",
    models=["elastic_net", "decision_tree_regressor"],
    maxtime=120,
    num_iterations=3,
)

optimizer.fit(X_train, y_train, feature_names=feature_names)
X_train_opt = optimizer.transform(X_train)
X_test_opt = optimizer.transform(X_test)

report = optimizer.report_final_selection(feature_names)
```

## Project Layout

- `evopt/`: installable Python package.
- `evopt/optimizer.py`: high-level evolutionary optimizer.
- `evopt/ga_feature_selection/`: genetic algorithm feature selector.
- `evopt/gp_feature_generation/`: genetic programming feature generator.
- `evopt/model_utils.py`: model configuration, tuning, metrics, and evaluation helpers.
- `examples/experiment_runner.py`: experiment runner and dataset preparation utilities.
- `examples/`: one executable script per configured dataset.

## Notes

- Generated caches, local environments, and experiment outputs are ignored by
  Git through `.gitignore`.
