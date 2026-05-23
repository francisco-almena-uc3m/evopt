# EVOPT

EVOPT is an evolutionary feature optimization library for tabular machine
learning. It combines genetic programming for feature generation with genetic
algorithms for feature selection.

## Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

## Running Experiments

Configure `CLASSIFICATION_EXPERIMENTS` and `REGRESSION_EXPERIMENTS` in
`main.py`, then run:

```bash
python main.py
```

`main.py` includes the dataset configurations used in the experiments.

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

- `evopt.py`: high-level evolutionary optimizer.
- `ga_feature_selection/`: genetic algorithm feature selector.
- `gp_feature_generation/`: genetic programming feature generator.
- `model_utils.py`: model configuration, tuning, metrics, and evaluation helpers.

## Notes

- `xgboost` is optional at runtime unless you select XGBoost models.
- Generated caches, local environments, and experiment outputs are ignored by
  Git through `.gitignore`.
