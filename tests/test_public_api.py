import numpy as np

from evopt import EvolutionaryOptimizer


def test_fit_transform_regression_smoke():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(36, 4))
    y = 2.0 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.1, size=36)

    optimizer = EvolutionaryOptimizer(
        task_type="regression",
        metric_name="mse",
        models=["linear_regression"],
        maxtime=30,
        num_iterations=1,
        run_final_ga=False,
        estimate_pop_size=False,
        model_tuning_cv_folds=2,
        gp_population_size=8,
        gp_num_generations=1,
        gp_maxtime=10,
        gp_patience=1,
        gp_cv_folds=2,
        gp_num_new_features=1,
        ga_population_size=8,
        ga_num_generations=1,
        ga_maxtime=10,
        ga_patience=1,
        ga_cv_folds=2,
        random_state=0,
    )

    feature_names = ["x0", "x1", "x2", "x3"]
    optimizer.fit(X, y, feature_names=feature_names)

    X_transformed = optimizer.transform(X)
    report = optimizer.report_final_selection(feature_names)

    assert X_transformed.shape[0] == X.shape[0]
    assert X_transformed.shape[1] >= 1
    assert "selected_originals" in report
    assert "selected_transformations" in report
