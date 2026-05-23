from run_dataset import run_dataset


if __name__ == "__main__":
    run_dataset(
        "california",
        opt_kwargs={
            "maxtime": 300,
            "num_iterations": 3,
            "estimate_pop_size": False,
            "gp_population_size": 20,
            "ga_population_size": 20,
            "gp_num_generations": 5,
            "ga_num_generations": 5,
            "verbose": True,
        },
    )
