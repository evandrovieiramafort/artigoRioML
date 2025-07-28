from skopt.space import Real, Integer, Categorical

"""
Dicionários com espaços de busca de algoritmo para serem usados pelos seus respectivos tunadores de hiperparametros
search_spaces_skopt = Regressão logística
search_spaces_optuna = Todo o restante

(ainda não encontrei uma forma de utilizar o optuna ou o skopt para todos, mas não acredito ser problema - a busca bayesiana funciona do mesmo jeito
nas duas bibliotecas)
"""

search_spaces_optuna = {
    "decision_tree": {
        "criterion": lambda trial: trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "max_depth": lambda trial: trial.suggest_int("max_depth", 1, 100),
        "min_samples_split": lambda trial: trial.suggest_int("min_samples_split", 2, 50),
        "min_samples_leaf": lambda trial: trial.suggest_int("min_samples_leaf", 1, 50),
        "max_features": lambda trial: trial.suggest_categorical("max_features", [None, "sqrt", "log2", 0.5]),
        "ccp_alpha": lambda trial: trial.suggest_float("ccp_alpha", 1e-6, 0.1)
    },

    "random_forest": {
        "n_estimators": lambda trial: trial.suggest_int("n_estimators", 1, 200),
        "criterion": lambda trial: trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "max_depth": lambda trial: trial.suggest_int("max_depth", 1, 100),
        "min_samples_split": lambda trial: trial.suggest_int("min_samples_split", 2, 50),
        "min_samples_leaf": lambda trial: trial.suggest_int("min_samples_leaf", 1, 50),
        "max_features": lambda trial: trial.suggest_categorical("max_features", [None, "sqrt", "log2", 0.5]),
        "bootstrap": lambda trial: trial.suggest_categorical("bootstrap", [True, False]),
        "ccp_alpha": lambda trial: trial.suggest_float("ccp_alpha", 1e-6, 0.1)
    },

    "naive_bayes": {
        "var_smoothing": lambda trial: trial.suggest_float("var_smoothing", 1e-10, 1e-2, log=True)
    },

    "knn": {
        "n_neighbors": lambda trial: trial.suggest_int("n_neighbors", 1, 50),
        "weights": lambda trial: trial.suggest_categorical("weights", ["uniform", "distance"]),
        "metric": lambda trial: trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
        "algorithm": lambda trial: trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
        "leaf_size": lambda trial: trial.suggest_int("leaf_size", 20, 100),
        "p": lambda trial: trial.suggest_categorical("p", [1, 2])
    },

    "svc": {
        "C": lambda trial: trial.suggest_float("C", 1e-6, 1e6, log=True),
        "kernel": lambda trial: trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
        "tol": lambda trial: trial.suggest_float("tol", 1e-6, 1e-2, log=True),
        "max_iter": lambda trial: trial.suggest_int("max_iter", 50, 2000)
    },

    "mlp": {
        "hidden_layer_sizes": lambda trial: trial.suggest_categorical("hidden_layer_sizes", [(50,), (100,), (50, 50), (100, 100), (50, 100), (100, 50)]),
        "activation": lambda trial: trial.suggest_categorical("activation", ["relu", "tanh", "logistic"]),
        "solver": lambda trial: trial.suggest_categorical("solver", ["adam", "sgd"]),
        "alpha": lambda trial: trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        "learning_rate": lambda trial: trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"]),
        "learning_rate_init": lambda trial: trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
        "max_iter": lambda trial: trial.suggest_int("max_iter", 50, 10000),
        "tol": lambda trial: trial.suggest_float("tol", 1e-4, 1e-2, log=True)
    }
}


common_params = {
    "C": Real(1e-6, 1e6, prior="log-uniform"),
    "max_iter": Integer(50, 2000),
    "tol": Real(1e-6, 1e-2, prior="log-uniform"),
    "fit_intercept": Categorical([True, False])
}

search_spaces_skopt = [
    {
        **common_params,
        "penalty": Categorical(["l1"]),
        "solver": Categorical(["liblinear", "saga"])
    },
    {
        **common_params,
        "penalty": Categorical(["l2"]),
        "solver": Categorical(["liblinear", "lbfgs", "saga"])
    },
    {
        **common_params,
        "penalty": Categorical(["elasticnet"]),
        "solver": Categorical(["saga"]),
        "l1_ratio": Real(0, 1)
    },
    {
        **common_params,
        "penalty": Categorical([None]),
        "solver": Categorical(["lbfgs", "saga"])
    }
]