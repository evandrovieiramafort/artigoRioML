from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

"""
Lista de modelos que serão testados, com:
- nome,
- classe do modelo,
- método de tuning ('optuna' ou 'skopt'),
- parâmetros fixos adicionais (precisei fazer isso para o probability=True do SVC, não consegui pensar em outra forma).
"""

modelos = [
    {
        "nome": "logistic_regression",
        "classe": LogisticRegression,
        "metodo": "skopt",
        "search_space": "search_spaces_skopt",
        "parametros_fixos": {}
    },
    {
        "nome": "decision_tree",
        "classe": DecisionTreeClassifier,
        "metodo": "optuna",
        "search_space": "search_spaces_optuna",
        "parametros_fixos": {}
    },
    {
        "nome": "random_forest",
        "classe": RandomForestClassifier,
        "metodo": "optuna",
        "search_space": "search_spaces_optuna",
        "parametros_fixos": {}
    },
    {
        "nome": "naive_bayes",
        "classe": GaussianNB,
        "metodo": "optuna",
        "search_space": "search_spaces_optuna",
        "parametros_fixos": {}
    },
    {
        "nome": "knn",
        "classe": KNeighborsClassifier,
        "metodo": "optuna",
        "search_space": "search_spaces_optuna",
        "parametros_fixos": {}
    },
    {
        "nome": "svc",
        "classe": SVC,
        "metodo": "optuna",
        "search_space": "search_spaces_optuna",
        "parametros_fixos": {"probability": True}
    },
    {
        "nome": "mlp",
        "classe": MLPClassifier,
        "metodo": "optuna",
        "search_space": "search_spaces_optuna",
        "parametros_fixos": {}
    }
]