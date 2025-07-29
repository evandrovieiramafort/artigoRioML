from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from functools import partial
from datetime import datetime
import optuna
from skopt import BayesSearchCV


def log(mensagem: str, inicio: datetime = None) -> None:
    """
    Registra mensagens de log com marcação de tempo e duração opcional, utilizada nas funções de tuning e validação
    cruzada dos modelos.

    Parâmetros:
        mensagem (str): A mensagem a ser exibida.
        inicio (datetime, opcional): O tempo inicial para calcular a duração.
    """
    fim = datetime.now()
    if inicio:
        duracao = fim - inicio
        horas, resto = divmod(duracao.total_seconds(), 3600)
        minutos, segundos = divmod(resto, 60)
        tempo_formatado = f"{int(horas):02}:{int(minutos):02}:{int(segundos):02}"
        print(f"----- {mensagem} concluído! Tempo de execução: {tempo_formatado} ----- {fim.strftime('%H:%M:%S')}\n")
    else:
        print(f"----- {mensagem} iniciado ----- {fim.strftime('%H:%M:%S')}\n")


def objetivo_skopt(classe_modelo, espaco_busca, X, y, n_iteracoes: int, metrica: str = "accuracy") -> dict:
    """
    Executa a busca bayesiana (BayesSearchCV ou Skopt, a depender do algoritmo) para encontrar os melhores hiperparâmetros.

    Parâmetros:
        classe_modelo: Classe do modelo (ex: RandomForestClassifier).
        espaco_busca (dict): Espaço de busca dos hiperparâmetros.
        X: Conjunto de atributos de treino.
        y: Conjunto de rótulos de treino.
        n_iteracoes (int): Número de iterações da busca.
        metrica (str): Métrica de avaliação (default = "accuracy").

    Retorna:
        dict: Dicionário com os melhores hiperparâmetros encontrados.
    """
    try:
        kfold = KFold(n_splits=3, shuffle=True, random_state=42)
        modelo = classe_modelo()
        busca = BayesSearchCV(
            estimator=modelo,
            search_spaces=espaco_busca,
            n_iter=n_iteracoes,
            cv=kfold,
            scoring=metrica,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        busca.fit(X, y)
        melhores_parametros = dict(busca.best_params_)
        print(f"Melhores Hiperparâmetros (skopt): {melhores_parametros}\n")
        return melhores_parametros
    except Exception as erro:
        print(f"Erro durante a execução do BayesSearchCV: {erro}")
        raise


def objetivo_optuna(tentativa, classe_modelo, nome_modelo: str, espacos_busca_optuna: dict,
                    X_treino, y_treino, X_validacao, y_validacao) -> float:
    """
    Função de objetivo para a otimização com Optuna.

    Parâmetros:
        tentativa: Objeto de tentativa do Optuna.
        classe_modelo: Classe do modelo (ex: RandomForestClassifier).
        nome_modelo (str): Nome do modelo.
        espacos_busca_optuna (dict): Espaços de busca definidos para o Optuna.
        X_treino, y_treino: Dados de treino.
        X_validacao, y_validacao: Dados de validação.

    Retorna:
        float: A acurácia do modelo com os parâmetros testados.
    """
    try:
        parametros = {
            hiperparametro: gerador(tentativa)
            for hiperparametro, gerador in espacos_busca_optuna[nome_modelo].items()
        }
        modelo = classe_modelo(**parametros)
        modelo.fit(X_treino, y_treino)
        y_predito = modelo.predict(X_validacao)
        return accuracy_score(y_validacao, y_predito)
    except Exception as erro:
        print(f"Erro durante a execução do Optuna: {erro}")
        raise


def tunar_modelo(nome_modelo: str,
                 classe_modelo,
                 X_treino,
                 y_treino,
                 n_iteracoes: int,
                 espacos_busca_optuna: dict = None,
                 espaco_busca_skopt: dict = None,
                 metodo: str = "optuna") -> dict:
    """
    Executa a otimização de hiperparâmetros utilizando Optuna ou skopt.

    Parâmetros:
        nome_modelo (str): Nome do modelo.
        classe_modelo: Classe do modelo (ex: RandomForestClassifier).
        X_treino: Conjunto de atributos de treino.
        y_treino: Conjunto de rótulos de treino.
        n_iteracoes (int): Número de iterações ou tentativas da busca.
        espacos_busca_optuna (dict, opcional): Espaço de busca de hiperparâmetros para Optuna.
        espaco_busca_skopt (dict, opcional): Espaço de busca de hiperparâmetros para skopt.
        metodo (str): Método de tuning a ser utilizado ("optuna" ou "skopt").

    Retorna:
        dict: Dicionário com os melhores hiperparâmetros encontrados.
    """
    try:
        inicio = datetime.now()
        log(f"Tuning para {classe_modelo.__name__} ({metodo})")

        if metodo == "optuna":
            if espacos_busca_optuna is None or nome_modelo not in espacos_busca_optuna:
                raise ValueError(f"Hiperparâmetros para '{nome_modelo}' não encontrados no espaço de busca.")

            kfold = KFold(n_splits=3, shuffle=True, random_state=42)
            estudos = []

            optuna.logging.set_verbosity(optuna.logging.WARNING)

            for indice_treino, indice_validacao in kfold.split(X_treino):
                X_tr, X_val = X_treino.iloc[indice_treino], X_treino.iloc[indice_validacao]
                y_tr, y_val = y_treino.iloc[indice_treino], y_treino.iloc[indice_validacao]

                estudo = optuna.create_study(direction="maximize")
                estudo.optimize(partial(objetivo_optuna,
                                        classe_modelo=classe_modelo,
                                        espacos_busca_optuna=espacos_busca_optuna,
                                        nome_modelo=nome_modelo,
                                        X_treino=X_tr,
                                        y_treino=y_tr,
                                        X_validacao=X_val,
                                        y_validacao=y_val),
                                n_trials=n_iteracoes)
                estudos.append(estudo)

            melhor_estudo = max(estudos, key=lambda e: e.best_value)
            melhores_parametros = melhor_estudo.best_params
            print(f"Melhores Hiperparâmetros (optuna): {melhores_parametros}\n")
            log(f"Tuning para {classe_modelo.__name__} ({metodo})", inicio=inicio)
            return melhores_parametros

        elif metodo == "skopt":
            if espaco_busca_skopt is None:
                raise ValueError("Um espaço de busca deve ser fornecido para o método 'skopt'.")

            melhores_parametros = objetivo_skopt(
                classe_modelo=classe_modelo,
                espaco_busca=espaco_busca_skopt,
                X=X_treino,
                y=y_treino,
                n_iteracoes=n_iteracoes
            )
            log(f"Tuning para {classe_modelo.__name__} ({metodo})", inicio=inicio)
            return melhores_parametros

        else:
            raise ValueError("Método de tuning inválido. Escolha entre 'optuna' ou 'skopt'.")

    except ValueError as erro_valor:
        print(f"Erro de valor: {erro_valor}")
        raise
    except Exception as erro_geral:
        print(f"Erro inesperado: {erro_geral}")
        raise
