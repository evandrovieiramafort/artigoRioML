import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, 
                             precision_score, 
                             recall_score, 
                             f1_score, 
                             confusion_matrix, 
                             roc_auc_score, 
                             roc_curve,
                             precision_recall_fscore_support)


def avaliar_modelo(modelo, X_teste, y_teste, salvar_figuras=False, nomes_arquivos=None, exibir_figura=True):
    """
    Avalia o desempenho de um modelo de classificação, gerando métricas de avaliação como 
    acurácia, AUC-ROC, precisão, recall, F1-score e matriz de confusão.
    Também exibe e/ou salva gráficos da matriz de confusão e da curva ROC, caso seja solicitado.

    Parâmetros:
        modelo: Objeto classificador treinado (como um modelo do scikit-learn).
        X_teste: Dados de teste.
        y_teste: Rótulos verdadeiros correspondentes a X_teste.
        salvar_figuras (bool): Se True, salva os gráficos gerados em arquivos.
        nomes_arquivos (dict): Dicionário com os nomes dos arquivos para salvar as figuras.
        exibir_figura (bool): Se True, exibe os gráficos no momento da execução.

    Retorno:
        None
    """
    
    # Realiza as previsões
    y_pred = modelo.predict(X_teste)
    y_prob = modelo.predict_proba(X_teste)[:, 1]
    
    # Calcula e exibe métricas globais
    acuracia = accuracy_score(y_teste, y_pred)
    auc_roc = roc_auc_score(y_teste, y_prob)
    print(f"Acurácia: {acuracia:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}\n")
    
    # Calcula e exibe métricas por classe
    precisao = precision_score(y_teste, y_pred, average=None)
    recall = recall_score(y_teste, y_pred, average=None)
    f1 = f1_score(y_teste, y_pred, average=None)
    
    print("Métricas de Classificação por Classe:")
    print(f"---Classe 0:\nPrecisão: {precisao[0]:.4f}\nRecall: {recall[0]:.4f}\nF1-Score: {f1[0]:.4f}")
    print(f"---Classe 1:\nPrecisão: {precisao[1]:.4f}\nRecall: {recall[1]:.4f}\nF1-Score: {f1[1]:.4f}\n")
    
    # Exibe a matriz de confusão
    matriz_confusao = confusion_matrix(y_teste, y_pred)
    print("Matriz de Confusão:\n", matriz_confusao)
    
    # Plota a matriz de confusão e exibe em formato de gráfico
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz_confusao, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Classe 0', 'Classe 1'],
                yticklabels=['Classe 0', 'Classe 1'])
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    plt.tight_layout()
    
    # Se True, salva a matriz de confusão
    if salvar_figuras:
        nome_matriz = nomes_arquivos.get('matriz', 'Matriz_Confusao.jpg') if nomes_arquivos else 'Matriz_Confusao.jpg'
        plt.savefig(nome_matriz)
    
    if exibir_figura:
        plt.show()
    else:
        plt.close()

    # Calcula os pontos para a curva ROC
    fpr, tpr, _ = roc_curve(y_teste, y_prob)
    
    # Plota o gráfico de exibição da AUC-ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'AUC-ROC = {auc_roc:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Linha de chance
    plt.xlabel("Falso Positivo (FPR)")
    plt.ylabel("Verdadeiro Positivo (TPR)")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    
    # Se True, salva o gráfico da AUC-ROC
    if salvar_figuras:
        nome_roc = nomes_arquivos.get('roc', 'Curva_ROC.jpg') if nomes_arquivos else 'Curva_ROC.jpg'
        plt.savefig(nome_roc)
    
    if exibir_figura:
        plt.show()
    else:
        plt.close()

def validador_cruzado(modelo, X, y, n_splits=5, exibir_resultados=True):
    """
    Executa validação cruzada estratificada para modelos de classificação, tendo a mesma funcionalidade da função "avaliarModelo" mas
    aplicado aos folds feitos nos dados

    Parameters:
        modelo: Instância de classificador do scikit-learn.
        X: Conjunto de dados de entrada (features).
        y: Vetor de rótulos (targets).
        n_splits (int): Número de folds na validação cruzada.
        exibir_resultados (bool): Se True, imprime as métricas médias e desvios padrão.

    Returns:
        dict: Dicionário contendo as métricas:
            'Acurácia': (media, desvio),
            'AUC-ROC': (media, desvio),
            'Precisão': (media por classe, desvio),
            'Recall': (media por classe, desvio),
            'F1-Score': (media por classe, desvio),
            'Matriz de Confusão': (media, desvio),
            'Curva ROC': (mean_fpr, mean_tpr, std_tpr)
    """
    
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2024)

    lista_acuracia, lista_auc_roc = [], []
    lista_precisao, lista_recall, lista_f1 = [], [], []
    lista_matrizes = []
    lista_fpr, lista_tpr = [], []

    for indices_treino, indices_teste in kfold.split(X, y):
        if isinstance(X, pd.DataFrame):
            X_treino, X_teste = X.iloc[indices_treino], X.iloc[indices_teste]
            y_treino, y_teste = y.iloc[indices_treino], y.iloc[indices_teste]
        else:
            X_treino, X_teste = X[indices_treino], X[indices_teste]
            y_treino, y_teste = y[indices_treino], y[indices_teste]

        modelo.fit(X_treino, y_treino)
        y_predito = modelo.predict(X_teste)
        y_probabilidade = modelo.predict_proba(X_teste)[:, 1] if hasattr(modelo, 'predict_proba') else None

        lista_acuracia.append(accuracy_score(y_teste, y_predito))
        
        if y_probabilidade is not None:
            auc_roc = roc_auc_score(y_teste, y_probabilidade)
            lista_auc_roc.append(auc_roc)
            fpr, tpr, _ = roc_curve(y_teste, y_probabilidade)
            lista_fpr.append(fpr)
            lista_tpr.append(tpr)
        
        precisao, recall, f1, _ = precision_recall_fscore_support(
            y_teste, y_predito, average=None, labels=np.unique(y)
        )
        lista_precisao.append(precisao)
        lista_recall.append(recall)
        lista_f1.append(f1)
        lista_matrizes.append(confusion_matrix(y_teste, y_predito))

    matriz_media = np.mean(lista_matrizes, axis=0)
    matriz_std = np.std(lista_matrizes, axis=0)

    
    media_auc_roc = np.mean(lista_auc_roc)
    desvio_auc_roc = np.std(lista_auc_roc)

    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr_interp = np.zeros_like(mean_fpr)
    
    for i in range(len(lista_fpr)):
        mean_tpr_interp += np.interp(mean_fpr, lista_fpr[i], lista_tpr[i])
    mean_tpr_interp /= len(lista_fpr)
    
    std_tpr = np.std([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(lista_fpr, lista_tpr)], axis=0)

    resultados = {
        'Acurácia': (np.mean(lista_acuracia), np.std(lista_acuracia)),
        'AUC-ROC': (media_auc_roc, desvio_auc_roc) if lista_auc_roc else (None, None),
        'Precisão': (np.mean(lista_precisao, axis=0), np.std(lista_precisao, axis=0)),
        'Recall': (np.mean(lista_recall, axis=0), np.std(lista_recall, axis=0)),
        'F1-Score': (np.mean(lista_f1, axis=0), np.std(lista_f1, axis=0)),
        'Matriz de Confusão': (matriz_media, matriz_std),
        'Curva ROC': (mean_fpr, mean_tpr_interp, std_tpr)
    }

    if exibir_resultados:
        print("Resultados de Validação Cruzada:")
        for metrica, valores in resultados.items():
            if metrica not in ['Matriz de Confusão', 'Curva ROC']:
                if isinstance(valores[0], np.ndarray):
                    print(f"{metrica}:")
                    for i, (media, desvio) in enumerate(zip(valores[0], valores[1])):
                        print(f"  Classe {i}: Média = {media:.4f}, Desvio Padrão = {desvio:.4f}")
                else:
                    print(f"{metrica}: Média = {valores[0]:.4f}, Desvio Padrão = {valores[1]:.4f}")
            elif metrica == 'Matriz de Confusão':
                print(f"{metrica}:")
                print(f"Média:\n{valores[0]}")
                print(f"Desvio Padrão:\n{valores[1]}")
    
    return resultados