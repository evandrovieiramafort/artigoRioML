import os
import glob
import pandas as pd
import numpy as np
import warnings
from typing import Callable, Dict, Any, List, Tuple
from itables import init_notebook_mode
from config.feriados import obter_feriados_ano

def carregar_datasets(caminho: str, configs: dict) -> pd.DataFrame:
    """
    Lê todos os arquivos CSV em um diretório e concatena-os verticalmente.

    Parâmetros:
        caminho (str): Caminho do diretório onde os arquivos CSV estão localizados.
        configs (dict): Dicionário com as configurações do pd.read_csv.

    Retorna:
        pd.DataFrame: DataFrame resultante da concatenação vertical dos arquivos.
    """
    padrao_busca = os.path.join(caminho, "*.csv")
    lista_arquivos = glob.glob(padrao_busca)

    if not lista_arquivos:
        raise FileNotFoundError(f"Nenhum arquivo CSV encontrado no caminho: {caminho}")

    dataframes = []

    for arquivo in lista_arquivos:
        print(f"Lendo arquivo: {arquivo}")
        df = pd.read_csv(arquivo, **configs)
        dataframes.append(df)

    df_concatenado = pd.concat(dataframes, axis=0, ignore_index=True)

    return df_concatenado

def caminho_saida_figura(nome_arquivo: str) -> str:
    """
    Retorna o caminho completo para salvar figuras no diretório 'figs'.

    Parâmetros:
        nome_arquivo (str): Nome do arquivo da figura (incluindo extensão).

    Retorna:
        str: Caminho completo do arquivo na pasta 'figs'.
    """
    return os.path.join("figs", nome_arquivo)

def configurar_ambiente() -> None:
    """
    Configura o ambiente de execução do notebook:
    - Exibe todas as colunas de DataFrames do pandas.
    - Formata números em ponto flutuante com três casas decimais.
    - Oculta warnings desnecessários.
    - Ativa o modo interativo do itables para exibição de tabelas HTML.

    Retorna:
        None
    """
    try:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.float_format', lambda x: '%.3f' % x)
        warnings.filterwarnings("ignore")
        init_notebook_mode(all_interactive=True)
    except (AttributeError, ImportError) as e:
        print(f"[Erro] Falha ao configurar ambiente — problema com dependências ou atributos: {e}")
        raise
    except Exception as e:
        print(f"[Erro] Erro inesperado ao configurar ambiente: {e}")
        raise

def eh_feriado(row: pd.Series) -> int:
    """
    Verifica se a data da linha é um feriado nacional brasileiro.

    Parâmetros:
        row (pd.Series): Linha do DataFrame contendo a coluna 'data_inversa'.

    Retorna:
        int: 1 se a data for feriado, 0 caso contrário.
    """
    try:
        data = pd.to_datetime(row['data_inversa']).normalize()
        ano = data.year
        feriados = obter_feriados_ano(ano)
        return int(data in feriados)
    except KeyError:
        print(f"[Erro] Coluna 'data_inversa' ausente na linha {row.name}")
        return 0
    except Exception as e:
        print(f"[Erro] Falha ao verificar feriado na linha {row.name}: {e}")
        return 0

def mapeador(dicionario: Dict[Tuple[Any, ...], Any]) -> Callable[[Any], Any]:
    """
    Cria uma função de mapeamento baseada em um dicionário de tuplas como chaves.

    Parâmetros:
        dicionario (Dict[Tuple[Any, ...], Any]): Dicionário onde as chaves são tuplas e os valores são categorias.

    Retorna:
        Callable[[Any], Any]: Função que mapeia um valor individual para sua categoria.
    """
    def mapper(valor: Any) -> Any:
        try:
            for chaves, resultado in dicionario.items():
                if valor in chaves:
                    return resultado
            return None
        except TypeError as e:
            print(f"[Erro] Tipo inválido no mapeamento do valor '{valor}': {e}")
            return None
        except Exception as e:
            print(f"[Erro] Erro inesperado no mapeamento do valor '{valor}': {e}")
            return None

    return mapper

def extrair_categorias_tracado(tracado: str) -> pd.Series:
    """
    Extrai categorias do campo 'tracado_via'.

    Parâmetros:
        tracado (str): Descrição do traçado da via.

    Retorna:
        pd.Series: Série com as categorias [condicaoPista, tipoInclinacao, tipoSuperficie, tipoManobra, tipoEstrutura].
    """
    condicao_pista = 'Normal'
    tipo_inclinacao = 'Plano'
    tipo_superficie = 'Reta'
    tipo_manobra = 'Nenhum'
    tipo_estrutura = 'Nenhum'

    if not isinstance(tracado, str):
        raise TypeError(f"Valor esperado do tipo str para 'tracado_via', recebido {type(tracado)}")

    if 'Em Obras' in tracado or 'Desvio Temporário' in tracado:
        condicao_pista = 'Em obras'

    if 'Aclive' in tracado:
        tipo_inclinacao = 'Aclive'
    elif 'Declive' in tracado:
        tipo_inclinacao = 'Declive'

    if 'Curva' in tracado:
        tipo_superficie = 'Curva'

    if 'Rotatória' in tracado:
        tipo_manobra = 'Rotatória'
    elif 'Interseção' in tracado:
        tipo_manobra = 'Interseção'
    elif 'Retorno Regulamentado' in tracado:
        tipo_manobra = 'Retorno Regulamentado'

    if 'Ponte' in tracado:
        tipo_estrutura = 'Ponte'
    elif 'Túnel' in tracado:
        tipo_estrutura = 'Túnel'
    elif 'Viaduto' in tracado:
        tipo_estrutura = 'Viaduto'

    return pd.Series([
        condicao_pista,
        tipo_inclinacao,
        tipo_superficie,
        tipo_manobra,
        tipo_estrutura
    ])

def categoriza_tracado_via(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica a extração de categorias do traçado da via em todas as linhas do DataFrame.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo a coluna 'tracado_via'.

    Retorna:
        pd.DataFrame: DataFrame original com as colunas adicionais [condicaoPista, tipoInclinacao, tipoSuperficie, tipoManobra, tipoEstrutura].
    """
    try:
        df[['condicaoPista', 'tipoInclinacao', 'tipoSuperficie', 'tipoManobra', 'tipoEstrutura']] = (
            df['tracado_via'].apply(extrair_categorias_tracado)
        )
        return df
    except KeyError as e:
        print(f"[Erro] Coluna 'tracado_via' não encontrada no DataFrame: {e}")
        raise
    except TypeError as e:
        print(f"[Erro] Tipo inválido ao processar a coluna 'tracado_via': {e}")
        raise
    except Exception as e:
        print(f"[Erro] Erro inesperado ao categorizar traçado da via: {e}")
        raise

def converte_features_ciclicas(df: pd.DataFrame, colunas: List[str], valores_maximos: List[int]) -> pd.DataFrame:
    """
    Converte variáveis cíclicas (hora, mês, dia) em componentes senoides (Sen, Cos).

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo as colunas a serem transformadas.
        colunas (List[str]): Lista de nomes das colunas cíclicas.
        valores_maximos (List[int]): Lista de valores máximos para cada coluna (ex: 24 para hora).

    Retorna:
        pd.DataFrame: DataFrame com as colunas cíclicas convertidas em componentes senoides.
    """
    try:
        for coluna, valorMaximo in zip(colunas, valores_maximos):
            if coluna not in df.columns:
                print(f"[Aviso] Coluna '{coluna}' não encontrada no DataFrame. Pulando conversão cíclica.")
                continue

            if not np.issubdtype(df[coluna].dtype, np.number):
                raise TypeError(f"Coluna '{coluna}' deve ser numérica para conversão cíclica")

            df[coluna + 'Sen'] = np.sin(2 * np.pi * df[coluna] / valorMaximo)
            df[coluna + 'Cos'] = np.cos(2 * np.pi * df[coluna] / valorMaximo)
            df.drop(coluna, axis=1, inplace=True)
        return df
    except TypeError as e:
        print(f"[Erro] Tipo inválido durante conversão cíclica: {e}")
        raise
    except Exception as e:
        print(f"[Erro] Erro inesperado na conversão cíclica das colunas {colunas}: {e}")
        raise

def remove_outliers(df: pd.DataFrame, colunas: List[str]) -> pd.DataFrame:
    """
    Remove outliers de colunas numéricas com base na regra do IQR (1.5 * intervalo interquartil).

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo as colunas.
        colunas (List[str]): Lista de colunas onde será aplicada a remoção de outliers.

    Retorna:
        pd.DataFrame: DataFrame com outliers removidos.
    """
    try:
        for coluna in colunas:
            if coluna not in df.columns:
                print(f"[Aviso] Coluna '{coluna}' não encontrada no DataFrame. Pulando remoção de outliers.")
                continue

            if not np.issubdtype(df[coluna].dtype, np.number):
                print(f"[Aviso] Coluna '{coluna}' não é numérica. Pulando remoção de outliers nessa coluna.")
                continue

            q1 = df[coluna].quantile(0.25)
            q3 = df[coluna].quantile(0.75)
            iqr = q3 - q1
            limite_inferior = q1 - 1.5 * iqr
            limite_superior = q3 + 1.5 * iqr
            df = df[(df[coluna] >= limite_inferior) & (df[coluna] <= limite_superior)]
        return df
    except Exception as e:
        print(f"[Erro] Erro inesperado ao remover outliers das colunas {colunas}: {e}")
        raise

def define_gravidade(row: pd.Series) -> int:
    """
    Define a gravidade do acidente com base nas vítimas, criando o novo target para o estudo.

    Parâmetros:
        row (pd.Series): Linha do DataFrame contendo as colunas 'mortos', 'feridos_graves', 'feridos_leves', 'ilesos'.

    Retorna:
        int: 1 para acidentes graves, 0 para não-graves.
    """
    try:
        mortos = int(row.get('mortos', 0))
        feridos_graves = int(row.get('feridos_graves', 0))
        feridos_leves = int(row.get('feridos_leves', 0))
        ilesos = int(row.get('ilesos', 0))

        if mortos > 0 or feridos_graves > 0:
            return 1
        elif feridos_leves > 0 or ilesos > 0:
            return 0
        else:
            return 0
    except (ValueError, TypeError) as e:
        print(f"[Erro] Dados inválidos para definir gravidade na linha {row.name}: {e}")
        return 0
    except Exception as e:
        print(f"[Erro] Erro inesperado ao definir gravidade na linha {row.name}: {e}")
        return 0

def define_fase_do_dia(coluna: List[int]) -> List[str]:
    """
    Determina a fase do dia (Dia ou Noite) com base na hora informada.

    Parâmetros:
        coluna (List[int]): Lista com valores de hora (0-23).

    Retorna:
        List[str]: Lista com as fases do dia correspondentes ('Dia', 'Noite', 'Desconhecido').
    """
    try:
        fase = []
        for i, hora in enumerate(coluna):
            if not isinstance(hora, (int, float)):
                print(f"[Aviso] Valor inválido para hora na posição {i}: {hora}. Definindo fase como 'Desconhecido'.")
                fase.append('Desconhecido')
                continue
            if 6 < hora < 18:
                fase.append('Dia')
            else:
                fase.append('Noite')
        return fase
    except Exception as e:
        print(f"[Erro] Erro inesperado ao definir fase do dia para valores {coluna}: {e}")
        return ['Desconhecido' for _ in coluna]

def calcular_frequencia_acidente(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria a feature 'frequenciaAcidente', que representa a frequência relativa de acidentes por BR e km.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo as colunas 'id', 'br', 'km'.

    Retorna:
        pd.DataFrame: DataFrame original com a coluna 'frequenciaAcidente' adicionada.
    """
    n = len(df)

    df[['br', 'km']] = df[['br', 'km']].replace(',', '.', regex=True).astype(float).astype(int)
    df['frequenciaAcidente'] = df.groupby(['br', 'km'])['id'].transform('count') / n

    df['br'] = df['br'].astype(str)

    return df
