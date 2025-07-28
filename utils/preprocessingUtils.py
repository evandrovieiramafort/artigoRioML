import pandas as pd
import numpy as np
import warnings
from typing import Callable, Dict, Any, List, Tuple
from itables import init_notebook_mode

from config.feriados import obter_feriados_ano


def configurar_ambiente() -> None:
    """
        Configura o ambiente de execução do notebook:
        - Exibe todas as colunas de DataFrames do pandas.
        - Formata números em ponto flutuante com três casas decimais.
        - Oculta warnings desnecessários.
        - Ativa o modo interativo do itables para exibição de tabelas HTML.
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
    Utiliza um dicionário cacheado (importado de config/feriados.py) com feriados por ano.
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
    Retorna uma função de mapeamento que converte valores com base em um dicionário de tuplas como chaves.

    O dicionário deve conter tuplas como chaves, e cada valor associado representa a categoria desejada.
    A função resultante (`mapper`) verifica se o valor fornecido pertence a alguma das tuplas (chaves)
    e retorna o valor correspondente.
    """

    def mapper(valor: Any) -> Any:
        """
        Mapeia um valor individual para sua categoria, com base no dicionário de tuplas definido.
        """
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
    Extrai categorias do campo 'tracado_via', retornando uma série com:
    [condicaoPista, tipoInclinacao, tipoSuperficie, tipoManobra, tipoEstrutura]
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
    Aplica a extração de categorias do traçado da via (em extrair_categorias_tracado) para cada linha do DataFrame.
    Cria as colunas:
    - condicaoPista
    - tipoInclinacao
    - tipoSuperficie
    - tipoManobra
    - tipoEstrutura
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
        Converte variáveis cíclicas (como hora, mês, dia) em componentes senoidais (Sen, Cos).
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
        Define a gravidade do acidente com base nas vítimas, criando o novo target para o estudo. Retorna 0 para
        acidentes considerados não-graves e 1 para acidentes graves (AMORIM, 2019).
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
        Determina a fase do dia (Dia ou Noite) com base na hora.
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
