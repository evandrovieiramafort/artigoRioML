# config/feriados.py
from datetime import date
import pandas as pd
from pandas.tseries.offsets import Day
from dateutil.easter import easter

# Dicionário que armazenará feriados calculados para cada ano
FERIADOS_POR_ANO: dict[int, list[pd.Timestamp]] = {}

def gerar_feriados(ano: int) -> list[pd.Timestamp]:
    """
    Gera a lista de feriados nacionais brasileiros para um determinado ano.

    Os feriados considerados são:
    - Confraternização Universal (1º de Janeiro)
    - Segunda-feira de Carnaval (48 dias antes da Páscoa)
    - Terça-feira de Carnaval (47 dias antes da Páscoa)
    - Sexta-feira Santa (2 dias antes da Páscoa)
    - Tiradentes (21 de Abril)
    - Dia de São Jorge (feriado estadual, 23 de Abril)
    - Dia do Trabalho (1º de Maio)
    - Corpus Christi (60 dias após a Páscoa)
    - Independência do Brasil (7 de Setembro)
    - Nossa Senhora Aparecida (12 de Outubro)
    - Finados (2 de Novembro)
    - Proclamação da República (15 de Novembro)
    - Natal (25 de Dezembro)

    Parâmetros:
        ano (int): Ano para o qual os feriados serão calculados.

    Retorna:
        list[pd.Timestamp]: Lista de feriados no formato Timestamp para o ano especificado.
    """
    pascoa = pd.Timestamp(easter(ano))

    return [
        pd.Timestamp(date(ano, 1, 1)),
        pascoa - Day(48),
        pascoa - Day(47),
        pascoa - Day(2),
        pd.Timestamp(date(ano, 4, 21)),
        pd.Timestamp(date(ano, 5, 1)),
        pascoa + Day(60),
        pd.Timestamp(date(ano, 9, 7)),
        pd.Timestamp(date(ano, 10, 12)),
        pd.Timestamp(date(ano, 11, 2)),
        pd.Timestamp(date(ano, 11, 15)),
        pd.Timestamp(date(ano, 12, 25)),
        pd.Timestamp(date(ano, 4, 23)),
    ]


def obter_feriados_ano(ano: int) -> list[pd.Timestamp]:
    """Retorna a lista de feriados para o ano, usando cache para não recalcular."""
    if ano not in FERIADOS_POR_ANO:
        FERIADOS_POR_ANO[ano] = gerar_feriados(ano)
    return FERIADOS_POR_ANO[ano]
