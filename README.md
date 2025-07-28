## Objetivo

Hospedar os códigos utilizados no artigo *Técnicas de Aprendizado de Máquina para Predição de Gravidade de Acidentes em Rodovias do Estado do Rio de Janeiro*, autoria de Evandro Vieira Mafort e Marco André Abud Kappel, publicado na REIC (Revista Eletrônica de Iniciação Científica).

O gerenciador de dependências utilizado aqui é o [UV](https://github.com/astral-sh/uv).'

## Itens

1. Dois notebooks:
   - *notebookPrincipal*, onde foi feito todo o estudo
   - *analiseDeDados*, onde foram geradas as figuras

2. Quatro diretórios:
   - *config*: Diretório contendo strings de caminho de arquivo, assim como dicionários contendo as instâncias de cada algoritmo e os hiperparâmetros dos algoritmos.
   - *data*: Onde os dados são guardados. Por motivos de boas práticas acerca de armazenamento do GitHub, esta pasta estará vazia. Baixe os datasets (indicado no artigo) e o código faz o resto.
   - *utils*: Onde todas as funções utilizadas nos códigos estão. Todas estão devidamente documentadas com docstrings, a fim de facilitar o entendimento do que foi usado.
   - *figs*: Local onde todas as figuras geradas nos dois notebook são armazenadas, figuras essas que são identificadas de acordo com os nomes dados no artigo.
   
3. Um script de geração automatizada para o *pyproject.toml* (*gerador_dependencies_toml.py* - mais detalhes na docstring das funções utilizadas).

4. *O requirements.txt* contendo todas as biblliotecas usadas  no estudo.
