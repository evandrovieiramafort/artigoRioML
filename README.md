## Objetivo

Hospedar os códigos utilizados no artigo *Técnicas de Aprendizado de Máquina para Predição de Gravidade de Acidentes em Rodovias do Estado do Rio de Janeiro*, de autoria de **Evandro Vieira Mafort** e **Marco André Abud Kappel**, publicado na **REIC (Revista Eletrônica de Iniciação Científica)**.

O gerenciador de dependências utilizado é o [UV](https://github.com/astral-sh/uv).

## Itens

### Notebooks
- `notebookPrincipal`: onde foi realizado todo o estudo.
- `analiseDeDados`: responsável por gerar as figuras.

### Diretórios
- `config`: contém strings de caminhos de arquivos, além dos dicionários com as instâncias dos algoritmos e seus hiperparâmetros.
- `data`: pasta destinada aos datasets. Por boas práticas de versionamento, ela estará vazia. O artigo indica onde baixar os arquivos, e o código cuidará do restante.
- `utils`: contém todas as funções utilizadas no projeto, devidamente documentadas com docstrings.
- `figs`: armazena todas as figuras geradas nos notebooks, nomeadas conforme indicado no artigo.

### Scripts
- `gerador_dependencies_toml.py`: script automatizado para gerar o `pyproject.toml`. Detalhes adicionais podem ser encontrados nas docstrings das funções ali contidas.

### Arquivos de Dependências
- `requirements.txt`: lista todas as bibliotecas utilizadas no estudo.
