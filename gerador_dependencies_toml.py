import toml
import re
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


def gerar_toml_atualizado():
    """
    Script que criei para automatizar a atualização das dependências no pyproject.toml com base no requirements.txt.

    A partir do momento que o script é executado, ele atualiza o toml em tempo real. Sempre que um requirements.txt for
    gerado, o código pega o conteúdo do txt atualizado e cria um pyproject.toml com o "dependencies" atualizado.

    IMPORTANTE:
    Adicionou alguma biblioteca?
    1. Rode este código LOGO EM SEGUIDA
    2. Faça o "uv pip freze > requirements.txt"
    """
    try:
        with open("pyproject.toml", "r", encoding="utf-8") as f:
            pyproject = toml.load(f)
    except Exception as erro_leitura_toml:
        print(f"[ERRO] Falha ao ler pyproject.toml: {erro_leitura_toml}")
        return

    try:
        with open("requirements.txt", "r", encoding="utf-16") as f:
            dependencias = [linha.strip() for linha in f if linha.strip() and not linha.startswith("#")]
    except Exception as erro_leitura_reqs:
        print(f"[ERRO] Falha ao ler requirements.txt: {erro_leitura_reqs}")
        return

    if "project" not in pyproject:
        pyproject["project"] = {}

    pyproject["project"]["dependencies"] = dependencias

    try:
        toml_str = toml.dumps(pyproject)
    except Exception as erro_dumps_toml:
        print(f"[ERRO] Falha ao converter para TOML string: {erro_dumps_toml}")
        return

    padrao_regex = r'dependencies = \[(.*?)\]'
    try:
        if re.search(padrao_regex, toml_str, re.DOTALL):
            deps_formatadas = ',\n    '.join(f'"{dep}"' for dep in dependencias)
            toml_str = re.sub(
                padrao_regex,
                f'dependencies = [\n    {deps_formatadas}\n]',
                toml_str,
                flags=re.DOTALL
            )
    except Exception as erro_regex:
        print(f"[ERRO] Falha ao formatar dependências no TOML: {erro_regex}")
        return

    try:
        with open("pyproject.toml", "w", encoding="utf-8") as f:
            f.write(toml_str)
    except Exception as erro_gravacao_toml:
        print(f"[ERRO] Falha ao salvar pyproject.toml: {erro_gravacao_toml}")


class RequisitosHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith("requirements.txt"):
            try:
                print("[INFO] Arquivo requirements.txt modificado. Atualizando pyproject.toml...")
                gerar_toml_atualizado()
                print("[INFO] pyproject.toml atualizado com sucesso.")
            except Exception as erro_atualizacao:
                print(f"[ERRO] Exceção durante atualização: {erro_atualizacao}")


if __name__ == "__main__":
    path = "."
    event_handler = RequisitosHandler()
    observer = Observer()
    observer.schedule(event_handler, path=path, recursive=False)

    try:
        observer.start()
        print("Monitorando requirements.txt por alterações (Pressione Ctrl+C para sair)")
        while True:
            time.sleep(1)
    except Exception as erro_observer:
        print(f"[ERRO] Problema encontrado ao executar o event handler: {erro_observer}")
        observer.stop()

    observer.join()
