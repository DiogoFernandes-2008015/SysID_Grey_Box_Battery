# python
import sys
import subprocess
from pathlib import Path

def run_scripts_in_sequence(scripts, continue_on_error=False):
    """
    Executa uma lista de scripts Python em sequência usando o mesmo
    interpretador (sys.executable). Retorna True se todos terminaram OK.
    """
    py = sys.executable  # garante usar o mesmo Python do PyCharm
    for script in scripts:
        script_path = Path(script)
        if not script_path.exists():
            print(f"Arquivo não encontrado: `{script_path}`")
            if not continue_on_error:
                return False
            else:
                continue
        print(f"Executando `{script_path}` ...")
        try:
            res = subprocess.run([py, str(script_path)], check=True)
            print(f"`{script_path}` finalizado com código {res.returncode}\n")
        except subprocess.CalledProcessError as e:
            print(f"Erro ao executar `{script_path}`: código {e.returncode}")
            if not continue_on_error:
                return False
    return True

if __name__ == "__main__":
    # Exemplo: liste aqui os seus scripts na ordem desejada
    scripts = [
        "ID_Grey_Battery_1RC_model.py",
        "ID_Grey_Battery_2RC_model.py",
        "ID_Grey_Battery_PNGV.py"
    ]
    ok = run_scripts_in_sequence(scripts, continue_on_error=False)
    print("Sequência concluída com sucesso." if ok else "Sequência interrompida por erro.")