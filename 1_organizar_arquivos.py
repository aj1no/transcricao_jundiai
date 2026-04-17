import os
import shutil
import subprocess

def organizar_arquivos():
    user_path = os.path.expanduser("~")
    downloads_path = os.path.join(user_path, "Downloads")
    project_path = os.getcwd()
    raw_path = os.path.join(project_path, "data", "raw")
    temp_path = os.path.join(raw_path, "temp_extracted")

    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    # Ferramentas de extração
    seven_zip = r"C:\Program Files\7-Zip\7z.exe"
    unrar = r"C:\Program Files\WinRAR\UnRAR.exe"

    arquivos_interesse = [
        "Transcrições.rar",
        "Photos-1-001.zip",
        "Photos-3-001 (1).zip",
        "Photos-3-001 (2).zip",
        "Photos-3-001.zip"
    ]

    print("--- 1. Movendo arquivos do Downloads ---")
    for arq in arquivos_interesse:
        origem = os.path.join(downloads_path, arq)
        destino = os.path.join(raw_path, arq)
        
        if os.path.exists(origem):
            print(f"Movendo: {arq}")
            shutil.move(origem, destino)
        elif os.path.exists(destino):
            print(f"Arquivo já estava no destino: {arq}")
        else:
            print(f"AVISO: {arq} não encontrado.")

    print("\n--- 2. Extraindo arquivos ---")
    for arq in os.listdir(raw_path):
        completo = os.path.join(raw_path, arq)
        if arq.endswith(".zip"):
            print(f"Extraindo ZIP: {arq}")
            # Usando 7z para garantir sucesso
            subprocess.run([seven_zip, "x", completo, f"-o{temp_path}", "-y"], check=True)
        elif arq.endswith(".rar"):
            print(f"Extraindo RAR: {arq}")
            # Usando UnRAR
            subprocess.run([unrar, "x", completo, temp_path, "-y"], check=True)

    print("\nSucesso! Tudo extraído em data/raw/temp_extracted/")

if __name__ == "__main__":
    organizar_arquivos()
