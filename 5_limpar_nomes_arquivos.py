import os
import shutil
import re

def slugify(text):
    # Converte para ASCII básico removendo acentos e caracteres especiais
    text = text.lower()
    text = text.replace('ã', 'a').replace('á', 'a').replace('â', 'a')
    text = text.replace('é', 'e').replace('ê', 'e')
    text = text.replace('í', 'i')
    text = text.replace('õ', 'o').replace('ó', 'o').replace('ô', 'o')
    text = text.replace('ú', 'u')
    text = text.replace('ç', 'c')
    # Remove qualquer outro caracter não alfanumérico exceto . e _
    text = re.sub(r'[^a-z0-9\._-]', '_', text)
    # Remove múltiplos underscores
    text = re.sub(r'_+', '_', text)
    return text.strip('_')

def rename_to_ascii(directory):
    if not os.path.exists(directory):
        print(f"Diretório {directory} não existe.")
        return

    print(f"Renomeando arquivos em: {directory}")
    for filename in os.listdir(directory):
        old_path = os.path.join(directory, filename)
        if os.path.isdir(old_path):
            continue
            
        new_filename = slugify(filename)
        new_path = os.path.join(directory, new_filename)
        
        if old_path != new_path:
            try:
                # Se o destino já existir, remover primeiro para evitar erro no Windows
                if os.path.exists(new_path):
                    os.remove(new_path)
                os.rename(old_path, new_path)
                print(f"  {filename} -> {new_filename}")
            except Exception as e:
                print(f"  ERRO ao renomear {filename}: {e}")

if __name__ == "__main__":
    rename_to_ascii("data/raw/images")
    rename_to_ascii("data/raw/transcriptions")
