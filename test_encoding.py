import fitz
import os

def test_encodings(pdf_path):
    doc = fitz.open(pdf_path)
    # Pegar uma página que sabemos ter problemas
    for i in range(len(doc)):
        text = doc[i].get_text()
        print(f"--- Página {i+1} ---")
        print(text[:200])
        # Tentar extrair com diferentes flags
        # text_raw = doc[i].get_text("rawdict") # Muito detalhado
        # print("Exemplo de caracter problemático:")
        # for b in text:
        #     if ord(b) > 127:
        #         print(f"Char: {b} | Hex: {hex(ord(b))} | Name: {fitz.get_font_name(doc[i], b) if hasattr(fitz, 'get_font_name') else '?'}")

if __name__ == "__main__":
    # Caminho para um PDF específico
    path = "data/raw/temp_extracted/Transcrições/Traslado do Auto de Criação da Vila de Jundiahy/Transcrição..pdf"
    if os.path.exists(path):
        test_encodings(path)
    else:
        print("Arquivo não encontrado")
