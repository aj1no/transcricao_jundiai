import os
import fitz
import re
import shutil

# Tabela de correção de caracteres que aparecem com lixo do PDF
def clean_text(text):
    # Remover caracteres nulos e lixo comum de PDFs mal codificados
    text = text.replace('\x00', '').replace('\xc3\x80', '')
    # Remover o caractere '' (substituição do unicode) se possível, ou tentar decodificar
    # Mas como o PyMuPDF já deu o texto, vamos limpar os nulls e normalizar
    lines = text.split('\n')
    cleaned_lines = [l.strip() for l in lines]
    cleaned_lines = [l for l in cleaned_lines if l]
    return "\n".join(cleaned_lines)

def is_valid_transcription(text):
    text_lower = text.lower()
    # Bloquear explicitamente páginas que são APENAS introdução/normas
    if "advertência ao leitor" in text_lower: return False
    if "normas e critérios" in text_lower: return False
    if "referência documental" in text_lower and len(text) < 400: return False
    
    # Se chegamos aqui e tem conteúdo razoável, vamos assumir que é transcrição
    if len(text) > 50:
        return True
    return False

def finalize_organization():
    temp_path = "data/raw/temp_extracted"
    final_images_path = "data/raw/images"
    final_trans_path = "data/raw/transcriptions"

    # Resetar pastas
    for p in [final_images_path, final_trans_path]:
        if os.path.exists(p): shutil.rmtree(p)
        os.makedirs(p)

    print("--- Extração de Alta Fidelidade (Final) ---")

    for root, dirs, files in os.walk(temp_path):
        folder_slug_raw = os.path.basename(root)
        if not folder_slug_raw: continue
        
        folder_slug = re.sub(r'[^\w\s-]', '', folder_slug_raw.lower()).replace(' ', '_')
        
        pdfs = [f for f in files if f.lower().endswith(".pdf")]
        imgs = [f for f in files if f.lower().endswith(".jpg") or f.lower().endswith(".png")]
        
        trans_pdf = next((p for p in pdfs if "transcri" in p.lower()), None)
        folio_pdf = next((p for p in pdfs if "folio" in p.lower() or "traslado" in p.lower()), None)

        if not trans_pdf: continue

        print(f"\n Pasta: {folder_slug}")
        doc_trans = fitz.open(os.path.join(root, trans_pdf))
        
        valid_pages = []
        for i in range(len(doc_trans)):
            raw_text = doc_trans[i].get_text()
            if is_valid_transcription(raw_text):
                valid_pages.append(clean_text(raw_text))
        
        print(f"  > Páginas válidas: {len(valid_pages)}")

        # Extrair/Associar
        if folio_pdf:
            doc_folios = fitz.open(os.path.join(root, folio_pdf))
            for idx, text in enumerate(valid_pages):
                if idx < len(doc_folios):
                    img_name = f"{folder_slug}_p{idx+1:03d}.jpg"
                    txt_name = f"{folder_slug}_p{idx+1:03d}.txt"
                    
                    page_img = doc_folios[idx] # Tentar mapeamento direto
                    pix = page_img.get_pixmap(dpi=300)
                    pix.save(os.path.join(final_images_path, img_name))
                    with open(os.path.join(final_trans_path, txt_name), "w", encoding="utf-8") as f:
                        f.write(text)
            doc_folios.close()
        elif imgs:
            imgs.sort()
            for idx, text in enumerate(valid_pages):
                if idx < len(imgs):
                    ext = os.path.splitext(imgs[idx])[1]
                    img_name = f"{folder_slug}_p{idx+1:03d}{ext}"
                    txt_name = f"{folder_slug}_p{idx+1:03d}.txt"
                    shutil.copy(os.path.join(root, imgs[idx]), os.path.join(final_images_path, img_name))
                    with open(os.path.join(final_trans_path, txt_name), "w", encoding="utf-8") as f:
                        f.write(text)
        
        doc_trans.close()

if __name__ == "__main__":
    finalize_organization()
