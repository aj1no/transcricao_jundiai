import os
import fitz  # PyMuPDF
import shutil

def survey_and_organize():
    project_path = os.getcwd()
    temp_path = os.path.join(project_path, "data", "raw", "temp_extracted")
    final_images_path = os.path.join(project_path, "data", "raw", "images")
    final_trans_path = os.path.join(project_path, "data", "raw", "transcriptions")

    if not os.path.exists(final_images_path): os.makedirs(final_images_path)
    if not os.path.exists(final_trans_path): os.makedirs(final_trans_path)

    print("--- Analisando Estrutura Extraída ---")
    
    for root, dirs, files in os.walk(temp_path):
        pdfs = [f for f in files if f.lower().endswith(".pdf")]
        imgs = [f for f in files if f.lower().endswith(".jpg") or f.lower().endswith(".png")]
        
        if not pdfs:
            continue
            
        rel_path = os.path.relpath(root, temp_path)
        print(f"\nPasta: {rel_path}")
        print(f"  PDFs encontrados: {len(pdfs)}")
        print(f"  Imagens encontradas: {len(imgs)}")

        # Identificar o PDF de transcrição (geralmente tem 'transcri' no nome)
        trans_pdf = None
        folio_pdf = None
        
        for p in pdfs:
            if "transcri" in p.lower():
                trans_pdf = p
            elif "folio" in p.lower() or "traslado" in p.lower():
                folio_pdf = p

        if trans_pdf:
            print(f"  > PDF de Transcrição: {trans_pdf}")
            # Tentar processar
            processar_par(root, trans_pdf, imgs, folio_pdf, final_images_path, final_trans_path)

def processar_par(root, trans_pdf_name, imgs, folio_pdf_name, dest_img, dest_trans):
    folder_slug = os.path.basename(root).replace(" ", "_").lower()
    trans_path = os.path.join(root, trans_pdf_name)
    
    # 1. Abrir PDF de transcrição
    doc_trans = fitz.open(trans_path)
    num_paginas = len(doc_trans)
    
    # 2. Se houver um PDF de fólios (imagens), extrair páginas como imagens
    if folio_pdf_name:
        print(f"  > Extraindo imagens do PDF: {folio_pdf_name}")
        doc_folios = fitz.open(os.path.join(root, folio_pdf_name))
        for i in range(len(doc_folios)):
            page = doc_folios[i]
            pix = page.get_pixmap(dpi=300)
            img_name = f"{folder_slug}_folio_{i+1:03d}.jpg"
            pix.save(os.path.join(dest_img, img_name))
            
            # Salvar texto correspondente (se existir página no trans_pdf)
            if i < num_paginas:
                text = doc_trans[i].get_text()
                with open(os.path.join(dest_trans, f"{folder_slug}_folio_{i+1:03d}.txt"), "w", encoding="utf-8") as f:
                    f.write(text)
        doc_folios.close()
    
    # 3. Se houver imagens JPG soltas na pasta
    elif imgs:
        imgs.sort() # Tentar manter ordem alfabética/numérica
        print(f"  > Associando {len(imgs)} imagens soltas...")
        for i, img_file in enumerate(imgs):
            # Mover/Copiar imagem
            ext = os.path.splitext(img_file)[1]
            novo_nome_img = f"{folder_slug}_folio_{i+1:03d}{ext}"
            shutil.copy(os.path.join(root, img_file), os.path.join(dest_img, novo_nome_img))
            
            # Salvar texto correspondente
            if i < num_paginas:
                text = doc_trans[i].get_text()
                with open(os.path.join(dest_trans, f"{folder_slug}_folio_{i+1:03d}.txt"), "w", encoding="utf-8") as f:
                    f.write(text)
    
    doc_trans.close()

if __name__ == "__main__":
    survey_and_organize()
