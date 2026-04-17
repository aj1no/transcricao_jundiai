import cv2
import os
import pandas as pd
import numpy as np

def segment_lines(image_path, transcription_path, output_dir, prefix):
    img = cv2.imread(image_path)
    if img is None: return []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 35, 15
    )
    
    # Kernel horizontal para destacar linhas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Ordenar contornos de cima para baixo
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
    
    # Ler texto e filtrar linhas vazias/curtas demais (boilerplate)
    with open(transcription_path, 'r', encoding='utf-8') as f:
        text_lines = [l.strip() for l in f.readlines() if len(l.strip()) > 2]
    
    valid_lines = []
    line_idx = 0
    
    print(f"  > Processando {prefix}: {len(contours)} contornos vs {len(text_lines)} linhas de texto.")

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Filtros de sanidade para linhas de texto manuscrito
        if w > img.shape[1] * 0.2 and h > 15 and h < 250:
            if line_idx < len(text_lines):
                line_img = img[y:y+h, x:x+w]
                filename = f"{prefix}_line_{line_idx:03d}.jpg"
                cv2.imwrite(os.path.join(output_dir, filename), line_img)
                
                valid_lines.append({
                    "file_name": filename,
                    "text": text_lines[line_idx]
                })
                line_idx += 1
                
    return valid_lines

def build_dataset():
    raw_images = "data/raw/images"
    raw_trans = "data/raw/transcriptions"
    output_images = "data/processed/train/images"
    metadata_path = "data/processed/train/metadata.csv"

    if not os.path.exists(output_images): os.makedirs(output_images)

    all_data = []
    
    files = sorted(os.listdir(raw_images))
    for f in files:
        if f.endswith(('.jpg', '.png')):
            base = os.path.splitext(f)[0]
            img_path = os.path.join(raw_images, f)
            txt_path = os.path.join(raw_trans, base + ".txt")
            
            if os.path.exists(txt_path):
                lines = segment_lines(img_path, txt_path, output_images, base)
                all_data.extend(lines)

    df = pd.DataFrame(all_data)
    df.to_csv(metadata_path, index=False)
    print(f"\n--- Dataset Gerado! ---")
    print(f"Total de linhas recortadas: {len(df)}")
    print(f"Manifesto salvo em: {metadata_path}")

if __name__ == "__main__":
    build_dataset()
