import cv2
import numpy as np
import os
import pandas as pd
from scipy.signal import find_peaks

def segment_lines_peaks(image_path, transcription_path, output_dir, prefix):
    img = cv2.imread(image_path)
    if img is None: return []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Binarização mais fina para não engrossar as letras
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 10
    )
    
    # 2. Projeção Horizontal
    projection = np.sum(thresh, axis=1)
    
    # 3. Suavizar a projeção para ignorar ruídos intra-linha
    kernel_size = 20
    projection_smooth = np.convolve(projection, np.ones(kernel_size)/kernel_size, mode='same')
    
    # 4. Encontrar picos (centros das linhas)
    # distance: distância mínima entre picos (altura esperada de uma linha ~40-80px)
    peaks, _ = find_peaks(projection_smooth, distance=40, prominence=img.shape[1]*2)
    
    # 5. Definir bordas das linhas baseadas nos vales entre picos
    line_ranges = []
    for i in range(len(peaks)):
        p = peaks[i]
        
        # Início: vale entre p anterior e p atual (ou 0 se for o primeiro)
        if i == 0:
            start = max(0, p - 30)
        else:
            prev_p = peaks[i-1]
            # O ponto de corte é o ponto mais baixo da projeção entre os dois picos
            valley = prev_p + np.argmin(projection_smooth[prev_p:p])
            start = valley
            
        # Fim: vale entre p atual e p próximo (ou fim da imagem)
        if i == len(peaks) - 1:
            end = min(img.shape[0], p + 30)
        else:
            next_p = peaks[i+1]
            valley = p + np.argmin(projection_smooth[p:next_p])
            end = valley
            
        line_ranges.append((start, end))
            
    # Ler texto
    with open(transcription_path, 'r', encoding='utf-8') as f:
        text_lines = [l.strip() for l in f.readlines() if len(l.strip()) > 3]
        
    print(f"  > {prefix}: {len(line_ranges)} picos detectados vs {len(text_lines)} transcritas.")
    
    valid_data = []
    n_lines = min(len(line_ranges), len(text_lines))
    
    for idx in range(n_lines):
        y_start, y_end = line_ranges[idx]
        
        # Recortar (largura total)
        line_img = img[y_start:y_end, :]
        
        filename = f"{prefix}_p{idx:03d}.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), line_img)
        
        valid_data.append({
            "file_name": filename,
            "text": text_lines[idx]
        })
        
    return valid_data

def build_peak_dataset():
    raw_images = "data/raw/images"
    raw_trans = "data/raw/transcriptions"
    output_images = "data/processed/train/images"
    metadata_path = "data/processed/train/metadata.csv"

    if not os.path.exists(output_images): os.makedirs(output_images)

    all_data = []
    files = sorted([f for f in os.listdir(raw_images) if f.endswith('.jpg')])
    
    for f in files:
        base = os.path.splitext(f)[0]
        img_p = os.path.join(raw_images, f)
        txt_p = os.path.join(raw_trans, base + ".txt")
        
        if os.path.exists(txt_p):
            lines = segment_lines_peaks(img_p, txt_p, output_images, base)
            all_data.extend(lines)

    df = pd.DataFrame(all_data)
    df.to_csv(metadata_path, index=False, encoding='utf-8')
    print(f"\n--- Novo Dataset (Peak Detection) Gerado! ---")
    print(f"Total de linhas extraídas: {len(df)}")

if __name__ == "__main__":
    build_peak_dataset()
