import os
import sys
import time

# --- MODO DE SEGURANÇA (Early Check) ---
force_cpu = os.environ.get("FORCE_CPU") == "1"
if force_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print(">>> [DEBUG] CUDA Bloqueado via CUDA_VISIBLE_DEVICES.")

print(">>> [DEBUG] Carregando bibliotecas core...")
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np

print(">>> [DEBUG] Carregando Torch...")
import torch
# Limitar threads para evitar picos de energia e calor no notebook
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
print(f">>> [DEBUG] Threads limitadas a 1 para estabilidade.")

print(">>> [DEBUG] Carregando Transformers e PIL...")
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

print(">>> [DEBUG] Carregando Scipy e utilitários...")
from scipy.signal import find_peaks
import io
import uuid
import gc
import psutil

def get_vram_info():
    if torch.cuda.is_available():
        t = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        r = torch.cuda.memory_reserved(0) / (1024**2)
        a = torch.cuda.memory_allocated(0) / (1024**2)
        f = t - a  # Simplified free
        return f"{a:.0f}MB / {t:.0f}MB"
    return "N/A"

app = FastAPI(title="PaleographIA Engine")

# Habilitar CORS para o Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurações de Caminhos
UPLOAD_DIR = "data/uploads"
MODELS_DIR = "models/trocr-jundiai-final"
DEFAULT_MODEL = "microsoft/trocr-small-handwritten"

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Carregar Modelo e Processador
print("\n" + "="*40)
print("--- Iniciando PaleographIA Engine ---")
print("="*40)

# Verificação de Modo de Segurança (Final Check)
device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"

if force_cpu:
    print(">>> MODO DE SEGURANÇA: Force CPU ativado.")
print(f"Dispositivo de processamento: {device.upper()}")
if device == "cuda":
    print(f"GPU Detectada: {torch.cuda.get_device_name(0)}")
    print(f"VRAM disponível: {get_vram_info()}")
else:
    print(">>> Utilizando Processamento via CPU.")
    print(f">>> CPU Threads: {torch.get_num_threads()}")

try:
    path = MODELS_DIR if os.path.exists(MODELS_DIR) else DEFAULT_MODEL
    print(f"Carregando pesos de: {path} ...")
    
    processor = TrOCRProcessor.from_pretrained(path)
    model = VisionEncoderDecoderModel.from_pretrained(path).to(device)
    
    # Colocar em modo de avaliação
    model.eval()
    
    print(f"--- Modelo carregado com sucesso! ({get_vram_info()}) ---")
except Exception as e:
    print(f"!!! ERRO FATAL ao carregar modelo: {e}")
    print("DICA: Tente rodar com a variável FORCE_CPU=1 se a GPU estiver travando.")

# --- Funções de Processamento ---

def segment_lines(image_np):
    """Detecta linhas de manuscrito ignorando texturas de papel antigo (vergê) e manchas."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    # 1. Suavização para remover fios/textura do papel
    # O MedianBlur é ótimo para remover linhas finas e ruído de 'sal e pimenta'
    denoised = cv2.medianBlur(gray, 3)
    
    # 2. Binarização Adaptativa
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 31, 15
    )
    
    # 3. Limpeza de ruído e fios finos (textura italiana)
    # Usamos uma abertura morfológica com kernel pequeno para remover o que for muito fino
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean)
    
    # 4. Borramento Horizontal (Smearing)
    # Aumentamos a largura do kernel para garantir que palavras distantes se conectem
    # e a altura para não perdermos letras com traços finos
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 5))
    dilated = cv2.dilate(thresh, kernel_line, iterations=2)
    
    # 5. Encontrar contornos
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    line_data = []
    img_h, img_w = image_np.shape[:2]
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # FILTROS GEOMÉTRICOS:
        # - Largura mínima (evita manchas isoladas)
        # - Proporção (linhas são muito mais largas que altas)
        # - Área (evita sujeira)
        aspect_ratio = w / float(h)
        if w > img_w * 0.15 and aspect_ratio > 3.0 and h > 20: 
            padding = 10
            y_start = max(0, y - padding)
            y_end = min(img_h, y + h + padding)
            
            line_img = image_np[y_start:y_end, :]
            line_data.append({
                "image": line_img,
                "y_start": int(y_start),
                "y_end": int(y_end)
            })
    
    # Ordenar de cima para baixo
    line_data.sort(key=lambda x: x["y_start"])
    
    # Debug no console
    print(f"   [SEGMENTAÇÃO] {len(contours)} contornos brutos -> {len(line_data)} linhas filtradas.")
    
    return line_data

# --- Endpoints API ---

@app.get("/")
def health_check():
    model_type = "TrOCR-Small" if "small" in DEFAULT_MODEL else "TrOCR-Base"
    return {
        "status": "online", 
        "engine": "PaleographIA",
        "model": model_type, 
        "device": device,
        "vram_usage": get_vram_info() if device == "cuda" else "N/A"
    }

@app.post("/process-folio")
async def process_folio(file: UploadFile = File(...)):
    """Recebe um folio, segmenta em linhas e retorna as linhas e o texto sugerido."""
    try:
        # Ler imagem
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Imagem inválida")

        # Gerar ID único para esta sessão
        session_id = str(uuid.uuid4())
        session_path = os.path.join(UPLOAD_DIR, session_id)
        os.makedirs(session_path)

        # Segmentar linhas
        line_data = segment_lines(img)
        results = []

        print(f"Processando folio: {len(line_data)} linhas detectadas.")

        # Altura total para cálculo de porcentagem no frontend
        img_height = img.shape[0]

        # Usar Inference Mode para economizar memória e processamento
        with torch.inference_mode():
            for idx, item in enumerate(line_data):
                line_img = item["image"]
                
                # Salvar recorte da linha
                line_filename = f"line_{idx}.jpg"
                line_path = os.path.join(session_path, line_filename)
                cv2.imwrite(line_path, line_img)

                # Transcrever com TrOCR
                pil_img = Image.fromarray(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
                pixel_values = processor(pil_img, return_tensors="pt").pixel_values.to(device)
                
                # Transcrever com TrOCR (Modo literal para evitar alucinações)
                generated_ids = model.generate(
                    pixel_values, 
                    max_new_tokens=48, 
                    repetition_penalty=1.5,
                    num_beams=1,  # Greedy Search: lê letra por letra, sem inventar frases
                    early_stopping=False
                )
                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                # Pequena pausa para resfriamento (Notebook safety)
                time.sleep(0.3)

                results.append({
                    "id": idx,
                    "text": text,
                    "image_url": f"/lines/{session_id}/{line_filename}",
                    "y_percent": round((item["y_start"] / img_height) * 100, 2),
                    "height_percent": round(((item["y_end"] - item["y_start"]) / img_height) * 100, 2)
                })
                
                # Feedback de progresso no console
                if (idx + 1) % 5 == 0 or (idx + 1) == len(line_data):
                    print(f"   > Transcritas {idx + 1}/{len(line_data)} linhas... ({get_vram_info()})")

        # Limpar cache e disparar GC
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        print(f"Limpeza concluída. RAM Livre: {psutil.virtual_memory().available / (1024**2):.0f}MB")

        return {
            "session_id": session_id,
            "total_lines": len(line_data),
            "lines": results
        }

    except Exception as e:
        print(f"Erro no processamento: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
