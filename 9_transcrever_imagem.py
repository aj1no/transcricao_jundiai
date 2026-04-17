import os
import cv2
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import sys

# Tentar carregar o modelo treinado, caso contrário usar o base
MODEL_PATH = "./models/trocr-jundiai-final"
DEFAULT_MODEL = "microsoft/trocr-base-handwritten"

def load_model():
    path = MODEL_PATH if os.path.exists(MODEL_PATH) else DEFAULT_MODEL
    print(f"Carregando modelo de: {path}")
    processor = TrOCRProcessor.from_pretrained(path)
    model = VisionEncoderDecoderModel.from_pretrained(path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

def transcribe_image(image, processor, model, device):
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def run_inference(image_path):
    processor, model, device = load_model()
    
    # Carregar imagem
    if not os.path.exists(image_path):
        print(f"Erro: Arquivo {image_path} não encontrado.")
        return

    # Se for uma imagem pequena (já recortada como linha)
    image = Image.open(image_path).convert("RGB")
    text = transcribe_image(image, processor, model, device)
    
    print("-" * 30)
    print(f"Resultado da Transcrição:")
    print(f"'{text}'")
    print("-" * 30)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_inference(sys.argv[1])
    else:
        # Tentar com uma imagem do dataset caso não informe o caminho
        sample_path = "data/processed/train/images/codigo_de_posturas_-_1831_p001_p001.jpg"
        if os.path.exists(sample_path):
            run_inference(sample_path)
        else:
            print("Uso: python 9_transcrever_imagem.py <caminho_da_imagem>")
