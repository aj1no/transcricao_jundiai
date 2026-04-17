import os
import torch
import pandas as pd
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# --- SEGURANÇA: Limitar CPU para não travar o Notebook ---
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Caminhos
MODEL_BASE = "microsoft/trocr-small-handwritten"
OUTPUT_DIR = "./models/trocr-jundiai-final"
CSV_PATH = "data/processed/train/metadata.csv"
IMG_DIR = "data/processed/train/images"

class SimpleJundiaiDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['file_name'].iloc[idx]
        img_path = os.path.join(self.root_dir, file_name)
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(
            self.df['text'].iloc[idx], 
            padding="max_length", 
            max_length=self.max_target_length
        ).input_ids
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        return {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}

def run_fast_train():
    if not os.path.exists(CSV_PATH):
        print(f"Erro: {CSV_PATH} não encontrado.")
        return

    print("\n[INFO] Carregando dados (280 linhas)...")
    df = pd.read_csv(CSV_PATH)
    
    # Usar apenas uma amostra se for muito lento, mas 280 é ok
    processor = TrOCRProcessor.from_pretrained(MODEL_BASE)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_BASE)

    # Configurações do modelo
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    dataset = SimpleJundiaiDataset(IMG_DIR, df, processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./models/checkpoints_fast",
        per_device_train_batch_size=1,
        num_train_epochs=3, # Rápido: apenas 3 passadas
        learning_rate=4e-5,
        logging_steps=10,
        save_strategy="no", # Não salvar checkpoints para poupar espaço
        report_to="none",
        fp16=False, # CPU não suporta fp16 bem
        use_cpu=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )

    print("\n[ALERTA] Inciando Treino Relâmpago (CPU)...")
    print("Isso vai levar cerca de 5-10 minutos. O computador pode ficar lento.")
    
    trainer.train()

    print(f"\n[SUCESSO] Salvando modelo em {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("Treino concluído!")

if __name__ == "__main__":
    run_fast_train()
