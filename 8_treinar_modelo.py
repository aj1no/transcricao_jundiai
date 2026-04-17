import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
from sklearn.model_selection import train_test_split
import jiwer

# Configurações do ambiente
os.environ["WANDB_DISABLED"] = "true"  # Desabilita o wandb para evitar logs extras

class JundiaiDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Caminho da imagem
        file_name = self.df['file_name'].iloc[idx]
        img_path = os.path.join(self.root_dir, file_name)
        
        # Carregar e processar imagem
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        # Carregar e processar texto
        labels = self.processor.tokenizer(
            self.df['text'].iloc[idx], 
            padding="max_length", 
            max_length=self.max_target_length
        ).input_ids
        
        # Importante: substituir padding token por -100 para ser ignorado na loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

def compute_metrics(pred):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = jiwer.cer(label_str, pred_str)
    return {"cer": cer}

def train():
    # 1. Carregar Dados
    csv_path = "data/processed/train/metadata.csv"
    img_dir = "data/processed/train/images"
    
    if not os.path.exists(csv_path):
        print(f"Erro: {csv_path} não encontrado. Execute o segmentador primeiro.")
        return

    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=0.1)
    
    # 2. Carregar Processador e Modelo
    # Usando o 'small' para economizar VRAM na 1050Ti
    model_name = "microsoft/trocr-small-handwritten"
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    # Configurações do modelo
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # 3. Criar Datasets
    train_dataset = JundiaiDataset(img_dir, train_df, processor)
    val_dataset = JundiaiDataset(img_dir, val_df, processor)

    # 4. Configurar Treinamento
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=2,  # Baixo para economizar VRAM (GTX 1050 Ti tem 4GB)
        per_device_eval_batch_size=2,
        fp16=True,                     # Usar ponto flutuante de 16 bits se possível
        output_dir="./models/trocr-jundiai-checkpoints",
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        num_train_epochs=20,
        learning_rate=5e-5,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to="none"
    )

    # 5. Iniciar Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    print("\n--- Iniciando Treinamento ---")
    print(f"Dispositivo: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    trainer.train()
    
    # Salvar final
    model.save_pretrained("./models/trocr-jundiai-final")
    processor.save_pretrained("./models/trocr-jundiai-final")
    print("\n--- Treinamento Concluído! Modelo salvo em ./models/trocr-jundiai-final ---")

if __name__ == "__main__":
    train()
