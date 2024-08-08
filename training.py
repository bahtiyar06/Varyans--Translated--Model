# !/usr/bin/env python3
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import MarianMTModel, MarianTokenizer
import os
import numpy as np
from sklearn.model_selection import train_test_split

print("Gerekli kütüphaneler yüklendi.")

# Tokenizer ve model
model_name = 'Helsinki-NLP/opus-mt-tr-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
print("Model ve tokenizer yüklendi.")

class TranslationDataset(Dataset):
    def __init__(self, source_file, target_file):
        self.source_file = source_file
        self.target_file = target_file

        with open(self.source_file, 'r', encoding='utf-8') as src_f, open(self.target_file, 'r', encoding='utf-8') as tgt_f:
            self.source_lines = src_f.readlines()
            self.target_lines = tgt_f.readlines()
        print(f"Veri dosyaları yüklendi: {len(self.source_lines)} satır.")

        self.train_source, self.val_source, self.train_target, self.val_target = train_test_split(
            self.source_lines, self.target_lines, test_size=0.1, random_state=42
        )
        print("Veri seti eğitim ve doğrulama olarak bölündü.")

    def __len__(self):
        return len(self.train_source)

    def __getitem__(self, idx):
        return self.train_source[idx].strip(), self.train_target[idx].strip()

source_file = 'CCMatrix.en-tr-tr-revize.txt'
target_file = 'CCMatrix.en-tr-en-revize.txt'
dataset = TranslationDataset(source_file, target_file)
print("Dataset oluşturuldu.")

batch_size = 24
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

val_dataset = list(zip(dataset.val_source, dataset.val_target))
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
print("Veri yükleyiciler oluşturuldu.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
print("Optimizasyon ve planlayıcı ayarlandı.")

def save_checkpoint(epoch, model, optimizer, scheduler, save_dir, best=False):
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    filename = 'model_best.pt' if best else f'model_epoch_{epoch+1}.pt'
    torch.save(checkpoint, os.path.join(save_dir, filename))
    print(f"Model ağırlıkları '{filename}' dosyasına kaydedildi.")

def load_checkpoint(filepath, model, optimizer, scheduler):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        return epoch
    else:
        return 0

def evaluate_model(model, dataloader, tokenizer, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            source_texts, target_texts = batch
            inputs = tokenizer(source_texts, return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)
            labels = tokenizer(target_texts, return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)

            outputs = model(**inputs, labels=labels['input_ids'])
            loss = outputs.loss
            total_loss += loss.item()

            # Bellek temizleme
            del inputs, labels, outputs, loss
            torch.cuda.empty_cache()
    model.train()  # Modeli eğitim moduna geri al
    return total_loss / len(dataloader)

save_directory = './translated-model'
os.makedirs(save_directory, exist_ok=True)

best_val_loss = float('inf')
resume_from_checkpoint = True
checkpoint_path = os.path.join(save_directory, 'model_best.pt')

if resume_from_checkpoint:
    start_epoch = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
else:
    start_epoch = 0

print("Eğitim başlıyor.")
for epoch in range(start_epoch, 5):
    model.train()
    total_loss = 0.0
    for batch_idx, batch in enumerate(dataloader):
        print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}")
        source_texts, target_texts = batch
        inputs = tokenizer(source_texts, return_tensors='pt', padding=True, truncation=True, max_length=256)
        labels = tokenizer(target_texts, return_tensors='pt', padding=True, truncation=True, max_length=256)

        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = {k: v.to(device) for k, v in labels.items()}

        outputs = model(**inputs, labels=labels['input_ids'])
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        del inputs, labels, outputs, loss
        torch.cuda.empty_cache()

    scheduler.step()

    avg_train_loss = total_loss / len(dataloader)
    avg_val_loss = evaluate_model(model, val_dataloader, tokenizer, device)

    print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")

    save_checkpoint(epoch, model, optimizer, scheduler, save_directory)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        save_checkpoint(epoch, model, optimizer, scheduler, save_directory, best=True)

print("Model eğitimi tamamlandı.")
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
print(f"Model ve tokenizer '{save_directory}' dizinine kaydedildi.")
