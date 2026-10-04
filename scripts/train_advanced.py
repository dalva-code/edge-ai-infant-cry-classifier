import os, glob, librosa, numpy as np
from pathlib import Path
from sklearn.metrics import classification_report
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- Configuración Estándar ---
SR, N_MELS, FMAX, N_FFT, HOP = 22050, 128, 8000, 1024, 256
IMSIZE = 224
BATCH = 16
ROOT = Path("data_split")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def get_labels():
    return sorted([p.name for p in (ROOT/"train").iterdir() if p.is_dir()])

LABELS = get_labels()
IDX = {l:i for i,l in enumerate(LABELS)}

class MelDataset(Dataset):
    def __init__(self, split):
        self.paths, self.labels = [], []
        base = ROOT / split
        for lab in LABELS:
            # Buscamos todos los audios (originales y aumentados v2)
            for wav in glob.glob(str(base / lab / "*.wav")):
                self.paths.append(wav)
                self.labels.append(IDX[lab])
        self.to_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((IMSIZE, IMSIZE)),
            transforms.Lambda(lambda x: x.repeat(3,1,1)),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        path = self.paths[i]
        y, sr = librosa.load(path, sr=SR, mono=True)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX, n_fft=N_FFT, hop_length=HOP)
        S_dB = librosa.power_to_db(S, ref=np.max)
        S_img = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + 1e-8)
        x = self.to_img(S_img.astype(np.float32))
        return x, self.labels[i]

def get_loaders():
    tr = DataLoader(MelDataset("train"), batch_size=BATCH, shuffle=True)
    va = DataLoader(MelDataset("val"),   batch_size=BATCH, shuffle=False)
    te = DataLoader(MelDataset("test"),  batch_size=BATCH, shuffle=False)
    return tr, va, te

def train_one_epoch(model, loader, opt, crit):
    model.train()
    loss_sum = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * x.size(0)
    return loss_sum / len(loader.dataset)

def validate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total if total else 0

def main():
    if not LABELS:
        print("⚠️ No hay datos. Revisa data_split/train")
        return
    
    tr, va, te = get_loaders()
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(LABELS))
    model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    best_va = 0

    # --- FASE 1: CONGELADO (5 Épocas) ---
    print("\n🚀 FASE 1: Entrenando solo la capa final (Cuerpo congelado)...")
    for param in model.features.parameters():
        param.requires_grad = False
    
    optimizer = AdamW(model.parameters(), lr=1e-3)
    
    for epoch in range(1, 6):
        loss = train_one_epoch(model, tr, optimizer, criterion)
        acc = validate(model, va)
        print(f"Época {epoch} [Congelado]: Loss={loss:.4f} | Val_Acc={acc:.3f}")

    # --- FASE 2: DESCONGELADO + SCHEDULER (15 Épocas) ---
    print("\n🧠 FASE 2: Ajuste fino (Descongelando todo el cerebro)...")
    for param in model.parameters():
        param.requires_grad = True
    
    # Bajamos el LR para que el ajuste sea muy preciso
    optimizer = AdamW(model.parameters(), lr=1e-5)
    # El Scheduler ayudará a no estancarse
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    for epoch in range(6, 21):
        loss = train_one_epoch(model, tr, optimizer, criterion)
        acc = validate(model, va)
        scheduler.step(acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Época {epoch} [Fine-tuning]: Loss={loss:.4f} | Val_Acc={acc:.3f} | LR={current_lr:.6f}")
        
        if acc > best_va:
            best_va = acc
            torch.save(model.state_dict(), "best_model_avanzado_80.pt")
            print("⭐ Nuevo mejor modelo guardado!")

    # --- TEST FINAL ---
    print("\n🏁 EVALUACIÓN FINAL SOBRE DATOS VIRGENES:")
    model.load_state_dict(torch.load("best_model_avanzado_80.pt"))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in te:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            y_true += y.cpu().tolist()
            y_pred += out.argmax(1).cpu().tolist()
    
    print(classification_report(y_true, y_pred, target_names=LABELS))

if __name__ == "__main__":
    main()