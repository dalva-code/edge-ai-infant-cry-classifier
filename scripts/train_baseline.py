import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

ROOT = Path('.').resolve()
CONFIG = yaml.safe_load(open(ROOT / 'config.yaml','r'))
FEAT_IDX = Path(CONFIG['paths']['features']) / 'features_index.csv'
LABELS_CSV = Path(CONFIG['paths']['labels_csv'])
MODELS_DIR = Path(CONFIG['paths']['models'])
RESULTS_DIR = Path(CONFIG['paths']['results'])
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class LogMelDataset(Dataset):
    def __init__(self, df, fixed_frames=128, feature_col='logmel_path', label_col='class', class_to_idx=None):
        self.df = df.reset_index(drop=True)
        self.fixed_frames = fixed_frames
        self.feature_col = feature_col
        self.label_col = label_col
        self.class_to_idx = class_to_idx or {c:i for i,c in enumerate(sorted(self.df[self.label_col].unique()))}
    def __len__(self): return len(self.df)
    def _pad_or_trim(self, m):
        T = m.shape[1]
        if T == self.fixed_frames: return m
        if T > self.fixed_frames: return m[:, :self.fixed_frames]
        pad = self.fixed_frames - T
        return np.pad(m, ((0,0),(0,pad)), mode='constant')
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = Path(row[self.feature_col])
        m = np.load(path)
        m = self._pad_or_trim(m)
        m = (m - m.mean()) / (m.std() + 1e-9)
        x = torch.from_numpy(m).float().unsqueeze(0)
        y = torch.tensor(self.class_to_idx[row[self.label_col]], dtype=torch.long)
        return x, y

class SmallCNN(nn.Module):
    def __init__(self, n_mels=64, n_classes=3, hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(32, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, n_classes)
    def forward(self, x):
        h = self.net(x)
        h = h.mean(dim=[2,3])
        h = self.dropout(h)
        return self.classifier(h)

def plot_confusion(cm, classes, out_png):
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title('Matriz de confusión')
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.yticks(range(len(classes)), classes)
    plt.xlabel('Predicción'); plt.ylabel('Real')
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def main():
    if not FEAT_IDX.exists():
        print(f"[error] No existe {FEAT_IDX}. Ejecuta scripts/extract_features.py primero.")
        return
    df_idx = pd.read_csv(FEAT_IDX)
    df_lbl = pd.read_csv(LABELS_CSV)
    df = df_idx.merge(df_lbl[['filename','clase']], on='filename', how='inner')
    df = df.rename(columns={'clase':'class'})
    df = df[~df['class'].isna()]
    if len(df) < 10:
        print(f"[warn] Muy pocos ejemplos ({len(df)}). Añade más audios para entrenar.")
    classes = sorted(df['class'].unique())
    class_to_idx = {c:i for i,c in enumerate(classes)}
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['class'], random_state=CONFIG['training']['seed'])

    train_set = LogMelDataset(train_df, class_to_idx=class_to_idx)
    val_set   = LogMelDataset(val_df, class_to_idx=class_to_idx)
    train_loader = DataLoader(train_set, batch_size=CONFIG['training']['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=CONFIG['training']['batch_size'], shuffle=False)

    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    n_classes = len(classes)
    model = SmallCNN(n_mels=CONFIG['features']['n_mels'], n_classes=n_classes,
                     hidden=CONFIG['model']['cnn_hidden'], dropout=CONFIG['model']['dropout']).to(device)

    counts = train_df['class'].value_counts().reindex(classes).fillna(0).values
    weights = 1.0 / (counts + 1e-9)
    weights = torch.tensor(weights / weights.sum() * n_classes, dtype=torch.float32).to(device)

    crit = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG['training']['lr'])

    best_f1 = -1
    for epoch in range(CONFIG['training']['epochs']):
        model.train()
        tr_loss, tr_ok, tr_n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
            tr_ok += (logits.argmax(1) == yb).sum().item()
            tr_n += xb.size(0)

        model.eval()
        va_ok, va_n = 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                va_ok += (logits.argmax(1) == yb).sum().item()
                va_n += xb.size(0)
                y_true.extend(yb.cpu().numpy().tolist())
                y_pred.extend(logits.argmax(1).cpu().numpy().tolist())

        report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
        f1_macro = report['macro avg']['f1-score']
        print(f"Epoch {epoch+1:02d} | train_acc={tr_ok/tr_n:.3f} val_acc={va_ok/va_n:.3f} val_f1_macro={f1_macro:.3f}")

        if f1_macro > best_f1:
            best_f1 = f1_macro
            torch.save({'model_state': model.state_dict(), 'classes': classes}, MODELS_DIR / 'baseline.pt')

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    plot_confusion(cm, classes, RESULTS_DIR / 'cm_baseline.png')
    with open(RESULTS_DIR / 'metrics_baseline.json','w') as f:
        json.dump({'classes': classes, 'best_val_f1_macro': best_f1}, f, indent=2)
    print(f"[ok] Guardado modelo en {MODELS_DIR/'baseline.pt'}")
    print(f"[ok] Métricas en {RESULTS_DIR/'metrics_baseline.json'}")
    print(f"[ok] Matriz de confusión en {RESULTS_DIR/'cm_baseline.png'}")

if __name__ == '__main__':
    main()
