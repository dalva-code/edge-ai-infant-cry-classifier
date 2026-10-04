import torch, librosa, os, numpy as np
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path

# Configuración técnica (tu estándar)
SR, N_MELS, FMAX, N_FFT, HOP = 22050, 128, 8000, 1024, 256
IMSIZE = 224
LABELS = ['gases', 'sueno'] 
TEST_DIR = Path("data_split/test")
MODEL_PATH = "best_model_fase_contaminada.pt" # Tu modelo antiguo

def evaluate():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Cargar el "Cerebro" antiguo
    model = models.mobilenet_v3_small()
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(LABELS))
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: No encuentro el archivo {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMSIZE, IMSIZE)),
        transforms.Lambda(lambda x: x.repeat(3,1,1)),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    correctos = 0
    total = 0

    print(f"📊 Evaluando el Test Set Limpio con el modelo antiguo...\n")
    print(f"{'ARCHIVO':<30} | {'REAL':<10} | {'PREDICCIÓN':<10} | {'ESTADO'}")
    print("-" * 70)

    for label in LABELS:
        folder = TEST_DIR / label
        for audio_path in folder.glob("*.wav"):
            # Procesamiento
            y, sr = librosa.load(str(audio_path), sr=SR, mono=True)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX, n_fft=N_FFT, hop_length=HOP)
            S_dB = librosa.power_to_db(S, ref=np.max)
            S_img = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + 1e-8)
            x = transform(S_img.astype(np.float32)).unsqueeze(0)

            with torch.no_grad():
                out = model(x.to(device))
                pred_idx = out.argmax(1).item()
                pred_label = LABELS[pred_idx]

            total += 1
            es_correcto = (pred_label == label)
            if es_correcto: correctos += 1
            
            estado = "✅" if es_correcto else "❌"
            print(f"{audio_path.name:<30} | {label:<10} | {pred_label:<10} | {estado}")

    accuracy = (correctos / total) * 100 if total > 0 else 0
    print("-" * 70)
    print(f"📈 RESULTADO FINAL: {accuracy:.1f}% de acierto sobre datos limpios.")

if __name__ == "__main__":
    evaluate()