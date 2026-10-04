import sys, torch, librosa, numpy as np
import torch.nn as nn
from torchvision import models, transforms

# Configuración idéntica a tu entrenamiento para no romper la física del sonido
SR, N_MELS, FMAX, N_FFT, HOP = 22050, 128, 8000, 1024, 256
IMSIZE = 224
LABELS = ['gases', 'sueno'] 

def predict(audio_path):
    print(f"⏳ Procesando el llanto de Elliot...")
    
    # 1. Cargar audio y convertir a Espectrograma de Mel
    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX, n_fft=N_FFT, hop_length=HOP)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_img = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + 1e-8)

    # 2. Transformaciones (igual que en tu Dataset)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMSIZE, IMSIZE)),
        transforms.Lambda(lambda x: x.repeat(3,1,1)),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    x = transform(S_img.astype(np.float32)).unsqueeze(0) # Añadir dimensión de "lote"

    # 3. Cargar el "Cerebro" que acabas de entrenar
    model = models.mobilenet_v3_small()
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(LABELS))
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # IMPORTANTE: Cargamos tu best_model.pt
    model.load_state_dict(torch.load("best_model_fase_contaminada.pt", map_location=device))
    model.to(device)
    model.eval() # Modo evaluación (no aprende, solo deduce)

    # 4. El Veredicto
    with torch.no_grad():
        out = model(x.to(device))
        prob = torch.nn.functional.softmax(out, dim=1)[0]
        pred_idx = out.argmax(1).item()
        
    print(f"\n======================================")
    print(f"🎧 Archivo: {audio_path}")
    print(f"🤖 DIAGNÓSTICO DE LA IA: ¡{LABELS[pred_idx].upper()}!")
    print(f"📊 Seguridad: Gases ({prob[0]*100:.1f}%) | Sueño ({prob[1]*100:.1f}%)")
    print(f"======================================\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Uso correcto: python3 predecir_llanto.py ruta_al_audio.wav")
    else:
        predict(sys.argv[1])