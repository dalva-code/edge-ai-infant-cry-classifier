import os
import librosa
import soundfile as sf
import numpy as np

INBOX_DIR = "mobile_inbox/_processed"
OUT_DIR = "mobile_inbox"

def augment_and_save():
    # Buscamos los audios originales
    files = [f for f in os.listdir(INBOX_DIR) if f.endswith('.wav')]
    print(f"🚀 Iniciando Data Augmentation sobre {len(files)} archivos...")
    
    for f in files:
        path = os.path.join(INBOX_DIR, f)
        y, sr = librosa.load(path, sr=22050)
        
        # 1. Variación con Ruido Blanco Suave (Simula ruido de fondo en la habitación)
        noise = np.random.randn(len(y))
        y_noise = y + 0.005 * noise
        sf.write(os.path.join(OUT_DIR, f.replace('.wav', '_noise.wav')), y_noise, sr)
        
        # 2. Variación de Tono Agudo (Pitch Shift Up)
        y_pitch_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
        sf.write(os.path.join(OUT_DIR, f.replace('.wav', '_pitchUP.wav')), y_pitch_up, sr)
        
        # 3. Variación de Tono Grave (Pitch Shift Down)
        y_pitch_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
        sf.write(os.path.join(OUT_DIR, f.replace('.wav', '_pitchDOWN.wav')), y_pitch_down, sr)

    print("✅ ¡Aumento de datos completado! Nuevos audios listos en mobile_inbox/")

if __name__ == "__main__":
    augment_and_save()