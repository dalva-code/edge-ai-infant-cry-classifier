import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

TRAIN_DIR = Path("data_split/train")
N_VARIANTS = 8 # Subimos a 8 variantes para tener más volumen de datos

def augment_audio(y, sr):
    # 1. Ruido blanco aleatorio
    noise = y + 0.005 * np.random.randn(len(y))
    # 2. Cambio de velocidad (Time Stretch) - Muy útil para llantos
    speed = librosa.effects.time_stretch(y, rate=np.random.uniform(0.8, 1.2))
    # 3. Cambio de volumen (Gain)
    gain = y * np.random.uniform(0.7, 1.3)
    # 4. Cambio de tono (Pitch)
    pitch = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=np.random.uniform(-2, 2))
    return [noise, speed, gain, pitch]

def main():
    for label in ["gases", "sueno"]:
        folder = TRAIN_DIR / label
        originals = [f for f in folder.glob("*.wav") if "_aug" not in f.name]
        for audio_path in originals:
            y, sr = librosa.load(str(audio_path), sr=22050)
            variants = augment_audio(y, sr)
            for i, v in enumerate(variants):
                out_name = f"{audio_path.stem}_aug_v2_{i}.wav"
                sf.write(folder / out_name, v, sr)
    print("✅ Aumento de alta diversidad completado.")

if __name__ == "__main__":
    main()