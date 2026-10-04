import os, csv, unicodedata
from datetime import datetime
import numpy as np
import sounddevice as sd
import soundfile as sf

import matplotlib
try:
    matplotlib.use('MacOSX')   # en macOS; si diera guerra, usa 'TkAgg'
except Exception:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import librosa, librosa.display

# ---------- Config ----------
SR          = 22050    # Hz
DUR         = 5        # s
N_MELS      = 128
FMAX        = 8000
N_FFT       = 1024
HOP_LENGTH  = 256
DATA_DIR    = "data"
GLOBAL_CSV  = os.path.join(DATA_DIR, "metadata.csv")

CLASSES = {
    "1": "hambre",
    "2": "sueño",
    "3": "pañal",
    "4": "gases",
    "5": "dolor",
    "6": "otro",
}

# ---------- Utils ----------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def ts_parts():
    """Timestamp local + nombre y offset de zona horaria (p.ej., CEST, +0200)."""
    now = datetime.now().astimezone()
    stamp   = now.strftime("%Y%m%d_%H%M%S")
    tzname  = now.tzname() or "LOCAL"
    tzoff   = now.strftime("%z")  # +HHMM
    return stamp, tzname, tzoff

def slugify(text: str) -> str:
    """Normaliza a ASCII seguro para rutas (ñ->n, tildes fuera, espacios->_)."""
    s = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return s.replace(" ", "_").lower()

def ask_label():
    print("\nEtiquetas:")
    for k, v in CLASSES.items():
        print(f"  {k}) {v}")
    print("  q) salir")
    lab = input("→ Elige (1-6) o escribe tu propia etiqueta: ").strip().lower()
    if lab == "q":
        return None
    if lab in CLASSES:
        return CLASSES[lab]
    return lab if lab else "otro"

def append_csv(csv_path, row):
    new_file = not os.path.exists(csv_path)
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([
                "filename","label","timestamp","tzname","tzoffset",
                "sr","dur_s","n_fft","hop_length","n_mels","fmax","rel_dir"
            ])
        w.writerow(row)

def save_mel_png(y, sr, png_path):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, fmax=FMAX, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", fmax=FMAX)
    plt.colorbar(format="%+2.0f dB")
    plt.title("Espectrograma Mel")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

# ---------- Bucle principal ----------
def main():
    ensure_dir(DATA_DIR)
    try:
        print("Dispositivos de audio (resumen):")
        print(sd.query_devices())
    except Exception as e:
        print("No pude listar dispositivos:", e)

    while True:
        label = ask_label()
        if label is None:
            print("Saliendo. ¡Buen trabajo! 👶")
            break

        label_slug = slugify(label)                 # 'sueño' -> 'sueno'
        label_dir  = os.path.join(DATA_DIR, label_slug)
        ensure_dir(label_dir)

        stamp, tzname, tzoff = ts_parts()
        base    = f"{label_slug}_{stamp}"           # nombre limpio (sin tz en filename)
        wav_fn  = f"{base}.wav"
        png_fn  = f"{base}.png"
        wav_path = os.path.join(label_dir, wav_fn)
        png_path = os.path.join(label_dir, png_fn)

        print(f"\n⏺️ Grabando {DUR}s a {SR} Hz…")
        audio = sd.rec(int(DUR * SR), samplerate=SR, channels=1, dtype="float32")
        sd.wait()
        y = audio.flatten()
        print("✅ Grabación finalizada.")

        # Guardar WAV + PNG
        sf.write(wav_path, y, SR);             print(f"💾 WAV: {wav_path}")
        save_mel_png(y, SR, png_path);         print(f"🖼️ PNG: {png_path}")

        # Fila de metadatos (global y por clase)
        row = [wav_fn, label_slug, stamp, tzname, tzoff,
               SR, DUR, N_FFT, HOP_LENGTH, N_MELS, FMAX, label_slug]

        # CSV global (data/metadata.csv)
        append_csv(GLOBAL_CSV, row)
        # CSV por clase (data/<clase>/metadata.csv)
        per_label_csv = os.path.join(label_dir, "metadata.csv")
        append_csv(per_label_csv, row)

if __name__ == "__main__":
    main()