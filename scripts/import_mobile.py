# import_mobile.py
import os, csv, unicodedata, shutil
from datetime import datetime
from pathlib import Path
import librosa, soundfile as sf
import numpy as np
import matplotlib
try:
    matplotlib.use('MacOSX')
except Exception:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import librosa.display

# --- Config ---
SRC_INBOX   = Path("mobile_inbox")
DATA_DIR    = Path("data")
GLOBAL_CSV  = DATA_DIR / "metadata.csv"
SR          = 22050
N_MELS      = 128
FMAX        = 8000
N_FFT       = 1024
HOP_LENGTH  = 256
EXTS        = {".wav", ".m4a", ".mp3", ".aac", ".ogg"}

CLASSES = {
    "1": "hambre",
    "2": "sueño",
    "3": "pañal",
    "4": "gases",
    "5": "dolor",
    "6": "otro",
}

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def slugify(text: str) -> str:
    s = unicodedata.normalize("NFKD", text).encode("ascii","ignore").decode("ascii")
    return s.replace(" ", "_").lower()

def ts_parts():
    now = datetime.now().astimezone()
    return now.strftime("%Y%m%d_%H%M%S"), (now.tzname() or "LOCAL"), now.strftime("%z")

def append_csv(csv_path: Path, row):
    new_file = not csv_path.exists()
    ensure_dir(csv_path.parent)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([
                "filename","label","timestamp","tzname","tzoffset",
                "sr","dur_s","n_fft","hop_length","n_mels","fmax","rel_dir"
            ])
        w.writerow(row)

def save_mel_png(y, sr, png_path: Path):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, fmax=8000, n_fft=1024, hop_length=256
    )
    S_dB = librosa.power_to_db(S, ref=np.max)
    ensure_dir(png_path.parent)
    plt.figure(figsize=(10,4))
    librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", fmax=8000)
    plt.colorbar(format="%+2.0f dB"); plt.title("Espectrograma Mel")
    plt.tight_layout(); plt.savefig(png_path, dpi=150); plt.close()

def guess_label_from_path(p: Path):
    parent = slugify(p.parent.name)
    class_slugs = {slugify(v) for v in CLASSES.values()}
    if parent in class_slugs:
        return parent
    name = slugify(p.stem)
    for v in CLASSES.values():
        sv = slugify(v)
        if sv in name:
            return sv
    return None

def ask_label():
    print("\nEtiquetas:")
    for k,v in CLASSES.items(): print(f"  {k}) {v}")
    print("  q) saltar / terminar")
    lab = input("→ Elige (1-6) o escribe etiqueta: ").strip().lower()
    if lab == "q": return None
    if lab in CLASSES: return slugify(CLASSES[lab])
    return slugify(lab) if lab else "otro"

def main():
    ensure_dir(SRC_INBOX)
    ensure_dir(DATA_DIR)
    files = [p for p in SRC_INBOX.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]
    if not files:
        print(f"No hay audios en {SRC_INBOX}. Copia aquí tus grabaciones del móvil.")
        return

    print(f"Encontrados {len(files)} archivos para importar.")
    for i, src in enumerate(sorted(files)):
        print("\n—"*40); print(f"Archivo: {src}")
        label = guess_label_from_path(src)
        if label: print(f"Etiqueta detectada: {label}")
        else:
            label = ask_label()
            if label is None:
                print("Saliendo por petición del usuario."); break

        label_dir = DATA_DIR / label
        ensure_dir(label_dir)

        try:
            y, sr = librosa.load(str(src), sr=SR, mono=True)
        except Exception as e:
            print(f"❌ No pude leer {src.name}: {e}"); continue

        stamp, tzname, tzoff = ts_parts()
        # Usamos el nombre original del archivo (src.stem) para mantener tus horas
        base = f"{label}_{src.stem}" 
        wav_fn, png_fn = f"{base}.wav", f"{base}.png"
        wav_path, png_path = label_dir / wav_fn, label_dir / png_fn

        sf.write(str(wav_path), y, SR)
        save_mel_png(y, SR, png_path)

        row = [wav_fn, label, stamp, tzname, tzoff,
               SR, round(len(y)/SR), N_FFT, HOP_LENGTH, N_MELS, FMAX, label]
        append_csv(GLOBAL_CSV, row)
        append_csv(label_dir / "metadata.csv", row)

        print(f"✅ Importado: {wav_path}")
        processed_dir = SRC_INBOX / "_processed"
        ensure_dir(processed_dir)
        shutil.move(str(src), processed_dir / src.name)

    print("\n🎉 Importación terminada.")

if __name__ == "__main__":
    main()