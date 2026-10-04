import json
from pathlib import Path
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import yaml
from tqdm import tqdm

ROOT = Path('.').resolve()
CONFIG = yaml.safe_load(open(ROOT / 'config.yaml', 'r'))

RAW = Path(CONFIG['paths']['raw'])
FEAT_DIR = Path(CONFIG['paths']['features'])
LABELS_CSV = Path(CONFIG['paths']['labels_csv'])

SR = CONFIG['audio']['sample_rate']
N_MELS = CONFIG['features']['n_mels']
N_MFCC = CONFIG['features']['n_mfcc']
WIN_MS = CONFIG['features']['win_length_ms']
HOP_MS = CONFIG['features']['hop_length_ms']

WIN = int(SR * WIN_MS / 1000)
HOP = int(SR * HOP_MS / 1000)

FEAT_DIR.mkdir(parents=True, exist_ok=True)

def load_wav(path, sr):
    y, s = librosa.load(path, sr=sr, mono=True)
    peak = np.max(np.abs(y)) + 1e-9
    y = y / peak
    return y, s

def extract_logmel(y, sr, n_mels, win, hop):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=win*2, hop_length=hop, win_length=win)
    logmel = librosa.power_to_db(S, ref=np.max)
    return logmel.astype(np.float32)

def extract_mfcc(y, sr, n_mfcc, win, hop):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=win*2, hop_length=hop, win_length=win)
    mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-9)
    return mfcc.astype(np.float32)

def main():
    if not LABELS_CSV.exists():
        print(f"[error] No existe labels.csv en {LABELS_CSV}")
        return
    df = pd.read_csv(LABELS_CSV)
    if 'filename' not in df.columns:
        print("[error] labels.csv debe contener columna 'filename'")
        return

    index_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        fname = row['filename']
        wav_path = RAW / fname
        if not wav_path.exists():
            print(f"[skip] No existe: {wav_path}")
            continue
        try:
            y, _ = load_wav(wav_path, SR)
            logmel = extract_logmel(y, SR, N_MELS, WIN, HOP)
            mfcc = extract_mfcc(y, SR, N_MFCC, WIN, HOP)

            stem = Path(fname).stem
            npy_logmel = FEAT_DIR / f"{stem}_logmel.npy"
            npy_mfcc = FEAT_DIR / f"{stem}_mfcc.npy"
            np.save(npy_logmel, logmel)
            np.save(npy_mfcc, mfcc)

            index_rows.append({
                'filename': fname,
                'class': row.get('clase', ''),
                'logmel_path': str(npy_logmel),
                'mfcc_path': str(npy_mfcc),
                'session_id': row.get('session_id', ''),
            })
        except Exception as e:
            print(f"[error] {fname}: {e}")

    if index_rows:
        idx = pd.DataFrame(index_rows)
        idx.to_csv(FEAT_DIR / 'features_index.csv', index=False)
        print(f"[ok] Guardado índice de features: {FEAT_DIR / 'features_index.csv'}")
    else:
        print("[warn] No se generaron features. Revisa labels.csv y data/raw/")

if __name__ == '__main__':
    main()
