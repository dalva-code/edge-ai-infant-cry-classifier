# make_splits.py
import csv, random, shutil
from pathlib import Path

SRC = Path("data_original")
DST = Path("data_split")
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}
EXT = ".wav"

random.seed(42)

def load_global_metadata(meta_path: Path):
    if not meta_path.exists():
        header = ["filename","label","timestamp","tzname","tzoffset",
                  "sr","dur_s","n_fft","hop_length","n_mels","fmax","rel_dir"]
        return header, {}
    with open(meta_path) as f:
        rdr = csv.reader(f)
        rows = list(rdr)
    header, rows = rows[0], rows[1:]
    # dict por filename
    meta = {r[0]: r for r in rows if r}
    return header, meta

def collect_label_files(label_dir: Path):
    return sorted([p for p in label_dir.glob(f"*{EXT}") if p.is_file()])

def main():
    assert SRC.exists(), "No existe la carpeta data/"
    DST.mkdir(exist_ok=True)

    header, meta_index = load_global_metadata(SRC / "metadata.csv")

    writers, files = {}, {}
    for split in SPLITS:
        split_dir = DST / split
        split_dir.mkdir(parents=True, exist_ok=True)
        csv_path = split_dir / "metadata.csv"
        f = open(csv_path, "w", newline="")
        w = csv.writer(f); w.writerow(header)
        writers[split] = (w, f)

    labels = [p.name for p in SRC.iterdir() if p.is_dir() and p.name != "_processed"]
    for lab in labels:
        label_dir = SRC / lab
        wavs = collect_label_files(label_dir)
        random.shuffle(wavs)
        n = len(wavs)
        n_train = int(n * SPLITS["train"])
        n_val = int(n * SPLITS["val"])
        splits = {
            "train": wavs[:n_train],
            "val": wavs[n_train:n_train+n_val],
            "test": wavs[n_train+n_val:]
        }
        for split, items in splits.items():
            out_dir = DST / split / lab
            out_dir.mkdir(parents=True, exist_ok=True)
            for wav in items:
                png = wav.with_suffix(".png")
                shutil.copy2(wav, out_dir / wav.name)
                if png.exists():
                    shutil.copy2(png, out_dir / png.name)
                row = meta_index.get(wav.name,
                       [wav.name, lab, "", "", "", 22050, 5, 1024, 256, 128, 8000, lab])
                writers[split][0].writerow(row)

    for _, f in writers.values():
        f.close()

    print("✅ data_split creado en:", DST.resolve())

if __name__ == "__main__":
    main()