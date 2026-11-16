import sys
from pathlib import Path

# Tambahkan root proyek ke sys.path agar modul starter dan games dapat diimpor
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from starter.load_atd import load as load_atd
from starter.utils import draw_pitch, plot_frame

# 1. Muat dataset tracking (menggunakan loader starter)
dataset = load_atd()

# 2. Muat hasil prediksi CSV dari model V2-r1
# (Ganti path ke lokasi file CSV V2-r1 Anda)
preds_df = pd.read_csv("local_official_artifacts/v2-r1-operator-run1/submission.csv")

# 3. Pilih nomor frame yang ingin Anda inspeksi (misal frame 12000)
frame_target = 40950

# Ambil label prediksi untuk frame tersebut (H, A, D, atau X)
predicted_label = preds_df.loc[preds_df["frame"] == frame_target, "label"].values[0]

# Terjemahkan label ke teks deskriptif
label_desc = {
    "H": "Barcelona (Home)",
    "A": "Athletic Club (Away)",
    "D": "Dead Ball (Bola Mati / Out)",
    "X": "Di Luar Periode Permainan"
}.get(predicted_label, predicted_label)

# 4. Gambar visualisasi lapangan dan pemain
fig, ax = plt.subplots(figsize=(12, 7))

# Gambar pemain dan bola pada frame tersebut
plot_frame(dataset, frame_index=frame_target, ax=ax)

# Tambahkan label hasil prediksi V2-r1 pada judul gambar
ax.set_title(
    f"Visualisasi Frame {frame_target} | Prediksi V2-r1: {predicted_label} ({label_desc})",
    fontsize=14, fontweight="bold", pad=12
)

# 5. Tampilkan gambar ke layar atau simpan ke file
plt.tight_layout()
plt.savefig(f"visualisasi_frame_{frame_target}_v2.png", dpi=200)
plt.show()