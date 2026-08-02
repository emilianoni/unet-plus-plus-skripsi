"""
visualisasi_S16.py
------------------
Visualisasi seluruh gambar test set untuk model S16.
Layout 5 panel:
  [Citra Asli] [Ground Truth] [Legend GT] [Prediksi] [Legend+Bar+Keputusan]

Jalankan:
    python visualisasi_S16.py
"""

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import albumentations as A
from albumentations.pytorch import ToTensorV2

from unet_model import UNetPlusPlus

# ==========================================
# KONFIGURASI
# ==========================================
MODEL_PATH   = 'models/model_S16_StepLR05_Drop01_7020_BEST_MIOU.pth'
DATASET_PATH = 'split_np_7020/test'
OUTPUT_DIR   = 'outputs/visualisasi_S16'
IMG_SIZE     = 256
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

KELAS = {
    0: ('Bacteria',     'yellow'),
    1: ('Fungi',        'magenta'),
    2: ('Background',   'black'),
    3: ('Nematode',     'cyan'),
    4: ('Pest',         'red'),
    5: ('Phytophthora', 'lime'),
    6: ('Virus',        'blue'),
}
WARNA_LIST = [KELAS[i][1] for i in range(7)]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# LOAD MODEL
# ==========================================
print(f"Loading model dari {MODEL_PATH}...")
model = UNetPlusPlus(
    input_channels=3,
    num_classes=7,
    deep_supervision=True,
    drop_rate=0.1,
    model_size='normal'
).to(DEVICE)
model.load_state_dict(torch.load(
    MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()
print(f"Model berhasil dimuat ke {DEVICE}")

# ==========================================
# TRANSFORM — normalisasi [0,1]
# ==========================================
transform = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.ToFloat(max_value=255.0),
    ToTensorV2()
])

# ==========================================
# LOAD DAFTAR GAMBAR TEST
# ==========================================
images_dir    = os.path.join(DATASET_PATH, 'images')
masks_idx_dir = os.path.join(DATASET_PATH, 'masks_index')

gambar_list = sorted(os.listdir(images_dir))
print(f"Ditemukan {len(gambar_list)} gambar di test set")

# ==========================================
# FUNGSI BANTU
# ==========================================
def keputusan_kelas(mask_pred):
    unik, jumlah = np.unique(mask_pred, return_counts=True)
    kelas_jumlah = {
        k: n for k, n in zip(unik, jumlah) if k != 2
    }
    if not kelas_jumlah:
        return 2, 'Background'
    kelas_terpilih = max(kelas_jumlah, key=kelas_jumlah.get)
    return kelas_terpilih, KELAS[kelas_terpilih][0]


def buat_legend_patches(kelas_hadir):
    patches = []
    for k in sorted(kelas_hadir):
        nama  = KELAS[k][0]
        warna = KELAS[k][1]
        ec = 'white' if warna == 'black' else 'none'
        patches.append(mpatches.Patch(
            facecolor=warna, edgecolor=ec,
            linewidth=0.8, label=nama))
    return patches


def gambar_colormap(mask_array):
    rgb = np.zeros((*mask_array.shape, 3), dtype=np.uint8)
    warna_rgb = {
        'yellow':  (255, 255,   0),
        'magenta': (255,   0, 255),
        'black':   (  0,   0,   0),
        'cyan':    (  0, 255, 255),
        'red':     (255,   0,   0),
        'lime':    (  0, 255,   0),
        'blue':    (  0,   0, 255),
    }
    for idx, warna in enumerate(WARNA_LIST):
        r, g, b = warna_rgb[warna]
        rgb[mask_array == idx] = [r, g, b]
    return rgb


# ==========================================
# LOOP VISUALISASI
# ==========================================
warna_hex = {
    'yellow':  '#FFFF00',
    'magenta': '#FF00FF',
    'black':   '#000000',
    'cyan':    '#00FFFF',
    'red':     '#FF0000',
    'lime':    '#00FF00',
    'blue':    '#0000FF',
}

print(f"\nMemulai visualisasi {len(gambar_list)} gambar...")
print(f"Output di: {OUTPUT_DIR}\n")

for fname in tqdm(gambar_list, desc="Visualisasi"):
    img_path   = os.path.join(images_dir, fname)
    mask_fname = os.path.splitext(fname)[0] + '.png'
    mask_path  = os.path.join(masks_idx_dir, mask_fname)

    if not os.path.exists(mask_path):
        continue

    img_asli = np.array(Image.open(img_path).convert('RGB'))
    mask_gt  = np.array(Image.open(mask_path))

    # Inferensi
    aug        = transform(image=img_asli)
    img_tensor = aug['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(img_tensor)
        if isinstance(out, list):
            final_out = torch.stack(out, dim=0).mean(dim=0)
        else:
            final_out = out
        pred = torch.argmax(final_out, dim=1)[0].cpu().numpy()

    mask_gt_resized = np.array(
        Image.fromarray(mask_gt.astype(np.uint8)).resize(
            (IMG_SIZE, IMG_SIZE), Image.NEAREST))

    # Denormalisasi — cukup clip karena sudah [0,1]
    img_vis = img_tensor[0].cpu().permute(1, 2, 0).numpy()
    img_vis = np.clip(img_vis, 0, 1)

    gt_rgb   = gambar_colormap(mask_gt_resized)
    pred_rgb = gambar_colormap(pred)

    kelas_gt_hadir = set(np.unique(mask_gt_resized).tolist())

    total_piksel = pred.size
    persen = {k: (np.sum(pred == k) / total_piksel) * 100
              for k in range(7)}

    idx_keputusan, nama_keputusan = keputusan_kelas(pred)

    # ==========================================
    # FIGURE — 5 PANEL
    # ==========================================
    fig = plt.figure(figsize=(18, 4.5))
    gs  = GridSpec(1, 5, figure=fig,
                   width_ratios=[3, 3, 1.8, 3, 2.4],
                   wspace=0.08)

    # Panel 1 — Citra Asli
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(img_vis)
    ax1.set_title('Citra Asli', fontsize=10,
                  fontweight='bold', pad=6)
    ax1.axis('off')

    # Panel 2 — Ground Truth
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(gt_rgb)
    ax2.set_title('Ground Truth', fontsize=10,
                  fontweight='bold', pad=6)
    ax2.axis('off')

    # Panel 3 — Legend GT
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')
    ax3.set_title('Kelas GT', fontsize=9,
                  fontweight='bold', pad=6)
    patches_gt = buat_legend_patches(kelas_gt_hadir)
    ax3.legend(
        handles=patches_gt,
        loc='upper left',
        bbox_to_anchor=(0.0, 0.98),
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        edgecolor='gray',
        handlelength=1.2,
        handleheight=1.2,
        borderpad=0.6,
        labelspacing=0.5
    )

    # Panel 4 — Prediksi
    ax4 = fig.add_subplot(gs[3])
    ax4.imshow(pred_rgb)
    ax4.set_title('Prediksi Model', fontsize=10,
                  fontweight='bold', pad=6)
    ax4.axis('off')

    # Panel 5 — Legend + Bar + Keputusan
    ax5 = fig.add_subplot(gs[4])
    ax5.axis('off')
    ax5.set_title('Kelas Prediksi', fontsize=9,
                  fontweight='bold', pad=6)

    kelas_tampil = [k for k in range(7) if persen[k] > 0]
    n_kelas      = len(kelas_tampil)

    if n_kelas > 0:
        bar_height = 0.7 / n_kelas
        y_start    = 0.95

        for i, k in enumerate(kelas_tampil):
            y_pos    = y_start - i * (bar_height + 0.02)
            nama_k   = KELAS[k][0]
            warna_k  = warna_hex[KELAS[k][1]]
            persen_k = persen[k]

            ax5.text(0.02, y_pos, f"{nama_k}",
                     transform=ax5.transAxes,
                     fontsize=7.5, va='top', ha='left',
                     color='black')

            ax5.add_patch(plt.Rectangle(
                (0.02, y_pos - bar_height - 0.005),
                0.82, bar_height,
                transform=ax5.transAxes,
                facecolor='#EEEEEE', edgecolor='#CCCCCC',
                linewidth=0.5, clip_on=False
            ))

            bar_len = 0.82 * (persen_k / 100)
            if bar_len > 0:
                ec_bar = '#333333' \
                         if KELAS[k][1] == 'black' else 'none'
                ax5.add_patch(plt.Rectangle(
                    (0.02, y_pos - bar_height - 0.005),
                    bar_len, bar_height,
                    transform=ax5.transAxes,
                    facecolor=warna_k,
                    edgecolor=ec_bar,
                    linewidth=0.5, clip_on=False
                ))

            ax5.text(0.86, y_pos - bar_height / 2 - 0.005,
                     f"{persen_k:.1f}%",
                     transform=ax5.transAxes,
                     fontsize=7, va='center', ha='left',
                     color='black')

        y_garis = y_start - n_kelas * (bar_height + 0.02) - 0.02
        ax5.plot([0.02, 0.98], [y_garis, y_garis],
                 color='gray', linewidth=0.5,
                 transform=ax5.transAxes)

        warna_keputusan = warna_hex[KELAS[idx_keputusan][1]]
        ec_box = '#333333' \
                 if KELAS[idx_keputusan][1] == 'black' \
                 else warna_keputusan

        ax5.text(0.5, y_garis - 0.04,
                 'Kelas Prediksi:',
                 transform=ax5.transAxes,
                 fontsize=7.5, va='top', ha='center',
                 color='#444444')
        ax5.text(0.5, y_garis - 0.13,
                 nama_keputusan,
                 transform=ax5.transAxes,
                 fontsize=9, va='top', ha='center',
                 fontweight='bold',
                 color='black',
                 bbox=dict(
                     boxstyle='round,pad=0.4',
                     facecolor=warna_keputusan,
                     edgecolor=ec_box,
                     linewidth=1.2,
                     alpha=0.85
                 ))

    nama_output = os.path.splitext(fname)[0] + '_vis.png'
    out_path    = os.path.join(OUTPUT_DIR, nama_output)
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)

print(f"\nSelesai! {len(gambar_list)} gambar tersimpan di:")
print(f"   {OUTPUT_DIR}")