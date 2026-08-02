# Segmentasi Semantik Multi-Kelas Citra Daun Kentang Berpenyakit Menggunakan Arsitektur U-Net++

Skripsi ini menggunakan UNet++ (Nested U-Net) untuk segmentasi semantik 7 kelas (6 jenis penyakit tanaman + background) pada citra daun.

## Referensi Arsitektur

Arsitektur mengacu pada:
- Zhou, Z., Rahman Siddiquee, M. M., Tajbakhsh, N., & Liang, J. (2018). *UNet++: A Nested U-Net Architecture for Medical Image Segmentation*. DLMIA 2018.
- Zhou, Z., dkk. (2019/2020). *UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation*. IEEE Transactions on Medical Imaging.
- Repository resmi penulis (acuan arsitektur): [github.com/MrGiovanni/Nested-UNet](https://github.com/MrGiovanni/Nested-UNet)

## Status Implementasi

Arsitektur UNet++ diimplementasikan ulang secara mandiri dalam PyTorch, mengacu pada paper Zhou et al. (2018, 2019), tanpa melakukan clone langsung dari repository resmi penulis. Fungsi loss yang digunakan pada dasarnya sama seperti yang umum dipakai untuk segmentasi (kombinasi Cross-Entropy dan Dice Loss), dengan tambahan pembobotan kelas (class weighting) pada salah satu komponen loss untuk membantu menangani ketidakseimbangan jumlah data antar kelas penyakit. Pipeline training dan evaluasi disusun sendiri untuk dataset segmentasi penyakit tanaman 7 kelas.

## Sumber Dataset

Dataset yang digunakan adalah **Potato Leaf Disease Dataset in Uncontrolled Environment**, yang aslinya dipublikasikan sebagai dataset akademik oleh Shabrina dkk. (2024):

> Shabrina, N. H., Indarti, S., Maharani, R., Kristiyanti, D. A., Irmawati, Prastomo, N., & Adilah M, T. (2024). A novel dataset of potato leaf disease in uncontrolled environment. *Data in Brief*, 52, 109955. https://doi.org/10.1016/j.dib.2023.109955

Data mentah (foto asli, tanpa anotasi segmentasi) tersedia di Mendeley Data: https://data.mendeley.com/datasets/ptz377bwb8/1

Untuk kebutuhan segmentasi semantik, digunakan versi yang sudah dianotasi (mask per piksel) dari Roboflow Universe:
https://universe.roboflow.com/potato-leaf-disease-in-uncontrolled-environment/potato-leaf-disease-nzxek

Lisensi: **CC BY 4.0** (boleh dipakai ulang dengan syarat mencantumkan atribusi/sitasi di atas).

Ketujuh kelas pada dataset asli (Bacteria, Fungi, Healthy, Nematode, Pest, Phytophthora, Virus) dipetakan ke 7 label yang dipakai pada model ini (kelas "Healthy" dipetakan sebagai kelas Background/index 2).

## Struktur Kode

| File | Fungsi |
|---|---|
| `unet_model.py` | Arsitektur UNetPlusPlus (opsi drop_rate & model_size: normal/small), ModifiedHybridLoss (CE+Dice, opsi Zhou Dice Loss & pembobotan kelas berbasis IoU) |
| `dataset_loader.py` | Dataset, augmentasi, get_loaders() |
| `mesin_unet.py` | Pipeline training & evaluasi (argparse, training loop, learning rate scheduler) |
| `test_only.py` | Evaluasi model pada valid set & test set |
| `visualisasi_S16.py` | Visualisasi hasil prediksi (demo) |

## Cara Menjalankan

### 1. Instalasi

```bash
pip install -r requirements.txt
```

### 2. Struktur Dataset

```
dataset/
  train/images/*.jpg   train/masks_index/*.png
  valid/images/*.jpg   valid/masks_index/*.png
  test/images/*.jpg    test/masks_index/*.png
```

### 3. Training

```bash
python mesin_unet.py --tag S16_StepLR05_Drop01_7020 \
  --dataset_path "PATH/KE/DATASET/split_np_7020" \
  --img_size 256 --batch_size 4 --lr 0.0002 --epochs 100 \
  --optimizer adam --ce_w 0.5 --dice_w 0.5 \
  --use_zhou_dice --use_iou_weights \
  --scheduler step --step_size 40 --gamma 0.5 \
  --drop_rate 0.1
```

Ganti `PATH/KE/DATASET/split_np_7020` dengan lokasi folder dataset kamu sendiri.

### 4. Testing / Evaluasi

```bash
python test_only.py --tag S16_StepLR05_Drop01_7020 --dataset_path split_np_7020
```

Argumen `--img_size` (default 256), `--batch_size` (default 4), dan `--model_type` (default `best_miou`) tidak perlu diisi kalau memang mau pakai nilai default tersebut.

Script ini mengevaluasi model ke valid set dan test set sekaligus (Loss, mIoU, Dice/F1, Pixel Accuracy, per kelas), dan menyimpan preview prediksi di `outputs/preview_S16_StepLR05_Drop01_7020_TEST.png`.

### 5. Demo (Visualisasi Hasil)

```bash
python visualisasi_S16.py
```

Hasil visualisasi tersimpan di `outputs/visualisasi_S16/`.
