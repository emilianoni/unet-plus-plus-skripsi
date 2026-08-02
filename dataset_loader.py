import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2


# ==========================================
# AUGMENTASI
# ==========================================
def get_transform(img_size, use_augment=False, aug_type='balanced'):
    transform_list = [A.Resize(height=img_size, width=img_size)]

    if use_augment:
        if aug_type == 'spatial':
            transform_list.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=45, p=0.5,
                         border_mode=cv2.BORDER_CONSTANT,
                         value=0, mask_value=2,
                         mask_interpolation=cv2.INTER_NEAREST)
            ])
        elif aug_type == 'extreme':
            transform_list.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=90, p=0.7,
                         border_mode=cv2.BORDER_CONSTANT,
                         value=0, mask_value=2,
                         mask_interpolation=cv2.INTER_NEAREST),
                A.RandomBrightnessContrast(
                    brightness_limit=0.6, contrast_limit=0.6, p=0.8),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3)
            ])
        elif aug_type == 'balanced':
            transform_list.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=45, p=0.5,
                         border_mode=cv2.BORDER_CONSTANT,
                         value=0, mask_value=2,
                         mask_interpolation=cv2.INTER_NEAREST),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.2, p=0.5)
            ])

    transform_list.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0),
        ToTensorV2()
    ])

    return A.Compose(transform_list,
                     additional_targets={'mask': 'mask'})


# ==========================================
# DATASET
# ==========================================
class PotatoLeafDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.images    = sorted(os.listdir(img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path  = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(
            self.mask_dir.replace('masks', 'masks_index'),
            self.images[idx].replace('.jpg', '.png')
        )
        image = np.array(Image.open(img_path).convert("RGB"))
        mask  = np.array(Image.open(mask_path))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask  = augmented['mask']

        mask = mask.long()
        return image, mask


# ==========================================
# CLASS-AWARE WEIGHTED SAMPLER
#
# Cara kerja:
# - Tentukan kelas dominan tiap gambar
#   (kelas non-background terbanyak di mask)
# - Gambar kelas minoritas diberi bobot sampling tinggi
# - Setiap batch lebih sering mengambil kelas minoritas
# - Dataset asli tidak berubah
# ==========================================
def get_class_aware_sampler(dataset, mask_dir):
    print("   🔍 Menghitung distribusi kelas untuk class-aware sampler...")

    nama_kelas = {
        0: 'Bacteria', 1: 'Fungi',   2: 'Background',
        3: 'Nematode', 4: 'Pest',    5: 'Phytophthora',
        6: 'Virus'
    }

    label_per_gambar = []
    for img_name in dataset.images:
        mask_path = os.path.join(
            mask_dir.replace('masks', 'masks_index'),
            img_name.replace('.jpg', '.png')
        )
        mask = np.array(Image.open(mask_path))

        # Kelas dominan = kelas non-background terbanyak
        kelas_unik, jumlah = np.unique(mask, return_counts=True)
        kelas_jumlah = {
            k: n for k, n in zip(kelas_unik, jumlah) if k != 2
        }

        if kelas_jumlah:
            kelas_dominan = max(kelas_jumlah, key=kelas_jumlah.get)
        else:
            kelas_dominan = 2  # fallback ke background

        label_per_gambar.append(kelas_dominan)

    label_per_gambar = np.array(label_per_gambar)

    # Hitung frekuensi per kelas
    kelas_unik, jumlah_per_kelas = np.unique(
        label_per_gambar, return_counts=True)
    frekuensi = dict(zip(kelas_unik, jumlah_per_kelas))

    print(f"   📊 Distribusi gambar per kelas di data train:")
    for k, n in sorted(frekuensi.items()):
        print(f"      {nama_kelas.get(k, str(k)):<15}: {n} gambar")

    # Bobot per kelas = 1 / jumlah_gambar_kelas
    bobot_kelas = {k: 1.0 / n for k, n in frekuensi.items()}

    # Bobot per sampel
    bobot_per_sampel = np.array([
        bobot_kelas[label] for label in label_per_gambar
    ])

    # Normalisasi
    bobot_per_sampel = (bobot_per_sampel / bobot_per_sampel.sum()
                        * len(label_per_gambar))

    sampler = WeightedRandomSampler(
        weights     = torch.DoubleTensor(bobot_per_sampel),
        num_samples = len(label_per_gambar),
        replacement = True
    )

    print(f"   ✅ Class-aware sampler siap "
          f"({len(label_per_gambar)} sampel per epoch)")
    return sampler


# ==========================================
# DATALOADER
# ==========================================
def get_loaders(base_path, batch_size=8, img_size=256,
                use_augment=False, aug_type='balanced',
                use_class_aware=False):
    train_transform = get_transform(
        img_size, use_augment=use_augment, aug_type=aug_type)
    eval_transform  = get_transform(img_size, use_augment=False)

    train_mask_dir = os.path.join(base_path, 'train', 'masks')

    train_dataset = PotatoLeafDataset(
        os.path.join(base_path, 'train', 'images'),
        train_mask_dir,
        transform=train_transform
    )
    valid_dataset = PotatoLeafDataset(
        os.path.join(base_path, 'valid', 'images'),
        os.path.join(base_path, 'valid', 'masks'),
        transform=eval_transform
    )
    test_dataset = PotatoLeafDataset(
        os.path.join(base_path, 'test', 'images'),
        os.path.join(base_path, 'test', 'masks'),
        transform=eval_transform
    )

    # Train loader — class-aware atau shuffle biasa
    if use_class_aware:
        sampler = get_class_aware_sampler(train_dataset, train_mask_dir)
        train_loader = DataLoader(
            train_dataset,
            batch_size  = batch_size,
            sampler     = sampler,  # shuffle harus False kalau pakai sampler
            num_workers = 4,
            pin_memory  = True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size  = batch_size,
            shuffle     = True,
            num_workers = 4,
            pin_memory  = True
        )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 4,
        pin_memory  = True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = 4,
        shuffle     = False,
        num_workers = 4,
        pin_memory  = True
    )

    return train_loader, valid_loader, test_loader