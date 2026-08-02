import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from sklearn.metrics import jaccard_score
import os
import argparse
from matplotlib.colors import ListedColormap

from dataset_loader import get_loaders
from unet_model import UNetPlusPlus, ModifiedHybridLoss

os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ==========================================
# 1. ARGPARSE
# ==========================================
parser = argparse.ArgumentParser(description="Mesin U-Net Skripsi Emil")
parser.add_argument('--tag',        type=str,   required=True)
parser.add_argument('--img_size',   type=int,   required=True)
parser.add_argument('--batch_size', type=int,   required=True)
parser.add_argument('--lr',         type=float, required=True)
parser.add_argument('--epochs',     type=int,   default=100)
parser.add_argument('--optimizer',  type=str,   default='adam',
                    choices=['adam', 'adamw', 'sgd'])

# Loss
parser.add_argument('--ce_w',        type=float, default=0.5)
parser.add_argument('--dice_w',      type=float, default=0.5)
parser.add_argument('--focal_w',     type=float, default=0.0)
parser.add_argument('--focal_gamma', type=float, default=2)

# Augmentasi
parser.add_argument('--use_augment',     action='store_true')
parser.add_argument('--aug_type',        type=str, default='balanced',
                    choices=['balanced', 'spatial', 'extreme'])

# Class-aware sampling
parser.add_argument('--use_class_aware', action='store_true',
                    help='Aktifkan WeightedRandomSampler untuk '
                         'class-aware sampling')

args = parser.parse_args()

# ==========================================
# 2. SETUP GLOBAL
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ACTIVE_CLASSES = [0, 1, 3, 4, 5, 6]
CLASS_NAMES    = ["Bacteria", "Fungi", "Nematode",
                  "Pest", "Phytophthora", "Virus"]

warna_penyakit = ['yellow', 'magenta', 'black', 'red', 'lime', 'blue', 'orange']
custom_cmap    = ListedColormap(warna_penyakit)

# ==========================================
# 3. LOSS FUNCTION
# ==========================================
criterion = ModifiedHybridLoss(
    device,
    ce_weight    = args.ce_w,
    dice_weight  = args.dice_w,
    focal_weight = args.focal_w,
    focal_gamma  = args.focal_gamma
)

# ==========================================
# 4. FUNGSI EVALUASI
# ==========================================
def evaluasi_model(loader, model_obj,
                   simpan_preview=False, nama_fase=""):
    model_obj.eval()
    all_preds, all_targets = [], []
    val_loss = 0.0

    with torch.no_grad():
        for batch_idx, (imgs, msks) in enumerate(loader):
            imgs = imgs.to(device)
            msks = msks.to(device)
            out  = model_obj(imgs)

            # DS: rata-rata murni sesuai paper Zhou et al.
            if isinstance(out, list):
                loss = sum([criterion(out[k], msks)
                            for k in range(len(out))]) / len(out)
                final_out = torch.stack(out, dim=0).mean(dim=0)
            else:
                loss      = criterion(out, msks)
                final_out = out

            val_loss += loss.item()

            preds   = torch.argmax(final_out, dim=1).cpu().numpy()
            targets = msks.cpu().numpy()
            all_preds.extend(preds.flatten())
            all_targets.extend(targets.flatten())

            if simpan_preview and batch_idx == 0:
                img_vis = imgs[0].cpu().permute(1, 2, 0).numpy()
                img_vis = (img_vis * np.array([0.229, 0.224, 0.225])
                           + np.array([0.485, 0.456, 0.406]))
                img_vis = np.clip(img_vis, 0, 1)

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(img_vis)
                axes[0].set_title(f"Citra Asli ({nama_fase})")
                axes[1].imshow(targets[0], cmap=custom_cmap,
                               vmin=0, vmax=6)
                axes[1].set_title("Ground Truth")
                axes[2].imshow(preds[0], cmap=custom_cmap,
                               vmin=0, vmax=6)
                axes[2].set_title(f"Prediksi — {args.tag}")
                plt.tight_layout()
                plt.savefig(
                    f'outputs/preview_{args.tag}_{nama_fase}.png',
                    dpi=150)
                plt.close(fig)

    avg_val_loss  = val_loss / len(loader)
    iou_per_class = jaccard_score(
        all_targets, all_preds,
        average=None, labels=ACTIVE_CLASSES, zero_division=0
    )
    macro_iou = float(np.mean(iou_per_class))
    return avg_val_loss, macro_iou, iou_per_class


# ==========================================
# 5. MAIN TRAINING LOOP
# ==========================================
if __name__ == '__main__':
    print("=" * 60)
    print(f"⚙️  MESIN MENYALA: {args.tag}")
    print(f"📐 Res:{args.img_size} | BS:{args.batch_size} | "
          f"LR:{args.lr} | Opt:{args.optimizer}")
    print(f"⚖️  CE:{args.ce_w} | Dice:{args.dice_w} | "
          f"Focal:{args.focal_w}(γ={args.focal_gamma})")
    print(f"🌿 Aug:{args.use_augment}({args.aug_type}) | "
          f"ClassAware:{args.use_class_aware}")
    print(f"📊 DS: rata-rata murni (1/4 per cabang, sesuai Zhou et al.)")
    print("=" * 60)

    # --- DataLoader ---
    train_loader, valid_loader, test_loader = get_loaders(
        'dataset',
        batch_size      = args.batch_size,
        img_size        = args.img_size,
        use_augment     = args.use_augment,
        aug_type        = args.aug_type,
        use_class_aware = args.use_class_aware
    )

    # --- Model ---
    model = UNetPlusPlus(input_channels=3, num_classes=7).to(device)

    # --- Optimizer ---
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr,
            momentum=0.9, weight_decay=1e-4)

    # --- Variabel tracking ---
    best_val_loss = float('inf')
    best_val_miou = 0.0
    history_data  = []
    global_start  = time.time()

    # ==========================================
    # EPOCH LOOP
    # ==========================================
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # ── TRAINING ──────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader,
                    desc=f"Ep {epoch:03d}/{args.epochs}")

        for imgs, msks in pbar:
            imgs, msks = imgs.to(device), msks.to(device)
            optimizer.zero_grad()

            outputs = model(imgs)

            # DS: rata-rata murni
            if isinstance(outputs, list):
                loss = sum([criterion(outputs[k], msks)
                            for k in range(len(outputs))]) / len(outputs)
            else:
                loss = criterion(outputs, msks)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # ── VALIDASI PER EPOCH ────────────────────────────────────
        avg_val_loss, current_val_miou, _ = evaluasi_model(
            valid_loader, model,
            simpan_preview=False, nama_fase=""
        )

        current_lr     = optimizer.param_groups[0]['lr']
        epoch_duration = time.time() - epoch_start

        # ── CATAT HISTORY ─────────────────────────────────────────
        history_data.append({
            'epoch':        epoch,
            'train_loss':   avg_train_loss,
            'val_loss':     avg_val_loss,
            'val_miou':     current_val_miou,
            'current_lr':   current_lr,
            'duration_sec': epoch_duration
        })
        pd.DataFrame(history_data).to_csv(
            f'outputs/hasil_{args.tag}.csv', index=False)

        print(f"   📉 Train:{avg_train_loss:.4f} | "
              f"Val:{avg_val_loss:.4f} | "
              f"mIoU:{current_val_miou:.4f} | "
              f"LR:{current_lr:.2e}")

        # ── SIMPAN MODEL TERBAIK ───────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(),
                       f'models/model_{args.tag}_BEST_LOSS.pth')
            print(f"      🟢 REKOR LOSS! ({best_val_loss:.4f})")

        if current_val_miou > best_val_miou:
            best_val_miou = current_val_miou
            torch.save(model.state_dict(),
                       f'models/model_{args.tag}_BEST_MIOU.pth')
            print(f"      👑 REKOR mIoU! ({best_val_miou:.4f})")

    # ==========================================
    # 6. EVALUASI FINAL
    # ==========================================
    total_jam = (time.time() - global_start) / 3600
    print(f"\n🎉 TRAINING SELESAI! Total: {total_jam:.2f} jam")
    print(f"   Best Val Loss : {best_val_loss:.4f}")
    print(f"   Best Val mIoU : {best_val_miou:.4f}")

    model.load_state_dict(torch.load(
        f'models/model_{args.tag}_BEST_MIOU.pth',
        map_location=device, weights_only=True
    ))

    _, final_miou, final_iou_per_class = evaluasi_model(
        valid_loader, model,
        simpan_preview=True, nama_fase="Valid_Akhir"
    )

    print(f"\n📊 IoU Per Kelas (BEST_MIOU di validasi):")
    for cls_name, iou_val in zip(CLASS_NAMES, final_iou_per_class):
        print(f"   {cls_name:<15}: {iou_val:.4f}")
    print(f"   {'mIoU (macro)':<15}: {final_miou:.4f}")

    # ==========================================
    # 7. GRAFIK OTOMATIS
    # ==========================================
    df = pd.DataFrame(history_data)

    # Kurva Loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['epoch'], df['train_loss'],
            label='Train Loss', color='blue', lw=2)
    ax.plot(df['epoch'], df['val_loss'],
            label='Val Loss', color='orange', lw=2)
    best_ep = int(df.loc[df['val_loss'].idxmin(), 'epoch'])
    ax.scatter(best_ep, df['val_loss'].min(),
               color='red', s=120, zorder=5,
               label=f'Best Val Loss (Ep {best_ep})')
    ax.set_title(f'Kurva Loss — {args.tag}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'outputs/loss_curve_{args.tag}.png', dpi=300)
    plt.close(fig)

    # Kurva mIoU
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['epoch'], df['val_miou'],
            label='Val mIoU', color='green', lw=2)
    best_ep_miou = int(df.loc[df['val_miou'].idxmax(), 'epoch'])
    ax.scatter(best_ep_miou, df['val_miou'].max(),
               color='purple', s=120, zorder=5,
               label=f'Best mIoU (Ep {best_ep_miou}) '
                     f'= {df["val_miou"].max():.4f}')
    ax.set_title(f'Kurva mIoU — {args.tag}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mIoU')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'outputs/miou_curve_{args.tag}.png', dpi=300)
    plt.close(fig)

    print(f"\n✅ Grafik tersimpan di folder outputs/")
    print(f"✅ Model tersimpan di folder models/")