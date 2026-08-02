import torch
import numpy as np
import argparse
import os
from sklearn.metrics import (jaccard_score, f1_score, 
                              accuracy_score)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from dataset_loader import get_loaders
from unet_model import UNetPlusPlus, ModifiedHybridLoss

parser = argparse.ArgumentParser()
parser.add_argument('--tag',          type=str, required=True)
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--img_size',     type=int, default=256)
parser.add_argument('--batch_size',   type=int, default=4)
parser.add_argument('--model_type',   type=str, default='best_miou',
                    choices=['best_miou', 'best_loss'])
args = parser.parse_args()

if __name__ == '__main__':

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    ACTIVE_CLASSES = [0, 1, 3, 4, 5, 6]
    CLASS_NAMES    = ["Bacteria", "Fungi", "Nematode",
                      "Pest", "Phytophthora", "Virus"]

    warna_penyakit = ['yellow', 'magenta', 'black',
                      'red', 'lime', 'blue', 'orange']
    custom_cmap    = ListedColormap(warna_penyakit)

    # ==========================================
    # LOAD MODEL
    # ==========================================
    model_path = f'models/model_{args.tag}_BEST_MIOU.pth' \
                 if args.model_type == 'best_miou' \
                 else f'models/model_{args.tag}_BEST_LOSS.pth'

    if not os.path.exists(model_path):
        print(f"ERROR: Model tidak ditemukan di {model_path}")
        exit(1)

    model = UNetPlusPlus(
        input_channels=3,
        num_classes=7,
        drop_rate=0.1
    ).to(device)

    model.load_state_dict(torch.load(
        model_path, map_location=device, weights_only=True))
    print(f"Model loaded: {model_path}")

    # ==========================================
    # DATA LOADER
    # ==========================================
    _, valid_loader, test_loader = get_loaders(
        args.dataset_path,
        batch_size  = args.batch_size,
        img_size    = args.img_size,
        use_augment = False
    )

    # ==========================================
    # LOSS
    # ==========================================
    criterion = ModifiedHybridLoss(
        device,
        ce_weight       = 0.5,
        dice_weight     = 0.5,
        use_zhou_dice   = True,
        use_iou_weights = True
    )

    # ==========================================
    # FUNGSI EVALUASI
    # ==========================================
    def evaluasi(loader, nama):
        model.eval()
        all_preds, all_targets = [], []
        total_loss = 0.0

        with torch.no_grad():
            for batch_idx, (imgs, msks) in enumerate(loader):
                imgs = imgs.to(device)
                msks = msks.to(device)
                out  = model(imgs)

                if isinstance(out, list):
                    loss      = sum([criterion(out[k], msks)
                                     for k in range(len(out))
                                     ]) / len(out)
                    final_out = torch.stack(
                        out, dim=0).mean(dim=0)
                else:
                    loss      = criterion(out, msks)
                    final_out = out

                total_loss += loss.item()
                preds   = torch.argmax(
                    final_out, dim=1).cpu().numpy()
                targets = msks.cpu().numpy()
                all_preds.extend(preds.flatten())
                all_targets.extend(targets.flatten())

                if batch_idx == 0 and nama == 'TEST':
                    img_vis = imgs[0].cpu().permute(
                        1, 2, 0).numpy()
                    img_vis = np.clip(img_vis, 0, 1)
                    fig, axes = plt.subplots(
                        1, 3, figsize=(15, 5))
                    axes[0].imshow(img_vis)
                    axes[0].set_title('Citra Asli')
                    axes[0].axis('off')
                    axes[1].imshow(targets[0],
                                   cmap=custom_cmap,
                                   vmin=0, vmax=6)
                    axes[1].set_title('Ground Truth')
                    axes[1].axis('off')
                    axes[2].imshow(preds[0],
                                   cmap=custom_cmap,
                                   vmin=0, vmax=6)
                    axes[2].set_title(
                        f'Prediksi — {args.tag}')
                    axes[2].axis('off')
                    plt.tight_layout()
                    plt.savefig(
                        f'outputs/preview_'
                        f'{args.tag}_TEST.png',
                        dpi=150)
                    plt.close(fig)

        avg_loss = total_loss / len(loader)

        iou_per_class = jaccard_score(
            all_targets, all_preds,
            average=None, labels=ACTIVE_CLASSES,
            zero_division=0
        )
        dice_per_class = f1_score(
            all_targets, all_preds,
            average=None, labels=ACTIVE_CLASSES,
            zero_division=0
        )
        pixel_acc  = accuracy_score(all_targets, all_preds)
        macro_iou  = float(np.mean(iou_per_class))
        macro_dice = float(np.mean(dice_per_class))

        print(f"\n{'='*60}")
        print(f"HASIL {nama} — {args.tag} "
              f"({args.model_type.upper()})")
        print(f"{'='*60}")
        print(f"Loss          : {avg_loss:.4f}")
        print(f"mIoU          : {macro_iou:.4f}")
        print(f"Dice (F1)     : {macro_dice:.4f}")
        print(f"Pixel Accuracy: {pixel_acc:.4f}")
        print(f"\nIoU per Kelas:")
        for cls_name, iou_val in zip(CLASS_NAMES,
                                      iou_per_class):
            print(f"   {cls_name:<15}: {iou_val:.4f}")
        print(f"\nDice per Kelas:")
        for cls_name, dice_val in zip(CLASS_NAMES,
                                       dice_per_class):
            print(f"   {cls_name:<15}: {dice_val:.4f}")

    # ==========================================
    # JALANKAN
    # ==========================================
    evaluasi(valid_loader, 'VALIDASI')
    evaluasi(test_loader,  'TEST')
    print(f"\nPreview tersimpan: "
          f"outputs/preview_{args.tag}_TEST.png")