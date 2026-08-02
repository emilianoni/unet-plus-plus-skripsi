import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_channels, out_channels, drop_rate=0.0):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if drop_rate > 0:
        layers.append(nn.Dropout2d(p=drop_rate))
    return nn.Sequential(*layers)

class UNetPlusPlus(nn.Module):
    def __init__(self, num_classes=7, input_channels=3, deep_supervision=True, drop_rate=0.0):
        super().__init__()
        nb = [32, 64, 128, 256, 512]
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block(input_channels, nb[0], drop_rate)
        self.conv0_1 = conv_block(nb[0] + nb[1], nb[0], drop_rate)
        self.conv0_2 = conv_block(nb[0]*2 + nb[1], nb[0], drop_rate)
        self.conv0_3 = conv_block(nb[0]*3 + nb[1], nb[0], drop_rate)
        self.conv0_4 = conv_block(nb[0]*4 + nb[1], nb[0], drop_rate)

        self.conv1_0 = conv_block(nb[0], nb[1], drop_rate)
        self.conv1_1 = conv_block(nb[1] + nb[2], nb[1], drop_rate)
        self.conv1_2 = conv_block(nb[1]*2 + nb[2], nb[1], drop_rate)
        self.conv1_3 = conv_block(nb[1]*3 + nb[2], nb[1], drop_rate)

        self.conv2_0 = conv_block(nb[1], nb[2], drop_rate)
        self.conv2_1 = conv_block(nb[2] + nb[3], nb[2], drop_rate)
        self.conv2_2 = conv_block(nb[2]*2 + nb[3], nb[2], drop_rate)

        self.conv3_0 = conv_block(nb[2], nb[3], drop_rate)
        self.conv3_1 = conv_block(nb[3] + nb[4], nb[3], drop_rate)

        self.conv4_0 = conv_block(nb[3], nb[4], drop_rate)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            # Mengembalikan list dari keempat cabang output (X^0,1 s/d X^0,4)
            # sesuai paper Zhou et al. Section II-B
            return [self.final1(x0_1), self.final2(x0_2), 
                    self.final3(x0_3), self.final4(x0_4)]
        else:
            return self.final(x0_4)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()


class ModifiedHybridLoss(nn.Module):
    def __init__(self, device, ce_weight=0.5, dice_weight=0.5, 
                 focal_weight=0.0, focal_gamma=2):  # ✅ tambahkan focal_gamma
        super(ModifiedHybridLoss, self).__init__()
        self.ce_weight    = ce_weight
        self.dice_weight  = dice_weight
        self.focal_weight = focal_weight

        # ✅ GANTI INI: weights baru hasil perhitungan dari dataset bersih
        self.class_weights = torch.tensor(
            [0.9663, 0.6596, 0.2595, 2.3612, 0.7483, 1.1964, 0.8088]
        ).to(device)

        self.ce    = nn.CrossEntropyLoss(weight=self.class_weights)
        self.focal = FocalLoss(weight=self.class_weights, gamma=focal_gamma)  # ✅ pakai focal_gamma

    def dice_loss(self, pred, target, num_classes=7):
        """
        PERBAIKAN: Dice loss dengan class weighting agar konsisten
        dengan class_weights yang dipakai di CCE.
        """
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        pred_softmax = F.softmax(pred, dim=1)

        # Hitung dice per kelas
        intersection = torch.sum(pred_softmax * target_one_hot, dim=(2, 3))
        union = torch.sum(pred_softmax + target_one_hot, dim=(2, 3))
        dice_per_class = (2. * intersection + 1e-5) / (union + 1e-5)

        # ✅ PERBAIKAN: Weighted average menggunakan class_weights
        # agar kelas langka (Virus, Phytophthora) lebih diperhatikan
        weights = self.class_weights / self.class_weights.sum()
        weighted_dice = (dice_per_class * weights.unsqueeze(0)).sum(dim=1)
        return 1 - weighted_dice.mean()

    def forward(self, pred, target):
        loss = 0
        if self.ce_weight > 0:
            loss += self.ce_weight * self.ce(pred, target)
        if self.dice_weight > 0:
            loss += self.dice_weight * self.dice_loss(pred, target)
        if self.focal_weight > 0:
            loss += self.focal_weight * self.focal(pred, target)
        return loss