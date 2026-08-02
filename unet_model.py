import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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
    def __init__(self, num_classes=7, input_channels=3,
                 deep_supervision=True, drop_rate=0.0,
                 model_size='normal'):
        super().__init__()

        if model_size == 'small':
            nb = [16, 32, 64, 128, 256]
        else:
            nb = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up   = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

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
        x0_3 = self.conv0_3(
            torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(
            torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(
            torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            return [self.final1(x0_1), self.final2(x0_2),
                    self.final3(x0_3), self.final4(x0_4)]
        else:
            return self.final(x0_4)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss    = F.cross_entropy(
            inputs, targets, weight=self.weight, reduction='none')
        pt         = torch.exp(-ce_loss)
        focal_loss = (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()


class ZhouDiceLoss(nn.Module):
    """
    Dice Loss persis seperti Zhou et al. (2020) UNet++.
    Penyebut: y^2 + p^2 (bukan y + p).
    Tanpa class weighting.
    """
    def __init__(self, num_classes=7, epsilon=1e-5):
        super(ZhouDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon     = epsilon

    def forward(self, pred, target):
        target_one_hot = F.one_hot(
            target, self.num_classes
        ).permute(0, 3, 1, 2).float()
        pred_softmax = F.softmax(pred, dim=1)

        intersection = torch.sum(
            pred_softmax * target_one_hot, dim=(2, 3))
        # Penyebut persis Zhou: y^2 + p^2
        denominator  = torch.sum(
            pred_softmax**2 + target_one_hot**2, dim=(2, 3))

        dice_per_class = (2. * intersection + self.epsilon) / \
                         (denominator + self.epsilon)
        return 1 - dice_per_class.mean()


class ModifiedHybridLoss(nn.Module):
    """
    Hybrid Loss yang dapat dikonfigurasi.

    Parameter:
    - use_class_weights : True  → pakai class weights (default, S2B.3)
                          False → tanpa class weights (Eksperimen A & C)
    - use_zhou_dice     : False → Dice penyebut y+p (default, S2B.3)
                          True  → Dice penyebut y^2+p^2 (Eksperimen B & C)
    """
    def __init__(self, device, ce_weight=0.5, dice_weight=0.5,
                focal_weight=0.0, focal_gamma=2,
                use_class_weights=True,
                use_zhou_dice=False,
                use_iou_weights=False):  # ← tambah ini
        super(ModifiedHybridLoss, self).__init__()
        self.ce_weight    = ce_weight
        self.dice_weight  = dice_weight
        self.focal_weight = focal_weight
        self.use_zhou_dice = use_zhou_dice

        if not use_class_weights:
            self.class_weights = torch.ones(7).to(device)
            self.ce    = nn.CrossEntropyLoss()
            self.focal = FocalLoss(gamma=focal_gamma)
        elif use_iou_weights:
            # Berbasis 1/IoU dari hasil S10
            # Bacteria=0.584, Fungi=0.627, BG=rendah,
            # Nematode=0.666, Pest=0.540, Phyto=0.559, Virus=0.561
            self.class_weights = torch.tensor(
                [1.712, 1.594, 0.260, 1.501, 1.850, 1.790, 1.784]
            ).to(device)
            self.ce    = nn.CrossEntropyLoss(weight=self.class_weights)
            self.focal = FocalLoss(
                weight=self.class_weights, gamma=focal_gamma)
        else:
            # Default — berbasis frekuensi piksel
            self.class_weights = torch.tensor(
                [0.9663, 0.6596, 0.2595, 2.3612, 0.7483, 1.1964, 0.8088]
            ).to(device)
            self.ce    = nn.CrossEntropyLoss(weight=self.class_weights)
            self.focal = FocalLoss(
                weight=self.class_weights, gamma=focal_gamma)

        # Komponen Dice
        if use_zhou_dice:
            # Penyebut y^2 + p^2, tanpa class weighting
            self.zhou_dice = ZhouDiceLoss(num_classes=7)
        else:
            self.zhou_dice = None

    def dice_loss(self, pred, target, num_classes=7):
        """Dice Loss internal: penyebut y+p, dengan class weighting."""
        target_one_hot = F.one_hot(
            target, num_classes).permute(0, 3, 1, 2).float()
        pred_softmax   = F.softmax(pred, dim=1)

        intersection    = torch.sum(
            pred_softmax * target_one_hot, dim=(2, 3))
        union           = torch.sum(
            pred_softmax + target_one_hot, dim=(2, 3))
        dice_per_class  = (2. * intersection + 1e-5) / (union + 1e-5)

        weights       = self.class_weights / self.class_weights.sum()
        weighted_dice = (
            dice_per_class * weights.unsqueeze(0)).sum(dim=1)
        return 1 - weighted_dice.mean()

    def forward(self, pred, target):
        loss = 0
        if self.ce_weight > 0:
            loss += self.ce_weight * self.ce(pred, target)
        if self.dice_weight > 0:
            if self.zhou_dice is not None:
                # Eksperimen B & C: penyebut y^2+p^2
                loss += self.dice_weight * self.zhou_dice(pred, target)
            else:
                # Default (S2B.3): penyebut y+p dengan class weighting
                loss += self.dice_weight * self.dice_loss(pred, target)
        if self.focal_weight > 0:
            loss += self.focal_weight * self.focal(pred, target)
        return loss
    
class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(
            weights='IMAGENET1K_V1' if pretrained else None
        )
        self.stem   = nn.Sequential(
            resnet.conv1, resnet.bn1,
            resnet.relu, resnet.maxpool
        )  # 64ch, 64x64
        self.layer1 = resnet.layer1  # 256ch, 64x64
        self.layer2 = resnet.layer2  # 512ch, 32x32
        self.layer3 = resnet.layer3  # 1024ch, 16x16
        self.layer4 = resnet.layer4  # 2048ch, 8x8

    def forward(self, x):
        e0 = self.stem(x)     # 64ch,   64x64
        e1 = self.layer1(e0)  # 256ch,  64x64
        e2 = self.layer2(e1)  # 512ch,  32x32
        e3 = self.layer3(e2)  # 1024ch, 16x16
        e4 = self.layer4(e3)  # 2048ch, 8x8
        return e0, e1, e2, e3, e4


class UNetPlusPlusResNet50(nn.Module):
    """
    U-Net++ dengan encoder ResNet50 pretrained.
    Karena ResNet50 stem melakukan downsampling 4x,
    decoder hanya memiliki 3 level (baris 2, 1, 0)
    dengan resolusi maksimal 64x64, kemudian
    di-upsample ke 256x256 di output.
    Deep supervision tetap dipertahankan (4 cabang).
    """
    def __init__(self, num_classes=7, pretrained=True,
                 drop_rate=0.1, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = ResNet50Encoder(pretrained=pretrained)

        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        # Projection: encoder ch → decoder ch
        # d = [256, 128, 64, 32] sesuai level
        self.proj0 = nn.Conv2d(64,   256, kernel_size=1)  # stem
        self.proj1 = nn.Conv2d(256,  256, kernel_size=1)  # layer1, 64x64
        self.proj2 = nn.Conv2d(512,  128, kernel_size=1)  # layer2, 32x32
        self.proj3 = nn.Conv2d(1024, 64,  kernel_size=1)  # layer3, 16x16
        self.proj4 = nn.Conv2d(2048, 32,  kernel_size=1)  # layer4, 8x8

        # ── Baris 3 — resolusi 16x16 ──────────────────────────
        # x3_1: x3_0[64] + up(x4_0)[32] = 96 → 64
        self.conv3_1 = conv_block(64 + 32, 64, drop_rate)

        # ── Baris 2 — resolusi 32x32 ──────────────────────────
        # x2_1: x2_0[128] + up(x3_0)[64] = 192 → 128
        self.conv2_1 = conv_block(128 + 64, 128, drop_rate)
        # x2_2: x2_0[128] + x2_1[128] + up(x3_1)[64] = 320 → 128
        self.conv2_2 = conv_block(128*2 + 64, 128, drop_rate)

        # ── Baris 1 — resolusi 64x64 ──────────────────────────
        # x1_0 = proj1(e1), 256ch, 64x64
        # x1_1: x1_0[256] + up(x2_0)[128] = 384 → 256
        self.conv1_1 = conv_block(256 + 128, 256, drop_rate)
        # x1_2: x1_0[256] + x1_1[256] + up(x2_1)[128] = 640 → 256
        self.conv1_2 = conv_block(256*2 + 128, 256, drop_rate)
        # x1_3: x1_0[256] + x1_1[256] + x1_2[256] + up(x2_2)[128] = 896 → 256
        self.conv1_3 = conv_block(256*3 + 128, 256, drop_rate)

        # ── Baris 0 — resolusi 64x64 (pakai stem sebagai x0_0) ─
        # x0_0 = proj0(e0), 256ch, 64x64
        # x0_1: x0_0[256] + x1_1[256] = 512 → 256
        self.conv0_1 = conv_block(256 + 256, 256, drop_rate)
        # x0_2: x0_0[256] + x0_1[256] + x1_2[256] = 768 → 256
        self.conv0_2 = conv_block(256*3, 256, drop_rate)
        # x0_3: x0_0[256] + x0_1[256] + x0_2[256] + x1_3[256] = 1024 → 256
        self.conv0_3 = conv_block(256*4, 256, drop_rate)
        # x0_4: x0_0[256] + x0_1[256] + x0_2[256] + x0_3[256] + x1_3[256] = 1280 → 256
        self.conv0_4 = conv_block(256*5, 256, drop_rate)

        # ── Output heads ──────────────────────────────────────
        if self.deep_supervision:
            self.final1 = nn.Conv2d(256, num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(256, num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(256, num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(256, num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(256, num_classes, kernel_size=1)

        # Upsample output dari 64x64 ke 256x256
        self.upsample_final = nn.Upsample(
            size=(256, 256), mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder
        e0, e1, e2, e3, e4 = self.encoder(x)

        # Projection
        x0_0 = self.proj0(e0)  # 256ch, 64x64
        x1_0 = self.proj1(e1)  # 256ch, 64x64
        x2_0 = self.proj2(e2)  # 128ch, 32x32
        x3_0 = self.proj3(e3)  # 64ch,  16x16
        x4_0 = self.proj4(e4)  # 32ch,  8x8

        # Baris 3 — 16x16
        x3_1 = self.conv3_1(
            torch.cat([x3_0, self.up(x4_0)], dim=1))
        # 64+32=96

        # Baris 2 — 32x32
        x2_1 = self.conv2_1(
            torch.cat([x2_0, self.up(x3_0)], dim=1))
        # 128+64=192
        x2_2 = self.conv2_2(
            torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))
        # 128+128+64=320

        # Baris 1 — 64x64
        x1_1 = self.conv1_1(
            torch.cat([x1_0, self.up(x2_0)], dim=1))
        # 256+128=384
        x1_2 = self.conv1_2(
            torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        # 256+256+128=640
        x1_3 = self.conv1_3(
            torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))
        # 256+256+256+128=896

        # Baris 0 — 64x64
        x0_1 = self.conv0_1(
            torch.cat([x0_0, x1_1], dim=1))
        # 256+256=512
        x0_2 = self.conv0_2(
            torch.cat([x0_0, x0_1, x1_2], dim=1))
        # 256+256+256=768
        x0_3 = self.conv0_3(
            torch.cat([x0_0, x0_1, x0_2, x1_3], dim=1))
        # 256+256+256+256=1024
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, x1_3], dim=1))
        # 256+256+256+256+256=1280

        # Output — upsample dari 64x64 ke 256x256
        if self.deep_supervision:
            out1 = self.upsample_final(self.final1(x0_1))
            out2 = self.upsample_final(self.final2(x0_2))
            out3 = self.upsample_final(self.final3(x0_3))
            out4 = self.upsample_final(self.final4(x0_4))
            return [out1, out2, out3, out4]
        else:
            return self.upsample_final(self.final(x0_4))
        
class VGG16Encoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        vgg = models.vgg16(
            weights='IMAGENET1K_V1' if pretrained else None
        )
        features = vgg.features
        # Ambil feature map SEBELUM tiap maxpool (bukan sesudahnya)
        self.stage0 = features[0:4]    # 64ch,  stride1 (sebelum pool1)
        self.pool0  = features[4]
        self.stage1 = features[5:9]    # 128ch, stride2 (sebelum pool2)
        self.pool1  = features[9]
        self.stage2 = features[10:16]  # 256ch, stride4 (sebelum pool3)
        self.pool2  = features[16]
        self.stage3 = features[17:23]  # 512ch, stride8 (sebelum pool4)
        self.pool3  = features[23]
        self.stage4 = features[24:30]  # 512ch, stride16 (sebelum pool5)

    def forward(self, x):
        e0 = self.stage0(x)                    # 64ch,  256x256
        e1 = self.stage1(self.pool0(e0))        # 128ch, 128x128
        e2 = self.stage2(self.pool1(e1))        # 256ch, 64x64
        e3 = self.stage3(self.pool2(e2))        # 512ch, 32x32
        e4 = self.stage4(self.pool3(e3))        # 512ch, 16x16
        return e0, e1, e2, e3, e4


class UNetPlusPlusVGG16(nn.Module):
    """
    U-Net++ dengan encoder VGG16.
    Berbeda dari versi ResNet50: VGG16 punya 5 resolusi
    berbeda sebelum tiap maxpool, jadi decoder bisa
    memakai nested skip pathway LENGKAP 5 baris,
    sejajar dengan struktur UNetPlusPlus custom.
    """
    def __init__(self, num_classes=7, pretrained=False,
                 drop_rate=0.1, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = VGG16Encoder(pretrained=pretrained)
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        # Projection: encoder ch (VGG) → decoder ch (nb)
        nb = [32, 64, 128, 256, 512]
        self.proj0 = nn.Conv2d(64,  nb[0], kernel_size=1)
        self.proj1 = nn.Conv2d(128, nb[1], kernel_size=1)
        self.proj2 = nn.Conv2d(256, nb[2], kernel_size=1)
        self.proj3 = nn.Conv2d(512, nb[3], kernel_size=1)
        self.proj4 = nn.Conv2d(512, nb[4], kernel_size=1)

        # Nested skip pathway — identik strukturnya dengan
        # UNetPlusPlus custom kamu, cuma row0_0..row4_0
        # digantikan hasil proj0..proj4 dari encoder VGG16
        self.conv0_1 = conv_block(nb[0] + nb[1], nb[0], drop_rate)
        self.conv0_2 = conv_block(nb[0]*2 + nb[1], nb[0], drop_rate)
        self.conv0_3 = conv_block(nb[0]*3 + nb[1], nb[0], drop_rate)
        self.conv0_4 = conv_block(nb[0]*4 + nb[1], nb[0], drop_rate)

        self.conv1_1 = conv_block(nb[1] + nb[2], nb[1], drop_rate)
        self.conv1_2 = conv_block(nb[1]*2 + nb[2], nb[1], drop_rate)
        self.conv1_3 = conv_block(nb[1]*3 + nb[2], nb[1], drop_rate)

        self.conv2_1 = conv_block(nb[2] + nb[3], nb[2], drop_rate)
        self.conv2_2 = conv_block(nb[2]*2 + nb[3], nb[2], drop_rate)

        self.conv3_1 = conv_block(nb[3] + nb[4], nb[3], drop_rate)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb[0], num_classes, kernel_size=1)

    def forward(self, x):
        e0, e1, e2, e3, e4 = self.encoder(x)

        x0_0 = self.proj0(e0)  # 32ch,  256x256
        x1_0 = self.proj1(e1)  # 64ch,  128x128
        x2_0 = self.proj2(e2)  # 128ch, 64x64
        x3_0 = self.proj3(e3)  # 256ch, 32x32
        x4_0 = self.proj4(e4)  # 512ch, 16x16

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(
            torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(
            torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            return [self.final1(x0_1), self.final2(x0_2),
                    self.final3(x0_3), self.final4(x0_4)]
        else:
            return self.final(x0_4)
        

class MobileNetV2Encoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        mobilenet = models.mobilenet_v2(
            weights='IMAGENET1K_V1' if pretrained else None
        )
        features = mobilenet.features
        # Sesuai stride milestone bawaan MobileNetV2
        self.stage0 = features[0:2]    # 16ch,   stride2
        self.stage1 = features[2:4]    # 24ch,   stride4
        self.stage2 = features[4:7]    # 32ch,   stride8
        self.stage3 = features[7:14]   # 96ch,   stride16
        self.stage4 = features[14:19]  # 1280ch, stride32

    def forward(self, x):
        e0 = self.stage0(x)  # 16ch,   128x128 (input 256)
        e1 = self.stage1(e0) # 24ch,   64x64
        e2 = self.stage2(e1) # 32ch,   32x32
        e3 = self.stage3(e2) # 96ch,   16x16
        e4 = self.stage4(e3) # 1280ch, 8x8
        return e0, e1, e2, e3, e4


class UNetPlusPlusMobileNetV2(nn.Module):
    """
    U-Net++ dengan encoder MobileNetV2.
    Encoder ini downsample langsung sejak stage awal (stride2),
    tanpa stage stride1 seperti VGG16, sehingga resolusi
    tertinggi decoder adalah 128x128 (bukan 256x256) —
    perlu upsample_final ke 256x256 di output, mirip versi ResNet50.
    """
    def __init__(self, num_classes=7, pretrained=False,
                 drop_rate=0.1, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = MobileNetV2Encoder(pretrained=pretrained)
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        nb = [32, 64, 128, 256, 512]
        self.proj0 = nn.Conv2d(16,   nb[0], kernel_size=1)
        self.proj1 = nn.Conv2d(24,   nb[1], kernel_size=1)
        self.proj2 = nn.Conv2d(32,   nb[2], kernel_size=1)
        self.proj3 = nn.Conv2d(96,   nb[3], kernel_size=1)
        self.proj4 = nn.Conv2d(1280, nb[4], kernel_size=1)

        self.conv0_1 = conv_block(nb[0] + nb[1], nb[0], drop_rate)
        self.conv0_2 = conv_block(nb[0]*2 + nb[1], nb[0], drop_rate)
        self.conv0_3 = conv_block(nb[0]*3 + nb[1], nb[0], drop_rate)
        self.conv0_4 = conv_block(nb[0]*4 + nb[1], nb[0], drop_rate)

        self.conv1_1 = conv_block(nb[1] + nb[2], nb[1], drop_rate)
        self.conv1_2 = conv_block(nb[1]*2 + nb[2], nb[1], drop_rate)
        self.conv1_3 = conv_block(nb[1]*3 + nb[2], nb[1], drop_rate)

        self.conv2_1 = conv_block(nb[2] + nb[3], nb[2], drop_rate)
        self.conv2_2 = conv_block(nb[2]*2 + nb[3], nb[2], drop_rate)

        self.conv3_1 = conv_block(nb[3] + nb[4], nb[3], drop_rate)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb[0], num_classes, kernel_size=1)

        # e0 cuma 128x128, perlu upsample ke 256x256 di output
        self.upsample_final = nn.Upsample(
            size=(256, 256), mode='bilinear', align_corners=True)

    def forward(self, x):
        e0, e1, e2, e3, e4 = self.encoder(x)

        x0_0 = self.proj0(e0)  # 32ch,  128x128
        x1_0 = self.proj1(e1)  # 64ch,  64x64
        x2_0 = self.proj2(e2)  # 128ch, 32x32
        x3_0 = self.proj3(e3)  # 256ch, 16x16
        x4_0 = self.proj4(e4)  # 512ch, 8x8

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(
            torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(
            torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            return [self.upsample_final(self.final1(x0_1)),
                    self.upsample_final(self.final2(x0_2)),
                    self.upsample_final(self.final3(x0_3)),
                    self.upsample_final(self.final4(x0_4))]
        else:
            return self.upsample_final(self.final(x0_4))
