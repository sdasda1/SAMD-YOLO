# ================================================================
# 完整优化版: ResNeSt50 + Attention UNet + AMP + Augmentation
# ================================================================

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import torch.nn.functional as F
import timm

# [新增 1] 引入混合精度训练库
from torch.cuda.amp import autocast, GradScaler
# [新增 2] 引入图像增强库 (如果没有请 pip install albumentations)
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 颜色
COLOR_MAP = {
    0: (0, 0, 0),  # empty
    1: (0, 255, 0),  # castingsurface
    2: (255, 0, 0),  # Machinedsurface
}


# ======================
# 0. 可视化工具
# ======================
def colorize_mask(mask):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for k, v in COLOR_MAP.items():
        color[mask == k] = v
    return color


def visualize_predictions(model, dataloader, device, save_dir, num=5):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for i, (img, mask) in enumerate(dataloader):
            if i >= num:
                break

            img = img.to(device)
            mask = mask.to(device)

            # 预测时也需要 AMP 上下文，虽然主要影响速度
            with autocast():
                pred = model(img).argmax(1)

            # 反归一化用于显示 (针对 ImageNet Mean/Std)
            # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            img_np = img[0].cpu().permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)
            img_np = (img_np * 255).astype(np.uint8)

            gt = mask[0].cpu().numpy()
            pr = pred[0].cpu().numpy()

            gt_color = colorize_mask(gt)
            pr_color = colorize_mask(pr)

            vis = np.hstack([img_np, gt_color, pr_color])
            vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, f"sample_{i}.png"), vis)


# ======================
# 1. Dataset (集成增强与标准化)
# ======================
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform  # [修改] 接收增强变换
        self.names = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        # 兼容 mask 可能是 png 但 image 是 jpg 的情况
        base = os.path.splitext(name)[0]

        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, base + ".png")

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # [修改] 使用 albumentations 进行增强和标准化
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        else:
            # 验证集如果不传 transform，至少要 resize 和 toTensor
            # 但通常我们在外面定义 val_transform
            pass

        return img, mask.long()  # mask 不需要 float, long 即可


# ======================
# 2. ResNeSt50_UNet_Attention
# ======================
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        if g.size()[2:] != x.size()[2:]:
            g = F.interpolate(g, size=x.size()[2:], mode='bilinear', align_corners=False)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResNeSt50AttentionUNet(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            'resnest50d',
            features_only=True,
            pretrained=pretrained,
            out_indices=(0, 1, 2, 3, 4)
        )
        filters = [64, 256, 512, 1024, 2048]

        # Decoder & Attention Gates
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.att4 = AttentionGate(F_g=filters[4], F_l=filters[3], F_int=512)
        self.dec4 = ConvBlock(filters[4] + filters[3], 512)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.att3 = AttentionGate(F_g=512, F_l=filters[2], F_int=256)
        self.dec3 = ConvBlock(512 + filters[2], 256)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.att2 = AttentionGate(F_g=256, F_l=filters[1], F_int=128)
        self.dec2 = ConvBlock(256 + filters[1], 128)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.att1 = AttentionGate(F_g=128, F_l=filters[0], F_int=64)
        self.dec1 = ConvBlock(128 + filters[0], 64)

        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x, return_features=False):
        features = self.backbone(x)
        e1, e2, e3, e4, e5 = features[0], features[1], features[2], features[3], features[4]

        d4_up = self.up4(e5)
        if d4_up.size()[2:] != e4.size()[2:]:
            d4_up = F.interpolate(d4_up, size=e4.shape[2:], mode='bilinear', align_corners=False)
        e4_att = self.att4(g=d4_up, x=e4)
        d4 = self.dec4(torch.cat([d4_up, e4_att], dim=1))

        d3_up = self.up3(d4)
        if d3_up.size()[2:] != e3.size()[2:]:
            d3_up = F.interpolate(d3_up, size=e3.shape[2:], mode='bilinear', align_corners=False)
        e3_att = self.att3(g=d3_up, x=e3)
        d3 = self.dec3(torch.cat([d3_up, e3_att], dim=1))

        d2_up = self.up2(d3)
        if d2_up.size()[2:] != e2.size()[2:]:
            d2_up = F.interpolate(d2_up, size=e2.shape[2:], mode='bilinear', align_corners=False)
        e2_att = self.att2(g=d2_up, x=e2)
        d2 = self.dec2(torch.cat([d2_up, e2_att], dim=1))

        d1_up = self.up1(d2)
        if d1_up.size()[2:] != e1.size()[2:]:
            d1_up = F.interpolate(d1_up, size=e1.shape[2:], mode='bilinear', align_corners=False)
        e1_att = self.att1(g=d1_up, x=e1)
        d1 = self.dec1(torch.cat([d1_up, e1_att], dim=1))

        d0 = self.up0(d1)
        if d0.size()[2:] != x.size()[2:]:
            d0 = F.interpolate(d0, size=x.shape[2:], mode='bilinear', align_corners=False)

        # ===== [导师注入：支持特征导出] =====
        out = self.final(d0)
        if return_features:
            # d0 的通道数正好是 64，这是我们需要的高维语义特征！
            return out, d0
        else:
            return out
        # =================================


# ======================
# 3. Loss Functions
# ======================
def lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1. - intersection / union
    jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax_flat(probs, labels, classes='present'):
    C = probs.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0:
            continue
        errors = (fg - probs[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    return torch.mean(torch.stack(losses))


class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        B, C, H, W = probs.shape
        probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, C)
        targets = targets.view(-1)
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            probs = probs[mask]
            targets = targets[mask]
        return lovasz_softmax_flat(probs, targets)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiceLossSingleClass(nn.Module):
    def __init__(self, class_index, smooth=1.0):
        super().__init__()
        self.class_index = class_index
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        pred = probs[:, self.class_index, :, :]
        gt = (targets == self.class_index).float()
        intersection = (pred * gt).sum(dim=(1, 2))
        union = pred.sum(dim=(1, 2)) + gt.sum(dim=(1, 2))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalDiceLoss(nn.Module):
    def __init__(self, class_weights, cast_class=1, gamma=2.0, dice_weight=1.0):
        super().__init__()
        self.focal = FocalLoss(alpha=class_weights, gamma=gamma)
        self.dice = DiceLossSingleClass(class_index=cast_class)
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        loss_focal = self.focal(logits, targets)
        loss_dice = self.dice(logits, targets)
        return loss_focal + self.dice_weight * loss_dice


class FocalDiceLovaszLoss(nn.Module):
    def __init__(self, class_weights, cast_class=1, gamma=2.0, dice_weight=2.0, lovasz_weight=0.75):
        super().__init__()
        self.focal_dice = FocalDiceLoss(class_weights=class_weights, cast_class=cast_class, gamma=gamma,
                                        dice_weight=dice_weight)
        self.lovasz = LovaszSoftmaxLoss(ignore_index=None)
        self.lovasz_weight = lovasz_weight

    def update_lovasz_weight(self, new_weight):
        self.lovasz_weight = new_weight

    def forward(self, logits, targets):
        loss_fd = self.focal_dice(logits, targets)
        loss_lz = self.lovasz(logits, targets)
        return loss_fd + self.lovasz_weight * loss_lz


# ======================
# 4. Metrics
# ======================
def compute_miou(pred, mask, num_classes=3):
    pred = pred.view(-1)
    mask = mask.view(-1)
    ious = []
    for c in range(num_classes):
        inter = ((pred == c) & (mask == c)).sum().item()
        union = ((pred == c) | (mask == c)).sum().item()
        if union > 0:
            ious.append(inter / union)
    return sum(ious) / len(ious) if ious else 0.0


# ======================
# 5. Train Main
# ======================
def main():
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    set_seed(42)

    # [新增 3] 定义数据增强和 ImageNet 标准化
    # 工业零件通常可以翻转、旋转而不改变属性
    train_transform = A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # A.RandomBrightnessContrast(p=0.2), # 如果需要更强增强可解开
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 实例化数据集
    train_ds = SegDataset("datasets/train/images", "datasets/train/masks", transform=train_transform)
    val_ds = SegDataset("datasets/val/images", "datasets/val/masks", transform=val_transform)

    # [修改] Batch Size 设为 16 (A5000 + AMP 可以轻松跑)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, pin_memory=True)

    model = ResNeSt50AttentionUNet(num_classes=3, pretrained=True).to(device)

    # 权重计算: 你可以使用计算出的，也可以使用之前讨论的强力压制背景的 [0.2, 1.0, 1.0]
    # 这里为了稳妥，保留了自动计算逻辑。
    # 建议：如果训练出来背景误报依然高，请强制设为 torch.tensor([0.2, 1.0, 1.0]).to(device)
    def compute_class_weights(mask_dir, num_classes=3):
        counts = np.zeros(num_classes, dtype=np.float64)
        for f in os.listdir(mask_dir):
            if not f.endswith(".png"): continue
            mask = cv2.imread(os.path.join(mask_dir, f), 0)
            for c in range(num_classes):
                counts[c] += np.sum(mask == c)
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * num_classes
        return torch.tensor(weights, dtype=torch.float32)


    class_weights = compute_class_weights("datasets/train/masks", num_classes=3).to(device)

    # class_weights = torch.tensor([0.4, 2.0, 1.0]).to(device) # 手动强力压制背景版

    criterion = FocalDiceLovaszLoss(
        class_weights=class_weights,
        cast_class=1,
        gamma=2.0,
        dice_weight=2.0,
        lovasz_weight=0.1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10, verbose=True)

    # [新增 4] AMP Scaler
    scaler = GradScaler()

    best_miou = 0.0
    os.makedirs("checkpoints", exist_ok=True)


    for epoch in range(1, 151):
        # ---- train ----
        model.train()
        train_loss = 0

        # 动态权重策略
        if epoch < 20:
            criterion.update_lovasz_weight(0.3)
        else:
            # Batch Size >= 16 时，直接拉满到 1.0
            criterion.update_lovasz_weight(0.75)

        for img, mask in tqdm(train_loader, desc=f"Epoch {epoch}"):
            img, mask = img.to(device), mask.to(device)

            optimizer.zero_grad()

            # [新增 5] 使用 AMP 前向传播
            with autocast():
                pred = model(img)
                loss = criterion(pred, mask)

            # [新增 6] 使用 Scaler 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        # ---- Validation ----
        model.eval()
        miou = 0
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(device), mask.to(device)
                # 验证时也建议开启 autocast 以节省显存（可选）
                with autocast():
                    pred = model(img).argmax(1)
                miou += compute_miou(pred, mask)

        avg_miou = miou / len(val_loader)
        scheduler.step(avg_miou)

        if avg_miou > best_miou:
            best_miou = avg_miou
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "miou": best_miou
            }, "checkpoints/train_ResNeSt50_UNet_best9.pth")

        if epoch % 10 == 0:
            visualize_predictions(model, val_loader, device, save_dir=f"outputs/train_ResNeSt50_UNet9/epoch_{epoch}", num=3)

        print(f"[Epoch {epoch}] Train Loss: {train_loss / len(train_loader):.4f} | Val mIoU: {avg_miou:.4f}")

    print(f"Training Done. Best mIoU = {best_miou:.4f}")


if __name__ == "__main__":
    main()