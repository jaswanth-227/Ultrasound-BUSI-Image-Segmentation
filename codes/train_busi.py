# =========================================================
# DDTI THYROID ULTRASOUND SEGMENTATION - KAGGLE A TO Z
# STAGE 2 ONLY
# MODEL: ConvNeXt-Base + ASPP + Attention Decoder + CBAM + Edge Supervision
# =========================================================

# =========================
# 0. INSTALLS
# =========================
!pip install -q timm albumentations opencv-python-headless

# =========================
# 1. IMPORTS
# =========================
import os
import gc
import cv2
import math
import glob
import time
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

HF_TOKEN = os.getenv("HF_TOKEN")
warnings.filterwarnings("ignore")

# =========================
# 2. SYSTEM INFO
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("DEVICE           :", device)
print("CUDA AVAILABLE   :", torch.cuda.is_available())
print("CUDA DEVICE COUNT:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU NAME         :", torch.cuda.get_device_name(0))
print("PYTORCH VERSION  :", torch.__version__)
print("MODEL            : ConvNeXt-Base + ASPP + Attention Decoder + Edge Head")
print("LOSS             : BCE + Dice + Aux + Edge")
print("OPTIMIZER        : AdamW")
print("=" * 60)

# =========================
# 3. SEED
# =========================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

seed_everything(42)
# =========================
# 4. PATHS - BUSI DATASET
# =========================

base_dir = "/kaggle/input/datasets/mdtasnimulhassankhan/busi-dataset/Dataset BUSI"

img_dirs = [
    os.path.join(base_dir, "benign"),
    os.path.join(base_dir, "malignant"),
    os.path.join(base_dir, "normal")
]

print("BASE DIR :", base_dir)

# =========================
# 5. DATA PAIRING
# =========================
def build_pairs(img_dirs):
    data = []

    for img_dir in img_dirs:
        img_files = glob.glob(os.path.join(img_dir, "*.png"))

        for img_path in img_files:
            if "_mask" in img_path:
                continue

            base_name = os.path.splitext(img_path)[0]

            # Try both mask formats
            possible_masks = [
                base_name + "_mask.png",
                base_name + "_mask_1.png"
            ]

            mask_path = None
            for m in possible_masks:
                if os.path.exists(m):
                    mask_path = m
                    break

            if mask_path is not None:
                data.append({
                    "id": os.path.basename(img_path),
                    "image_path": img_path,
                    "mask_path": mask_path
                })

    df = pd.DataFrame(data)
    return df
df = build_pairs(img_dirs)
print("Total paired samples:", len(df))

# =========================
# 6. TRAIN/VAL/TEST SPLIT
# =========================
train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, shuffle=True)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Train samples: {len(train_df)}")
print(f"Val samples  : {len(val_df)}")
print(f"Test samples : {len(test_df)}")

# =========================
# 7. CONFIG
# =========================
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
PIN_MEMORY = True
MODEL_SAVE_PATH = "/kaggle/working/best_ddti_stage2_convnext_boundary.pth"

print("=" * 60)
print("IMG_SIZE    :", IMG_SIZE)
print("BATCH_SIZE  :", BATCH_SIZE)
print("EPOCHS      :", EPOCHS)
print("LR          :", LR)
print("WEIGHT_DECAY:", WEIGHT_DECAY)
print("SAVE PATH   :", MODEL_SAVE_PATH)
print("=" * 60)

# =========================
# 8. AUGMENTATIONS
# =========================
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.10, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    A.OneOf([
        A.GaussNoise(p=1.0),
        A.MotionBlur(blur_limit=3, p=1.0),
        A.MedianBlur(blur_limit=3, p=1.0),
    ], p=0.2),
    A.OneOf([
        A.RandomBrightnessContrast(p=1.0),
        A.CLAHE(p=1.0),
    ], p=0.25),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# =========================
# 9. DATASET
# =========================
class DDTIDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["image_path"]
        mask_path = self.df.iloc[idx]["mask_path"]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")

        mask = (mask > 127).astype(np.float32)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        mask = mask.unsqueeze(0).float()

        return image, mask, os.path.basename(img_path)

train_dataset = DDTIDataset(train_df, transform=train_transform)
val_dataset = DDTIDataset(val_df, transform=val_transform)
test_dataset = DDTIDataset(test_df, transform=val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=False
)

# =========================
# 10. VISUALIZE FEW SAMPLES
# =========================
def denorm_image(t):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    x = t.permute(1, 2, 0).cpu().numpy()
    x = (x * std + mean)
    x = np.clip(x, 0, 1)
    return x

sample_batch = next(iter(train_loader))
sample_imgs, sample_masks, sample_names = sample_batch

plt.figure(figsize=(12, 8))
for i in range(min(4, len(sample_imgs))):
    plt.subplot(4, 2, 2 * i + 1)
    plt.imshow(denorm_image(sample_imgs[i]))
    plt.title(f"Image: {sample_names[i]}")
    plt.axis("off")

    plt.subplot(4, 2, 2 * i + 2)
    plt.imshow(sample_masks[i][0].cpu().numpy(), cmap="gray")
    plt.title("Mask")
    plt.axis("off")
plt.tight_layout()
plt.show()

# =========================
# 11. MODEL BLOCKS
# =========================
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ChannelAttention(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        hidden = max(ch // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, ch, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = self.mlp(self.avg_pool(x))
        m = self.mlp(self.max_pool(x))
        return x * self.sigmoid(a + m)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca = ChannelAttention(ch)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )
        self.b4 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 4, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return self.project(x)


class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch)
        )
        self.psi = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(inter_ch, 1, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(g1 + x1)
        return x * psi


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.attn = AttentionGate(in_ch, skip_ch, out_ch)
        self.conv1 = ConvBNAct(in_ch + skip_ch, out_ch, k=3, p=1)
        self.conv2 = ConvBNAct(out_ch, out_ch, k=3, p=1)
        self.cbam = CBAM(out_ch)
        self.drop = nn.Dropout2d(0.1)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        skip = self.attn(x, skip)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.cbam(x)
        x = self.drop(x)
        return x


class EdgeHead(nn.Module):
    def __init__(self, in_ch, mid_ch=64):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_ch, mid_ch, k=3, p=1),
            ConvBNAct(mid_ch, mid_ch, k=3, p=1),
            nn.Conv2d(mid_ch, 1, kernel_size=1)
        )

    def forward(self, x, out_size):
        e = self.block(x)
        e = F.interpolate(e, size=out_size, mode="bilinear", align_corners=False)
        return e


# =========================
# 12. MAIN MODEL
# =========================
class DDTIBoundaryNet(nn.Module):
    def __init__(self, encoder_name="convnext_base.fb_in22k_ft_in1k", pretrained=True, num_classes=1):
        super().__init__()

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )

        encoder_channels = self.encoder.feature_info.channels()
        c1, c2, c3, c4 = encoder_channels

        self.stem_refine = nn.Sequential(
            ConvBNAct(c1, c1, k=3, p=1),
            CBAM(c1)
        )

        self.bridge = ASPP(c4, 512)

        self.dec3 = DecoderBlock(512, c3, 256)
        self.dec2 = DecoderBlock(256, c2, 128)
        self.dec1 = DecoderBlock(128, c1, 96)

        edge_feat_ch = min(c1, 64)

        self.edge_proj = nn.Sequential(
            ConvBNAct(c1, edge_feat_ch, k=3, p=1),
            CBAM(edge_feat_ch)
        )

        self.edge_head = EdgeHead(c1, 64)

        self.fuse = nn.Sequential(
            ConvBNAct(96 + edge_feat_ch, 96, k=3, p=1),
            CBAM(96),
            ConvBNAct(96, 96, k=3, p=1)
        )

        self.seg_head = nn.Conv2d(96, num_classes, kernel_size=1)

        self.aux3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.aux2 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.aux1 = nn.Conv2d(96, num_classes, kernel_size=1)

    def forward(self, x):
        h, w = x.shape[2:]

        feats = self.encoder(x)
        f1, f2, f3, f4 = feats

        f1 = self.stem_refine(f1)
        bridge = self.bridge(f4)

        d3 = self.dec3(bridge, f3)
        d2 = self.dec2(d3, f2)
        d1 = self.dec1(d2, f1)

        edge_feat = F.interpolate(self.edge_proj(f1), size=d1.shape[2:], mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat([d1, edge_feat], dim=1))

        seg = self.seg_head(fused)
        seg = F.interpolate(seg, size=(h, w), mode="bilinear", align_corners=False)

        edge = self.edge_head(f1, (h, w))

        aux3 = F.interpolate(self.aux3(d3), size=(h, w), mode="bilinear", align_corners=False)
        aux2 = F.interpolate(self.aux2(d2), size=(h, w), mode="bilinear", align_corners=False)
        aux1 = F.interpolate(self.aux1(d1), size=(h, w), mode="bilinear", align_corners=False)

        return {
            "mask": seg,
            "edge": edge,
            "aux3": aux3,
            "aux2": aux2,
            "aux1": aux1
        }

# =========================
# 13. LOSSES
# =========================
def dice_loss(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    probs = probs.contiguous().view(probs.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)
    intersection = (probs * targets).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + smooth)
    return 1.0 - dice.mean()

def bce_dice_loss(logits, targets, bce_weight=0.5, dice_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dloss = dice_loss(logits, targets)
    return bce_weight * bce + dice_weight * dloss

def make_boundary_targets(mask):
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)

    gx = F.conv2d(mask, sobel_x, padding=1)
    gy = F.conv2d(mask, sobel_y, padding=1)
    g = torch.sqrt(gx ** 2 + gy ** 2)
    g = (g > 0).float()
    return g

def total_loss(outputs, mask_gt, aux_weight=0.2, edge_weight=0.2):
    main_loss = bce_dice_loss(outputs["mask"], mask_gt)
    aux1_loss = bce_dice_loss(outputs["aux1"], mask_gt)
    aux2_loss = bce_dice_loss(outputs["aux2"], mask_gt)
    aux3_loss = bce_dice_loss(outputs["aux3"], mask_gt)
    edge_gt = make_boundary_targets(mask_gt)
    edge_loss = F.binary_cross_entropy_with_logits(outputs["edge"], edge_gt)
    total = main_loss + aux_weight * (aux1_loss + aux2_loss + aux3_loss) + edge_weight * edge_loss
    return total

# =========================
# 14. METRICS
# =========================
def compute_batch_metrics(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds_np = preds.detach().cpu().numpy().astype(np.uint8).reshape(-1)
    targets_np = targets.detach().cpu().numpy().astype(np.uint8).reshape(-1)

    precision = precision_score(targets_np, preds_np, zero_division=0)
    recall = recall_score(targets_np, preds_np, zero_division=0)
    f1 = f1_score(targets_np, preds_np, zero_division=0)
    acc = accuracy_score(targets_np, preds_np)

    intersection = np.logical_and(preds_np, targets_np).sum()
    union = np.logical_or(preds_np, targets_np).sum()
    iou = intersection / (union + 1e-7)
    dice = (2.0 * intersection) / (preds_np.sum() + targets_np.sum() + 1e-7)

    return precision, recall, f1, iou, dice, acc

# =========================
# 15. TRAIN / VALIDATE
# =========================
def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    precision_list, recall_list, f1_list, iou_list, dice_list, acc_list = [], [], [], [], [], []

    pbar = tqdm(loader, total=len(loader), desc="Train", leave=False)

    for images, masks, _ in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = total_loss(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        precision, recall, f1, iou, dice, acc = compute_batch_metrics(outputs["mask"], masks)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        iou_list.append(iou)
        dice_list.append(dice)
        acc_list.append(acc)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "dice": f"{dice:.4f}",
            "iou": f"{iou:.4f}"
        })

    metrics = {
        "loss": running_loss / len(loader),
        "precision": np.mean(precision_list),
        "recall": np.mean(recall_list),
        "f1": np.mean(f1_list),
        "iou": np.mean(iou_list),
        "dice": np.mean(dice_list),
        "accuracy": np.mean(acc_list)
    }
    return metrics

@torch.no_grad()
def valid_one_epoch(model, loader, device):
    model.eval()
    running_loss = 0.0
    precision_list, recall_list, f1_list, iou_list, dice_list, acc_list = [], [], [], [], [], []

    pbar = tqdm(loader, total=len(loader), desc="Valid", leave=False)

    for images, masks, _ in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = total_loss(outputs, masks)

        running_loss += loss.item()

        precision, recall, f1, iou, dice, acc = compute_batch_metrics(outputs["mask"], masks)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        iou_list.append(iou)
        dice_list.append(dice)
        acc_list.append(acc)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "dice": f"{dice:.4f}",
            "iou": f"{iou:.4f}"
        })

    metrics = {
        "loss": running_loss / len(loader),
        "precision": np.mean(precision_list),
        "recall": np.mean(recall_list),
        "f1": np.mean(f1_list),
        "iou": np.mean(iou_list),
        "dice": np.mean(dice_list),
        "accuracy": np.mean(acc_list)
    }
    return metrics

# =========================
# 16. MODEL INIT
# =========================
model = DDTIBoundaryNet(
    encoder_name="convnext_base.fb_in22k_ft_in1k",
    pretrained=True,
    num_classes=1
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

# =========================
# 17. TRAINING LOOP
# =========================
history = []
best_dice = -1

start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch [{epoch}/{EPOCHS}]")
    print("-" * 60)

    train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device)
    val_metrics = valid_one_epoch(model, val_loader, device)

    scheduler.step()

    row = {
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "train_precision": train_metrics["precision"],
        "train_recall": train_metrics["recall"],
        "train_f1": train_metrics["f1"],
        "train_iou": train_metrics["iou"],
        "train_dice": train_metrics["dice"],
        "train_accuracy": train_metrics["accuracy"],
        "val_loss": val_metrics["loss"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
        "val_iou": val_metrics["iou"],
        "val_dice": val_metrics["dice"],
        "val_accuracy": val_metrics["accuracy"],
        "lr": optimizer.param_groups[0]["lr"]
    }
    history.append(row)

    print(
        f"Train Loss: {train_metrics['loss']:.4f} | "
        f"P: {train_metrics['precision']:.4f} | "
        f"R: {train_metrics['recall']:.4f} | "
        f"F1: {train_metrics['f1']:.4f} | "
        f"IoU: {train_metrics['iou']:.4f} | "
        f"Dice: {train_metrics['dice']:.4f} | "
        f"Acc: {train_metrics['accuracy']:.4f}"
    )

    print(
        f"Val   Loss: {val_metrics['loss']:.4f} | "
        f"P: {val_metrics['precision']:.4f} | "
        f"R: {val_metrics['recall']:.4f} | "
        f"F1: {val_metrics['f1']:.4f} | "
        f"IoU: {val_metrics['iou']:.4f} | "
        f"Dice: {val_metrics['dice']:.4f} | "
        f"Acc: {val_metrics['accuracy']:.4f}"
    )

    if val_metrics["dice"] > best_dice:
        best_dice = val_metrics["dice"]
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Best model saved at epoch {epoch} with Val Dice = {best_dice:.4f}")

history_df = pd.DataFrame(history)
history_df.to_csv("/kaggle/working/training_history_stage2.csv", index=False)

total_time = time.time() - start_time
print(f"\nTraining completed in {total_time/60:.2f} minutes")
print("Best Val Dice:", best_dice)

# =========================
# 18. PLOTS
# =========================
plt.figure(figsize=(8, 5))
plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history_df["epoch"], history_df["train_dice"], label="Train Dice")
plt.plot(history_df["epoch"], history_df["val_dice"], label="Val Dice")
plt.xlabel("Epoch")
plt.ylabel("Dice")
plt.title("Dice Curve")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history_df["epoch"], history_df["train_iou"], label="Train IoU")
plt.plot(history_df["epoch"], history_df["val_iou"], label="Val IoU")
plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.title("IoU Curve")
plt.legend()
plt.grid(True)
plt.show()

# =========================
# 19. LOAD BEST MODEL
# =========================
best_model = DDTIBoundaryNet(
    encoder_name="convnext_base.fb_in22k_ft_in1k",
    pretrained=False,
    num_classes=1
).to(device)

best_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
best_model.eval()

# =========================
# 20. TEST EVALUATION
# =========================
@torch.no_grad()
def evaluate_test(model, loader, device):
    model.eval()

    precision_list, recall_list, f1_list, iou_list, dice_list, acc_list = [], [], [], [], [], []
    total_loss = 0.0

    for images, masks, _ in tqdm(loader, total=len(loader), desc="Test", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(images)
        loss = bce_dice_loss(outputs["mask"], masks)
        total_loss += loss.item()

        precision, recall, f1, iou, dice, acc = compute_batch_metrics(outputs["mask"], masks)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        iou_list.append(iou)
        dice_list.append(dice)
        acc_list.append(acc)

    metrics = {
        "loss": total_loss / len(loader),
        "precision": np.mean(precision_list),
        "recall": np.mean(recall_list),
        "f1": np.mean(f1_list),
        "iou": np.mean(iou_list),
        "dice": np.mean(dice_list),
        "accuracy": np.mean(acc_list)
    }
    return metrics

test_metrics = evaluate_test(best_model, test_loader, device)

results_df = pd.DataFrame([{
    "Model": "ConvNeXtBase_ASPP_Attn_CBAM_Edge",
    "Precision": round(test_metrics["precision"], 4),
    "Recall": round(test_metrics["recall"], 4),
    "F1": round(test_metrics["f1"], 4),
    "IoU": round(test_metrics["iou"], 4),
    "Dice": round(test_metrics["dice"], 4),
    "Accuracy": round(test_metrics["accuracy"], 4)
}])

print("\nFinal Test Results")
display(results_df)
results_df.to_csv("/kaggle/working/final_test_results_stage2.csv", index=False)

# =========================
# 21. VISUALIZE PREDICTIONS
# =========================
@torch.no_grad()
def visualize_predictions(model, loader, device, num_samples=6):
    model.eval()
    shown = 0

    for images, masks, names in loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        probs = torch.sigmoid(outputs["mask"])
        preds = (probs > 0.5).float()

        for i in range(images.size(0)):
            if shown >= num_samples:
                return

            img = denorm_image(images[i].cpu())
            gt = masks[i, 0].cpu().numpy()
            pred = preds[i, 0].cpu().numpy()
            heat = probs[i, 0].cpu().numpy()

            plt.figure(figsize=(14, 4))

            plt.subplot(1, 4, 1)
            plt.imshow(img)
            plt.title(f"Image\n{names[i]}")
            plt.axis("off")

            plt.subplot(1, 4, 2)
            plt.imshow(gt, cmap="gray")
            plt.title("Ground Truth")
            plt.axis("off")

            plt.subplot(1, 4, 3)
            plt.imshow(pred, cmap="gray")
            plt.title("Prediction")
            plt.axis("off")

            plt.subplot(1, 4, 4)
            plt.imshow(heat, cmap="jet")
            plt.title("Probability Map")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

            shown += 1

visualize_predictions(best_model, test_loader, device, num_samples=6)

# =========================
# 22. SAVE TEST PREDICTIONS
# =========================
save_pred_dir = "/kaggle/working/test_predictions_stage2"
os.makedirs(save_pred_dir, exist_ok=True)

@torch.no_grad()
def save_predictions(model, loader, device, save_dir):
    model.eval()
    for images, masks, names in tqdm(loader, total=len(loader), desc="Saving Predictions", leave=False):
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs["mask"])
        preds = (probs > 0.5).float()

        for i in range(images.size(0)):
            pred = preds[i, 0].cpu().numpy().astype(np.uint8) * 255
            save_path = os.path.join(save_dir, names[i])
            cv2.imwrite(save_path, pred)

save_predictions(best_model, test_loader, device, save_pred_dir)
print("Saved predicted masks to:", save_pred_dir)

# =========================
# 23. FINAL SUMMARY
# =========================
print("=" * 60)
print("STAGE 2 ONLY TRAINING DONE")
print("Best model path:", MODEL_SAVE_PATH)
print("History path   : /kaggle/working/training_history_stage2.csv")
print("Results path   : /kaggle/working/final_test_results_stage2.csv")
print("Pred masks dir : /kaggle/working/test_predictions_stage2")
print("=" * 60)

print("\nFinal Test Metrics")
for k, v in test_metrics.items():
    print(f"{k:10s}: {v:.4f}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()