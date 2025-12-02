#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage A training script: sRGB patch -> XYZ patch

- 使用 ReRAW 风格的 cfg（dataset, cfg_sample, cfg_train）
- 使用我们已经生成的 patch:
    cfg_sample["output_folder_root"]/srgb-sample/*.npy  (input)
    cfg_sample["output_folder_root"]/xyz-target/*.npy   (GT)
    cfg_sample["output_folder_root"]/srgb-context/*.npy (暂时不用，只是保留接口)
- 模型: 简单的 3->3 卷积网络 (可视为 mini-UNet)
- Loss: L1Loss (sRGB 已 /255, XYZ 假定在 [0,1])
- Eval: 在 val 集上计算 XYZ 域 PSNR
"""

import os
import time
import argparse
from math import log10

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from resources.utils import load_cfg, save_cfg, save_model


# ==========================
# Dataset: Stage A XYZ
# ==========================

class StageAXYZDataset(Dataset):
    """
    从 srgb-sample / xyz-target / srgb-context 目录读取 patch，
    按 split=train/val 做划分。
    """

    def __init__(
        self,
        root_patches: str,
        sample_subdir: str = "srgb-sample",
        target_subdir: str = "xyz-target",
        context_subdir: str = "srgb-context",
        split: str = "train",
        val_ratio: float = 0.1,
        max_samples: int = -1,
        seed: int = 1234,
    ):
        super().__init__()
        self.root_patches = root_patches
        self.sample_dir = os.path.join(root_patches, sample_subdir)
        self.target_dir = os.path.join(root_patches, target_subdir)
        self.context_dir = os.path.join(root_patches, context_subdir)
        self.split = split

        # 收集所有 sample 文件名
        all_files = sorted(
            [f for f in os.listdir(self.sample_dir) if f.endswith(".npy")]
        )

        if max_samples is not None and max_samples > 0:
            all_files = all_files[:max_samples]

        # 固定随机划分 train/val
        rng = np.random.RandomState(seed)
        indices = np.arange(len(all_files))
        rng.shuffle(indices)

        n_total = len(all_files)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_val

        if split == "train":
            use_idx = indices[:n_train]
        else:
            use_idx = indices[n_train:]  # val

        self.files = [all_files[i] for i in use_idx]

        print(
            f"[StageAXYZDataset] split={split}, total={n_total}, "
            f"train={n_train}, val={n_val}, using={len(self.files)}"
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        # 路径
        sample_path = os.path.join(self.sample_dir, fname)
        target_path = os.path.join(self.target_dir, fname)
        context_path = os.path.join(self.context_dir, fname)

        # 读 npy
        sample_np = np.load(sample_path).astype(np.float32)  # sRGB (H, W, 3), 0~255
        target_np = np.load(target_path).astype(np.float32)  # XYZ  (H, W, 3), ~[0,1]
        # context 暂时不用，但保持接口
        if os.path.exists(context_path):
            context_np = np.load(context_path).astype(np.float32)
        else:
            context_np = np.zeros_like(sample_np, dtype=np.float32)

        # 归一化 & 转为 tensor，(C, H, W)
        # sRGB: 0~255 -> 0~1
        sample_np = sample_np / 255.0
        sample = torch.from_numpy(sample_np.transpose(2, 0, 1))  # (3, H, W)

        # XYZ 假定已经在 0~1 之间，如果有越界可以 clamp
        target_np = np.clip(target_np, 0.0, 1.0)
        target = torch.from_numpy(target_np.transpose(2, 0, 1))

        context_np = context_np / 255.0
        context = torch.from_numpy(context_np.transpose(2, 0, 1))

        return sample, target, context, fname


# ==========================
# Model: 简单 3->3 网络
# ==========================

class StageAXYZNet(nn.Module):
    """
    一个简单的 3->3 卷积网络（可看作 mini-UNet 的 encoder-decoder 版本）

    输入: sRGB patch (B, 3, H, W)
    输出: XYZ patch (B, 3, H, W)
    """

    def __init__(self, hidden_size=128, depth=4):
        super().__init__()

        layers = []
        in_ch = 3
        for i in range(depth):
            layers.append(nn.Conv2d(in_ch, hidden_size, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_ch = hidden_size
        self.encoder = nn.Sequential(*layers)

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, 3, kernel_size=3, padding=1),
        )

    def forward(self, sample, context):
        # 当前版本先不使用 context，只使用 sample。
        x = self.encoder(sample)
        out = self.decoder(x)
        # 为了兼容原 train.py 的调用方式，这里返回 (y, outputs, aux)
        y = out        # 主输出
        outputs = out  # 第二个输出也用它
        aux = None
        return y, outputs, aux


# ==========================
# Test: XYZ PSNR
# ==========================

def test_xyz_psnr(model, dataloader, device):
    model.eval()
    mse_sum = 0.0
    n = 0
    with torch.no_grad():
        for data in dataloader:
            sample, target, context, _ = data
            sample = sample.to(device)
            target = target.to(device)

            _, outputs, _ = model(sample, context.to(device))
            mse = torch.mean((outputs - target) ** 2).item()
            mse_sum += mse
            n += 1

    if n == 0:
        return 0.0

    mse_mean = mse_sum / n
    if mse_mean <= 1e-12:
        return 99.0

    # 假定 XYZ 已经近似在 [0,1]
    psnr = 10 * log10(1.0 / mse_mean)
    return psnr


# ==========================
# Train loop
# ==========================

def train(model, trainloader, valloader, save_path, cfg_train, writer, device):
    lr = cfg_train["lr"]
    lr_scaling = cfg_train["lr_scaling"]
    restart = cfg_train["restart"]

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-9)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=restart, eta_min=lr * lr_scaling
    )

    criterion = nn.L1Loss()

    best_psnr = 0.0

    for epoch in range(cfg_train["epochs"]):
        model.train()
        running_loss = 0.0
        last_loss = 0.0

        with tqdm(trainloader, unit="batch", disable=False) as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            for i, data in enumerate(tepoch):
                sample, target, context, _ = data
                sample = sample.to(device)
                target = target.to(device)
                context = context.to(device)

                optimizer.zero_grad()
                _, outputs, _ = model(sample, context)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i == 0:
                    last_loss = loss.item()
                else:
                    last_loss = last_loss * 0.9 + loss.item() * 0.1

                tepoch.set_postfix(loss=f"{last_loss:.5f}")

        avg_train_loss = running_loss / max(len(trainloader), 1)
        writer.add_scalar("train_loss", avg_train_loss, epoch)

        # 验证 PSNR
        psnr = test_xyz_psnr(model, valloader, device)
        writer.add_scalar("val_psnr_xyz", psnr, epoch)

        print(f"[Epoch {epoch}] train_loss={avg_train_loss:.6f}, val_PSNR_XYZ={psnr:.3f} dB")

        # 保存 best 模型
        if psnr > best_psnr:
            best_psnr = psnr
            save_model(model, os.path.join(save_path, "best_stageA_xyz.pth"))
            print(f"  -> New best PSNR: {best_psnr:.3f} dB, model saved.")

        lr_scheduler.step()


# ==========================
# Main
# ==========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage A training: sRGB patch -> XYZ patch."
    )
    parser.add_argument(
        "-g", "--gpu", type=int, default=0, help="GPU used to train."
    )
    parser.add_argument(
        "-c", "--cfg", type=str, default="", help="config file"
    )
    args = parser.parse_args()

    cuda_no = str(args.gpu)
    device = torch.device(f"cuda:{cuda_no}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # 读取 cfg（沿用 ReRAW 的 load_cfg）
    dataset_cfg, cfg_sample, cfg_train = load_cfg(args.cfg)

    # Patch 根目录 (我们在 cfg_sample 里已经定义过)
    patches_root = cfg_sample["output_folder_root"]  # 例如 /mnt/data/FiveK_Canon/stageA_patches_train

    # 构建 train/val Dataset
    max_samples = cfg_train.get("max_samples", -1)
    val_ratio = cfg_train.get("val_ratio", 0.1)

    train_dataset = StageAXYZDataset(
        root_patches=patches_root,
        sample_subdir=cfg_sample["sample_path"].rstrip("/"),
        target_subdir=cfg_sample["target_path"].rstrip("/"),
        context_subdir=cfg_sample["context_path"].rstrip("/"),
        split="train",
        val_ratio=val_ratio,
        max_samples=max_samples,
    )

    val_dataset = StageAXYZDataset(
        root_patches=patches_root,
        sample_subdir=cfg_sample["sample_path"].rstrip("/"),
        target_subdir=cfg_sample["target_path"].rstrip("/"),
        context_subdir=cfg_sample["context_path"].rstrip("/"),
        split="val",
        val_ratio=val_ratio,
        max_samples=max_samples,
    )

    trainloader = DataLoader(
        train_dataset,
        batch_size=cfg_train["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    valloader = DataLoader(
        val_dataset,
        batch_size=cfg_train["batch_size"],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # 模型
    model = StageAXYZNet(
        hidden_size=cfg_train["hidden_size"],
        depth=cfg_train["depth"],
    ).to(device)

    # 工作空间
    TIME_STAMP = int(time.time())
    project_name = f"{TIME_STAMP}"

    save_path = os.path.join(cfg_train["save_path"], project_name)
    os.makedirs(save_path, exist_ok=True)
    print("Saving model at:", save_path)

    writer = SummaryWriter(os.path.join(cfg_train["tensorboard_path"], project_name))

    # 保存实际使用的 config
    save_cfg(cfg_train, cfg_sample, dataset_cfg, os.path.join(save_path, "cfg_stageA_xyz.py"))

    # 开始训练
    train(model, trainloader, valloader, save_path, cfg_train, writer, device)