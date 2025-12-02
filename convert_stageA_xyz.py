#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage A inference: sRGB image -> XYZ image

- 加载用 train_stageA_xyz.py 训练好的 StageAXYZNet 权重
- 输入: 整张 sRGB 图 (.png / .jpg / .jpeg / .tif / .npy)
- 输出:
    1) XYZ 整图 .npy (float32, 原汁原味保留网络输出，不做任何颜色空间变换)
    2) (可选 --vis) “原始 XYZ 可视化” PNG:
       - 不做 XYZ->sRGB 矩阵变换
       - 不做 gamma
       - X/Y/Z 三个通道直接映射到 PNG 的 R/G/B 通道
       - 只做 clip 到 [0,1] 然后线性映射到 [0,255]
       - 与 rawpy_raw_to_xyz_phase1.py 中的 xyz_to_png_raw 完全一致

用法示例:
    python3 convert_stageA_xyz.py \\
        -g 0 \\
        -f ./outputs_xyz/1764653715 \\
        -i /mnt/data/FiveK_Canon/sRGB_test3 \\
        -o ./converted_stageA_test3 \\
        --vis
"""

import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn

from resources.utils import load_cfg


# ==========================
# 模型: 与 train_stageA_xyz.py 保持一致
# ==========================

class StageAXYZNet(nn.Module):
    """
    一个简单的 3->3 卷积网络（与 train_stageA_xyz.py 中的定义保持一致）

    输入: sRGB (B, 3, H, W) in [0,1]
    输出: XYZ  (B, 3, H, W) in [0,1] (由训练数据分布决定)
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

    def forward(self, sample, context=None):
        # 当前版本忽略 context，只使用 sample
        x = self.encoder(sample)
        out = self.decoder(x)
        # 为了兼容 train_stageA_xyz 风格，返回三元组
        y = out
        outputs = out
        aux = None
        return y, outputs, aux


# ==========================
# XYZ -> PNG 可视化 (原汁原味，和 rawpy 脚本一致)
# ==========================

def xyz_to_png_raw(xyz_float: np.ndarray) -> np.ndarray:
    """
    将原始 XYZ(float32) 直接可视化为 PNG：
    - 不做矩阵变换
    - 不做 gamma
    - 不做色彩空间转换
    - 不做任何美化

    只是简单地把每个通道裁剪到 [0,1]，映射到 [0,255]，以便保存 PNG。

    与 rawpy_raw_to_xyz_phase1.py 中的 xyz_to_png_raw 完全一致。
    """
    img = np.clip(xyz_float, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    return img


# ==========================
# 单张图推理
# ==========================

def run_stageA_xyz(model, file_name, input_path, output_path, device, to_vis):
    # 拆文件名 & 扩展名
    if "." in file_name:
        stem, ext = file_name.rsplit(".", 1)
    else:
        stem, ext = file_name, ""

    input_file_path = os.path.join(input_path, file_name)

    # 读入 sRGB
    if ext.lower() == "npy":
        img = np.load(input_file_path).astype(np.float32)
        # 如果是 (H, W) 单通道，就简单复制成三通道
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
    elif ext.lower() in ["jpg", "jpeg", "png", "tif", "tiff", "bmp"]:
        bgr = cv2.imread(input_file_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] Cannot read image {input_file_path}, skip.")
            return
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    else:
        print(f"[WARN] Unsupported extension .{ext} for {file_name}, skip.")
        return

    H, W = img.shape[:2]

    # 归一化到 [0,1]，与训练时保持一致 (train_stageA_xyz: /255.0)
    img_norm = img / 255.0

    # 转成 tensor: (1, 3, H, W)
    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # context 暂时不用，但为了接口完整，传一个全零即可
    context_tensor = torch.zeros_like(img_tensor)

    model.eval()
    with torch.no_grad():
        # 如果有 GPU，就用 FP16 autocast；否则直接 float32
        if device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                _, out_xyz, _ = model(img_tensor, context_tensor)
        else:
            _, out_xyz, _ = model(img_tensor, context_tensor)

    # 回到 numpy: (H, W, 3)
    out_xyz_np = out_xyz.squeeze(0).cpu().float().numpy().transpose(1, 2, 0)

    # === 保存 XYZ (原汁原味，不做任何处理) ===
    os.makedirs(output_path, exist_ok=True)
    save_xyz_path = os.path.join(output_path, f"{stem}_xyz.npy")
    np.save(save_xyz_path, out_xyz_np.astype(np.float32))

    # === 可视化: 原始 XYZ 映射到 PNG，仅用于 eyeball，不影响 .npy 内容 ===
    if to_vis:
        xyz_vis = xyz_to_png_raw(out_xyz_np)     # (H, W, 3), uint8, X/Y/Z 映射到 R/G/B
        # 通道顺序已经是 "RGB" 语义，这里只是给 OpenCV 转成 BGR 存盘
        vis_bgr = cv2.cvtColor(xyz_vis, cv2.COLOR_RGB2BGR)
        vis_path = os.path.join(output_path, f"{stem}_xyz_raw_preview.png")
        cv2.imwrite(vis_path, vis_bgr)

    # 清理 GPU 显存碎片
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ==========================
# main
# ==========================

def main():
    parser = argparse.ArgumentParser(
        description="Stage A: convert sRGB images to XYZ using StageAXYZNet."
    )
    parser.add_argument(
        "-g", "--gpu", type=int, default=0, help="GPU id to use."
    )
    parser.add_argument(
        "-f", "--folder", type=str, required=True,
        help="Checkpoint folder (e.g., ./outputs_xyz/1764653715)"
    )
    parser.add_argument(
        "-i", "--input_path", type=str, required=True,
        help="Input image folder (sRGB)"
    )
    parser.add_argument(
        "-o", "--output_path", type=str, required=True,
        help="Output folder for XYZ npy and optional visualisations"
    )
    parser.add_argument(
        "--vis", action="store_true",
        help="Also generate raw-style XYZ visualisation PNG (X/Y/Z -> R/G/B)."
    )
    args = parser.parse_args()

    cuda_no = str(args.gpu)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cuda_no}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    checkpoint_folder = args.folder
    input_path = args.input_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    # ===== 载入 cfg =====
    # train_stageA_xyz.py 中保存的是 cfg_stageA_xyz.py
    cfg_stageA_path = os.path.join(checkpoint_folder, "cfg_stageA_xyz.py")
    if not os.path.exists(cfg_stageA_path):
        # 兜底: 如果没有这个文件，可以尝试 cfg.py
        cfg_stageA_path = os.path.join(checkpoint_folder, "cfg.py")
        if not os.path.exists(cfg_stageA_path):
            raise FileNotFoundError(
                f"Cannot find cfg_stageA_xyz.py or cfg.py in {checkpoint_folder}"
            )

    dataset_cfg, cfg_sample, cfg_train = load_cfg(cfg_stageA_path)

    # ===== 初始化模型 =====
    hidden_size = cfg_train.get("hidden_size", 128)
    depth = cfg_train.get("depth", 4)

    model = StageAXYZNet(
        hidden_size=hidden_size,
        depth=depth,
    ).to(device)

    # 权重文件: train_stageA_xyz.py 中保存的是 best_stageA_xyz.pth
    model_path = os.path.join(checkpoint_folder, "best_stageA_xyz.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find model weights: {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # ===== 收集输入文件 =====
    input_files = [
        f for f in os.listdir(input_path)
        if os.path.isfile(os.path.join(input_path, f))
    ]
    input_files.sort()

    print("=== Stage A sRGB -> XYZ convert ===")
    print(f"Checkpoint folder: {checkpoint_folder}")
    print(f"Config file      : {cfg_stageA_path}")
    print(f"Model weights    : {model_path}")
    print(f"Input path       : {input_path}")
    print(f"Output path      : {output_path}")
    print(f"Num images       : {len(input_files)}")
    print(f"Visualisation    : {args.vis}")
    print("====================================")

    # ===== 逐张处理 =====
    with tqdm(total=len(input_files)) as pbar:
        for fname in input_files:
            run_stageA_xyz(
                model,
                fname,
                input_path,
                output_path,
                device,
                args.vis,
            )
            pbar.update(1)


if __name__ == "__main__":
    main()