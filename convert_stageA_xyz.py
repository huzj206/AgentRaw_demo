#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage A Inference: sRGB -> CIE XYZ (full-image, fully convolutional)

- 使用你已经修改好的 ReRAW（out_size=3 + no downsample）
- 整图前向：把整张 sRGB 作为输入 patch（不再切 64x64 小块）
- 使用 StageA 的 cfg.py 和 best.pth
- 输入: sRGB 图像 (png/jpg/npy)，H×W×3, uint8 或 float32
- 输出: 预测的 XYZ .npy (float32, H×W×3, 约在 [0,1])

用法示例：
    python convert_stageA_xyz.py \
        -g 0 \
        -f ./outputs/1764840231 \
        -i /data/umihebi0/users/z-hu/AgentRAW/FiveK_Nikon/sRGB_test3 \
        -o ./result/converted_stageA_test3_nikon_org
"""

import os
import argparse

import cv2
import numpy as np
import torch
from tqdm import tqdm

from resources.models import ReRAW
from resources.utils import load_cfg


# ------------------------------
# 整图前向: sRGB -> XYZ
# ------------------------------
def convert_image_xyz_full(
    model,
    img_rgb,
    context_size=(128, 128),
    scale_back=1.0,
):
    """
    Args:
        model:   ReRAW(in_size=3, out_size=3)，已 .eval() 且在 CUDA 上
        img_rgb: H×W×3, float32, 已经 /256 归一化
        context_size: (Hc, Wc)，给 GlobalContextModule 用
        scale_back: dataset['rggb_max']，用于把训练时缩放还原回 [0,1] 域

    Returns:
        out_xyz: H×W×3, float32, 约在 [0,1]
    """
    model.eval()
    img_h, img_w = img_rgb.shape[:2]

    # 整图 context
    context = cv2.resize(img_rgb, context_size, interpolation=cv2.INTER_AREA)

    # HWC -> NCHW
    x = img_rgb.transpose(2, 0, 1)[None, ...]      # (1,3,H,W)
    g = context.transpose(2, 0, 1)[None, ...]      # (1,3,Hc,Wc)

    x_t = torch.from_numpy(x).cuda(non_blocking=True)
    g_t = torch.from_numpy(g).cuda(non_blocking=True)

    with torch.no_grad():
        y, _, _ = model(x_t, g_t)                  # (1,3,H,W)

    y_np = y[0].detach().cpu().numpy()             # (3,H,W)
    y_np = y_np.transpose(1, 2, 0)                 # (H,W,3)

    # 训练时: target = XYZ / rggb_max
    # 推理:   输出 * rggb_max，再 clip 到 [0,1]
    out_xyz = np.clip(y_np * scale_back, 0.0, 1.0)

    return out_xyz


# ------------------------------
# 单张图像推理
# ------------------------------
def run_stageA_xyz_single(
    model,
    file_name,
    input_path,
    output_path,
    cfg_sample,
    cfg_train,
    dataset,
    cuda_no,
):
    file_stem, ext = os.path.splitext(file_name)
    ext = ext.lstrip(".")

    input_file_path = os.path.join(input_path, file_name)

    # 读 sRGB
    if ext == "npy":
        original_rgb = np.load(input_file_path).astype(np.float32)
    elif ext in ["JPG", "JPEG", "jpg", "jpeg", "png", "PNG"]:
        original_rgb = cv2.imread(input_file_path)
        if original_rgb is None:
            print(f"[WARN] 读取失败，跳过: {input_file_path}")
            return
        original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_BGR2RGB).astype(
            np.float32
        )
    else:
        print(f"[WARN] unsupported extension: {file_name}, skip.")
        return

    # 按训练时的方式归一化：/256
    original_rgb = original_rgb / 256.0

    context_size = cfg_sample["context_size"]
    rggb_max = dataset.get("rggb_max", 1.0)

    # 整图推理
    pred_xyz = convert_image_xyz_full(
        model,
        original_rgb,
        context_size=context_size,
        scale_back=rggb_max,
    )

    # 保存 .npy (H,W,3)，float32
    save_xyz_path = os.path.join(output_path, file_stem + "_pred_xyz.npy")
    np.save(save_xyz_path, pred_xyz)


# ------------------------------
# main
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage A (sRGB -> CIE XYZ) full-image conversion (no multiprocessing)."
    )
    parser.add_argument(
        "-g", "--gpu", type=int, default=0, help="GPU id used to convert the images."
    )
    parser.add_argument(
        "-f", "--folder", type=str, default="", help="Checkpoint folder (with best.pth + cfg.py)"
    )
    parser.add_argument(
        "-i", "--input_path", type=str, default="", help="Input folder with sRGB images"
    )
    parser.add_argument(
        "-o", "--output_path", type=str, default="", help="Output folder for XYZ .npy"
    )
    args = parser.parse_args()

    cuda_no = str(args.gpu)
    device = torch.device(f"cuda:{cuda_no}")
    torch.cuda.set_device(device)

    checkpoint_folder = args.folder
    input_path = args.input_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    # 从训练 workspace 里加载 cfg
    cfg_path = os.path.join(checkpoint_folder, "cfg.py")
    model_path = os.path.join(checkpoint_folder, "best.pth")

    dataset, cfg_sample, cfg_train = load_cfg(cfg_path)

    # 构建一个模型（单模型，无多进程）
    model = ReRAW(
        in_size=3,
        out_size=3,
        target_size=cfg_train["target_size"],
        hidden_size=cfg_train["hidden_size"],
        n_layers=cfg_train["depth"],
        gammas=cfg_train["gammas"],
    )
    state_dict = torch.load(model_path, map_location=f"cuda:{cuda_no}")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 拿输入文件列表
    input_files = [
        f
        for f in os.listdir(input_path)
        if os.path.isfile(os.path.join(input_path, f))
    ]
    input_files.sort()

    print(f"Found {len(input_files)} input images in {input_path}")

    # 单进程顺序跑，便于 debug，也避免 multiprocessing + CUDA 的各种坑
    for file_name in tqdm(input_files, desc="StageA sRGB->XYZ"):
        run_stageA_xyz_single(
            model,
            file_name,
            input_path,
            output_path,
            cfg_sample,
            cfg_train,
            dataset,
            args.gpu,
        )
