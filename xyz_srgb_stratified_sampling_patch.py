#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XYZ + sRGB patch-level stratified sampling (ReRAW-style)

完全参考 ReRAW 的 stratified_sampling.py:

- dataset["root"], ["rgb_path"], ["raw_path"]
- cfg_sample["output_folder_root"], ["sample_path"], ["target_path"], ["context_path"]
- 使用 sRGB 图像做亮度分层（/256，等宽 bin）
- 保存:
    sample_path: sRGB patch (模型输入)
    target_path: XYZ patch (模型输出 / GT)
    context_path: sRGB context patch

注意:
    raw_path 这里放的是 XYZ .npy (H, W, 3)，不是 Bayer RAW。
"""

import cv2
import os
import numpy as np
import multiprocessing
from tqdm import tqdm
import argparse

from resources.utils import large_crop  # 与 ReRAW 一样复用 large_crop


def sample_from_image(
    file_name,
    rgb_path,
    raw_path,
    rgb_extension,
    raw_extension,
    cfg_sample,
):
    """
    对单张 {sRGB, XYZ} 图做 ReRAW 风格的分层 patch 采样。

    file_name: 不含扩展名的 base name
    rgb_path : sRGB 图目录
    raw_path : XYZ .npy 目录（替代 ReRAW 的 RAW）
    """

    # 输出目录 (字段名和 ReRAW 完全一致)
    output_root = cfg_sample["output_folder_root"]
    sample_path = os.path.join(output_root, cfg_sample["sample_path"])   # sRGB patch (输入)
    target_path = os.path.join(output_root, cfg_sample["target_path"])   # XYZ patch (GT)
    context_path = os.path.join(output_root, cfg_sample["context_path"]) # sRGB context

    n_samples = cfg_sample["samples_per_channel"]

    # 读 sRGB (和 ReRAW 一样，作为亮度分层的基础)
    rgb_file = os.path.join(rgb_path, f"{file_name}.{rgb_extension}")
    if not os.path.exists(rgb_file):
        print(f"[WARN] RGB file not found, skip: {rgb_file}")
        return

    if rgb_extension in ["jpg", "JPG", "jpeg", "png"]:
        rgb_img = cv2.imread(rgb_file)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    elif rgb_extension == "npy":
        rgb_img = np.load(rgb_file).astype(np.float32)
    else:
        raise ValueError(f"Unsupported rgb_extension: {rgb_extension}")

    # 读 XYZ (raw 分支，但实际上是 XYZ .npy)
    raw_file = os.path.join(raw_path, f"{file_name}.{raw_extension}")
    if not os.path.exists(raw_file):
        print(f"[WARN] XYZ file not found, skip: {raw_file}")
        return

    if raw_extension == "npy":
        xyz_img = np.load(raw_file).astype(np.float32)
    else:
        raise ValueError(f"XYZ is expected to be .npy, got extension: {raw_extension}")

    # 尺寸对齐：裁掉多余部分，保证两者一样大
    h = min(rgb_img.shape[0], xyz_img.shape[0])
    w = min(rgb_img.shape[1], xyz_img.shape[1])
    rgb_img = rgb_img[:h, :w]
    xyz_img = xyz_img[:h, :w]

    img_h, img_w = rgb_img.shape[:2]

    # 和 ReRAW 一样的网格计算
    count_h = (img_h - cfg_sample["sample_size"][0]) // cfg_sample["delta"]
    count_w = (img_w - cfg_sample["sample_size"][1]) // cfg_sample["delta"]

    if count_h <= 2 or count_w <= 2:
        print(
            f"[WARN] Image {file_name} too small for sampling, skip. "
            f"img_h={img_h}, img_w={img_w}"
        )
        return

    bright = np.zeros((count_h, count_w, 3), dtype=np.float32)

    # 亮度计算：完全照抄 ReRAW，只是变量名 rgb_img
    for i in range(1, count_h - 1):
        for j in range(1, count_w - 1):
            y = cfg_sample["sample_size"][0] + i * cfg_sample["delta"]
            x = cfg_sample["sample_size"][1] + j * cfg_sample["delta"]
            for c in range(3):
                patch = rgb_img[
                    y - cfg_sample["sample_size"][0] // 2 : y + cfg_sample["sample_size"][0] // 2,
                    x - cfg_sample["sample_size"][1] // 2 : x + cfg_sample["sample_size"][1] // 2,
                    c,
                ]
                brightness = np.mean(patch) / 256.0
                bright[i, j, c] = brightness

    # 等宽 [0,1] bin，和 ReRAW 一致
    bins = np.linspace(0, 1, cfg_sample["n_bins"] + 1)
    counter = 0

    for c in range(3):
        idx = np.digitize(bright[:, :, c], bins)

        # 只用 1..n_bins-1（最亮一档没用到，保持和 ReRAW 一致）
        locations = [np.where(idx == i) for i in range(1, cfg_sample["n_bins"])]

        # ReRAW 代码里 prob/sample_indexes 基本没用，这里也照搬
        prob = [int(len(loc) > 0) for loc in locations]
        if np.sum(prob) > 0:
            prob = [p / np.sum(prob) for p in prob]
        sampled_indexes = [x for x in range(len(prob))]

        for i_sample in range(n_samples):
            if cfg_sample["type"] == "random":
                # 完全随机选网格点（ReRAW 的 random 分支）
                y = np.random.randint(1, count_h - 1)
                x = np.random.randint(1, count_w - 1)
            else:
                # stratified 分支：先随机选 bin，再在 bin 内选位置
                try:
                    sampled_index = np.random.randint(cfg_sample["n_bins"])
                    sampled_location = np.random.randint(
                        len(locations[sampled_index][0])
                    )
                    y = locations[sampled_index][0][sampled_location]
                    x = locations[sampled_index][1][sampled_location]
                except Exception:
                    # 如果 bin 没样本或者越界，就退回随机
                    y = np.random.randint(1, count_h - 1)
                    x = np.random.randint(1, count_w - 1)

            # 从网格坐标到像素坐标（完全照抄 ReRAW）
            y1 = np.clip(y * cfg_sample["delta"], 2, None)
            y2 = np.clip(y1 + cfg_sample["sample_size"][0], None, img_h - 2)

            x1 = np.clip(x * cfg_sample["delta"], 2, None)
            x2 = np.clip(x1 + cfg_sample["sample_size"][1], None, img_w - 2)

            # sRGB patch（上下左右各留 1 px，作为网络输入）
            selected_patch_rgb = rgb_img[y1 - 1 : y2 + 1, x1 - 1 : x2 + 1]
            # 同位置 XYZ patch（网络 GT）
            selected_patch_xyz = xyz_img[y1 - 1 : y2 + 1, x1 - 1 : x2 + 1]
            # context：使用 large_crop 基于 sRGB
            context = large_crop(
                rgb_img,
                (x1, y1),
                cfg_sample["context_size_scale"],
                cfg_sample["context_size"],
            )

            # === 保存 patch ===
            # sample_path: sRGB (输入)
            np.save(
                os.path.join(sample_path, f"{file_name}_{counter}.npy"),
                selected_patch_rgb,
            )
            # target_path: XYZ (GT)
            np.save(
                os.path.join(target_path, f"{file_name}_{counter}.npy"),
                selected_patch_xyz,
            )
            # context_path: sRGB context
            np.save(
                os.path.join(context_path, f"{file_name}_{counter}.npy"),
                context,
            )

            counter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform stratified sampling for sRGB->XYZ patches (ReRAW-style)."
    )
    parser.add_argument("-c", "--cfg", type=str, default="", help="Config path.")
    parser.add_argument("-n", "--n", type=int, default=2, help="number of workers")
    args = parser.parse_args()

    cfg_path = args.cfg
    N_WORKERS = int(args.n)

    variables = {}
    with open(cfg_path) as file:
        exec(file.read(), variables)

    dataset = variables["dataset"]
    cfg_sample = variables["cfg_sample"]

    rgb_path = os.path.join(dataset["root"], dataset["rgb_path"])
    raw_path = os.path.join(dataset["root"], dataset["raw_path"])

    rgb_files = [
        f for f in os.listdir(rgb_path)
        if os.path.isfile(os.path.join(rgb_path, f))
    ]
    raw_files = [
        f for f in os.listdir(raw_path)
        if os.path.isfile(os.path.join(raw_path, f))
    ]

    if len(rgb_files) == 0:
        raise RuntimeError(f"No RGB files found in {rgb_path}")
    if len(raw_files) == 0:
        raise RuntimeError(f"No RAW/XYZ files found in {raw_path}")

    rgb_extension = rgb_files[0].split(".")[-1]
    raw_extension = raw_files[0].split(".")[-1]

    # 输出目录创建
    output_root = cfg_sample["output_folder_root"]
    sample_path = os.path.join(output_root, cfg_sample["sample_path"])
    target_path = os.path.join(output_root, cfg_sample["target_path"])
    context_path = os.path.join(output_root, cfg_sample["context_path"])

    for folder in [sample_path, target_path, context_path]:
        os.makedirs(folder, exist_ok=True)

    print("=== StageA sRGB->XYZ Patch Stratified Sampling (ReRAW-style) ===")
    print(f"RGB root : {rgb_path} (*.{rgb_extension})")
    print(f"RAW/XYZ  : {raw_path} (*.{raw_extension})")
    print(f"Output   : {output_root}")
    print(f"sample_path : {cfg_sample['sample_path']} (sRGB patches, input)")
    print(f"target_path : {cfg_sample['target_path']} (XYZ patches, GT)")
    print(f"context_path: {cfg_sample['context_path']} (sRGB context)")
    print(f"sample_size : {cfg_sample['sample_size']}")
    print(f"delta       : {cfg_sample['delta']}")
    print(f"n_bins      : {cfg_sample['n_bins']}")
    print(f"type        : {cfg_sample['type']}")
    print(f"samples_per_channel: {cfg_sample['samples_per_channel']}")
    print("===============================================")

    # 和 ReRAW 一样：用 rgb 文件名作为基准
    file_names = [f.split(".")[0] for f in rgb_files]

    tasks = [
        [file_names[i], rgb_path, raw_path, rgb_extension, raw_extension, cfg_sample]
        for i in range(len(file_names))
    ]

    pool = multiprocessing.Pool(processes=N_WORKERS)
    with tqdm(total=len(tasks)) as pbar:

        def update(_):
            pbar.update()

        for t in tasks:
            pool.apply_async(sample_from_image, args=tuple(t), callback=update)

        pool.close()
        pool.join()

