#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase1 - rawpy 版本 RAW -> XYZ 小测试脚本（原汁原味 XYZ 可视化版）

使用 rawpy 直接输出 XYZ（ColorSpace.XYZ），并把结果保存为 .npy，
同时基于同一个 XYZ(.npy) 生成一张“原始 XYZ 可视化”的 PNG：

- PNG 的 R/G/B 三个通道分别对应 X/Y/Z 分量
- 不做 XYZ->sRGB 矩阵变换
- 不做 gamma
- 不做任何为了人眼好看的处理，只做 [0,1] -> [0,255] 映射

当前默认处理两个目录：
- /data/umihebi0/users/z-hu/AgentRAW/FiveK_Canon/raw_test3
- /data/umihebi0/users/z-hu/AgentRAW/FiveK_Nikon/raw_test3
"""

import pathlib
import argparse

import numpy as np
import rawpy
import imageio.v2 as imageio


# 允许的 RAW 后缀
RAW_EXTS = {".dng", ".DNG", ".cr2", ".CR2", ".cr3", ".CR3",
            ".nef", ".NEF", ".arw", ".ARW"}


def list_raw_files(folder):
    folder = pathlib.Path(folder)
    files = [p for p in sorted(folder.iterdir()) if p.suffix in RAW_EXTS]
    return files


def ensure_dir(path):
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def xyz_to_png_raw(xyz_float: np.ndarray) -> np.ndarray:
    """
    将原始 XYZ(float32) 直接可视化为 PNG：
    - 不做矩阵变换
    - 不做 gamma
    - 不做色彩空间转换
    - 不做任何美化

    只是简单地把每个通道裁剪到 [0,1]，映射到 [0,255]，以便保存 PNG。

    xyz_float: (H, W, 3), float32, 原始 XYZ 数据
    返回: (H, W, 3) uint8，用于 imageio.imwrite
    """
    img = np.clip(xyz_float, 0.0, 1.0)       # 保护一下，避免异常值
    img = (img * 255.0).astype(np.uint8)    # 线性映射到 8bit
    return img


def raw_to_xyz_and_preview(raw_path, out_xyz_dir, out_png_dir):
    """
    用 rawpy 读取 RAW：
      1) 输出 ColorSpace.XYZ（线性） -> 保存为 .npy
      2) 从同一个 XYZ(.npy) 生成“原始 XYZ 可视化” PNG -> 保存为 .png
    """
    raw_path = pathlib.Path(raw_path)
    stem = raw_path.stem

    print(f"\n>>> 处理：{raw_path}")

    with rawpy.imread(str(raw_path)) as raw:

        # 1. XYZ 输出（线性、无 gamma、无 auto_bright）
        xyz_img = raw.postprocess(
            use_camera_wb=True,           # 用相机自带 WB（后面可以改成 GreyWorld）
            no_auto_bright=True,          # 禁用自动提亮
            output_bps=16,                # 16 bit 整数
            gamma=(1, 1),                 # 线性
            output_color=rawpy.ColorSpace.XYZ
        )

        # 转成 float32 归一化到 [0,1]
        xyz_float = xyz_img.astype(np.float32) / 65535.0

        # 保存 .npy
        out_xyz_path = ensure_dir(out_xyz_dir) / f"{stem}.npy"
        np.save(out_xyz_path, xyz_float)
        print(f"  [XYZ] 形状: {xyz_float.shape}, min={xyz_float.min():.6f}, max={xyz_float.max():.6f}")
        print(f"  [XYZ] 已保存: {out_xyz_path}")

        # 2. 基于 xyz_float 生成“原汁原味” XYZ PNG 预览（用于 eyeball / diff）
        xyz_preview_png = xyz_to_png_raw(xyz_float)

        out_png_path = ensure_dir(out_png_dir) / f"{stem}_xyz_raw_preview.png"
        imageio.imwrite(out_png_path, xyz_preview_png)
        print(f"  [PREVIEW] 原始 XYZ 可视化 PNG 已保存: {out_png_path}")


def process_dataset(name, raw_dir, base_out_dir):
    """
    处理一个数据集：
      - raw_dir：raw_test3 目录
      - base_out_dir：比如 ./FiveK_Canon
    会在 base_out_dir 下创建：
      - 同名.npy     (保存 .npy)
      - preview/ (保存 XYZ 可视化 PNG)
    """
    print(f"\n========== 数据集: {name} ==========")
    raw_files = list_raw_files(raw_dir)
    if not raw_files:
        print(f"  警告：目录中没有找到 RAW 文件: {raw_dir}")
        return

    print(f"  在 {raw_dir} 中找到 {len(raw_files)} 张 RAW，将全部处理。")

    out_xyz_dir = pathlib.Path(base_out_dir)
    out_png_dir = pathlib.Path(base_out_dir) / "preview"

    for rp in raw_files:
        raw_to_xyz_and_preview(rp, out_xyz_dir, out_png_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Phase1 rawpy 版 RAW->XYZ 小测试脚本（原汁原味 XYZ 可视化）"
    )

    parser.add_argument(
        "--canon_raw_dir",
        type=str,
        default="/data/umihebi0/users/z-hu/AgentRAW/FiveK_Canon/raw_train",
        help="FiveK_Canon RAW 测试目录 (raw_test)"
    )
    parser.add_argument(
        "--nikon_raw_dir",
        type=str,
        default="/data/umihebi0/users/z-hu/AgentRAW/FiveK_Nikon/raw_train",
        help="FiveK_Nikon RAW 测试目录 (raw_test)"
    )
    parser.add_argument(
        "--samsung_raw_dir",
        type=str,
        default="/data/umihebi0/users/z-hu/AgentRAW/samsung/raw_train",
        help="samsung RAW 测试目录 (raw_test)"
    )

    parser.add_argument(
        "--canon_base_out",
        type=str,
        default="/data/umihebi0/users/z-hu/AgentRAW/FiveK_Canon/xyz_train",
        help="Canon 输出基目录（会在下面创建 preview 等文件夹）"
    )
    parser.add_argument(
        "--nikon_base_out",
        type=str,
        default="/data/umihebi0/users/z-hu/AgentRAW/FiveK_Nikon/xyz_train",
        help="Nikon 输出基目录（会在下面创建 preview 等文件夹）"
    )

    parser.add_argument(
        "--samsung_base_out",
        type=str,
        default="/data/umihebi0/users/z-hu/AgentRAW/samsung/xyz_train",
        help="samsung 输出基目录（会在下面创建 preview 等文件夹）"
    )

    args = parser.parse_args()

    # process_dataset("FiveK_Canon",
    #                 raw_dir=args.canon_raw_dir,
    #                 base_out_dir=args.canon_base_out)

    # process_dataset("FiveK_Nikon",
    #                 raw_dir=args.nikon_raw_dir,
    #                 base_out_dir=args.nikon_base_out)

    process_dataset("samsung",
                    raw_dir=args.samsung_raw_dir,
                    base_out_dir=args.samsung_base_out)


if __name__ == "__main__":
    main()
