#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase1 - rawpy 版本 RAW -> XYZ 小测试脚本（原汁原味 XYZ 可视化版, 支持多进程）

使用 rawpy 直接输出 XYZ（ColorSpace.XYZ），并把结果保存为 .npy，
同时基于同一个 XYZ(.npy) 生成一张“原始 XYZ 可视化”的 PNG：

- PNG 的 R/G/B 三个通道分别对应 X/Y/Z 分量
- 不做 XYZ->sRGB 矩阵变换
- 不做 gamma
- 不做任何为了人眼好看的处理，只做 [0,1] -> [0,255] 映射

支持多进程并行处理 RAW 文件，加快整个 RAW->XYZ 的转换速度。
"""

import pathlib
import argparse
import multiprocessing

import numpy as np
import rawpy
import imageio.v2 as imageio
from tqdm import tqdm

# 允许的 RAW 后缀
RAW_EXTS = {
    ".dng", ".DNG",
    ".cr2", ".CR2",
    ".cr3", ".CR3",
    ".nef", ".NEF",
    ".arw", ".ARW"
}


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
    img = np.clip(xyz_float, 0.0, 1.0)   # 保护一下，避免异常值
    img = (img * 255.0).astype(np.uint8)  # 线性映射到 8bit
    return img


def raw_to_xyz_and_preview_one(args):
    """
    单个 RAW 文件的处理函数，用于多进程：
      args: (raw_path_str, out_xyz_dir_str, out_png_dir_str)
    """
    raw_path_str, out_xyz_dir_str, out_png_dir_str = args

    raw_path = pathlib.Path(raw_path_str)
    out_xyz_dir = pathlib.Path(out_xyz_dir_str)
    out_png_dir = pathlib.Path(out_png_dir_str)

    stem = raw_path.stem

    try:
        with rawpy.imread(str(raw_path)) as raw:
            # 1. XYZ 输出（线性、无 gamma、无 auto_bright）
            xyz_img = raw.postprocess(
                use_camera_wb=True,           # 用相机自带 WB（后面可以改成 GreyWorld）
                no_auto_bright=True,          # 禁用自动提亮
                output_bps=16,                # 16 bit 整数
                gamma=(1, 1),                 # 线性
                output_color=rawpy.ColorSpace.XYZ
                # 不再设置 user_flip，使用 rawpy 默认的 orientation 处理
            )

        # 转成 float32 归一化到 [0,1]
        xyz_float = xyz_img.astype(np.float32) / 65535.0

        # 保存 .npy
        out_xyz_path = out_xyz_dir / f"{stem}.npy"
        np.save(out_xyz_path, xyz_float)

        # 基于 xyz_float 生成“原汁原味” XYZ PNG 预览
        xyz_preview_png = xyz_to_png_raw(xyz_float)
        out_png_path = out_png_dir / f"{stem}_xyz_raw_preview.png"
        imageio.imwrite(out_png_path, xyz_preview_png)

        # 返回一点简单信息（可选）
        return (str(raw_path), xyz_float.shape, float(xyz_float.min()), float(xyz_float.max()))
    except Exception as e:
        # 避免多进程中异常直接吞掉，打印出来
        print(f"[ERROR] 处理 {raw_path} 时出错: {e}")
        return None


def process_dataset(name, raw_dir, base_out_dir, num_workers=4):
    """
    处理一个数据集：
      - raw_dir：RAW 目录（raw_train / raw_test 等）
      - base_out_dir：比如 ./FiveK_Canon/xyz_train
    会在 base_out_dir 下创建：
      - *.npy     (保存 XYZ .npy)
      - preview/  (保存 XYZ 可视化 PNG)

    支持多进程并行处理。
    """
    print(f"\n========== 数据集: {name} ==========")
    raw_files = list_raw_files(raw_dir)
    if not raw_files:
        print(f"  警告：目录中没有找到 RAW 文件: {raw_dir}")
        return

    print(f"  在 {raw_dir} 中找到 {len(raw_files)} 张 RAW，将全部处理。")
    print(f"  输出目录: {base_out_dir} (含 preview 子目录)")
    print(f"  使用进程数: {num_workers}")

    out_xyz_dir = ensure_dir(base_out_dir)
    out_png_dir = ensure_dir(pathlib.Path(base_out_dir) / "preview")

    # 构建任务列表：每个元素都是 (raw_path_str, out_xyz_dir_str, out_png_dir_str)
    tasks = [
        (str(rp), str(out_xyz_dir), str(out_png_dir))
        for rp in raw_files
    ]

    if num_workers <= 1:
        # 单进程 fallback，方便 debug
        for t in tqdm(tasks, desc=f"{name} (single process)"):
            raw_to_xyz_and_preview_one(t)
    else:
        # 多进程
        with multiprocessing.Pool(processes=num_workers) as pool:
            for _ in tqdm(
                pool.imap_unordered(raw_to_xyz_and_preview_one, tasks),
                total=len(tasks),
                desc=f"{name} (multi-process)"
            ):
                pass  # 只是为了驱动 tqdm 进度条


def main():
    parser = argparse.ArgumentParser(
        description="Phase1 rawpy 版 RAW->XYZ 小测试脚本（原汁原味 XYZ 可视化，支持多进程）"
    )

    parser.add_argument(
        "--canon_raw_dir",
        type=str,
        default="/data/umihebi0/users/z-hu/AgentRAW/FiveK_Canon/raw_train",
        help="FiveK_Canon RAW 目录 (raw_train/raw_test)"
    )
    parser.add_argument(
        "--nikon_raw_dir",
        type=str,
        default="/data/umihebi0/users/z-hu/AgentRAW/FiveK_Nikon/raw_train",
        help="FiveK_Nikon RAW 目录 (raw_train/raw_test)"
    )
    parser.add_argument(
        "--samsung_raw_dir",
        type=str,
        default="/data/umihebi0/users/z-hu/AgentRAW/samsung/raw_train",
        help="Samsung RAW 目录 (raw_train/raw_test)"
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
        help="Samsung 输出基目录（会在下面创建 preview 等文件夹）"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="多进程 worker 数量（每个 worker 处理一张 RAW）"
    )

    args = parser.parse_args()

    # 注意：在某些平台（特别是 Windows）需要放在 if __name__ == '__main__' 保护下
    # 你当前环境是 Linux + HPC，multiprocessing 默认 fork，通常没问题。

    if args.canon_raw_dir:
        process_dataset(
            "FiveK_Canon",
            raw_dir=args.canon_raw_dir,
            base_out_dir=args.canon_base_out,
            num_workers=args.num_workers,
        )

    if args.nikon_raw_dir:
        process_dataset(
            "FiveK_Nikon",
            raw_dir=args.nikon_raw_dir,
            base_out_dir=args.nikon_base_out,
            num_workers=args.num_workers,
        )

    # 如果还想处理 samsung，就解除注释：
    # if args.samsung_raw_dir:
    #     process_dataset(
    #         "Samsung",
    #         raw_dir=args.samsung_raw_dir,
    #         base_out_dir=args.samsung_base_out,
    #         num_workers=args.num_workers,
    #     )


if __name__ == "__main__":
    main()
