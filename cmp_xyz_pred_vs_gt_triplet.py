#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对比 GT XYZ(.npy) 和 StageA 预测的 XYZ(.npy)，生成 triplet diff 图：
  [ GT XYZ | Pred XYZ | diff(heatmap) ]

- 左图：GT XYZ 原汁原味可视化 (X/Y/Z -> R/G/B, 0~1 -> 0~255)
- 中图：Pred XYZ 同样风格可视化
- 右图：GT vs Pred 差异的热力图（JET colormap）
- 右图上标注 PSNR, SSIM

PSNR / SSIM 都在 XYZ 线性 [0,1] 空间上计算。

新增：
- 将每张图的 PSNR / SSIM / MSE 写入 CSV
- 在 CSV 末尾追加一行 AVERAGE，记录平均 PSNR / SSIM / MSE
"""

import argparse
import pathlib
import numpy as np
import imageio.v2 as imageio  # 没用到也保留，兼容旧环境
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import csv
import math


def ensure_dir(p):
    p = pathlib.Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_npy_files(folder):
    folder = pathlib.Path(folder)
    return [p for p in sorted(folder.iterdir()) if p.suffix == ".npy"]


def compute_psnr(mse, max_val=1.0):
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * np.log10((max_val ** 2) / mse)


def draw_label_cv(img, text, org,
                  font_scale=3.0,
                  thickness=5):
    """
    在 numpy 图像上用 OpenCV 画大号文字（带简单黑边）。
    img: H,W,3 (RGB uint8)
    org: (x, y) 左下角坐标
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 先画黑色粗描边
    cv2.putText(
        img,
        text,
        org,
        fontFace=font,
        fontScale=font_scale,
        color=(0, 0, 0),
        thickness=thickness + 2,
        lineType=cv2.LINE_AA,
    )

    # 再画白色文字
    cv2.putText(
        img,
        text,
        org,
        fontFace=font,
        fontScale=font_scale,
        color=(255, 255, 255),
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def xyz_to_png_raw(xyz_float: np.ndarray) -> np.ndarray:
    """
    原汁原味 XYZ 可视化：
    - 不做 XYZ->sRGB 矩阵变换
    - 不做 gamma
    - 不做色彩空间转换
    - 不做任何美化
    只是 clip 到 [0,1] 然后映射到 0~255，X/Y/Z -> R/G/B。
    """
    img = np.clip(xyz_float, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    return img


def find_pred_xyz_path(stem: str, pred_xyz_dir: pathlib.Path):
    """
    按顺序尝试多种命名，优先匹配你当前 StageA 的命名：
      1) <stem>_pred_xyz.npy   (convert_stageA_xyz.py 输出)
      2) <stem>_xyz.npy        (旧代码命名)
      3) <stem>.npy            (最原始兼容)
    """
    cand1 = pred_xyz_dir / f"{stem}_pred_xyz.npy"
    cand2 = pred_xyz_dir / f"{stem}_xyz.npy"
    cand3 = pred_xyz_dir / f"{stem}.npy"

    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    if cand3.exists():
        return cand3
    return None


def make_triplet_one_pair(gt_xyz_path, pred_xyz_dir, out_dir, dataset_name=None):
    """
    对单张图片做：
      - 加载 GT XYZ(.npy) 和 Pred XYZ(.npy)
      - 计算 PSNR & SSIM & MSE
      - 生成 [GT | Pred | diff heatmap] triplet
    返回:
      (image_id, psnr, ssim_val, mse) 或 None(如果找不到 pred)
    """
    gt_xyz_path = pathlib.Path(gt_xyz_path)
    stem = gt_xyz_path.stem

    pred_dir = pathlib.Path(pred_xyz_dir)
    pred_xyz_path = find_pred_xyz_path(stem, pred_dir)
    if pred_xyz_path is None:
        print(f"[跳过] 找不到对应的 Pred XYZ for {stem}: *_pred_xyz.npy / *_xyz.npy / .npy 都不存在")
        return None

    print(f"\n=== 处理配对 ===")
    print(f"GT   XYZ: {gt_xyz_path}")
    print(f"Pred XYZ: {pred_xyz_path}")

    # 1) 读 GT & Pred XYZ
    gt_xyz = np.load(gt_xyz_path).astype(np.float32)    # (H,W,3)
    pred_xyz = np.load(pred_xyz_path).astype(np.float32)

    if gt_xyz.shape != pred_xyz.shape:
        print(f"[警告] shape 不一致: GT={gt_xyz.shape}, Pred={pred_xyz.shape}，尝试 resize Pred 到 GT 尺寸")
        H, W = gt_xyz.shape[:2]
        # 逐通道 resize 更安全
        pred_resized = np.zeros_like(gt_xyz)
        for c in range(3):
            pred_resized[..., c] = np.array(
                Image.fromarray(pred_xyz[..., c]).resize((W, H), resample=Image.BILINEAR)
            )
        pred_xyz = pred_resized

    # 2) clip 到 [0,1]，用于 PSNR / SSIM / 可视化
    gt_clipped = np.clip(gt_xyz, 0.0, 1.0)
    pred_clipped = np.clip(pred_xyz, 0.0, 1.0)

    # 3) 计算 diff / MSE / PSNR / SSIM
    diff = pred_clipped - gt_clipped
    mse = float(np.mean(diff ** 2))
    psnr_val = compute_psnr(mse, max_val=1.0)
    ssim_val = float(ssim(gt_clipped, pred_clipped, channel_axis=-1, data_range=1.0))

    print(f"  PSNR = {psnr_val:.2f} dB, SSIM = {ssim_val:.4f}, MSE = {mse:.6e}")

    # 4) 生成 GT / Pred 的原汁原味 XYZ 预览
    gt_vis = xyz_to_png_raw(gt_clipped)
    pred_vis = xyz_to_png_raw(pred_clipped)

    # 5) 生成 diff heatmap
    diff_abs = np.abs(diff)             # (H,W,3)
    diff_gray = diff_abs.mean(axis=2)   # (H,W)

    max_d = diff_gray.max()
    if max_d < 1e-12:
        diff_norm = diff_gray  # 全零
    else:
        diff_norm = diff_gray / max_d

    diff_uint8 = (diff_norm * 255.0).clip(0, 255).astype(np.uint8)
    diff_color_bgr = cv2.applyColorMap(diff_uint8, cv2.COLORMAP_JET)
    diff_color_rgb = cv2.cvtColor(diff_color_bgr, cv2.COLOR_BGR2RGB)

    # 6) 在三张图上写标注
    H, W = gt_vis.shape[:2]
    gt_anno   = gt_vis.copy()
    pred_anno = pred_vis.copy()
    diff_anno = diff_color_rgb.copy()

    font_scale = 3.0
    thickness = 5

    draw_label_cv(gt_anno,   "GT XYZ",        (40, 120), font_scale, thickness)
    draw_label_cv(pred_anno, "Pred XYZ",      (40, 120), font_scale, thickness)

    diff_text_1 = "diff"
    diff_text_2 = f"PSNR={psnr_val:.2f} dB"
    diff_text_3 = f"SSIM={ssim_val:.4f}"

    draw_label_cv(diff_anno, diff_text_1, (40, 120),  font_scale, thickness)
    draw_label_cv(diff_anno, diff_text_2, (40, 250),  font_scale * 0.9, thickness - 1)
    draw_label_cv(diff_anno, diff_text_3, (40, 380),  font_scale * 0.9, thickness - 1)

    # 7) 横向拼接 triplet: [GT | Pred | diff]
    triplet_np = np.zeros((H, W * 3, 3), dtype=np.uint8)
    triplet_np[:, 0:W, :]       = gt_anno
    triplet_np[:, W:2*W, :]     = pred_anno
    triplet_np[:, 2*W:3*W, :]   = diff_anno

    triplet = Image.fromarray(triplet_np)

    out_dir = ensure_dir(out_dir)
    out_path = out_dir / f"{stem}_triplet_xyz_pred_vs_gt.png"
    triplet.save(out_path)
    print(f"  已保存 triplet: {out_path}")

    # 返回数值结果
    image_id = stem
    return image_id, psnr_val, ssim_val, mse, H, W


def process_dataset(name, gt_xyz_dir, pred_xyz_dir, out_dir=None, csv_path=None):
    print(f"\n========== 数据集: {name} ==========")
    gt_xyz_dir = pathlib.Path(gt_xyz_dir)
    pred_xyz_dir = pathlib.Path(pred_xyz_dir)

    if out_dir is None:
        out_dir = gt_xyz_dir.parent / "triplet_xyz_pred_vs_gt"
    out_dir = ensure_dir(out_dir)

    if csv_path is None:
        csv_path = out_dir / f"{name}_metrics.csv"
    else:
        csv_path = pathlib.Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    gt_files = list_npy_files(gt_xyz_dir)
    if not gt_files:
        print(f"  [警告] 没找到 GT XYZ .npy 文件: {gt_xyz_dir}")
        return

    print(f"  在 {gt_xyz_dir} 中找到 {len(gt_files)} 个 GT XYZ，将尝试生成 triplet 和 CSV。")

    rows = []
    for gp in gt_files:
        res = make_triplet_one_pair(gp, pred_xyz_dir, out_dir, dataset_name=name)
        if res is None:
            continue
        image_id, psnr_val, ssim_val, mse, H, W = res
        rows.append(
            {
                "dataset": name,
                "image": image_id,
                "height": H,
                "width": W,
                "psnr": psnr_val,
                "ssim": ssim_val,
                "mse": mse,
            }
        )

    if not rows:
        print("  [提示] 没有成功生成任何配对结果，CSV 将不会写入。")
        return

    # 计算平均值（忽略 inf PSNR 的样本）
    psnrs = [r["psnr"] for r in rows if math.isfinite(r["psnr"])]
    ssims = [r["ssim"] for r in rows]
    mses  = [r["mse"] for r in rows]

    mean_psnr = float(np.mean(psnrs)) if psnrs else float("inf")
    mean_ssim = float(np.mean(ssims))
    mean_mse  = float(np.mean(mses))

    avg_row = {
        "dataset": name,
        "image": "AVERAGE",
        "height": -1,
        "width": -1,
        "psnr": mean_psnr,
        "ssim": mean_ssim,
        "mse": mean_mse,
    }
    rows.append(avg_row)

    # 写 CSV
    fieldnames = ["dataset", "image", "height", "width", "psnr", "ssim", "mse"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"  已保存 CSV: {csv_path}")
    print(f"  平均 PSNR = {mean_psnr:.3f} dB, 平均 SSIM = {mean_ssim:.4f}, 平均 MSE = {mean_mse:.6e}")


def main():
    parser = argparse.ArgumentParser(
        description="生成 GT XYZ vs Pred XYZ 的 triplet diff 图 + CSV 指标"
    )

    # Canon
    parser.add_argument(
        "--canon_gt_xyz_dir",
        type=str,
        default="/mnt/data/FiveK_Canon/xyz_test3",
        help="Canon GT XYZ .npy 目录 (rawpy 输出)"
    )
    parser.add_argument(
        "--canon_pred_xyz_dir",
        type=str,
        default="./converted_stageA_test3",
        help="Canon Pred XYZ .npy 目录 (Stage A convert 输出)"
    )

    # Nikon
    parser.add_argument(
        "--nikon_gt_xyz_dir",
        type=str,
        default="/mnt/data/FiveK_Nikon/xyz_test3",
        help="Nikon GT XYZ .npy 目录"
    )
    parser.add_argument(
        "--nikon_pred_xyz_dir",
        type=str,
        default="./converted_stageA_test3_nikon",
        help="Nikon Pred XYZ .npy 目录"
    )

    parser.add_argument(
        "--canon_out_dir",
        type=str,
        default="./cmp_stageA_test3_canon",
        help="Canon triplet 输出目录"
    )
    parser.add_argument(
        "--nikon_out_dir",
        type=str,
        default="./cmp_stageA_test3_nikon",
        help="Nikon triplet 输出目录"
    )

    args = parser.parse_args()

    # Canon
    if pathlib.Path(args.canon_gt_xyz_dir).exists():
        process_dataset(
            "FiveK_Canon",
            gt_xyz_dir=args.canon_gt_xyz_dir,
            pred_xyz_dir=args.canon_pred_xyz_dir,
            out_dir=args.canon_out_dir,
            csv_path=pathlib.Path(args.canon_out_dir) / "FiveK_Canon_metrics.csv",
        )

    # Nikon
    if pathlib.Path(args.nikon_gt_xyz_dir).exists():
        process_dataset(
            "FiveK_Nikon",
            gt_xyz_dir=args.nikon_gt_xyz_dir,
            pred_xyz_dir=args.nikon_pred_xyz_dir,
            out_dir=args.nikon_out_dir,
            csv_path=pathlib.Path(args.nikon_out_dir) / "FiveK_Nikon_metrics.csv",
        )


if __name__ == "__main__":
    main()
