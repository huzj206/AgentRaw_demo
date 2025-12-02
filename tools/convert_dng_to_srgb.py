#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import rawpy
import imageio.v2 as imageio


RAW_DIR_DEFAULT = "/data/umihebi0/users/z-hu/AgentRAW/FiveK_Nikon/raw"
RGB_DIR_DEFAULT = "/data/umihebi0/users/z-hu/AgentRAW/FiveK_Nikon/sRGB"


def convert_one_dng_to_srgb(dng_path, out_path):
    """
    读取单张 .dng，使用 rawpy 转成 sRGB，并保存为 8-bit PNG
    """
    with rawpy.imread(dng_path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,                      # 使用相机白平衡
            no_auto_bright=True,                     # 关闭自动增亮，避免过曝
            output_bps=8,                           # 16-bit 输出
            output_color=rawpy.ColorSpace.sRGB,      # sRGB 颜色空间
            gamma=(2.222, 4.5),                      # 标准 sRGB gamma
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
        )

    # 保存为 PNG（你也可以改成 .jpg）
    imageio.imwrite(out_path, rgb)


def main(
    raw_dir=RAW_DIR_DEFAULT,
    rgb_dir=RGB_DIR_DEFAULT,
):
    # 创建输出目录
    os.makedirs(rgb_dir, exist_ok=True)

    # 找到所有 .dng 文件
    dng_files = sorted(
        glob.glob(os.path.join(raw_dir, "*.dng"))
        + glob.glob(os.path.join(raw_dir, "*.DNG"))
    )

    if not dng_files:
        print(f"在目录中没有找到 .dng 文件：{raw_dir}")
        return

    print(f"发现 {len(dng_files)} 个 DNG 文件，开始转换...")
    print(f"输入目录: {raw_dir}")
    print(f"输出目录: {rgb_dir}")

    for idx, dng_path in enumerate(dng_files, 1):
        basename = os.path.splitext(os.path.basename(dng_path))[0]
        out_path = os.path.join(rgb_dir, f"{basename}.png")

        # 已经存在就跳过（可按需删掉）
        if os.path.exists(out_path):
            print(f"[{idx}/{len(dng_files)}] 跳过（已存在）：{out_path}")
            continue

        print(f"[{idx}/{len(dng_files)}] 处理：{dng_path}")
        try:
            convert_one_dng_to_srgb(dng_path, out_path)
        except Exception as e:
            print(f"  -> 转换失败：{e}")
        else:
            print(f"  -> 已保存：{out_path}")


if __name__ == "__main__":
    # 直接用默认路径；如果以后想换路径，可以改 main() 的参数
    main()
