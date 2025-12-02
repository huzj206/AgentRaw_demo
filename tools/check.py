#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import rawpy
from rawpy._rawpy import LibRawDataError, LibRawFileUnsupportedError, LibRawIOError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_path",
        type=str,
        default="/data/umihebi0/users/z-hu/AgentRAW/samsung/raw_train/20251115_130052.dng",
        help="要测试的 RAW 文件路径",
    )
    args = parser.parse_args()

    p = Path(args.raw_path)
    print(p, "exists:", p.exists())
    if not p.exists():
        return

    # 1. 先试试 imread
    try:
        raw = rawpy.imread(str(p))
        print("imread ok:", raw)
    except Exception as e:
        print("[ERROR] imread 失败：", repr(e))
        return

    # 打印一点基本信息
    try:
        print("  sizes:", raw.sizes)
        print("  camera:", raw.camera_whitebalance, raw.color_desc)
    except Exception as e:
        print("  [WARN] 打印 metadata 时出错：", repr(e))

    # 2. 试着访问 raw_image（这一步也会触发 unpack）
    try:
        _ = raw.raw_image
        print("raw.raw_image 访问成功")
    except Exception as e:
        print("[ERROR] 访问 raw.raw_image 失败：", repr(e))

    # 3. 默认 postprocess
    print("\n[TEST] raw.postprocess() 默认参数")
    try:
        rgb = raw.postprocess()
        print("  postprocess 默认参数 OK，shape:", rgb.shape)
    except LibRawDataError as e:
        print("  [LibRawDataError] 默认 postprocess 失败：", repr(e))
    except (LibRawFileUnsupportedError, LibRawIOError) as e:
        print("  [LibRaw*Error] 默认 postprocess 失败：", repr(e))
    except Exception as e:
        print("  [OtherError] 默认 postprocess 失败：", repr(e))

    # 4. 指定 output_color = XYZ 再试一次（模拟你主脚本的情况）
    print("\n[TEST] raw.postprocess(output_color=XYZ)")
    try:
        xyz_img = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=16,
            gamma=(1, 1),
            output_color=rawpy.ColorSpace.XYZ,
        )
        print("  postprocess XYZ OK，shape:", xyz_img.shape)
    except LibRawDataError as e:
        print("  [LibRawDataError] XYZ postprocess 失败：", repr(e))
    except (LibRawFileUnsupportedError, LibRawIOError) as e:
        print("  [LibRaw*Error] XYZ postprocess 失败：", repr(e))
    except Exception as e:
        print("  [OtherError] XYZ postprocess 失败：", repr(e))

    raw.close()


if __name__ == "__main__":
    main()
