#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import shutil
from pathlib import Path

# === 路径与参数配置 ===
SRGB_DIR = Path("/data/umihebi0/users/z-hu/AgentRAW/samung/sRGB")
RAW_DIR = Path("/data/umihebi0/users/z-hu/AgentRAW/samung/raw")

SRGB_TRAIN_DIR = Path("/data/umihebi0/users/z-hu/AgentRAW/samung/sRGB_train")
SRGB_TEST_DIR = Path("/data/umihebi0/users/z-hu/AgentRAW/samung/sRGB_test")
RAW_TRAIN_DIR = Path("/data/umihebi0/users/z-hu/AgentRAW/samung/raw_train")
RAW_TEST_DIR = Path("/data/umihebi0/users/z-hu/AgentRAW/samung/raw_test")

# 从 test 中再抽 3 对
SRGB_TEST3_DIR = Path("/data/umihebi0/users/z-hu/AgentRAW/samung/sRGB_test3")
RAW_TEST3_DIR = Path("/data/umihebi0/users/z-hu/AgentRAW/samung/raw_test3")

TRAIN_RATIO = 0.8        # 8:2 划分
RANDOM_SEED = 42         # 固定随机种子，保证可复现
MOVE_FILES = False        # True=移动文件, False=复制文件 (从原始 sRGB/raw -> train/test)
TEST3_SAMPLE_NUM = 3     # 从 test 中抽取的配对数量


# ========== 通用工具函数 ==========

def list_files_by_stem(folder: Path):
    """
    返回: {stem: [Path1, Path2, ...]}（一般每个 stem 只对应一个文件）
    比如: image_0001.dng -> stem=image_0001
    """
    mapping = {}
    for p in folder.iterdir():
        if not p.is_file():
            continue
        stem = p.stem
        mapping.setdefault(stem, []).append(p)
    return mapping


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def move_or_copy(src: Path, dst: Path, move: bool = True):
    ensure_dir(dst.parent)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


# ========== 第一步：划分 train/test ==========

def split_train_test():
    print("==== 第一步：按 8:2 划分 train/test ====")
    random.seed(RANDOM_SEED)

    if not SRGB_DIR.exists():
        raise FileNotFoundError(f"sRGB 目录不存在: {SRGB_DIR}")
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"RAW 目录不存在: {RAW_DIR}")

    print(f"扫描 sRGB 目录: {SRGB_DIR}")
    srgb_map = list_files_by_stem(SRGB_DIR)
    print(f"找到 sRGB 文件数: {sum(len(v) for v in srgb_map.values())}")

    print(f"扫描 RAW 目录: {RAW_DIR}")
    raw_map = list_files_by_stem(RAW_DIR)
    print(f"找到 RAW 文件数: {sum(len(v) for v in raw_map.values())}")

    srgb_stems = set(srgb_map.keys())
    raw_stems = set(raw_map.keys())
    common_stems = sorted(srgb_stems & raw_stems)

    if not common_stems:
        raise RuntimeError("在 sRGB 和 RAW 两个目录中没有找到任何公共的文件名（stem）。")

    only_srgb = srgb_stems - raw_stems
    only_raw = raw_stems - srgb_stems
    if only_srgb:
        print(f"警告: 有 {len(only_srgb)} 个文件只出现在 sRGB 目录中，将被忽略。")
    if only_raw:
        print(f"警告: 有 {len(only_raw)} 个文件只出现在 RAW 目录中，将被忽略。")

    print(f"可配对的样本数量: {len(common_stems)}")

    # 打乱并按比例切分
    random.shuffle(common_stems)
    n_total = len(common_stems)
    n_train = int(n_total * TRAIN_RATIO)
    if n_train == 0 or n_train == n_total:
        # 极端情况，至少留 1 个给 test 或 train
        n_train = max(1, min(n_total - 1, n_train))

    train_stems = common_stems[:n_train]
    test_stems = common_stems[n_train:]

    print(f"训练集: {len(train_stems)}  | 测试集: {len(test_stems)}")
    print("开始创建输出目录...")
    for d in [SRGB_TRAIN_DIR, SRGB_TEST_DIR, RAW_TRAIN_DIR, RAW_TEST_DIR]:
        ensure_dir(d)

    action = "移动" if MOVE_FILES else "复制"
    print(f"开始{action}文件...")

    def process_split(stems, srgb_out_dir, raw_out_dir, split_name):
        for i, stem in enumerate(stems, 1):
            srgb_files = srgb_map[stem]
            raw_files = raw_map[stem]

            # 一般情况下，每个 stem 应该只对应一个文件；如果有多个，就全部处理
            for s in srgb_files:
                dst = srgb_out_dir / s.name
                move_or_copy(s, dst, MOVE_FILES)

            for r in raw_files:
                dst = raw_out_dir / r.name
                move_or_copy(r, dst, MOVE_FILES)

            if i % 100 == 0 or i == len(stems):
                print(f"[{split_name}] 已{action} {i}/{len(stems)} 对样本")

    process_split(train_stems, SRGB_TRAIN_DIR, RAW_TRAIN_DIR, "train")
    process_split(test_stems, SRGB_TEST_DIR, RAW_TEST_DIR, "test")

    print("train/test 划分完成！")
    print(f"训练集 sRGB 目录: {SRGB_TRAIN_DIR}")
    print(f"训练集 RAW 目录 : {RAW_TRAIN_DIR}")
    print(f"测试集 sRGB 目录: {SRGB_TEST_DIR}")
    print(f"测试集 RAW 目录 : {RAW_TEST_DIR}")
    print(f"操作方式: {'移动(move)' if MOVE_FILES else '复制(copy)'}")


# ========== 第二步：从 test 中随机抽取 3 对到 *_test3 ==========

def sample_test3():
    print("\n==== 第二步：从 *_test 中随机抽取 3 对到 *_test3 ====")
    random.seed(RANDOM_SEED)  # 保持可复现，也可以单独设置另一个 seed

    if not RAW_TEST_DIR.exists() or not SRGB_TEST_DIR.exists():
        raise FileNotFoundError("raw_test 或 sRGB_test 目录不存在，请先完成 train/test 划分。")

    raw_map = list_files_by_stem(RAW_TEST_DIR)
    srgb_map = list_files_by_stem(SRGB_TEST_DIR)

    common_stems = sorted(set(raw_map.keys()) & set(srgb_map.keys()))
    if len(common_stems) < TEST3_SAMPLE_NUM:
        raise RuntimeError(
            f"测试集中可配对的数量只有 {len(common_stems)}，不足 {TEST3_SAMPLE_NUM} 对，无法抽样。"
        )

    selected = random.sample(common_stems, TEST3_SAMPLE_NUM)

    print(f"从测试集中随机抽取的 {TEST3_SAMPLE_NUM} 个样本 stem：")
    for s in selected:
        print(" -", s)

    # 创建输出目录
    ensure_dir(RAW_TEST3_DIR)
    ensure_dir(SRGB_TEST3_DIR)

    # 这里对 test3 采用复制，不从 test 中删除
    print("开始从 *_test 复制文件到 *_test3 ...")
    for stem in selected:
        for r in raw_map[stem]:
            dst = RAW_TEST3_DIR / r.name
            shutil.copy2(r, dst)

        for s in srgb_map[stem]:
            dst = SRGB_TEST3_DIR / s.name
            shutil.copy2(s, dst)

    print("test3 子集抽样完成！")
    print(f"RAW test3 目录 : {RAW_TEST3_DIR}")
    print(f"sRGB test3 目录: {SRGB_TEST3_DIR}")


# ========== 主函数 ==========

def main():
    split_train_test()
    sample_test3()


if __name__ == "__main__":
    main()
