#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import rawpy
from rawpy._rawpy import LibRawDataError, LibRawFileUnsupportedError, LibRawIOError

RAW_EXTS = {".dng", ".DNG", ".cr2", ".CR2", ".cr3", ".CR3",
            ".nef", ".NEF", ".arw", ".ARW"}


def list_raw_files(folder):
    folder = Path(folder)
    return [p for p in sorted(folder.iterdir()) if p.suffix in RAW_EXTS]


def main():
    raw_dir = Path("/data/umihebi0/users/z-hu/AgentRAW/samsung/raw_train")
    files = list_raw_files(raw_dir)

    print(f"在 {raw_dir} 中找到 {len(files)} 张 RAW，开始体检…")

    ok_list = []
    bad_list = []

    for p in files:
        print(f"\n>>> 检查：{p.name}")
        try:
            with rawpy.imread(str(p)) as raw:
                # 不访问任何 metadata，直接 postprocess 一次
                _ = raw.postprocess()
            print("   [OK] imread + postprocess 成功")
            ok_list.append(p)
        except LibRawDataError as e:
            print(f"   [BAD] LibRawDataError: {repr(e)}")
            bad_list.append((p, e))
        except (LibRawFileUnsupportedError, LibRawIOError) as e:
            print(f"   [BAD] LibRaw*Error: {repr(e)}")
            bad_list.append((p, e))
        except Exception as e:
            print(f"   [BAD] 其他异常: {repr(e)}")
            bad_list.append((p, e))

    print("\n========== 体检总结 ==========")
    print(f"  成功: {len(ok_list)} 张")
    print(f"  失败: {len(bad_list)} 张")

    if bad_list:
        print("\n  失败列表：")
        for p, e in bad_list:
            print(f"    - {p.name}: {repr(e)}")


if __name__ == "__main__":
    main()
