#!/usr/bin/env python3
"""
测试脚本：用于测试 benchmark_atten+concat_0-40step_1.08token_down+up_test 目录下
animals 前缀的文件夹的图像相似度

使用方法:
1. 使用默认设置（dreamsim模式，测试animals前缀的文件夹）:
   python resource/test_benchmark_animals.py

2. 使用CLIP图像相似度模式:
   python resource/test_benchmark_animals.py --mode clip_image

3. 使用CLIP文本-图像相似度模式:
   python resource/test_benchmark_animals.py --mode clip_text

4. 指定自定义文件夹路径:
   python resource/test_benchmark_animals.py --folder result/your_benchmark_folder

5. 移除背景后计算相似度:
   python resource/test_benchmark_animals.py --remove_background

6. 测试其他前缀（如fairy_tales）:
   python resource/test_benchmark_animals.py --prefix fairy_tales

示例（从项目根目录运行）:
   python resource/test_benchmark_animals.py --mode dreamsim --folder result/benchmark_atten+concat_0-40step_1.08token_down+up_test --prefix animals
"""

import sys
import os
import math
from pathlib import Path

# 添加当前目录到路径，以便导入 test_benchmark 中的函数
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 直接导入 test_benchmark（它们在同一个目录下）
from test_benchmark import (
    calculate_avg_distance,
    calculate_avg_clip_text_score
)
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="测试 benchmark_atten+concat_0-40step_1.08token_down+up_test 目录下 animals 前缀的文件夹"
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['dreamsim', 'clip_image', 'clip_text'],
        default='dreamsim',
        help='测试模式: "dreamsim", "clip_image" 或 "clip_text" (默认: dreamsim)'
    )
    parser.add_argument(
        '--folder',
        type=str,
        default='result/benchmark_atten+concat_0-40step_1.08token_down+up_test',
        help='测试文件夹路径 (默认: result/benchmark_atten+concat_0-40step_1.08token_down+up_test)'
    )
    parser.add_argument(
        '--remove_background',
        action='store_true',
        help='移除背景后计算相似度'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='animals',
        help='文件夹前缀过滤 (默认: animals)'
    )

    args = parser.parse_args()

    # 确保使用绝对路径
    if not os.path.isabs(args.folder):
        # 尝试从脚本所在目录的父目录（项目根目录）查找
        script_dir = Path(__file__).parent.parent
        folder_path = script_dir / args.folder
        # 如果不存在，尝试当前工作目录
        if not folder_path.exists():
            folder_path = Path(args.folder).resolve()
    else:
        folder_path = Path(args.folder)

    if not folder_path.exists():
        print(f"错误: 文件夹不存在: {folder_path}")
        print(f"提示: 请确保文件夹路径正确，或者使用绝对路径")
        return

    print(f"测试文件夹: {folder_path}")
    print(f"测试模式: {args.mode}")
    print(f"前缀过滤: {args.prefix}")
    print(f"移除背景: {args.remove_background}")
    print("=" * 80)

    # 执行测试
    if args.mode == 'clip_text':
        avg_distances = calculate_avg_clip_text_score(
            str(folder_path),
            single_folder=False,
            prefix=args.prefix
        )
    else:
        avg_distances = calculate_avg_distance(
            str(folder_path),
            mode=args.mode,
            remove_bg=args.remove_background,
            single_folder=False,
            prefix=args.prefix
        )

    # 输出结果
    print("\n" + "=" * 80)
    print("测试结果:")
    print("=" * 80)

    total_avg_distance = 0
    valid_count = 0

    for subfolder, avg_distance in sorted(avg_distances.items()):
        # 检查是否是有效数值（不是NaN）
        is_valid = isinstance(avg_distance, (int, float)) and not math.isnan(avg_distance)
        if is_valid:
            total_avg_distance += avg_distance
            valid_count += 1
        # 格式化输出（NaN会显示为nan）
        if is_valid:
            print(f"{subfolder}: {avg_distance:.5f}")
        else:
            print(f"{subfolder}: {avg_distance}")

    if valid_count > 0:
        total_avg_distance /= valid_count
        print("=" * 80)
        print(f"{args.mode}_total_avg_distance: {total_avg_distance:.5f}")
        print(f"测试的文件夹数量: {valid_count}")
    else:
        print("警告: 没有有效的测试结果")

    print("=" * 80)


if __name__ == "__main__":
    main()

