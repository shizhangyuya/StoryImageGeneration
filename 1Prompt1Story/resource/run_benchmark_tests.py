#!/usr/bin/env python3
"""
测试benchmark结果的脚本
分别使用dreamsim、clip_image、clip_text三种方法测试result/benchmark目录下的所有结果
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import subprocess

def run_test(mode, folder_path, output_file=None):
    """
    运行单个测试模式
    :param mode: 测试模式 ('dreamsim', 'clip_image', 'clip_text')
    :param folder_path: 要测试的文件夹路径
    :param output_file: 输出文件路径（可选）
    :return: 测试结果字典
    """
    print(f"\n{'='*80}")
    print(f"开始测试: {mode}")
    print(f"测试目录: {folder_path}")
    print(f"{'='*80}\n")
    
    # 构建命令
    script_path = os.path.join(os.path.dirname(__file__), "test_benchmark.py")
    cmd = [
        sys.executable, script_path,
        "--mode", mode,
        "--folder", folder_path
    ]
    
    # 运行测试
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        output = result.stdout
        print(output)
        
        # 解析结果，提取total_avg_distance
        total_avg = None
        for line in output.split('\n'):
            if f"{mode}_total_avg_distance" in line:
                try:
                    total_avg = float(line.split(':')[-1].strip())
                except:
                    pass
        
        # 如果指定了输出文件，保存结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"测试模式: {mode}\n")
                f.write(f"测试目录: {folder_path}\n")
                f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")
                f.write(output)
            print(f"\n结果已保存至: {output_file}")
        
        return {
            'status': 'success',
            'output': output,
            'total_avg_distance': total_avg
        }
    except subprocess.CalledProcessError as e:
        error_msg = f"错误: {e}\n标准输出: {e.stdout}\n标准错误: {e.stderr}"
        print(error_msg)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"测试模式: {mode}\n")
                f.write(f"测试目录: {folder_path}\n")
                f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")
                f.write("测试失败！\n\n")
                f.write(error_msg)
        
        return {
            'status': 'failed',
            'output': error_msg,
            'total_avg_distance': None
        }


def main():
    parser = argparse.ArgumentParser(description="测试benchmark结果")
    parser.add_argument(
        '--benchmark_dir',
        type=str,
        default='./result/benchmark',
        help='benchmark结果目录路径（默认: ./result/benchmark）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./result/benchmark_test_results',
        help='测试结果输出目录（默认: ./result/benchmark_test_results）'
    )
    parser.add_argument(
        '--modes',
        type=str,
        nargs='+',
        choices=['dreamsim', 'clip_image', 'clip_text'],
        default=['dreamsim', 'clip_image', 'clip_text'],
        help='要运行的测试模式（默认: 全部三种）'
    )
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='跳过已存在的测试结果文件'
    )
    
    args = parser.parse_args()
    
    # 检查benchmark目录是否存在
    benchmark_dir = os.path.abspath(args.benchmark_dir)
    if not os.path.exists(benchmark_dir):
        print(f"错误: benchmark目录不存在: {benchmark_dir}")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*80}")
    print(f"Benchmark测试脚本")
    print(f"{'='*80}")
    print(f"Benchmark目录: {benchmark_dir}")
    print(f"输出目录: {output_dir}")
    print(f"测试模式: {', '.join(args.modes)}")
    print(f"时间戳: {timestamp}")
    print(f"{'='*80}\n")
    
    # 运行每种测试模式
    all_results = {}
    
    for mode in args.modes:
        output_file = os.path.join(output_dir, f"{mode}_results_{timestamp}.txt")
        
        # 如果跳过已存在的文件且文件存在，则跳过
        if args.skip_existing and os.path.exists(output_file):
            print(f"跳过已存在的文件: {output_file}")
            continue
        
        # 运行测试
        result = run_test(mode, benchmark_dir, output_file)
        all_results[mode] = {
            'output_file': output_file,
            'status': result['status'],
            'total_avg_distance': result.get('total_avg_distance')
        }
    
    # 生成汇总报告
    summary_file = os.path.join(output_dir, f"summary_{timestamp}.txt")
    json_file = os.path.join(output_dir, f"summary_{timestamp}.json")
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Benchmark测试汇总报告\n")
        f.write("="*80 + "\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Benchmark目录: {benchmark_dir}\n")
        f.write(f"测试模式: {', '.join(args.modes)}\n")
        f.write("\n" + "="*80 + "\n\n")
        
        for mode, info in all_results.items():
            f.write(f"{mode}:\n")
            f.write(f"  状态: {info['status']}\n")
            if info['total_avg_distance'] is not None:
                f.write(f"  平均距离/分数: {info['total_avg_distance']:.6f}\n")
            f.write(f"  结果文件: {info['output_file']}\n")
            f.write("\n")
    
    # 保存JSON格式的汇总
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'benchmark_dir': benchmark_dir,
            'modes': args.modes,
            'results': all_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("测试完成！")
    print(f"{'='*80}")
    print(f"汇总报告: {summary_file}")
    print(f"JSON报告: {json_file}")
    print("\n测试结果摘要:")
    for mode, info in all_results.items():
        status_icon = "✓" if info['status'] == 'success' else "✗"
        print(f"  {status_icon} {mode:15s}: ", end="")
        if info['total_avg_distance'] is not None:
            print(f"{info['total_avg_distance']:.6f}")
        else:
            print(f"{info['status']}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

