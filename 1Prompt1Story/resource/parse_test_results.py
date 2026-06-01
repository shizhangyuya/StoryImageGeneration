#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
解析测试结果文件，生成对比表格
"""

import os
import re
from collections import defaultdict
from pathlib import Path

def parse_result_file(file_path):
    """解析单个测试结果文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取测试目录
    dir_match = re.search(r'测试目录:\s*(.+?)\n', content)
    test_dir = dir_match.group(1).strip() if dir_match else "unknown"
    
    # 提取测试模式
    mode_match = re.search(r'测试结果\s*-\s*(\w+)', content)
    mode = mode_match.group(1).strip() if mode_match else "unknown"
    
    # 提取每个类别的平均距离
    category_results = {}
    category_pattern = r'类别:\s*(\w+)\s*\n\s*平均距离:\s*([\d.]+)'
    for match in re.finditer(category_pattern, content):
        category = match.group(1)
        avg_distance = float(match.group(2))
        category_results[category] = avg_distance
    
    # 提取总体平均距离
    total_match = re.search(r'(\w+)_total_avg_distance:\s*([\d.]+)', content)
    total_avg = None
    if total_match:
        total_avg = float(total_match.group(2))
    
    return {
        'test_dir': test_dir,
        'mode': mode,
        'categories': category_results,
        'total_avg': total_avg
    }

def main():
    test_result_dir = Path(__file__).parent / 'test_result'
    
    # 查找所有结果文件
    result_files = {
        'clip_image': [],
        'clip_text': [],
        'dreamsim': []
    }
    
    for file_path in test_result_dir.glob('*.txt'):
        if 'clip_image' in file_path.name:
            result_files['clip_image'].append(file_path)
        elif 'clip_text' in file_path.name:
            result_files['clip_text'].append(file_path)
        elif 'dreamsim' in file_path.name:
            result_files['dreamsim'].append(file_path)
    
    # 解析所有文件
    all_results = defaultdict(lambda: defaultdict(dict))
    
    for mode, files in result_files.items():
        for file_path in files:
            result = parse_result_file(file_path)
            test_dir = result['test_dir']
            
            # 识别方法名称（只保留两个目标方法）
            if 'benchmark_consistory+id_replacement_0-50step_down+mid+upblock_no-ipca_no-svr_diff_prompt_+new_subject_test2' in test_dir:
                method_name = 'consistory+id_replacement'
            elif 'benchmark_1p1s' in test_dir:
                method_name = '1p1s'
            else:
                # 跳过其他方法
                continue
            
            all_results[mode][method_name] = result['categories']
    
    # 只保留两个目标方法
    target_methods = ['1p1s', 'consistory+id_replacement']
    for mode in all_results:
        all_results[mode] = {k: v for k, v in all_results[mode].items() if k in target_methods}
    
    # 生成表格
    print("\n" + "="*80)
    print("测试结果对比表格")
    print("="*80 + "\n")
    print("方法1: 1p1s (benchmark_1p1s)")
    print("方法2: consistory+id_replacement (benchmark_consistory+id_replacement_0-50step_down+mid+upblock_no-ipca_no-svr_diff_prompt_+new_subject_test2)")
    print("\n")
    
    # 获取所有类别
    all_categories = set()
    for mode_results in all_results.values():
        for method_results in mode_results.values():
            all_categories.update(method_results.keys())
    all_categories = sorted(list(all_categories))
    
    # 获取所有方法（应该只有两个）
    all_methods = ['1p1s', 'consistory+id_replacement']
    
    # 为每个指标生成表格
    for mode in ['clip_image', 'clip_text', 'dreamsim']:
        print(f"\n{'='*80}")
        print(f"指标: {mode.upper()}")
        print(f"{'='*80}\n")
        
        # 表头
        header = f"{'类别':<15}"
        header += f"{'1p1s':<20}"
        header += f"{'consistory+id_replacement':<30}"
        print(header)
        print("-" * 65)
        
        # 数据行
        for category in all_categories:
            row = f"{category:<15}"
            # 1p1s
            if '1p1s' in all_results[mode] and category in all_results[mode]['1p1s']:
                value = all_results[mode]['1p1s'][category]
                row += f"{value:<20.6f}"
            else:
                row += f"{'N/A':<20}"
            # consistory+id_replacement
            if 'consistory+id_replacement' in all_results[mode] and category in all_results[mode]['consistory+id_replacement']:
                value = all_results[mode]['consistory+id_replacement'][category]
                row += f"{value:<30.6f}"
            else:
                row += f"{'N/A':<30}"
            print(row)
        
        # 总体平均
        print("-" * 65)
        row = f"{'总体平均':<15}"
        # 1p1s
        if '1p1s' in all_results[mode]:
            values = list(all_results[mode]['1p1s'].values())
            if values:
                total_avg = sum(values) / len(values)
                row += f"{total_avg:<20.6f}"
            else:
                row += f"{'N/A':<20}"
        else:
            row += f"{'N/A':<20}"
        # consistory+id_replacement
        if 'consistory+id_replacement' in all_results[mode]:
            values = list(all_results[mode]['consistory+id_replacement'].values())
            if values:
                total_avg = sum(values) / len(values)
                row += f"{total_avg:<30.6f}"
            else:
                row += f"{'N/A':<30}"
        else:
            row += f"{'N/A':<30}"
        print(row)
    
    # 生成CSV格式
    print("\n" + "="*80)
    print("CSV格式（便于导入Excel）")
    print("="*80 + "\n")
    
    for mode in ['clip_image', 'clip_text', 'dreamsim']:
        print(f"\n{mode.upper()}")
        print("类别," + ",".join(all_methods))
        for category in all_categories:
            row = category
            for method in all_methods:
                if method in all_results[mode] and category in all_results[mode][method]:
                    value = all_results[mode][method][category]
                    row += f",{value:.6f}"
                else:
                    row += ",N/A"
            print(row)
        
        # 总体平均
        row = "总体平均"
        for method in all_methods:
            if method in all_results[mode]:
                values = list(all_results[mode][method].values())
                if values:
                    total_avg = sum(values) / len(values)
                    row += f",{total_avg:.6f}"
                else:
                    row += ",N/A"
            else:
                row += ",N/A"
        print(row)

if __name__ == "__main__":
    main()

