#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
交互式故事配置生成器
根据故事描述生成 YAML 配置文件
"""

import os
import sys
import yaml
import re

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 直接导入函数（避免循环导入）
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# 导入生成函数
import importlib.util
spec = importlib.util.spec_from_file_location(
    "generate_story_yaml",
    os.path.join(os.path.dirname(__file__), "generate_story_yaml.py")
)
generate_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate_module)

generate_story_config_with_chatgpt = generate_module.generate_story_config_with_chatgpt
save_to_yaml = generate_module.save_to_yaml


def main_interactive():
    """交互式主函数"""
    print("\n" + "="*60)
    print("故事配置生成器 - 交互式模式")
    print("="*60 + "\n")
    
    # 检查依赖
    if not OPENAI_AVAILABLE:
        print("❌ 错误: openai 库未安装")
        print("请运行: pip install openai")
        return
    
    # 输入参数
    print("请输入以下信息:\n")
    
    story_description = input("故事描述 (例如: 生成火柴人的故事): ").strip()
    if not story_description:
        print("❌ 错误: 故事描述不能为空")
        return
    
    try:
        sequence_length = int(input("序列长度/图像数量 (默认 5): ").strip() or "5")
        if sequence_length < 1 or sequence_length > 50:
            print("⚠️  警告: 序列长度建议在 1-50 之间，将使用 5")
            sequence_length = 5
    except ValueError:
        print("⚠️  警告: 输入无效，使用默认值 5")
        sequence_length = 5
    
    api_key = input("OpenAI API Key: ").strip()
    if not api_key:
        print("❌ 错误: API Key 不能为空")
        return
    
    base_url = input("API Base URL (可选，直接回车使用默认): ").strip()
    if not base_url:
        base_url = None
    
    model = input("模型名称 (默认 gpt-3.5-turbo): ").strip() or "gpt-3.5-turbo"
    
    output_file = input("输出文件路径 (默认 generated_story.yaml): ").strip() or "generated_story.yaml"
    
    category = input("分类名称 (默认 custom): ").strip() or "custom"
    
    print("\n" + "="*60)
    print("开始生成配置...")
    print("="*60 + "\n")
    
    try:
        # 生成配置
        config = generate_story_config_with_chatgpt(
            story_description=story_description,
            sequence_length=sequence_length,
            api_key=api_key,
            base_url=base_url,
            model=model
        )
        
        # 显示生成的配置
        print("\n" + "="*60)
        print("生成的配置预览:")
        print("="*60)
        print(f"\nconcept_token: {config['concept_token']}")
        print(f"\nstyle: {config['style']}")
        print(f"\nsubject: {config['subject']}")
        print(f"\nnew_subject: {config['new_subject']}")
        print(f"\nsettings ({len(config['settings'])} 个场景):")
        for i, setting in enumerate(config['settings'], 1):
            print(f"  {i}. {setting}")
        
        # 确认保存
        print("\n" + "="*60)
        confirm = input(f"\n是否保存到文件 '{output_file}'? (y/n, 默认 y): ").strip().lower()
        if confirm in ['', 'y', 'yes']:
            save_to_yaml(config, output_file, category)
            print(f"\n✅ 完成！配置文件已保存到: {output_file}")
            print(f"\n可以在 gen_benchmark.py 中使用此配置文件进行图像生成。")
        else:
            print("\n已取消保存。")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main_interactive()

