#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据故事描述生成 YAML 配置文件
用于图像生成的故事序列配置
"""

import os
import sys
import yaml
import argparse
import re
from typing import Dict, List

# 尝试导入 openai
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("警告: openai 库未安装，请使用 'pip install openai' 安装")


def generate_story_config_with_chatgpt(
    story_description: str,
    sequence_length: int,
    api_key: str,
    base_url: str = None,
    model: str = "gpt-3.5-turbo"
) -> Dict:
    """
    使用 ChatGPT API 根据故事描述生成配置
    
    Args:
        story_description: 故事描述，如"生成火柴人的故事"
        sequence_length: 序列长度（图像数量）
        api_key: OpenAI API key
        base_url: API base URL (可选)
        model: 使用的模型名称
    
    Returns:
        dict: 包含 concept_token, settings, style, subject, new_subject 的字典
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("openai 库未安装，请使用 'pip install openai' 安装")
    
    if not api_key:
        raise ValueError("请提供 OpenAI API Key")
    
    try:
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        
        # 提取主体（从故事描述中）
        # 例如："生成火柴人的故事" -> "火柴人"
        subject_match = re.search(r'生成(.+?)的', story_description)
        if subject_match:
            concept_token = subject_match.group(1).strip()
        else:
            # 如果没有匹配，尝试提取关键词
            concept_token = story_description.replace("生成", "").replace("的故事", "").strip()
            if not concept_token:
                concept_token = "subject"
        
        # 构建 prompt
        prompt = f"""根据以下故事描述，生成一个图像生成配置。

故事描述：{story_description}
需要生成 {sequence_length} 张图像的序列。

请生成以下内容：

1. **concept_token**: 一个简短的英文概念词（如 phoenix, zebra, stick_figure），用于标识主体
2. **settings**: 一个包含 {sequence_length} 个场景描述的列表，每个描述应该：
   - 简洁明了（5-10个英文单词）
   - 描述具体的动作或场景
   - 使用现在分词形式（如 "rising from", "soaring through"）
   - 每个场景应该连贯，形成一个完整的故事序列
3. **style**: 一个风格描述，格式为 "A [形容词] [类型] illustration of"（如 "A fiery and majestic illustration of"）
4. **subject**: 一个简洁的主体描述，格式为 "A [主体] with [特征]"（如 "A phoenix with bright orange feathers"）
5. **new_subject**: 一个详细的主体描述，包含更多视觉细节（如 "Phoenix with blazing bright orange feathers, scarlet wing edges, golden crest, amber tail plumes and sapphire eyes"）

请以 JSON 格式返回，格式如下：
{{
    "concept_token": "concept_name",
    "settings": ["scene 1", "scene 2", ...],
    "style": "A [adjective] [type] illustration of",
    "subject": "A [subject] with [features]",
    "new_subject": "Detailed description with visual details"
}}

只返回 JSON，不要添加其他文字："""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个专业的图像生成提示词助手，擅长将故事描述转换为结构化的图像生成配置。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        # 提取生成的文本
        generated_text = response.choices[0].message.content.strip()
        
        # 尝试提取 JSON
        json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
        if json_match:
            import json
            config = json.loads(json_match.group(0))
        else:
            # 如果无法解析 JSON，尝试手动解析
            print("警告: 无法解析 JSON，尝试手动解析...")
            config = parse_text_to_config(generated_text, concept_token, sequence_length)
        
        # 验证和补充配置
        if "concept_token" not in config or not config["concept_token"]:
            config["concept_token"] = concept_token.lower().replace(" ", "_")
        
        if "settings" not in config or len(config["settings"]) != sequence_length:
            print(f"警告: settings 数量不正确，期望 {sequence_length}，实际 {len(config.get('settings', []))}")
            # 补充或截断
            if len(config.get("settings", [])) < sequence_length:
                while len(config["settings"]) < sequence_length:
                    config["settings"].append(f"{concept_token} in scene {len(config['settings']) + 1}")
            else:
                config["settings"] = config["settings"][:sequence_length]
        
        # 确保所有字段都存在
        if "style" not in config:
            config["style"] = f"A vibrant and detailed illustration of"
        if "subject" not in config:
            config["subject"] = f"A {concept_token}"
        if "new_subject" not in config:
            config["new_subject"] = config.get("subject", f"{concept_token}")
        
        return config
    
    except Exception as e:
        raise Exception(f"调用 ChatGPT API 时出错: {str(e)}")


def parse_text_to_config(text: str, concept_token: str, sequence_length: int) -> Dict:
    """手动解析文本为配置"""
    config = {
        "concept_token": concept_token.lower().replace(" ", "_"),
        "settings": [],
        "style": "A vibrant and detailed illustration of",
        "subject": f"A {concept_token}",
        "new_subject": f"{concept_token} with detailed features"
    }
    
    # 尝试提取 settings
    lines = text.split('\n')
    in_settings = False
    for line in lines:
        line = line.strip()
        if 'settings' in line.lower() or '场景' in line or 'scene' in line.lower():
            in_settings = True
            continue
        if in_settings and line and (line.startswith('-') or line.startswith('•') or re.match(r'^\d+[\.\)]', line)):
            # 清理行
            line = re.sub(r'^[-•\d\.\)]\s*', '', line)
            if line:
                config["settings"].append(line)
                if len(config["settings"]) >= sequence_length:
                    break
    
    # 如果 settings 不够，补充
    while len(config["settings"]) < sequence_length:
        config["settings"].append(f"{concept_token} in scene {len(config['settings']) + 1}")
    
    # 提取 style
    style_match = re.search(r'style[:\s]+(.+)', text, re.IGNORECASE)
    if style_match:
        config["style"] = style_match.group(1).strip()
    
    # 提取 subject
    subject_match = re.search(r'subject[:\s]+(.+)', text, re.IGNORECASE)
    if subject_match:
        config["subject"] = subject_match.group(1).strip()
    
    return config


def save_to_yaml(config: Dict, output_path: str, category: str = "custom"):
    """
    保存配置到 YAML 文件
    
    Args:
        config: 配置字典
        output_path: 输出文件路径
        category: 分类名称（YAML 的顶级键）
    """
    # 构建 YAML 结构
    yaml_data = {
        category: [config]
    }
    
    # 如果文件已存在，读取并追加
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_data = yaml.safe_load(f) or {}
            
            if category in existing_data:
                existing_data[category].append(config)
            else:
                existing_data[category] = [config]
            
            yaml_data = existing_data
        except Exception as e:
            print(f"警告: 读取现有文件失败，将创建新文件: {e}")
    
    # 保存到文件
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    
    print(f"✅ 配置已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="根据故事描述生成 YAML 配置文件")
    parser.add_argument('--story', type=str, required=True, help='故事描述，如"生成火柴人的故事"')
    parser.add_argument('--length', type=int, default=5, help='序列长度（图像数量），默认 5')
    parser.add_argument('--api-key', type=str, required=True, help='OpenAI API Key')
    parser.add_argument('--base-url', type=str, default=None, help='API Base URL (可选)')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='使用的模型，默认 gpt-3.5-turbo')
    parser.add_argument('--output', type=str, default='generated_story.yaml', help='输出文件路径，默认 generated_story.yaml')
    parser.add_argument('--category', type=str, default='custom', help='YAML 分类名称，默认 custom')
    parser.add_argument('--append', action='store_true', help='追加到现有文件（如果存在）')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("故事配置生成器")
    print(f"{'='*60}")
    print(f"故事描述: {args.story}")
    print(f"序列长度: {args.length}")
    print(f"输出文件: {args.output}")
    print(f"{'='*60}\n")
    
    try:
        # 生成配置
        print("正在调用 ChatGPT API 生成配置...")
        config = generate_story_config_with_chatgpt(
            story_description=args.story,
            sequence_length=args.length,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model
        )
        
        # 显示生成的配置
        print("\n生成的配置:")
        print(f"  concept_token: {config['concept_token']}")
        print(f"  style: {config['style']}")
        print(f"  subject: {config['subject']}")
        print(f"  new_subject: {config['new_subject']}")
        print(f"  settings ({len(config['settings'])} 个场景):")
        for i, setting in enumerate(config['settings'], 1):
            print(f"    {i}. {setting}")
        
        # 保存到文件
        save_to_yaml(config, args.output, args.category)
        
        print(f"\n✅ 完成！配置文件已保存到: {args.output}")
        print(f"\n可以在 gen_benchmark.py 中使用此配置文件进行图像生成。")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

