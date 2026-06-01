#!/usr/bin/env python3
"""
脚本：将指定文件夹中的图像横向合并成一张大图
用法: 
  - 单个文件夹: python merge_images.py <文件夹路径>
  - 批量处理: python merge_images.py <目录路径> [--prefix 前缀]
  
示例: 
  - 单个文件夹: python merge_images.py ./result/benchmark_refinedprompt_2/animals_2
  - 批量处理所有子文件夹: python merge_images.py ./result/benchmark_1p1s
  - 批量处理指定前缀的子文件夹: python merge_images.py ./result/benchmark_1p1s --prefix animals
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont
import argparse
from pathlib import Path


def merge_pil_horizontally(images, common_prefix, action_prompts):
    """
    将已加载的 PIL 图像按顺序横向拼接，顶部绘制 common_prefix，每张图下方绘制对应动作文案。
    与 merge_images_horizontally 的排版逻辑一致；适用于内存中已有有序图像列表的场景。
    """
    if not images:
        raise ValueError("images 不能为空")
    if len(action_prompts) != len(images):
        raise ValueError(
            f"action_prompts 长度 ({len(action_prompts)}) 与 images ({len(images)}) 不一致"
        )

    heights = [img.height for img in images]
    min_height = min(heights)
    max_height = max(heights)

    if min_height != max_height:
        resized_images = []
        for img in images:
            aspect_ratio = img.width / img.height
            new_width = int(min_height * aspect_ratio)
            resized_images.append(img.resize((new_width, min_height), Image.Resampling.LANCZOS))
        images = resized_images

    total_width = sum(img.width for img in images)
    uniform_height = images[0].height

    try:
        font_size_title = 60
        font_size_action = 48
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size_title)
            action_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size_action)
        except Exception:
            try:
                title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size_title)
                action_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size_action)
            except Exception:
                title_font = ImageFont.load_default()
                action_font = ImageFont.load_default()
    except Exception:
        title_font = ImageFont.load_default()
        action_font = ImageFont.load_default()

    padding_top = 20
    padding_bottom = 20
    padding_between = 15
    padding_action = 10

    try:
        test_img = Image.new("RGB", (100, 100), "white")
        test_draw = ImageDraw.Draw(test_img)
        bbox_title = test_draw.textbbox((0, 0), common_prefix, font=title_font)
        title_height = bbox_title[3] - bbox_title[1] + padding_top + padding_between
        bbox_action = test_draw.textbbox((0, 0), "Test", font=action_font)
        action_height = bbox_action[3] - bbox_action[1] + padding_action
    except Exception:
        title_height = 60
        action_height = 40

    image_y = title_height
    temp_image = Image.new("RGB", (total_width, 100), color="white")
    draw = ImageDraw.Draw(temp_image)

    max_lines = 1
    action_prompts_lines = []
    for action_prompt, img in zip(action_prompts, images):
        if action_prompt:
            img_width = img.width
            margin = 20
            max_text_width = img_width - margin

            try:
                bbox = draw.textbbox((0, 0), action_prompt, font=action_font)
                text_width = bbox[2] - bbox[0]
            except Exception:
                text_width = len(action_prompt) * 8

            if text_width > max_text_width:
                words = action_prompt.split(" ")
                lines = []
                current_line = []

                for word in words:
                    test_line = " ".join(current_line + [word]) if current_line else word
                    try:
                        bbox = draw.textbbox((0, 0), test_line, font=action_font)
                        line_width = bbox[2] - bbox[0]
                    except Exception:
                        line_width = len(test_line) * 8

                    if line_width <= max_text_width:
                        current_line.append(word)
                    else:
                        if current_line:
                            lines.append(" ".join(current_line))
                        current_line = [word]

                if current_line:
                    lines.append(" ".join(current_line))

                action_prompts_lines.append(lines)
                max_lines = max(max_lines, len(lines))
            else:
                action_prompts_lines.append([action_prompt])
        else:
            action_prompts_lines.append([])

    try:
        bbox = draw.textbbox((0, 0), "Test", font=action_font)
        single_line_height = bbox[3] - bbox[1]
    except Exception:
        single_line_height = 35

    line_spacing = 5
    action_height = max_lines * single_line_height + (max_lines - 1) * line_spacing + padding_action * 2

    final_height = title_height + uniform_height + action_height + padding_bottom
    merged_image = Image.new("RGB", (total_width, final_height), color="white")
    draw = ImageDraw.Draw(merged_image)

    if common_prefix:
        try:
            bbox = draw.textbbox((0, 0), common_prefix, font=title_font)
            text_width = bbox[2] - bbox[0]
            text_x = (total_width - text_width) // 2
            text_y = padding_top
            draw.text((text_x, text_y), common_prefix, fill="black", font=title_font)
        except Exception as e:
            print(f"警告: 无法绘制标题文字: {e}")
            draw.text((10, padding_top), common_prefix, fill="black")

    x_offset = 0
    image_widths = []
    for img in images:
        merged_image.paste(img, (x_offset, image_y))
        image_widths.append(x_offset)
        x_offset += img.width

    action_y = image_y + uniform_height + padding_action
    for x_offset, lines, img in zip(image_widths, action_prompts_lines, images):
        if lines:
            img_width = img.width
            for line_idx, line in enumerate(lines):
                try:
                    bbox = draw.textbbox((0, 0), line, font=action_font)
                    line_text_width = bbox[2] - bbox[0]
                except Exception:
                    line_text_width = len(line) * 8

                line_x = x_offset + (img_width - line_text_width) // 2
                line_y = action_y + line_idx * (single_line_height + line_spacing)
                try:
                    draw.text((line_x, line_y), line, fill="black", font=action_font)
                except Exception:
                    draw.text((line_x, line_y), line, fill="black")

    return merged_image


def merge_images_horizontally(folder_path, output_dir=None):
    """
    将指定文件夹中的所有图像横向合并成一张大图
    
    :param folder_path: 包含图像的文件夹路径
    :param output_dir: 输出目录，如果为None则使用 resource/test_result
    :return: 保存的文件路径
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        raise ValueError(f"文件夹不存在: {folder_path}")
    
    if not folder_path.is_dir():
        raise ValueError(f"路径不是文件夹: {folder_path}")
    
    # 获取所有图像文件（支持常见格式）
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(folder_path.glob(f'*{ext}'))
        image_files.extend(folder_path.glob(f'*{ext.upper()}'))
    
    # 过滤掉 story_image 等特殊文件
    image_files = [f for f in image_files if not f.name.startswith('story_image')]
    
    if not image_files:
        raise ValueError(f"文件夹中没有找到图像文件: {folder_path}")
    
    # 按文件名排序
    image_files.sort(key=lambda x: x.name)
    
    print(f"找到 {len(image_files)} 张图像:")
    for img_file in image_files:
        print(f"  - {img_file.name}")
    
    # 提取文件名的公共前缀（id prompt）和每张图的后缀部分（动作 prompt）
    filenames = [f.stem for f in image_files]  # 去除扩展名
    
    # 找到所有文件名的公共前缀
    def find_common_prefix(strings):
        if not strings:
            return ""
        if len(strings) == 1:
            return strings[0]
        # 找到最短字符串的长度
        min_len = min(len(s) for s in strings)
        # 逐字符比较
        prefix = ""
        for i in range(min_len):
            char = strings[0][i]
            if all(s[i] == char for s in strings):
                prefix += char
            else:
                break
        # 找到最后一个完整的单词边界（避免截断单词）
        last_space = prefix.rfind(' ')
        if last_space > 0:
            prefix = prefix[:last_space]
        return prefix
    
    common_prefix = find_common_prefix(filenames)
    
    # 提取每张图的后缀部分（动作 prompt）
    action_prompts = []
    for filename in filenames:
        if filename.startswith(common_prefix):
            action_prompt = filename[len(common_prefix):].strip()
            # 去除前导空格和可能的标点
            if action_prompt.startswith(' '):
                action_prompt = action_prompt[1:]
            action_prompts.append(action_prompt)
        else:
            # 如果没有公共前缀，使用整个文件名
            action_prompts.append(filename)
    
    print(f"\n公共 id prompt: {common_prefix}")
    print(f"动作 prompts:")
    for i, action in enumerate(action_prompts):
        print(f"  [{i+1}] {action}")
    
    # 读取所有图像（与 action_prompts 按下标对齐，跳过读取失败的文件）
    images = []
    action_prompts_kept = []
    for idx, img_file in enumerate(image_files):
        try:
            img = Image.open(img_file)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)
            action_prompts_kept.append(action_prompts[idx])
        except Exception as e:
            print(f"警告: 无法读取图像 {img_file.name}: {e}")
            continue

    if not images:
        raise ValueError(f"无法读取任何图像文件")

    merged_image = merge_pil_horizontally(images, common_prefix, action_prompts_kept)

    # 确定输出目录
    if output_dir is None:
        # 使用 resource/test_result 作为默认输出目录
        script_dir = Path(__file__).parent
        output_dir = script_dir / 'test_result'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 从文件夹路径提取名称
    # 例如: ./result/benchmark_refinedprompt_2/animals_2 -> benchmark_refinedprompt_2+animals_2
    folder_parts = folder_path.parts
    
    # 找到 result 目录的索引
    result_idx = -1
    for i, part in enumerate(folder_parts):
        if 'result' in part.lower():
            result_idx = i
            break
    
    if result_idx >= 0 and len(folder_parts) >= result_idx + 2:
        # 提取 benchmark_xxx 和 animals_xxx
        benchmark_name = folder_parts[result_idx + 1]  # benchmark_refinedprompt_2
        folder_name = folder_parts[-1]  # animals_2
        output_filename = f"{benchmark_name}+{folder_name}.jpg"
    else:
        # 如果无法从路径提取，使用文件夹名
        output_filename = f"{folder_path.name}_merged.jpg"
    
    output_path = output_dir / output_filename
    
    # 保存图像
    merged_image.save(output_path, 'JPEG', quality=95)
    
    tw, fh = merged_image.size
    print(f"\n合并完成！")
    print(f"输出文件: {output_path}")
    print(f"图像尺寸: {tw} x {fh}")
    print(f"包含图像数: {len(images)}")
    
    return output_path


def batch_merge_images(base_dir, prefix=None, output_base_dir=None):
    """
    批量处理：遍历指定目录下所有子文件夹（或所有以指定前缀开头的子文件夹），对每个子文件夹中的图像进行合并
    
    :param base_dir: 基础目录路径（例如: ./result/benchmark_1p1s）
    :param prefix: 子文件夹前缀（例如: animals），如果为None则处理所有子文件夹
    :param output_base_dir: 输出基础目录，如果为None则使用 result/visual_result
    :return: 处理的文件列表
    """
    base_dir = Path(base_dir)
    
    if not base_dir.exists():
        raise ValueError(f"目录不存在: {base_dir}")
    
    if not base_dir.is_dir():
        raise ValueError(f"路径不是目录: {base_dir}")
    
    # 找到所有子文件夹（如果指定了prefix，则只找以prefix开头的）
    if prefix:
        matching_folders = [f for f in base_dir.iterdir() if f.is_dir() and f.name.startswith(prefix)]
        if not matching_folders:
            raise ValueError(f"在 {base_dir} 中没有找到以 '{prefix}' 开头的子文件夹")
    else:
        matching_folders = [f for f in base_dir.iterdir() if f.is_dir()]
        if not matching_folders:
            raise ValueError(f"在 {base_dir} 中没有找到子文件夹")
    
    # 按文件夹名排序
    matching_folders.sort(key=lambda x: x.name)
    
    if prefix:
        print(f"找到 {len(matching_folders)} 个以 '{prefix}' 开头的子文件夹:")
    else:
        print(f"找到 {len(matching_folders)} 个子文件夹:")
    for folder in matching_folders:
        print(f"  - {folder.name}")
    
    # 确定输出基础目录
    if output_base_dir is None:
        # 使用 result/visual_result 作为默认输出目录
        # 从 base_dir 提取 result 路径
        base_parts = base_dir.parts
        result_idx = -1
        for i, part in enumerate(base_parts):
            if 'result' in part.lower():
                result_idx = i
                break
        
        if result_idx >= 0:
            # 构建 result/visual_result/benchmark_xxx 路径
            result_path = Path(*base_parts[:result_idx + 1])  # 到 result 目录
            output_base_dir = result_path / 'visual_result' / base_dir.name
        else:
            # 如果找不到 result 目录，使用 base_dir 的父目录
            output_base_dir = base_dir.parent / 'visual_result' / base_dir.name
    else:
        output_base_dir = Path(output_base_dir) / base_dir.name
    
    output_base_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n输出目录: {output_base_dir}")
    
    # 处理每个子文件夹
    output_paths = []
    for folder in matching_folders:
        print(f"\n{'='*60}")
        print(f"处理文件夹: {folder.name}")
        print(f"{'='*60}")
        try:
            output_path = merge_images_horizontally(folder, output_base_dir)
            output_paths.append(output_path)
            print(f"✓ 成功处理: {folder.name}")
        except Exception as e:
            print(f"✗ 处理失败 {folder.name}: {e}", file=sys.stderr)
            continue
    
    print(f"\n{'='*60}")
    print(f"批量处理完成！共处理 {len(output_paths)} 个文件夹")
    print(f"输出目录: {output_base_dir}")
    print(f"{'='*60}")
    
    return output_paths


def main():
    parser = argparse.ArgumentParser(
        description='将指定文件夹中的图像横向合并成一张大图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单个文件夹处理
  python merge_images.py ./result/benchmark_refinedprompt_2/animals_2
  python merge_images.py ./result/benchmark_refinedprompt_2/animals_2 --output-dir ./output
  
  # 批量处理：处理指定目录下所有子文件夹（默认行为）
  python merge_images.py ./result/benchmark_1p1s
  python merge_images.py ./result/benchmark_1p1s --output-dir ./custom_output
  
  # 批量处理：处理指定目录下所有以指定前缀开头的子文件夹
  python merge_images.py ./result/benchmark_1p1s --prefix animals
  python merge_images.py ./result/benchmark_1p1s --prefix animals --output-dir ./custom_output
        """
    )
    parser.add_argument('folder_path', type=str, help='包含图像的文件夹路径（单个文件夹）或包含子文件夹的目录（批量处理时）')
    parser.add_argument('--output-dir', type=str, default=None, 
                       help='输出目录（默认: resource/test_result 或 result/visual_result）')
    parser.add_argument('--prefix', type=str, default=None,
                       help='批量处理模式：指定子文件夹前缀（例如: animals）。如果不指定prefix且输入路径是目录，则自动处理所有子文件夹')
    
    args = parser.parse_args()
    
    try:
        input_path = Path(args.folder_path)
        
        if not input_path.exists():
            raise ValueError(f"路径不存在: {input_path}")
        
        # 判断处理模式
        if args.prefix:
            # 指定了prefix，使用批量处理模式（只处理以prefix开头的子文件夹）
            output_paths = batch_merge_images(args.folder_path, args.prefix, args.output_dir)
            print(f"\n成功处理 {len(output_paths)} 个文件夹")
        elif input_path.is_dir():
            # 未指定prefix且是目录，检查是否有子文件夹
            subdirs = [f for f in input_path.iterdir() if f.is_dir()]
            if subdirs:
                # 有子文件夹，自动批量处理所有子文件夹
                print(f"检测到目录下有 {len(subdirs)} 个子文件夹，自动启用批量处理模式")
                output_paths = batch_merge_images(args.folder_path, None, args.output_dir)
                print(f"\n成功处理 {len(output_paths)} 个文件夹")
            else:
                # 没有子文件夹，使用单个文件夹处理模式
                output_path = merge_images_horizontally(args.folder_path, args.output_dir)
                print(f"\n成功保存到: {output_path}")
        else:
            # 单个文件夹处理模式（文件路径或不存在子文件夹的目录）
            output_path = merge_images_horizontally(args.folder_path, args.output_dir)
            print(f"\n成功保存到: {output_path}")
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
