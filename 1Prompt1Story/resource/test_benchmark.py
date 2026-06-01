import os
# 设置 HuggingFace 镜像地址（必须在导入其他库之前设置）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"

# 设置 GitHub 镜像地址（用于加速 GitHub 下载，必须在导入 dreamsim 之前设置）
# 通过 monkey patch urllib 来拦截和修改 GitHub URL
try:
    import urllib.request
    import urllib.error
    original_urlopen = urllib.request.urlopen
    
    def urlopen_with_github_mirror(req, *args, **kwargs):
        # 获取 URL
        if isinstance(req, str):
            url = req
            is_string = True
        else:
            # 尝试多种方式获取 URL
            if hasattr(req, 'full_url'):
                url = req.full_url
            elif hasattr(req, 'get_full_url'):
                url = req.get_full_url()
            else:
                url = str(req)
            is_string = False
        
        # 对于 zipball/tarball 类型的下载，镜像站点通常不支持，直接使用原始 URL
        if '/zipball/' in url or '/tarball/' in url:
            # 直接使用原始 GitHub URL，不进行镜像转换
            if not is_string and isinstance(req, urllib.request.Request):
                final_req = req
            else:
                final_req = url
            return original_urlopen(final_req, *args, **kwargs)
        
        # 对于其他 GitHub URL，尝试使用镜像站点
        if 'github.com' in url and 'ghfast.top' not in url and 'ghproxy.com' not in url:
            # 将 https://github.com 替换为 https://ghfast.top/github.com
            url = url.replace('https://github.com/', 'https://ghfast.top/github.com/')
            url = url.replace('http://github.com/', 'https://ghfast.top/github.com/')
        
        # 创建最终的 Request 对象
        if not is_string and isinstance(req, urllib.request.Request):
            final_req = urllib.request.Request(url, data=req.data if hasattr(req, 'data') else None, 
                                             headers=dict(req.headers) if hasattr(req, 'headers') else {},
                                             method=req.get_method() if hasattr(req, 'get_method') else None)
        else:
            final_req = url
        
        return original_urlopen(final_req, *args, **kwargs)
    
    urllib.request.urlopen = urlopen_with_github_mirror
except Exception as e:
    print(f"警告: 无法设置 GitHub 镜像 URL 重写: {e}")

# 尝试 monkey patch requests 库（如果使用 requests 下载）
try:
    import requests
    original_get = requests.get
    original_post = requests.post
    
    def get_with_github_mirror(url, *args, **kwargs):
        if 'github.com' in url and 'ghfast.top' not in url:
            url = url.replace('https://github.com/', 'https://ghfast.top/github.com/')
            url = url.replace('http://github.com/', 'https://ghfast.top/github.com/')
        return original_get(url, *args, **kwargs)
    
    def post_with_github_mirror(url, *args, **kwargs):
        if 'github.com' in url and 'ghfast.top' not in url:
            url = url.replace('https://github.com/', 'https://ghfast.top/github.com/')
            url = url.replace('http://github.com/', 'https://ghfast.top/github.com/')
        return original_post(url, *args, **kwargs)
    
    requests.get = get_with_github_mirror
    requests.post = post_with_github_mirror
except ImportError:
    pass  # requests 可能未安装
except Exception as e:
    print(f"警告: 无法设置 requests GitHub 镜像 URL 重写: {e}")

# 设置 git URL 重写（如果使用 git 命令下载）
os.environ["GIT_TERMINAL_PROMPT"] = "0"

# 尝试 monkey patch subprocess（如果使用 subprocess 调用 git clone）
try:
    import subprocess
    original_run = subprocess.run
    original_call = subprocess.call
    original_Popen = subprocess.Popen
    
    def process_git_command(cmd):
        """处理 git 命令中的 GitHub URL"""
        if not isinstance(cmd, (list, tuple)):
            return cmd
        cmd = list(cmd)
        # 如果命令是 git clone 或包含 github.com
        if any('git' in str(arg).lower() for arg in cmd):
            for i, arg in enumerate(cmd):
                if isinstance(arg, str) and 'github.com' in arg and 'ghfast.top' not in arg:
                    cmd[i] = arg.replace('https://github.com/', 'https://ghfast.top/github.com/')
                    cmd[i] = cmd[i].replace('http://github.com/', 'https://ghfast.top/github.com/')
        return cmd
    
    def run_with_github_mirror(*args, **kwargs):
        if args and isinstance(args[0], (list, tuple)):
            args = (process_git_command(args[0]),) + args[1:]
        return original_run(*args, **kwargs)
    
    def call_with_github_mirror(*args, **kwargs):
        if args and isinstance(args[0], (list, tuple)):
            args = (process_git_command(args[0]),) + args[1:]
        return original_call(*args, **kwargs)
    
    def Popen_with_github_mirror(*args, **kwargs):
        if args and isinstance(args[0], (list, tuple)):
            args = (process_git_command(args[0]),) + args[1:]
        return original_Popen(*args, **kwargs)
    
    subprocess.run = run_with_github_mirror
    subprocess.call = call_with_github_mirror
    subprocess.Popen = Popen_with_github_mirror
except Exception as e:
    print(f"警告: 无法设置 subprocess GitHub 镜像 URL 重写: {e}")

from dreamsim import dreamsim
from PIL import Image
from pathlib import Path
from itertools import combinations
import torch
from transformers import CLIPProcessor, CLIPModel
import argparse
import clip
import shutil

from carvekit.api.high import HiInterface
# 修改 carvekit 的下载器使用镜像地址
try:
    from carvekit.utils import download_models
    # 修改 fallback_downloader 的 base_url 为镜像地址
    if hasattr(download_models, 'fallback_downloader'):
        download_models.fallback_downloader.base_url = "https://hf-mirror.com"
        download_models.fallback_downloader._name = "hf-mirror.com"
    # 如果主下载器也有 fallback，也修改它
    if hasattr(download_models, 'downloader') and hasattr(download_models.downloader, '_fallback_downloader'):
        if download_models.downloader._fallback_downloader:
            download_models.downloader._fallback_downloader.base_url = "https://hf-mirror.com"
            download_models.downloader._fallback_downloader._name = "hf-mirror.com"
except Exception as e:
    print(f"警告: 无法修改 carvekit 下载器配置: {e}")

import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import sklearn.preprocessing
import yaml

interface = HiInterface(object_type="object",  # Can be "object" or "hairs-like".
                        batch_size_seg=5,
                        batch_size_matting=1,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                        matting_mask_size=2048,
                        trimap_prob_threshold=231,
                        trimap_dilation=30,
                        trimap_erosion_iters=5,
                        fp16=False)


def replace_bg_with_noise(image_path):
    image = interface([image_path])[0].convert('RGB')
    image_array = np.array(image)

    # Identify regions with all black pixels
    black_pixels = np.all(image_array == [130, 130, 130], axis=-1)

    # Generate random uniform noise
    noise = np.random.randint(0, 256, size=image_array.shape, dtype=np.uint8)

    # Replace black regions with noise
    image_array[black_pixels] = noise[black_pixels]

    # Convert the modified NumPy array back to a PIL image
    image = Image.fromarray(image_array)

    return image


def get_clip_score_for_image_text(image, caption, model, device):
    # Define the preprocessing steps for the image
    preprocess = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # Load and preprocess the image
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Tokenize and encode the caption
    caption = 'A photo depicts ' + caption if not caption.startswith('A photo depicts') else caption
    caption_tokens = clip.tokenize(caption, truncate=True).squeeze()
    caption_tokens = caption_tokens.unsqueeze(0)  # Add batch dimension

    # Move image and caption tokens to the device
    image = image.to(device)
    caption_tokens = caption_tokens.to(device)

    # Encode image and caption
    image_features = model.encode_image(image)
    text_features = model.encode_text(caption_tokens)

    # Normalize features
    image_features = sklearn.preprocessing.normalize(image_features.cpu().detach().numpy(), axis=1)
    text_features = sklearn.preprocessing.normalize(text_features.cpu().detach().numpy(), axis=1)

    # Calculate CLIP score
    clip_score = 2.5 * np.sum(image_features * text_features)

    return clip_score


def load_yaml_subject_mapping(yaml_path):
    """
    加载YAML文件并构建subject到new_subject的映射
    返回一个字典：{(style, subject): new_subject}
    """
    subject_mapping = {}
    if not os.path.exists(yaml_path):
        print(f"警告: YAML文件不存在: {yaml_path}")
        return subject_mapping
    
    with open(yaml_path, 'r', encoding='utf-8') as file:
        refined_data = yaml.safe_load(file)
    
    for subject_domain, subject_domain_instances in refined_data.items():
        for refined_instance in subject_domain_instances:
            refined_style = refined_instance.get("style", "").strip()
            refined_subject = refined_instance.get("subject", "").strip()
            new_subject = refined_instance.get("new_subject", "").strip()
            
            if refined_subject and new_subject:
                # 使用 (style, subject) 作为键
                key = (refined_style, refined_subject)
                subject_mapping[key] = new_subject
    
    return subject_mapping


def replace_subject_with_new_subject(filename, subject_mapping):
    """
    从文件名中提取style和subject，如果找到匹配的条目，用new_subject替换subject
    文件名格式: {style} {subject} {setting}.jpg
    返回替换后的prompt（不包含扩展名）
    """
    # 移除扩展名
    filename_without_ext = os.path.splitext(filename)[0]
    
    # 按前缀长度排序，优先匹配更长的前缀（避免短前缀误匹配）
    sorted_items = sorted(
        subject_mapping.items(),
        key=lambda x: len(x[0][0] + " " + x[0][1]) if x[0][0] else len(x[0][1]),
        reverse=True
    )
    
    # 尝试匹配所有可能的 (style, subject) 组合
    for (style, subject), new_subject in sorted_items:
        # 构建匹配的前缀：{style} {subject}
        if style:
            prefix = f"{style} {subject}"
        else:
            prefix = subject
        
        # 检查文件名是否以该前缀开头，并且前缀后是空格或字符串结束（避免部分匹配）
        if filename_without_ext.startswith(prefix):
            # 检查前缀后是否是空格或字符串结束，避免部分匹配
            if len(filename_without_ext) == len(prefix) or filename_without_ext[len(prefix)] == ' ':
                # 获取剩余的setting部分
                remaining = filename_without_ext[len(prefix):].strip()
                # 构建新的prompt：{style} {new_subject} {setting}
                if style:
                    if remaining:
                        new_prompt = f"{style} {new_subject} {remaining}"
                    else:
                        new_prompt = f"{style} {new_subject}"
                else:
                    if remaining:
                        new_prompt = f"{new_subject} {remaining}"
                    else:
                        new_prompt = new_subject
                return new_prompt
    
    # 如果没有找到匹配，返回原始文件名（不含扩展名）
    return filename_without_ext


def calculate_avg_clip_text_score(folder_path, single_folder, prefix=None, use_new_subject=False, yaml_path=None):
    model, _ = clip.load('ViT-B/32', device="cuda", jit=False)
    model.eval()
    
    # 如果启用了new_subject替换，加载YAML映射
    subject_mapping = {}
    if use_new_subject:
        if yaml_path is None:
            # 默认使用当前目录下的YAML文件
            yaml_path = os.path.join(os.path.dirname(__file__), "consistory+1p1s_allprompt_refined.yaml")
        subject_mapping = load_yaml_subject_mapping(yaml_path)
        if subject_mapping:
            print(f"已加载 {len(subject_mapping)} 个subject映射")
        else:
            print("警告: 未找到任何subject映射，将使用原始文件名")
    
    if single_folder:
        subfolders = [folder_path]
    else:
        subfolders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
        # Filter by prefix if specified
        # print(subfolders)
        if prefix:
            subfolders = [name for name in subfolders if name.startswith(prefix)]
    avg_distances = {}
    for subfolder_name in subfolders:
        images = []
        text_prompts = []
        avg_distance = 0
        subfolder_path = os.path.join(folder_path, subfolder_name)
        for f in Path(subfolder_path).rglob('*'):
            if f.suffix.lower() in {'.png', '.jpg'}:
                images.append(Image.open(f))
                filename = os.path.basename(f)
                # 如果启用了new_subject替换，替换subject部分
                if use_new_subject and subject_mapping:
                    text_prompt = replace_subject_with_new_subject(filename, subject_mapping)
                else:
                    text_prompt = filename
                text_prompts.append(text_prompt)

        # Skip if no images are found
        if not images:
            print(f"No images found in {subfolder_name}, skipping...")
            continue

        for image, text_prompt in zip(images, text_prompts):
            distance = get_clip_score_for_image_text(image, text_prompt, model, "cuda")
            avg_distance += distance.item()

        avg_distance /= len(images)
        avg_distances[subfolder_name] = avg_distance
    return avg_distances


# Compute DreamSim distance between two images
def compute_dreamsim_distance(image1, image2, model, processor):
    features1 = processor(image1).to("cuda")
    features2 = processor(image2).to("cuda")
    distance = model(features1, features2)
    return distance


# Compute CLIP distance between two images
def compute_clip_distance(image1, image2, model, processor):
    inputs1 = processor(images=image1, return_tensors="pt")
    with torch.no_grad():
        features1 = model.get_image_features(**inputs1)

    inputs2 = processor(images=image2, return_tensors="pt")
    with torch.no_grad():
        features2 = model.get_image_features(**inputs2)

    distance = torch.nn.functional.cosine_similarity(features1, features2)
    return distance


def safe_load_dreamsim(max_retries=2):
    """
    安全加载 dreamsim 模型，如果遇到缓存损坏问题会自动清理并重试
    """
    for attempt in range(max_retries):
        try:
            model, processor = dreamsim(pretrained=True)
            return model, processor
        except (RuntimeError, OSError) as e:
            error_msg = str(e).lower()
            # 检查是否是缓存损坏相关的错误
            if 'zip archive' in error_msg or 'central directory' in error_msg or 'failed reading' in error_msg:
                print(f"检测到缓存损坏错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                
                # 清理可能的缓存位置
                cache_dirs = [
                    './models/facebookresearch_dino_main',
                    os.path.join(torch.hub.get_dir(), 'facebookresearch_dino_main'),
                ]
                
                for cache_dir in cache_dirs:
                    if os.path.exists(cache_dir):
                        print(f"清理缓存目录: {cache_dir}")
                        try:
                            shutil.rmtree(cache_dir)
                        except Exception as cleanup_error:
                            print(f"清理缓存目录失败: {cleanup_error}")
                
                # 清理 PyTorch hub 缓存中的 DINO 权重文件
                hub_cache_dir = torch.hub.get_dir()
                dino_weight_patterns = ['dino_vitbase16_pretrain.pth']
                for root, dirs, files in os.walk(hub_cache_dir):
                    for file in files:
                        if any(pattern in file for pattern in dino_weight_patterns):
                            file_path = os.path.join(root, file)
                            print(f"删除损坏的权重文件: {file_path}")
                            try:
                                os.remove(file_path)
                            except Exception as cleanup_error:
                                print(f"删除权重文件失败: {cleanup_error}")
                
                # 如果是最后一次尝试，抛出异常
                if attempt == max_retries - 1:
                    raise RuntimeError(f"加载 dreamsim 模型失败，已重试 {max_retries} 次。请检查网络连接或手动清理缓存。") from e
                
                print("缓存已清理，正在重试...")
            else:
                # 其他类型的错误直接抛出
                raise
    raise RuntimeError(f"加载 dreamsim 模型失败，已重试 {max_retries} 次。")


# Compute average distance for images in a folder
def calculate_avg_distance(folder_path, mode, remove_bg, single_folder, prefix=None):
    if mode == "dreamsim":
        model, processor = safe_load_dreamsim()
    elif mode == "clip_image":
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    if single_folder:
        subfolders = [folder_path]
    else:
        subfolders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
        # Filter by prefix if specified
        if prefix:
            subfolders = [name for name in subfolders if name.startswith(prefix)]

    avg_distances = {}
    for subfolder_name in subfolders:
        images = []
        avg_distance = 0
        subfolder_path = os.path.join(folder_path, subfolder_name)
        for f in Path(subfolder_path).rglob('*'):
            if f.suffix.lower() in {'.png', '.jpg'}:
                if remove_bg:
                    images.append(replace_bg_with_noise(f))
                else:
                    images.append(Image.open(f))

        # Skip if no images are found
        if not images:
            print(f"No images found in {subfolder_name}, skipping...")
            continue

        if len(images) < 2:
            avg_distances[subfolder_name] = float('nan')
            continue

        for combo in combinations(images, 2):
            if mode == "dreamsim":
                distance = compute_dreamsim_distance(combo[0], combo[1], model, processor)
            elif mode == "clip_image":
                distance = compute_clip_distance(combo[0], combo[1], model, processor)

            avg_distance += distance.item()

        avg_distance /= len(images) * (len(images) - 1) / 2
        avg_distances[subfolder_name] = avg_distance
        print(f"Average distance for {subfolder_name}: {avg_distance}")

    return avg_distances


# Main function to parse arguments and execute tasks
def main():
    parser = argparse.ArgumentParser(description="Calculate image similarities using DreamSim or CLIP.")
    parser.add_argument('--mode', type=str, choices=['dreamsim', 'clip_image', 'clip_text'],
                        help='Mode of operation: "dreamsim" or "clip_image"')
    parser.add_argument('--folder', type=str, help='Path to the folder containing image subfolders')
    parser.add_argument('--image1', type=str, help='Path to the first image for pairwise comparison')
    parser.add_argument('--image2', type=str, help='Path to the second image for pairwise comparison')
    parser.add_argument('--image_clip_text', type=str,
                        help='Path to the image for clip_text score, image_name as text_prompt')
    parser.add_argument('--remove_background', action='store_true', help='Remove background from images')
    parser.add_argument('--single_folder', action='store_true', help='single_folder')
    parser.add_argument('--prefix', type=str, default=None, help='Only test folders with this prefix (e.g., "human" to test only folders starting with "human")')
    parser.add_argument('--allprefix', action='store_true', help='Test all different prefixes (humans, animals, etc.) separately and save results by category')
    parser.add_argument('--new_subject', action='store_true', help='Use new_subject from YAML file to replace subject in prompts when calculating clip_text score')
    parser.add_argument('--yaml_path', type=str, default=None, help='Path to YAML file containing subject and new_subject mappings (default: consistory+1p1s_allprompt_refined.yaml in same directory)')
    parser.add_argument('--output_file', type=str, default=None, help='Output file path to save results (default: test_result/{mode}_results_{timestamp}.txt)')

    args = parser.parse_args()

    if args.image_clip_text:
        image = Image.open(args.image_clip_text).convert("RGB")
        caption = os.path.basename(args.image_clip_text)
        model, _ = clip.load('ViT-B/32', device="cuda", jit=False)
        model.eval()
        clip_score = get_clip_score_for_image_text(image, caption, model, "cuda")

        print(f"Clip_score:{clip_score}")

    elif args.image1 and args.image2:
        if args.remove_background == True:
            image1 = replace_bg_with_noise(args.image1)
            image2 = replace_bg_with_noise(args.image2)
        else:
            image1 = Image.open(args.image1).convert("RGB")
            image2 = Image.open(args.image2).convert("RGB")

        if args.mode == 'dreamsim':
            model, processor = safe_load_dreamsim()
            distance = compute_dreamsim_distance(image1, image2, model, processor)
        elif args.mode == 'clip_image':
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
            distance = compute_clip_distance(image1, image2, model, processor)

        print(f"Distance: {distance}")

    else:
        if args.allprefix:
            # 检测所有不同的前缀类别
            import re
            from collections import defaultdict
            from datetime import datetime
            
            if not os.path.isdir(args.folder):
                print(f"错误: {args.folder} 不是一个有效的目录")
                return
            
            # 获取所有子目录
            subfolders = [name for name in os.listdir(args.folder) 
                         if os.path.isdir(os.path.join(args.folder, name))]
            
            # 提取所有不同的前缀（如 humans, animals）
            prefix_categories = defaultdict(list)
            for subfolder in subfolders:
                # 匹配模式：prefix_number (如 animals_0, humans_1)
                match = re.match(r'^([a-zA-Z]+)_\d+$', subfolder)
                if match:
                    prefix = match.group(1)
                    prefix_categories[prefix].append(subfolder)
            
            if not prefix_categories:
                print("警告: 未找到任何符合 prefix_number 格式的子目录，将使用普通模式")
                args.allprefix = False
            else:
                print(f"检测到 {len(prefix_categories)} 个类别: {', '.join(prefix_categories.keys())}")
                
                # 为每个类别分别测试
                category_results = {}
                all_subfolders_results = {}
                
                for category, category_subfolders in prefix_categories.items():
                    print(f"\n{'='*60}")
                    print(f"测试类别: {category} ({len(category_subfolders)} 个子目录)")
                    print(f"{'='*60}")
                    
                    if args.mode == 'dreamsim' or args.mode == 'clip_image':
                        category_avg_distances = calculate_avg_distance(
                            args.folder, args.mode, args.remove_background, 
                            args.single_folder, category
                        )
                    elif args.mode == 'clip_text':
                        category_avg_distances = calculate_avg_clip_text_score(
                            args.folder, args.single_folder, category, 
                            args.new_subject, args.yaml_path
                        )
                    
                    # 计算该类别的平均距离
                    category_total = 0
                    valid_count = 0
                    for subfolder, avg_distance in category_avg_distances.items():
                        # 检查是否为有效数值（不是 NaN）
                        is_valid = not (isinstance(avg_distance, float) and (avg_distance != avg_distance))
                        if is_valid:
                            category_total += avg_distance
                            valid_count += 1
                        all_subfolders_results[subfolder] = avg_distance
                        print(f"Average distance for {subfolder}: {avg_distance}")
                    
                    if valid_count > 0:
                        category_avg = category_total / valid_count
                        category_results[category] = {
                            'avg_distance': category_avg,
                            'subfolders': category_avg_distances,
                            'count': valid_count
                        }
                        print(f"\n{category} 类别平均距离: {category_avg}")
                    else:
                        category_results[category] = {
                            'avg_distance': float('nan'),
                            'subfolders': category_avg_distances,
                            'count': 0
                        }
                        print(f"\n{category} 类别: 无有效数据")
                
                # 计算总体平均距离
                print(f"\n{'='*60}")
                print("总体结果")
                print(f"{'='*60}")
                
                total_avg_distance = 0
                total_valid_count = 0
                for category, result in category_results.items():
                    if not (isinstance(result['avg_distance'], float) and (result['avg_distance'] != result['avg_distance'])):
                        total_avg_distance += result['avg_distance'] * result['count']
                        total_valid_count += result['count']
                
                if total_valid_count > 0:
                    overall_avg = total_avg_distance / total_valid_count
                    print(f"总体平均距离 ({args.mode}_total_avg_distance): {overall_avg}")
                else:
                    overall_avg = float('nan')
                    print("总体平均距离: 无有效数据")
                
                # 保存结果到文件
                if args.output_file is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = os.path.join(os.path.dirname(__file__), 'test_result')
                    os.makedirs(output_dir, exist_ok=True)
                    args.output_file = os.path.join(output_dir, f"{args.mode}_results_{timestamp}.txt")
                
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    f.write(f"{'='*60}\n")
                    f.write(f"测试结果 - {args.mode}\n")
                    f.write(f"测试目录: {args.folder}\n")
                    f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"{'='*60}\n\n")
                    
                    # 按类别保存结果
                    for category, result in category_results.items():
                        f.write(f"类别: {category}\n")
                        f.write(f"  平均距离: {result['avg_distance']}\n")
                        f.write(f"  子目录数量: {result['count']}\n")
                        f.write(f"  详细结果:\n")
                        for subfolder, distance in result['subfolders'].items():
                            f.write(f"    {subfolder}: {distance}\n")
                        f.write("\n")
                    
                    # 总体结果
                    f.write(f"{'='*60}\n")
                    f.write(f"总体结果\n")
                    f.write(f"{'='*60}\n")
                    f.write(f"{args.mode}_total_avg_distance: {overall_avg}\n")
                    f.write(f"总有效子目录数: {total_valid_count}\n")
                
                print(f"\n结果已保存到: {args.output_file}")
                return
        
        # 原有的单次测试逻辑
        if args.mode == 'dreamsim' or args.mode == 'clip_image':
            avg_distances = calculate_avg_distance(args.folder, args.mode, args.remove_background, args.single_folder, args.prefix)
        elif args.mode == 'clip_text':
            avg_distances = calculate_avg_clip_text_score(args.folder, args.single_folder, args.prefix, args.new_subject, args.yaml_path)

        total_avg_distance = 0
        for subfolder, avg_distance in avg_distances.items():
            total_avg_distance += round(avg_distance, 5)
            # if avg_distance < 0.84:
            print(f"Average distance for {subfolder}: {avg_distance}")

        total_avg_distance /= len(avg_distances)
        print(f"{args.mode}_total_avg_distance: {total_avg_distance}")


if __name__ == "__main__":
    main()