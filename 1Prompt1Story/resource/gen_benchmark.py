import os
import sys
# 添加父目录到 sys.path，以便导入 main 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置 HuggingFace 镜像地址（必须在导入其他库之前设置）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import yaml
from main import generate_images, load_unet_controller
from unet import utils
import torch
import queue 
import threading  
from tqdm import tqdm  # Import tqdm

def main_ben(unet_controller, pipe, save_dir, id_prompt, frame_prompt_list, seed, window_length, original_id_prompt=None):
    """
    生成图像的主函数
    :param id_prompt: 用于生成图像的 prompt（包含 new_subject）
    :param original_id_prompt: 用于生成文件名的 id_prompt（包含 subject，如果提供，将重命名文件）
    """
    unet_controller.ipca_index = -1
    unet_controller.ipca_time_step = -1
    # 开始新的图像序列时，清空 kv 存储（避免使用上一组序列的 kv）
    if unet_controller is not None:
        unet_controller.reset_first_frame_kv()
    # Ensure each process uses its own assigned device
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存原始的 id_prompt 用于生成
    generation_id_prompt = id_prompt
    
    # 使用 refined id_prompt 进行生成（refined 版本的 prompts）
    # 注意：modulation scores可视化已在movement_gen_story_slide_windows中处理
    images, story_image = generate_images(
        unet_controller, pipe, generation_id_prompt, frame_prompt_list, 
        save_dir, window_length, seed, verbose=False
    )
    
    # 如果提供了原始 id_prompt，需要重命名文件
    if original_id_prompt is not None and original_id_prompt != generation_id_prompt:
        # 重命名所有生成的图像文件
        # 文件名格式："{id_prompt} {frame_prompt_express}.jpg"
        prefix_to_replace = f'{generation_id_prompt} '
        for image_file in os.listdir(save_dir):
            if image_file.endswith('.jpg') and not image_file.startswith('story_image'):
                # 检查文件名是否以 generation_id_prompt + 空格开头
                if image_file.startswith(prefix_to_replace):
                    # 获取 frame_prompt_express 部分（去掉 id_prompt 前缀和空格）
                    frame_part = image_file[len(prefix_to_replace):]
                    # 构造新的文件名：original_id_prompt + 空格 + frame_part
                    new_filename = f'{original_id_prompt} {frame_part}'
                    old_path = os.path.join(save_dir, image_file)
                    new_path = os.path.join(save_dir, new_filename)
                    if os.path.exists(old_path) and old_path != new_path:
                        os.rename(old_path, new_path)
    
    return images, story_image

def process_instance(unet_controller, pipe, instance):
    # Unpack instance and execute task
    save_dir, id_prompt, frame_prompt_list, seed, window_length, subject_text, original_id_prompt = instance
    # 设置subject_text用于modulation
    unet_controller.subject_text = subject_text
    return main_ben(unet_controller, pipe, save_dir, id_prompt, frame_prompt_list, seed, window_length, original_id_prompt)

def worker(device, unet_controller, pipe, task_queue, pbar):
    # Process tasks until queue is empty
    while not task_queue.empty():
        instance = task_queue.get()
        if instance is None:  # If None is encountered, stop the worker
            break
        # Process the instance
        result = process_instance(unet_controller, pipe, instance)
        # Log the completion
        print(f"Finished processing {instance[1]}")  # Log the processed instance (id_prompt)
        task_queue.task_done()  # Mark the task as done
        pbar.update(1)  # Update the progress bar

def main():
    parser = argparse.ArgumentParser(description="Calculate image similarities using DreamSim or CLIP.")
    parser.add_argument('--device', type=str, choices=['cuda:0', 'cuda:1', 'cuda'], default='cuda')
    parser.add_argument('--save_dir', type=str, default="./result/benchmark_consistory+baseline")
    parser.add_argument('--refined_benchmark_path', type=str, default="resource/consistory+1p1s_fantasy+foods+humans+inanimate_refined.yaml", help='Path to yaml file containing style, subject, new_subject and settings')
    parser.add_argument(
        '--prefix',
        type=str,
        default="",
        help="仅生成 YAML 顶层 key 以该前缀开头的类别；支持逗号分隔多个，例如：foods,fantasy。留空则生成全部类别。",
    )
    parser.add_argument('--model_path', type=str, default='stabilityai/stable-diffusion-xl-base-1.0', help='Path to the model')
    parser.add_argument('--precision', type=str, choices=["fp16", "fp32"], default="fp16", help='Model precision')
    parser.add_argument('--window_length', type=int, default=10, help='Window length for story generation')
    parser.add_argument('--num_gpus', type=int, default=2, help='Number of GPUs to use')
    parser.add_argument('--fix_seed', type=int, default=42, help='-1 for random seed')
    parser.add_argument('--use_new_subject', action='store_true', help='Use new_subject for image generation (filename always uses subject)')
    parser.add_argument('--visualize_modulation_scores', action='store_true', help='Enable visualization of modulation scores')
    parser.add_argument('--modulation_vis_positions', type=str, nargs='+', default=['up1'], 
                        help='UNet positions to visualize (e.g., up1 up0 mid). Options: down0, down1, down2, mid, up0, up1, up2')
    parser.add_argument('--modulation_vis_layer_indices', type=int, nargs='+', default=[2], 
                        help='Layer indices to visualize (e.g., 0 1 2). 0=first Transformer2DModel, 1=second, 2=third, etc.')
    args = parser.parse_args()

    # 检查可用的GPU数量
    if torch.cuda.is_available():
        num_available_gpus = torch.cuda.device_count()
        print(f"检测到 {num_available_gpus} 个GPU")
    else:
        num_available_gpus = 0
        print("警告：未检测到CUDA设备，将使用CPU")
    
    # Create a list of devices
    if args.num_gpus == 1:
        # 如果指定了cuda设备但不可用，回退到cpu
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            print(f"警告：{args.device} 不可用，将使用 CPU")
            devices = ['cpu']
        else:
            devices = [args.device]
    else:
        # 限制使用的GPU数量不超过可用数量
        if num_available_gpus == 0:
            print("警告：没有可用的GPU，将使用CPU")
            devices = ['cpu']
        else:
            actual_num_gpus = min(args.num_gpus, num_available_gpus)
            if actual_num_gpus < args.num_gpus:
                print(f"警告：请求使用 {args.num_gpus} 个GPU，但只有 {num_available_gpus} 个可用。将使用 {actual_num_gpus} 个GPU。")
            devices = [f'cuda:{i}' for i in range(actual_num_gpus)]  # List of device names

    # Load unet_controllers and pipes for each device
    unet_controllers = {}
    pipes = {}
    for device in devices:
        pipe, _, tokenizer = utils.load_pipe_from_path(args.model_path, device, torch.float16 if args.precision == "fp16" else torch.float32, args.precision)
        unet_controller = load_unet_controller(pipe, device)
        unet_controller.Save_story_image = False
        # unet_controller.Prompt_embeds_mode = "svr-eot"
        unet_controller.Prompt_embeds_mode = "single"
        # unet_controller.Is_freeu_enabled = True
        # 优化内存：默认只保存交叉注意力，使用float16
        unet_controller.save_self_attention = False  # 只保存交叉注意力以节省内存
        unet_controller.use_half_precision = True  # 使用float16节省内存
        unet_controller.max_heads_to_save = 4  # 只保存4个头（如果总头数更多）
        # 启用modulation scores可视化（如果指定）
        if args.visualize_modulation_scores:
            unet_controller.Visualize_modulation_scores = True
            unet_controller.Modulation_scores_vis_positions = args.modulation_vis_positions
            unet_controller.Modulation_scores_vis_layer_indices = args.modulation_vis_layer_indices
            print(f"启用modulation scores可视化 - 位置: {args.modulation_vis_positions}, 层索引: {args.modulation_vis_layer_indices}")
        unet_controllers[device] = unet_controller
        pipes[device] = pipe

    # Load the refined benchmark data
    with open(os.path.expanduser(args.refined_benchmark_path), 'r') as file:
        refined_data = yaml.safe_load(file)

    # 可选：按顶层类别前缀过滤（例如 foods,fantasy）
    prefixes = [p.strip() for p in (args.prefix or "").split(",") if p.strip()]
    if prefixes:
        refined_data = {
            k: v for k, v in refined_data.items()
            if any(str(k).startswith(p) for p in prefixes)
        }
        if not refined_data:
            print(f"警告：--prefix={args.prefix!r} 未匹配到任何类别，将不会生成任何实例。")

    instances = []
    for subject_domain, subject_domain_instances in refined_data.items():
        for index, refined_instance in enumerate(subject_domain_instances):
            # 提取字段
            refined_style = refined_instance.get("style", "").strip()
            refined_subject = refined_instance.get("subject", "")
            frame_prompt_list = refined_instance["settings"]
            
            # 根据开关决定是否使用 new_subject 来生成图像
            if args.use_new_subject:
                # 使用 new_subject 来生成图像（如果存在），否则使用 subject
                subject_for_generation = refined_instance.get("new_subject", refined_subject)
            else:
                # 使用 subject 来生成图像
                subject_for_generation = refined_subject
            
            # 构建生成图像的 prompt
            if refined_style:
                generation_id_prompt = f'{refined_style} {subject_for_generation}'
            else:
                generation_id_prompt = subject_for_generation
            
            # 始终使用 subject 来构建文件名的 prompt
            if refined_style:
                filename_id_prompt = f'{refined_style} {refined_subject}'
            else:
                filename_id_prompt = refined_subject
            
            subject_text = refined_subject  # 提取subject部分用于modulation
            
            save_dir = os.path.join(args.save_dir, f"{subject_domain}_{index}")

            # 如果该实例对应的保存目录已经存在且包含已生成的图像，则跳过该实例
            if os.path.exists(save_dir):
                try:
                    existing_files = os.listdir(save_dir)
                    has_images = any(
                        fname.lower().endswith(('.jpg', '.jpeg', '.png'))
                        for fname in existing_files
                    )
                except Exception as e:
                    print(f"检查目录 {save_dir} 时出错：{e}，将继续重新生成该实例。")
                    has_images = False

                if has_images:
                    print(f"检测到目录 {save_dir} 已存在生成图像，跳过该实例。")
                    continue
            if args.fix_seed != -1:
                seed = args.fix_seed
            else:
                import random
                seed = random.randint(0, 2**32 - 1)
            instances.append((save_dir, generation_id_prompt, frame_prompt_list, seed, args.window_length, subject_text, filename_id_prompt))

    # Create a task queue and populate it with instances
    task_queue = queue.Queue()
    for instance in instances:
        task_queue.put(instance)

    # Initialize tqdm progress bar
    pbar = tqdm(total=len(instances))

    # Create threads for each device to process instances
    threads = []
    for device in devices:
        unet_controller = unet_controllers[device]
        pipe = pipes[device]
        thread = threading.Thread(target=worker, args=(device, unet_controller, pipe, task_queue, pbar))
        threads.append(thread)
        thread.start()
        import time
        time.sleep(1)  # Wait for 1 second before starting the next thread

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # Close the progress bar
    pbar.close()

if __name__ == "__main__":
    main()
