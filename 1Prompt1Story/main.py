import os
# 设置 HuggingFace 镜像地址（必须在导入 diffusers 和 transformers 之前设置）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"

import torch
import random
import diffusers
import torch.utils
import unet.utils as utils
from unet.unet_controller import UNetController
import argparse
from datetime import datetime

#yu
# 注意：注意力可视化已移至utils.py中，每生成完一张图自动保存

diffusers.utils.logging.set_verbosity_error()

def load_unet_controller(pipe, device):
    unet_controller = UNetController()
    unet_controller.device = device
    unet_controller.tokenizer = pipe.tokenizer

    return unet_controller


def generate_images(unet_controller: UNetController, pipe, id_prompt, frame_prompt_list, save_dir, window_length, seed, verbose=True):
    generate = torch.Generator().manual_seed(seed)
    # 设置id_prompt到unet_controller（用于ID replacement功能）
    if unet_controller is not None:
        unet_controller.id_prompt = id_prompt
    # 开始新的图像序列时，清空 kv 存储（避免使用上一组序列的 kv）
    if unet_controller is not None:
        unet_controller.reset_first_frame_kv()
    if unet_controller.Use_ipca is True:
        unet_controller.Store_qkv = True
        original_prompt_embeds_mode = unet_controller.Prompt_embeds_mode
        unet_controller.Prompt_embeds_mode = "original"
        _ = pipe(id_prompt, generator=generate, unet_controller=unet_controller).images

        unet_controller.Prompt_embeds_mode = original_prompt_embeds_mode


    unet_controller.Store_qkv = False
    images, story_image = utils.movement_gen_story_slide_windows(
        id_prompt, frame_prompt_list, pipe, window_length, seed, unet_controller, save_dir, verbose=verbose
    )

    return images, story_image


def main(device, model_path, save_dir, id_prompt, frame_prompt_list, precision, seed, window_length):
    pipe, _ ,tokenizer= utils.load_pipe_from_path(model_path, device, torch.float16 if precision == "fp16" else torch.float32, precision)
    
    unet_controller = load_unet_controller(pipe, device)
    # 优化内存：默认只保存交叉注意力，使用float16
    unet_controller.save_self_attention = False  # 只保存交叉注意力以节省内存
    unet_controller.use_half_precision = True  # 使用float16节省内存
    unet_controller.max_heads_to_save = 4  # 只保存4个头（如果总头数更多）
    # 注意：三个独立开关已在unet_controller中定义，使用默认值
    # 如需修改，可以在这里设置：
    # unet_controller.Use_kv_interpolation = True  # 启用KV插值（会自动保存第一张图的kv）
    # unet_controller.Visualize_attn_map = True  # 启用attention map可视化
          
    images, story_image = generate_images(unet_controller, pipe, id_prompt, frame_prompt_list, save_dir, window_length, seed)

    # 注意：交叉注意力可视化已移至utils.py中，每生成完一张图自动保存
    # 可视化结果保存在 save_dir/cross_attn_vis/frame_XXX/ 目录下

    return images, story_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using a specific device.")
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for computation (e.g., cuda:0, cpu)')
    # parser.add_argument('--model_path', type=str, default='./model/models--playgroundai--playground-v2.5-1024px-aesthetic/snapshots/1e032f13f2fe6db2dc49947dbdbd196e753de573',help='Path to the model')
    parser.add_argument('--model_path', type=str, default='playgroundai/playground-v2.5-1024px-aesthetic', help='Path to the model')
    parser.add_argument('--project_base_path', type=str, default='./results', help='Path to save the generated images')
    parser.add_argument('--id_prompt', type=str, default="A photo of a red fox with coat", help='Initial prompt for image generation')
    parser.add_argument('--frame_prompt_list', type=str, nargs='+', default=[
        "wearing a scarf in a meadow",
        "playing in the snow",
        "at the edge of a village with river",
    ], help='List of frame prompts')
    parser.add_argument('--precision', type=str, choices=["fp16", "fp32"], default="fp16", help='Model precision')
    parser.add_argument('--seed', type=int, default=421, help='Random seed for generation')
    parser.add_argument('--window_length', type=int, default=10, help='Window length for story generation')
    parser.add_argument('--save_padding', type=str, default='test', help='Padding for save directory')
    parser.add_argument('--random_seed', action='store_true', help='Use random seed')
    parser.add_argument('--json_path', type=str,)

    # -------------------------- Debug专用：传入参数列表 --------------------------
    # debug_args_list = [
    #     '--device', 'cuda:0',
    #     '--model_path', 'model/playground-v2.5-1024px-aesthetic',
    #     '--project_base_path', './results',
    #     '--id_prompt', 'A photo of a red fox with coat',
    #     '--frame_prompt_list', 'wearing a scarf in a meadow', 'playing in the snow',
    #     'at the edge of a village with river',
    #     '--precision', 'fp16',
    #     '--seed', '420',
    #     '--window_length', '10',
    #     '--save_padding', 'test',
    #     # '--json_path', './debug_config.json'
    #     # 若需启用随机种子，添加下面一行
    #     '--random_seed'
    # ]
    # args = parser.parse_args(debug_args_list)

    args = parser.parse_args()
    if args.random_seed:
        args.seed = random.randint(0, 1000000)

    current_time = datetime.now().strftime("%Y%m%d%H")
    current_time_ = datetime.now().strftime("%M%S")
    save_dir = os.path.join(args.project_base_path, f'result/{current_time}/{current_time_}_{args.save_padding}_seed{args.seed}')
    os.makedirs(save_dir, exist_ok=True)

    if args.json_path is None:
        main(args.device, args.model_path, save_dir, args.id_prompt, args.frame_prompt_list, args.precision, args.seed, args.window_length)
    else:
        import json
        with open(args.json_path, "r") as file:
            data = json.load(file)

        combinations = data["combinations"]

        for combo in combinations:
            main(args.device, args.model_path, save_dir, combo['id_prompt'], combo['frame_prompt_list'], args.precision, args.seed, args.window_length)
