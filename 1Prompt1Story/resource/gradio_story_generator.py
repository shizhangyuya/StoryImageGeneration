import os
import sys
# 添加父目录到 sys.path，以便导入 main 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置 HuggingFace 镜像地址（必须在导入其他库之前设置）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"

import gradio as gr
import diffusers
import random
import torch
from PIL import Image
import numpy as np
from datetime import datetime
import json
import re

diffusers.utils.logging.set_verbosity_error()

from unet.unet_controller import UNetController
from main import load_unet_controller
from unet import utils

# 全局变量
interrupt_flag = False
pipe = None
unet_controller = None

# 尝试导入 openai，如果没有安装则提示
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("警告: openai 库未安装，请使用 'pip install openai' 安装")


def generate_prompts_with_chatgpt(story_description, num_images, api_key, base_url=None):
    """
    使用 ChatGPT API 根据故事描述生成行为 prompt 列表
    
    Args:
        story_description: 故事描述，如"生成火柴人的故事"
        num_images: 需要生成的图像数量
        api_key: OpenAI API key
        base_url: API base URL (可选，用于自定义 API 端点)
    
    Returns:
        list: 行为 prompt 列表，如 ["[火柴人]打羽毛球", "[火柴人]跳舞"]
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("openai 库未安装，请使用 'pip install openai' 安装")
    
    if not api_key:
        raise ValueError("请提供 OpenAI API Key")
    
    try:
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        
        prompt = f"""根据以下故事描述，生成 {num_images} 个图像行为描述，每个描述应该简洁明了，适合用于图像生成。

故事描述：{story_description}

要求：
1. 生成 {num_images} 个不同的行为描述
2. 每个描述应该是一个具体的动作或场景
3. 如果故事描述中提到了特定主体（如"火柴人"），请在每个描述中使用相同的格式（如 [火柴人] 或 【火柴人】）
4. 描述应该连贯，形成一个完整的故事序列
5. 只返回行为描述列表，每行一个，不要添加编号或其他文字

请直接返回 {num_images} 个行为描述，每行一个："""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 或使用 "gpt-4" 如果需要更好的效果
            messages=[
                {"role": "system", "content": "你是一个专业的图像生成提示词助手，擅长将故事描述转换为具体的图像行为描述。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        # 提取生成的文本
        generated_text = response.choices[0].message.content.strip()
        
        # 解析生成的行为描述列表
        prompts = []
        for line in generated_text.split('\n'):
            line = line.strip()
            # 移除可能的编号（如 "1. ", "1)", "- " 等）
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            line = re.sub(r'^-\s*', '', line)
            line = re.sub(r'^\*\s*', '', line)
            if line and len(line) > 0:
                prompts.append(line)
        
        # 如果生成的 prompt 数量不够，补充一些
        if len(prompts) < num_images:
            print(f"警告: 只生成了 {len(prompts)} 个 prompt，需要 {num_images} 个")
            # 可以尝试再次生成或使用默认值
            while len(prompts) < num_images:
                prompts.append(f"{story_description} - 场景 {len(prompts) + 1}")
        elif len(prompts) > num_images:
            prompts = prompts[:num_images]
        
        return prompts
    
    except Exception as e:
        raise Exception(f"调用 ChatGPT API 时出错: {str(e)}")


def generate_story_images(
    story_description,
    num_images,
    api_key,
    model_path,
    precision,
    seed,
    window_length,
    device,
    base_url=None
):
    """
    生成故事图像的主函数（支持实时进度更新）
    
    Args:
        story_description: 故事描述
        num_images: 图像数量
        api_key: OpenAI API key
        model_path: 模型路径
        precision: 精度 (fp16/fp32)
        seed: 随机种子
        window_length: 窗口长度
        device: 设备 (cuda:0, cuda:1, cpu)
        base_url: API base URL (可选)
    
    Yields:
        tuple: (图像列表, 进度文本)
    """
    global interrupt_flag, pipe, unet_controller
    
    interrupt_flag = False
    result_images = []
    result_data = []  # 存储 (image, prompt) 元组列表
    
    try:
        # 步骤1: 生成行为 prompt 列表
        yield [], f"正在使用 ChatGPT 生成 {num_images} 个行为描述..."
        
        try:
            frame_prompt_list = generate_prompts_with_chatgpt(
                story_description, num_images, api_key, base_url
            )
        except Exception as e:
            error_msg = f"生成 prompt 失败: {str(e)}"
            yield [], error_msg
            return
        
        prompt_list_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(frame_prompt_list)])
        yield [], f"已生成 {len(frame_prompt_list)} 个行为描述:\n\n{prompt_list_text}\n\n开始加载模型..."
        
        # 步骤2: 加载模型和控制器
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        
        # 检查设备是否可用
        if device.startswith('cuda') and not torch.cuda.is_available():
            device = 'cpu'
            yield [], f"警告: CUDA 不可用，使用 CPU"
        
        pipe, _, tokenizer = utils.load_pipe_from_path(
            model_path,
            device,
            torch.float16 if precision == "fp16" else torch.float32,
            precision
        )
        
        if interrupt_flag:
            yield [], "生成已中断"
            return
        
        unet_controller = load_unet_controller(pipe, device)
        unet_controller.save_self_attention = False
        unet_controller.use_half_precision = True
        unet_controller.max_heads_to_save = 4
        unet_controller.Prompt_embeds_mode = "single"
        unet_controller.Save_story_image = False
        
        # 创建保存目录
        current_time = datetime.now().strftime("%Y%m%d%H")
        current_time_ = datetime.now().strftime("%M%S")
        save_dir = os.path.join(".", f'result/{current_time}/{current_time_}_story_seed{seed}')
        os.makedirs(save_dir, exist_ok=True)
        
        # 使用故事描述作为 id_prompt
        id_prompt = story_description
        
        # 步骤3: 准备生成参数
        generate = torch.Generator().manual_seed(seed)
        unet_controller.id_prompt = id_prompt
        
        # 处理 ipca
        if unet_controller.Use_ipca is True:
            unet_controller.Store_qkv = True
            original_prompt_embeds_mode = unet_controller.Prompt_embeds_mode
            unet_controller.Prompt_embeds_mode = "original"
            _ = pipe(id_prompt, generator=generate, unet_controller=unet_controller).images
            unet_controller.Prompt_embeds_mode = original_prompt_embeds_mode
        
        unet_controller.Store_qkv = False
        unet_controller.reset_first_frame_kv()
        
        # 计算窗口
        max_window_length = utils.get_max_window_length(unet_controller, id_prompt, frame_prompt_list)
        window_length = min(window_length, max_window_length)
        if window_length < len(frame_prompt_list):
            movement_lists = utils.circular_sliding_windows(frame_prompt_list, window_length)
        else:
            movement_lists = [movement for movement in frame_prompt_list]
        
        # 步骤4: 逐张生成图像（实时更新）
        yield [], f"开始生成图像... (共 {len(frame_prompt_list)} 张)"
        
        for index, movement in enumerate(frame_prompt_list):
            if interrupt_flag:
                yield result_data, "生成已中断"
                return
            
            # 设置当前帧索引
            if unet_controller is not None:
                unet_controller.current_frame_index = index
                unet_controller.reset_attn_store(clear_modulation_scores=True)
                if index == 0:
                    unet_controller.reset_first_frame_kv()
            
            # 准备生成 prompt
            if unet_controller is not None:
                if window_length < len(frame_prompt_list):
                    unet_controller.frame_prompt_suppress = movement_lists[index][1:]
                    unet_controller.frame_prompt_express = movement_lists[index][0]
                    gen_propmts = [f'{id_prompt} {movement_lists[index][0]}']
                else:
                    unet_controller.frame_prompt_suppress = movement_lists[:index] + movement_lists[index+1:]
                    unet_controller.frame_prompt_express = movement_lists[index]
                    gen_propmts = [f'{id_prompt} {movement_lists[index]}']
            else:
                gen_propmts = f'{id_prompt} {movement}'
            
            # 生成图像
            if unet_controller is not None and unet_controller.Use_same_init_noise is True:
                generate = torch.Generator().manual_seed(seed)
            
            images = pipe(gen_propmts, generator=generate, unet_controller=unet_controller).images
            generated_image = images[0]
            
            # 保存图像
            if unet_controller is not None:
                frame_prompt_express = unet_controller.frame_prompt_express
            else:
                frame_prompt_express = movement
            generated_image.save(os.path.join(save_dir, f'{id_prompt} {frame_prompt_express}.jpg'))
            
            # 添加到结果列表
            result_images.append(generated_image)
            result_data.append((generated_image, frame_prompt_express))
            
            # 实时更新进度
            progress_msg = f"已完成 {index + 1}/{len(frame_prompt_list)} 张图像\n\n当前生成: {frame_prompt_express}\n\n已生成的行为描述:\n" + "\n".join([f"{i+1}. {p}" for i, p in enumerate(frame_prompt_list[:index+1])])
            yield result_data.copy(), progress_msg
        
        # 最终结果
        final_msg = f"✅ 全部完成！共生成 {len(result_images)} 张图像。\n\n所有行为描述:\n" + "\n".join([f"{i+1}. {p}" for i, p in enumerate(frame_prompt_list)])
        yield result_data, final_msg
        
        # 清理
        import gc
        del pipe
        del unet_controller
        pipe = None
        unet_controller = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        error_msg = f"❌ 生成过程中出错: {str(e)}\n\n详细错误:\n{error_trace}"
        print(f"错误详情: {error_trace}")  # 在服务器端打印完整错误
        yield result_data, error_msg




def gradio_interface():
    """创建 Gradio 界面"""
    global interrupt_flag
    
    # 启用 Gradio 的详细错误显示
    import logging
    logging.basicConfig(level=logging.INFO)
    
    with gr.Blocks(title="故事图像生成器") as demo:
        gr.Markdown("""
        # 故事图像生成器
        
        根据故事描述自动生成图像序列。
        
        1. 输入故事描述（如"生成火柴人的故事"）
        2. 设置要生成的图像数量
        3. 系统会自动调用 ChatGPT 生成行为描述
        4. 根据行为描述生成图像并展示
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # 输入区域
                story_description = gr.Textbox(
                    label="故事描述",
                    placeholder="例如：生成火柴人的故事",
                    value="生成火柴人的故事",
                    lines=2
                )
                
                num_images = gr.Slider(
                    label="图像数量",
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1
                )
                
                api_key = gr.Textbox(
                    label="OpenAI API Key",
                    type="password",
                    placeholder="输入你的 OpenAI API Key",
                    info="需要 OpenAI API Key 来生成行为描述",
                    value=""
                )
                
                base_url = gr.Textbox(
                    label="API Base URL (可选)",
                    placeholder="留空使用默认 OpenAI API，或输入自定义 API 地址",
                    value="",
                    info="例如: https://api.openai.com/v1",
                    visible=False  # 默认隐藏，减少界面复杂度
                )
                
                with gr.Row():
                    device = gr.Dropdown(
                        label="设备",
                        choices=["cuda:0", "cuda:1", "cpu"],
                        value="cuda:0" if torch.cuda.is_available() else "cpu"
                    )
                    
                    model_path = gr.Dropdown(
                        label="模型路径",
                        choices=[
                            "playgroundai/playground-v2.5-1024px-aesthetic",
                            "stabilityai/stable-diffusion-xl-base-1.0",
                            "RunDiffusion/Juggernaut-X-v10",
                            "SG161222/RealVisXL_V4.0"
                        ],
                        value="playgroundai/playground-v2.5-1024px-aesthetic",
                        allow_custom_value=True
                    )
                
                with gr.Row():
                    precision = gr.Dropdown(
                        label="精度",
                        choices=["fp16", "fp32"],
                        value="fp16"
                    )
                    
                    seed = gr.Number(
                        label="随机种子 (-1 为随机)",
                        value=-1,
                        precision=0
                    )
                    
                    window_length = gr.Slider(
                        label="窗口长度",
                        minimum=1,
                        maximum=20,
                        value=10,
                        step=1
                    )
                
                generate_button = gr.Button("生成图像", variant="primary", size="lg")
                interrupt_button = gr.Button("中断生成", variant="stop")
            
            with gr.Column(scale=2):
                # 输出区域
                progress_text = gr.Textbox(
                    label="生成进度",
                    lines=10,
                    interactive=False
                )
                
                output_gallery = gr.Gallery(
                    label="生成的图像（点击查看大图，下方显示对应的 prompt）",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    rows=2,
                    height="auto",
                    show_share_button=False
                )
        
        # 中断功能
        def interrupt_generation():
            global interrupt_flag
            interrupt_flag = True
            return "生成已中断"
        
        interrupt_button.click(
            fn=interrupt_generation,
            inputs=[],
            outputs=[progress_text]
        )
        
        # 生成功能
        def generate_wrapper(*args):
            try:
                # 解包参数
                story_desc, num_img, api_key_val, base_url_val, model_path_val, precision_val, seed_val, window_len, device_val = args
                
                # 参数验证
                if not story_desc or not story_desc.strip():
                    yield [], "错误: 请输入故事描述"
                    return
                
                if num_img is None or num_img < 1 or num_img > 20:
                    yield [], "错误: 图像数量必须在 1-20 之间"
                    return
                
                if not api_key_val or not api_key_val.strip():
                    yield [], "错误: 请输入 OpenAI API Key"
                    return
                
                if not model_path_val or not model_path_val.strip():
                    yield [], "错误: 请选择模型路径"
                    return
                
                # 处理空值
                if base_url_val is None:
                    base_url_val = ""
                if device_val is None:
                    device_val = "cuda:0" if torch.cuda.is_available() else "cpu"
                if precision_val is None:
                    precision_val = "fp16"
                if seed_val is None:
                    seed_val = -1
                if window_len is None:
                    window_len = 10
                
                # 调用生成函数
                images_data = []
                current_progress = ""
                
                for images_data_new, progress in generate_story_images(
                    story_desc, num_img, api_key_val, model_path_val,
                    precision_val, seed_val, window_len, device_val, base_url_val if base_url_val else None
                ):
                    if images_data_new is not None:
                        images_data = images_data_new
                    if progress:
                        current_progress = progress
                    yield images_data, current_progress
                    
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                error_msg = f"发生错误: {str(e)}\n\n详细错误信息:\n{error_trace}"
                print(error_msg)  # 在服务器端打印
                yield [], f"❌ 错误: {str(e)}\n\n请检查:\n1. API Key 是否正确\n2. 模型路径是否正确\n3. 设备是否可用\n4. 网络连接是否正常"
        
        generate_button.click(
            fn=generate_wrapper,
            inputs=[
                story_description,
                num_images,
                api_key,
                base_url,
                model_path,
                precision,
                seed,
                window_length,
                device
            ],
            outputs=[output_gallery, progress_text]
        )
        
        gr.Markdown("""
        ### 使用说明
        
        1. **故事描述**: 输入你想要生成的故事主题，例如"生成火柴人的故事"
        2. **图像数量**: 设置要生成的图像数量（1-20）
        3. **API Key**: 输入你的 OpenAI API Key（用于生成行为描述）
        4. **设备**: 选择使用的计算设备（GPU 或 CPU）
        5. **模型**: 选择要使用的图像生成模型
        6. 点击"生成图像"按钮开始生成
        
        **注意**: 
        - 首次运行需要下载模型，可能需要一些时间
        - 图像生成过程可能需要几分钟，请耐心等待
        - 如果遇到 API 错误，请检查 API Key 是否正确
        """)
    
    return demo


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="启动故事图像生成器 Gradio 界面")
    parser.add_argument('--port', type=int, default=7860, help='服务器端口号')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器主机地址 (0.0.0.0 表示监听所有接口)')
    parser.add_argument('--share', action='store_true', help='使用 Gradio 的公共链接（需要网络连接）')
    parser.add_argument('--server-name', type=str, default=None, help='服务器名称（默认自动检测）')
    args = parser.parse_args()
    
    demo = gradio_interface()
    
    # 如果 host 是 0.0.0.0，尝试使用 None 让 Gradio 自动处理
    # 或者明确设置为 0.0.0.0
    server_name = args.server_name if args.server_name is not None else args.host
    
    print(f"\n{'='*60}")
    print(f"Gradio 界面启动中...")
    print(f"监听地址: {server_name}:{args.port}")
    
    if args.share:
        print(f"公共链接: 将在启动后显示")
        print(f"注意: 使用公共链接时，数据会经过 Gradio 服务器")
    else:
        print(f"\n如果使用 SSH 端口转发，请在本地终端运行:")
        print(f"  ssh -p 42050 -L {args.port}:localhost:{args.port} root@connect.westd.seetacloud.com")
        print(f"然后在浏览器访问: http://localhost:{args.port}")
        print(f"\n如果无法访问，可以尝试使用 --share 参数创建公共链接")
    
    print(f"{'='*60}\n")
    
    try:
        demo.launch(
            share=args.share, 
            server_name=server_name, 
            server_port=args.port,
            show_error=True
        )
    except ValueError as e:
        if "shareable link must be created" in str(e):
            print(f"\n{'='*60}")
            print("错误: Gradio 检测到 localhost 不可访问")
            print("解决方案:")
            print("1. 使用 --share 参数创建公共链接:")
            print(f"   python resource/gradio_story_generator.py --share")
            print("2. 或者使用 SSH 端口转发（推荐）")
            print(f"{'='*60}\n")
            # 自动尝试使用 share=True
            print("自动尝试使用公共链接模式...")
            demo.launch(
                share=True,
                server_name=server_name,
                server_port=args.port,
                show_error=True
            )
        else:
            raise

