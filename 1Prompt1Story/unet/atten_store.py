import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import datetime
from typing import List, Optional
import torch
import os
from .unet_controller import UNetController

# 全局设置：提升可视化速度
plt.switch_backend('Agg')
plt.rcParams['font.size'] = 8
plt.rcParams['figure.max_open_warning'] = 0


def aggregate_attn_from_controller(
        unet_controller: UNetController,
        res: int,
        from_where: List[str],
        is_cross: bool,
        select: int = 0
) -> torch.Tensor:
    """
    从unet_controller聚合注意力权重（映射为2D图像尺寸）
    注意：现在存储的tensor已经是降维后的，形状为 (res, res, num_tokens) 或 (res*res, num_tokens)
    :param unet_controller: UNet控制器（存储了注意力权重）
    :param res: 图像分辨率（如32，对应32x32分块）
    :param from_where: 选择UNet的层（["down", "mid", "up"]或["down0", "down1", "down2", "mid", "up0", "up1", "up2"]）
    :param is_cross: 是否聚合交叉注意力
    :param select: 选择第几个样本（批量推理时，现在已聚合，通常为0）
    :return: 聚合后的注意力张量，形状为 (res, res, num_tokens) 或 (res, res, res*res)
    """
    attn_maps = unet_controller.get_average_attn()
    num_pixels = res ** 2
    aggregated_out = []

    # 遍历指定的UNet层
    for layer_pos in from_where:
        # 支持两种格式：["down", "mid", "up"] 或 ["down0", "down1", "down2", "mid", "up0", "up1", "up2"]
        # 如果layer_pos是"down"、"mid"或"up"，则匹配所有以该前缀开头的key
        if layer_pos in ["down", "mid", "up"]:
            # 匹配所有以layer_pos开头的key
            matching_keys = [k for k in attn_maps.keys() if k.startswith(layer_pos) and k.endswith(f"_{'cross' if is_cross else 'self'}")]
        else:
            # 精确匹配（需要映射到前缀）
            if layer_pos.startswith("down"):
                prefix = "down"
            elif layer_pos.startswith("up"):
                prefix = "up"
            elif layer_pos.startswith("mid"):
                prefix = "mid"
            else:
                prefix = layer_pos
            attn_key = f"{prefix}_{'cross' if is_cross else 'self'}"
            matching_keys = [attn_key] if attn_key in attn_maps else []
        
        for attn_key in matching_keys:
            if attn_key not in attn_maps:
                continue

            # 遍历该层下的所有注意力权重（已经是降维后的）
            for attn_tensor in attn_maps[attn_key]:
                # 现在tensor已经是聚合后的，形状可能是：
                # - (res, res, num_tokens) - 交叉注意力，已reshape为2D空间
                # - (num_pixels, num_tokens) - 交叉注意力，未reshape
                # - (num_pixels, num_pixels) - 自注意力
                
                if attn_tensor.dim() == 3:
                    # 3维：(res, res, num_tokens) - 交叉注意力，已reshape
                    if attn_tensor.shape[0] == res and attn_tensor.shape[1] == res:
                        aggregated_out.append(attn_tensor)
                elif attn_tensor.dim() == 2:
                    # 2维：(seq_len_q, seq_len_k)
                    if is_cross:
                        # 交叉注意力：(num_pixels, num_tokens) -> (res, res, num_tokens)
                        if attn_tensor.shape[0] == num_pixels:
                            aggregated_out.append(attn_tensor.view(res, res, -1))
                    else:
                        # 自注意力：(num_pixels, num_pixels) - 保持原样或reshape
                        if attn_tensor.shape[0] == num_pixels and attn_tensor.shape[1] == num_pixels:
                            aggregated_out.append(attn_tensor)

    # 聚合所有层的注意力
    if not aggregated_out:
        # 如果没找到匹配的注意力，返回None而不是抛出异常
        return None

    # 对所有层的注意力取均值
    # 对于交叉注意力，形状应该是 (res, res, num_tokens)
    if is_cross:
        # 确保所有tensor形状一致
        target_shape = aggregated_out[0].shape
        if all(t.shape == target_shape for t in aggregated_out):
            aggregated_tensor = torch.stack(aggregated_out, dim=0).mean(dim=0)
        else:
            # 如果形状不一致，取第一个
            aggregated_tensor = aggregated_out[0]
    else:
        # 自注意力：直接取第一个（或可以取均值）
        aggregated_tensor = aggregated_out[0] if len(aggregated_out) == 1 else torch.stack(aggregated_out, dim=0).mean(dim=0)
    
    return aggregated_tensor


def visualize_cross_attn_from_controller(
        tokenizer,
        prompt: str,
        unet_controller: UNetController,
        res: int,
        from_where: List[str] = ["down", "mid", "up"],
        select: int = 0,
        save_dir: str = "./cross_attn_vis",
        save_filename: str = "cross_attn_final.png",
        colormap: str = "hot",
        show_fig: bool = False
) -> List[np.ndarray]:
    """
    从unet_controller可视化交叉注意力（图像-文本关联）
    :param tokenizer: 文本tokenizer
    :param prompt: 生成图像的文本prompt
    :param unet_controller: UNet控制器
    :param res: 图像分块分辨率（如32）
    :param from_where: 选择UNet的层
    :param select: 选择第几个样本
    :param save_dir: 保存目录
    :param save_filename: 保存文件名
    :param colormap: 配色方案（"hot", "white", "gray"等）
    :param show_fig: 是否显示图像
    :return: 注意力可视化图像列表
    """
    # 1. 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 2. 编码文本token
    tokens = tokenizer.encode(prompt)
    token_decoder = tokenizer.decode
    if len(tokens) == 0:
        raise ValueError("prompt编码后无有效token")

    # 3. 聚合交叉注意力
    cross_attn_maps = aggregate_attn_from_controller(
        unet_controller=unet_controller,
        res=res,
        from_where=from_where,
        is_cross=True,
        select=select
    )

    # 检查是否成功聚合注意力
    if cross_attn_maps is None:
        print(f"警告：未找到匹配的交叉注意力权重（res={res}, from_where={from_where}）")
        return []

    # 4. 处理注意力维度
    # 现在cross_attn_maps的形状应该是 (res, res, num_tokens)
    if cross_attn_maps.dim() == 2:
        # 如果是2D (num_pixels, num_tokens)，reshape为3D
        if cross_attn_maps.shape[0] == res ** 2:
            cross_attn_maps = cross_attn_maps.view(res, res, -1)
    elif cross_attn_maps.dim() != 3:
        raise ValueError(f"意外的注意力维度：{cross_attn_maps.shape}，期望 (res, res, num_tokens)")

    # 5. 可视化每个token的注意力图
    attn_vis_list = []
    
    # 限制token数量，避免图像过大
    num_tokens = cross_attn_maps.shape[-1]
    max_tokens_to_show = min(num_tokens, len(tokens), 20)  # 最多显示20个token
    
    token_images = []
    for token_idx in range(max_tokens_to_show):
        # 提取单个token对应的注意力图
        # cross_attn_maps shape: (res, res, num_tokens)
        attn_map = cross_attn_maps[:, :, token_idx]

        # 归一化到[0, 1]（最后输出时归一化）
        attn_map_np = attn_map.float().numpy()
        if attn_map_np.max() > attn_map_np.min():
            attn_map_np = (attn_map_np - attn_map_np.min()) / (attn_map_np.max() - attn_map_np.min() + 1e-8)
        else:
            attn_map_np = np.zeros_like(attn_map_np)
        
        # 转换为0-255的uint8
        attn_map_np = (attn_map_np * 255).astype(np.uint8)

        # 应用colormap增强可视化效果
        try:
            import matplotlib.cm as cm
            if colormap == "white":
                # 白色配色：高注意力区域为白色，低注意力区域为黑色
                # attn_map_np已经是0-255的值，高注意力=高值=白色
                # 直接使用灰度值，白色背景
                attn_map_colored = np.repeat(np.expand_dims(attn_map_np, axis=-1), 3, axis=-1)
            elif colormap == "gray":
                # 灰度配色
                attn_map_colored = np.repeat(np.expand_dims(attn_map_np, axis=-1), 3, axis=-1)
            else:
                # 其他colormap（如hot）
                cmap = cm.get_cmap(colormap if colormap != "white" else "hot")
                attn_map_colored = (cmap(attn_map_np / 255.0)[:, :, :3] * 255).astype(np.uint8)
        except:
            # 如果colormap不可用，使用灰度图
            attn_map_colored = np.repeat(np.expand_dims(attn_map_np, axis=-1), 3, axis=-1)

        # 缩放为256x256
        attn_img = Image.fromarray(attn_map_colored).resize((256, 256), Image.BILINEAR)
        attn_img = np.array(attn_img)

        # 在图像下方添加token文本
        if token_idx < len(tokens):
            token_text = token_decoder(int(tokens[token_idx])).replace("<|endoftext|>", "").replace("<|startoftext|>", "")
            if not token_text or token_text.strip() == "":
                token_text = f"token_{token_idx}"
        else:
            token_text = f"token_{token_idx}"
        
        text_img = _add_text_under_image(attn_img, token_text)
        token_images.append(text_img)

    # 拼接所有token的注意力图
    if token_images:
        final_attn_img = np.concatenate(token_images, axis=1)
        attn_vis_list.append(final_attn_img)

        # 保存注意力图
        final_save_path = os.path.join(save_dir, save_filename)
        plt.imsave(final_save_path, final_attn_img)

    # 6. 保存结果（现在只保存一张聚合后的图）
    if attn_vis_list:
        print(f"交叉注意力可视化完成，保存至：{save_dir}")

    # 7. 显示图像（可选）
    if show_fig and attn_vis_list:
        plt.figure(figsize=(16, 4))
        plt.imshow(attn_vis_list[-1])
        plt.axis("off")
        plt.title("Cross Attention")
        plt.show()

    return attn_vis_list


def visualize_self_attn_from_controller(
        unet_controller: UNetController,
        res: int,
        from_where: List[str] = ["down", "mid", "up"],
        select: int = 0,
        max_components: int = 10,
        save_path: str = "./self_attn_vis.png",
        show_fig: bool = False
):
    """
    从unet_controller可视化自注意力（图像内部关联，基于SVD分解）
    :param unet_controller: UNet控制器
    :param res: 图像分块分辨率（如32）
    :param from_where: 选择UNet的层
    :param select: 选择第几个样本
    :param max_components: SVD分解的最大成分数
    :param save_path: 保存路径
    :param show_fig: 是否显示图像
    """
    # 1. 聚合自注意力
    self_attn_maps = aggregate_attn_from_controller(
        unet_controller=unet_controller,
        res=res,
        from_where=from_where,
        is_cross=False,
        select=select
    )

    # 检查是否成功聚合注意力
    if self_attn_maps is None:
        print(f"警告：未找到匹配的自注意力权重（res={res}, from_where={from_where}）")
        return None

    # 2. 处理自注意力矩阵
    # 现在self_attn_maps应该是 (res*res, res*res) 的形状（已降维）
    if self_attn_maps.dim() == 2:
        # 2维：(num_pixels, num_pixels)
        if self_attn_maps.shape[0] == res ** 2 and self_attn_maps.shape[1] == res ** 2:
            self_attn_flat = self_attn_maps.float().numpy()
        else:
            raise ValueError(f"自注意力形状不匹配：{self_attn_maps.shape}，期望 ({res**2}, {res**2})")
    elif self_attn_maps.dim() == 1:
        # 如果是1维，可能是对角线或其他形式，reshape为2D
        self_attn_flat = self_attn_maps.float().numpy().reshape((res ** 2, res ** 2))
    else:
        # 其他维度，尝试reshape
        self_attn_flat = self_attn_maps.float().numpy().reshape((res ** 2, res ** 2))

    # 3. SVD分解提取核心成分
    self_attn_flat = self_attn_flat - np.mean(self_attn_flat, axis=1, keepdims=True)
    u, s, vh = np.linalg.svd(self_attn_flat)

    # 4. 渲染核心成分
    component_images = []
    for comp_idx in range(min(max_components, vh.shape[0])):
        # 重塑为2D图像
        comp_map = vh[comp_idx].reshape(res, res)
        # 归一化
        comp_map = comp_map - comp_map.min()
        comp_map = 255 * comp_map / (comp_map.max() + 1e-8)
        comp_map = comp_map.astype(np.uint8)
        # 转为RGB并缩放
        comp_map_rgb = np.repeat(np.expand_dims(comp_map, axis=-1), 3, axis=-1)
        comp_img = Image.fromarray(comp_map_rgb).resize((256, 256), Image.BILINEAR)
        component_images.append(np.array(comp_img))

    # 5. 拼接并保存
    self_attn_img = np.concatenate(component_images, axis=1)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.imsave(save_path, self_attn_img)

    # 6. 显示图像（可选）
    if show_fig:
        plt.figure(figsize=(16, 4))
        plt.imshow(self_attn_img)
        plt.axis("off")
        plt.title(f"Self Attention Components (Top {max_components})")
        plt.show()

    print(f"自注意力可视化完成，保存至：{save_path}")
    return self_attn_img


# -------------------------- 辅助函数 --------------------------
def _add_text_under_image(image: np.ndarray, text: str, text_height: int = 40) -> np.ndarray:
    """在图像下方添加文本（使用PIL绘制）"""
    from PIL import Image, ImageDraw, ImageFont
    h, w, c = image.shape
    # 创建文本背景
    text_bg = np.ones((text_height, w, c), dtype=np.uint8) * 255
    # 拼接图像和文本背景
    combined = np.concatenate([image, text_bg], axis=0)
    
    # 使用PIL绘制文本
    pil_img = Image.fromarray(combined)
    draw = ImageDraw.Draw(pil_img)
    
    # 尝试使用默认字体，如果失败则使用基本字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except:
            font = ImageFont.load_default()
    
    # 计算文本位置（居中）
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = (w - text_width) // 2
    text_y = h + (text_height - (bbox[3] - bbox[1])) // 2
    
    # 绘制文本
    draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
    
    return np.array(pil_img)


def _save_images_as_gif(images: List[np.ndarray], save_path: str, duration: int = 500):
    """将图像列表保存为GIF"""
    pil_images = [Image.fromarray(img) for img in images]
    pil_images[0].save(
        save_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=0
    )