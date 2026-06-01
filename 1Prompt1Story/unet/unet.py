# modified from the https://github.com/cloneofsimo/minSDXL


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin
from typing import Optional

from unet.unet_controller import UNetController
import unet.utils as utils
# SDXL


def create_rectangular_mask(seq_len_q, coords, device, dtype):
    """
    根据坐标创建矩形mask
    :param seq_len_q: query序列长度（空间维度）
    :param coords: [left, right, top, bottom]，0-1范围
    :param device: 设备
    :param dtype: 数据类型
    :return: mask张量，形状为(seq_len_q,)，mask>0表示在矩形区域内
    """
    left, right, top, bottom = coords
    
    # 尝试推断空间分辨率（假设是正方形）
    spatial_size = int(math.sqrt(seq_len_q))
    if spatial_size * spatial_size != seq_len_q:
        # 如果不是完全平方数，尝试找到最接近的因数分解
        # 对于非正方形，返回全1的mask（不进行modulation）
        return torch.ones(seq_len_q, device=device, dtype=dtype)
    
    # 创建坐标网格
    y_coords, x_coords = torch.meshgrid(
        torch.arange(spatial_size, device=device, dtype=dtype),
        torch.arange(spatial_size, device=device, dtype=dtype),
        indexing='ij'
    )
    
    # 归一化坐标到0-1范围
    x_norm = x_coords.float() / spatial_size
    y_norm = y_coords.float() / spatial_size
    
    # 创建mask：在矩形区域内的为1，否则为0
    # 注意：坐标系统：x从左到右(0-1)，y从上到下(0-1)
    mask = ((x_norm >= left) & (x_norm <= right) & 
            (y_norm >= bottom) & (y_norm <= top)).float()

    count = torch.sum(mask).item()
    
    # reshape为1D
    mask = mask.reshape(-1)  # (seq_len_q,)
    
    return mask


class Timesteps(nn.Module):
    def __init__(self, num_channels: int = 320):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps):
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(
            half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, in_features, out_features):
        super(TimestepEmbedding, self).__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(out_features, out_features, bias=True)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)

        return sample


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, conv_shortcut=True):
        super(ResnetBlock2D, self).__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-05, affine=True)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.time_emb_proj = nn.Linear(1280, out_channels, bias=True)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-05, affine=True)
        self.dropout = nn.Dropout(p=0.0, inplace=False)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.nonlinearity = nn.SiLU()
        self.conv_shortcut = None
        if conv_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1
            )

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        temb = self.nonlinearity(temb)
        temb = self.time_emb_proj(temb)[:, :, None, None]
        hidden_states = hidden_states + temb
        hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states

        return output_tensor


class Attention(nn.Module):
    def __init__(
        self, inner_dim, cross_attention_dim=None, num_heads=None, dropout=0.0
    ):
        super(Attention, self).__init__()
        if num_heads is None:
            self.head_dim = 64
            self.num_heads = inner_dim // self.head_dim
        else:
            self.num_heads = num_heads
            self.head_dim = inner_dim // num_heads

        self.scale = self.head_dim**-0.5
        if cross_attention_dim is None:
            cross_attention_dim = inner_dim
        self.to_q = nn.Linear(inner_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)



        self.to_out = nn.ModuleList(
            [nn.Linear(inner_dim, inner_dim), nn.Dropout(dropout, inplace=False)]
        )

    def forward(self, hidden_states, encoder_hidden_states=None, unet_controller: Optional[UNetController] = None, layer_idx: int = 0):
        # -------------------------- 新增：ID prompt embedding替换功能 --------------------------
        is_cross_attention = (encoder_hidden_states is not None)
        if (unet_controller is not None and 
            unet_controller.Use_id_replacement and 
            is_cross_attention and 
            encoder_hidden_states is not None and
            unet_controller.current_time_step is not None):
            
            # 检查位置限制：将 'down0', 'down1', 'down2' 映射到 'down'，'up0', 'up1', 'up2' 映射到 'up'
            unet_position = unet_controller.current_unet_position
            position_prefix = None
            if unet_position:
                if unet_position.startswith('down'):
                    position_prefix = 'down'
                elif unet_position.startswith('up'):
                    position_prefix = 'up'
                elif unet_position.startswith('mid'):
                    position_prefix = 'mid'
            
            # 检查timestep区间限制
            timestep = unet_controller.current_time_step
            if isinstance(timestep, torch.Tensor):
                timestep_value = timestep.item() if timestep.numel() == 1 else int(timestep[0].item())
            else:
                timestep_value = int(timestep)
            
            in_timestep_range = (
                timestep_value >= unet_controller.Id_replacement_start_step and
                timestep_value <= unet_controller.Id_replacement_end_step
            )
            
            # 检查位置是否在允许的列表中
            position_allowed = (
                position_prefix is not None and
                position_prefix in unet_controller.Id_replacement_position
            )
            
            # 只有在位置和timestep都满足条件时才进行替换
            if position_allowed and in_timestep_range:
                # 如果是第一张图像，保存id_prompt对应的embedding
                if unet_controller.current_frame_index == 0:
                    unet_controller.save_first_frame_id_embeds(encoder_hidden_states)
                else:
                    # 如果是后续图像，根据IPR算法进行替换和归一化
                    # IPR算法：hat{T} = (T_iden^1, (||T_iden^1|| / ||T_iden^n||) * T_exp^n)
                    first_id_embeds = unet_controller.get_first_frame_id_embeds()
                    first_id_norm = unet_controller.get_first_frame_id_norm()
                    
                    if (first_id_embeds is not None and 
                        first_id_norm is not None and 
                        first_id_norm > 0 and
                        unet_controller.id_prompt_token_indices is not None):
                        start_idx, end_idx = unet_controller.id_prompt_token_indices
                        
                        # 确保索引在有效范围内
                        if (start_idx >= 0 and end_idx <= encoder_hidden_states.shape[1] and 
                            end_idx - start_idx == first_id_embeds.shape[1]):
                            
                            # 创建encoder_hidden_states的副本（避免修改原始tensor）
                            encoder_hidden_states = encoder_hidden_states.clone()
                            
                            # 对于CFG，只处理positive prompt部分（后半部分）
                            if (unet_controller.do_classifier_free_guidance and 
                                encoder_hidden_states.shape[0] >= 2):
                                half_batch = encoder_hidden_states.shape[0] // 2
                                positive_embeds = encoder_hidden_states[half_batch:]
                                
                                # 1. 计算当前图像的id_prompt embedding的L2范数（||T_iden^n||）
                                current_id_embeds = positive_embeds[:, start_idx:end_idx, :]
                                current_id_norm = torch.norm(current_id_embeds.view(-1)).item()
                                
                                if current_id_norm > 0:
                                    # 2. 计算归一化因子：||T_iden^1|| / ||T_iden^n||
                                    normalization_factor = first_id_norm / current_id_norm
                                    
                                    # 3. 替换id_prompt部分为T_iden^1
                                    first_id_embeds_device = first_id_embeds.to(positive_embeds.device).to(positive_embeds.dtype)
                                    if first_id_embeds_device.shape[0] == 1 and half_batch > 1:
                                        first_id_embeds_device = first_id_embeds_device.expand(half_batch, -1, -1)
                                    positive_embeds[:, start_idx:end_idx, :] = first_id_embeds_device
                                    
                                    # 4. 对expression部分进行归一化：T_exp^n * (||T_iden^1|| / ||T_iden^n||)
                                    # expression部分包括：id_prompt之前的所有token和id_prompt之后的所有token
                                    if start_idx > 0:
                                        # 归一化id_prompt之前的部分
                                        positive_embeds[:, :start_idx, :] = positive_embeds[:, :start_idx, :] * normalization_factor
                                    
                                    if end_idx < positive_embeds.shape[1]:
                                        # 归一化id_prompt之后的部分
                                        positive_embeds[:, end_idx:, :] = positive_embeds[:, end_idx:, :] * normalization_factor
                                    
                                    # 注意：positive_embeds是encoder_hidden_states[half_batch:]的视图（view），
                                    # 所以对positive_embeds的修改已经直接反映到encoder_hidden_states上了，无需再次赋值
                            else:
                                # 没有CFG，直接处理
                                # 1. 计算当前图像的id_prompt embedding的L2范数（||T_iden^n||）
                                current_id_embeds = encoder_hidden_states[:, start_idx:end_idx, :]
                                current_id_norm = torch.norm(current_id_embeds.view(-1)).item()
                                
                                if current_id_norm > 0:
                                    # 2. 计算归一化因子：||T_iden^1|| / ||T_iden^n||
                                    normalization_factor = first_id_norm / current_id_norm
                                    
                                    # 3. 替换id_prompt部分为T_iden^1
                                    first_id_embeds_device = first_id_embeds.to(encoder_hidden_states.device).to(encoder_hidden_states.dtype)
                                    if first_id_embeds_device.shape[0] == 1 and encoder_hidden_states.shape[0] > 1:
                                        first_id_embeds_device = first_id_embeds_device.expand(encoder_hidden_states.shape[0], -1, -1)
                                    encoder_hidden_states[:, start_idx:end_idx, :] = first_id_embeds_device
                                    
                                    # 4. 对expression部分进行归一化：T_exp^n * (||T_iden^1|| / ||T_iden^n||)
                                    if start_idx > 0:
                                        # 归一化id_prompt之前的部分
                                        encoder_hidden_states[:, :start_idx, :] = encoder_hidden_states[:, :start_idx, :] * normalization_factor
                                    
                                    if end_idx < encoder_hidden_states.shape[1]:
                                        # 归一化id_prompt之后的部分
                                        encoder_hidden_states[:, end_idx:, :] = encoder_hidden_states[:, end_idx:, :] * normalization_factor
        # --------------------------------------------------------------------------
        
        q = self.to_q(hidden_states)
        k = (
            self.to_k(encoder_hidden_states)
            if encoder_hidden_states is not None
            else self.to_k(hidden_states)
        )
        v = (
            self.to_v(encoder_hidden_states)
            if encoder_hidden_states is not None
            else self.to_v(hidden_states)
        )
        b, t, c = q.size()

        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        
        # -------------------------- 新增：k和v权重插值功能 --------------------------
        is_self_attention = (encoder_hidden_states is None)
        if (unet_controller is not None and 
            unet_controller.Use_kv_interpolation and 
            is_self_attention and 
            unet_controller.current_time_step is not None):
            
            timestep = unet_controller.current_time_step
            unet_position = unet_controller.current_unet_position
            
            # 检查位置限制：将 'down0', 'down1', 'down2' 映射到 'down'，'up0', 'up1', 'up2' 映射到 'up'
            position_prefix = None
            if unet_position:
                if unet_position.startswith('down'):
                    position_prefix = 'down'
                elif unet_position.startswith('up'):
                    position_prefix = 'up'
                elif unet_position.startswith('mid'):
                    position_prefix = 'mid'
            
            # 检查timestep区间限制
            in_timestep_range = (
                timestep >= unet_controller.Kv_interpolation_start_step and
                timestep <= unet_controller.Kv_interpolation_end_step
            )
            
            # 检查位置是否在允许的列表中
            position_allowed = (
                position_prefix is not None and
                position_prefix in unet_controller.Kv_interpolation_position
            )
            
            # 只有在位置和timestep都满足条件时才进行插值
            if position_allowed and in_timestep_range:
                if unet_controller.current_frame_index == 0:
                    # 第一张图：保存k和v权重（仅在启用KV插值时保存）
                    unet_controller.save_first_frame_kv(
                        k=k, v=v, 
                        timestep=timestep, 
                        unet_position=unet_position, 
                        layer_idx=layer_idx
                    )
                else:
                    # 后续图像：进行线性插值
                    first_k, first_v = unet_controller.get_first_frame_kv(
                        timestep=timestep, 
                        unet_position=unet_position, 
                        layer_idx=layer_idx
                    )
                    
                    if first_k is not None and first_v is not None:
                        # 确保形状匹配
                        if (first_k.shape == k.shape and first_v.shape == v.shape):
                            alpha = unet_controller.Kv_interpolation_alpha
                            # 线性插值：alpha * current + (1 - alpha) * first_frame
                            k = alpha * k + (1 - alpha) * first_k.to(k.device).to(k.dtype)
                            v = alpha * v + (1 - alpha) * first_v.to(v.device).to(v.dtype)
        # --------------------------------------------------------------------------
        
        # -------------------------- 新增：k和v的concat功能（Consistent Self-Attention） --------------------------
        is_self_attention = (encoder_hidden_states is None)
        if (unet_controller is not None and 
            unet_controller.Use_kv_concat and 
            is_self_attention and 
            unet_controller.current_time_step is not None):
            
            timestep = unet_controller.current_time_step
            unet_position = unet_controller.current_unet_position
            
            # 检查位置限制：将 'down0', 'down1', 'down2' 映射到 'down'，'up0', 'up1', 'up2' 映射到 'up'
            position_prefix = None
            if unet_position:
                if unet_position.startswith('down'):
                    position_prefix = 'down'
                elif unet_position.startswith('up'):
                    position_prefix = 'up'
                elif unet_position.startswith('mid'):
                    position_prefix = 'mid'
            
            # 检查timestep区间限制
            in_timestep_range = (
                timestep >= unet_controller.Kv_concat_start_step and
                timestep <= unet_controller.Kv_concat_end_step
            )
            
            # 检查位置是否在允许的列表中
            position_allowed = (
                position_prefix is not None and
                position_prefix in unet_controller.Kv_concat_position
            )
            
            # 只有在位置和timestep都满足条件时才进行concat
            if position_allowed and in_timestep_range:
                # 所有图像都保存自己的k和v权重（concat功能需要所有图像的kv，不仅仅是第一张）
                # 如果同时启用了interpolation，这里也会保存（覆盖之前保存的相同数据，这是可以接受的）
                unet_controller.save_frame_kv(
                    k=k, v=v, 
                    timestep=timestep, 
                    unet_position=unet_position, 
                    layer_idx=layer_idx
                )
                
                # 对于后续图像（frame_index > 0），使用当前Q和前序K计算attention scores，选择得分最高的token进行concat
                if unet_controller.current_frame_index > 0:
                    # 使用当前图像的Q和前序图像的K计算attention scores，选择得分最高的token
                    previous_k, previous_v = unet_controller.get_previous_frames_kv(
                        q=q,  # 当前图像的query
                        timestep=timestep, 
                        unet_position=unet_position, 
                        layer_idx=layer_idx,
                        sampling_rate=unet_controller.Kv_concat_sampling_rate,
                        scale=self.scale  # attention scale因子
                    )
                    
                    if previous_k is not None and previous_v is not None:
                        # 确保形状匹配（batch, num_heads, seq_len, head_dim）
                        if (previous_k.shape[0] == k.shape[0] and 
                            previous_k.shape[1] == k.shape[1] and 
                            previous_k.shape[3] == k.shape[3] and
                            previous_v.shape[0] == v.shape[0] and 
                            previous_v.shape[1] == v.shape[1] and 
                            previous_v.shape[3] == v.shape[3]):
                            
                            # 将previous_k和previous_v移到正确的设备和数据类型
                            previous_k = previous_k.to(k.device).to(k.dtype)
                            previous_v = previous_v.to(v.device).to(v.dtype)
                            
                            # 将前序图像采样的token与当前帧的k和v拼接（沿着seq_len维度，即dim=2）
                            # 按照伪代码逻辑：token_KV = concat([sampled_tokens, tile_features], dim=1)
                            # 在PyTorch中，对于形状(batch, num_heads, seq_len, head_dim)，dim=2对应seq_len
                            k = torch.cat([previous_k, k], dim=2)  # (batch, num_heads, total_samples + seq_len_current, head_dim)
                            v = torch.cat([previous_v, v], dim=2)  # (batch, num_heads, total_samples + seq_len_current, head_dim)
                            
                            # 注意：根据伪代码，Q应该保持为tile_features（当前帧），即token_Q = tile_features
                            # 所以q不需要改变，attention计算时q的seq_len保持不变，k和v的seq_len增加
                            # 这样attention scores的形状为 (batch, num_heads, seq_len_q, seq_len_k)
                            # 其中seq_len_k = total_samples + seq_len_current > seq_len_q
        # --------------------------------------------------------------------------
        

        # 计算attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # -------------------------- 新增：Mask modulation功能 --------------------------
        is_self_attention = (encoder_hidden_states is None)
        
        # 保存modulation前的scores（用于可视化）
        # 注意：需要在modulation判断之前保存，以便即使modulation不应用也能保存before
        # 保存指定位置的scores（用于可视化）
        scores_before = None
        if (unet_controller is not None and 
            unet_controller.Visualize_modulation_scores and 
            unet_controller.Use_mask_modulation and 
            unet_controller.current_time_step is not None and
            unet_controller.current_unet_position in unet_controller.Modulation_scores_vis_positions and
            unet_controller.current_attn_idx_in_block is not None and
            unet_controller.current_attn_idx_in_block in unet_controller.Modulation_scores_vis_layer_indices):
            # 只保存positive prompt部分（如果有CFG）
            batch_size = scores.shape[0]
            half_batch = batch_size // 2
            if (unet_controller.do_classifier_free_guidance and batch_size >= 2):
                scores_before = scores[half_batch:].detach().cpu()  # 只保存positive部分
            else:
                scores_before = scores.detach().cpu()
        
        if (unet_controller is not None and 
            unet_controller.Use_mask_modulation and 
            unet_controller.current_time_step is not None):
            
            # 检查是否在modulation的时间范围内
            # timestep范围是0-49，总共50步
            total_steps = 50
            timestep = unet_controller.current_time_step
            # 将timestep转换为整数步数
            if isinstance(timestep, torch.Tensor):
                current_step = timestep.item() if timestep.numel() == 1 else int(timestep[0].item())
            else:
                current_step = int(timestep)
            max_modulation_steps = int(total_steps * unet_controller.Mask_modulation_part)
            
            if current_step < max_modulation_steps:
                # 保存scores的原始数据类型和设备
                scores_dtype = scores.dtype
                scores_device = scores.device
                
                # 计算timestep权重（类似参考代码中的treg）
                # timestep范围是0-49，归一化到0-1范围，然后计算5次方
                if isinstance(timestep, torch.Tensor):
                    treg = torch.pow(timestep.float() / float(total_steps), 1)
                else:
                    treg = torch.pow(torch.tensor(float(timestep) / float(total_steps), dtype=torch.float32), 1)
                # 确保treg与scores具有相同的数据类型和设备
                treg = treg.to(device=scores_device, dtype=scores_dtype)
                
                # 获取scores的形状：(batch, num_heads, seq_len_q, seq_len_k)
                # 对于classifier-free guidance，只对后半部分（positive prompt）进行modulation
                batch_size = scores.shape[0]
                half_batch = batch_size // 2
                
                # 只处理positive prompt部分（后半部分）
                if (unet_controller.do_classifier_free_guidance and batch_size >= 2):
                    scores_to_modulate = scores[half_batch:]  # (half_batch, num_heads, seq_len_q, seq_len_k)
                else:
                    scores_to_modulate = scores
                
                seq_len_q = scores_to_modulate.shape[-2]
                seq_len_k = scores_to_modulate.shape[-1]

                mask = create_rectangular_mask(
                            seq_len_q, 
                            unet_controller.Mask_coords, 
                            scores_device, 
                            scores_dtype
                        )
                mask_expanded = mask.view(1, 1, seq_len_q, 1)  
                
                # 对于cross-attention，如果指定了subject_text，只对subject对应的token进行modulation
                # 对于self-attention，使用空间mask
                if not is_self_attention and unet_controller.subject_text is not None and unet_controller.tokenizer is not None and unet_controller.prompts is not None:
                    # Cross-attention: 只对subject对应的token位置进行modulation（所有空间位置）
                        # 获取完整prompt的tokens
                    prompt_tokens = utils.prompt2tokens(unet_controller.tokenizer, unet_controller.prompts[0])
                    # 获取subject的tokens
                    subject_tokens = utils.prompt2tokens(unet_controller.tokenizer, unet_controller.subject_text)
                    subject_tokens = [word for word in subject_tokens if word != '<|endoftext|>' and word != '<|startoftext|>']
                    # 找到subject在prompt中的位置
                    index_of_subject = utils.find_sublist_index(prompt_tokens, subject_tokens)
                    
                    if index_of_subject != -1:
                        # 创建subject token位置的mask（在key维度上）
                        subject_token_indices = list(range(index_of_subject, index_of_subject + len(subject_tokens)))
                        # 确保索引在有效范围内（seq_len_k通常是77）
                        subject_token_indices = [idx for idx in subject_token_indices if 0 <= idx < seq_len_k]
                        
                        if len(subject_token_indices) > 0:
                            # 创建key维度的mask：subject位置为1，其他位置为0
                            key_mask = torch.zeros(seq_len_k, device=scores_device, dtype=scores_dtype)
                            key_mask[subject_token_indices] = 1.0
                            # 扩展维度：(1, 1, 1, seq_len_k)，对所有query位置都应用
                            key_mask_expanded = key_mask.view(1, 1, 1, seq_len_k)
                            
                            # 计算min和max值（在最后一个维度上，即key维度）
                            min_value = scores_to_modulate.min(dim=-1, keepdim=True)[0]  # (half_batch, num_heads, seq_len_q, 1)
                            max_value = scores_to_modulate.max(dim=-1, keepdim=True)[0]  # (half_batch, num_heads, seq_len_q, 1)
                            
                            # 计算modulation强度
                            reg = float(unet_controller.Mask_modulation_reg)
                            size_reg = 1.0
                            modulation_factor = size_reg * reg * treg
                            
                            # 应用modulation：只对subject token位置进行（所有query位置）
                            # subject token位置：增加scores（向max_value靠近）
                            # 非subject token位置：减少scores（向min_value靠近）
                            modulation_mask = (key_mask_expanded > 0).to(dtype=scores_dtype)  # (1, 1, 1, seq_len_k)
                            modulation_mask_inv = (key_mask_expanded <= 0).to(dtype=scores_dtype)  # (1, 1, 1, seq_len_k)
                            
                            #测试
                            # scores_to_modulate = mask_expanded * key_mask_expanded * scores_to_modulate

                            #测试

                            scores_to_modulate = scores_to_modulate + (
                                mask_expanded * modulation_mask * modulation_factor * (max_value - scores_to_modulate)
                            )
                            scores_to_modulate = scores_to_modulate - (
                                (1 - mask_expanded) * modulation_mask * modulation_factor * (scores_to_modulate - min_value)
                            )
                        else:
                            # subject token索引无效，跳过modulation
                            pass
                    else:
                        # 未找到subject，跳过modulation
                        pass
                else:
                    # Self-attention或未指定subject: 使用原有的空间mask逻辑
                    # 生成mask（只针对query的空间维度）
                    mask = create_rectangular_mask(
                        seq_len_q, 
                        unet_controller.Mask_coords, 
                        scores_device, 
                        scores_dtype
                    )  # (seq_len_q,)
                    
                    # 扩展mask维度以匹配scores的形状
                    # mask需要扩展到 (1, 1, seq_len_q, 1) 以便广播
                    mask_expanded = mask.view(1, 1, seq_len_q, 1)  # (1, 1, seq_len_q, 1)
                    
                    # 计算min和max值（在最后一个维度上，即key维度）
                    min_value = scores_to_modulate.min(dim=-1, keepdim=True)[0]  # (half_batch, num_heads, seq_len_q, 1)
                    max_value = scores_to_modulate.max(dim=-1, keepdim=True)[0]  # (half_batch, num_heads, seq_len_q, 1)
                    
                    
                    # 计算modulation强度（确保是标量，并转换为正确的数据类型）
                    reg = float(unet_controller.Mask_modulation_reg)
                    size_reg = 1.0  # 可以根据需要调整，参考代码中有reg_sizes
                    
                    # 应用modulation
                    # mask>0的区域：增加scores（向max_value靠近）
                    # mask<=0的区域：减少scores（向min_value靠近）
                    modulation_mask = (mask_expanded > 0).to(dtype=scores_dtype)  # 转换为数值类型
                    modulation_mask_inv = (mask_expanded <= 0).to(dtype=scores_dtype)  # 转换为数值类型
                    # 确保所有运算保持原始数据类型
                    modulation_factor = size_reg * reg * treg

                    #测试
                    # modulation_factor = 1.0
                    # max_value = 10.
                    # min_value = -10.
                    count = torch.sum(mask_expanded).item()
                    # scores_to_modulate = (1 - mask_expanded) * scores_to_modulate


                    scores_to_modulate = scores_to_modulate + (
                        modulation_mask * modulation_factor * (max_value - scores_to_modulate)
                    )
                    scores_to_modulate = scores_to_modulate - (
                        modulation_mask_inv * modulation_factor * (scores_to_modulate - min_value)
                    )
                
                # 确保scores_to_modulate保持原始数据类型
                scores_to_modulate = scores_to_modulate.to(dtype=scores_dtype)
                
                # 保存modulation后的scores（用于可视化）
                # 保存指定位置的scores
                if (unet_controller is not None and 
                    unet_controller.Visualize_modulation_scores and
                    unet_controller.current_unet_position in unet_controller.Modulation_scores_vis_positions and
                    unet_controller.current_attn_idx_in_block is not None and
                    unet_controller.current_attn_idx_in_block in unet_controller.Modulation_scores_vis_layer_indices):
                    scores_after = scores_to_modulate.detach().cpu()
                    # 直接保存可视化结果（不存储到modulation_scores_store）
                    if scores_before is not None and unet_controller.result_save_dir is not None:
                        unet_position = unet_controller.current_unet_position if unet_controller.current_unet_position else "unknown"
                        timestep = unet_controller.current_time_step
                        if isinstance(timestep, torch.Tensor):
                            timestep = timestep.item() if timestep.numel() == 1 else int(timestep[0].item())
                        else:
                            timestep = int(timestep)
                        
                        # 保存scores（对batch和head维度取平均，只保留空间维度）
                        scores_before_avg = scores_before.mean(dim=(0, 1))  # (seq_len_q, seq_len_k)
                        scores_after_avg = scores_after.mean(dim=(0, 1))  # (seq_len_q, seq_len_k)
                        
                        import os
                        instance_id = os.path.basename(os.path.normpath(unet_controller.result_save_dir))
                        modulation_vis_dir = os.path.join("result", "Attention-modulation-visualization-part0.5-reg5.0-factor1.0-subMask+attenMask-tregpow1up-test", instance_id)
                        block_position = f"{unet_position}_{'self' if is_self_attention else 'cross'}_{layer_idx}"
                        
                        # 获取当前帧索引和prompt信息
                        frame_index = getattr(unet_controller, 'current_frame_index', None)
                        prompt = None
                        if unet_controller.prompts is not None and len(unet_controller.prompts) > 0:
                            prompt = unet_controller.prompts[0]
                        
                        utils.save_single_modulation_score(
                            before_scores=scores_before_avg,
                            after_scores=scores_after_avg,
                            timestep=timestep,
                            block_position=block_position,
                            save_dir=modulation_vis_dir,
                            frame_index=frame_index,
                            is_cross_attention=not is_self_attention,
                            tokenizer=unet_controller.tokenizer,
                            prompt=prompt
                        )
                
                # 将modulated scores放回原位置
                if (unet_controller.do_classifier_free_guidance and batch_size >= 2):
                    scores = torch.cat([scores[:half_batch], scores_to_modulate], dim=0)
                else:
                    scores = scores_to_modulate
            else:
                # 如果不在modulation范围内，after应该等于before
                # 直接保存可视化结果（不存储到modulation_scores_store）
                if (unet_controller is not None and 
                    unet_controller.Visualize_modulation_scores and 
                    scores_before is not None and
                    unet_controller.current_unet_position in unet_controller.Modulation_scores_vis_positions and
                    unet_controller.current_attn_idx_in_block is not None and
                    unet_controller.current_attn_idx_in_block in unet_controller.Modulation_scores_vis_layer_indices and
                    unet_controller.result_save_dir is not None):
                    unet_position = unet_controller.current_unet_position if unet_controller.current_unet_position else "unknown"
                    timestep = unet_controller.current_time_step
                    if isinstance(timestep, torch.Tensor):
                        timestep = timestep.item() if timestep.numel() == 1 else int(timestep[0].item())
                    else:
                        timestep = int(timestep)
                    
                    # 保存scores（对batch和head维度取平均）
                    scores_before_avg = scores_before.mean(dim=(0, 1))  # (seq_len_q, seq_len_k)
                    scores_after_avg = scores_before_avg.clone()  # after等于before（没有modulation）
                    
                    import os
                    instance_id = os.path.basename(os.path.normpath(unet_controller.result_save_dir))
                    modulation_vis_dir = os.path.join("result", "Attention-modulation-visualization", instance_id)
                    block_position = f"{unet_position}_{'self' if is_self_attention else 'cross'}_{layer_idx}"
                    
                    # 获取当前帧索引和prompt信息
                    frame_index = getattr(unet_controller, 'current_frame_index', None)
                    prompt = None
                    if unet_controller.prompts is not None and len(unet_controller.prompts) > 0:
                        prompt = unet_controller.prompts[0]
                    
                    utils.save_single_modulation_score(
                        before_scores=scores_before_avg,
                        after_scores=scores_after_avg,
                        timestep=timestep,
                        block_position=block_position,
                        save_dir=modulation_vis_dir,
                        frame_index=frame_index,
                        is_cross_attention=not is_self_attention,
                        tokenizer=unet_controller.tokenizer,
                        prompt=prompt
                    )
        # --------------------------------------------------------------------------
        
        if (unet_controller is not None and unet_controller.Use_ipca and unet_controller.current_unet_position in unet_controller.Ipca_position
              and encoder_hidden_states is not None and unet_controller.current_time_step >= unet_controller.Ipca_start_step):
            
            if unet_controller.do_classifier_free_guidance is True:
                attn_weights = torch.softmax(scores, dim=-1)  # this is only used by cross_attn_map store
                # -------------------------- 新增：保存注意力权重 --------------------------
                if unet_controller is not None and unet_controller.Visualize_attn_map and is_cross:
                    unet_controller.save_attn_map(
                        attn_weights=attn_weights,
                        place_in_unet=unet_controller.current_unet_position  # down/mid/up
                    )
                # --------------------------------------------------------------------------
                ipca_attn_output = utils.ipca2(q,k,v,self.scale,unet_controller=unet_controller)
                attn_output = ipca_attn_output
            else:
                exit("current doesn't support cfg=1.0")

        else:
            attn_weights = torch.softmax(scores, dim=-1)
            # -------------------------- 新增：保存注意力权重 --------------------------
            if unet_controller is not None:
                is_cross = (encoder_hidden_states is not None)
                unet_controller.save_attn_map(
                    attn_weights=attn_weights,
                    is_cross=is_cross,
                    place_in_unet=unet_controller.current_unet_position
                )
            # --------------------------------------------------------------------------
            attn_output = torch.matmul(attn_weights, v)


        attn_output = attn_output.transpose(1, 2).contiguous().view(b, t, c)

        for layer in self.to_out:
            attn_output = layer(attn_output)

        return attn_output


class GEGLU(nn.Module):
    def __init__(self, in_features, out_features):
        super(GEGLU, self).__init__()
        self.proj = nn.Linear(in_features, out_features * 2, bias=True)

    def forward(self, x):
        x_proj = self.proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        return x1 * torch.nn.functional.gelu(x2)


class FeedForward(nn.Module):
    def __init__(self, in_features, out_features):
        super(FeedForward, self).__init__()

        self.net = nn.ModuleList(
            [
                GEGLU(in_features, out_features * 4),
                nn.Dropout(p=0.0, inplace=False),
                nn.Linear(out_features * 4, out_features, bias=True),
            ]
        )

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(self, hidden_size):
        super(BasicTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.attn1 = Attention(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.attn2 = Attention(hidden_size, 2048)
        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.ff = FeedForward(hidden_size, hidden_size)

    def forward(self, x, encoder_hidden_states=None, unet_controller: Optional[UNetController] = None, layer_idx: int = 0):
        residual = x

        x = self.norm1(x)
        # attn1是自注意力，传递layer_idx用于k和v插值
        x = self.attn1(x, unet_controller=unet_controller, layer_idx=layer_idx * 2)
        x = x + residual

        residual = x

        x = self.norm2(x)
        if encoder_hidden_states is not None:
            # attn2是交叉注意力，不需要插值，但传递layer_idx保持一致性
            x = self.attn2(x, encoder_hidden_states, unet_controller=unet_controller, layer_idx=layer_idx * 2 + 1)
        else:
            # attn2也可能是自注意力（如果没有encoder_hidden_states）
            x = self.attn2(x, unet_controller=unet_controller, layer_idx=layer_idx * 2 + 1)
        x = x + residual

        residual = x

        x = self.norm3(x)
        x = self.ff(x)
        x = x + residual
        return x


class Transformer2DModel(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers):
        super(Transformer2DModel, self).__init__()
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-06, affine=True)
        self.proj_in = nn.Linear(in_channels, out_channels, bias=True)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(out_channels) for _ in range(n_layers)]
        )
        self.proj_out = nn.Linear(out_channels, out_channels, bias=True)

    def forward(self, hidden_states, encoder_hidden_states=None, unet_controller: Optional[UNetController] = None):
        batch, _, height, width = hidden_states.shape
        res = hidden_states
        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
            batch, height * width, inner_dim
        )
        hidden_states = self.proj_in(hidden_states)

        for block_idx, block in enumerate(self.transformer_blocks):
            hidden_states = block(hidden_states, encoder_hidden_states, unet_controller=unet_controller, layer_idx=block_idx)

        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch, height, width, inner_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return hidden_states + res


class Downsample2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample2D, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        return self.conv(x)


class Upsample2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample2D, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class DownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock2D, self).__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(in_channels, out_channels, conv_shortcut=False),
                ResnetBlock2D(out_channels, out_channels, conv_shortcut=False),
            ]
        )
        self.downsamplers = nn.ModuleList([Downsample2D(out_channels, out_channels)])

    def forward(self, hidden_states, temb):
        output_states = []
        for module in self.resnets:
            hidden_states = module(hidden_states, temb)
            output_states.append(hidden_states)

        hidden_states = self.downsamplers[0](hidden_states)
        output_states.append(hidden_states)

        return hidden_states, output_states


class CrossAttnDownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, has_downsamplers=True):
        super(CrossAttnDownBlock2D, self).__init__()
        self.attentions = nn.ModuleList(
            [
                Transformer2DModel(out_channels, out_channels, n_layers),
                Transformer2DModel(out_channels, out_channels, n_layers),
            ]
        )
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(in_channels, out_channels),
                ResnetBlock2D(out_channels, out_channels, conv_shortcut=False),
            ]
        )
        self.downsamplers = None
        if has_downsamplers:
            self.downsamplers = nn.ModuleList(
                [Downsample2D(out_channels, out_channels)]
            )

    def forward(self, hidden_states, temb, encoder_hidden_states, unet_controller: Optional[UNetController] = None):
        output_states = []
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                unet_controller=unet_controller,
            )
            output_states.append(hidden_states)

        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
            output_states.append(hidden_states)

        return hidden_states, output_states


class CrossAttnUpBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, prev_output_channel, n_layers):
        super(CrossAttnUpBlock2D, self).__init__()
        self.attentions = nn.ModuleList(
            [
                Transformer2DModel(out_channels, out_channels, n_layers),
                Transformer2DModel(out_channels, out_channels, n_layers),
                Transformer2DModel(out_channels, out_channels, n_layers),
            ]
        )
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(prev_output_channel + out_channels, out_channels),
                ResnetBlock2D(2 * out_channels, out_channels),
                ResnetBlock2D(out_channels + in_channels, out_channels),
            ]
        )
        self.upsamplers = nn.ModuleList([Upsample2D(out_channels, out_channels)])

    def forward(
        self, hidden_states, res_hidden_states_tuple, temb, encoder_hidden_states, unet_controller: Optional[UNetController] = None, 
    ):
        for attn_idx, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            if unet_controller is not None and unet_controller.Is_freeu_enabled:
                hidden_states, res_hidden_states = utils.apply_freeu(
                    0 if unet_controller.current_unet_position == 'up0' else 1,
                    hidden_states,
                    res_hidden_states,
                    s1=unet_controller.Freeu_parm['s1'],
                    s2=unet_controller.Freeu_parm['s2'],
                    b1=unet_controller.Freeu_parm['b1'],
                    b2=unet_controller.Freeu_parm['b2'],
                )

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            # 传递attn_idx给Transformer2DModel，用于识别是第几个Transformer2DModel
            if unet_controller is not None:
                unet_controller.current_attn_idx_in_block = attn_idx
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                unet_controller=unet_controller,
            )
            if unet_controller is not None:
                unet_controller.current_attn_idx_in_block = None

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class UpBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, prev_output_channel):
        super(UpBlock2D, self).__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(out_channels + prev_output_channel, out_channels),
                ResnetBlock2D(out_channels * 2, out_channels),
                ResnetBlock2D(out_channels + in_channels, out_channels),
            ]
        )

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None):
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(self, in_features):
        super(UNetMidBlock2DCrossAttn, self).__init__()
        self.attentions = nn.ModuleList(
            [Transformer2DModel(in_features, in_features, n_layers=10)]
        )
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(in_features, in_features, conv_shortcut=False),
                ResnetBlock2D(in_features, in_features, conv_shortcut=False),
            ]
        )

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, unet_controller: Optional[UNetController] = None):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                unet_controller=unet_controller,
            )
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class UNet2DConditionModel(ModelMixin, ConfigMixin):
    def __init__(self):
        super(UNet2DConditionModel, self).__init__() ## init child class first

        # This is needed to imitate huggingface config behavior
        # has nothing to do with the model itself
        # remove this if you don't use diffuser's pipeline
        # self.config = namedtuple(
        #             "config", "in_channels addition_time_embed_dim sample_size time_cond_proj_dim"
        #         )
        # self.config.in_channels = 4
        # self.config.addition_time_embed_dim = 256
        # self.config.sample_size = 128
        # self.config.time_cond_proj_dim = None

        self.conv_in = nn.Conv2d(4, 320, kernel_size=3, stride=1, padding=1)
        self.time_proj = Timesteps()
        self.time_embedding = TimestepEmbedding(in_features=320, out_features=1280)
        self.add_time_proj = Timesteps(256)
        self.add_embedding = TimestepEmbedding(in_features=2816, out_features=1280)
        self.down_blocks = nn.ModuleList(
            [
                DownBlock2D(in_channels=320, out_channels=320),
                CrossAttnDownBlock2D(in_channels=320, out_channels=640, n_layers=2),
                CrossAttnDownBlock2D(
                    in_channels=640,
                    out_channels=1280,
                    n_layers=10,
                    has_downsamplers=False,
                ),
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                CrossAttnUpBlock2D(
                    in_channels=640,
                    out_channels=1280,
                    prev_output_channel=1280,
                    n_layers=10,
                ),
                CrossAttnUpBlock2D(
                    in_channels=320,
                    out_channels=640,
                    prev_output_channel=1280,
                    n_layers=2,
                ),
                UpBlock2D(in_channels=320, out_channels=320, prev_output_channel=640),
            ]
        )
        self.mid_block = UNetMidBlock2DCrossAttn(1280)
        self.conv_norm_out = nn.GroupNorm(32, 320, eps=1e-05, affine=True)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(320, 4, kernel_size=3, stride=1, padding=1)

    
    def forward(
        self, sample, timesteps, encoder_hidden_states, added_cond_kwargs, unet_controller: Optional[UNetController] = None, **kwargs
    ):
        # Implement the forward pass through the model
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)

        text_embeds = added_cond_kwargs.get("text_embeds")
        time_ids = added_cond_kwargs.get("time_ids")

        time_embeds = self.add_time_proj(time_ids.flatten())
        time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

        add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
        add_embeds = add_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(add_embeds)

        emb = emb + aug_emb

        sample = self.conv_in(sample)

        # 3. down
        if unet_controller is not None:
            unet_controller.current_unet_position = 'down0'
        
        s0 = sample
        sample, [s1, s2, s3] = self.down_blocks[0](
            sample,
            temb=emb,
        )

        if unet_controller is not None:
            unet_controller.current_unet_position = 'down1'

        # encoder_hidden_states is prompt_embedings, so here do cross_attn
        sample, [s4, s5, s6] = self.down_blocks[1](
            sample,
            temb=emb, # time_embbeding
            encoder_hidden_states=encoder_hidden_states, #[2,77,2048], 2 means two branch, 1 for prompt, 1 for negative prompt
            unet_controller=unet_controller,
        )

        if unet_controller is not None:
            unet_controller.current_unet_position = 'down2'

        sample, [s7, s8] = self.down_blocks[2](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            unet_controller=unet_controller,
        )

        # 4. mid
        if unet_controller is not None:
            unet_controller.current_unet_position = 'mid'

        sample = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states, unet_controller=unet_controller,
        )

        # 5. up
        if unet_controller is not None:
            unet_controller.current_unet_position = 'up0'

        sample = self.up_blocks[0](
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=[s6, s7, s8],
            encoder_hidden_states=encoder_hidden_states,
            unet_controller=unet_controller,
        )

        if unet_controller is not None:
            unet_controller.current_unet_position = 'up1'

        sample = self.up_blocks[1](
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=[s3, s4, s5],
            encoder_hidden_states=encoder_hidden_states,
            unet_controller=unet_controller,
        )

        if unet_controller is not None:
            unet_controller.current_unet_position = 'up2'

        sample = self.up_blocks[2](
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=[s0, s1, s2],
        )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return [sample]
