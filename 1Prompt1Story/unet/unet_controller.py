from asyncio import FastChildWatcher
import torch

# ---新增
import copy
import os
from typing import List, Dict, Optional

class UNetController():
    # Static variables (Hyperparameters)
    Is_freeu_enabled = False
    Freeu_parm = {'s1': 0.6, 's2': 0.4, 'b1': 1.1, 'b2': 1.2}

    # Ipca parameters
    Use_ipca = False
    Ipca_position = ['down0', 'down1', 'down2', 'mid', 'up0', 'up1', 'up2']
    Ipca_start_step = 0
    Ipca_dropout = 0.0
    Use_embeds_mask = True

    # SVR parameters
    Alpha_weaken = 0.01  # 0.01~0.5
    Beta_weaken = 0.05  # 0.05~1.0
    Alpha_enhance = -0.01  # -0.001~-0.02
    Beta_enhance = 1.0  # 1.0~2.0

    # SVR settings
    Prompt_embeds_mode = 'original'
    Remove_pool_embeds = False
    Prompt_embeds_start_step = 0

    Store_qkv = True

    # other settings
    Use_same_latents = True
    Use_same_init_noise = True
    Save_story_image = True
    
    # KV插值设置
    Use_kv_interpolation = False  # 是否使用k和v权重插值（默认关闭）
    Kv_interpolation_alpha = 0.6  # 插值权重（0.0=完全使用第一张图，1.0=完全使用当前图）
    Kv_interpolation_position = ['down','up']  # 在哪些UNet位置进行插值（可选：'down', 'mid', 'up'）
    Kv_interpolation_start_step = 20  # 开始插值的timestep（默认0，即从开始就插值）
    Kv_interpolation_end_step = 35  # 结束插值的timestep（默认1000，即整个去噪过程都插值）
    
    # KV Concat设置（基于Consistent Self-Attention的concat方案）
    Use_kv_concat = False  # 是否使用k和v的concat方法（默认关闭）
    Kv_concat_sampling_rate = 0.2  # 从第一帧k和v中采样的比例（0-1范围，类似伪代码中的sampling_rate）
    Kv_concat_position = ['up']  # 在哪些UNet位置进行concat（可选：'down', 'mid', 'up'）
    Kv_concat_start_step = 0  # 开始concat的timestep
    Kv_concat_end_step = 20  # 结束concat的timestep
    
    # Attention map存储和可视化设置
    Visualize_attn_map = False  # 是否可视化attention map（默认关闭以节省计算）
    
    # Attention mask modulation设置
    Use_mask_modulation = False  # 是否使用mask modulation（默认关闭）
    Mask_coords = [0.3, 0.7, 0.8, 0.3]  # 矩形mask坐标 [left, right, top, bottom]，0-1范围
    Mask_modulation_reg = 10.0  # modulation强度（类似参考代码中的sreg/creg）
    Mask_modulation_part = 0.3  # 在多少比例的去噪步骤中应用modulation（类似参考代码中的reg_part，0-1范围）
    Visualize_modulation_scores = False  # 是否可视化modulation前后的attention scores（默认关闭）
    Modulation_scores_vis_positions = ['up1']  # 要可视化的UNet位置列表，例如：['up1', 'up0', 'mid']
    Modulation_scores_vis_layer_indices = [2]  # 要可视化的层索引列表，例如：[2] 表示第三个Transformer2DModel，[0, 1, 2] 表示前三个
    
    # CLIP 引导设置（推理阶段 CLIP 引导）
    Use_clip_guidance = False  # 是否使用 CLIP 引导（默认关闭）
    Clip_guidance_lambda = 10.0  # CLIP 引导强度 λ，范围5-15
    Clip_guidance_eta = 0.5  # CLIP 引导学习率 η，范围0.01-0.05
    Clip_guidance_interval = 10  # CLIP 引导的 timestep 间隔 n，范围25-50步
    
    # ID prompt embedding 替换设置
    Use_id_replacement = False  # 是否使用id prompt embedding替换（默认关闭）
    Id_replacement_position = ['down', 'mid', 'up']  # 在哪些UNet位置进行替换（可选：'down', 'mid', 'up'）
    Id_replacement_start_step = 0  # 开始替换的timestep（默认0，即从开始就替换）
    Id_replacement_end_step = 1000  # 结束替换的timestep（默认1000，即整个去噪过程都替换）

    def __init__(self):
        self._variables = {}

        ## Variables (updated during inference) ##
        self.device = "cuda"
        self.current_unet_position = 'down'  # down, mid or up
        self.torch_dtype = torch.float16

        self.prompts = None
        self.negative_prompt = None
        self.id_prompt = None
        self.subject_text = None  # subject部分的文本，用于modulation时定位token
        self.frame_prompt_express = None
        self.frame_prompt_suppress = None

        self.frame_prompt_express_list = None
        self.frame_prompt_suppress_list = None

        self.tokenizer = None
        self.result_save_dir = None
        self.current_time_step = None
        self.do_classifier_free_guidance = None

        self.q_store = {}
        self.k_store = {}
        self.v_store = {}
        
        # 所有图像的k和v权重存储（用于插值和concat）
        # 结构：{frame_index: {timestep: {unet_position: {layer_idx: tensor}}}}
        self.all_frames_k_store = {}  # 存储所有图像的k权重
        self.all_frames_v_store = {}  # 存储所有图像的v权重
        self.current_frame_index = 0  # 当前是第几张图（0表示第一张）

        self.do_classifier_free_guidance = None
        self.current_unet_position = None
        self.current_attn_idx_in_block = 2  # 当前在block中是第几个Transformer2DModel（0,1,2）

        self.ipca2_index = -1
        self.ipca_time_step = -1
        
        # CLIP 引导相关变量
        self.clip_model = None  # CLIP 模型
        self.clip_processor = None  # CLIP 处理器
        self.clip_text_embeds = None  # CLIP 文本嵌入（用于计算相似度）
        
        # ID prompt embedding 替换相关变量
        self.first_frame_id_embeds = None  # 第一张图像的id_prompt对应的text embedding
        self.first_frame_id_norm = None  # 第一张图像的id_prompt embedding的L2范数（||T_iden^1||）
        self.id_prompt_token_indices = None  # id_prompt在完整prompt中的token索引范围（start_idx, end_idx）
        ## Variables End ##

# -------------------------- 新增：注意力存储相关属性/方法 --------------------------
        self.save_self_attention = False  # 是否保存自注意力（默认False以节省内存）
        self.attn_store_disk = False  # 是否存磁盘（默认存内存）
        self.attn_store_dir = "./unet_attn_cache"  # 磁盘存储目录
        self.cur_denoise_step = 0  # 当前去噪步数
        self.step_attn_store = self._get_empty_attn_store()  # 单步注意力存储
        self.total_attn_store = {}  # 所有步骤注意力累加存储
        self.all_step_attn_paths = []  # 所有步骤注意力路径/数据
        self.is_start_frame = True
        self.attn_map_store = {}
        self.attn_count = 0
        # 内存优化参数
        self.max_attn_resolution = 32  # 最大保存分辨率（32²=1024）
        self.use_half_precision = True  # 使用float16节省内存
        self.max_heads_to_save = 4  # 最多保存的注意力头数（None表示保存所有）
        # 初始化磁盘存储目录（若开启）
        if self.attn_store_disk:
            import datetime
            time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.attn_store_dir = f"./unet_attn_cache_{time_str}"
            os.makedirs(self.attn_store_dir, exist_ok=True)
        
        # Modulation scores存储（用于可视化）
        # 结构：{block_key: {'before': [tensor_list], 'after': [tensor_list]}}
        # block_key格式："{unet_position}_{is_self_attention}_{layer_idx}"
        self.modulation_scores_store = {}

    def _get_empty_attn_store(self) -> Dict[str, List[torch.Tensor]]:
        """获取空的注意力存储结构（按UNet位置+注意力类型分类）"""
        return {
            "down_cross": [], "mid_cross": [], "up_cross": [],
            "down_self": [], "mid_self": [], "up_self": []
        }

    def reset_attn_store(self, clear_modulation_scores=True):
        """重置注意力存储（每次推理前调用）
        :param clear_modulation_scores: 是否清空modulation_scores_store（默认True）
                                        如果需要在所有图像生成完成后可视化，应设置为False
        """
        self.cur_denoise_step = 0
        self.attn_count = 0
        self.step_attn_store = self._get_empty_attn_store()
        self.total_attn_store = {}
        self.all_step_attn_paths = []
        # 重置modulation scores存储（可选）
        if clear_modulation_scores:
            self.modulation_scores_store = {}

    def save_attn_map(self, attn_weights: torch.Tensor, is_cross: bool, place_in_unet: str):
        """
        保存注意力权重到unet_controller（核心存储方法）
        :param attn_weights: 注意力权重张量
        :param is_cross: 是否为交叉注意力
        :param place_in_unet: 注意力层在UNet的位置（down/mid/up 或 down0/down1/down2/mid/up0/up1/up2）
        """
        
        # 1. 将精确位置映射到前缀（down0/down1/down2 -> down, up0/up1/up2 -> up, mid -> mid）
        if place_in_unet.startswith("down"):
            place_prefix = "down"
        elif place_in_unet.startswith("up"):
            place_prefix = "up"
        elif place_in_unet.startswith("mid"):
            place_prefix = "mid"
        else:
            # 如果无法识别，使用原值
            place_prefix = place_in_unet
        
        # 2. 生成存储key
        attn_key = f"{place_prefix}_{'cross' if is_cross else 'self'}"

        # 3. 确保存储key存在（防止KeyError）
        if attn_key not in self.step_attn_store:
            self.step_attn_store[attn_key] = []

        # 4. 筛选：仅保存序列长度≤max_attn_resolution²的注意力（避免内存过载）
        max_seq_len = self.max_attn_resolution ** 2
        if attn_weights.shape[-2] <= max_seq_len:
            # 根据开关决定是否保存：交叉注意力总是保存，自注意力根据save_self_attention决定
            if is_cross or (not is_cross and self.save_self_attention):
                # 脱离计算图+迁移到CPU（避免占用GPU内存）
                save_tensor = attn_weights.detach().cpu()
                
                # 内存优化：提前聚合，减少维度
                # 原始shape可能是 (batch, num_heads, seq_len_q, seq_len_k)
                # 例如：(2, 20, 1024, 77) -> (32, 32, 77)
                if save_tensor.dim() == 4:
                    # 4维：(batch, num_heads, seq_len_q, seq_len_k)
                    # 对batch和head维度取均值
                    save_tensor = save_tensor.mean(dim=(0, 1))  # -> (seq_len_q, seq_len_k)
                    
                    # 如果是交叉注意力且seq_len_q是空间维度，reshape为2D空间
                    if is_cross and save_tensor.shape[0] == max_seq_len:
                        # (1024, 77) -> (32, 32, 77)
                        spatial_size = int(save_tensor.shape[0] ** 0.5)
                        if spatial_size * spatial_size == save_tensor.shape[0]:
                            save_tensor = save_tensor.view(spatial_size, spatial_size, -1)
                    elif not is_cross and save_tensor.shape[0] == max_seq_len and save_tensor.shape[1] == max_seq_len:
                        # 自注意力：(1024, 1024) -> (32, 32, 32, 32) 或直接保持 (1024, 1024)
                        # 为了节省内存，可以只保存对角线或聚合结果
                        pass  # 保持原形状，后续可视化时再处理
                
                elif save_tensor.dim() == 3:
                    # 3维：(batch, seq_len_q, seq_len_k) 或 (num_heads, seq_len_q, seq_len_k)
                    # 对第一个维度取均值
                    save_tensor = save_tensor.mean(dim=0)  # -> (seq_len_q, seq_len_k)
                    
                    # 如果是交叉注意力且seq_len_q是空间维度，reshape为2D空间
                    if is_cross and save_tensor.shape[0] == max_seq_len:
                        spatial_size = int(save_tensor.shape[0] ** 0.5)
                        if spatial_size * spatial_size == save_tensor.shape[0]:
                            save_tensor = save_tensor.view(spatial_size, spatial_size, -1)
                
                # 转换为float16以节省内存
                if self.use_half_precision:
                    save_tensor = save_tensor.half()
                
                # 保存聚合后的结果（已经降维，内存占用大幅减少）
                self.step_attn_store[attn_key].append(save_tensor.clone())

        self.attn_count += 1

    def aggregate_step_attn(self):
        """每步去噪完成后，聚合单步注意力到总存储"""
        # 1. 累加单步注意力到总存储（使用浅拷贝和原地操作以节省内存）
        if len(self.total_attn_store) == 0:
            # 初始化时使用浅拷贝
            self.total_attn_store = {}
            for key, value_list in self.step_attn_store.items():
                self.total_attn_store[key] = [t.clone() for t in value_list]
        else:
            for key in self.total_attn_store.keys():
                if key in self.step_attn_store:
                    for i in range(min(len(self.total_attn_store[key]), len(self.step_attn_store[key]))):
                        # 原地累加以节省内存
                        self.total_attn_store[key][i] = self.total_attn_store[key][i] + self.step_attn_store[key][i]

        # 2. 保存单步注意力（内存/磁盘）- 只在需要时保存
        # 为了节省内存，默认不保存所有步骤的详细数据，只保存聚合结果
        if self.attn_store_disk:
            step_save_path = os.path.join(self.attn_store_dir, f"denoise_step_{self.cur_denoise_step:03d}.pt")
            # 使用浅拷贝
            step_data = {k: [t.clone() for t in v] for k, v in self.step_attn_store.items()}
            torch.save(step_data, step_save_path)
            self.all_step_attn_paths.append(step_save_path)
            # 立即释放内存
            del step_data
        # 注意：不保存到内存中的all_step_attn_paths，只保留聚合结果（节省内存）

        # 3. 重置单步存储并释放内存
        for key in list(self.step_attn_store.keys()):
            self.step_attn_store[key].clear()
        self.step_attn_store = self._get_empty_attn_store()
        self.cur_denoise_step += 1
        
        # 4. 强制垃圾回收（可选，在内存紧张时启用）
        if self.cur_denoise_step % 10 == 0:  # 每10步清理一次
            import gc
            gc.collect()

    def get_average_attn(self) -> Dict[str, List[torch.Tensor]]:
        """获取所有去噪步骤的平均注意力权重（消除单步波动）"""
        if self.cur_denoise_step == 0:
            return copy.deepcopy(self.total_attn_store)

        average_attn = {}
        for key in self.total_attn_store.keys():
            average_attn[key] = [
                item / self.cur_denoise_step for item in self.total_attn_store[key]
            ]
        return average_attn

    # -------------------------- 新增：注意力存储相关属性/方法 --------------------------

    

    def save_frame_kv(self, k: torch.Tensor, v: torch.Tensor, timestep: int, unet_position: str, layer_idx: int = 0):
        """
        保存当前图像的k和v权重
        :param k: k权重张量
        :param v: v权重张量
        :param timestep: 当前时间步
        :param unet_position: UNet位置（down0/down1/mid/up0等）
        :param layer_idx: 层索引（用于区分同一位置的多个层）
        """
        frame_index = self.current_frame_index
        
        # 初始化存储结构
        if frame_index not in self.all_frames_k_store:
            self.all_frames_k_store[frame_index] = {}
            self.all_frames_v_store[frame_index] = {}
        
        if timestep not in self.all_frames_k_store[frame_index]:
            self.all_frames_k_store[frame_index][timestep] = {}
            self.all_frames_v_store[frame_index][timestep] = {}
        
        if unet_position not in self.all_frames_k_store[frame_index][timestep]:
            self.all_frames_k_store[frame_index][timestep][unet_position] = {}
            self.all_frames_v_store[frame_index][timestep][unet_position] = {}
        
        # 保存k和v（detach并移到CPU以节省GPU内存）
        self.all_frames_k_store[frame_index][timestep][unet_position][layer_idx] = k.detach().cpu()
        self.all_frames_v_store[frame_index][timestep][unet_position][layer_idx] = v.detach().cpu()
    
    def save_first_frame_kv(self, k: torch.Tensor, v: torch.Tensor, timestep: int, unet_position: str, layer_idx: int = 0):
        """
        保存当前图像的k和v权重（向后兼容的别名，实际调用save_frame_kv）
        """
        self.save_frame_kv(k, v, timestep, unet_position, layer_idx)
    
    def get_first_frame_kv(self, timestep: int, unet_position: str, layer_idx: int = 0):
        """
        获取第一张图的k和v权重（向后兼容方法，用于interpolation功能）
        :param timestep: 当前时间步
        :param unet_position: UNet位置
        :param layer_idx: 层索引
        :return: (k, v) 或 (None, None) 如果不存在
        """
        frame_index = 0  # 第一张图的索引是0
        if (frame_index in self.all_frames_k_store and
            timestep in self.all_frames_k_store[frame_index] and 
            unet_position in self.all_frames_k_store[frame_index][timestep] and
            layer_idx in self.all_frames_k_store[frame_index][timestep][unet_position]):
            k = self.all_frames_k_store[frame_index][timestep][unet_position][layer_idx]
            v = self.all_frames_v_store[frame_index][timestep][unet_position][layer_idx]
            # 移回原始设备
            return k.to(self.device), v.to(self.device)
        return None, None
    
    def get_previous_frames_kv(self, q: torch.Tensor, timestep: int, unet_position: str, layer_idx: int = 0, 
                                sampling_rate: float = 0.6, scale: float = 1.0):
        """
        使用当前图像的Q和前序图像的K计算attention scores，选择得分最高的token进行采样和合并
        :param q: 当前图像的query张量，形状为 (batch, num_heads, seq_len_q, head_dim)
        :param timestep: 当前时间步
        :param unet_position: UNet位置
        :param layer_idx: 层索引
        :param sampling_rate: 采样率（0-1范围），表示从每个前序图像中采样多少比例的token
        :param scale: attention scale因子（用于计算attention scores）
        :return: 合并后的 (k, v) 或 (None, None) 如果没有前序图像
        """
        current_frame = self.current_frame_index
        
        # 如果没有前序图像，返回None
        if current_frame == 0:
            return None, None
        
        # 收集所有前序图像的kv（frame_index < current_frame_index）
        all_previous_k_list = []
        all_previous_v_list = []
        
        for prev_frame_idx in range(current_frame):
            if (prev_frame_idx in self.all_frames_k_store and
                timestep in self.all_frames_k_store[prev_frame_idx] and
                unet_position in self.all_frames_k_store[prev_frame_idx][timestep] and
                layer_idx in self.all_frames_k_store[prev_frame_idx][timestep][unet_position]):
                
                prev_k = self.all_frames_k_store[prev_frame_idx][timestep][unet_position][layer_idx]
                prev_v = self.all_frames_v_store[prev_frame_idx][timestep][unet_position][layer_idx]
                
                # 移回GPU并确保正确的数据类型
                prev_k = prev_k.to(q.device).to(q.dtype)
                prev_v = prev_v.to(q.device).to(q.dtype)
                
                # 计算attention scores: Q @ K_prev^T * scale
                # q形状: (batch, num_heads, seq_len_q, head_dim)
                # prev_k形状: (batch, num_heads, seq_len_prev, head_dim)
                # scores形状: (batch, num_heads, seq_len_q, seq_len_prev)
                scores = torch.matmul(q, prev_k.transpose(-2, -1)) * scale
                
                # 为了选择最重要的token，我们需要聚合scores
                # 对query维度取平均或最大值，得到每个前序token的重要性分数
                # scores: (batch, num_heads, seq_len_q, seq_len_prev)
                # -> (batch, num_heads, seq_len_prev) 对query维度取平均
                importance_scores = scores.mean(dim=2)  # (batch, num_heads, seq_len_prev)
                
                # 对batch维度取平均，得到每个head的重要性分数
                # 这样可以处理CFG的情况（如果batch包含多个分支）
                # importance_scores: (batch, num_heads, seq_len_prev)
                # -> (num_heads, seq_len_prev)
                head_importance = importance_scores.mean(dim=0)  # (num_heads, seq_len_prev)
                
                # 根据采样率计算需要采样的token数量
                seq_len_prev = prev_k.shape[2]
                num_samples = max(1, int(seq_len_prev * sampling_rate))
                
                # 对每个head分别选择top-k个token（因为不同head关注不同的特征）
                # 但为了保持一致性，我们也可以对所有head取平均后选择全局top-k
                # 这里选择对所有head取平均，得到一个全局重要性分数
                global_importance = head_importance.mean(dim=0)  # (seq_len_prev,)
                
                # 选择得分最高的top-k个token
                top_k_values, top_k_indices = torch.topk(global_importance, k=num_samples, dim=-1)
                
                # 采样对应的k和v（对所有batch和head使用相同的采样索引）
                # prev_k形状: (batch, num_heads, seq_len_prev, head_dim)
                sampled_k = prev_k[:, :, top_k_indices, :]  # (batch, num_heads, num_samples, head_dim)
                sampled_v = prev_v[:, :, top_k_indices, :]  # (batch, num_heads, num_samples, head_dim)
                
                all_previous_k_list.append(sampled_k)
                all_previous_v_list.append(sampled_v)
        
        # 如果没有找到任何前序图像的kv，返回None
        if len(all_previous_k_list) == 0:
            return None, None
        
        # 合并所有采样的kv（沿着seq_len维度，即dim=2）
        merged_k = torch.cat(all_previous_k_list, dim=2)  # (batch, num_heads, total_samples, head_dim)
        merged_v = torch.cat(all_previous_v_list, dim=2)  # (batch, num_heads, total_samples, head_dim)
        
        return merged_k, merged_v
    
    def reset_first_frame_kv(self):
        """重置所有图像的k和v存储（开始新的序列时调用）"""
        self.all_frames_k_store = {}
        self.all_frames_v_store = {}
        # 同时重置id_prompt embedding
        self.reset_first_frame_id_embeds()
    
    def save_first_frame_id_embeds(self, encoder_hidden_states: torch.Tensor):
        """
        保存第一张图像的id_prompt对应的text embedding
        :param encoder_hidden_states: 文本embedding，形状为 (batch, seq_len, embed_dim)
        """
        if not self.Use_id_replacement:
            return
        
        if self.id_prompt is None or self.tokenizer is None or self.prompts is None or len(self.prompts) == 0:
            return
        
        try:
            import unet.utils as utils
            # 获取完整prompt的tokens
            prompt_tokens = utils.prompt2tokens(self.tokenizer, self.prompts[0])
            # 获取id_prompt的tokens
            id_prompt_tokens = utils.prompt2tokens(self.tokenizer, self.id_prompt)
            id_prompt_tokens = [word for word in id_prompt_tokens if word != '<|endoftext|>' and word != '<|startoftext|>']
            
            # 找到id_prompt在prompt中的位置
            start_idx = utils.find_sublist_index(prompt_tokens, id_prompt_tokens)
            
            if start_idx == -1:
                # 如果找不到，尝试直接匹配（可能id_prompt就是完整的prompt）
                if len(id_prompt_tokens) > 0 and prompt_tokens[:len(id_prompt_tokens)] == id_prompt_tokens:
                    start_idx = 0
                else:
                    return  # 无法找到id_prompt位置，跳过保存
            
            end_idx = start_idx + len(id_prompt_tokens)
            
            # 保存token索引范围
            self.id_prompt_token_indices = (start_idx, end_idx)
            
            # 保存第一张图像的id_prompt embedding（只保存positive prompt部分，如果有CFG）
            # encoder_hidden_states形状: (batch, seq_len, embed_dim)
            # 对于CFG，batch=2，前半部分是negative，后半部分是positive
            if self.do_classifier_free_guidance and encoder_hidden_states.shape[0] >= 2:
                # 只保存positive prompt部分（后半部分）
                positive_embeds = encoder_hidden_states[encoder_hidden_states.shape[0] // 2:]
                # 提取id_prompt对应的embedding部分
                id_embeds = positive_embeds[:, start_idx:end_idx, :].detach().clone()
                self.first_frame_id_embeds = id_embeds
                # 计算L2范数：||T_iden^1|| = sqrt(sum(id_embeds^2))
                # 对每个token的embedding计算L2范数，然后取平均（或者对整个id_prompt部分计算一个总的范数）
                # 根据论文，应该是对整个id_prompt embedding计算L2范数
                self.first_frame_id_norm = torch.norm(id_embeds.view(-1)).item()  # 将id_embeds展平后计算L2范数
            else:
                # 没有CFG，直接保存
                id_embeds = encoder_hidden_states[:, start_idx:end_idx, :].detach().clone()
                self.first_frame_id_embeds = id_embeds
                # 计算L2范数
                self.first_frame_id_norm = torch.norm(id_embeds.view(-1)).item()
        except Exception as e:
            print(f"警告: 保存第一张图像的id_prompt embedding失败: {e}")
            self.first_frame_id_embeds = None
            self.first_frame_id_norm = None
            self.id_prompt_token_indices = None
    
    def get_first_frame_id_embeds(self) -> Optional[torch.Tensor]:
        """
        获取第一张图像的id_prompt对应的text embedding
        :return: id_prompt embedding，形状为 (1, token_len, embed_dim) 或 None
        """
        if not self.Use_id_replacement:
            return None
        
        if self.first_frame_id_embeds is None or self.id_prompt_token_indices is None:
            return None
        
        return self.first_frame_id_embeds
    
    def get_first_frame_id_norm(self) -> Optional[float]:
        """
        获取第一张图像的id_prompt embedding的L2范数（||T_iden^1||）
        :return: L2范数值或None
        """
        if not self.Use_id_replacement:
            return None
        
        return self.first_frame_id_norm
    
    def reset_first_frame_id_embeds(self):
        """重置第一张图像的id_prompt embedding（开始新的序列时调用）"""
        self.first_frame_id_embeds = None
        self.first_frame_id_norm = None
        self.id_prompt_token_indices = None
    
    def init_clip_model(self, prompt):
        """
        初始化 CLIP 模型并计算文本嵌入
        使用与 test_benchmark.py 相同的 CLIP 模型和配置
        :param prompt: 用于计算相似度的文本提示
        """
        if not self.Use_clip_guidance:
            return
        
        try:
            # 使用标准的 CLIP 库，与 test_benchmark.py 保持一致
            import clip
            self.clip_model, self.clip_preprocess = clip.load('ViT-B/32', device=self.device, jit=False)
            self.clip_model.eval()  # 设置为评估模式
            
            # 计算文本嵌入（与 test_benchmark.py 一致，不使用 'A photo depicts' 前缀）
            text_tokens = clip.tokenize([prompt], truncate=True).to(self.device)
            with torch.no_grad():
                self.clip_text_embeds = self.clip_model.encode_text(text_tokens)
                # 归一化（与 test_benchmark.py 中的处理一致）
                # test_benchmark.py 中使用 sklearn.preprocessing.normalize，这里使用简单的 L2 归一化
                self.clip_text_embeds = self.clip_text_embeds / self.clip_text_embeds.norm(dim=-1, keepdim=True)
        except ImportError:
            print("警告: clip 库未安装，CLIP 引导功能将被禁用。请运行: pip install clip-by-openai")
            self.Use_clip_guidance = False
        except Exception as e:
            print(f"警告: CLIP 模型初始化失败: {e}")
            self.Use_clip_guidance = False
    
    def apply_clip_guidance(self, latents, vae, timestep_index, timestep, num_inference_steps):
        """
        应用 CLIP 引导来更新 latents（参考标准实现）
        :param latents: 当前的隐变量 [B, C, H, W]（scheduler.step 后的结果）
        :param vae: VAE 模型（用于解码 latents 到图像）
        :param timestep_index: 当前的 timestep 索引 i
        :param timestep: 当前的 timestep 值 t
        :param num_inference_steps: 总推理步数
        :return: 更新后的 latents
        """
        if not self.Use_clip_guidance:
            return latents
        
        # 如果 eta 为 0，直接返回原始 latents，不执行任何计算
        if self.Clip_guidance_eta == 0:
            return latents
        
        # 检查是否应该在这一步应用 CLIP 引导（参考代码使用 (i+1) % interval == 0）
        if (timestep_index + 1) % self.Clip_guidance_interval != 0:
            return latents
        
        if self.clip_text_embeds is None or self.clip_model is None:
            return latents
        
        try:
            # 保存原始 latents 的副本，确保即使出错也能返回原始值
            original_latents = latents.clone().detach()
            
            # 检查输入 latents 是否有效
            if torch.isnan(latents).any() or torch.isinf(latents).any():
                print(f"警告: 输入 latents 包含 NaN 或 Inf，跳过 CLIP 引导")
                return original_latents
            
            # 需要梯度来计算 loss
            latents_requires_grad = latents.detach().requires_grad_(True)
            
            # 解码 latents 为图像（参考 pipeline 的处理方式）
            with torch.enable_grad():
                # 根据 VAE 配置处理 latents（与 pipeline 最终解码时的处理一致）
                has_latents_mean = hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None
                has_latents_std = hasattr(vae.config, "latents_std") and vae.config.latents_std is not None
                
                # 保存缩放因子，用于后续梯度缩放
                if has_latents_mean and has_latents_std:
                    # SDXL 的处理方式：latents = latents * latents_std / scaling_factor + latents_mean
                    scaling_factor = vae.config.scaling_factor
                    latents_mean = torch.tensor(vae.config.latents_mean, device=latents_requires_grad.device, dtype=latents_requires_grad.dtype).view(1, 4, 1, 1)
                    latents_std = torch.tensor(vae.config.latents_std, device=latents_requires_grad.device, dtype=latents_requires_grad.dtype).view(1, 4, 1, 1)
                    scaled_latents = latents_requires_grad * latents_std / scaling_factor + latents_mean
                    # 对于 SDXL，梯度需要乘以 latents_std / scaling_factor 来反向缩放
                    grad_scale = latents_std / scaling_factor
                else:
                    # 非 SDXL 的处理方式：latents = latents / scaling_factor
                    scaling_factor = vae.config.scaling_factor if hasattr(vae.config, 'scaling_factor') else 0.18215
                    scaled_latents = latents_requires_grad / scaling_factor
                    # 对于非 SDXL，梯度需要除以 scaling_factor 来反向缩放
                    grad_scale = 1.0 / scaling_factor
                
                # 检查 scaled_latents 是否有效
                if torch.isnan(scaled_latents).any() or torch.isinf(scaled_latents).any():
                    print(f"警告: scaled_latents 包含 NaN 或 Inf，跳过 CLIP 引导")
                    return original_latents
                
                # 限制 scaled_latents 的范围，防止 VAE 解码时溢出
                # 根据 SDXL VAE 的配置，latents 通常在合理范围内
                # 但为了安全，我们限制在更大的范围内
                scaled_latents_clamped = torch.clamp(scaled_latents, -20.0, 20.0)
                
                # 检查 clamped 后的值是否有效
                if torch.isnan(scaled_latents_clamped).any() or torch.isinf(scaled_latents_clamped).any():
                    print(f"警告: clamped scaled_latents 包含 NaN 或 Inf，跳过 CLIP 引导")
                    return original_latents
                
                # 解码为图像（VAE decode 需要 float32）
                # 参考 pipeline 的处理：如果 VAE 是 float16 且 force_upcast=True，需要 upcast
                needs_upcasting = vae.dtype == torch.float16 and hasattr(vae.config, "force_upcast") and vae.config.force_upcast
                original_vae_dtype = vae.dtype  # 保存原始 dtype，无论是否需要 upcast
                
                if needs_upcasting:
                    # 临时 upcast VAE 到 float32（参考 pipeline 的 upcast_vae 方法）
                    vae.to(dtype=torch.float32)
                    # 只 upcast 必要的层（参考 pipeline 的实现）
                    use_torch_2_0_or_xformers = hasattr(vae, 'decoder') and hasattr(vae.decoder, 'mid_block')
                    if use_torch_2_0_or_xformers:
                        # 只 upcast post_quant_conv, conv_in, mid_block
                        if hasattr(vae, 'post_quant_conv'):
                            vae.post_quant_conv.to(dtype=torch.float32)
                        if hasattr(vae.decoder, 'conv_in'):
                            vae.decoder.conv_in.to(dtype=torch.float32)
                        if hasattr(vae.decoder, 'mid_block'):
                            vae.decoder.mid_block.to(dtype=torch.float32)
                    else:
                        vae.to(dtype=torch.float32)
                
                vae_dtype = next(iter(vae.post_quant_conv.parameters())).dtype
                scaled_latents_fp32 = scaled_latents_clamped.to(dtype=vae_dtype)
                
                # 检查转换后的 latents 是否有效
                if torch.isnan(scaled_latents_fp32).any() or torch.isinf(scaled_latents_fp32).any():
                    print(f"警告: scaled_latents_fp32 包含 NaN 或 Inf，跳过 CLIP 引导")
                    if needs_upcasting:
                        vae.to(dtype=original_vae_dtype)
                    return original_latents
                
                try:
                    decoded_image = vae.decode(scaled_latents_fp32, return_dict=False)[0]
                except RuntimeError as e:
                    print(f"警告: VAE 解码失败: {e}，跳过 CLIP 引导")
                    if needs_upcasting:
                        vae.to(dtype=original_vae_dtype)
                    return original_latents
                finally:
                    # 恢复 VAE 的原始 dtype
                    if needs_upcasting:
                        vae.to(dtype=original_vae_dtype)
                
                # 检查 VAE 解码结果是否有效
                if torch.isnan(decoded_image).any() or torch.isinf(decoded_image).any():
                    nan_count = torch.isnan(decoded_image).sum().item()
                    inf_count = torch.isinf(decoded_image).sum().item()
                    print(f"警告: VAE 解码结果包含 NaN ({nan_count} 个) 或 Inf ({inf_count} 个)，跳过 CLIP 引导 (timestep={timestep_index}, latents_range=[{scaled_latents_clamped.min().item():.2f}, {scaled_latents_clamped.max().item():.2f}])")
                    return original_latents
                
                # VAE 输出在 [-1, 1] 范围，转换为 [0, 1]
                decoded_image = (decoded_image + 1.0) / 2.0
                decoded_image = torch.clamp(decoded_image, 0.0, 1.0)
                
                # 使用与 test_benchmark.py 相同的预处理方式
                from torch.nn.functional import interpolate
                
                # 1. Resize 到至少 224x224（如果小于的话）
                _, _, h, w = decoded_image.shape
                if h < 224 or w < 224:
                    scale = max(224 / h, 224 / w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    decoded_image = interpolate(decoded_image, size=(new_h, new_w), mode='bilinear', align_corners=False)
                
                # 2. CenterCrop 到 224x224
                _, _, h, w = decoded_image.shape
                top = (h - 224) // 2
                left = (w - 224) // 2
                decoded_image = decoded_image[:, :, top:top+224, left:left+224]
                
                # 3. 应用 CLIP 的归一化（与 test_benchmark.py 中的 Normalize 参数一致）
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=decoded_image.device, dtype=decoded_image.dtype).view(1, 3, 1, 1)
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=decoded_image.device, dtype=decoded_image.dtype).view(1, 3, 1, 1)
                decoded_image_normalized = (decoded_image - mean) / std
                
                # 检查归一化后的图像是否有效
                if torch.isnan(decoded_image_normalized).any() or torch.isinf(decoded_image_normalized).any():
                    print(f"警告: 归一化后的图像包含 NaN 或 Inf，跳过 CLIP 引导")
                    return original_latents
                
                # 使用 CLIP 编码图像
                image_embeds = self.clip_model.encode_image(decoded_image_normalized)
                
                # 检查 CLIP 编码结果是否有效
                if torch.isnan(image_embeds).any() or torch.isinf(image_embeds).any():
                    print(f"警告: CLIP 编码结果包含 NaN 或 Inf，跳过 CLIP 引导")
                    return original_latents
                
                # 归一化（添加 epsilon 防止除零）
                image_embeds_norm = image_embeds.norm(dim=-1, keepdim=True)
                # 如果 norm 为 0 或非常小，使用单位向量
                eps = 1e-8
                image_embeds = image_embeds / torch.clamp(image_embeds_norm, min=eps)
                
                # 检查归一化后的 image_embeds 是否有效
                if torch.isnan(image_embeds).any() or torch.isinf(image_embeds).any():
                    print(f"警告: 归一化后的 image_embeds 包含 NaN 或 Inf，跳过 CLIP 引导")
                    return original_latents
                
                # 计算 CLIP loss: L_CLIP = -λ * cos(f_img(I_t), f_text(c_i))
                cosine_sim = (image_embeds * self.clip_text_embeds).sum(dim=-1).mean()
                
                # 检查 cosine_sim 是否有效
                if torch.isnan(cosine_sim) or torch.isinf(cosine_sim):
                    print(f"警告: cosine_sim 为 NaN 或 Inf，跳过 CLIP 引导")
                    return original_latents
                
                clip_loss = -self.Clip_guidance_lambda * cosine_sim
                
                # 检查 clip_loss 是否有效
                if torch.isnan(clip_loss) or torch.isinf(clip_loss):
                    print(f"警告: clip_loss 为 NaN 或 Inf，跳过 CLIP 引导")
                    return original_latents
                
                # 计算梯度（相对于 scaled_latents）
                try:
                    grad_scaled = torch.autograd.grad(
                        clip_loss, 
                        scaled_latents, 
                        retain_graph=False, 
                        create_graph=False, 
                        allow_unused=False
                    )[0]
                except RuntimeError as e:
                    print(f"警告: 梯度计算失败: {e}，跳过 CLIP 引导")
                    return original_latents
                
                # 检查梯度是否有效（不包含 NaN 或 Inf）
                if grad_scaled is None:
                    print(f"警告: 梯度为 None，跳过 CLIP 引导")
                    return original_latents
                    
                if torch.isnan(grad_scaled).any() or torch.isinf(grad_scaled).any():
                    nan_count = torch.isnan(grad_scaled).sum().item()
                    inf_count = torch.isinf(grad_scaled).sum().item()
                    print(f"警告: CLIP 引导梯度包含 NaN ({nan_count} 个) 或 Inf ({inf_count} 个)，跳过此步骤 (timestep={timestep_index})")
                    return original_latents
                
                # 将梯度从 scaled_latents 空间转换回原始 latents 空间
                # 由于 scaled_latents = latents * grad_scale (或 latents / scaling_factor)
                # 所以 grad_latents = grad_scaled * grad_scale
                if has_latents_mean and has_latents_std:
                    # SDXL: scaled_latents = latents * latents_std / scaling_factor + latents_mean
                    # 梯度需要乘以 latents_std / scaling_factor（忽略常数项 latents_mean）
                    grad = grad_scaled * grad_scale
                else:
                    # 非 SDXL: scaled_latents = latents / scaling_factor
                    # 梯度需要除以 scaling_factor，即乘以 1/scaling_factor
                    grad = grad_scaled * grad_scale
                
                # 检查缩放后的梯度是否有效
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    print(f"警告: 缩放后的梯度包含 NaN 或 Inf，跳过 CLIP 引导")
                    return original_latents
                
                # 梯度归一化，防止梯度爆炸（使用更保守的裁剪）
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1, keepdim=True).view(-1, 1, 1, 1)
                
                # 检查 grad_norm 是否有效
                if torch.isnan(grad_norm).any() or torch.isinf(grad_norm).any():
                    print(f"警告: 梯度范数包含 NaN 或 Inf，跳过 CLIP 引导")
                    return original_latents
                
                # 使用更保守的梯度裁剪，并添加 epsilon 防止除零
                max_grad_norm = 1.0  # 更保守的梯度裁剪
                eps = 1e-8
                grad_normalized = grad / torch.clamp(grad_norm / max_grad_norm, min=1.0)
                
                # 检查归一化后的梯度是否有效
                if torch.isnan(grad_normalized).any() or torch.isinf(grad_normalized).any():
                    print(f"警告: 归一化后的梯度包含 NaN 或 Inf，跳过 CLIP 引导")
                    return original_latents
                
                # 进一步裁剪梯度值，防止极端值
                grad = torch.clamp(grad_normalized, -0.5, 0.5)
                
                # 检查裁剪后的梯度是否有效
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    print(f"警告: 裁剪后的梯度包含 NaN 或 Inf，跳过 CLIP 引导")
                    return original_latents
                
                # 更新 latents: Z_t ← Z_t - η * ∇_z L_CLIP
                # 注意：这里使用 original_latents 而不是 latents，确保基于原始值更新
                # 限制 eta 的最大值，防止更新过大
                safe_eta = min(self.Clip_guidance_eta, 0.1)  # 限制最大 eta 为 0.1
                latents_updated = original_latents - safe_eta * grad
                
                # 检查更新后的 latents 是否有效
                if torch.isnan(latents_updated).any() or torch.isinf(latents_updated).any():
                    print(f"警告: 更新后的 latents 包含 NaN 或 Inf，跳过 CLIP 引导")
                    return original_latents
                
                # 限制 latents 的范围，防止超出有效值域（使用更保守的范围）
                # 根据 SDXL 的 latents 分布，通常范围在 [-2, 2] 左右
                latents_updated = torch.clamp(latents_updated, -5.0, 5.0)
                
                # 最终检查
                if torch.isnan(latents_updated).any() or torch.isinf(latents_updated).any():
                    print(f"警告: 最终 latents 包含 NaN 或 Inf，返回原始 latents")
                    return original_latents
                
                # 将 latents 分离梯度，避免影响后续计算
                return latents_updated.detach()
                
        except Exception as e:
            print(f"警告: CLIP 引导应用失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回原始 latents，确保不会返回损坏的值
            return original_latents if 'original_latents' in locals() else latents
        
        return latents

    def print_attributes(self):
        """
        Prints all attributes and their values of the object.
        """
        for attr, value in vars(self).items():
            print(f"{attr}: {value}")
