import torch
from typing import Optional
from PIL import Image
from diffusers import AutoencoderKL, EulerDiscreteScheduler, EDMDPMSolverMultistepScheduler
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from scipy.spatial.distance import cdist
import numpy as np
import unet.pipeline_stable_diffusion_xl as pipeline_stable_diffusion_xl
from torch.fft import fftn, fftshift, ifftn, ifftshift
from typing import Optional, Tuple

from unet.unet import UNet2DConditionModel
from unet.unet_controller import UNetController

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os


def ipca(q, k, v, scale, unet_controller: Optional[UNetController] = None): # eg. q: [4,20,1024,64] k,v: [4,20,77,64] 
    q_neg, q_pos = torch.split(q, q.size(0) // 2, dim=0)
    k_neg, k_pos = torch.split(k, k.size(0) // 2, dim=0)
    v_neg, v_pos = torch.split(v, v.size(0) // 2, dim=0)

    # 1. negative_attn

    scores_neg = torch.matmul(q_neg, k_neg.transpose(-2, -1)) * scale
    attn_weights_neg = torch.softmax(scores_neg, dim=-1)
    attn_output_neg = torch.matmul(attn_weights_neg, v_neg)

    # 2. positive_attn (we do ipca only on positive branch)

    # 2.1 ipca 
    k_plus = torch.cat(tuple(k_pos.transpose(-2, -1)), dim=2).unsqueeze(0).repeat(k_pos.size(0),1,1,1) # 𝐾+ = [𝐾1 ⊕ 𝐾2 ⊕ . . . ⊕ 𝐾𝑁 ]
    v_plus = torch.cat(tuple(v_pos), dim=1).unsqueeze(0).repeat(v_pos.size(0),1,1,1) # 𝑉+ = [𝑉1 ⊕ 𝑉2 ⊕ . . . ⊕ 𝑉𝑁 ]


    # 2.2 apply mask
    if unet_controller is not None:
        scores_pos = torch.matmul(q_pos, k_plus) * scale

        
        # 2.2.1 apply dropout mask
        dropout_mask = gen_dropout_mask(scores_pos.shape, unet_controller, unet_controller.Ipca_dropout) # eg: [a,1024,154]   


        # 2.2.3 apply embeds mask
        if unet_controller.Use_embeds_mask:
            apply_embeds_mask(unet_controller,dropout_mask, add_eot=False)

        mask = dropout_mask

        mask = mask.unsqueeze(1).repeat(1,scores_pos.size(1),1,1)
        attn_weights_pos = torch.softmax(scores_pos + torch.log(mask), dim=-1)

    else:
        scores_pos = torch.matmul(q_pos, k_plus) * scale
        attn_weights_pos = torch.softmax(scores_pos, dim=-1)


    attn_output_pos = torch.matmul(attn_weights_pos, v_plus)
    # 3. combine
    attn_output = torch.cat((attn_output_neg, attn_output_pos), dim=0)

    return attn_output


def ipca2(q, k, v, scale, unet_controller: Optional[UNetController] = None): # eg. q: [4,20,1024,64] k,v: [4,20,77,64] 
    if unet_controller.ipca_time_step != unet_controller.current_time_step:
        unet_controller.ipca_time_step = unet_controller.current_time_step
        unet_controller.ipca2_index = 0
    else:
        unet_controller.ipca2_index += 1

    if unet_controller.Store_qkv is True:

        key = f"cross {unet_controller.current_time_step} {unet_controller.current_unet_position} {unet_controller.ipca2_index}"
        unet_controller.k_store[key] = k
        unet_controller.v_store[key] = v

        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
    else:
        # batch > 1
        if unet_controller.frame_prompt_express_list is not None:
            batch_size = q.size(0) // 2
            attn_output_list = []

            for i in range(batch_size):
                q_i = q[[i, i + batch_size], :, :, :]
                k_i = k[[i, i + batch_size], :, :, :]
                v_i = v[[i, i + batch_size], :, :, :]

                q_neg_i, q_pos_i = torch.split(q_i, q_i.size(0) // 2, dim=0)
                k_neg_i, k_pos_i = torch.split(k_i, k_i.size(0) // 2, dim=0)
                v_neg_i, v_pos_i = torch.split(v_i, v_i.size(0) // 2, dim=0)

                key = f"cross {unet_controller.current_time_step} {unet_controller.current_unet_position} {unet_controller.ipca2_index}"
                q_store = q_i
                k_store = unet_controller.k_store[key]
                v_store = unet_controller.v_store[key]

                q_store_neg, q_store_pos = torch.split(q_store, q_store.size(0) // 2, dim=0)
                k_store_neg, k_store_pos = torch.split(k_store, k_store.size(0) // 2, dim=0)
                v_store_neg, v_store_pos = torch.split(v_store, v_store.size(0) // 2, dim=0)    

                q_neg = torch.cat((q_neg_i, q_store_neg), dim=0)
                q_pos = torch.cat((q_pos_i, q_store_pos), dim=0)
                k_neg = torch.cat((k_neg_i, k_store_neg), dim=0)
                k_pos = torch.cat((k_pos_i, k_store_pos), dim=0)
                v_neg = torch.cat((v_neg_i, v_store_neg), dim=0)
                v_pos = torch.cat((v_pos_i, v_store_pos), dim=0)

                q_i = torch.cat((q_neg, q_pos), dim=0)
                k_i = torch.cat((k_neg, k_pos), dim=0)
                v_i = torch.cat((v_neg, v_pos), dim=0)

                attn_output_i = ipca(q_i, k_i, v_i, scale, unet_controller)
                attn_output_i = attn_output_i[[0, 2], :, :, :]
                attn_output_list.append(attn_output_i)
            
            attn_output_ = torch.cat(attn_output_list, dim=0)
            attn_output = torch.zeros(size=(q.size(0), attn_output_i.size(1), attn_output_i.size(2), attn_output_i.size(3)), device=q.device, dtype=q.dtype)
            for i in range(batch_size):
                attn_output[i] = attn_output_[i*2]
            for i in range(batch_size):
                attn_output[i + batch_size] = attn_output_[i*2 + 1]
        # batch = 1
        else:
            q_neg, q_pos = torch.split(q, q.size(0) // 2, dim=0)
            k_neg, k_pos = torch.split(k, k.size(0) // 2, dim=0)
            v_neg, v_pos = torch.split(v, v.size(0) // 2, dim=0)

            key = f"cross {unet_controller.current_time_step} {unet_controller.current_unet_position} {unet_controller.ipca2_index}"
            q_store = q
            k_store = unet_controller.k_store[key]
            v_store = unet_controller.v_store[key]

            q_store_neg, q_store_pos = torch.split(q_store, q_store.size(0) // 2, dim=0)
            k_store_neg, k_store_pos = torch.split(k_store, k_store.size(0) // 2, dim=0)
            v_store_neg, v_store_pos = torch.split(v_store, v_store.size(0) // 2, dim=0)    

            q_neg = torch.cat((q_neg, q_store_neg), dim=0)
            q_pos = torch.cat((q_pos, q_store_pos), dim=0)
            k_neg = torch.cat((k_neg, k_store_neg), dim=0)
            k_pos = torch.cat((k_pos, k_store_pos), dim=0)
            v_neg = torch.cat((v_neg, v_store_neg), dim=0)
            v_pos = torch.cat((v_pos, v_store_pos), dim=0)

            q = torch.cat((q_neg, q_pos), dim=0)
            k = torch.cat((k_neg, k_pos), dim=0)
            v = torch.cat((v_neg, v_pos), dim=0)

            attn_output = ipca(q, k, v, scale, unet_controller)
            attn_output = attn_output[[0, 2], :, :, :]
    
    return attn_output


def apply_embeds_mask(unet_controller: Optional[UNetController],dropout_mask, add_eot=False):   
    id_prompt = unet_controller.id_prompt
    prompt_tokens = prompt2tokens(unet_controller.tokenizer,unet_controller.prompts[0])
    
    words_tokens = prompt2tokens(unet_controller.tokenizer,id_prompt)
    words_tokens = [word for word in words_tokens if word != '<|endoftext|>' and word != '<|startoftext|>']
    index_of_words = find_sublist_index(prompt_tokens,words_tokens)    
    index_list = [index+77 for index in range(index_of_words, index_of_words+len(words_tokens))]
    if add_eot:
        index_list.extend([index+77 for index, word in enumerate(prompt_tokens) if word == '<|endoftext|>'])

    mask_indices = torch.arange(dropout_mask.size(-1), device=dropout_mask.device)
    mask = (mask_indices >= 78) & (~torch.isin(mask_indices, torch.tensor(index_list, device=dropout_mask.device)))
    dropout_mask[0, :, mask] = 0


def gen_dropout_mask(out_shape, unet_controller: Optional[UNetController], drop_out):
    gen_length = out_shape[3]
    attn_map_side_length = out_shape[2]

    batch_num = out_shape[0]
    mask_list = []
    
    for prompt_index in range(batch_num):
        start = prompt_index * int(gen_length / batch_num)
        end = (prompt_index + 1) * int(gen_length / batch_num)
    
        mask = torch.bernoulli(torch.full((attn_map_side_length,gen_length), 1 - drop_out, dtype=unet_controller.torch_dtype, device=unet_controller.device))        
        mask[:, start:end] = 1

        mask_list.append(mask)

    concatenated_mask = torch.stack(mask_list, dim=0)
    return concatenated_mask


def load_pipe_from_path(model_path, device, torch_dtype, variant):
    # 检查设备是否可用
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA不可用，但尝试使用设备: {device}")
        # 如果是cuda:X格式，检查设备索引
        if ':' in device:
            device_idx = int(device.split(':')[1])
            if device_idx >= torch.cuda.device_count():
                raise RuntimeError(f"设备 {device} 不存在。系统只有 {torch.cuda.device_count()} 个GPU (0-{torch.cuda.device_count()-1})")
    
    model_name = model_path.split('/')[-1]
    if model_path.split('/')[-1] == 'playground-v2.5-1024px-aesthetic':
        scheduler = EDMDPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler", torch_dtype=torch_dtype, variant=variant,)
    else:
        scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler", torch_dtype=torch_dtype, variant=variant,)
    
    if model_path.split('/')[-1] == 'Juggernaut-X-v10' or model_path.split('/')[-1] == 'Juggernaut-XI-v11':
        variant = None

    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=torch_dtype, variant=variant,)
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", torch_dtype=torch_dtype, variant=variant,)
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2", torch_dtype=torch_dtype, variant=variant,)
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=torch_dtype, variant=variant,)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder="text_encoder_2", torch_dtype=torch_dtype, variant=variant,)
    unet_new = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=torch_dtype, variant=variant,)
    
    pipe = pipeline_stable_diffusion_xl.StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=unet_new,
        scheduler=scheduler,
    )
    
    try:
        pipe.to(device)
    except RuntimeError as e:
        if "invalid device ordinal" in str(e) or "CUDA error" in str(e):
            raise RuntimeError(f"无法将模型移动到设备 {device}。请检查设备是否可用。系统有 {torch.cuda.device_count()} 个GPU。") from e
        raise

    return pipe, model_name, tokenizer


def get_max_window_length(unet_controller: Optional[UNetController],id_prompt, frame_prompt_list):
    single_long_prompt = id_prompt
    max_window_length = 0
    for index, movement in enumerate(frame_prompt_list):
        single_long_prompt += ' ' + movement
        token_length = len(single_long_prompt.split())
        if token_length >= 77:
            break
        max_window_length += 1
    return max_window_length


def movement_gen_story_slide_windows(id_prompt, frame_prompt_list, pipe, window_length, seed, unet_controller: Optional[UNetController], save_dir, verbose=True):  
    import os
    # 设置保存目录，用于直接保存modulation scores可视化
    if unet_controller is not None:
        unet_controller.result_save_dir = save_dir
    max_window_length = get_max_window_length(unet_controller,id_prompt,frame_prompt_list)
    window_length = min(window_length,max_window_length)
    if window_length < len(frame_prompt_list):
        movement_lists = circular_sliding_windows(frame_prompt_list, window_length)
    else:
        movement_lists = [movement for movement in frame_prompt_list]
    story_images = []


    if verbose: 
        print("seed:", seed)
    generate = torch.Generator().manual_seed(seed)
    unet_controller.id_prompt = id_prompt

    for index, movement in enumerate(frame_prompt_list):
        # 重置注意力存储（每张图生成前）
        # 如果启用了modulation scores可视化，不清空modulation_scores_store，以便在最后统一可视化
        if unet_controller is not None:
            clear_modulation = not (unet_controller.Visualize_modulation_scores if hasattr(unet_controller, 'Visualize_modulation_scores') else False)
            unet_controller.reset_attn_store(clear_modulation_scores=clear_modulation)
            # 设置当前帧索引（用于判断是否是第一张图）
            unet_controller.current_frame_index = index
            # 如果是第一张图，重置第一张图的k和v存储
            if index == 0:
                unet_controller.reset_first_frame_kv()
        if unet_controller is not None:
            if window_length < len(frame_prompt_list):
                unet_controller.frame_prompt_suppress = movement_lists[index][1:]
                unet_controller.frame_prompt_express = movement_lists[index][0]
                # 根据Prompt_embeds_mode选择不同的prompt拼接方式
                if unet_controller.Prompt_embeds_mode in ['svr-single', 'single']:
                    # 新模式：id_prompt + 单个express prompt
                    gen_propmts = [f'{id_prompt} {movement_lists[index][0]}']
                else:
                    # 原模式：id_prompt + 所有express prompt串联
                    gen_propmts = [f'{id_prompt} {" ".join(movement_lists[index])}']

            else:
                unet_controller.frame_prompt_suppress = movement_lists[:index] + movement_lists[index+1:]
                unet_controller.frame_prompt_express = movement_lists[index]
                # 根据Prompt_embeds_mode选择不同的prompt拼接方式
                if unet_controller.Prompt_embeds_mode in ['svr-single', 'single']:
                    # 新模式：id_prompt + 单个express prompt
                    gen_propmts = [f'{id_prompt} {movement_lists[index]}']
                else:
                    # 原模式：id_prompt + 所有express prompt串联
                    gen_propmts = [f'{id_prompt} {" ".join(movement_lists)}']
            
            if verbose:
                print(f"suppress: {unet_controller.frame_prompt_suppress}")
                print(f"express: {unet_controller.frame_prompt_express}")
                print(f'id_prompt: {id_prompt}')
                print(f"gen_propmts: {gen_propmts}")


        else:
            gen_propmts = f'{id_prompt} {movement}'

        if unet_controller is not None and unet_controller.Use_same_init_noise is True:     
            generate = torch.Generator().manual_seed(seed)

        images = pipe(gen_propmts, generator=generate, unet_controller=unet_controller).images
        story_images.append(images[0])
        images[0].save(os.path.join(save_dir, f'{id_prompt} {unet_controller.frame_prompt_express}.jpg'))

        # 保存该图像生成过程中的交叉注意力map（分别可视化down/mid/up三部分）
        if (unet_controller is not None and 
            unet_controller.tokenizer is not None and 
            unet_controller.Visualize_attn_map):
            try:
                from unet.atten_store import visualize_cross_attn_from_controller
                cross_attn_save_dir = os.path.join(save_dir, "cross_attn_vis", f"frame_{index:03d}")
                os.makedirs(cross_attn_save_dir, exist_ok=True)
                
                # 使用当前帧的prompt进行可视化
                current_prompt = gen_propmts[0] if isinstance(gen_propmts, list) else gen_propmts
                
                # 分别可视化down/mid/up三部分
                layer_groups = {
                    "down": ["down0", "down1", "down2"],
                    "mid": ["mid"],
                    "up": ["up0", "up1", "up2"]
                }
                
                for layer_name, layer_list in layer_groups.items():
                    cross_attn_vis = visualize_cross_attn_from_controller(
                        tokenizer=unet_controller.tokenizer,
                        prompt=current_prompt,
                        unet_controller=unet_controller,
                        res=32,  # 图像分块分辨率（32x32=1024像素）
                        from_where=layer_list,  # 分别指定down/mid/up
                        save_dir=cross_attn_save_dir,
                        save_filename=f"cross_attn_{layer_name}.png",  # 使用不同的文件名
                        colormap="white",  # 使用白色配色
                        show_fig=False
                    )
                    if verbose:
                        print(f"交叉注意力可视化完成 ({layer_name})，保存至：{cross_attn_save_dir}")
                
            except Exception as e:
                if verbose:
                    print(f"交叉注意力可视化失败：{e}")
                    import traceback
                    traceback.print_exc()

        


    image_array_list = [np.array(pil_img) for pil_img in story_images]

    # Concatenate images horizontally
    story_image = np.concatenate(image_array_list, axis=1)
    story_image = Image.fromarray(story_image.astype(np.uint8))

    if unet_controller.Save_story_image:
        story_image.save(os.path.join(save_dir, f'story_image_{id_prompt}.jpg'))

    return story_images, story_image

# this function set batch > 1 to generate multiple images at once
def movement_gen_story_slide_windows_batch(id_prompt, frame_prompt_list, pipe, window_length, seed, unet_controller: Optional[UNetController], save_dir, batch_size=3):  
    import os
    max_window_length = get_max_window_length(unet_controller,id_prompt,frame_prompt_list)
    window_length = min(window_length,max_window_length)
    if window_length < len(frame_prompt_list):
        movement_lists = circular_sliding_windows(frame_prompt_list, window_length)
    else:
        movement_lists = [movement for movement in frame_prompt_list]
    story_images = []

    print("seed:", seed)
    generate = torch.Generator().manual_seed(seed)
    unet_controller.id_prompt = id_prompt

    gen_prompt_info_list = []
    gen_prompt = None
    for index, _ in enumerate(frame_prompt_list):
        if window_length < len(frame_prompt_list):
            frame_prompt_suppress = movement_lists[index][1:]
            frame_prompt_express = movement_lists[index][0]
            # 根据Prompt_embeds_mode选择不同的prompt拼接方式
            if unet_controller.Prompt_embeds_mode in ['svr-single', 'single']:
                # 新模式：id_prompt + 单个express prompt
                gen_prompt = f'{id_prompt} {movement_lists[index][0]}'
            else:
                # 原模式：id_prompt + 所有express prompt串联
                gen_prompt = f'{id_prompt} {" ".join(movement_lists[index])}'

        else:
            frame_prompt_suppress = movement_lists[:index] + movement_lists[index+1:]
            frame_prompt_express = movement_lists[index]
            # 根据Prompt_embeds_mode选择不同的prompt拼接方式
            if unet_controller.Prompt_embeds_mode in ['svr-single', 'single']:
                # 新模式：id_prompt + 单个express prompt
                gen_prompt = f'{id_prompt} {movement_lists[index]}'
            else:
                # 原模式：id_prompt + 所有express prompt串联
                gen_prompt = f'{id_prompt} {" ".join(movement_lists)}'

        gen_prompt_info_list.append({'frame_prompt_suppress': frame_prompt_suppress, 'frame_prompt_express': frame_prompt_express, 'gen_prompt': gen_prompt})
    
    story_images = []
    for i in range(0, len(gen_prompt_info_list), batch_size):
        batch = gen_prompt_info_list[i:i + batch_size]
        gen_prompts = [gen_prompt_info['gen_prompt'] for gen_prompt_info in batch]
        unet_controller.frame_prompt_express_list = [gen_prompt_info['frame_prompt_express'] for gen_prompt_info in batch]
        unet_controller.frame_prompt_suppress_list = [gen_prompt_info['frame_prompt_suppress'] for gen_prompt_info in batch]
                
        if unet_controller is not None and unet_controller.Use_same_init_noise is True:     
            generate = torch.Generator().manual_seed(seed)
        
        images = pipe(gen_prompts, generator=generate, unet_controller=unet_controller).images    
        for index,image in enumerate(images):
            story_images.append(image)
            image.save(os.path.join(save_dir, f'{id_prompt} {unet_controller.frame_prompt_express_list[index]}.jpg'))

    image_array_list = [np.array(pil_img) for pil_img in story_images]

    # Concatenate images horizontally
    story_image = np.concatenate(image_array_list, axis=1)
    story_image = Image.fromarray(story_image.astype(np.uint8))

    if unet_controller.Save_story_image:
        story_image.save(os.path.join(save_dir, 'story_image.jpg'))

    return story_images, story_image


def prompt2tokens(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    tokens = []
    for text_input_id in text_input_ids[0]:
        token = tokenizer.decoder[text_input_id.item()]
        tokens.append(token)
    return tokens


def punish_wight(tensor, latent_size, alpha=1.0, beta=1.2, calc_similarity=False):
    u, s, vh = torch.linalg.svd(tensor)
    u = u[:,:latent_size]
    zero_idx = int(latent_size * alpha)

    if calc_similarity:
        _s = s.clone()
        _s *= torch.exp(-alpha*_s) * beta
        _s[zero_idx:] = 0
        _tensor = u @ torch.diag(_s) @ vh
        dist = cdist(tensor[:,0].unsqueeze(0).cpu(), _tensor[:,0].unsqueeze(0).cpu(), metric='cosine')
        print(f'The distance between the word embedding before and after the punishment: {dist}')
    s *= torch.exp(-alpha*s) * beta
    tensor = u @ torch.diag(s) @ vh
    return tensor


def swr_single_prompt_embeds(swr_words,prompt_embeds,prompt,tokenizer,alpha=1.0, beta=1.2, zero_eot=False):
    punish_indices = []

    prompt_tokens = prompt2tokens(tokenizer,prompt)
    
    words_tokens = prompt2tokens(tokenizer,swr_words)
    words_tokens = [word for word in words_tokens if word != '<|endoftext|>' and word != '<|startoftext|>']
    index_of_words = find_sublist_index(prompt_tokens,words_tokens)
    
    if index_of_words != -1:
        punish_indices.extend([num for num in range(index_of_words, index_of_words+len(words_tokens))])
    
    if zero_eot:
        eot_indices = [index for index, word in enumerate(prompt_tokens) if word == '<|endoftext|>']
        prompt_embeds[eot_indices] *= 9e-1
        pass
    else:
        punish_indices.extend([index for index, word in enumerate(prompt_tokens) if word == '<|endoftext|>'])

    punish_indices = list(set(punish_indices))
    
    wo_batch = prompt_embeds[punish_indices]
    wo_batch = punish_wight(wo_batch.T.to(float), wo_batch.size(0), alpha=alpha, beta=beta, calc_similarity=False).T.to(prompt_embeds.dtype)

    prompt_embeds[punish_indices] = wo_batch


def find_sublist_index(list1, list2):
    for i in range(len(list1) - len(list2) + 1):
        if list1[i:i + len(list2)] == list2:
            return i
    return -1  # If sublist is not found


def fourier_filter(x_in: "torch.Tensor", threshold: int, scale: int) -> "torch.Tensor":
    """Fourier filter as introduced in FreeU (https://arxiv.org/abs/2309.11497).

    This version of the method comes from here:
    https://github.com/huggingface/diffusers/pull/5164#issuecomment-1732638706
    """
    x = x_in
    B, C, H, W = x.shape

    x = x.to(dtype=torch.float32)

    # FFT
    x_freq = fftn(x, dim=(-2, -1))
    x_freq = fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)

    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold : crow + threshold, ccol - threshold : ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = ifftshift(x_freq, dim=(-2, -1))
    x_filtered = ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered.to(dtype=x_in.dtype)


def apply_freeu(
    resolution_idx: int, hidden_states: "torch.Tensor", res_hidden_states: "torch.Tensor", **freeu_kwargs
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Applies the FreeU mechanism as introduced in https:
    //arxiv.org/abs/2309.11497. Adapted from the official code repository: https://github.com/ChenyangSi/FreeU.

    Args:
        resolution_idx (`int`): Integer denoting the UNet block where FreeU is being applied.
        hidden_states (`torch.Tensor`): Inputs to the underlying block.
        res_hidden_states (`torch.Tensor`): Features from the skip block corresponding to the underlying block.
        s1 (`float`): Scaling factor for stage 1 to attenuate the contributions of the skip features.
        s2 (`float`): Scaling factor for stage 2 to attenuate the contributions of the skip features.
        b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
        b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
    """
    if resolution_idx == 0:
        num_half_channels = hidden_states.shape[1] // 2
        hidden_states[:, :num_half_channels] = hidden_states[:, :num_half_channels] * freeu_kwargs["b1"]
        res_hidden_states = fourier_filter(res_hidden_states, threshold=1, scale=freeu_kwargs["s1"])
    if resolution_idx == 1:
        num_half_channels = hidden_states.shape[1] // 2
        hidden_states[:, :num_half_channels] = hidden_states[:, :num_half_channels] * freeu_kwargs["b2"]
        res_hidden_states = fourier_filter(res_hidden_states, threshold=1, scale=freeu_kwargs["s2"])

    return hidden_states, res_hidden_states


def circular_sliding_windows(lst, w):
    n = len(lst)
    windows = []
    for i in range(n):
        window = [lst[(i + j) % n] for j in range(w)]
        windows.append(window)
    return


def visualize_image_text_cross_attn(
        attn_weights,  # 交叉注意力权重 (t_q, t_k) 或 (num_heads, t_q, t_k)
        text_tokens,  # 文本Token列表（如["jeep", "countryside"]）
        target_token,  # 要可视化的目标Token（如"jeep"）
        image_size=(512, 512),  # 原始图像尺寸
        patch_size=(32, 32),  # 图像分块大小
        save_path="./image_text_attn.png",
        dpi=200
):
    """
    生成“图像分块-文本Token”对应的交叉注意力可视化（与示例一致）
    """
    # -------------------------- 1. 预处理注意力权重 --------------------------
    # 转换为NumPy数组
    if hasattr(attn_weights, 'detach'):
        attn_weights = attn_weights.detach().cpu().numpy()
    # 若为多头权重，取均值
    if len(attn_weights.shape) == 3:
        attn_weights = np.mean(attn_weights, axis=0)  # (t_q, t_k)

    # 找到目标Token对应的索引
    if target_token not in text_tokens:
        raise ValueError(f"目标Token {target_token} 不在文本Token列表中")
    token_idx = text_tokens.index(target_token)
    # 提取目标Token对应的注意力权重（t_q,）
    token_attn = attn_weights[:, token_idx]

    # -------------------------- 2. 图像分块与注意力映射 --------------------------
    # 计算分块数量
    patch_num_h = image_size[0] // patch_size[0]
    patch_num_w = image_size[1] // patch_size[1]
    t_q = patch_num_h * patch_num_w
    assert t_q == attn_weights.shape[0], f"图像分块数 {t_q} 与Query序列长度 {attn_weights.shape[0]} 不匹配"

    # 将注意力权重重塑为图像分块网格
    attn_map = token_attn.reshape(patch_num_h, patch_num_w)
    # 归一化到[0,1]（增强可视化对比度）
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # -------------------------- 3. 可视化布局（与示例一致） --------------------------
    fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={'width_ratios': [1, 1]})
    fig.subplots_adjust(wspace=0.1)  # 减小子图间距

    # -------- 左侧：注意力加权的图像分块 --------
    # 绘制注意力热图（模拟模糊图像效果）
    im = ax_img.imshow(attn_map, cmap='viridis', aspect='equal', interpolation='bilinear')
    # 添加虚线边框（与示例一致）
    for spine in ax_img.spines.values():
        spine.set_color('blue')
        spine.set_linestyle('--')
        spine.set_linewidth(2)
    # 隐藏坐标轴
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    # 添加Token标签（底部）
    ax_img.text(0.5, -0.1, target_token, ha='center', va='top', transform=ax_img.transAxes, fontsize=14,
                fontweight='bold')

    # -------- 右侧：文本Token（模拟模糊背景） --------
    # 绘制模糊背景
    ax_text.imshow(np.random.rand(patch_num_h, patch_num_w) * 0.2, cmap='gray', aspect='equal',
                   interpolation='bilinear')
    # 添加Token文本
    ax_text.text(0.5, 0.5, text_tokens[1] if len(text_tokens) >= 2 else "text", ha='center', va='center',
                 fontsize=14, fontweight='bold', color='black', transform=ax_text.transAxes)
    # 隐藏坐标轴
    ax_text.set_xticks([])
    ax_text.set_yticks([])
    # 添加虚线边框
    for spine in ax_text.spines.values():
        spine.set_color('blue')
        spine.set_linestyle('--')
        spine.set_linewidth(2)

    # -------- 中间连接点（与示例一致） --------
    # 在两个子图之间添加三个点
    fig.text(0.5, 0.5, '...', ha='center', va='center', fontsize=20, color='black')

    # -------------------------- 4. 保存图像 --------------------------
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"图像-文本交叉注意力图已保存至：{save_path}")


def save_single_modulation_score(
    before_scores: torch.Tensor,
    after_scores: torch.Tensor,
    timestep: int,
    block_position: str,
    save_dir: str,
    frame_index: int = None,
    is_cross_attention: bool = False,
    tokenizer=None,
    prompt: str = None
):
    """
    直接保存单个timestep和block的modulation scores可视化
    :param before_scores: modulation前的scores，形状为(seq_len_q, seq_len_k)
    :param after_scores: modulation后的scores，形状为(seq_len_q, seq_len_k)
    :param timestep: timestep数值
    :param block_position: block位置字符串，格式为"{unet_position}_{attention_type}_{layer_idx}"
    :param save_dir: 保存目录
    :param frame_index: 当前帧索引（用于创建单独的文件夹）
    :param is_cross_attention: 是否为cross-attention（需要按token可视化）
    :param tokenizer: tokenizer（用于decode tokens）
    :param prompt: prompt文本（用于显示）
    """
    import os
    import math
    import torch.nn.functional as F
    
    # 为每张图像创建单独的文件夹
    if frame_index is not None:
        frame_dir = os.path.join(save_dir, f"frame_{frame_index:03d}")
    else:
        frame_dir = save_dir
    os.makedirs(frame_dir, exist_ok=True)
    
    # 获取原始空间维度（假设是正方形）
    seq_len_q = before_scores.shape[0]
    original_size = int(math.sqrt(seq_len_q))
    
    # 如果seq_len_q不是完全平方数，跳过可视化
    if original_size * original_size != seq_len_q:
        return
    
    # 对于cross-attention，按token可视化并横向拼接
    if is_cross_attention and tokenizer is not None and prompt is not None:
        seq_len_k = before_scores.shape[1]
        
        # 获取prompt tokens
        try:
            prompt_tokens = prompt2tokens(tokenizer, prompt)
            # 过滤特殊token
            prompt_tokens = [tok for tok in prompt_tokens if tok not in ['<|endoftext|>', '<|startoftext|>']]
        except:
            prompt_tokens = None
        
        # 为每个token创建可视化
        token_images_before = []
        token_images_after = []
        token_texts = []
        
        # 遍历每个key token
        for token_idx in range(min(seq_len_k, len(prompt_tokens) if prompt_tokens else seq_len_k)):
            # 提取该token对应的attention scores (seq_len_q,)
            before_token_scores = before_scores[:, token_idx]  # (seq_len_q,)
            after_token_scores = after_scores[:, token_idx]    # (seq_len_q,)
            
            # 归一化
            before_sum = before_token_scores.sum()
            after_sum = after_token_scores.sum()
            if before_sum > 0:
                before_token_scores = before_token_scores / before_sum
            if after_sum > 0:
                after_token_scores = after_token_scores / after_sum
            
            # Reshape为2D
            before_vis_2d = before_token_scores.view(1, 1, original_size, original_size)
            after_vis_2d = after_token_scores.view(1, 1, original_size, original_size)
            
            # 下采样到目标分辨率
            target_size = min(256, original_size) if original_size < 256 else 256
            before_vis_2d_resized = F.interpolate(
                before_vis_2d, 
                size=(target_size, target_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0).squeeze(0)
            
            after_vis_2d_resized = F.interpolate(
                after_vis_2d, 
                size=(target_size, target_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0).squeeze(0)
            
            # 归一化到0-1
            before_min = before_vis_2d_resized.min()
            before_max = before_vis_2d_resized.max()
            if before_max > before_min:
                before_vis_2d_resized = (before_vis_2d_resized - before_min) / (before_max - before_min)
            else:
                before_vis_2d_resized = torch.zeros_like(before_vis_2d_resized)
            
            after_min = after_vis_2d_resized.min()
            after_max = after_vis_2d_resized.max()
            if after_max > after_min:
                after_vis_2d_resized = (after_vis_2d_resized - after_min) / (after_max - after_min)
            else:
                after_vis_2d_resized = torch.zeros_like(after_vis_2d_resized)
            
            # 转换为numpy
            before_np = (before_vis_2d_resized.numpy() * 255).astype(np.uint8)
            after_np = (after_vis_2d_resized.numpy() * 255).astype(np.uint8)
            
            token_images_before.append(before_np)
            token_images_after.append(after_np)
            
            # 获取token文本，去除</w>等标识
            if prompt_tokens and token_idx < len(prompt_tokens):
                token_text = prompt_tokens[token_idx]
                # 去除</w>等标识符
                token_text = token_text.replace('</w>', '').replace('<|endoftext|>', '').replace('<|startoftext|>', '')
                token_texts.append(token_text)
            else:
                token_texts.append(f"token_{token_idx}")
        
        # 横向拼接所有token的可视化
        if token_images_before:
            # 计算需要添加的文本高度
            text_height = 40
            img_height, img_width = token_images_before[0].shape
            total_width = img_width * len(token_images_before)
            total_height = img_height + text_height
            
            # 创建拼接图像（包含文本区域）
            before_concatenated = np.ones((total_height, total_width), dtype=np.uint8) * 255
            after_concatenated = np.ones((total_height, total_width), dtype=np.uint8) * 255
            
            # 放置图像
            for idx, (before_img, after_img) in enumerate(zip(token_images_before, token_images_after)):
                x_start = idx * img_width
                x_end = x_start + img_width
                before_concatenated[:img_height, x_start:x_end] = before_img
                after_concatenated[:img_height, x_start:x_end] = after_img
            
            # 转换为PIL Image并添加文本
            before_pil = Image.fromarray(before_concatenated, mode='L').convert('RGB')
            after_pil = Image.fromarray(after_concatenated, mode='L').convert('RGB')
            
            from PIL import ImageDraw, ImageFont
            draw_before = ImageDraw.Draw(before_pil)
            draw_after = ImageDraw.Draw(after_pil)
            
            # 尝试使用默认字体
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            except:
                font = ImageFont.load_default()
            
            # 在每个token图像下方添加文本
            for idx, token_text in enumerate(token_texts):
                x_start = idx * img_width
                x_center = x_start + img_width // 2
                y_text = img_height + 10
                text_bbox = draw_before.textbbox((0, 0), token_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                draw_before.text((x_center - text_width // 2, y_text), token_text, fill='black', font=font)
                draw_after.text((x_center - text_width // 2, y_text), token_text, fill='black', font=font)
            
            # 保存拼接后的图像
            before_path = os.path.join(frame_dir, f"t{timestep:03d}_{block_position}_before.png")
            after_path = os.path.join(frame_dir, f"t{timestep:03d}_{block_position}_after.png")
            before_pil.save(before_path)
            after_pil.save(after_path)
        
    else:
        # Self-attention: 使用原有的逻辑（对key维度求和）
        # 对key维度求和（归一化），得到每个query位置的注意力强度
        before_vis = before_scores.sum(dim=-1)  # (seq_len_q,)
        after_vis = after_scores.sum(dim=-1)    # (seq_len_q,)
        
        # 归一化到0-1范围（在求和之后归一化）
        before_sum = before_vis.sum()
        after_sum = after_vis.sum()
        if before_sum > 0:
            before_vis = before_vis / before_sum
        if after_sum > 0:
            after_vis = after_vis / after_sum
        
        # Reshape为2D空间 (original_size, original_size)
        before_vis_2d = before_vis.view(1, 1, original_size, original_size)  # (1, 1, H, W)
        after_vis_2d = after_vis.view(1, 1, original_size, original_size)   # (1, 1, H, W)
        
        # 使用双线性插值下采样到目标分辨率
        target_size = min(256, original_size) if original_size < 256 else 256
        before_vis_2d_resized = F.interpolate(
            before_vis_2d, 
            size=(target_size, target_size), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0).squeeze(0)  # (target_size, target_size)
        
        after_vis_2d_resized = F.interpolate(
            after_vis_2d, 
            size=(target_size, target_size), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0).squeeze(0)  # (target_size, target_size)
        
        # 再次归一化到0-1范围
        before_min = before_vis_2d_resized.min()
        before_max = before_vis_2d_resized.max()
        if before_max > before_min:
            before_vis_2d_resized = (before_vis_2d_resized - before_min) / (before_max - before_min)
        else:
            before_vis_2d_resized = torch.zeros_like(before_vis_2d_resized)
        
        after_min = after_vis_2d_resized.min()
        after_max = after_vis_2d_resized.max()
        if after_max > after_min:
            after_vis_2d_resized = (after_vis_2d_resized - after_min) / (after_max - after_min)
        else:
            after_vis_2d_resized = torch.zeros_like(after_vis_2d_resized)
        
        # 转换为numpy并映射为0-255的uint8
        before_np = (before_vis_2d_resized.numpy() * 255).astype(np.uint8)
        after_np = (after_vis_2d_resized.numpy() * 255).astype(np.uint8)
        
        # 使用PIL创建位图（灰度图，权重高的地方是白色）
        before_img = Image.fromarray(before_np, mode='L')
        after_img = Image.fromarray(after_np, mode='L')
        
        # 保存单独的位图文件
        before_bitmap_path = os.path.join(frame_dir, f"t{timestep:03d}_{block_position}_before.png")
        after_bitmap_path = os.path.join(frame_dir, f"t{timestep:03d}_{block_position}_after.png")
        before_img.save(before_bitmap_path)
        after_img.save(after_bitmap_path)


def visualize_modulation_scores(
    unet_controller: UNetController,
    save_dir: str = "result/Attention-modulation-visualization",
    save_filename: str = "modulation_scores.png"
):
    """
    可视化modulation前后的attention scores
    按照timestep数+block位置+modulation前后的格式以png保存在指定目录下
    :param unet_controller: UNet控制器（包含modulation_scores_store）
    :param save_dir: 保存目录（默认为result/Attention-modulation-visualization）
    :param save_filename: 保存文件名（用于汇总图，如果生成的话）
    """
    import os
    import math
    import torch.nn.functional as F
    
    if not unet_controller.modulation_scores_store:
        print("警告：没有找到modulation scores数据，跳过可视化")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 收集所有block的数据（包含timestep和block位置信息）
    block_scores = {}  # {(timestep, block_position): {'before': tensor, 'after': tensor}}
    
    for block_key, scores_dict in unet_controller.modulation_scores_store.items():
        # 解析block_key：{unet_position}_{is_self_attention}_{layer_idx}_t{timestep}
        # 例如：up1_cross_2_t50
        parts = block_key.split('_')
        if len(parts) < 4 or not parts[-1].startswith('t'):
            continue
        
        # 提取timestep
        timestep_str = parts[-1][1:]  # 去掉't'前缀
        try:
            timestep = int(timestep_str)
        except ValueError:
            continue
        
        # 提取block位置（去掉最后的_t{timestep}部分）
        # block_position格式：{unet_position}_{is_self_attention}_{layer_idx}
        block_position = '_'.join(parts[:-1])  # 例如：up1_cross_2
        
        before_scores = scores_dict.get('before')
        after_scores = scores_dict.get('after')
        
        if before_scores is None or after_scores is None:
            continue
        
        # 保存该block的scores
        block_scores[(timestep, block_position)] = {
            'before': before_scores,
            'after': after_scores
        }
    
    if not block_scores:
        print("警告：没有找到modulation scores数据，跳过可视化")
        return
    
    # 2. 对每个block进行可视化
    for (timestep, block_position) in sorted(block_scores.keys()):
        scores_dict = block_scores[(timestep, block_position)]
        before_scores = scores_dict['before']  # (seq_len_q, seq_len_k)，例如(4096, 77)
        after_scores = scores_dict['after']   # (seq_len_q, seq_len_k)
        
        # 对key维度求和（归一化），得到每个query位置的注意力强度
        # 对于cross-attention，对key维度求和
        before_vis = before_scores.sum(dim=-1)  # (seq_len_q,)，例如(4096,)
        after_vis = after_scores.sum(dim=-1)    # (seq_len_q,)
        
        # 归一化到0-1范围（在求和之后归一化）
        before_sum = before_vis.sum()
        after_sum = after_vis.sum()
        if before_sum > 0:
            before_vis = before_vis / before_sum
        if after_sum > 0:
            after_vis = after_vis / after_sum
        
        # 获取原始空间维度（假设是正方形）
        seq_len_q = before_vis.shape[0]
        original_size = int(math.sqrt(seq_len_q))
        
        # 如果seq_len_q不是完全平方数，跳过可视化
        if original_size * original_size != seq_len_q:
            print(f"警告：timestep {timestep} 的seq_len_q={seq_len_q}不是完全平方数，跳过可视化")
            continue
        
        # Reshape为2D空间 (original_size, original_size)
        before_vis_2d = before_vis.view(1, 1, original_size, original_size)  # (1, 1, H, W)
        after_vis_2d = after_vis.view(1, 1, original_size, original_size)   # (1, 1, H, W)
        
        # 使用双线性插值下采样到目标分辨率（提高分辨率以获得更清晰的可视化）
        # 如果原始分辨率较小，使用原始分辨率；否则使用128x128
        target_size = min(128, original_size) if original_size < 128 else 128
        before_vis_2d_resized = F.interpolate(
            before_vis_2d, 
            size=(target_size, target_size), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0).squeeze(0)  # (target_size, target_size)
        
        after_vis_2d_resized = F.interpolate(
            after_vis_2d, 
            size=(target_size, target_size), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0).squeeze(0)  # (target_size, target_size)
        
        # 再次归一化到0-1范围（确保在0-1之间）
        before_min = before_vis_2d_resized.min()
        before_max = before_vis_2d_resized.max()
        if before_max > before_min:
            before_vis_2d_resized = (before_vis_2d_resized - before_min) / (before_max - before_min)
        else:
            before_vis_2d_resized = torch.zeros_like(before_vis_2d_resized)
        
        after_min = after_vis_2d_resized.min()
        after_max = after_vis_2d_resized.max()
        if after_max > after_min:
            after_vis_2d_resized = (after_vis_2d_resized - after_min) / (after_max - after_min)
        else:
            after_vis_2d_resized = torch.zeros_like(after_vis_2d_resized)
        
        # 转换为numpy并映射为0-255的uint8（权重高的地方是白色，即255）
        before_np = (before_vis_2d_resized.numpy() * 255).astype(np.uint8)
        after_np = (after_vis_2d_resized.numpy() * 255).astype(np.uint8)
        
        # 使用PIL创建位图（灰度图，权重高的地方是白色）
        before_img = Image.fromarray(before_np, mode='L')
        after_img = Image.fromarray(after_np, mode='L')
        
        # 保存单独的位图文件，格式：timestep数+block位置+modulation前后
        # 例如：50_up1_cross_2_before.png, 50_up1_cross_2_after.png
        before_bitmap_path = os.path.join(save_dir, f"{timestep}_{block_position}_before.png")
        after_bitmap_path = os.path.join(save_dir, f"{timestep}_{block_position}_after.png")
        before_img.save(before_bitmap_path)
        after_img.save(after_bitmap_path)
    
    # 获取绝对路径并打印
    abs_save_dir = os.path.abspath(save_dir)
    print(f"Modulation scores可视化已保存至：{abs_save_dir}，共{len(block_scores)}个block")