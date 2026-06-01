#!/usr/bin/env python3
import os
import sys
import textwrap
from datetime import datetime
from typing import Any
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENDPOINT", "https://hf-mirror.com")
_ROOT = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.join(_ROOT, "resource")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _RES)
import gradio as gr
import yaml
import torch
from PIL import Image
import diffusers
diffusers.utils.logging.set_verbosity_error()
import gen_benchmark
from main import load_unet_controller
from merge_images import merge_pil_horizontally
from unet import utils
APP_TITLE = "故事图像生成系统v1.0"
SAMPLE_YAML = textwrap.dedent("""\
concept_token: demo butterfly
settings:
- hovering near morning dew
- landing on a wildflower
- spreading wings in sunlight
- resting on a leaf
- flying toward the horizon
style: A soft watercolor illustration of
subject: A monarch butterfly with orange and black wings
new_subject: monarch butterfly, vivid wing patterns, delicate antennae, morning light
""").strip()
HELP_MD = textwrap.dedent("""\
## {title}

1. 编辑 **条目 YAML** 或点击「填入示例 YAML」。
2. 填写设备、模型、精度与输出目录后点击 **加载扩散模型**。
3. **开始生成** 调用 `gen_benchmark.main_ben` 与 `generate_images` 写入 `run_*` 目录；**更新拼接预览** 仅对当前画廊图像做简单拼接。
4. **输出质量测试** 对指定子目录调用 `resource/test_benchmark.py` 中的指标计算。
""").strip()
METRICS_INTRO = textwrap.dedent("""\
对 **输出根目录** 下子目录（或根目录）内图像运行与 `test_benchmark.py` 一致的指标。
""").strip()
_SD: dict[str, Any] = {}
def _load_image(img: Any) -> Image.Image | None:
	if img is None:
		return None
	if isinstance(img, Image.Image):
		return img.convert("RGB")
	if isinstance(img, str) and os.path.isfile(img):
		return Image.open(img).convert("RGB")
	try:
		import numpy as np
		if isinstance(img, np.ndarray):
			if img.ndim == 2:
				return Image.fromarray(img).convert("RGB")
			return Image.fromarray(img.astype("uint8")).convert("RGB")
	except Exception:
		pass
	return None
def _gallery_items_to_images(items: list | None) -> list[Image.Image]:
	out: list[Image.Image] = []
	if not items:
		return out
	for item in items:
		path_or_img: Any = item
		if isinstance(item, (list, tuple)) and item:
			path_or_img = item[0]
		elif isinstance(item, dict):
			path_or_img = item.get("image") or item.get("name") or item.get("path")
		im = _load_image(path_or_img)
		if im is not None:
			out.append(im)
	return out
def merge_pil_simple(images: list[Image.Image], gap: int = 8, bg: tuple[int, int, int] = (40, 44, 52)) -> Image.Image | None:
	if not images:
		return None
	target_h = max(im.height for im in images)
	resized: list[Image.Image] = []
	for im in images:
		if im.height != target_h:
			w = max(1, int(im.width * target_h / im.height))
			resized.append(im.resize((w, target_h), Image.Resampling.LANCZOS))
		else:
			resized.append(im)
	total_w = sum(im.width for im in resized) + gap * (len(resized) + 1)
	canvas = Image.new("RGB", (total_w, target_h), bg)
	x = gap
	for im in resized:
		canvas.paste(im, (x, 0))
		x += im.width + gap
	return canvas
def fill_sample_yaml() -> str:
	return SAMPLE_YAML
def _parse_instance_yaml(text: str) -> dict[str, Any]:
	data = yaml.safe_load(text)
	if not isinstance(data, dict):
		raise ValueError("YAML 须为 mapping")
	for k in ("settings", "subject"):
		if k not in data:
			raise ValueError(f"缺少字段: {k}")
	if not isinstance(data["settings"], list) or not data["settings"]:
		raise ValueError("settings 须为非空列表")
	return data
def _build_id_prompts(inst: dict[str, Any], use_new_subject: bool) -> tuple[str, str, str]:
	st = inst.get("style", "")
	if isinstance(st, str):
		st = st.strip()
	subj = inst.get("subject", "")
	gen_subj = inst.get("new_subject", subj) if use_new_subject else subj
	if st:
		gen_id = f"{st} {gen_subj}"
		file_id = f"{st} {subj}"
	else:
		gen_id = gen_subj
		file_id = subj
	return gen_id, file_id, subj
def _load_sd_stack(device: str, model_path: str, precision: str) -> str:
	key = (device.strip(), model_path.strip(), precision)
	if _SD.get("key") == key and _SD.get("pipe") is not None:
		return "模型已在内存中，未重新加载。"
	dev = device.strip()
	if dev.startswith("cuda") and not torch.cuda.is_available():
		dev = "cpu"
	dtype = torch.float16 if precision == "fp16" else torch.float32
	pipe, _, _ = utils.load_pipe_from_path(model_path.strip(), dev, dtype, precision)
	uc = load_unet_controller(pipe, dev)
	uc.Save_story_image = False
	uc.Prompt_embeds_mode = "single"
	uc.save_self_attention = False
	uc.use_half_precision = True
	uc.max_heads_to_save = 4
	_SD.clear()
	_SD["key"] = key
	_SD["pipe"] = pipe
	_SD["unet_controller"] = uc
	_SD["device"] = dev
	return f"已加载: {model_path.strip()} @ {dev}"
def run_preview(yaml_text: str, seq_len: float, theme: str, window_length: float, seed: float, use_new_subject: bool, upload_gallery: list | None) -> tuple[list, Image.Image | None, str, str]:
	pil_list = _gallery_items_to_images(upload_gallery)
	merged = merge_pil_simple(pil_list) if pil_list else None
	gallery_out: list = [(im, f"帧 {i + 1}") for i, im in enumerate(pil_list)]
	note = textwrap.dedent(f"""\
	YAML 字符数: {len(yaml_text or '')}；序列长度(settings): {int(seq_len)}
	主题: {theme or '—'}
	window_length={int(window_length)}，seed={int(seed)}，use_new_subject={use_new_subject}
	当前画廊 {len(pil_list)} 帧；下方为逐帧与简单横向拼接（与 merge_images 排版不同）。
	""").strip()
	save_hint = "可右键另存图像。"
	return gallery_out, merged, note, save_hint
def run_generate(yaml_text: str, seq_len: float, theme: str, window_length: float, seed: float, use_new_subject: bool, upload_gallery: list | None, output_root: str, device: str, model_path: str, precision: str) -> tuple[list, Image.Image | None, str, str]:
	if _SD.get("pipe") is None:
		return [], None, "请先点击「加载扩散模型」成功后再生成。", "—"
	try:
		inst = _parse_instance_yaml(yaml_text)
	except Exception as e:
		return [], None, f"YAML 解析失败: {e}", "—"
	gen_id, file_id, subject_text = _build_id_prompts(inst, use_new_subject)
	frames = [str(s) for s in inst["settings"]]
	pipe = _SD["pipe"]
	uc = _SD["unet_controller"]
	uc.subject_text = subject_text
	root = os.path.abspath(os.path.expanduser((output_root or ".").strip() or "."))
	os.makedirs(root, exist_ok=True)
	stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_dir = os.path.join(root, f"run_{stamp}")
	os.makedirs(run_dir, exist_ok=True)
	wl = int(window_length)
	sd = int(seed)
	try:
		images, _story = gen_benchmark.main_ben(uc, pipe, run_dir, gen_id, frames, sd, wl, original_id_prompt=file_id if file_id != gen_id else None)
	except Exception as e:
		return [], None, f"生成失败: {e}", run_dir
	merged = merge_pil_horizontally(images, gen_id, frames)
	mp = os.path.join(run_dir, "merged_display.jpg")
	merged.save(mp, "JPEG", quality=95)
	gallery_out = [(im, cap) for im, cap in zip(images, frames)]
	note = textwrap.dedent(f"""\
	生成完成。
	输出目录: {run_dir}
	id_prompt: {gen_id}
	帧数: {len(images)}
	合并图: {mp}
	主题备注: {theme or '—'}；声明长度 {int(seq_len)}（以 YAML settings 为准: {len(frames)}）
	""").strip()
	return gallery_out, merged, note, run_dir
def _count_images_in_dir(directory: str) -> int:
	if not os.path.isdir(directory):
		return 0
	exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
	n = 0
	for name in os.listdir(directory):
		p = os.path.join(directory, name)
		if os.path.isfile(p) and os.path.splitext(name.lower())[1] in exts:
			if name.startswith("story_image"):
				continue
			n += 1
	return n
def run_metrics_test(output_root: str, test_subdir: str, modes: list[str], use_new_subject: bool, yaml_path: str) -> str:
	root = os.path.abspath(os.path.expanduser((output_root or ".").strip() or "."))
	sub = (test_subdir or "").strip()
	if sub and os.path.isabs(sub):
		test_dir = sub
	elif sub:
		test_dir = os.path.join(root, sub)
	else:
		test_dir = root
	lines = [f"测试路径: {test_dir}", f"图像文件数: {_count_images_in_dir(test_dir)}", ""]
	if not os.path.isdir(test_dir):
		lines.append("目录不存在。")
		return "\n".join(lines)
	if not torch.cuda.is_available():
		lines.append("当前无 CUDA，test_benchmark 中 CLIP/DreamSim 需要 GPU。")
		return "\n".join(lines)
	order = ("clip_image", "clip_text", "dreamsim")
	selected = [m for m in order if m in (modes or [])]
	if not selected:
		lines.append("请至少勾选一种测试模式。")
		return "\n".join(lines)
	yp = (yaml_path or "").strip() or None
	try:
		from test_benchmark import calculate_avg_distance, calculate_avg_clip_text_score
	except Exception as e:
		lines.append(f"无法导入 test_benchmark: {e}")
		return "\n".join(lines)
	for m in selected:
		try:
			if m in ("clip_image", "dreamsim"):
				avg = calculate_avg_distance(test_dir, m, False, True, None)
			else:
				avg = calculate_avg_clip_text_score(test_dir, True, None, use_new_subject, yp)
			vals = [v for v in avg.values() if isinstance(v, float) and v == v]
			if vals:
				tot = sum(vals) / len(vals)
				lines.append(f"{m}_total_avg_distance: {tot}")
			else:
				lines.append(f"{m}: 无有效数值（需至少 2 张图计算 clip_image/dreamsim）")
			lines.append(f"明细: {avg}")
		except Exception as e:
			lines.append(f"{m} 失败: {e}")
	return "\n".join(lines)
def build_ui() -> gr.Blocks:
	with gr.Blocks(title=APP_TITLE) as demo:
		gr.Markdown(HELP_MD.format(title=APP_TITLE))
		with gr.Row():
			with gr.Column(scale=1):
				seq_len = gr.Slider(1, 16, value=5, step=1, label="声明序列长度 settings（以 YAML 为准）")
				theme = gr.Textbox(label="主题 / 创意提示", placeholder="例如：雨林里的机械蝴蝶……")
				btn_sample = gr.Button("填入示例 YAML")
			with gr.Column(scale=2):
				yaml_box = gr.Code(label="条目 YAML", language="yaml", lines=24, value=SAMPLE_YAML)
		gr.Markdown("### 图像生成参数")
		with gr.Row():
			device = gr.Textbox(value="cuda:0", label="运算设备")
			sd_model = gr.Textbox(value="stabilityai/stable-diffusion-xl-base-1.0", label="模型路径 / Hugging Face ID")
			precision = gr.Dropdown(choices=["fp16", "fp32"], value="fp16", label="精度")
		with gr.Row():
			window_length = gr.Slider(2, 20, value=10, step=1, label="窗口长度 window_length")
			seed = gr.Number(value=42, label="随机种子", precision=0)
			use_new_subject = gr.Checkbox(value=True, label="使用 new_subject 作为生成主体描述")
			output_root = gr.Textbox(value="./results/gradio_ui", label="输出根目录")
		with gr.Row():
			btn_load = gr.Button("加载扩散模型")
			load_status = gr.Textbox(label="加载状态", lines=2, value="未加载模型。")
		gr.Markdown("### 图像序列")
		upload = gr.Gallery(label="图像序列（可选上传；生成后由模型结果覆盖）", columns=4, height=240, object_fit="contain")
		with gr.Row():
			btn_generate = gr.Button("开始生成", variant="primary")
			btn_preview = gr.Button("更新拼接预览")
		gen_status = gr.Textbox(label="任务状态", lines=8)
		save_dir_txt = gr.Textbox(label="输出说明", lines=2)
		with gr.Row():
			merged_img = gr.Image(label="横向合并长图（生成时用 merge_images.merge_pil_horizontally）", type="pil")
			gallery = gr.Gallery(label="逐帧图像", columns=4, height="auto")
		gr.Markdown("### 输出质量测试")
		gr.Markdown(METRICS_INTRO)
		with gr.Row():
			test_subdir = gr.Textbox(label="测试子目录（相对输出根目录，留空测根目录）", placeholder="run_20260101_120000", lines=1)
			test_modes = gr.CheckboxGroup(label="测试模式", choices=[("CLIP-图像", "clip_image"), ("CLIP-文本", "clip_text"), ("DreamSim", "dreamsim")], value=["clip_image", "clip_text", "dreamsim"])
		metrics_new_subject = gr.Checkbox(value=False, label="clip_text 使用 YAML 替换 subject（与 --new_subject 一致）")
		metrics_yaml = gr.Textbox(label="clip_text 用 YAML 路径（可选）", placeholder="留空用 test_benchmark 默认", lines=1)
		btn_run_test = gr.Button("运行测试")
		metrics_report = gr.Textbox(label="测评报告", lines=16, max_lines=28)
		btn_sample.click(fill_sample_yaml, outputs=[yaml_box])
		btn_load.click(lambda d, m, p: _load_sd_stack(d, m, p), inputs=[device, sd_model, precision], outputs=[load_status])
		btn_preview.click(run_preview, inputs=[yaml_box, seq_len, theme, window_length, seed, use_new_subject, upload], outputs=[gallery, merged_img, gen_status, save_dir_txt])
		btn_generate.click(run_generate, inputs=[yaml_box, seq_len, theme, window_length, seed, use_new_subject, upload, output_root, device, sd_model, precision], outputs=[gallery, merged_img, gen_status, save_dir_txt])
		btn_run_test.click(run_metrics_test, inputs=[output_root, test_subdir, test_modes, metrics_new_subject, metrics_yaml], outputs=[metrics_report])
	return demo
if __name__ == "__main__":
	ui = build_ui()
	ui.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"), server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
