# 故事配置生成器使用说明

## 功能

根据故事描述自动生成 YAML 配置文件，用于图像生成的故事序列。

## 使用方法

### 方法一：命令行模式

```bash
python resource/generate_story_yaml.py \
    --story "生成火柴人的故事" \
    --length 5 \
    --api-key "your-api-key-here" \
    --output "my_story.yaml" \
    --category "custom"
```

**参数说明：**
- `--story`: 故事描述（必需）
- `--length`: 序列长度/图像数量（默认 5）
- `--api-key`: OpenAI API Key（必需）
- `--base-url`: API Base URL（可选，用于自定义 API 端点）
- `--model`: 使用的模型（默认 gpt-3.5-turbo）
- `--output`: 输出文件路径（默认 generated_story.yaml）
- `--category`: YAML 分类名称（默认 custom）
- `--append`: 追加到现有文件（如果存在）

### 方法二：交互式模式

```bash
python resource/generate_story_yaml_interactive.py
```

然后按照提示输入信息。

## 示例

### 示例 1：生成火柴人的故事

```bash
python resource/generate_story_yaml.py \
    --story "生成火柴人的故事" \
    --length 5 \
    --api-key "sk-..." \
    --output "stick_figure_story.yaml"
```

**生成的配置示例：**
```yaml
custom:
- concept_token: stick_figure
  settings:
  - playing basketball on a court
  - dancing at a party
  - running in a park
  - reading a book in a library
  - cooking in a kitchen
  style: A simple and minimalist illustration of
  subject: A stick figure with basic features
  new_subject: Stick figure with simple line body, round head, expressive gestures and dynamic poses
```

### 示例 2：生成动物的故事

```bash
python resource/generate_story_yaml.py \
    --story "生成一只猫的日常生活故事" \
    --length 8 \
    --api-key "sk-..." \
    --output "cat_story.yaml" \
    --category "animals"
```

## 输出格式

生成的 YAML 文件格式与 `consistory+1p1s_allprompt_refined.yaml` 相同：

```yaml
category_name:
- concept_token: token_name
  settings:
  - scene description 1
  - scene description 2
  - ...
  style: Style description
  subject: Subject description
  new_subject: Detailed subject description
```

## 使用生成的配置文件

生成配置文件后，可以在 `gen_benchmark.py` 中使用：

```bash
python resource/gen_benchmark.py \
    --refined_benchmark_path "generated_story.yaml" \
    --save_dir "./result/my_story" \
    --device cuda:0
```

## 注意事项

1. **API Key**: 需要有效的 OpenAI API Key
2. **网络连接**: 需要能够访问 OpenAI API（或自定义 API 端点）
3. **序列长度**: 建议在 1-20 之间，过长可能导致生成质量下降
4. **模型选择**: 
   - `gpt-3.5-turbo`: 快速、经济
   - `gpt-4`: 更高质量，但更慢更贵

## 故障排除

### 问题 1: API 调用失败
- 检查 API Key 是否正确
- 检查网络连接
- 检查 API 配额是否充足

### 问题 2: 生成的配置不完整
- 尝试增加 `max_tokens` 参数
- 使用 `gpt-4` 模型
- 简化故事描述

### 问题 3: JSON 解析失败
- 脚本会自动尝试手动解析
- 如果失败，检查 API 返回的内容

## 高级用法

### 使用自定义 API 端点

```bash
python resource/generate_story_yaml.py \
    --story "生成火柴人的故事" \
    --api-key "your-key" \
    --base-url "https://api.example.com/v1"
```

### 追加到现有文件

```bash
python resource/generate_story_yaml.py \
    --story "生成另一个故事" \
    --api-key "your-key" \
    --output "existing_file.yaml" \
    --append
```

## 相关文件

- `generate_story_yaml.py`: 命令行版本
- `generate_story_yaml_interactive.py`: 交互式版本
- `consistory+1p1s_allprompt_refined.yaml`: 参考格式示例
- `gen_benchmark.py`: 使用生成的配置文件进行图像生成

