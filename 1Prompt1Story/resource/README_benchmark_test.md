# Benchmark测试脚本使用说明

## 功能
使用 `run_benchmark_tests.py` 脚本可以自动测试 `result/benchmark` 目录下的所有结果，分别使用三种方法：
- `dreamsim`: 使用DreamSim模型计算图像相似度
- `clip_image`: 使用CLIP模型计算图像-图像相似度
- `clip_text`: 使用CLIP模型计算图像-文本相似度

## 使用方法

### 基本用法（测试所有三种方法）
```bash
cd /root/autodl-tmp/dit/1Prompt1Story
python -m resource.run_benchmark_tests --benchmark_dir ./result/benchmark
```

### 指定测试模式
```bash
# 只测试dreamsim
python -m resource.run_benchmark_tests --benchmark_dir ./result/benchmark --modes dreamsim

# 测试dreamsim和clip_image
python -m resource.run_benchmark_tests --benchmark_dir ./result/benchmark --modes dreamsim clip_image
```

### 指定输出目录
```bash
python -m resource.run_benchmark_tests \
    --benchmark_dir ./result/benchmark \
    --output_dir ./result/benchmark_test_results
```

### 跳过已存在的测试结果
```bash
python -m resource.run_benchmark_tests \
    --benchmark_dir ./result/benchmark \
    --skip_existing
```

## 参数说明

- `--benchmark_dir`: benchmark结果目录路径（默认: `./result/benchmark`）
- `--output_dir`: 测试结果输出目录（默认: `./result/benchmark_test_results`）
- `--modes`: 要运行的测试模式，可选值：`dreamsim`, `clip_image`, `clip_text`（默认: 全部三种）
- `--skip_existing`: 跳过已存在的测试结果文件

## 输出文件

测试完成后会在输出目录生成以下文件：

1. `{mode}_results_{timestamp}.txt`: 每个测试模式的详细结果
2. `summary_{timestamp}.txt`: 文本格式的汇总报告
3. `summary_{timestamp}.json`: JSON格式的汇总报告（便于程序处理）

## 示例输出

```
================================================================================
Benchmark测试脚本
================================================================================
Benchmark目录: /path/to/result/benchmark
输出目录: /path/to/result/benchmark_test_results
测试模式: dreamsim, clip_image, clip_text
时间戳: 20251225_120000
================================================================================

================================================================================
开始测试: dreamsim
测试目录: /path/to/result/benchmark
================================================================================

... (测试过程输出) ...

dreamsim_total_avg_distance: 0.12345

... (其他测试模式) ...

================================================================================
测试完成！
================================================================================
汇总报告: /path/to/result/benchmark_test_results/summary_20251225_120000.txt
JSON报告: /path/to/result/benchmark_test_results/summary_20251225_120000.json

测试结果摘要:
  ✓ dreamsim       : 0.123456
  ✓ clip_image     : 0.234567
  ✓ clip_text      : 0.345678
================================================================================
```

## 注意事项

1. 确保已安装所有依赖：
   - `dreamsim`
   - `transformers` (用于CLIP)
   - `clip` (用于CLIP文本)
   - `torch`
   - `PIL`

2. 测试过程可能需要较长时间，特别是对于大量图像

3. 确保有足够的GPU内存或使用CPU模式

4. 如果测试中断，可以使用 `--skip_existing` 参数继续未完成的测试




