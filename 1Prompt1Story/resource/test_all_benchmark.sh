#!/bin/bash
# 测试benchmark结果的简单脚本
# 使用方法: ./test_all_benchmark.sh [benchmark_dir] [output_dir]

BENCHMARK_DIR="${1:-./result/benchmark}"
OUTPUT_DIR="${2:-./result/benchmark_test_results}"

echo "=========================================="
echo "Benchmark测试脚本"
echo "=========================================="
echo "Benchmark目录: $BENCHMARK_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="
echo ""

# 检查benchmark目录是否存在
if [ ! -d "$BENCHMARK_DIR" ]; then
    echo "错误: benchmark目录不存在: $BENCHMARK_DIR"
    exit 1
fi

# 运行测试脚本
python -m resource.run_benchmark_tests \
    --benchmark_dir "$BENCHMARK_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --modes dreamsim clip_image clip_text

echo ""
echo "测试完成！"




