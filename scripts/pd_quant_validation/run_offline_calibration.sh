#!/bin/bash
# run_offline_calibration.sh - 运行 KVTuner 离线校准

set -e

# 配置
MODEL_PATH="${1:-/data/Qwen/Qwen2.5-7B}"
OUTPUT_NAME="${2:-layer_quant_config}"
CALIB_SAMPLES="${3:-512}"
WORK_DIR="/data/kvtuner_calib"
OUTPUT_DIR="/data"

echo "=========================================="
echo "KVTuner 离线校准"
echo "=========================================="
echo ""
echo "配置:"
echo "  模型：$MODEL_PATH"
echo "  输出：$OUTPUT_DIR/$OUTPUT_NAME.json"
echo "  校准样本：$CALIB_SAMPLES"
echo "  工作目录：$WORK_DIR"
echo ""

# 创建工作目录
mkdir -p "$WORK_DIR"

# 步骤 1: 准备校准数据集
echo "步骤 1: 准备校准数据集..."
python3 ~/kvtuner_offline/prepare_calib_dataset.py \
    --output "$WORK_DIR/calib_dataset.json" \
    --samples $CALIB_SAMPLES \
    --type wikitext

# 步骤 2: 运行离线校准
echo ""
echo "步骤 2: 运行离线校准..."
echo "这可能需要 30 分钟到 2 小时，取决于样本数和模型大小..."
echo ""

python3 ~/kvtuner_offline/kvtuner_offline_calib.py \
    --model "$MODEL_PATH" \
    --calib-data "$WORK_DIR/calib_dataset.json" \
    --output "$WORK_DIR/$OUTPUT_NAME.json" \
    --default-nbits 4 \
    --calib-samples $CALIB_SAMPLES

# 步骤 3: 验证配置
echo ""
echo "步骤 3: 验证配置..."
python3 -c "
import json
with open('$WORK_DIR/$OUTPUT_NAME.json') as f:
    config = json.load(f)
    
print(f'模型：{config[\"model_name\"]}')
print(f'层数：{len(config[\"layers\"])}')
print(f'默认 nbits: {config[\"default_nbits\"]}')
print(f'校准样本：{config[\"calib_samples\"]}')

# 统计 nbits 分布
nbits_dist = {}
for layer in config['layers'].values():
    nbits = layer['nbits_key']
    nbits_dist[nbits] = nbits_dist.get(nbits, 0) + 1

print(f'nbits 分布：{nbits_dist}')
"

# 步骤 4: 复制到最终位置
echo ""
echo "步骤 4: 保存配置..."
cp "$WORK_DIR/$OUTPUT_NAME.json" "$OUTPUT_DIR/$OUTPUT_NAME.json"

echo ""
echo "=========================================="
echo "✅ 校准完成!"
echo "=========================================="
echo ""
echo "配置文件:"
echo "  $OUTPUT_DIR/$OUTPUT_NAME.json"
echo ""
echo "下一步:"
echo "  1. 将配置文件部署到 P/D 节点"
echo "  2. 启动带层级量化的 P/D 服务"
echo ""
