# KVTuner 离线校准工具

> **最后更新**: 2026-04-08

本目录包含 KVTuner KV Cache 量化的离线校准工具，用于分析各层 KV Cache 敏感性并生成最优量化配置。

## 工具说明

| 文件 | 用途 |
|------|------|
| `kvtuner_offline_calib.py` | 离线校准主程序，分析各层 KV 敏感性 |
| `prepare_calib_dataset.py` | 准备校准数据集 |
| `run_offline_calibration.sh` | 一键运行离线校准 |

## 量化配置格式

校准输出为 JSON 配置文件，格式示例：

```json
{
  "layer_configs": [
    {"layer_id": 0, "nbits_key": 8, "nbits_value": 8, "residual_length": 256},
    {"layer_id": 1, "nbits_key": 4, "nbits_value": 4, "residual_length": 128},
    ...
  ]
}
```

**注意**: `nbits_key` 和 `nbits_value` 只支持 **2、4、8**。

## 使用方式

```bash
# 1. 准备校准数据集
python prepare_calib_dataset.py --model-path /path/to/model --output calib_data/

# 2. 运行离线校准
python kvtuner_offline_calib.py --model-path /path/to/model --calib-data calib_data/ --output layer_quant.json

# 或使用一键脚本
bash run_offline_calibration.sh /path/to/model layer_quant.json
```

## P/D 分离部署测试

KVTuner + P/D 分离的部署和测试脚本已迁移到 `../pd_disagg_test/` 目录。
