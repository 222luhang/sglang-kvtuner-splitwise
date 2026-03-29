# P/D 分离与量化验证 - 设置摘要

## ✅ 已完成

### 1. 验证脚本创建

已在 `/home/ubuntu/.openclaw/workspace/sglang-kvtuner/scripts/pd_quant_validation/` 创建完整工具链：

| 文件 | 状态 |
|------|------|
| `deploy_to_nodes.sh` | ✅ 就绪 |
| `start_prefill.sh` | ✅ 就绪 |
| `start_decode.sh` | ✅ 就绪 |
| `start_router.sh` | ✅ 就绪 |
| `run_validation.sh` | ✅ 就绪 |
| `validate_pd_quant.py` | ✅ 就绪 |
| `check_status.sh` | ✅ 就绪 |
| `QUICK_START.md` | ✅ 就绪 |
| `DEPLOYMENT_GUIDE.md` | ✅ 就绪 |

### 2. 目标机器配置

| 角色 | IP | 端口 | 状态 |
|------|-----|------|------|
| Prefill 1 | 10.60.6.75 | 30000 | ⏳ 待部署 |
| Decode 1 | 10.60.19.152 | 30001 | ⏳ 待部署 |
| Prefill 2 | 10.60.9.62 | 30000 | ⏳ 待部署 |
| Decode 2 | 10.60.176.217 | 30001 | ⏳ 待部署 |

### 3. 凭据配置

- **用户名**: ubuntu
- **密码**: luhang222
- **Python 环境**: `.venv` (uv)

## ⚠️ 注意事项

### 模型路径

**`/data` 目录当前为空**

脚本将默认使用 HuggingFace 模型：
- `meta-llama/Llama-3.1-8B-Instruct`

如果有本地模型，设置环境变量：
```bash
export MODEL_PATH="/path/to/your/model"
```

### 网络要求

- 所有节点间需要高速网络连接
- 推荐：InfiniBand 或 RoCE (≥100Gbps)
- 最低：10GbE

### 依赖检查

在部署前，确保控制节点已安装 `sshpass`：
```bash
sudo apt-get install -y sshpass
```

## 🚀 下一步操作

### 选项 A: 自动部署（推荐）

```bash
cd /home/ubuntu/.openclaw/workspace/sglang-kvtuner/scripts/pd_quant_validation
./deploy_to_nodes.sh
```

### 选项 B: 手动部署

1. **SSH 到每台机器**，复制脚本：
   ```bash
   scp -r ./pd_quant_validation ubuntu@10.60.6.75:/tmp/
   ```

2. **在每台机器上启动服务**：
   - Node 1 & 3: `./start_prefill.sh`
   - Node 2 & 4: `./start_decode.sh`
   - 任一节点：`./start_router.sh`

3. **运行验证**：
   ```bash
   ./run_validation.sh
   ```

## 📊 验证测试项

1. ✅ 服务健康检查
2. ✅ 服务器信息验证（P/D 模式、量化配置）
3. ✅ 量化功能验证（KVTuner、KV Cache）
4. ✅ 基本推理测试
5. ✅ P/D 通信验证
6. ✅ 性能基准测试
7. ✅ 内存使用检查

## 📈 预期结果

| 指标 | 预期值 |
|------|--------|
| Prefill 吞吐 | ≥1000 tokens/s (8B, 4-bit) |
| Decode 延迟 | ≤50ms/token (8B, 8-bit) |
| 内存节省 | ≥50% (相比 BF16) |
| 精度损失 | <1% (相比 BF16) |

## 🔧 量化配置

### Prefill 服务（高吞吐）
```bash
export KV_TUNER_NBITS_KEY=4
export KV_TUNER_NBITS_VALUE=4
export KV_TUNER_RESIDUAL_LENGTH=256
```

### Decode 服务（低延迟）
```bash
export KV_TUNER_NBITS_KEY=8
export KV_TUNER_NBITS_VALUE=8
export KV_TUNER_RESIDUAL_LENGTH=32
```

### KV Cache 量化
```bash
# 推荐
export KV_CACHE_DTYPE=fp8_e5m2

# 更高精度
export KV_CACHE_DTYPE=fp8_e4m3

# 实验性（最大压缩）
export KV_CACHE_DTYPE=fp4_e2m1
```

## 📝 日志位置

服务启动后，日志将保存在：
```
/tmp/sglang_logs/prefill_YYYYMMDD_HHMMSS.log
/tmp/sglang_logs/decode_YYYYMMDD_HHMMSS.log
/tmp/sglang_logs/router_YYYYMMDD_HHMMSS.log
```

## ❓ 需要帮助？

1. 查看 `QUICK_START.md` - 快速开始指南
2. 查看 `DEPLOYMENT_GUIDE.md` - 详细部署指南
3. 运行 `./check_status.sh` - 检查服务状态

## 📞 联系

如遇到问题，请提供：
- 错误日志（`/tmp/sglang_logs/*.log`）
- 服务状态（`./check_status.sh` 输出）
- 网络连通性测试结果
