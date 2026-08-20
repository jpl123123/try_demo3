# KV 压缩工具 × vllm-ascend v0.23.0（monkeypatch 适配器合集）

把 [kvpress](https://github.com/NVIDIA/kvpress) 与 [SqueezeAttention](https://github.com/Ledzy/SqueezeAttention) 两个 KV cache 压缩工具，以**纯 monkeypatch 方式**适配到 **vllm-ascend v0.23.0**：**不修改 vllm-ascend 任何一行源码**，`pip install` 即用，环境变量开关控制。

## 目录结构

```
├── kvpress-ascend/            # kvpress 适配包（打分剪枝：SnapKV/Knorm/TOVA/StreamingLLM/PyramidKV/…）
│   ├── kvpress_ascend/        #   源码（envs/log/registry/kvcore/presses/capture/engine/simulate）
│   ├── tests/                 #   L0/L1/L2 离线测试
│   └── README.md / RISK_REGISTER.md
├── SqueezeAttention-ascend/   # SqueezeAttention 适配包（逐层滑动窗口：聚类 → 每层窗口）
│   ├── squeeze_ascend/        #   源码（同构）
│   ├── tests/
│   └── README.md / RISK_REGISTER.md
├── launch_vllm_ascend.sh      # 一键拉起示例脚本
└── PLAN.md / TODO.md          # 设计文档与施工清单
```

## 安装

```bash
pip install ./kvpress-ascend
pip install ./SqueezeAttention-ascend
# 安装后 site-packages 里会多出 kvpress_ascend.pth / squeeze_ascend.pth，
# 每个 Python 进程启动时自动导入（API server / engine-core / worker 全覆盖）
```

## 激活开关（环境变量门控）

```bash
export kvpress=1        # 开启 kvpress 压缩（候选名：kvpress_ascend / exportkvpress）
export squeeze=1        # 开启 SqueezeAttention（候选名：squeezeattention / squeeze_ascend / exportsqueeze）
```

- **不 export = 完全无感**：包不会 import torch/vllm，进程零开销。
- **两个同时 export**：默认 **kvpress 生效**（squeeze 自动让位并打日志）；要换成 squeeze 生效：`export SQUEEZE_ASCEND_POLICY=squeeze`。两者机制会竞争同一份注意力元数据，**不要**同时让两个都重写（见各自 README）。

## 怎么一起拉起 vllm-ascend

```bash
# 1) 安装（一次）
pip install ./kvpress-ascend ./SqueezeAttention-ascend

# 2) 开开关（每次拉起前）
export kvpress=1
export squeeze=1

# 3) 原样拉起 vllm serve（命令与未装工具时完全一致）
vllm serve /softwarePlatform/c00879303/Qwen3.5-27B-w8a8-mtp \
  --served-model-name "qwen3.5" \
  --host 0.0.0.0 \
  --port 1144 \
  --data-parallel-size 1 \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.9 \
  --compilation-config '{"cudagraph_capture_sizes":[1,4,8,12,16,24,32,48,56,64,72,84,96,108,112,128,160,172,196,200,212,232,272,288,312,328,344,360,384,400,416,432,448,480,512], "cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
  --trust-remote-code \
  --async-scheduling \
  --allowed-local-media-path / \
  --quantization ascend \
  --enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --additional-config '{"enable_cpu_binding":true}' \
  --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
```

或直接：`bash launch_vllm_ascend.sh`

## 每次推理的开关验证（心跳日志）

每个 `execute_model` 步两个包各打一行，证明 patch 进了核心代码（seam 探针）与核心参数：

```
[kvpress-ascend]  INFO step=42 seams=6/8 press=snapkv ratio=0.50 window=64 sink=4 mode=view layers=4 active_compressed=2 attn_state=DecodeOnly compressed=3 ...
[squeeze-ascend]  INFO step=42 seams=6/6 ini=0.30 class3=0.10 start=4 w_min=512 w_max=65536 clustered=2 ...
```

- `seams=x/N`：本步命中的钩子数；`FAIL=...` 点名未安装的钩子 → patch 没进核心代码，先查 `[kvpress-ascend]`/`[squeeze-ascend]` 错误日志。
- 激活时每个进程打印一次全 seam 汇总。
- 关闭心跳：`export KVPRESS_ASCEND_STEP_LOG=0` / `export SQUEEZE_ASCEND_STEP_LOG=0`；日志级别：`KVPRESS_ASCEND_LOG=debug|info|warning`（squeeze 同理）。

## 常用调优参数速查

| 目标 | 变量 | 默认 | 说明 |
|---|---|---|---|
| 压缩方法 | `KVPRESS_ASCEND_PRESS` | `snapkv` | `knorm`/`random`/`streamingllm`/`snapkv`/`tova`/`pyramidkv`/`expected_attention`/`criticalkv` |
| 压缩比例 | `KVPRESS_ASCEND_RATIO` | `0.5` | 保留比例 = 1−ratio |
| 观察窗口 | `KVPRESS_ASCEND_WINDOW` | `64` | SnapKV/TOVA 用 |
| 布局模式 | `KVPRESS_ASCEND_MODE` | `view` | `view`（默认，前缀缓存安全）或 `compact`（物理搬移，需 `KVPRESS_ASCEND_PREFIX_CACHE=force`） |
| 安全排练 | `KVPRESS_ASCEND_DRY_RUN` | `0` | 只打分不生效，先确认统计正常 |
| 总预算比例 | `SQUEEZE_ASCEND_INI_SIZE` | `0.3` | 总 KV 预算 = 层数 × ini × prompt 长 |
| class-3 窗口 | `SQUEEZE_ASCEND_CLASS3_RATIO` | `0.1` | 高 cos 相似层的窗口比例 |
| sink 数 | `SQUEEZE_ASCEND_START_SIZE` | `4` | StreamingLLM 风格首部保留 |
| 短 prompt 阈值 | `KVPRESS_ASCEND_MIN_PROMPT` / `SQUEEZE_ASCEND_MIN_PROMPT` | `512` / `256` | 更短不压缩 |

完整清单见各包 `README.md`。

## 机制一句话

- **kvpress**：prefill 完成瞬间按层打分（head 统一）→ 块粒度保留集 → 每步只重写 `attn_metadata[layer]` 的 `block_tables` 视图行与 `seq_lens`（写路径零改动 → 前缀缓存 hash 有效、MTP 兼容）。
- **SqueezeAttention**：prefill 捕获逐层 `cos_sim(层输入, 层输入+attn输出)` → 1D KMeans 三类 → 原版预算公式 → decode 每步逐层"窗口视图"（sink 块 + 最近块，视图长度在末块内封顶）。
- 两者都**不回收块内存**（worker 侧无法还块给调度器），省的是注意力计算/带宽（FIA 按 `actual_seq_lengths_kv` 计工作量）。

## 离线模拟调试（本机无 NPU 也全绿）

```bash
cd kvpress-ascend          && python -m kvpress_ascend.simulate --steps 8 --mode view
cd SqueezeAttention-ascend && python -m squeeze_ascend.simulate --steps 8
python -m pytest kvpress-ascend/tests SqueezeAttention-ascend/tests -q   # 46 tests
```

L2 端到端不变量：多步 decode 越过块边界，`attention(视图 K/V) == attention(参考保留集+新 token)`（误差 < 1e-4）。

## 真机核对清单（第一跑）

1. 启动日志出现 `activation summary: seams=N/N`（每个进程一次）。
2. 首个长 prompt 完成后出现 `compressed=N` / `clustered=1` 与逐层窗口日志。
3. 先 `KVPRESS_ASCEND_DRY_RUN=1` 跑一轮确认打分/统计，再开真压缩。
4. 精度对比（压缩 vs 不压缩）、前缀缓存对照、MTP draft 接受率、长跑 `skipped_error` 不增长、性能基线 —— 详见两包 README 附录与 `RISK_REGISTER.md`。

## 已知限制（如实说明）

- 块粒度/head 统一保留集是 vllm v1 块式缓存的硬约束（逐 head 剪枝无法表达）。
- 两个包同时 export 时只有主策略生效（默认 kvpress）。
- 不回收显存；`compact` 模式与 `--enable-prefix-caching` 冲突需 `force`。
- 模拟覆盖 L0–L2；CANN 算子数值/图回放/MTP 接受率/性能需真机确认。
