# kvpress / SqueezeAttention → vllm-ascend v0.23.0 适配计划（monkeypatch 外置，不改 vllm-ascend 一行代码）

> 产物：`kvpress-ascend/` 与 `SqueezeAttention-ascend/` 两个可 `pip install` 的独立包。
> 激活方式：`export kvpress=1` / `export squeeze=1`（大小写不敏感，多候选名），安装后 `.pth` 自动导入，未 export 时**零 torch/vllm 导入**（完全惰性）。
> 开关：环境变量门控 + 每次推理（每个 execute_model 步）心跳日志，报告两个 patch 是否进入各自核心代码（seam 探针）与核心参数。

---

## 0. 版本锚定（已从源码逐行核实）

| 事实 | 值 |
|---|---|
| vllm-ascend | v0.23.0（`vllm_ascend/__init__.py` 插件入口：`register()/register_connector()/register_model_loader()` → `adapt_patch(is_global_patch=True)`） |
| worker 类 | `NPUModelRunner(GPUModelRunner)`（`vllm_ascend/worker/model_runner_v1.py`，5249 行） |
| 注意力后端 | `AscendAttentionBackendImpl.forward(layer, query, key, value, kv_cache, attn_metadata, output, ...)`；query/key/value 为 **TND `(T, heads, hd)`**；KV 写走 `reshape_and_cache`（`attn_metadata.slot_mapping`）；读走 FIA：`_get_fia_params` → `block_table = attn_metadata.block_tables`、`actual_seq_lengths_kv = attn_metadata.seq_lens_list` |
| 元数据 | `AscendMetadata`：`seq_lens`(CPU tensor)、`seq_lens_cpu`、`seq_lens_list`、`block_tables`(GPU)、`slot_mapping`、`actual_seq_lengths_q`、`attn_state`（**Enum**：PrefillNoCache=0/PrefillCacheHit=1/DecodeOnly=2/ChunkedPrefill=3/SpecDecoding=4） |
| 公共元数据 | `AscendCommonAttentionMetadata`：`seq_lens`、`_seq_lens_cpu`(optimistic)、`block_table_tensor`、`slot_mapping`、`positions`、`num_computed_tokens_cpu`、`attn_state` |
| 块表 | `BlockTable.block_table.np/gpu`（CpuGpuBuffer，int32 行）、`num_blocks_per_row`、`commit_block_table(n)` 入口即 CPU→GPU、`compute_slot_mapping`（核函数 `slot = row[pos//bs]*bs + pos%bs`） |
| KV cache 张量 | `layer.kv_cache = (key_cache, value_cache)`，每张 `(num_blocks, block_size, kv_heads, head_size)`；`static_forward_context[layer_name] → Attention 模块`（layer_name 如 `model.layers.0.self_attn.attn`） |
| 时序陷阱 | `num_computed_tokens` 在 `sample_tokens()` 才更新 → 完成判定用 `before + 本步 scheduled`；`_prepare_inputs` 开头（第 885 行）就 `commit_block_table` |
| 用户拉起配置 | `--speculative_config {qwen3_5_mtp, num_speculative_tokens:3}` → `using_paged_attention()` 恒 False → **FIA 路径**；`--enable-prefix-caching`；`--tensor-parallel-size 4`；`--max-num-batched-tokens 4096`（长 prompt 必分块 prefill）；`FULL_DECODE_ONLY` cudagraph |

---

## 1. 两个工具原生的 KV cache 交互机制（已读源码）

### 1.1 kvpress（`kvpress-main/kvpress/`）

- 数据形态：HF `DynamicCache`，每层 `(batch, kv_heads, seq, hd)` 稠密张量。
- 机制：`BasePress.__call__(model)` 上下文管理器 → 给每个 attention 层注册 **forward hook**（`forward_hook(module, input, kwargs, output)`）→ prefill 期间每层 forward 结束后：从 cache 取 keys/values → 调 `compress(module, hidden_states, keys, values, attentions, kwargs)` → **原地替换** `cache_layer.keys/values` 为压缩后张量。
- `ScorerPress`：`score()` → `scores.topk(n_kept, dim=-1)`（**逐 head**）→ `gather` 剪枝。
- 典型 press 所需数据：
  - `KnormPress`：只需 keys 的 L2 范数。
  - `SnapKVPress`：最后 `window_size` 个 query 与全部 keys 的窗口注意力（需要 pre-rope queries、RoPE cos/sin、`repeat_kv`）。
  - `TOVAPress`：最后 1 个 query 的注意力。
  - `StreamingLLMPress`：sink 保留（纯索引）。
  - `RandomPress`：随机。
  - `PyramidKVPress`：SnapKV 打分 + **逐层预算**。
  - `ExpectedAttentionPress`：pre-rope 查询统计（均值/协方差）+ RoPE 平均旋转。
  - `CriticalKVPress`：两层打分（需要 o_proj 权重）。
- 约束：压缩发生在 prefill 完成瞬间（每请求一次）；decode 不再压缩。

### 1.2 SqueezeAttention（`SqueezeAttention-main/utils_hh/modify_llama_drop.py` / `modify_mistral.py`）

- 机制分两段：
  1. **prefill**（`LlamaModel_squeeze.forward` + `LlamaDecoderLayer_squeeze`）：每层记录 `cosine_similarity(层输入 hidden, 层输入+attn 输出)`（逐 token，取均值）→ `KMeans(n_clusters=3)` 按均值聚成 3 类 → 预算公式：
     `a = (num_layers*ini_size - len(class3)*percent) / len(class1+class2)`；
     `window[class1/2] = int(a * prompt_len)`；`window[class3] = int(percent * prompt_len)`。
     （`ini_size`/`percent` 是 prompt 长度的比例；class3 = cos 相似度最高 = attention 改变表征最少的层，给最小窗口。）
  2. **decode（streaming with real drop）**：每层若缓存长度 > window：保留 `[0:start_size] + [-recent:]`（recent = window − start_size），position id 重映射为压缩后坐标。
- 机制本质：**每层独立的滑动窗口视图**（物理内容不删除、只是"只让注意力看到窗口"）。

---

## 2. vllm v1 块式 KV cache 下的机制转换（本计划核心决策）

### 2.1 块布局带来的三条硬约束（分析得出，全部写入 RISK_REGISTER）

1. **每请求一个共享块表行**：所有层共用一个 `BlockTable` 行（同一个物理块序列）。写路径 `slot_mapping` 是 per-token 的、**所有层共用同一个张量** → **逐层不同压缩布局会互相踩槽**。→ kvpress 逐 head / 逐层剪枝必须降级：**打分可逐层逐 head，但保留集必须 head 统一（scores 先对 kv_heads 求均值）且 per-layer 视图独立**。
2. **worker 侧无法把块还给调度器**（engine-core 才持有块分配/hash 表）→ 任何 worker 侧压缩**都不回收块内存**，省的是注意力计算/带宽（FIA 按 `actual_seq_lengths_kv` 计工作量）。
3. **物理改写缓存 = 前缀缓存 hash 失效**（用户命令里带 `--enable-prefix-caching`）。

### 2.2 采用「视图重写（view rewrite）」机制（对两个工具统一），不采用尾部物理搬移

对比两种实现：

| 维度 | 尾部块物理搬移（compact） | **视图重写（view，本计划采用）** |
|---|---|---|
| 物理缓存 | 把保留 token 搬进尾部块 | **不改**（只读 gather 打分） |
| 块表行 | 每步重写 np 行 + positions 减 delta + 槽映射位移 | **不改**（写路径完全保持真值坐标） |
| seq_lens | 每层减 delta | 每层替换为视图长度 |
| 前缀缓存 | hash 失效（默认 skip → 用户场景下等于禁用） | **hash 依然有效** |
| MTP draft | 必须统一布局、cm 同步 | draft 若共享 group-0 看到全量缓存（安全）；step3.5 各自独立 group，天然无冲突 |
| 逐层预算 | 不可能（共享槽映射） | **支持**（每层独立元数据） |
| 侵入面 | positions/slot_mapping/row/cm 四处 | 仅**每层元数据**（block_tables + seq_lens 视图） |

核心机制（两个包共用同一套）：**在 `_build_attention_metadata` 返回后，逐层重写 `attn_metadata[layer]` 的 `block_tables`（GPU 视图行，写入每层预分配缓冲）与 `seq_lens / seq_lens_cpu / seq_lens_list`（CPU 拷贝）**；`slot_mapping / actual_seq_lengths_q / cm` 一律不动。FIA 核、FULL_DECODE_ONLY 图回放（`update_graph_params` 每步从 `attn_metadata[layer].block_tables/seq_lens_list` 取参）天然吃到修正。

- **kvpress 视图**：prefill 完成时按层打分 → head 统一 top-k → 记录 `(n_kept, 保留 token 块序列 kept_blocks)`；每步视图行 = `[kept_blocks] + [真行 m 之后的块]`，视图长 = `n_kept + (真长 − orig_len)`。
- **SqueezeAttention 视图**：prefill 完成时聚类出逐层窗口 `w_l`；每步视图行 = `[sink 块] + [最后 recent 块的区间]`，视图长 = `min(真长, w_l + 1)`（+1 含本步新 token）。

### 2.3 每步数据流（复刻 vllm v1 真实时序）

```
execute_model 入口（pre-hook）：
  快照 input_batch.num_computed_tokens_cpu、scheduler_output.num_scheduled_tokens、req_ids
  → 计算本步完成 prefill 的请求集合
→ _update_states（上游填充/增长块表行，真值坐标）
→ _prepare_inputs（S3 钩子：kvpress 无动作（view 模式不改行）；仅 capture 上下文）
  → positions / seq_lens / compute_slot_mapping（真值坐标，不动）
→ _build_attention_metadata（S5 钩子：返回后逐层视图重写）
→ _model_forward（S1/S2 捕获钩子：prefill 期间逐层捕获 query/hidden）
→ execute_model 返回前（post-hook）：
  对"本步完成 prefill"的请求：kvpress 打分+记录保留集 / squeeze 聚类出窗口
  → 心跳日志（seams + 核心参数）
→ sample_tokens（num_computed 才更新 —— 我们不依赖它）
```

---

## 3. Seam 表（全部从 vllm-ascend v0.23.0 源码逐行核实）

| # | 目标 | 签名要点（源码行） | 钩子动作 | kvpress | squeeze |
|---|---|---|---|---|---|
| S1 | `AscendAttentionBackendImpl.forward`（attention_v1.py:1479）与 `AscendC8AttentionBackendImpl.forward`(:1557) | `(layer, query TND, key, value, kv_cache, attn_metadata, output, ...)` | prefill 且非 draft 且非图捕获时，按 `actual_seq_lengths_q` 切分 per-request query，滚动存入 per-request 捕获缓冲（仅保留最后 window 个） | ✅ | — |
| S2 | `vllm.model_executor.layers.attention.Attention.forward` | `(layer, hidden_states, position_embeddings, kv_cache, attn_metadata, ...)` | 捕获 hidden_states（ExpectedAttention 用）；squeeze 结合层 forward 捕获的输入算 cos 相似度 | 按需 | ✅ |
| S3 | `NPUModelRunner._prepare_inputs`（:862） | 入口包装（commit_block_table 之前） | 设置每步 CaptureContext（view 模式无行改写） | ✅ | ✅ |
| S4 | `NPUModelRunner._build_attention_metadata`（:3034） | 返回 `(attn_metadata, spec_decode_common_attn_metadata)` 后 | **逐层视图重写**（block_tables 视图行 + seq_lens 三元组）；跳过图捕获期与 profile/dummy 步 | ✅ | ✅ |
| S5 | `NPUModelRunner.execute_model`（:1950） | 入口快照 / 返回前 post-hook | 完成判定（`before + scheduled >= prompt`）；kvpress 压缩 pass / squeeze 聚类 | ✅ | ✅ |
| S6 | 解码层 forward 包装（`runner.get_model().model.layers[i].forward`） | 层输入 hidden（pre-layernorm residual） | squeeze 层重要性捕获 | — | ✅ |
| S7 | `MultiGroupBlockTable.compute_slot_mapping` | 仅 compact 模式需要 | view 模式不 patch | — | — |

MTP 说明：`qwen3_5_mtp` → `AscendStep3p5MTPProposer`（step3p5.py:31），draft 有**独立 KV group**（`set_per_group_attn_metadata`），draft 元数据由 `_build_step_attn_metadatas` 从 cm 构建 → **draft 不读 group-0 的视图**，无需 cm 重写；若检测到共享 group-0 的 drafter（`AscendEagleProposer` 且非 step3.5），视图重写对 draft 不可见 → draft 看到全量缓存（安全降级，draft 质量不受损）。

---

## 4. 交付物形态（两个包同构）

```
kvpress-ascend/  (SqueezeAttention-ascend/ 同构)
├── pyproject.toml            # hatchling；force-include *.pth 进 wheel 根 → pip 自动落 site-packages
├── kvpress_ascend.pth        # 内容: "import kvpress_ascend"
├── kvpress_ascend/
│   ├── __init__.py           # env 门控：未 export 完全不 import torch/vllm；开启后 apply()（惰性 + fail-soft）
│   ├── envs.py               # 全部 env 集中定义：KVPRESS_ASCEND_ENABLE/POLICY/RATIO/WINDOW/SINK/MODE/STEP_LOG/DRY_RUN/PREFIX_CACHE/…
│   ├── log.py                # 独立 logger，前缀 [kvpress-ascend]
│   ├── registry.py           # seams 探针 + 统计计数器 + 心跳
│   ├── kvcore.py             # 纯逻辑：视图行构造/seq-lens 修正/保留集数学（L0 直接驱动，无 torch/vllm 依赖或仅 torch）
│   ├── presses.py            # BasePress/ScorerPress + 移植 presses（Knorm/Random/StreamingLLM/SnapKV/TOVA/PyramidKV/ExpectedAttention/CriticalKV）
│   ├── capture.py            # per-request 捕获缓冲 + 打分/聚类 pass
│   ├── engine.py             # 全部 monkeypatch（S1–S6）+ fail-soft try/except
│   └── simulate.py           # L1/L2 离线模拟器 CLI（python -m kvpress_ascend.simulate）
├── tests/                    # L0/L1/L2 分级测试（全离线可跑）
├── README.md                 # 用法/限制/真机核对清单
└── RISK_REGISTER.md          # 运行时风险登记
```

## 5. 环境变量（开关）

- 主开关：`export kvpress=1`（候选名 `kvpress`/`kvpress_ascend`/`exportkvpress`，值非空即开）；`export squeeze=1`（候选 `squeeze`/`squeezeattention`/`squeeze_ascend`/`exportsqueeze`）。
- 细粒度：`KVPRESS_ASCEND_ENABLE=0/1`、`KVPRESS_ASCEND_POLICY`（两包同开时谁生效）、`KVPRESS_ASCEND_PRESS=snapkv|knorm|…`、`KVPRESS_ASCEND_RATIO=0.5`、`KVPRESS_ASCEND_WINDOW=64`、`KVPRESS_ASCEND_SINK=4`、`KVPRESS_ASCEND_PREFIX_CACHE=skip|force`（view 模式无冲突，默认安全）、`KVPRESS_ASCEND_DRY_RUN=1`（只打分不记录视图）、`KVPRESS_ASCEND_STEP_LOG=1`（默认开）、`KVPRESS_ASCEND_LOG=info|debug`。
- SqueezeAttention：`SQUEEZE_ASCEND_INI_SIZE=0.3`、`SQUEEZE_ASCEND_CLASS3_RATIO=0.1`、`SQUEEZE_ASCEND_START_SIZE=4`、`SQUEEZE_ASCEND_STEP_LOG=1`、`SQUEEZE_ASCEND_LOG=info|debug`。

## 6. 心跳日志（每次推理一行的开关验证）

每步 `execute_model` 打一行（两包各自）：
```
[kvpress-ascend] step=42 prefill=0 decode=12 seams=6/6 hit=[S1,S2,S3,S4,S5,S6] press=snapkv ratio=0.50 window=64 sink=4 mode=view layers=64 bs=128 compressed=3 skipped_short=0 skipped_prefix_cache=0 skipped_error=0
[squeeze-ascend] step=42 seams=5/5 hit=[...] ini=0.30 class3=0.10 start=4 w_min=512 w_max=65536 clustered=2 active=9
```
- seam 探针：每个钩子进入时标记 `hit`；激活时打一次全 seam 汇总；心跳缺失/FAIL = patch 没进核心代码 → 查 `[kvpress-ascend]` 错误日志。
- 核心参数：press 名/ratio/window/sink；squeeze 的 ini_size/class3_ratio/start_size/逐层窗口范围。

## 7. 离线模拟调试（本机无 NPU）

- L0：kvcore 纯函数（视图行构造、seq-lens 修正、keep 数学、窗口公式、slack 不变量）。
- L1：Fake 对象逐字段照抄 ascend 源码（FakeRunner/FakeInputBatch/FakeBlockTable/FakeAscendMetadata/FakeBackend）。
- L2：步骤驱动复刻 vllm v1 时序（execute_model 入口快照 → _update_states 行增长 → _build_attention_metadata → backend forward 捕获 → 完成判定 → post 压缩 → 视图重写），多步 decode 越过块边界，端到端不变量：`attention(视图 K/V) == attention(参考保留集 + 新 token)`（误差 < 1e-4）。
- 完成定义（DoD）：L0/L1/L2 全绿 + 不变量注册表 + RISK_REGISTER + 自检 CLI + 心跳单测。

## 8. 风险登记摘要（详见各包 RISK_REGISTER.md）

| 风险 | 兜底 |
|---|---|
| FIA 数值/图回放行为 | 只改每步重建的元数据对象；DRY_RUN 先行；真机核对清单 |
| 前缀缓存 | view 模式不碰物理缓存 → 无冲突；compact 模式默认 skip |
| MTP draft 视图 | step3.5 独立 group 天然隔离；共享 group 的 drafter 看到全量（安全） |
| 逐 head 剪枝语义变化 | 降级为 head 统一保留集（块布局硬约束），README 明示 |
| 多请求混合步的捕获 | 捕获只对"唯一 prefill 请求"的步生效，否则跳过该步 |
| per-step 视图行拷贝开销 | 每层预分配缓冲 + 无压缩请求时零拷贝 fast-path |
| preempt/recompute | 每步校验 `num_computed < num_prompt` 即清除该请求压缩状态 |
| `.item()`/同步 | 热路径无 `.item()`；打分 pass 每请求一次，允许少量同步 |
