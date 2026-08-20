# vllm-ascend v0.23.0 / vllm 0.23.0 已验证 Seam / API 表

> 全部条目从 vllm-ascend-releases-v0.23.0 源码逐行核实（本机无 vllm 安装时，以 ascend 仓库自身对 `vllm.*` 的 import 为地面真值）。
> 选缝、写 patch、真机排查时对照本表；版本升级后第一件事就是重核本表。
> §2/§3/§3b/§5 对任何模型优化类型通用；§6 是 KV 压缩案例的布局公式——新优化类型按同格式扩充自己的公式。
> 运行时的"谁调用谁、每步顺序、状态时序"另见 `runtime-scheduling-framework.md`（调度框架，先搭框架再选缝）。

## 1. 版本与插件

| 事实 | 值 / 证据 |
|---|---|
| 上游 vllm | 0.23.0（`vllm_ascend/utils.py::vllm_version_is("0.23.0")` 分支遍布仓库） |
| 依赖 | torch==2.10.0 / torch-npu==2.10.0.post4 / transformers==5.5.4 / triton-ascend==3.2.2 |
| 插件入口 | `vllm_ascend/__init__.py`：`register()`→NPUPlatform；`register_connector()`/`register_model_loader()` 先调 `_ensure_global_patch()`→`adapt_patch(is_global_patch=True)`（global=patch/platform，worker=patch/worker） |
| 引擎 | v1 引擎（`--async-scheduling`、`--additional-config`、v1 worker） |

## 2. Worker 侧关键类与方法（patch 目标）

| 位置 | 成员 | 角色 / 签名要点 |
|---|---|---|
| `vllm_ascend/worker/worker.py` | `NPUWorker(WorkerBase)` | `execute_model(scheduler_output)` → `self.model_runner.execute_model(...)`；`sample_tokens()` 在其后调用 |
| `vllm_ascend/worker/model_runner_v1.py` | `NPUModelRunner(GPUModelRunner)` | `__init__(vllm_config, device)` |
| 同上 | `execute_model(scheduler_output, intermediate_tensors=None)` | 步骤主入口；**`num_computed_tokens` 在 `sample_tokens()` 才更新**；async 路径返回 None 前 `execute_model_state` 暂存 attn_metadata 等 |
| 同上 | `_update_states(scheduler_output)` | 调 `super()`；块表行在本函数期间由上游 `append_row` 填充（行内容跨步持久，append 位置 = `num_blocks_per_row`） |
| 同上 | `_prepare_inputs(scheduler_output, num_scheduled_tokens)` | 开头 `block_table.commit_block_table(num_reqs)`（**此后改 np 行无效**）；`positions = num_computed + query_pos`；`seq_lens[:num_reqs] = num_computed + num_scheduled`；非 PCP 路径 `compute_slot_mapping(num_reqs, query_start_loc.gpu, self.positions[:total])` |
| 同上 | `_build_attention_metadata(...)` | 逐 KV group `cm = copy(cm_base)` → `_build_attn_group_metadata` 产出 `attn_metadata[layer_name] = AscendMetadata`；有 spec 时 `spec_decode_common_attn_metadata = cm`（group 0 的 cm）；返回 `(attn_metadata, spec_decode_common_attn_metadata)`；**ubatch 时 attn_metadata 是 list[dict]**（插件需守卫） |
| 同上 | `input_batch`（`NPUInputBatch`） | `req_ids`、`req_id_to_index`、`num_computed_tokens_cpu`(numpy)、`num_prompt_tokens`、`block_table`（`MultiGroupBlockTable`，`bt[gid]`→`BlockTable`） |
| 同上 | `requests` | `dict[req_id, RequestState]`（镜像调度器状态） |
| 同上 | `kv_cache_config.kv_cache_groups[g]` | `.layer_names`、`.kv_cache_spec`（`FullAttentionSpec`：`block_size`/`num_kv_heads`/`head_size`） |
| 同上 | `compilation_config.static_forward_context` | `dict[layer_name, Attention模块]`；KV cache 经 `bind_kv_cache` 绑定为 `模块.kv_cache` |
| `vllm_ascend/worker/block_table.py` | `BlockTable` | `block_table`（CpuGpuBuffer `.np`/`.gpu`）、`num_blocks_per_row`(np)、`append_row/add_row/clear_row/move_row/swap_row`、`commit_block_table(n)`（CPU→GPU）、`compute_slot_mapping(num_reqs, query_start_loc, positions)`（核函数语义：`slot = row[pos//bs]*bs + pos%bs`）、`compute_slot_mapping_draft(req_indices, positions)`（numpy） |
| `vllm_ascend/worker/block_table.py` | `MultiGroupBlockTable` | `compute_slot_mapping(...)` 循环各 group；**patch 它即可一次性位移 positions（compact 模式 S7）** |

## 3. 注意力后端（数据捕获点）

| 位置 | 成员 | 要点 |
|---|---|---|
| `vllm_ascend/attention/attention_v1.py` | `AscendAttentionBackendImpl.forward(layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None, output_block_scale=None)` | **query/key/value 为 TND `(T, heads, hd)` / `(T, kv_heads, hd)`**；`reshape_and_cache` 用 `attn_metadata.slot_mapping` 写缓存；`_get_fia_params` 按 `attn_state` 取 key_cache/view；FIA 核用 `block_table + actual_seq_lengths_kv`（= `seq_lens_list`） |
| 同上 | `AscendC8AttentionBackendImpl(AscendAttentionBackendImpl)` | 覆盖 `forward`（INT8 KV）；要 patch 需单独包装（S1b） |
| 同上 | `AscendAttentionState(Enum)` | `PrefillNoCache=0 / PrefillCacheHit=1 / DecodeOnly=2 / ChunkedPrefill=3 / SpecDecoding=4`；**`.value` 是 int**，比对用 `getattr(state, "name", state)` |
| 同上 | `AscendMetadata` | 字段：`attn_state`、**`seq_lens`(CPU tensor，取自 `_seq_lens_cpu[:num_reqs]`)**、`seq_lens_cpu`、`seq_lens_list`、`block_tables`(GPU 行)、`slot_mapping`、`actual_seq_lengths_q`(list)、`num_actual_tokens`、`num_prefills/num_decodes`；FIA padding 时 `seq_lens_list` 会补 1、`block_tables` 会补零行 |
| 同上 | `AscendAttentionMetadataBuilder.build(common_prefix_len, common_attn_metadata)` | `seq_lens_list = seq_lens.tolist()`；`block_tables` 取自 common 元数据（group 0 = input_batch 行）→ **换 per-layer 视图行 = 换新张量（不能原地改共享行）** |
| `vllm_ascend/attention/utils.py` | `AscendCommonAttentionMetadata` | `num_reqs`、`seq_lens`(GPU)、`_seq_lens_cpu`(optimistic, CPU)、`seq_lens_cpu_upper_bound`、`num_computed_tokens_cpu`、`is_prefilling`、`block_table_tensor`、`slot_mapping`、`query_start_loc`、`positions`、`attn_state`；`unpadded()` 给 eagle 用 |
| `vllm_ascend/ascend_forward_context.py` | `_EXTRA_CTX` | `is_draft_model`（draft forward 标志，捕获时跳过）、`capturing`（图捕获标志，假元数据别碰） |
| `vllm.model_executor.layers.attention` | `Attention`（上游 vllm 类） | `forward(self, layer, hidden_states, position_embeddings, kv_cache, attn_metadata, ...)`；`layer.layer_name` 形如 `model.layers.0.self_attn.attn`；输出为 attn 结果（T,H） |
| KV cache 张量 | ascend 布局 | `(2, num_blocks, block_size, num_kv_heads, head_size)`；split 后 key_cache `(num_blocks, bs, kv_heads, hd)`，可 `view(-1, kv_heads, hd)` 按槽索引 |

## 3b. MTP / 投机解码元数据流（v0.23.0 实测）

| 事实 | 要点 |
|---|---|
| `qwen3_5_mtp` → `AscendStep3p5MTPProposer`（spec_decode/step3p5.py:31） | draft 层有**独立 KV group**（`set_per_group_attn_metadata` 每步从 runner 捕获各组 block_table/slot_mapping）；draft 元数据 `multi_steps_attn_metadata`（每 draft 步 1 份 dict）在 **sample_tokens 的 `propose_draft_token_ids` → `_propose`** 里从 cm 重建 → **draft 不读 group-0 的视图**，视图重写无需 cm 同步 |
| 共享 group-0 的 drafter（`AscendEagleProposer` 等） | `spec_decode_common_attn_metadata = cm`；draft 元数据从 cm 重建 → 要让 draft 看到压缩视图必须重写 cm（seq_lens 拷贝）；否则 draft 看到全量（安全降级） |
| `spec_decode_common_attn_metadata` 的选取 | `_build_attention_metadata` :3305-3310：drafter 的 `attn_layer_names[0]` 在 group 层名里才取；step3.5 多 group 各自 `set_per_group_attn_metadata` |
| decode_threshold | 有 spec 时 = 1 + num_speculative_tokens；`split_decodes_and_prefills` 用它分 decode/prefill |

## 4. 调度侧（engine-core，只读参考、默认不 patch）

| 位置 | 成员 | 要点 |
|---|---|---|
| `vllm.v1.core.kv_cache_utils` | `KVCacheManager`、`KVCacheBlock`（`block_id/ref_cnt/is_null/prev_free_block/next_free_block`）、`FreeKVCacheBlockQueue` | 块生命周期在调度进程 |
| `vllm.v1.core.block_pool` | `BlockPool`：`get_new_blocks(n)`、`cache_full_blocks(request, blocks, ...)`、`get_cached_block(hash, gid)`、`touch(blocks)`、`free_blocks(blocks, prepend)` | **prefix-cache hash 表在这里**；物理改写缓存内容 = hash 失效；视图重写不碰内容 = hash 有效 |
| `vllm.v1.core.single_type_kv_cache_manager` | `SingleTypeKVCacheManager`/`FullAttentionManager`：`req_to_blocks[req_id]`、`enable_caching`、`allocate_slots` | DeepSeekV4 压缩 MLA 用 `CompressAttentionManager`（`vllm_ascend/core/single_type_kv_cache_manager.py`，`compress_ratio` 布局思路可借鉴） |
| `vllm.v1.kv_cache_spec_registry` | `KVCacheSpecRegistry.get_manager_class(spec)` | 模型 runner 建 manager 的工厂 |

## 5. 时序陷阱（最容易翻车）

1. **`num_computed` 更新在 `sample_tokens`**（execute_model 之后）→ 完成判定用 `before + 本步 scheduled`，且**允许 `before == 0`**（单步完成整个 prompt 的请求）。
2. **`_prepare_inputs` 开头就 `commit_block_table`** → 行重写必须在 `_prepare_inputs` 入口、commit 之前（包装入口即可）。
3. **FIA padding**：`actual_seq_lengths_q` 可能含 FIA 假 padding 请求（`num_reqs_fia`），按 `min(len(req_ids), len(q_lens))` 迭代；`seq_lens_list` 可能被 builder 补 1、`block_tables` 补零行。
4. **cudagraph FULL_DECODE_ONLY 回放**：`update_graph_params` 每步从当前 `attn_metadata[layer].seq_lens/block_tables` 取参（非 SWA 时 block_tables 刷新）→ 每步重建的 metadata 修正天然生效；图捕获期是假元数据，别碰。
5. **`using_paged_attention()`**：有 speculative_config 时返回 False → FIA 路径（只依赖 FIA 路径的 block_table/seq_lens 语义即可）。
6. **`num_blocks_per_row` 决定 append_row 落点**：compact 模式行重写后必须把计数缩减为 `k + (valid − m)`，否则后续 append 落点错乱。
7. **`positions` 同时服务 RoPE 与槽映射**：`update_cos_sin(positions)`（:2266）在 `_prepare_inputs` 之后、forward 之前——设备端位移 positions 会同时改两者（compact 模式按设计）。
8. **draft forward 在 sample_tokens 里跑**（execute_model 之后）→ 目标侧 per-layer 视图重写对 draft 无效（step3.5 独立 group 无需管；共享 group drafter 走 cm 或接受全量）。
9. **ubatch**：`_build_attention_metadata` 传 `ubatch_slices` 时 `attn_metadata` 是 `list[dict]`——插件一律守卫跳过（`skipped_ubatch`）。

## 6. 案例：压缩布局公式（KV 压缩插件；其它优化类型按同格式扩充）

### 6.1 视图重写（view 模式，默认推荐；写路径零改动，前缀缓存安全）

```
m = ceil(orig_len / bs)                 # prefill 块数
保留块：按块分数 top-k（块级粒度；head 统一分数）
强制规则：orig % bs != 0 时块 m-1 必须保留（新 decode token 落其 padding 槽，否则不可见；
         让位时只在已选块内 argmin 丢最低分块）
视图行 = [保留块（物理 id 升序）] + [真行 m .. valid]
view_len = Σ_{b∈保留块} min(bs, orig - b*bs) + (true_len - orig)
           # 按块 token 数（末块部分填充感知）+ 新增 token；末块内封顶，绝不读零 padding
FIA 读 view_row[p//bs] 槽 p%bs（块序列语义，不是 token 子集）
```

### 6.2 窗口视图（SqueezeAttention 风格；写路径零改动）

```
window, start_size, recent = window - start_size
sink_blocks = ceil(start_size / bs)
recent_first = max(sink_blocks, (true_len - recent) // bs)   # 重叠钳位（去重）
视图行 = [真行 0..sink_blocks) + [真行 recent_first .. ceil(true_len/bs))
view_len = true_len - (recent_first - sink_blocks) * bs      # 末块内封顶
true_len <= window → 不重写
```

### 6.3 尾部块物理搬移（compact 模式，可选；前缀缓存需 force）

```
m = ceil(orig_len / bs);  delta = orig_len - n_kept
k = m - delta // bs                # 保留块数（>= ceil(n_kept/bs)，保证 slack 不变量）
row' = [b_{m-k} .. b_{m-1}] + [b_m ..]     # 【一次性】重写（非幂等！）+ num_blocks_per_row = k + (valid - m)
packing 槽: t_slots = repeat(row'[:k], bs)[:n_kept] * bs + (arange(n_kept) % bs)   # [:n_kept] 截断
打分 gather 槽: repeat(row, bs)[:orig_len]   # 绝不取 m*bs（尾块 padding 污染 topk）
新 token 槽 = row'[(n_kept+j)//bs] * bs + (n_kept+j) % bs     # positions 设备端减 delta 后算
seq_len' = seq_len - delta         # 每层注意力元数据（+ MTP 时 cm 拷贝）
```

slack 不变量：`k*bs - n_kept >= m*bs - orig_len`（左 = delta mod bs + c ≥ c），保证压缩位置不跑赢调度器块表增长。
共享槽映射 → compact 模式必须每请求统一 n_kept（逐层预算取均值）；逐层预算只可能在 view 模式表达。
