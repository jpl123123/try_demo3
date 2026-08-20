# vllm-ascend v0.23.0 + monkeypatch 优化插件：运行调度框架（代码级，持续更新）

> **本文件是"先搭框架、再深入 coding、debug 反馈后持续更新框架"这一方法论的标准载体，对任何模型优化类型（KV 压缩/注意力变体/投机解码/采样/量化…）通用。**
> 规则：任何 vllm-ascend 集成工作开始前，先把本节框架（进程 → 每步流水线 → 状态时序 → 钩子叠加 → 数据流）按当前版本源码逐行核实并落成文档；**选缝、写代码、定位 bug 都先回到这张图**；每次 debug 有新发现（新 seam、新时序、新 bug 根因），先在框架上落一笔再动代码。
> 以下条目全部从 vllm-ascend-releases-v0.23.0 源码逐行核实（2025-08 实战：kvpress-ascend / SqueezeAttention-ascend 适配；§2/§3 对任何优化通用，§4 的钩子叠加表以 KV 压缩插件为示例，新优化类型按同格式追加自己的 seam）。

---

## 1. 进程架构（谁在哪个进程里干什么）

```
┌────────────────────────────────────────────────────────────────────┐
│ API server 进程（vllm serve 入口）                                  │
│   · 加载 vllm → 插件注册：vllm_ascend/__init__.py register() 系列    │
│   · site-packages 里的 <pkg>_ascend.pth 在解释器启动时已执行         │
│     → monkeypatch 插件在【每个 Python 进程】独立生效                 │
└───────────────┬────────────────────────────────────────────────────┘
                │ spawn（多进程）
┌───────────────▼────────────────┐   ┌──────────────────────────────────┐
│ engine-core（调度进程，1 个）     │   │ worker（执行进程，每 NPU rank 1 个）│
│ · Scheduler                    │   │ · NPUModelRunner                 │
│ · KVCacheManager / BlockPool   │◄──┤ · input_batch（块表行镜像）        │
│ · prefix-cache hash 表         │   │ · KV cache 张量（每层 1 对）       │
│ · RequestState（权威状态）       │   │ · requests（RequestState 镜像）   │
│ · 块分配/释放/refcount           │   │ · static_forward_context         │
└───────────────────────────────┘   └──────────────────────────────────┘
        scheduler_output（每步 RPC：num_scheduled_tokens、块增长指令）
        model_runner_output（采样结果回传）
```

推论（铁律）：
1. **worker 侧无法把块还给调度器** → 纯 worker 侧压缩**不回收块内存**，省的是注意力计算/带宽。
2. **prefix-cache hash 表在 engine-core** → 物理改写缓存内容 = 让 hash 失效；**只改"读视图"则 hash 依然有效**。
3. **worker 的块表行 np 缓冲是镜像**：`_update_states` 每步按调度器指令 append，`_prepare_inputs` 开头 commit 到 GPU —— 想在行上做手脚，窗口在 `_prepare_inputs` 入口、commit 之前。

---

## 2. worker 每步流水线（execute_model 精确调用顺序，行号 = v0.23.0）

```
NPUWorker.execute_model (worker.py:611)
 └─ NPUModelRunner.execute_model (model_runner_v1.py:1950)
     ├─ 状态清理/深拷贝护栏 (:1969-2022)
     ├─ _update_states(scheduler_output) (:2047)
     │     · 块表行增长（append_row / add_row，行号 == input_batch 内下标）
     │     · input_batch.num_computed_tokens_cpu / num_prompt_tokens 镜像刷新
     ├─ [无 token] 提前返回 EMPTY (:2060-2075)
     ├─ _prepare_inputs(scheduler_output, num_scheduled_tokens) (:862)
     │     ├─ commit_block_table(num_reqs) (:885)   ← CPU→GPU 拷贝；【此后改 np 行无效】
     │     ├─ positions_np = num_computed + query_pos (:908-916)  ← token_indices 用它 (:999)
     │     ├─ _compute_prev_positions(num_reqs) (:948)            ← spec-decode 簿记
     │     ├─ self.positions[:total]（GPU, int64）赋值 (:1249-1259)
     │     ├─ self.seq_lens[:num_reqs] = num_computed + scheduled (:1261-1264)
     │     └─ compute_slot_mapping(num_reqs, qsl, self.positions[:total]) (:1283)
     │           · 核函数语义：slot = row[pos//bs]*bs + pos%bs（row = block_table.gpu）
     ├─ _build_attention_metadata(...) (:3034)
     │     ├─ cm_base = AscendCommonAttentionMetadata(:3161)：seq_lens(GPU)、_seq_lens_cpu(optimistic)、
     │     │        block_table_tensor(group0 行)、slot_mapping、positions、attn_state、is_prefilling…
     │     ├─ per kv_cache_group: cm = copy(cm_base) (:3275)
     │     ├─ builder.build(cm) → attn_metadata[layer_name] (:3265-3266)
     │     │        · AscendMetadata：seq_lens(CPU 拷贝)、seq_lens_cpu、seq_lens_list、
     │     │          block_tables(GPU 行)、slot_mapping、actual_seq_lengths_q、attn_state(Enum)
     │     └─ return (attn_metadata, spec_decode_common_attn_metadata) (:3350)
     │           · MTP/eagle：spec_decode_common_attn_metadata = group-0 的 cm（:3305-3310）
     ├─ _preprocess → input_ids / positions (:2252-2263)
     ├─ update_cos_sin(positions) (:2266)            ← RoPE 查表用 positions（插件位移后=压缩坐标）
     ├─ set_ascend_forward_context(attn_metadata, …) (:2294-2317)
     │     · _EXTRA_CTX.is_draft_model / capturing 在这里的 forward_context 上
     ├─ _model_forward (:2320)
     │     ├─ 每层：Attention.forward → AscendAttentionBackendImpl.forward (attention_v1.py:1479)
     │     │     ├─ reshape_and_cache：按 attn_metadata.slot_mapping 写 KV (:1430)
     │     │     └─ forward_impl：FIA / paged attention (:1456)
     │     │           · FIA 读 attn_metadata.block_tables + seq_lens_list (actual_seq_lengths_kv) (:994/:1211)
     │     └─（图模式：update_graph_params 每步从 attn_metadata[layer].block_tables/seq_lens_list 取参）
     └─ 返回 None（async 路径 :2401；execute_model_state 暂存 attn_metadata 等）
sample_tokens (:2403)                                ← 【num_computed 在这里才更新】
 └─ propose_draft_token_ids (:2471)
     └─ AscendStep3p5MTPProposer._propose (step3p5.py:329)
           · 用 spec_decode_common_attn_metadata（cm）构建 multi_steps_attn_metadata（每 draft 步 1 份）
           · draft forward 在 sample_tokens 里跑（execute_model 之后！）
```

---

## 3. 状态时序表（谁在什么时候更新什么 —— 插件设计的地基）

| 状态 | 更新时机 | 对插件的结论 |
|---|---|---|
| `input_batch.num_computed_tokens_cpu` | **sample_tokens**（execute_model 返回之后） | 完成判定必须用 `before + 本步 scheduled` |
| 块表行 np（`block_table.np`） | `_update_states`（每步 append_row；行内容跨步持久） | 行重写窗口在 `_prepare_inputs` 入口、commit 之前；改完要缩减 `num_blocks_per_row` 才会影响后续 append 位置 |
| 块表行 gpu | `commit_block_table`（`_prepare_inputs` 第一件事） | 同上 |
| `self.positions`（GPU） | `_prepare_inputs` 每步重建（num_computed+query_pos） | 设备端位移安全（RoPE 与槽映射共用） |
| `self.seq_lens`（GPU） | `_prepare_inputs` 每步重建 | 不要原地改；改 per-layer 元数据拷贝 |
| `optimistic_seq_lens_cpu` | `_prepare_inputs`（async spec decode 还会修正） | 别动 |
| `attn_metadata[layer]` | **每步 `_build_attention_metadata` 重建** | **插件改写点**：`block_tables`（换新张量）、`seq_lens/seq_lens_cpu/seq_lens_list`（CPU 拷贝） |
| `spec_decode_common_attn_metadata` | = group-0 的 cm（有 spec 时） | step3.5 MTP：draft 有独立 KV group → 无需 cm 重写；共享 group 的 drafter 走 cm |
| KV cache 物理内容 | 每步 `reshape_and_cache` | view 模式不碰；compact 模式 prefill 完成时搬移一次 |
| **请求的"完成"状态** | `num_computed >= num_prompt`（sample_tokens 更新） | **优化触发点不能只挂在"完成"上**：长上下文下资源可能在完成前耗尽（抢占循环，`completed=0` 永远）→ 提供渐进式触发（按 token 预算推进锚点）；配套回归式状态清理（`before < 上次所见` 才判抢占，`before < prompt` 会误删渐进布局） |

---

## 4. 插件钩子叠加层（seam → 流水线节点 → 动作）

| seam | 触发节点 | 动作（kvpress-ascend / squeeze-ascend 实战验证） |
|---|---|---|
| S1 backend forward 捕获 | `_model_forward` 内每层（attention_v1.py:1479 / C8:1557） | prefill 态（PrefillNoCache/CacheHit/ChunkedPrefill）且非 draft 且非 capturing：按 `actual_seq_lengths_q` 切 TND query，per-request 滚动存最后 window 个 |
| S2 Attention.forward | 同上（vllm 的 Attention 类） | hidden 捕获（ExpectedAttention 用）；与层包装输入配对算 cos-sim（squeeze 层重要性） |
| S3 `_prepare_inputs` 入口 | :862 之前（包装整个方法） | compact：一次性行重写 `[b_{m-k}..b_{m-1}]+[b_m..]` + `num_blocks_per_row = k+(valid-m)` + `row_rewritten` 标志（**非幂等，只做一次**） |
| S4 `_build_attention_metadata` 之后 | :3350 返回后 | view：per-layer 视图行（写预分配缓冲，行尾补零）+ seq_lens 三元组；compact：seq_lens − delta + cm 拷贝；**ubatch（list 元数据）跳过** |
| S5 `execute_model` pre/post | :1950 入口 / 返回前（finally） | 快照 num_computed/num_scheduled/num_prompt → 完成判定 → 返回前压缩 pass（打分/记录布局/聚类）→ 心跳 |
| S6 解码层 forward 包装 | `_model_forward` 内（每层，惰性安装） | 捕获层输入（pre-layernorm residual），squeeze 用 |
| S7 `MultiGroupBlockTable.compute_slot_mapping` | :1283 | compact：positions 设备端减 per-request delta（repeat_interleave 按 query_start_loc 展开；**注意测试要走 engine 全局 ctx**） |
| S8（新增，kvpress/squeeze 二期） | 视图行 buffer 填充 | **物理块 id 地面真值**：`row[kept]`（保留集是逻辑下标，buffer 行必须存物理 id）；squeeze 窗口滑动 sync 的 append 分支必须校验 `recent_first` 未变；增量 sync 只拷新增尾块（append-only 前提靠 first-block 签名检测 add_row/move/swap/preemption 后全量重同步） |

---

## 5. 数据流（张量/对象关系速查）

```
input_batch.block_table[gid]  ─ BlockTable
   ├─ block_table.np  (CPU, int32 行)   ── commit_block_table ──► block_table.gpu
   ├─ num_blocks_per_row (np int32)     ← append_row/add_row 维护；插件可缩减（compact）
   └─ slot_mapping (CpuGpuBuffer)       ← compute_slot_mapping 写（slot = row[pos//bs]*bs + pos%bs）

static_forward_context[layer_name] ──► Attention 模块（.kv_cache = (key_cache, value_cache)）
   key_cache/value_cache: (num_blocks, block_size, kv_heads, head_size)
   ──► 平面索引：kv.view(-1, kv_heads, head_size)[slot]

AscendMetadata（每层，每步重建）
   seq_lens(CPU) / seq_lens_cpu / seq_lens_list  ← 注意力"KV 视图长度"（FIA actual_seq_lengths_kv）
   block_tables(GPU, (num_reqs_padded, max_blocks)) ← 注意力"KV 视图块行"
   slot_mapping(GPU) ← 写路径（reshape_and_cache），视图重写不动它
   actual_seq_lengths_q ← 查询长度（与 KV 视图无关，别改）

TND query/key/value: (T, heads, hd) / (T, kv_heads, hd)；捕获后按 actual_seq_lengths_q 切 per-request。
```

---

## 6. 更新纪律（debug 反馈 → 框架更新）

每次调试闭环（REPRO → ROOTCAUSE → FIX → VERIFY）完成后，回到本文件做三件事：
1. **更新流水线/时序表**：若 bug 源于某节点顺序或状态更新时机，在该行加注（如 `(:885) ← 此后改 np 无效`）；
2. **更新钩子叠加层**：新 seam 或既有 seam 的新动作，补一行；
3. **更新 bug 目录与 RTR**：新根因模式进 SKILL.md 的 bug 类目表；模拟覆盖不了的新风险进 RISK_REGISTER.md。
