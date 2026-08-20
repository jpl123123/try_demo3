---
name: vllm-ascend-debugging
description: Use when working on vllm-ascend / vLLM v1 (Ascend NPU) model-optimization integration — monkeypatching external libraries or model optimizations (KV-cache compression, attention variants, speculative decoding, sampling, quantization, custom layers, etc.) into vllm-ascend without touching its source, planning patch seams, offline simulated debugging without NPU hardware, or diagnosing on-machine failures (ImportError, gather_v3/AI Core, ACL stream synchronize failed, worker crash, skipped optimization, prefix-cache/MTP issues). Covers the scheduling-framework-first methodology (build the code-level runtime map of vllm-ascend + your hooks before coding, keep updating it from debug feedback), the systematic debug state machine, editing-time self-checking, the hardware-free simulated-debug protocol (fidelity tiers, step driver, invariant registry, runtime risk register), a verified seam/API reference for vllm-ascend v0.23.0, and a growing bug catalog — with the kvpress / SqueezeAttention KV-compression adaptation as the fully worked example.
whenToUse: Whenever the user mentions vllm-ascend, vLLM v1 on Ascend NPU, or any model optimization / monkeypatch / patch adaptation on it (KV-cache compression, attention changes, speculative decoding, sampling, quantization, model support), or pastes a vllm-ascend traceback (ImportError, partially initialized module, gather_v3 index out of range, ACL stream synchronize failed, AI Core error, worker crash) — or asks to plan, debug or troubleshoot such integration work. Load the skill BEFORE reading code, BEFORE proposing patches, BEFORE writing offline simulation tests, and BEFORE diagnosing real-machine logs.
---

# vllm-ascend 模型优化集成：框架先行 · 规划 · 调试 · 模拟 · 排查

**适用范围：任何对 vllm-ascend 上模型做优化的集成工作**——KV cache 压缩、注意力变体、投机解码、采样、量化、自定义层、新模型支持等，只要满足"**不修改 vllm-ascend 源码、以运行时 monkeypatch 注入**"这一形态，本技能的方法论全部适用。
**贯穿案例**：kvpress / SqueezeAttention → vllm-ascend v0.23.0 的 KV 压缩适配（完整跑通：规划 → 编码 → 模拟调试 → 交付）。文中标"案例"的条目是该项目的具体落地，其余条目对任何优化流程通用。

核心立场：vllm-ascend 源码**一行不改**，所有改动都是运行时 monkeypatch；每个 hook 必须 **fail-soft**（出错只告警、服务继续跑未优化）；**所有关键事实必须能从 vllm-ascend 源码本身验证**，不能靠猜；**没有机器不是不调试的借口——模拟调试是定义好的、可交付的一等流程**。

---

## 0. 方法论总纲：调度框架先行（先搭骨架，再深入 coding，debug 持续更新）

> 这条来自实战教训：直接埋头选缝/coding，会在中后期被"谁在什么时候更新了什么"反复绊倒。正确顺序是——**阅读和规划阶段的第一件事，是把"整个 vllm-ascend 和你的优化应该怎么运行"的代码级调度框架搭出来；框架立住之后再深入 coding；之后每次 debug 有反馈，先更新框架，再改代码。** 对任何优化类型（压缩/注意力/采样/量化…）都一样：优化的本质是"在框架的某个节点上插入/改写一段数据流"，没有框架图就不知道插在哪、会不会踩到别处的状态。

### 0.1 框架的五层（全部从源码逐行核实，落到文档）

| 层 | 内容 | 说明 |
|---|---|---|
| L0 进程/线程架构 | API server → engine-core（调度）↔ worker（执行，每 rank 一个）；`.pth` 在每个进程生效；scheduler_output / model_runner_output 两条 RPC | 决定"你的 hook 在哪个进程里能看到什么" |
| L1 每步流水线 | `execute_model` 内精确调用顺序 + 源码行号（`_update_states` → `_prepare_inputs`(commit/positions/seq_lens/slot_mapping) → `_build_attention_metadata` → `_model_forward` → 返回；`sample_tokens` 在其后） | 决定"你的 hook 插在哪个节点、前后分别是什么" |
| L2 状态时序 | 谁在什么时候更新什么（num_computed 在 sample_tokens 才更新、commit_block_table 后改 np 无效、attn_metadata 每步重建、块表行跨步持久） | 决定"你的 hook 读的状态是否新鲜、改的状态会不会被覆盖" |
| L3 钩子叠加 | 每个 seam 在流水线哪个节点触发、读什么写什么 | 你的优化方案的落地位置清单 |
| L4 数据/张量流 | 块表行 np/gpu、slot_mapping、positions、seq_lens、各 Metadata 字段关系、KV cache 张量、TND query | 决定"你的优化需要的数据从哪来、改哪里才生效" |

**标准载体**：`references/runtime-scheduling-framework.md`（v0.23.0 已核实版本；接到新版本/新任务时按同样结构重核重写一份，把你要优化的对象在 L1/L2 上标出来）。

### 0.2 工作流（每次集成任务照此循环）

```
阅读源码（版本锚定 → 插件骨架 → 优化对象机制 → vllm-ascend 数据位置）
  ↓
[1] 搭调度框架：进程 → 每步流水线（带行号）→ 状态时序 → 数据流      ← 先立骨架
  ↓
[2] 在框架上选缝：把候选 hook 点标注到流水线节点上（只选"每步重建的对象"）
  ↓
[3] 机制设计：把优化目标"转换"成框架能表达的形式（§2.3 的决策轴）
  ↓
[4] 深入 coding（小步快验，见 §4 编辑期自查）
  ↓
[5] debug 有反馈（真机日志 / 模拟测试失败）：
      先在框架上定位"哪个节点、哪个状态、什么时序"→ 更新框架文档（§6 更新纪律）
      → 再走调试状态机（§3）改代码
```

### 0.3 框架的用法（不是装饰）

- **选缝前**：每个候选 hook 点在框架 §2 的流水线上标注触发节点，检查"这个对象每步重建吗、状态谁更新"。
- **每个 bug**：先在框架上回答三个问题——发生在哪个节点？依赖哪个状态？该状态何时被谁更新？答不上来 = 还没到根因。
- **每次闭环后**：按框架文档 §6 的三件事（流水线/时序加注、钩子层补行、bug 目录与 RTR 更新）回写框架。

---

## 1. 两进程架构与三条铁律（vllm v1 通用）

vLLM v1 把调度与执行拆开：

| 进程 | 拥有什么 | 能做什么 |
|---|---|---|
| **engine-core（调度进程）** | `KVCacheManager` / `BlockPool`（块分配、refcount、**prefix-cache hash 表**）、`Scheduler`、`RequestState` | 分配/释放块、前缀缓存匹配、调度决策 |
| **worker（执行进程，每个 NPU rank 一个）** | `NPUModelRunner`、`input_batch`（块表行镜像）、KV cache 张量、`requests`（RequestState 镜像）、模型权重与算子 | 前向、注意力、KV 写入、采样 |

铁律（决定一切设计）：
1. **worker 侧无法把块还给调度器 / 无法改调度器状态** → 纯 worker 侧优化**动不了资源分配**（如不回收块内存），省的是计算/带宽；要动分配必须改 engine-core，如实告知用户。
2. **prefix-cache hash 表在 engine-core** → **物理改写缓存内容 = hash 失效**；**只改"读视图"（每步重建的元数据）则 hash 依然有效**。这条决定了很多优化（压缩、窗口、重排）该用"视图改写"还是"内容改写"。
3. **`input_batch.num_computed_tokens` 在 `sample_tokens()`（execute_model 返回之后）才更新** → 在 `execute_model` 尾部做任何"本步是否完成 X"的判断必须用 `before + 本步 scheduled tokens`，且**允许 `before == 0`**（单步完成整个 prompt 的请求——真实踩过的坑）。

---

## 2. 规划阶段：选缝与机制设计（Planning）

### 2.1 读代码的顺序（快而准）

1. **版本锚定**：`git describe`、`requirements.txt`、`vllm_ascend/utils.py` 的 `vllm_version_is(...)` → 确定上游 vllm 版本。**用 ascend 仓库自身对 `vllm.*` 的 import 作为上游 API 的地面真值**（本机常常没装 vllm）。
2. **插件骨架**：`vllm_ascend/__init__.py` 的 `register()` / `adapt_patch(is_global_patch)` → 知道哪些 vllm 模块会被 patch、入口在哪、进程拓扑如何。
3. **优化对象的机制**：抽象出你的优化**真正需要的数据与介入点**（案例：KV 压缩需要每层 dense K/V、窗口 query、层重要性 = hidden 输入输出余弦相似度；换成注意力变体就是 query/key/value 与 mask 语义；换成采样就是 logits 与 sampling_metadata），而不是它的具体实现。
4. **在 vllm-ascend 里找数据在哪**：`grep` 定位候选目标——注意力：`AscendAttentionBackendImpl.forward`（TND query/key/value）、`reshape_and_cache`、`_build_attention_metadata`、`BlockTable.compute_slot_mapping`、`static_forward_context`；采样：`_sample`/`AscendSampler`/`sampling_metadata`；量化：`quantization/` 方法注册；自定义层：`models/` 与 `model_loader`。每类优化都有自己的"数据位置"清单，先 grep 再动手。
5. **验证每个 seam 的签名**：只 patch 从源码能逐行确认的方法；签名不确定一律 `*args/**kwargs` 包装 + 属性探测（`getattr` 多候选名）。
6. **先搭框架**（§0），把上面找到的调用点按行号落进流水线图。

### 2.2 选缝原则（通用）

- 优先 patch **worker 侧、每步重建的对象**（`_prepare_inputs`、`_build_attention_metadata`、`compute_slot_mapping`、backend `forward`、`_sample` 等）——每步重建，改写后下一帧自动恢复，不污染调度器状态；图回放（FULL_DECODE_ONLY）每步从当前 metadata 取参，修正天然生效。
- 需要"请求身份"的 hook，包装 `execute_model` 设置**每步 CaptureContext**（req_ids 顺序 == TND batch 顺序 == `input_batch.req_ids`）。
- **永远不要 patch 后不重建的持久对象**（如 `BlockPool`、scheduler、`self.seq_lens`、`optimistic_seq_lens_cpu`），除非你真的要动 engine-core。
- **时序敏感的改写**：先查框架 L2（该状态何时更新）再决定 hook 在流水线里的位置（案例：行重写必须在 `_prepare_inputs` 入口、commit 之前；positions 位移在 `compute_slot_mapping` 调用前设备端完成）。
- 完整已验证 seam 表见 `references/vllm-ascend-v023-seam-map.md`；框架见 `references/runtime-scheduling-framework.md`。

### 2.3 机制设计：把优化目标"转换"成框架能表达的形式（通用决策轴）

任何优化的落地都走同一条决策链，答案是"在哪个节点、以什么形态改写什么数据"：

| 决策轴 | 选项 | 判断依据 |
|---|---|---|
| **介入层** | 前向计算（算子/模块）/ 数据改写（元数据/张量）/ 采样后处理 | 你的优化改的是"算得对不对"还是"看什么数据" |
| **数据形态转换** | 上游库是稠密/连续形态，vllm-ascend 是块式/打包/TND 形态 → 必须先转换（案例：HF 稠密 cache ↔ 块式 cache） | 形态不转换就 patch = 维度/语义错乱 |
| **改写读路径还是写路径** | 读视图（每步重建的 metadata：block_tables/seq_lens 等）vs 物理内容（KV 张量/权重） | **读视图 = 前缀缓存安全、侵入最小、逐层/逐请求可变**；物理改写 = 粒度更细但破坏 hash、需同步写路径（案例：KV 压缩三选一，见下） |
| **粒度** | token / 块 / 层 / 请求 | 块式缓存的共享结构决定了最小可表达粒度（案例：块内所有 kv head 共享物理块 → head 统一保留集；FIA 按块读取 → token 级子集退化为块并集） |
| **时序约束** | 状态何时更新、hook 插在哪个节点前后 | 查框架 L2（案例：draft forward 在 sample_tokens 里跑 → 目标侧视图对 draft 无效） |
| **触发时机** | 完成时一次性 / **渐进式（预算推进，mid-prefill）** | 长上下文场景：资源（如 KV 显存）可能在"完成"之前就耗尽 → 完成时触发的优化永远不触发（**鸡生蛋**，真实真机踩坑：16×262144-token prompt、KV 占用 91.5%、`completed=0` 永远）→ 必须提供按 token 预算推进的渐进触发点 |

**案例：KV 压缩布局的三种表达**（kvpress/squeeze 实战，展示决策轴怎么用）：

| 机制 | 表达 | 前缀缓存 | 逐层/逐请求可变 | 粒度 | 侵入面 | 适用 |
|---|---|---|---|---|---|---|
| **A. 视图重写（view，默认）** | 每层视图行 = `[保留块]+[真行 m 起]`；`view_len = Σ min(bs, orig−b·bs) + (true_len−orig)` | **有效** | **支持** | 块级 | 仅 per-layer 元数据 | 用户带 `--enable-prefix-caching` 的场景 |
| **B. 尾部块物理搬移（compact）** | 保留 token 物理写进尾部块；`k = m − delta//bs` | **失效**（需 force） | 不可能（共享槽映射） | token 级（head 统一） | positions 位移 + 槽映射 + 一次性行重写 + 计数缩减 + seq_lens/cm | 无前缀缓存、要 token 级精度 |
| **C. 窗口视图（squeeze）** | 视图行 = `[sink 块]+[最后 recent 块]`；`view_len = true_len − (recent_first − sink_blocks)·bs` | 有效 | 支持 | 块级（边界 ±1 块近似） | 仅 per-layer 元数据 | StreamingLLM 式逐层窗口 |

**view 模式关键规则**（案例三条，违反即错）：① 强制保留最后一个非对齐块（`orig % bs != 0` 时块 `m−1` 必须进视图——新 decode token 落其 padding 槽，不保留则新 token 不可见；让位时只在已选块内 argmin）；② view_len 按块 token 数（末块部分填充感知）加新增 token，**末块内封顶，绝不读零 padding**；③ FIA 读 `view_row[p//bs]` 槽 `p%bs`——视图是块序列，不是 token 子集。

**compact 模式关键规则**（案例）：① `rewrite_row` **非幂等**——只做一次（标志位）+ `num_blocks_per_row` 缩减为 `k + (valid−m)`（否则 append 落点错乱）；② packing 槽 `repeat(rew[:k], bs)[:n_kept]·bs + (arange%bs)` 必须 `[:n_kept]` 截断；③ 打分 gather 只取 `orig_len` 个槽（绝不 `m·bs`，padding 污染 topk）；④ slack 不变量 `k·bs − n_kept ≥ m·bs − orig_len`；⑤ positions 设备端位移（`repeat_interleave` 按 query_start_loc 展开），热路径严禁 `.item()`。

**MTP/投机解码语义**（v0.23.0 实测，做任何与 KV/注意力相关的优化都要查）：`qwen3_5_mtp` → `AscendStep3p5MTPProposer`：draft 是**独立 per-MTP-layer KV group**，draft 元数据在 **sample_tokens 里**从 cm 重建 → **draft 不读 group-0 的视图**；共享 group-0 的 drafter（eagle/旧式 MTP）走 cm——要么重写 cm、要么接受 draft 看全量（安全降级）。**统一布局约束的真正来源不是 MTP，而是共享槽映射**。

**TP 分片**：每 rank 独立优化自己的分片；跨 rank 一致的参数要同步（案例：聚类用 all-reduce(MAX)，失败则独立聚类）。
**分块 prefill**：捕获只保留**最后一个 chunk 的尾部窗口**（每步覆盖）；需要全量数据的 pass 要**逐层流式**，控峰值内存。

### 2.4 交付物形态（monkeypatch 适配包通用骨架）

```
<name>-ascend/
  pyproject.toml            # hatchling；force-include 把 *.pth 打进 wheel 根 → pip 自动落到 site-packages
  <name>_ascend.pth        # 内容: "import <name>_ascend" —— 解释器启动自动导入（API server/engine-core/worker 全覆盖）
  <name>_ascend/
    __init__.py             # env 门控：未 export 时**完全不 import torch/vllm**（惰性）；开启后 apply()
    envs.py                 # 全部 env 变量集中定义 + 文档（开关/旋钮/日志级别）
    log.py                  # 独立 logger，前缀 [xxx-ascend]
    registry.py             # seam 探针 + 统计计数器 + 每步心跳
    engine.py               # 所有 monkeypatch + fail-soft try/except + 导入环 defuse
    core.py                 # 与设备无关的纯逻辑（优化算法本体）—— L0/L1 直接驱动
    simulate.py             # L1/L2 模拟器：fakes + 步骤驱动 + 自检 CLI
  tests/                    # pytest，全离线可跑（L0/L1/L2 + fail-soft + heartbeat）
  RISK_REGISTER.md          # 运行时风险登记 —— 与代码一起交付
  README.md                 # 用法 + 限制 + 真机核对清单
```

要点：**env 门控（未开启零导入）**、**fail-soft 全钩子**、**seam 探针 + 每步心跳**（证明优化真的进了核心代码）、**导入环 defuse**（激活时先按安全入口 `import vllm_ascend.ops.fused_moe.fused_moe`，失败则中止安装不留残留）、两包同开用策略 env 决出主策略（互相竞争同一数据时**不要**让两个同时改写）。

---

## 3. 调试方法论：系统化定义（Debugging）

> 本节定义"调试"本身：不是碰运气改代码，而是**一条可复现的证据链 + 一个回归测试**。任何 bug 都必须走完下面的状态机。

### 3.1 调试状态机（每个 bug 严格按状态推进）

每个状态有明确的**进入条件、动作、出口条件**。不允许跨状态跳跃：**没复现就改代码 = 猜测**；每个状态失败就回退到前一状态，不允许带着未决问题前进。

| 状态 | 输入 | 动作 | 出口条件（进入下一状态的门槛） |
|---|---|---|---|
| **SPEC** 定义预期 | 需求/设计/公式 | 写出可判定的预期：不变量、边界值、时序语义、失败模式 | 预期能被一条断言或一个对照实现表达 |
| **REPRO** 复现 | SPEC + 代码 | 最小输入稳定复现偏差；固定种子/固定数据/固定步骤数 | 无任何修改时 10/10 复现且可脚本化 |
| **ISOLATE** 隔离 | 复现脚本 | 二分定位：函数级 vs 端到端对照；减规模到最小失败面 | 在最小调用面内稳定失败，且能指出失败发生在哪个 seam |
| **ROOTCAUSE** 根因 | 隔离点 | 用证据（值/形状/时序/别名）解释全部症状，排除巧合与第二根因 | 能用一句话 + 一处代码解释所有观察到的症状 |
| **FIX** 修复 | 根因 | 最小改动；同步更新公式/文档/不变量；一次只变一个变量 | 改动不引入新行为面（diff 可审） |
| **VERIFY** 验证 | 修复 | 原复现脚本转绿；全量不变量套件；相邻用例（边界/多请求/多步） | 全绿，无回归 |
| **REGRESS** 固化 | 验证 | 复现脚本固化为回归测试；更新 bug 目录与风险登记 | 测试入库，bug 目录有记录，DoD 达成 |

### 3.2 调试三定律

1. **复现优先**：不能复现的 bug 不修——修了也无法验证，改错的风险大于收益。复现成本 > 4 小时时，先写"最小复现申请"（需要什么输入/环境）而不是猜。
2. **最小化**：任何断言失败，先减输入、减层数、减步骤、减请求，直到最小失败面。最小失败面决定了根因所在层。
3. **一次只变一个变量**：修复、数据、环境必须分开变；每变一次重跑 REPRO，记录结果。

### 3.3 证据纪律（地面真值）

**定义"正确"的来源优先级**：
1. **原始输入快照**（写入前的张量/数据）——最高优先级；
2. **独立参考实现**（朴素算法重写，与引擎实现无关）；
3. **数学推导**（不变量公式等）。

**禁止**：从你改过的内存读回当参考（改写写回后，"原始"已经不存在了——真实踩过的坑）；**参考集必须跨步累积**（多步场景的参考 = 保留集 + **全部**新数据，只加当前步会漏）。

**取证手段**（按证据类型）：
- 值证据：`assert allclose` + 逐行 diff + `argmin` 反查（错的槽里到底是哪个 token 的值）；
- 形状证据：每个跨函数张量的 shape/dtype 断言（TND vs BNSD、int32 vs int64）；
- 时序证据：记录调用顺序与"谁在什么时候更新了什么"（复刻 sample_tokens 时序）；
- 别名证据：张量是否共享/alias——用 op spy 确认写到了哪。

### 3.4 已知 bug 类目（ROOTCAUSE 阶段的模式库）

**A. 通用模式（任何 vllm-ascend 优化都会踩）**

| # | bug 模式 | 症状 | 检查手法 |
|---|---|---|---|
| G1 | 完成/进度判定用了过期的计数器（`num_computed` 在 sample_tokens 才更新） | 分块场景永远"未完成" | 判定用 `before + 本步 tokens`，且**允许 before==0** |
| G2 | 所有层/所有请求误用同一个对象（layer-0 的缓存、req-0 的状态） | 内容错乱、随规模变化 | 逐层/逐请求解析独立对象 |
| G3 | TND/BNSD/稠密形态没转换就喂给算子 | matmul 维数报错 | 捕获处先转 `(1, heads, w, hd)` 等目标形态 |
| G4 | gather/索引结果维度顺序错（`(seq,kv,hd)` vs `(1,kv,seq,hd)`） | 评分/改写维度错 | `index_select` 后 `transpose(0,1).unsqueeze(0)` |
| G5 | 改写了每步不重建的持久对象（`self.seq_lens`、`optimistic_seq_lens_cpu`、BlockPool） | 状态泄漏、跨步污染 | 只改每步重建对象；要改共享缓冲就换新张量 |
| G6 | 测试自身从被改内存读参考 / 参考集漏累积 | 断言莫名失败 | 地面真值纪律（3.3）；参考跨步累积 |
| G7 | **预导入扰动上游潜在循环导入**（vllm-ascend `ops/__init__ ↔ fused_moe ↔ experts_selector ↔ device_op ↔ ops.triton.fla`） | 启动期 `ImportError: cannot import name 'X' from partially initialized module` | 激活时先按安全入口 `import vllm_ascend.ops.fused_moe.fused_moe` defuse；失败中止安装不留残留 |
| G8 | **Enum 状态用 `.value` 比对字符串**（`AscendAttentionState` 的 `.value` 是 int） | 真机分支永远不触发；离线 mock 用字符串假绿 | `getattr(state, "name", state)` 兼容；None 检查先于解包 |
| G9 | **非法索引未做 CPU 前置守卫**（内容损坏/错位 → 设备端越界） | 设备端 `gather_v3 index out of range` / AIV `IndexCheckKernel::CheckUpperBound` 断言 → **NPU 流被污染** → try/except 救不回 → 下一同步点 worker 崩 | 任何设备算子前 CPU 校验：行内块 id ∈ [0, num_cache_blocks)、派生槽 ∈ [0, num_blocks·bs)、保留块 id 同界（**包括渐进压缩路径的每次 gather 与视图行写入**，真机 AIV 越界即此）；校验失败跳过该请求并打 `skipped_bad_row` 诊断（req/anchor/ids min-max/num_blocks）；下标用 `req_id_to_index` |
| G10 | **非重入锁 + 锁内再取锁** | 进程**静默挂死**（无 traceback，表现为超时） | 锁内只取数据快照，日志输出移到锁外 |
| G11 | **全局单例状态跨测试/跨步污染**（模块级 ctx、心跳 step 守卫） | 单测偶发失败、hook 不生效 | 测试内重置全局；必要时注入 ctx |
| G12 | **多包测试文件同名** | pytest `import file mismatch` 收集失败 | 测试文件 basename 全局唯一 |
| G13 | **ubatch（list 形态 attn_metadata）未守卫** | `.items()` 崩溃 / 错改 | `isinstance(..., (list, tuple))` → 跳过该步 |
| G14 | **空输入守卫缺失**（softmax 空张量、零长度切片） | NaN / 崩溃 | 返回"不优化"等价路径 |
| G15 | **热路径 `.item()`/同步** | 性能崩塌（async 调度阻塞） | 设备端批量操作；CPU 值只在每请求一次的非热路径取 |
| G16 | **逻辑 seam 未标记 installed**（压缩 pass/聚类 pass 等"包装内子步骤"只在心跳里 mark_hit，从未 mark_installed） | 心跳永远 `FAIL=<逻辑 seam>`——**假报警**，误导排查方向 | 逻辑 seam 随其宿主钩子一起 mark_installed；心跳口径：`seams=installed/total hit=N FAIL=...` |
| G17 | **完成时触发的优化 + 长上下文 = 鸡生蛋**（优化只在 prefill 完成后执行，但 KV 显存在任何请求完成前就耗尽 → 抢占循环 → `completed=0` 永远） | 服务跑了几百步，优化从未发生（心跳 `compressed=0`、无任何 skipped 计数） | 提供**渐进式触发点**（按 token 预算推进，mid-prefill 压缩）：在完成前按预算锚点压缩并推进锚点；完成时再以 prompt 长度重锚定；配套"回归式"状态清理（`before < 上次所见` 才判定抢占，不能再用 `before < prompt`——带渐进布局的请求仍处于 prefill 是正常态） |
| G18 | **pip 安装后源码改动不生效**（site-packages 的 `.pth` 在解释器启动时已把旧包注册进 sys.modules） | 改了源码重跑测试仍是旧行为（AttributeError 找不到新属性） | 开发循环先 `pip uninstall` 再测；发布前重装并核验 `kvpress_ascend.__file__` 指向 site-packages 新版 |
| G19 | **设备张量直接 `.numpy()`/`.tolist()`**（NPU/CUDA tensor 没有 numpy 转换；**CPU mock 永远测不出**） | 真机 `can't convert npu:N device type tensor to numpy`（mid-prefill/压缩 pass 失败，`skipped_error` 增长） | 统一走 `t.detach().cpu().numpy()` 助手；CPU/GPU 混合运算前先对齐设备（per-layer `meta.seq_lens` 是 CPU 张量、delta 在 NPU → 先 `.cpu()`）；回归：CPU 测试 + 真机日志 |
| G20 | **完成判定漏检**（末块跨步：调度器把最后一块跨步调度或计数口径与 `num_computed` 更新不一致 → `before + n_sched >= prompt` 未触发，但下一步 `before >= prompt`） | 请求已进 decode 但 `completed=0`、优化从未触发 | **补检**：`last_before < prompt <= before` 时视为上一步完成、本步补压缩（一次性，`_compressed_done` 去重）；完成触发点本身也应是渐进式的（G17） |
| G21 | **`not kc` 拦不住 `(None, None)` 元组**（kv_cache 未绑定的层，真机形态） | 打分器拿到 `keys=None` → `'NoneType' object has no attribute 'shape'`，且**整请求**压缩被 abort | 显式检查 `kc[0] is None or kc[1] is None`（+ `skipped_no_kv` 计数与层名日志）；**逐层 try/except**：坏一层只跳该层，其它层照常压缩 |
| G22 | **主动降级/让位被当作失败打 ERROR**（两个包互斥时的让位、DRY_RUN 等"故意不做"的路径） | 日志出现误导性 `ERROR ... installed with FAILED seams`，还引用一个根本没打印的 summary | install() 区分"让位/降级"（设 DEFERRED_REASON 之类标记）与"真实失败"：让位打 INFO + 原因，真实失败才打 summary + ERROR；日志消息用 ASCII 破折号防终端乱码（`—` 在部分终端显示为 `�~@~T`） |
| G23 | **整理/清理代码时的语义漂移**（"看起来等价"的改写悄悄改了语义：块下标 `int(b)` vs 块起始位置 `int(b)*bs`、转置顺序、`+1`/`-1`） | 行为静默变化，通常数步后才炸 | **不变量/端到端测试兜底**（本例 L2 视图不变量当场抓住多读一个尾块）；"等价改写"后必须跑全套不变量；编辑后重读完整 diff，逐符号核对 |
| G25 | **投机解码的 draft/MTP 层混入目标层列表**（step3.5 的 `mtp.layers.N.self_attn.attn` 出现在 kv_cache_config 的层列表里，但其 kv_cache 未按基础层方式绑定） | 每轮优化尝试都打 `'NoneType' ... 'shape'` 告警（逐层 fail-soft 已兜住但刷屏） | 结构性排除：`runner.drafter.attn_layer_names` + 名字启发式（`.mtp.`/`.draft.` 前缀）；被排除层计数 `layers_excluded_draft`；kv_cache 缺失告警**每层只报一次** |
| G26 | **设备端 `.sort()` int64 索引降级 AiCpu**（`topk(...).indices.sort()`；Ascend ArgSort 不支持 int32/int64） | 启动/运行期 `ArgSortKernelNpuOpApi` WARNING + 性能损失 | 排序移到 CPU/numpy（`np.sort(_as_numpy(idx))`）；topk 保持设备端 |
| G24 | **模拟 harness 自身的坑**（fake 块 id 超出 fake 缓存张量尺寸、driver 不维护 num_computed 导致误判 recompute、注意力参考实现 einsum/softmax 维度错、测试文件 basename 全局不唯一） | 模拟器报错或假绿，浪费整个调试轮次 | harness 冒烟：无补丁输出 == 朴素参考；fake 物理块 id 落在缓存尺寸内；driver 每步更新 num_computed（复刻 sample_tokens 时序）；参考实现先单测再进不变量；测试文件 basename 全局唯一 |

**B. 案例模式（KV 压缩/窗口类优化专属，其它优化项目按同样方式扩充本表）**

| # | bug 模式 | 症状 | 检查手法 |
|---|---|---|---|
| K1 | 布局 slack 不满足（k 公式错 / view_len 公式错） | 生成中后期写到错块/读错块 | L2 仿真多步 decode 不变量；view_len 按块 token 数封顶 |
| K2 | 共享槽映射下逐层不同 delta | 同 token 全层写乱 → 上下文损坏 | view 模式无此问题；compact 模式每请求统一 n_kept |
| K3 | 打分 gather 含尾块 padding（`m·bs` 而非 `orig_len`） | topk 被 padding 污染，保留集偏移 | `repeat(row, bs)[:orig_len]` |
| K4 | 行重写非幂等却每步重放 | 二次重写后行内容错乱 | 一次性 + 标志 + `num_blocks_per_row` 缩减 |
| K5 | 窗口注意力 k 转置顺序错（`unsqueeze(1).transpose(1,2)` vs `transpose(0,1).unsqueeze(1)`） | matmul 维度错/结果乱 | keys `(k_len,kvh,hd)` → `transpose(0,1).unsqueeze(1)` |
| K6 | 未捕获对象的 None 直接解包（`rc.queries.get(layer)[:n]`） | `TypeError: 'NoneType' object is not subscriptable` | 先判 None 再切片 |
| K7 | 强制保留尾块时"让位"选错范围（`argmin(全部)` 而非 `argmin(已选)`） | 丢的不是最低分保留块 | `argmin(block_scores[bl])` |
| K8 | 窗口边界重叠未钳位（recent 伸进 sink 块） | 视图行重复块 → 同一 token 读两次 | `recent_first = max(sink_blocks, ...)`；去重 |

> 全量 bug 目录（含琐碎项：模拟 harness 坑、测试污染、公式语义漂移等）见 `references/bug-catalog.md`——REGRESS 纪律：每个修复必须能在此找到一行记录 + 一个回归测试。

### 3.5 编码过程中的自我排查（Editing-time Self-Debugging）

> 状态机（3.1）管的是"bug 已经出现之后"；本节管的是"**正在写代码的那一刻**"——在错误进入 REPRO 之前就拦住它。编辑期的自查是最高效的调试：**改一步、验一步，永远不攒到最后**。自查失败 = 立即进入状态机，不要带着疑问继续写。

**编辑前（每次动手前）**：
1. **先读后改**：编辑工具要求先读文件再改，这是纪律不是手续——你要改的代码可能是别人（或几小时前的你）写的，语义以当前文件为准。
2. **说清本次 diff 的最小面**：这一改动哪些函数、碰哪些不变量、影响哪些 seam；对照 3.4 bug 类目，预判自己正踩哪个模式。
3. **确认验证手段已就位**：这次改动有测试/断言/CLI 能立即证明对吗？没有就先写验证，再写实现（测试先行在 patch 工程里同样成立）。
4. **回框架看一眼**：这个改动在调度框架的哪个节点？依赖的状态何时更新？框架图对不上 = 先更新框架再动手。

**编辑中（小步快验）**：
1. **一次一个小改动**，改完立即跑最小验证（`py_compile`/import 冒烟/单测/自检 CLI），不要连改五个文件再一起跑——失败时无法定位是哪个改动引入的。
2. **形状与 dtype 自检**：每个跨函数张量在注释或断言里写明 shape/dtype；TND vs BNSD、`(seq,kv,hd)` vs `(1,kv,seq,hd)`、int32 vs int64 是这类工程的高频雷区。
3. **别名意识**：`view/reshape` 是否 alias 底层缓存？写透测试前先确认"写进去"真的写到了目标张量（op spy 或读回断言）。
4. **设备与导入纪律**：包入口保持惰性（未启用时零 torch/vllm 导入）；CPU 可测路径与 NPU 专属路径隔离；`torch` 只在函数内惰性导入。
5. **可观测性随代码一起写**：每个新 hook 同时写 fail-soft 包装（try/except + 日志 + 计数器）——错误路径在写的那一刻就可观测，而不是上线后才知道。
6. **时序意识**：改任何"读状态"的代码前，先回答"这个状态是谁、在什么时候更新的"。

**编辑后（提交前自审，10 分钟起步）**：
1. **重读自己的完整 diff**（不是片段）：找死代码、未使用 import、复制粘贴只改一处忘另一处、下标/索引错位。
2. **对照 seam 表**逐个核对 patch 签名：每个 `getattr`/`*args` 探测是否有源码依据，还是猜的。
3. **跑 affected tests + 全量套件 + 自检 CLI**；全绿后**故意制造一次失败**（如 mock 缺字段、env 不设）确认 fail-soft 路径真的降级而不是炸穿。
4. **验证编辑确实生效**：改完重新读该区域，确认写入的内容与意图一致。

**编码时自我排查三问**（每次编辑后问自己，任一答不上 = 未完成）：
1. 我刚改的东西，**怎么证明它对**？（有没有断言/测试/CLI 覆盖；没有 = 立即补）
2. 我依赖的每个 API，**我从源码验证过吗**？（还是凭印象猜的；猜的 = 回到 2.1 去 grep 验证）
3. 如果真机上这里出问题，**我能从日志/计数器定位到吗**？（不能 = 补可观测性再走）

**编辑时特有的自伤模式**（区别于运行时 bug）：测试与实现互相将就（为让测试过而改实现语义）、在热路径里加 `.item()`/同步、把调试打印留在生产路径、改完不跑测试就继续写、用"看起来对"代替"断言过"。

---

## 4. 无硬件模拟调试协议（Simulated Debugging Without a Machine）

> 本节回答："没有 NPU、甚至没有 vllm 安装时，怎么系统化地完成调试排查？"
> 答案：**我们模拟的不是 NPU，而是被 patch 的 seam 及其数据流**。被改的只有 worker 侧方法，它们的输入输出是普通张量/数组/元数据对象——在 CPU 上完全可构造。NPU 专属物不模拟，用"不变量代理 + fail-soft 兜底 + 风险登记"处理。**这条对任何优化类型都成立**：采样优化模拟 logits/sampling_metadata 流，注意力变体模拟 TND query/key/value 流，方法完全一样。

### 4.1 模拟的哲学（先定义边界）

- **模拟对象**：① 被 patch 方法的接口契约；② vllm v1 的调用顺序与**时序陷阱**；③ 数据流（块表/槽映射/seq-lens/缓存内容/logits/metadata）的数值语义。
- **不模拟**：CANN 算子数值行为、设备流/同步、图捕获、显存带宽、真实性能。这些进入风险登记，由真机核对清单承接。
- **纪律**：引擎代码**不写 mock 专用分支**——模拟器驱动的是真实引擎路径，mock 只站在 vllm/vllm_ascend 一侧。

### 4.2 保真度分级（先定级别，再写测试）

| 级别 | 模拟什么 | 能抓到 | 抓不到 |
|---|---|---|---|
| **L0 纯逻辑单测** | 优化算法本体（布局公式/转换/索引/边界） | 数学、索引、形状、边界、不变量 | 与 vllm 对象的交互 |
| **L1 API-surface mock** | 被 patch 方法的签名与字段（FakeRunner.input_batch、FakeBlockTable、FakeAttnMeta 逐字段照抄 ascend 源码） | 参数/返回契约、属性访问、对象生命周期 | 调用顺序、跨方法状态、时序 |
| **L2 行为仿真** | L1 + 步骤驱动复刻 vllm v1 真实顺序与**时序陷阱**（sample_tokens 延迟更新、commit 顺序、FIA padding、MTP draft 元数据流） | 时序 bug、状态泄漏、跨步状态、多请求干扰 | 设备语义（流/同步） |
| **L3 全栈仿真（可选）** | L2 + 调度器块增长模型 + 多请求 + draft/target 一致性 | 调度交互、前缀缓存命中路径 | 真机性能、CANN 算子行为 |

规则：每个测试文件头部标注级别；**级别不足导致的"假绿"必须记录到风险登记**，不许悄悄当作验证通过。

### 4.3 搭建协议（按顺序执行）

1. **列 seam 清单与数据流图**（对照 `references/runtime-scheduling-framework.md` 与 seam map）：哪些方法被 patch、谁调用谁、每步谁更新什么。
2. **按 L0 → L1 → L2 搭建**：先纯函数（无对象依赖），再 mock 对象（接口从源码逐字段照抄），最后步骤驱动。
3. **冒烟验证 mock 保真**：未启用补丁时，模拟器的输出必须与朴素参考一致（例如：无优化时模拟注意力 == 直接 matmul 参考；无优化时模拟采样 == 朴素 argmax）——mock 本身错了，后面全白搭。
4. **建不变量注册表**：每条不变量 = 一句断言 + 覆盖的环节 + 级别 + 对应的风险项。

### 4.4 步骤驱动（复刻 vllm v1 的真实顺序）

```
execute_model_pre（快照 before 状态：num_computed/num_scheduled/num_prompt/req_ids）
→ 调度器 grow 块表行（按 token 数 ceil，模拟 engine-core 的分配；
   物理块 id 必须落在 fake 缓存张量尺寸内）
→ _prepare_inputs 入口（改写点；随后 commit_block_table 拷贝 CPU→GPU）
→ positions / query_start_loc
→ compute_slot_mapping（compact：positions 设备端减 delta 后算槽）
→ backend forward（query 捕获）→ attention 模块 forward（hidden/attn-out 捕获）
→ 按层写 KV（reshape_and_cache 语义，**每层自己的缓存**）
→ execute_model_post（优化 pass）
→ 最后才更新 num_computed（复刻 sample_tokens 时序！driver 必须每步更新 fake 的
   num_computed_tokens_cpu，否则 recompute 检测会误删状态）
```

### 4.5 端到端不变量（一票否决级）

优化的核心语义必须能写成一条可判定的数值不变量，**跑多步**（**必须越过块/状态边界**，触发封顶/越界 bug），断言：

```
优化后可见数据参与的计算 == 参考实现（原始快照按保留规则 + 逐步累积的全部新数据）   # 误差 < 1e-4
```

- 可见集 = 用**改写后的视图/数据 + 修正后的长度** gather 的槽位或取值；
- 参考集 = **改写前保存的原始快照**按规则取 + **逐步累积的**新数据；
- 这条不变量同时验证：评分/选择、改写写入、长度修正、索引映射——全部环节。
- 附加断言：**最新数据永远可见**（优化后视图必须包含本步新增数据的槽）。

### 4.6 运行时风险登记（Runtime Risk Register，RTR）

模拟覆盖不了的东西逐项登记，**随代码一起交付**（每个项目逐条写清"为什么模拟覆盖不了 / 真机验证方法 / fail-soft 兜底"）。通用条目：CANN 算子数值差异、cudagraph 捕获/回放、流/同步竞态、前缀缓存 hash 交互、MTP draft 一致性、性能。案例条目见 kvpress-ascend/SqueezeAttention-ascend 的 RISK_REGISTER.md。

### 4.7 模拟调试的完成定义（Definition of Done）

同时满足才可声称"模拟调试完成"：
1. L0/L1/L2 测试全绿，且覆盖**全部被 patch 的 seam**；
2. 不变量注册表每条都有对应测试，端到端不变量跑到**多步越过边界**；
3. RTR 建立：每个"模拟覆盖不了"的项都有真机验证方法与兜底；
4. 自检 CLI 可运行（`python -m <pkg>.simulate`）；
5. bug 目录中新发现已固化（REGRESS）；
6. fail-soft 注入测试在（缺字段/env 未设/坏数据 → 降级不崩溃）；
7. **安装链路验证**：`pip install ./<pkg>` 成功、`.pth` 落 site-packages（`zipfile` 核验 wheel 根）、未 export 时 `assert 'torch' not in sys.modules`、无 vllm 环境激活时 fail-soft 降级不崩溃。

未满足 DoD，汇报时必须说"模拟调试进行到 X 级"，不得声称已调试完成。

### 4.8 模拟调试的交付物

- 分级测试套件（`tests/`，L0/L1/L2 + fail-soft + heartbeat）；自检 CLI；不变量注册表；
- `RISK_REGISTER.md`；**真机核对清单**（精度对比、前缀缓存对照、MTP 接受率、长跑稳定性、性能基线、心跳 seam 全 OK）。

---

## 5. 真机排查阶段（Troubleshooting）

### 5.1 与 RTR 对接（真机第一跑）

按 4.8 的真机核对清单逐项执行、逐项销项；新发现回填 bug 目录与 RTR；模拟阶段"假绿"项在此暴露。

### 5.2 日志驱动（通用）

- 包日志前缀 `[xxx-ascend]`，等级 `XXX_ASCEND_LOG=debug|info|warning`。
- **每步心跳（`XXX_ASCEND_STEP_LOG=1`，默认开）**：每步一行打印优化是否进入核心代码（seam 探针 `seams=N/N`、FAIL 项点名）与核心参数（优化名/关键旋钮/逐请求状态）；激活时还有一次全 seam 汇总。心跳缺失或 FAIL = patch 没进核心代码，先查错误日志。
- 统计计数器：`completed / applied / skipped_short / skipped_<原因> / skipped_error / dry_run` —— 一眼看出每次请求被跳过在哪一环。
- `XXX_ASCEND_DRY_RUN=1`：只算不改写，先确认统计正常再开真优化。

### 5.3 失败分类与对策（通用 + 案例）

| 现象 | 原因 | 对策 |
|---|---|---|
| 一直 `skipped_<原因>` | 用户配置与优化前提冲突（如前缀缓存/量化/图模式） | 逐项核对前提；给出两条路（改配置 or force 自担风险） |
| 服务跑几百步但优化从未发生（心跳 `compressed=0`、无 skipped 计数） | **长上下文鸡生蛋**：优化只在 prefill 完成后触发，但资源在完成前就耗尽（抢占循环） | 开渐进式触发（案例：`KVPRESS_ASCEND_MID_PREFILL=1` + `KVPRESS_ASCEND_MID_PREFILL_BUDGET`/`REFRESH`）；心跳新增 `prefilling`/`mid_anchored` 与恒显计数器定位 |
| 心跳 `FAIL=<逻辑 seam>`（如压缩 pass/聚类 pass） | 逻辑 seam 未随宿主钩子 mark_installed → 假报警 | 更新包（该 seam 应标记 installed）；真 FAIL 只可能是真实钩子未装上 |
| 优化了但收益不明显 | worker 侧物理边界（不动调度器分配） | 如实说明省的是什么；要动资源需改 engine-core |
| 逐层/逐请求参数一样 | 聚类/统计输入缺失 → 中性值兜底 | 检查捕获日志（多请求混合步会跳过捕获） |
| `skipped_error` 增长 | 某 seam API 对不上 / 守卫触发 | debug 级日志看 traceback；查 seam 表核对版本 |
| 服务照常但没有任何优化日志 | env 没生效 / .pth 没装上 | 检查 site-packages 里 `.pth`；`python -c "import <pkg>"`；未 export 时 `assert 'torch' not in sys.modules` 反证门控 |
| 多个优化包同时 export 但只有一个生效 | 策略机制（默认先装者/主策略优先） | 用策略 env 显式指定；互相竞争同一数据时不要同时改写 |
| 性能下降 | 热路径 `.item()`/每步分配/全量拷贝 | 设备端化；预分配缓冲；无优化请求 fast-path；每请求只做一次 |

### 5.4 用户启动命令的快速体检（通用 checklist）

1. **投机解码**（`--speculative_config`）→ 查框架 L2/L3：draft 何时跑、读什么元数据（step3.5 独立 group vs 共享 group）。
2. **前缀缓存**（`--enable-prefix-caching`）→ 你的优化是否碰物理缓存内容；碰了 = hash 失效风险。
3. **TP/DP/PP 并行** → 每 rank 独立执行；跨 rank 一致的参数要同步。
4. **图模式**（`--compilation-config` FULL_DECODE_ONLY）→ 只 patch 每步重建对象；图捕获期假元数据别碰；回放每步从当前 metadata 取参。
5. **分块 prefill**（小 `--max-num-batched-tokens`）→ 长 prompt 必然分块，完成/进度判定用 `before + 本步`。

---

## 6. 如实汇报（职业底线）

交付时必须在 README/总结里写清：
1. **机制取舍**：为什么选"读视图改写"而不是"物理改写"（或反之）——前缀缓存/侵入面/粒度的权衡，用户配置冲突时直接给推荐；
2. **物理边界**：worker 侧 patch 动不了调度器资源（内存回收等），如实说明收益范围；
3. **MTP/并行语义**：draft 可见性、跨 rank 一致性、分块 prefill 的完成判定；
4. **近似与约束**：块式/共享结构带来的粒度近似（head 统一、块粒度等），README 明示；
5. **模拟覆盖级别与 RTR**：哪些环节已由 L0-L2 离线验证、哪些仍需真机确认（逐项列）；未达 DoD 时明确说"模拟进行到哪一级"。

参考文件：
- `references/runtime-scheduling-framework.md`（**运行调度框架**：进程 → 每步流水线(带行号) → 状态时序 → 钩子叠加 → 数据流；先搭框架、debug 持续更新——对任何优化类型通用）
- `references/vllm-ascend-v023-seam-map.md`（v0.23.0 已验证 seam/API 表 + KV 压缩案例公式；新版本/新优化类型请按同样格式扩充）
- `references/bug-catalog.md`（**bug 目录**：实战全清单 26 条，含琐碎项——症状/根因/修复/类目/发现途径/回归测试；REGRESS 纪律：每个修复必须能在此找到一行记录 + 一个回归测试）
