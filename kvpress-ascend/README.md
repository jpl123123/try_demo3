# kvpress-ascend

Monkeypatch 适配器：把 [kvpress](https://github.com/NVIDIA/kvpress) 的 KV cache 压缩方法适配到 **vllm-ascend v0.23.0**，**不修改 vllm-ascend 任何源码**。

## 安装与激活

```bash
pip install ./kvpress-ascend          # 安装后 .pth 自动落 site-packages
export kvpress=1                       # 主开关（等价: kvpress_ascend=1 / exportkvpress=1）
# 然后正常拉起 vllm serve ...
```

- 未 export 时：**零 torch/vllm 导入**，进程完全无感。
- `.pth` 在每个 Python 进程启动时自动 `import kvpress_ascend`（API server / engine-core / worker 全覆盖）。
- 与 SqueezeAttention-ascend 同开时：默认 **kvpress 生效**（squeeze 自动让位）；若想 squeeze 生效，设 `export SQUEEZE_ASCEND_POLICY=squeeze`。

## 每次推理的开关验证（心跳日志）

每次 `execute_model` 打一行，证明 patch 进了核心代码（seam 探针）与核心参数：

```
[kvpress-ascend] INFO step=42 seams=6/8 FAIL=... press=snapkv ratio=0.50 window=64 sink=4 mode=view layers=4 completed=0 active_compressed=2 attn_state=DecodeOnly compressed=3 ...
```

- `seams=x/8`：本步命中探针数；`FAIL=` 点名未安装的 seam → patch 没进核心代码。
- 激活时打印一次全 seam 汇总。
- 关闭：`export KVPRESS_ASCEND_STEP_LOG=0`；日志级别 `export KVPRESS_ASCEND_LOG=debug|info|warning`。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `kvpress` / `kvpress_ascend` / `exportkvpress` | — | 主开关 |
| `KVPRESS_ASCEND_ENABLE` | — | 显式开关 |
| `KVPRESS_ASCEND_PRESS` | `snapkv` | `knorm` / `random` / `streamingllm` / `snapkv` / `tova` / `pyramidkv` / `expected_attention` / `criticalkv` |
| `KVPRESS_ASCEND_RATIO` | `0.5` | 压缩比例（块粒度） |
| `KVPRESS_ASCEND_WINDOW` | `64` | SnapKV/TOVA 观察窗口 |
| `KVPRESS_ASCEND_SINK` | `4` | StreamingLLM sink 数 |
| `KVPRESS_ASCEND_KERNEL` | `5` | SnapKV avg_pool 核 |
| `KVPRESS_ASCEND_MODE` | `view` | `view`（默认，见下）或 `compact` |
| `KVPRESS_ASCEND_PREFIX_CACHE` | — | view 模式无冲突；compact 模式需 `force` 自担风险 |
| `KVPRESS_ASCEND_DRY_RUN` | `0` | 只打分、不记录视图（安全排练） |
| `KVPRESS_ASCEND_MIN_PROMPT` | `512` | 短 prompt 不压缩 |
| `KVPRESS_ASCEND_STEP_LOG` | `1` | 每步心跳 |
| `KVPRESS_ASCEND_LOG` | `info` | 日志级别 |
| `KVPRESS_ASCEND_POLICY` | `both` | `squeeze` 时本包让位 |
| `KVPRESS_ASCEND_SKIP_DRAFT_STEPS` | `0` | 有 draft token 的步跳过视图重写（保守） |

## 机制（与上游 kvpress 的差异，务必阅读）

上游 kvpress 在 HF `DynamicCache`（稠密 `(bs, kv_heads, seq, hd)`）上做**逐 head、token 粒度**的 top-k 剪枝并原地替换缓存。vllm v1 是**块式共享 KV cache**，因此：

1. **逐 head 保留集 → head 统一保留集**：块内所有 kv head 共享同一物理块，无法表达逐 head 的 token 子集。所有 press 先对 kv heads 求平均得到 `(seq,)` 分数（README 明示的近似）。
2. **token 粒度 → 块粒度（`view` 模式）**：分数按块聚合（均值），保留整块。块边界最多各多读 ~1 块，文档化近似。
3. **写路径零改动**：`view` 模式**不碰**物理缓存、块表行、positions、slot mapping —— 每步只重写 `attn_metadata[layer]` 的 `block_tables`（视图行）与 `seq_lens/seq_lens_cpu/seq_lens_list`（视图长度）。因此：
   - **前缀缓存 hash 依然有效**（物理内容不变）；
   - MTP draft（step3.5 独立 KV group）天然无冲突；共享 group 的 drafter 看到全量缓存（安全降级）；
   - 逐层预算（PyramidKV）可用。
4. **`compact` 模式（可选）**：token 粒度 top-k 物理搬入尾部块（skill 的布局公式 `k = m - delta//bs` + slack 不变量），需要 positions 位移 + 槽映射重算 + 一次性行重写（`num_blocks_per_row` 缩减为 `k + rest`）+ 前缀缓存 `force`。**默认不启用**。
5. **内存边界**：worker 侧无法把块还给调度器，两种模式**都不回收块内存**，省的是注意力计算/带宽（FIA 按 `actual_seq_lengths_kv` 计工作量）。

## 离线模拟调试（本机无 NPU）

```bash
cd kvpress-ascend
python -m kvpress_ascend.simulate --steps 8 --mode view    # 场景自检
python -m kvpress_ascend.simulate --steps 8 --mode compact
python -m pytest tests/ -q                                   # L0/L1/L2 全绿
```

- L2 端到端不变量：多步 decode 越过块边界，`attention(视图 K/V) == attention(参考保留集+新 token)`（误差 < 1e-4）。
- 模拟覆盖级别与真机风险见 `RISK_REGISTER.md`。

## 真机核对清单（第一跑逐项勾选）

- [ ] 启动日志出现 `[kvpress-ascend] activation summary: seams=8/8`（每进程各一次）
- [ ] 首个长 prompt prefill 完成后日志出现 `compressed=N`，随后每步心跳 `active_compressed>=1`
- [ ] `KVPRESS_ASCEND_DRY_RUN=1` 先跑一轮：确认打分/统计正常再开真压缩
- [ ] 精度对比：同 prompt 压缩 vs 不压缩的输出差异（长文摘要类任务）
- [ ] 前缀缓存对照：开/关 `--enable-prefix-caching` 输出一致（view 模式应一致）
- [ ] MTP：`draft acceptance` 无明显劣化（draft 看到全量缓存，应无影响）
- [ ] 长跑稳定性：无 `skipped_error` 增长；心跳无 FAIL
- [ ] 性能基线：TTFT/TPOT 对比（压缩后 decode 步应下降）

## 已知限制

- 块粒度（view 模式）与 head 统一保留集是块式缓存的硬约束（见上）。
- 不回收显存块；收益来自注意力计算量下降。
- `ExpectedAttention`/`CriticalKV` 为 best-effort（依赖模块属性，缺失时自动退化为 Knorm 打分）。
- `compact` 模式与 `--enable-prefix-caching` 冲突，需 `KVPRESS_ASCEND_PREFIX_CACHE=force` 自担风险。
