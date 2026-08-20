# SqueezeAttention-ascend

Monkeypatch 适配器：把 [SqueezeAttention](https://github.com/Ledzy/SqueezeAttention) 的**逐层 KV 滑动窗口**机制适配到 **vllm-ascend v0.23.0**，**不修改 vllm-ascend 任何源码**。

## 安装与激活

```bash
pip install ./SqueezeAttention-ascend   # 安装后 .pth 自动落 site-packages
export squeeze=1                         # 主开关（等价: squeezeattention=1 / squeeze_ascend=1 / exportsqueeze=1）
# 然后正常拉起 vllm serve ...
```

- 未 export 时：**零 torch/vllm 导入**。
- 与 kvpress-ascend 同开时默认让位给 kvpress；强制本包：`export SQUEEZE_ASCEND_POLICY=squeeze`（并 `unset kvpress` 或保持 `KVPRESS_ASCEND_POLICY != squeeze`）。

## 每次推理的开关验证（心跳日志）

```
[squeeze-ascend] INFO step=42 seams=6/6 ini=0.30 class3=0.10 start=4 w_min=512 w_max=65536 active_windowed=2 completed=0 attn_state=DecodeOnly clustered=2 ...
```

- `seams=x/6` + `FAIL=` 点名未安装 seam；激活时打印一次全 seam 汇总。
- 核心参数：`ini`（总预算比例）、`class3`（class-3 层窗口比例）、`start`（sink）、`w_min/w_max`（当前逐层窗口范围）。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `squeeze` / `squeezeattention` / `squeeze_ascend` / `exportsqueeze` | — | 主开关 |
| `SQUEEZE_ASCEND_ENABLE` | — | 显式开关 |
| `SQUEEZE_ASCEND_INI_SIZE` | `0.3` | 总 KV 预算 = `num_layers × ini × prompt_len` |
| `SQUEEZE_ASCEND_CLASS3_RATIO` | `0.1` | class-3 层（cos 相似度最高、attention 改变最少）窗口 = `ratio × prompt_len` |
| `SQUEEZE_ASCEND_START_SIZE` | `4` | sink token 数（StreamingLLM 风格） |
| `SQUEEZE_ASCEND_MIN_PROMPT` | `256` | 更短的 prompt 不窗口化 |
| `SQUEEZE_ASCEND_DRY_RUN` | `0` | 只聚类不重写视图 |
| `SQUEEZE_ASCEND_STEP_LOG` | `1` | 每步心跳 |
| `SQUEEZE_ASCEND_LOG` | `info` | 日志级别 |
| `SQUEEZE_ASCEND_POLICY` | `auto` | `squeeze` 强制本包生效 |

## 机制（与上游的差异）

上游在 HF 逐层 forward 里直接裁剪 `past_key_value` 并重映射 position id。本适配器在 vllm v1 块式缓存上的转换：

1. **层重要性**：prefill 期间经两层包装（解码层 forward 捕获 pre-layernorm 输入、`Attention.forward` 捕获注意力输出）累积 `cos_sim(层输入, 层输入+attn输出)` 的逐 token 均值（单请求 prefill 步才捕获）。
2. **聚类**：prefill 完成时对逐层均值做自包含 1D KMeans（3 类，确定性初始化）→ 原版预算公式：
   `a = (N×ini − n3×ratio) / n12`；class1/2 窗口 = `a×prompt_len`，class3 = `ratio×prompt_len`；钳位到 `[start+1, prompt_len]`。
   TP 多卡时对逐层均值做 all-reduce(MAX) 保证各 rank 窗口一致（失败则跳过同步，各 rank 独立聚类）。
3. **窗口视图（decode 每步）**：对每层 `attn_metadata[layer]` 重写：
   - `block_tables` 行 = `[sink 块] + [最后 recent 块]`（recent = window − start；块级近似：边界块最多多读 ~1 块）；
   - `seq_lens/seq_lens_cpu/seq_lens_list` = `true_len − (recent_first − sink_blocks)×bs`（**在最后一块内部封顶，绝不读零 padding**）。
   - 写路径（slot mapping/positions/块表行）**零改动** → 前缀缓存 hash 有效、MTP step3.5 draft 独立 group 无冲突。
4. **内存边界**：worker 侧不回收块；省的是注意力计算/带宽。

## 离线模拟调试

```bash
cd SqueezeAttention-ascend
python -m squeeze_ascend.simulate --steps 8
python -m pytest tests/ -q     # L0/L1/L2 全绿
```

L2 端到端不变量：多步 decode 后 `attention(窗口视图) == attention(参考窗口 token 集)`（块边界近似误差有界）。

## 真机核对清单

- [ ] 启动日志 `[squeeze-ascend] activation summary: seams=6/6`
- [ ] 首个长 prompt 完成后出现 `clustered=1` 与逐层窗口日志；随后心跳 `w_min/w_max` 非零
- [ ] `SQUEEZE_ASCEND_DRY_RUN=1` 先跑一轮
- [ ] 精度对比：同 prompt 开/关 squeeze 输出对比（长文摘要）
- [ ] 前缀缓存对照：开/关 `--enable-prefix-caching` 一致
- [ ] MTP：draft acceptance 无劣化
- [ ] 长跑稳定性：`skipped_error` 不增长、心跳无 FAIL
- [ ] 性能基线：TTFT/TPOT 对比

## 已知限制

- 窗口为**块粒度**近似（每窗口边界最多 ±1 块）。
- 聚类只对"唯一 prefill 请求"的步有效；多请求混合 prefill 步跳过捕获（该请求窗口退化为全量）。
- 不回收显存块；收益来自注意力计算量下降。
- 逐层窗口在 MTP 共享 group-0 的 drafter 下不可见（draft 看到全量缓存，安全）。
