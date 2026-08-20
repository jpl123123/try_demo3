# kvpress-ascend 运行时风险登记（RTR）

> 模拟调试（L0/L1/L2）无法覆盖的项逐条登记：真机验证方法 + fail-soft 兜底。
> 模拟完成级别：**L0/L1/L2 全绿**（25 tests，含多步 decode 跨块边界端到端不变量）。

| # | 风险项 | 为什么模拟覆盖不了 | 真机验证方法 | fail-soft 兜底 |
|---|---|---|---|---|
| 1 | FIA 核的数值/图回放行为 | 无 NPU 算子 | 精度对比（压缩 vs 不压缩）；`KVPRESS_ASCEND_DRY_RUN=1` 先行 | 只重写每步重建的元数据对象；异常捕获降级为不压缩 |
| 2 | `update_graph_params`（FULL_DECODE_ONLY）对 `block_tables` 视图的逐层绑定 | 图回放是设备语义 | 真机开 FULL_DECODE_ONLY 跑通 + 精度 | 视图行写入每层预分配缓冲（形状与捕获一致，行尾补零） |
| 3 | 前缀缓存（view 模式） | 物理内容未改，理论上 hash 依然有效 | 真机开 `--enable-prefix-caching` 前后对照输出 | 不碰物理缓存；若实测异常，可设 `KVPRESS_ASCEND_PREFIX_CACHE=skip` 关闭 |
| 4 | 前缀缓存（compact 模式） | 物理改写尾部块会使部分前缀 hash 失效 | 仅 `KVPRESS_ASCEND_PREFIX_CACHE=force` 时启用；对照输出 | 默认不启用 compact；心跳统计显示模式 |
| 5 | MTP draft 与视图的交互 | step3.5 draft 有独立 KV group；共享 group 的 drafter 走 cm | 真机看 draft acceptance rate | draft 看到全量缓存（安全降级）；`KVPRESS_ASCEND_SKIP_DRAFT_STEPS=1` 可跳过含 draft 的步 |
| 6 | 每步视图行拷贝开销 | 无真实带宽模型 | benchmark TPOT | 无压缩请求时 fast-path 零拷贝；每层缓冲预分配；热路径无 `.item()` |
| 7 | preempt/recompute 后行与布局 | 只模拟了 `num_computed` 回退路径 | 真机压测抢占/恢复 | 每步校验 `num_computed < num_prompt` 即清布局 + 行重写标志 |
| 8 | 多请求混合 prefill 步的捕获 | 模拟只覆盖单请求 prefill | 并发长 prompt 场景 | 捕获只对"唯一 prefill 请求"的步生效；超过 `KVPRESS_ASCEND_MAX_PREFILLS` 丢弃捕获（`capture_dropped_cap`） |
| 9 | `ExpectedAttention`/`CriticalKV` 的模块属性探测 | 依赖真机 Attention 模块（qkv_proj/q_norm/o_proj/rotary_emb） | 真机选这两个 press 看日志是否 fallback | 缺失属性自动退化为 Knorm 打分 |
| 10 | `prev_positions`（spec decode 簿记）与压缩坐标 | 上游 `_compute_prev_positions` 无本地源码 | 真机 MTP 长跑 | compact 模式默认 MTP 统一布局；视图重写不动 positions；异常时心跳 `skipped_error` 可见 |
| 11 | C8（INT8 KV）后端 | `AscendC8AttentionBackendImpl` 的 forward 签名已包装但未在真机触发 | `--quantization ascend` + w8a8 下观察 S1b seam hit | 包装 fail-soft；捕获失败只丢该层打分 |
| 12 | 多 KV group 模型（如 step3.5 MTP 层组） | 只对 group-0 布局 | 真机观察各 group 行为 | 布局只作用于 group-0 层；其他组不触碰 |

## 不变量注册表（L2 端到端，一票否决级）

| 不变量 | 覆盖环节 | 级别 |
|---|---|---|
| I1 view: FIA 可见槽 == 保留块 token ∪ 新 token | 视图行 + seq-lens 重写 | L2 |
| I2 view: 最新 token 永远可见（含尾部块 padding 落入新 token 的边界情形） | 强制保留最后一个非对齐块 | L2 |
| I3 view: attention(视图) == attention(参考集)，误差 < 1e-4 | 评分→块聚合→视图→注意力全链路 | L2 |
| I4 compact: 尾部槽内容 == 保留参考 KV | 物理搬移写回 | L2 |
| I5 compact: 新 token 落压缩槽 `n_kept+j`（positions 位移 + 槽映射 + 行重写 + 计数缩减） | S3/S7 全链路 | L2 |
| I6 compact: attention(压缩视图) == attention(保留+新)，误差 < 1e-4 | 多步 decode 跨块边界 | L2 |
| L0: slack 不变量 `k*bs - n_kept >= m*bs - orig_len`、rewrite_row 一次性语义、n_kept 公式 | kvcore | L0 |
