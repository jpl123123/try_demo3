# SqueezeAttention-ascend 运行时风险登记（RTR）

> 模拟完成级别：**L0/L1/L2 全绿**（17 tests，含多步 decode 窗口视图端到端不变量）。

| # | 风险项 | 为什么模拟覆盖不了 | 真机验证方法 | fail-soft 兜底 |
|---|---|---|---|---|
| 1 | FIA 核数值/图回放 | 无 NPU 算子 | 精度对比；DRY_RUN 先行 | 只重写每步重建的元数据对象 |
| 2 | 层重要性捕获与真实层结构的吻合（vllm v1 decoder layer 结构） | 模拟用通用层包装 | 真机看 `clustered` 日志与窗口分布是否合理 | 捕获失败该层重要性取 0.5 中性值 |
| 3 | TP 多卡聚类一致性 | all-reduce(MAX) 失败路径未在真机验证 | 真机 4 卡看各 rank 心跳 `w_min/w_max` 一致 | 同步失败则各 rank 独立聚类（保守：窗口可能略异，不影响正确性） |
| 4 | 前缀缓存 | 物理内容未改，理论 hash 有效 | 真机对照 | 不碰物理缓存 |
| 5 | MTP draft 交互 | step3.5 draft 独立 group；共享 group drafter 看到全量 | 真机 draft acceptance | 不重写 cm；`SQUEEZE_ASCEND_DRY_RUN` 可验证 |
| 6 | 每步视图行拷贝开销 | 无带宽模型 | benchmark TPOT | 无窗口请求时 fast-path；缓冲预分配 |
| 7 | preempt/recompute | 模拟 num_computed 回退路径 | 真机压测 | 每步校验回退即清窗口 |
| 8 | 多请求混合 prefill 步 | 捕获只对单 prefill 步 | 并发长 prompt | 跳过捕获（窗口退化为全量，安全） |
| 9 | `start_size` 与窗口极小值下的块边界行为 | L0/L1 覆盖公式边界 | 真机小窗口长跑 | 窗口钳位 `[start+1, prompt_len]` |

## 不变量注册表（L2 端到端）

| 不变量 | 覆盖环节 | 级别 |
|---|---|---|
| I1 窗口视图可见槽 == `[0,start) ∪ [L−recent, L)` 的块覆盖（去重，重叠钳位） | 窗口行构造 | L2 |
| I2 最新 token 永远可见（视图长度在最后一块内封顶） | view_len 公式 | L2 |
| I3 attention(窗口视图) == attention(参考窗口)，误差有界（块边界近似） | 全链路 | L2 |
| I4 聚类：class3（最高 cos 相似）层窗口最小；预算守恒（含钳位容差） | 预算公式 | L0/L1 |
| L0: `window_block_ranges` 无重写判定、重叠钳位、确定性 KMeans | kvcore | L0 |
