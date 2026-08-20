# bug 目录：kvpress-ascend / SqueezeAttention-ascend 适配实战全清单

> 本目录收录实战中**所有** bug（含琐碎项），每条含：症状 → 根因一句话 → 修复 → 类目（对应 SKILL.md §3.4 的 Gx/Kx 模式）→ 发现途径 → 回归测试。
> 新项目请按同格式追加自己的 bug（REGRESS 纪律：每个修复必须能在这里找到一行记录 + 一个回归测试）。

## A. 开发期（L0/L1/L2 模拟测试发现）

| # | 症状 | 根因 | 修复 | 类目 | 发现途径 | 回归测试 |
|---|---|---|---|---|---|---|
| A1 | pytest 静默挂死（无 traceback，表现为超时） | registry 非重入锁：`with _lock:` 内调用会再取同一把锁的 `seams_summary()` | 锁内只取数据快照，日志输出移到锁外 | G10 | 测试超时排查 | test_heartbeat |
| A2 | topk 保留集偏移 | 打分 gather 用 `m*bs` 个槽（含尾块 padding），padding 内容污染分数 | `repeat(row, bs)[:orig_len]`，只取真实 token 槽 | K3 | L2 保留集断言 | test_l2_invariants |
| A3 | 行重写二次执行后内容错乱 | `rewrite_row` 非幂等，却每步重放 | 一次性重写 + `row_rewritten` 标志 + `num_blocks_per_row` 缩减为 `k+(valid−m)` | K4 | L2 compact 不变量 | test_l2_invariants |
| A4 | 窗口注意力 matmul 维度错/结果乱 | keys `unsqueeze(1).transpose(1,2)` 顺序错 | `transpose(0,1).unsqueeze(1)` → `(kvh,1,k_len,hd)` | K5 | 模拟器报错 | simulate/snapkv |
| A5 | 单步完成整个 prompt 的请求永远"未完成" | 完成判定 `0 < before` 漏掉 `before==0` 首步 | `before < prompt and before + n_sched >= prompt` | G1 | L1 测试 | test_l1_capture |
| A6 | `TypeError: 'NoneType' object is not subscriptable` | 未捕获层的 `rc.queries.get(layer)[:n]` 直接切片 | 先判 None 再切片 | K6 | 模拟器报错 | test_l1_capture |
| A7 | 强制保留尾块时丢错块 | "让位"用 `argmin(全部块分)` 而非 `argmin(已选块分)` | `argmin(bscores[bl])` | K7 | L2 保留集断言 | test_l2_invariants |
| A8 | 多步 decode 参考集断言莫名失败 | 测试参考集只含当前步新 token，漏累积 | 参考集跨步累积全部新 token | G6 | 不变量断言 | test_l2_invariants |
| A9 | 心跳单测偶发失败 | `registry._last_step` 全局守卫被前序测试污染 | 测试内重置 `_last_step = -1` | G11 | 测试隔离排查 | test_heartbeat |
| A10 | pytest `import file mismatch` 收集失败 | 两包测试文件同名（test_l0_kvcore.py） | basename 全局唯一（改名 test_l0_window_math.py） | G12 | pytest 收集报错 | — |
| A11 | S7 位移不生效 | `_shift_positions` 读 engine 模块级 `ctx`，测试未注入 | 测试设 `engine.ctx = mgr` | G11 | 断言失败 | test_l2_invariants |
| A12 | 测试注意力参考 einsum 报错 | `einsum("skh,skh->kh")` 用 2D attn 当 3D；softmax 维度错 | 参考实现改为 `"sk,skh->kh"`，softmax 沿 S 维 | G24 | 测试报错 | test_l2_invariants |
| A13 | 模拟器 `index out of bounds` | fake 物理块 id（1000+）超出 fake 缓存张量尺寸（256 块） | fake id 用 `r*100+b` 落在尺寸内 | G24 | 模拟器报错 | simulate |
| A14 | 模拟器误删布局（`layout_dropped_recompute` 乱涨） | driver 不维护 fake 的 num_computed，recompute 检测误判 | driver 每步按 sample_tokens 时序更新 num_computed | G24 | 模拟器统计 | simulate |
| A15 | ubatch 场景错改/崩溃 | `attn_metadata` 是 `list[dict]` 时 `.items()` 失败 | `isinstance(list/tuple)` → 跳过该步（`skipped_ubatch`） | G13 | 源码审查 | test_failsoft |
| A16 | 空窗口 softmax 空张量 → NaN/崩溃 | `k_len − w < 1` 未守卫 | 返回全 1 分数（等价不剪枝） | G14 | 边界推演 | presses 单测 |
| A17 | n_kept 突然变 48（应 36），视图多读一个尾块 | 整理代码时把 `orig_len − b*bs` 写成 `orig_len − int(b)`（块下标当起始位置） | 改回 `orig_len − int(b)*bs` | G23 | **L2 不变量当场抓住**（`slot 0` 尾项） | test_l2_invariants |

## B. 真机反馈（NPU 日志驱动）

| # | 症状 | 根因 | 修复 | 类目 | 发现途径 | 回归测试 |
|---|---|---|---|---|---|---|
| B1 | 心跳永远 `FAIL=S6_compress_pass` | 逻辑 seam（压缩 pass 寄生 S5 钩子）从未 mark_installed；`seams=N/M` 混用命中数 | 逻辑 seam 随宿主钩子一起标记；口径改 installed + 独立 hit | G16 | 真机心跳 | test_activation_policy |
| B2 | 16×262144-token prompt、KV 91.5%、`completed=0` 永远、压缩从不触发 | 完成时触发的设计与资源在完成前耗尽 = 鸡生蛋（抢占循环） | 渐进式 mid-prefill 压缩（预算锚点推进 + 完成时重锚定） | G17 | 真机日志统计 | test_l2_mid_prefill |
| B3 | 改了源码重跑测试仍是旧行为（AttributeError 找不到新属性） | pip 安装的 `.pth` 在解释器启动时把旧包注册进 sys.modules | 开发循环先 `pip uninstall`；发布前重装并核验 `__file__` | G18 | 测试环境排查 | — |
| B4 | `can't convert npu:N device type tensor to numpy` | 设备张量直接 `.numpy()`/`.tolist()`（CPU mock 测不出） | `_as_numpy()` 统一 `detach().cpu().numpy()` | G19 | 真机 mid-prefill 报错 | _as_numpy 单测 |
| B5 | compact 元数据设备不匹配 | per-layer `seq_lens` 是 CPU 张量、delta 在 NPU，直接相减 | `delta_t.cpu()` 后再减 | G19 | 源码审查 | — |
| B6 | 请求已进 decode 但 `completed=0`（step9 prefilling=1 → step10 prefilling=0 无完成步） | 末块跨步时严格判定漏检（MTP async 计数口径） | 补检 `last_before < prompt <= before` + `_compressed_done` 去重 | G20 | 真机心跳序列 | test_l1_capture |
| B7 | mid-prefill 失败：`'NoneType' object has no attribute 'shape'` | 某层 `kv_cache=(None,None)`，`not kc` 拦不住元组元素 | 显式元素检查 + `skipped_no_kv` + **逐层 fail-soft**（坏一层不毁整请求） | G21 | 真机 mid-prefill 报错 | test_failsoft |
| B8 | AIV `IndexCheckKernel::CheckUpperBound` 断言（index 341748 > 923136） | 非法块 id 打进设备 gather，污染 NPU 流（try/except 救不回） | G11 守卫：行 id ∈ [0,num_cache_blocks)、槽 ∈ [0,num_blocks·bs)、保留块同界；`skipped_bad_row` 诊断 | G9 | 真机 AIV 断言 | test_failsoft |
| B10 | 每轮压缩都打 `layer mtp.layers.0.self_attn.attn scoring failed: 'NoneType' ... 'shape'` | step3.5 的 MTP draft 层混进 group-0 层列表且 kv_cache 未绑定；逐层 fail-soft 兜住但刷屏 | 结构性排除 draft 层（drafter.attn_layer_names + `.mtp.`/`.draft.` 启发式，`layers_excluded_draft` 计数）；kv_cache 缺失告警每层只报一次 | G25 | 真机日志 | test_failsoft |
| B12 | 压缩健康运行 ~400 步后 AIV 越界崩 worker（索引值跨 run 确定性重复 ~341k） | 疑似坏视图布局（保留块/视图长度越界）到达 FIA；旧代码只校验了保留块 id | 元数据改写**原子化**（快照原值、异常恢复，杜绝半改写状态）+ **全量自检**（m/klen≤max_blocks、view_len∈[1,true_len]），任一违规丢弃布局并 `skipped_bad_view` 诊断；坏视图**永远到不了 FIA** | G9/G21 延伸 | 真机 AIV | test_failsoft |
| B11 | `ArgSortKernelNpuOpApi.cpp` WARNING：kernel [ArgSort] 不支持 int32/int64，降级 AiCpu | 设备端 `topk(...).indices.sort()`（int64 索引） | 排序移到 numpy（`np.sort(_as_numpy(...))`），topk 留在设备端 | G26 | 真机启动/运行期 WARNING | 既有 L2 套件 |
| B9 | 让位路径打 `ERROR installed with FAILED seams`（还引用没打印的 summary） | apply() 把"主动让位"当"安装失败" | `DEFERRED_REASON` 标记 → INFO；真实失败才 summary+ERROR；ASCII 破折号 | G22 | 真机启动日志 | test_activation_policy |

## 统计

- 共 **29 条**（开发期 17 + 真机 12）；其中模拟测试发现 15、真机日志驱动 9、代码审查/推演 2。
- 类目分布：通用模式 G1–G24 覆盖 19 条；KV 压缩案例模式 K3–K7 覆盖 5 条。
- 每条真机 bug（B 组）都有对应回归测试入库；开发期 A 组 15 条有测试/模拟器断言覆盖。

## C. kvpress-ascend / SqueezeAttention-ascend 适配新增（2025-08 第二期）

| # | 症状 | 根因 | 修复 | 类目 | 发现途径 | 回归测试 |
|---|---|---|---|---|---|---|
| C1 | 视图行内容错乱（多请求时保留块被读成物理块 id 错位） | 视图 buffer 行写入的是**逻辑块下标**（kept 数组值），FIA 需要**物理块 id**（`row[kept]`） | buffer 填充改为 `row[kept]`（同步时从真行取值） | G23 变体 | **L2 端到端不变量**（view slots == 参考槽集合） | test_l2_invariant（两包） |
| C2 | 窗口滑动后视图读错块（append 分支在 recent_first 前移时执行） | squeeze `_sync_window_row` 的 append-only 分支没校验 `recent_first` 未变 | 分支条件改为 `recent_first == marker.recent_first` 才 append；前移走全量重装 | G23 变体 | L2 不变量跨块滑动 | test_l2_window_invariant |
| C3 | 层 hook 全部拿到最后一层的 layer_name/forward（cos-sim 全错） | 循环内闭包**晚绑定**（layer_name/orig/residual_pos 捕获变量） | 默认参数绑定 `_layer=_layer` 等 | G2 变体 | stats 层数/均值断言 | test_layer_hook_measures |
| C4 | 双包同开时所有权竞态（谁先装取决于对方策略检查是否 eager import） | `.pth` 启动期互相 find_spec+import，`_APPLIED` 检查时机不定 | 共享 `KV_ASCEND_OWNER` 进程标记 + 策略只读 env/sys.modules，**不跨包 import** | G11 变体 | 安装链路双包测试 | test_engine_defers_* |
| C5 | 测试进程内 re-import 包导致 registry 双身份（计数器全空、断言假败） | gate-off 测试 pop sys.modules 后重 import，harness 引用旧模块对象 | gate-off 测试改子进程验证；测试内禁止 re-import 包 | G11/G24 | 全量跑批 | test_gate_off_zero_imports |
| C6 | `kv_heads=0` → gather reshape 崩溃（snapkv 全层跳过） | `_resolve_target_layers` 只解析了 block_size，未解析 num_kv_heads/head_size | 缓存张量 shape 为地面真值（`key_cache.shape[2]`）；spec 属性兜底 | G2 变体 | simulate CLI | test_l2（snapkv） |
| C7 | 层重要性方向断言反了 | cos-sim 低 = 表示变化大 = 更重要（上游 class3 = 最高 cos 拿最小预算） | 语义核对上游代码后修正断言与注释 | 语义 | 上游对照 | test_squeeze_mode_clustering |
| C8 | 首层 cos-sim 永远捕获不到（residual=None） | 首层 residual 为 None，hook 只查 residual 参数 | residual 缺失时回退 hidden_states（首层输入即 residual） | G14 变体 | stats 键数断言 | test_layer_hook_measures |
