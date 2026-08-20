# TODO.md — kvpress-ascend / SqueezeAttention-ascend 施工清单

> 状态：`[ ]` 待办 / `[x]` 完成。最终状态：**阶段 A–D 全部完成**；模拟调试 DoD 达成（L0/L1/L2 全绿 46 tests + 自检 CLI + 心跳单测 + fail-soft 注入 + RISK_REGISTER）。

## 阶段 A：规划（已完成）
- [x] A1 读技能 seam map（vllm-ascend v0.23.0 / vllm 0.23.0）
- [x] A2 读 vllm-ascend：`model_runner_v1.py`（execute_model/_prepare_inputs/_build_attention_metadata/块表/positions/seq_lens）、`attention_v1.py`（AscendMetadata/Builder/Backend forward/_get_fia_params/reshape_and_cache/C8）、`block_table.py`、`ascend_forward_context.py`（_EXTRA_CTX）、`step3p5.py`（MTP draft 元数据流）、`__init__.py`（插件入口）
- [x] A3 读 kvpress：`pipeline.py`/`base_press.py`/`scorer_press.py`/`utils.py`/`attention_patch.py` + 代表性 presses
- [x] A4 读 SqueezeAttention：`utils_hh/modify_llama_drop.py`（LlamaModel_squeeze/LlamaDecoderLayer_squeeze）、`modify_mistral.py`、`pred.py`（ini_size/KV_class3 语义）
- [x] A5 机制转换设计 → PLAN.md（视图重写机制、seam 表、时序、心跳、风险）

## 阶段 B：kvpress-ascend 包
- [x] B1 骨架：`pyproject.toml`（hatchling + .pth force-include）、`kvpress_ascend.pth`、`__init__.py`（env 门控 + apply()）、`envs.py`、`log.py`
- [x] B2 `registry.py`：seam 探针注册 + 统计计数器 + 心跳行格式化
- [x] B3 `kvcore.py`（L0 纯逻辑）：视图行构造（kept-blocks + 真行尾部）、seq-lens 三元组修正、keep 数学（head 统一 topk）、slack/边界不变量
- [x] B4 `presses.py`：BasePress/ScorerPress 移植 + Knorm/Random/StreamingLLM/SnapKV/TOVA/PyramidKV/ExpectedAttention/CriticalKV（head 统一保留集）
- [x] B5 `capture.py`：per-request query/hidden 滚动捕获、逐层流式 gather+打分 pass、prefill 完成判定（before+scheduled）
- [x] B6 `engine.py`：S1 后端 forward 捕获、S2 Attention.forward 捕获、S3 _prepare_inputs 上下文、S4 _build_attention_metadata 视图重写、S5 execute_model pre/post、S7 槽映射位移（compact）、fail-soft 全套 + 导入环 defuse
- [x] B7 L0/L1/L2 测试（30 tests）+ `simulate.py` 自检 CLI
- [x] B8 `README.md` + `RISK_REGISTER.md`

## 阶段 C：SqueezeAttention-ascend 包
- [x] C1 骨架（同构 B1）
- [x] C2 `kvcore.py`：窗口公式（ini_size/percent/start_size）、KMeans-1D（自包含）、视图行（sink+recent 块区间）、seq-lens 修正
- [x] C3 `capture.py`：层重要性（cos 相似度）滚动累积 + 聚类 pass + 完成判定
- [x] C4 `engine.py`：S6 层 forward 包装、S3 注意力输出捕获、S4 视图重写、S1/S5 pre/post、fail-soft
- [x] C5 L0/L1/L2 测试（16 tests）+ `simulate.py`
- [x] C6 `README.md` + `RISK_REGISTER.md`

## 阶段 D：联调与模拟 debug
- [x] D1 心跳日志单测：seam 探针 hit/FAIL 与核心参数输出（test_heartbeat.py）
- [x] D2 端到端不变量：多步 decode 越过块边界（L2），`attention(视图) == attention(参考)` 误差 < 1e-4（view/compact 双模式）
- [x] D3 故意失败注入：mock 缺字段/env 未设 → fail-soft 降级路径验证（test_failsoft.py：缺 kv_cache / 缺 static_forward_context / 坏块行 / 形状不匹配 / ubatch）
- [x] D4 全量测试 + 自检 CLI 绿：**46 passed**；`python -m kvpress_ascend.simulate` / `python -m squeeze_ascend.simulate` OK
- [x] D5 真机核对清单（README 附录）：精度对比/前缀缓存对照/MTP 接受率/长跑稳定性/性能基线

## 附：本机验证记录（模拟 debug，无 NPU）
- [x] pip 安装验证：两包 `pip install ./...` 成功，`.pth` 落 site-packages（wheel 内容核验）
- [x] env 门控验证：未 export 时零 torch 导入；`export kvpress=1` 且无 vllm_ascend 时 fail-soft 降级不崩溃
- [x] 模拟器全模式跑通：view/compact/window 场景 + 心跳样例输出
