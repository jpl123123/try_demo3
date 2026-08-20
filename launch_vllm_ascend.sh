#!/usr/bin/env bash
# 一键拉起 vllm-ascend + kvpress-ascend / SqueezeAttention-ascend（monkeypatch 适配器）
#
# 用法：
#   bash launch_vllm_ascend.sh
#
# 前提：已 pip install ./kvpress-ascend ./SqueezeAttention-ascend
# 说明：两个包同时 export 时默认 kvpress 生效（squeeze 让位）；
#       想换成 squeeze 生效：export SQUEEZE_ASCEND_POLICY=squeeze

set -e

# ---------- 开关：不 export 对应包 = 完全无感 ----------
export kvpress=1        # 开启 kvpress 压缩
export squeeze=1        # 开启 SqueezeAttention（同开时默认让位给 kvpress）

# ---------- 可选调优（按需取消注释） ----------
# export KVPRESS_ASCEND_PRESS=snapkv        # knorm/random/streamingllm/snapkv/tova/pyramidkv/...
# export KVPRESS_ASCEND_RATIO=0.5
# export KVPRESS_ASCEND_WINDOW=64
# export KVPRESS_ASCEND_SINK=4
# export KVPRESS_ASCEND_MODE=view           # view（默认）| compact
# export KVPRESS_ASCEND_STEP_LOG=1
# export KVPRESS_ASCEND_LOG=info
#
# ---- 262144 级长上下文必开：渐进式 mid-prefill 压缩 ----
# export KVPRESS_ASCEND_MID_PREFILL=1
# export KVPRESS_ASCEND_MID_PREFILL_BUDGET=65536     # 首个压缩锚点（每请求 token 数）
# export KVPRESS_ASCEND_MID_PREFILL_REFRESH=32768    # 锚点后每再增长多少 token 压一次
# export KVPRESS_ASCEND_PROGRESS_LOG=200             # 每 N 步打印 prefill 进度摘要（0=关）
#
# export SQUEEZE_ASCEND_INI_SIZE=0.3
# export SQUEEZE_ASCEND_CLASS3_RATIO=0.1
# export SQUEEZE_ASCEND_START_SIZE=4
# export SQUEEZE_ASCEND_STEP_LOG=1

exec vllm serve /softwarePlatform/c00879303/Qwen3.5-27B-w8a8-mtp \
  --served-model-name "qwen3.5" \
  --host 0.0.0.0 \
  --port 1144 \
  --data-parallel-size 1 \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.9 \
  --compilation-config '{"cudagraph_capture_sizes":[1,4,8,12,16,24,32,48,56,64,72,84,96,108,112,128,160,172,196,200,212,232,272,288,312,328,344,360,384,400,416,432,448,480,512], "cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
  --trust-remote-code \
  --async-scheduling \
  --allowed-local-media-path / \
  --quantization ascend \
  --enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --additional-config '{"enable_cpu_binding":true}' \
  --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
