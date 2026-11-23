#!/usr/bin/env bash
RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
read_config() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$RUN_DIR/config.json" "$1"; }
export BUNDLE_ROOT="$(read_config bundle_root)"
PYTHON_BIN="${SENSENOVA_PYTHON:-$(read_config runtime_python)}"
BASE_ENV="${SENSENOVA_BASE_ENV:-$(read_config base_environment)}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$(dirname "$PYTHON_BIN"):$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$BASE_ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$RUN_DIR/code_snapshot:$RUN_DIR/vendor/LightX2V:$RUN_DIR/vendor/LightLLM"
export SENSENOVA_RUN_DIR="$RUN_DIR"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR="/tmp/png2pptx-sensenova-triton"
export TORCHINDUCTOR_CACHE_DIR="/tmp/png2pptx-sensenova-inductor"
export FLASHINFER_WORKSPACE_BASE="/tmp/png2pptx-sensenova-flashinfer"
export TVM_FFI_CACHE_DIR="/tmp/png2pptx-sensenova-tvm-ffi"
export TORCH_EXTENSIONS_DIR="/tmp/png2pptx-sensenova-torch-extensions"
export XDG_CACHE_HOME="/tmp/png2pptx-sensenova-cache"
export MPLCONFIGDIR="/tmp/png2pptx-sensenova-matplotlib"
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
MODEL_DIR="$BUNDLE_ROOT/weights/sensenova/SenseNova-U1.5-8B-MoT"
GEN_CONFIG="$RUN_DIR/code_snapshot/neopp_dense_parallel_cfg.json"
test -x "$PYTHON_BIN"
test -x "$CUDA_HOME/bin/nvcc"
test -f "$MODEL_DIR/model.safetensors.index.json"
