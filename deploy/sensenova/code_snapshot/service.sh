#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
mkdir -p "$RUN_DIR/evidence"
exec > >(tee -a "$RUN_DIR/evidence/service.log") 2>&1
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
cd "$RUN_DIR/vendor/LightLLM"
exec "$PYTHON_BIN" -m lightllm.server.api_server \
  --model_dir "$MODEL_DIR" --enable_multimodal_x2i \
  --x2i_server_deploy_mode separate --x2i_server_used_gpus 1 \
  --x2v_gen_model_config "$GEN_CONFIG" --host 0.0.0.0 --port 8091 \
  --max_req_total_len 65536 --max_total_token_num 69632 --tp 1 --graph_max_batch_size 16 \
  --trust_remote_code --disable_log_stats
