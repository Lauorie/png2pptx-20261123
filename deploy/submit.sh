#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="$(realpath "${1:?usage: deploy/submit.sh DEPLOYMENT_DIR [--dry-run]}")"
SERVICE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["service"])' "$RUN_DIR/config.json")"
IMAGE="${RJOB_IMAGE:-registry.h.pjlab.org.cn/ailab-intern7-intern7_gpu/tianchangyao-workspace@sha256:da09d3a89a5370ea386ea25e97557521ebcaea1eb6eac27f5b5bf685621abd7b}"
GPU_COUNT=1
CPU_COUNT=8
MEMORY_MB=49152
if [[ "$SERVICE" == sensenova ]]; then
  GPU_COUNT=2
  CPU_COUNT=16
  MEMORY_MB=131072
fi
ARGS=(submit --name="png2pptx-$SERVICE-$(date +%Y%m%d-%H%M%S)"
  --namespace ailab-llmmultimodal --charged-group llmmultimodal_gpu_pool
  --gpu="$GPU_COUNT" --cpu="$CPU_COUNT" --memory="$MEMORY_MB"
  --private-machine=group -P 1 --share-host-shm=true --gang-start=true
  --image "$IMAGE"
  --mount=gpfs://gpfs1/intern7shared:/mnt/shared-storage-user/intern7shared)
if [[ "$SERVICE" == sam ]]; then
  ARGS+=(--mount="${SAM_ENV_MOUNT:-gpfs://gpfs1/dongdaozhu/conda/envs/sam3}:/data/dongdaozhu/conda/envs/sam3")
  printf -v REMOTE '%q ' /data/dongdaozhu/conda/envs/sam3/bin/python -B -u "$RUN_DIR/code/worker.py" --run-dir "$RUN_DIR" --power-pct 60
elif [[ "$SERVICE" == sensenova ]]; then
  BASE_ENV="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_environment"])' "$RUN_DIR/config.json")"
  ARGS+=(--mount="${SENSENOVA_BASE_ENV_MOUNT:-gpfs://gpfs1/yinjinhui/H-conda/anaconda3/envs/openrlhf}:$BASE_ENV"
         --mount="${CUDA_MOUNT:-gpfs://gpfs1/intern7shared/yinjinhui/cuda/cuda_installed/cuda128}:/usr/local/cuda")
  printf -v REMOTE '%q ' bash "$RUN_DIR/code_snapshot/worker.sh"
else
  echo "Unknown service: $SERVICE" >&2
  exit 2
fi
if [[ "${2:-}" == --dry-run ]]; then
  printf '%q ' rjob "${ARGS[@]}" -- bash -c "$REMOTE"
  printf '\n'
  exit 0
fi
if [[ "$SERVICE" == sensenova ]]; then
  RUNTIME_PYTHON="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime_python"])' "$RUN_DIR/config.json")"
  if [[ ! -x "$RUNTIME_PYTHON" ]]; then
    echo "SenseNova runtime missing; run deploy/sensenova/setup_runtime.py first: $RUNTIME_PYTHON" >&2
    exit 2
  fi
fi
exec 9>"$RUN_DIR/jobs/submit.lock"
flock -n 9
test ! -e "$RUN_DIR/jobs/job.id"
rjob "${ARGS[@]}" -- bash -c "$REMOTE" 2>&1 | tee "$RUN_DIR/jobs/submit.log"
python3 - "$RUN_DIR" <<'PY'
import re,sys
from pathlib import Path
run=Path(sys.argv[1])
match=re.search(r'created[ _]rjob_name["\x27]?\s*[:=]\s*["\x27]?([A-Za-z0-9_.-]+)',(run/"jobs/submit.log").read_text())
if not match:
    raise SystemExit("No metadata job ID; inspect submit.log before retrying")
(run/"jobs/job.id").write_text(match[1]+"\n")
print("metadata_job_id="+match[1])
PY
rjob get --namespace ailab-llmmultimodal "$(cat "$RUN_DIR/jobs/job.id")"
