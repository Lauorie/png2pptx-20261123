#!/usr/bin/env bash
# Run the SenseNova image-edit service in the foreground on one GPU (port 8091).
#
#   start_sensenova.sh RUN_DIR
#
# RUN_DIR must not exist yet. Loading takes about seven minutes. The service
# needs python3 on PATH and the CUDA toolkit (nvcc) under CUDA_HOME.
#
# Required environment:
#   PNG2PPTX_SENSENOVA_BASE_ENV  prefix of the base environment (ENV_ROOT/openrlhf)
# Optional environment:
#   PNG2PPTX_SENSENOVA_PYTHON    default <repo>/.runtime/sensenova/bin/python
#   CUDA_VISIBLE_DEVICES         default 0
#   CUDA_HOME                    default /usr/local/cuda
set -euo pipefail

if [ "$#" -ne 1 ]; then
  sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//' >&2
  exit 2
fi
: "${PNG2PPTX_SENSENOVA_BASE_ENV:?set PNG2PPTX_SENSENOVA_BASE_ENV}"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_DIR=$(realpath -m "$1")
PYTHON="${PNG2PPTX_SENSENOVA_PYTHON:-$REPO/.runtime/sensenova/bin/python}"

# A crashed or half-started LightLLM leaves helper processes holding GPU memory
# and a ~50 GB SysV segment; a second start on top of them exceeds the memory
# limit and its loader is killed without any message. Some helpers keep only
# the interpreter path in their command line, so match that as well.
pgrep -f 'lightllm[.]server[.]api_server' | xargs -r kill || true
sleep 3
for _ in 1 2 3; do
  { pgrep -f 'lightllm[:][:]' || true; pgrep -f "^$PYTHON " || true; pgrep -f 'code_snapshot/service[.]sh' || true; } \
    | sort -u | xargs -r kill -9 || true
  sleep 2
done
for id in $(ipcs -m | awk 'NR>3 && $6==0 {print $2}'); do
  ipcrm -m "$id" || true
done

"$PYTHON" -B "$REPO/deploy/prepare.py" sensenova --run-dir "$RUN_DIR" \
  --sensenova-python "$PYTHON" --base-environment "$PNG2PPTX_SENSENOVA_BASE_ENV" > /dev/null
# The bundled launcher splits understanding and generation over two GPUs and
# starts an artificial idle load; this deployment uses one GPU and no idle load.
sed -i 's/--x2i_server_deploy_mode separate/--x2i_server_deploy_mode colocate/' "$RUN_DIR/code_snapshot/service.sh"
grep -q -- '--x2i_server_deploy_mode colocate' "$RUN_DIR/code_snapshot/service.sh"
"$PYTHON" -I - "$RUN_DIR/config.json" <<'PY'
import json, sys
path = sys.argv[1]
config = json.load(open(path))
config.update(gpu_count=1, idle_resume_seconds=10**9)
json.dump(config, open(path, "w"), indent=2)
PY
# The launcher reads its configuration with "python3".
export PATH="$(dirname "$PYTHON"):$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec bash "$RUN_DIR/code_snapshot/service.sh"
