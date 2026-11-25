#!/usr/bin/env bash
# Run the SAM3 service in the foreground (port 18103).
#
#   start_sam.sh RUN_DIR
#
# RUN_DIR must not exist yet. Requests carry file paths, so the service must see
# the worker's run directories and input images under the same paths.
#
# Required environment:
#   PNG2PPTX_SAM_PYTHON  interpreter of the SAM3 environment
# Optional environment:
#   CUDA_VISIBLE_DEVICES  GPU to use; the service needs exactly one
#   POD_IP                address written to RUN_DIR/endpoint.json; default 127.0.0.1
set -euo pipefail

if [ "$#" -ne 1 ]; then
  sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//' >&2
  exit 2
fi
: "${PNG2PPTX_SAM_PYTHON:?set PNG2PPTX_SAM_PYTHON}"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_DIR=$(realpath -m "$1")

"$PNG2PPTX_SAM_PYTHON" -B "$REPO/deploy/prepare.py" sam --run-dir "$RUN_DIR" > /dev/null
# The cluster configuration starts an artificial GPU load when the service is
# idle, to keep its allocation. Outside that cluster the load is only a cost.
"$PNG2PPTX_SAM_PYTHON" -I - "$RUN_DIR/config.json" <<'PY'
import json, sys
path = sys.argv[1]
config = json.load(open(path))
config["idle_resume_seconds"] = 10**9
json.dump(config, open(path, "w"), indent=2)
PY
export POD_IP="${POD_IP:-127.0.0.1}"
exec "$PNG2PPTX_SAM_PYTHON" -B -u "$RUN_DIR/code/worker.py" --run-dir "$RUN_DIR" --power-pct 60
