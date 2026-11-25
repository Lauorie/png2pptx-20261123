#!/usr/bin/env bash
# Convert every PNG in one directory into an editable PPTX.
#
#   run_job.sh IMAGES_DIR RUN_DIR
#
# RUN_DIR must not exist yet; results land in RUN_DIR/delivery/<image name>/.
# IMAGES_DIR must stay in place until the job ends (images are linked, not copied),
# and both directories must be visible to the SAM service under the same paths.
#
# Required environment:
#   PNG2PPTX_API_BASE_URL    OpenAI-compatible chat completions base URL
#   PNG2PPTX_API_KEY_FILE    file holding the API key (mode 0600)
#   PNG2PPTX_SAM_URL         SAM3 service, e.g. http://127.0.0.1:18103
#   PNG2PPTX_IMAGE_EDIT_URL  SenseNova service, e.g. http://127.0.0.1:8091/v1
# Optional environment:
#   PNG2PPTX_API_MODEL         default z-ai/glm-5.3-flash
#   PNG2PPTX_REASONING_EFFORT  low | medium | high | none; default high
#   PNG2PPTX_CPU_ENV           Python environment prefix; default /opt/shape-cpu
set -euo pipefail

if [ "$#" -ne 2 ]; then
  sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//' >&2
  exit 2
fi
: "${PNG2PPTX_API_BASE_URL:?set PNG2PPTX_API_BASE_URL}"
: "${PNG2PPTX_API_KEY_FILE:?set PNG2PPTX_API_KEY_FILE}"
: "${PNG2PPTX_SAM_URL:?set PNG2PPTX_SAM_URL}"
: "${PNG2PPTX_IMAGE_EDIT_URL:?set PNG2PPTX_IMAGE_EDIT_URL}"

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
IMAGES=$(realpath "$1")
RUN_DIR=$(realpath -m "$2")
CATALOG="${RUN_DIR}.input"
export PNG2PPTX_CPU_ENV="${PNG2PPTX_CPU_ENV:-/opt/shape-cpu}"
export PATH="${PNG2PPTX_CPU_ENV}/bin:${PATH}"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}"
PYTHON="${PNG2PPTX_CPU_ENV}/bin/python"
EFFORT="${PNG2PPTX_REASONING_EFFORT:-high}"

if [ -e "$RUN_DIR" ] || [ -e "$CATALOG" ]; then
  echo "run directory or its .input catalog already exists: $RUN_DIR" >&2
  exit 2
fi

"$PYTHON" -B "$REPO/scripts/make_input_catalog.py" --images "$IMAGES" --output "$CATALOG"

PREPARE=("$PYTHON" -B "$REPO/scripts/prepare_forward.py"
  --input-root "$CATALOG" --run-dir "$RUN_DIR"
  --api-base-url "$PNG2PPTX_API_BASE_URL"
  --api-model "${PNG2PPTX_API_MODEL:-z-ai/glm-5.3-flash}"
  --api-key-file "$PNG2PPTX_API_KEY_FILE"
  --sam-url "$PNG2PPTX_SAM_URL"
  --image-edit-url "$PNG2PPTX_IMAGE_EDIT_URL")
if [ "$EFFORT" != "none" ]; then
  PREPARE+=(--force-reasoning-effort "$EFFORT")
fi
"${PREPARE[@]}"

cd "$RUN_DIR"
# The worker records failures in status.json; its exit code alone is not the verdict.
"$PYTHON" -B -u "$RUN_DIR/code/full_recovery/api_worker.py" --run-dir "$RUN_DIR" || true

STATE=$("$PYTHON" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "$RUN_DIR/status.json")
echo "state: $STATE  delivery: $RUN_DIR/delivery"
[ "$STATE" = "completed" ]
