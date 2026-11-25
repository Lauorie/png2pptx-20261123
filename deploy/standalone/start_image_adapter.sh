#!/usr/bin/env bash
# Run the images/edits -> chat completions adapter in the foreground (port 18300).
# It lets an image-output chat model (e.g. a Gemini image model) stand in for the
# SenseNova service; point PNG2PPTX_IMAGE_EDIT_URL at http://127.0.0.1:18300/v1.
#
# Required environment:
#   PNG2PPTX_IMAGE_UPSTREAM_BASE_URL  e.g. https://host/api/v1
#   PNG2PPTX_IMAGE_MODEL              upstream image model
#   PNG2PPTX_IMAGE_KEY_FILE           file holding the upstream key (mode 0600)
# Optional environment:
#   PNG2PPTX_IMAGE_USAGE_LOG          JSONL usage log; default ./image_usage.jsonl
#   PNG2PPTX_IMAGE_ADAPTER_PORT       default 18300
#   PNG2PPTX_CPU_ENV                  Python environment prefix; default /opt/shape-cpu
set -euo pipefail
: "${PNG2PPTX_IMAGE_UPSTREAM_BASE_URL:?set PNG2PPTX_IMAGE_UPSTREAM_BASE_URL}"
: "${PNG2PPTX_IMAGE_MODEL:?set PNG2PPTX_IMAGE_MODEL}"
: "${PNG2PPTX_IMAGE_KEY_FILE:?set PNG2PPTX_IMAGE_KEY_FILE}"
USAGE_LOG=$(realpath -m "${PNG2PPTX_IMAGE_USAGE_LOG:-image_usage.jsonl}")
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}"
exec "${PNG2PPTX_CPU_ENV:-/opt/shape-cpu}/bin/python" -B -m images_edits_adapter \
  --upstream-base-url "$PNG2PPTX_IMAGE_UPSTREAM_BASE_URL" --model "$PNG2PPTX_IMAGE_MODEL" \
  --api-key-file "$PNG2PPTX_IMAGE_KEY_FILE" --usage-log "$USAGE_LOG" \
  --host 127.0.0.1 --port "${PNG2PPTX_IMAGE_ADAPTER_PORT:-18300}"
