#!/usr/bin/env bash
# Run the chat-completions -> messages adapter in the foreground (port 18200).
# Only needed when the model endpoint serves the model through /messages.
#
# Required environment:
#   PNG2PPTX_UPSTREAM_BASE_URL  e.g. https://host/api/v1
# Optional environment:
#   PNG2PPTX_CPU_ENV            Python environment prefix; default /opt/shape-cpu
set -euo pipefail
: "${PNG2PPTX_UPSTREAM_BASE_URL:?set PNG2PPTX_UPSTREAM_BASE_URL}"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
exec "${PNG2PPTX_CPU_ENV:-/opt/shape-cpu}/bin/python" -B -m chat_to_messages_adapter \
  --upstream-base-url "$PNG2PPTX_UPSTREAM_BASE_URL" --host 127.0.0.1 --port "${PNG2PPTX_ADAPTER_PORT:-18200}"
