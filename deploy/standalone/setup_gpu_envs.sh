#!/usr/bin/env bash
# Build the Python environments of the two GPU services.
#
#   setup_gpu_envs.sh ENV_ROOT
#
# Creates ENV_ROOT/sam3 and ENV_ROOT/openrlhf, then the SenseNova runtime at
# <repo>/.runtime/sensenova, which links into ENV_ROOT/openrlhf and therefore
# needs that directory to stay in place.
#
# Optional environment:
#   PNG2PPTX_BASE_PYTHON  a CPython 3.12 interpreter; default python3.12
set -euo pipefail

if [ "$#" -ne 1 ]; then
  sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//' >&2
  exit 2
fi
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
ENV_ROOT=$(realpath -m "$1")
BASE_PYTHON="${PNG2PPTX_BASE_PYTHON:-python3.12}"

# A marker is written only after a complete install, so an interrupted run is redone.
if [ ! -e "$ENV_ROOT/sam3/.png2pptx-ready" ]; then
  rm -rf "$ENV_ROOT/sam3"
  "$BASE_PYTHON" -m venv "$ENV_ROOT/sam3"
  "$ENV_ROOT/sam3/bin/pip" install -r "$HERE/requirements.sam3.txt"
  touch "$ENV_ROOT/sam3/.png2pptx-ready"
fi
if [ ! -e "$ENV_ROOT/openrlhf/.png2pptx-ready" ]; then
  rm -rf "$ENV_ROOT/openrlhf"
  "$BASE_PYTHON" -m venv "$ENV_ROOT/openrlhf"
  # The recorded versions do not satisfy each other's declared requirements
  # (vllm 0.16.0 asks for transformers<5, the service ran with 5.2.0), so every
  # package is pinned in the file and installed without dependency resolution.
  "$ENV_ROOT/openrlhf/bin/pip" install --no-deps -r "$HERE/requirements.sensenova-base.txt"
  touch "$ENV_ROOT/openrlhf/.png2pptx-ready"
fi
if [ ! -d "$REPO/deploy/sensenova/runtime_overlay" ]; then
  echo "deploy/sensenova/runtime_overlay is absent; it ships outside git with the other large files" >&2
  exit 1
fi
# handoff_runtime.json is the last file setup_runtime.py writes.
if [ ! -e "$REPO/.runtime/sensenova/handoff_runtime.json" ]; then
  rm -rf "$REPO/.runtime/sensenova"
  "$ENV_ROOT/openrlhf/bin/python" "$REPO/deploy/sensenova/setup_runtime.py" \
    --base-python "$ENV_ROOT/openrlhf/bin/python" --output "$REPO/.runtime/sensenova"
fi
echo "sam3 python:      $ENV_ROOT/sam3/bin/python"
echo "sensenova python: $REPO/.runtime/sensenova/bin/python"
echo "sensenova base:   $ENV_ROOT/openrlhf"
