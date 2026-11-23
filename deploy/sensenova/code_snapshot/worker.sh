#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
mkdir -p "$RUN_DIR/evidence"
exec > >(tee -a "$RUN_DIR/evidence/supervisor.log") 2>&1
exec "$PYTHON_BIN" -B -u "$RUN_DIR/code_snapshot/supervisor.py"
