#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="$(realpath "${1:?prepared complete-PPT API run required}")"
exec 9>"${RUN_DIR}/jobs/.submit.lock"
flock -n 9
test ! -e "${RUN_DIR}/jobs/job.id"
JOB_NAME="${JOB_NAME_PREFIX:-pptx-full-api}-$(date +%Y%m%d-%H%M%S)"
IMAGE="${RJOB_IMAGE:-registry.h.pjlab.org.cn/ailab-intern7-intern7_gpu/tianchangyao-workspace:v6}"
RJOB_ARGS=(submit --name="${JOB_NAME}" --namespace ailab-intern10
  --gpu=0 --cpu="${CPU_COUNT:-16}" --memory="${MEMORY_MB:-65536}" --charged-group=intern10_proxy
  --private-machine=group -P 1 --share-host-shm=true --gang-start=true
  --image "${IMAGE}"
  --mount=gpfs://gpfs1/intern7shared:/mnt/shared-storage-user/intern7shared
  --mount="${CPU_ENV_MOUNT:-gpfs://gpfs1/kouyiqun/conda/envs/SVG}:/opt/shape-cpu")
ENV_ARGS=()
for name in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
  [[ -n "${!name:-}" ]] && ENV_ARGS+=(-e "${name}=${!name}")
done
printf -v REMOTE_COMMAND '%q ' env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt /opt/shape-cpu/bin/python -B -u "${RUN_DIR}/code/full_recovery/api_worker.py" --run-dir "${RUN_DIR}"
rjob "${RJOB_ARGS[@]}" "${ENV_ARGS[@]}" -- bash -lc "${REMOTE_COMMAND}" 2>&1 | tee "${RUN_DIR}/jobs/submit.log"
python3 - "${RUN_DIR}" <<'PY'
from pathlib import Path
import re,sys
run=Path(sys.argv[1]);text=(run/'jobs/submit.log').read_text()
m=re.search(r'created[ _]rjob_name["\x27]?\s*[:=]\s*["\x27]?([A-Za-z0-9_.-]+)',text)
if not m:raise SystemExit('Missing metadata job ID; inspect before resubmission')
(run/'jobs/job.id').write_text(m[1]+'\n');print('metadata_job_id='+m[1])
PY
