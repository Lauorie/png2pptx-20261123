#!/usr/bin/env bash
# Start the GLM adapter and the SAM3 service on this instance (idempotent for the adapter).
export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/png2pptx-20261123
L=/root/autodl-tmp/verify_logs
pgrep -f 'chat_to_messages[_]adapter' | xargs -r kill; sleep 1
setsid nohup .venv/bin/python -B -m chat_to_messages_adapter --upstream-base-url "${UPSTREAM_BASE_URL:?set the gateway base URL}" --port 18200 > $L/chat_to_messages_adapter.log 2>&1 < /dev/null &
if ! curl -s --noproxy '*' -m 3 -o /dev/null http://127.0.0.1:18103/health; then
  R=runtime_runs/sam_$(date +%Y%m%d_%H%M%S)
  python deploy/prepare.py sam --run-dir $R > /dev/null
  # Rented instance: no reclaim-avoidance needed, so keep the idle GPU load from ever starting.
  python - "$R/config.json" <<'PY'
import json, sys
p = sys.argv[1]; c = json.load(open(p)); c["idle_resume_seconds"] = 10**9; json.dump(c, open(p, "w"), indent=2)
PY
  POD_IP=127.0.0.1 setsid nohup /root/autodl-tmp/envs/sam3/bin/python -B -u $R/code/worker.py --run-dir $R --power-pct 60 > $L/sam_worker.log 2>&1 < /dev/null &
  echo "sam run dir: $R"
fi
