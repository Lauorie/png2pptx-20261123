#!/usr/bin/env bash
# (Re)start the single-GPU SenseNova service; extra env via "KEY=VAL ..." args.
# A crashed LightLLM leaves helper processes holding a 50 GB SysV segment; a second start on
# top of that exceeds the container memory limit, so stop everything and free segments first.
export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/png2pptx-20261123
L=/root/autodl-tmp/verify_logs; R=runtime_runs/sensenova_verify_20261123; OUT="$L/${SN_LOG:-sensenova_service.out}"
pgrep -f 'lightllm[.]server[.]api_server' | xargs -r kill; sleep 3
for round in 1 2 3; do
  pgrep -f 'lightllm[:][:]|[.]runtime/sensenova/bin/python' | xargs -r kill -9; sleep 2
done
for id in $(ipcs -m | awk 'NR>3 && $6==0 {print $2}'); do ipcrm -m "$id"; done
ipcs -m | awk 'NR>3 && NF>=6 {n++} END {print "shm segments left:", n+0}'
env CUDA_VISIBLE_DEVICES=0 "$@" setsid nohup bash "$R/code_snapshot/service.sh" > "$OUT" 2>&1 < /dev/null &
echo started "$OUT"
