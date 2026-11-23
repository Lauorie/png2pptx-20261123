#!/usr/bin/env python3
"""Run GLM-5.3-Flash from the image's own Python and verify its API."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-dir", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--tp", type=int, default=4)
parser.add_argument("--dp", type=int, default=2)
parser.add_argument("--port", type=int, default=10088)
parser.add_argument("--max-model-len", type=int, default=262144)
parser.add_argument("--max-num-seqs", type=int, default=48)
parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
parser.add_argument("--vllm-bin", default="/usr/local/bin/vllm")
parser.add_argument("--startup-timeout", type=int, default=3600)
parser.add_argument("--dry-run", action="store_true",
                    help="Print launch argv/environment without importing torch or starting a service")
args = parser.parse_args()
if args.tp <= 0 or args.dp <= 0 or args.tp * args.dp not in (4, 8):
    parser.error("This single-node launcher requires positive TP * DP = 4 or 8")
if not 0 < args.gpu_memory_utilization <= 1:
    parser.error("gpu-memory-utilization must be in (0, 1]")

# Keep the shared serving settings for either 4 or 8 GPUs; dry-run is CPU-only.
runtime_env = {
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES") or ",".join(str(i) for i in range(args.tp * args.dp)),
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "VLLM_USE_BREAKABLE_CUDAGRAPH": "1",
    "VLLM_ENGINE_READY_TIMEOUT_S": str(args.startup_timeout),
}
cuda_root = Path(os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", "/usr/local/cuda")))
if ("VLLM_USE_FLASHINFER_SAMPLER" not in os.environ
        and shutil.which("nvcc") is None and not (cuda_root / "bin/nvcc").is_file()):
    runtime_env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
elif "VLLM_USE_FLASHINFER_SAMPLER" in os.environ:
    runtime_env["VLLM_USE_FLASHINFER_SAMPLER"] = os.environ["VLLM_USE_FLASHINFER_SAMPLER"]
command = [
    args.vllm_bin, "serve", args.model,
    "--tensor-parallel-size", str(args.tp),
    "--data-parallel-size", str(args.dp),
    "--data-parallel-size-local", str(args.dp),
    "--data-parallel-backend", "mp",
    "--distributed-executor-backend", "mp",
    "--dtype", "bfloat16", "--kv-cache-dtype", "auto",
    "--max-model-len", str(args.max_model_len),
    "--max-num-seqs", str(args.max_num_seqs),
    "--max-num-batched-tokens", str(args.max_num_batched_tokens),
    "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    "--enable-prefix-caching", "--enable-chunked-prefill", "--async-scheduling",
    "--compilation-config", '{"cudagraph_mode":"FULL_AND_PIECEWISE"}',
    "--no-enable-flashinfer-autotune", "--safetensors-load-strategy", "prefetch",
    "--tool-call-parser", "glm47", "--reasoning-parser", "glm45",
    "--enable-auto-tool-choice", "--host", "0.0.0.0", "--port", str(args.port),
    "--served-model-name", "GLM-5.3-Flash",
]
if args.dry_run:
    print(json.dumps({"argv": command, "env": runtime_env}, ensure_ascii=False, indent=2))
    raise SystemExit(0)
os.environ.update(runtime_env)
out = Path(args.run_dir)
out.mkdir(parents=True, exist_ok=True)
state = {"phase": "initializing", "model": args.model, "port": args.port,
         "served_model_name": "GLM-5.3-Flash", "pid": os.getpid(),
         "tp": args.tp, "dp": args.dp}
server = None

def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def write_json(name, data):
    target = out / name
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temp.replace(target)

def update(phase, **fields):
    state.update(phase=phase, updated_at=stamp(), **fields)
    write_json("service-state.json", state)
    print(json.dumps(state, ensure_ascii=False), flush=True)

def stop(signum=None, frame=None):
    if server is not None and server.poll() is None:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait(timeout=15)
    if signum is not None:
        update("stopped", signal=signum)
        raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)

# Inspect the allocated runtime only. The user has confirmed the model snapshot.
import torch
environment = {
    "started_at": stamp(), "hostname": socket.gethostname(),
    "python": sys.executable, "python_version": sys.version,
    "packages": {name: importlib.metadata.version(name)
                 for name in ["vllm", "torch", "transformers", "flashinfer-python"]},
    "gpu_count": torch.cuda.device_count(),
    "gpus": [{"index": i, "name": torch.cuda.get_device_name(i),
              "capability": torch.cuda.get_device_capability(i),
              "memory_bytes": torch.cuda.get_device_properties(i).total_memory}
             for i in range(torch.cuda.device_count())],
}
write_json("environment.json", environment)
print(json.dumps(environment, ensure_ascii=False), flush=True)
if environment["gpu_count"] != args.tp * args.dp:
    update("failed", reason="GPU count does not match TP * DP")
    raise SystemExit(2)
if any(gpu["capability"][0] < 9 for gpu in environment["gpus"]):
    update("failed", reason="The GLM-5.3-Flash NVIDIA implementation requires Hopper or newer")
    raise SystemExit(2)
smi = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv"],
    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
(out / "gpus.csv").write_text(smi.stdout)

write_json("vllm-argv.json", command)
write_json("vllm-env.json", runtime_env)
(out / "vllm-command.sh").write_text(
    shlex.join(["env", *(f"{key}={value}" for key, value in runtime_env.items()), *command]) + "\n")
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
base = f"http://127.0.0.1:{args.port}"

with (out / "vllm.log").open("w", buffering=1) as log:
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True)
    update("loading", server_pid=server.pid)
    deadline = time.monotonic() + args.startup_timeout
    while server.poll() is None and time.monotonic() < deadline:
        try:
            with opener.open(base + "/health", timeout=5) as response:
                healthy = response.status == 200
            if healthy:
                break
        except Exception:
            pass
        time.sleep(5)
    else:
        update("failed", reason="server exited or startup timed out",
               exit_code=server.poll())
        stop()
        raise SystemExit(1)

    with opener.open(base + "/v1/models", timeout=15) as response:
        models = json.load(response)
    write_json("models.json", models)
    update("healthy", health_http_status=200)
    request_body = {
        "model": "GLM-5.3-Flash",
        "messages": [{"role": "user", "content": "1+1等于多少？最终答案只写一个数字。"}],
        "temperature": 0, "max_tokens": 512,
        "chat_template_kwargs": {"reasoning_effort": "low", "clear_thinking": True},
    }
    write_json("smoke-request.json", request_body)
    try:
        request = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps(request_body, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"})
        with opener.open(request, timeout=180) as response:
            result = json.load(response)
        write_json("smoke-response.json", result)
        message = result["choices"][0]["message"]
        answer = message.get("content")
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("No non-empty final answer returned")
        update("ready", smoke_http_status=200, smoke_answer=answer)
    except Exception as error:
        update("healthy_smoke_failed", smoke_error=str(error))

    # Keep the model service alive, with no synthetic GPU keepalive workload.
    code = server.wait()
    update("exited", exit_code=code)
    raise SystemExit(code)
