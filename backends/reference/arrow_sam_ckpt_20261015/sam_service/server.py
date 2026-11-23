#!/usr/bin/env python3
"""One resident SAM3 engine; health stays responsive while inference is busy."""
from __future__ import annotations

import argparse
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import threading
import time
import traceback
import uuid

from engine import SamEngine, save
from verify_assets import verify_assets


def make_server(engine, address, run_dir, state):
    run_dir = Path(run_dir)
    infer_lock = threading.Lock()
    state_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, body):
            payload = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            if status == 503:
                self.send_header('Retry-After', '1')
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass  # The committed artifact remains usable after a client disconnect.

        def do_GET(self):
            if self.path not in ('/', '/health'):
                return self.respond(404, {'error': 'Unknown endpoint'})
            with state_lock:
                snapshot = dict(state)
            return self.respond(200, {**snapshot, 'busy': infer_lock.locked()})

        def do_POST(self):
            if self.path not in ('/', '/v1/segment'):
                return self.respond(404, {'error': 'Unknown endpoint'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 1024 * 1024:
                    raise ValueError('Expected JSON body of at most 1 MiB')
                self.connection.settimeout(30)
                request = json.loads(self.rfile.read(size))
                engine.validate(request)
                request = dict(request)
                request.setdefault('output', str(run_dir / 'requests' / uuid.uuid4().hex))
            except (ValueError, TypeError, OSError) as error:
                return self.respond(400, {'error': str(error)})
            if not infer_lock.acquire(blocking=False):
                return self.respond(503, {'error': 'SAM is busy; retry after this request finishes'})
            try:
                with state_lock:
                    state.update(phase='busy', last_started_at=time.time())
                    save(run_dir / 'service-state.json', state)
                # The optional keepalive releases this lock after every short burst.
                with (run_dir / 'gpu-work.lock').open('a') as gpu_lock:
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX)
                    result = engine(request)
                status = 200
                with state_lock:
                    state['requests_completed'] += 1
                    state['last_output'] = request['output']
            except FileExistsError:
                status, result = 409, {'error': 'Output directory already exists; use a fresh path'}
            except Exception:
                detail = traceback.format_exc()
                print(detail, flush=True)
                status, result = 500, {'error': detail, 'output': request['output']}
                with state_lock:
                    state['requests_failed'] += 1
            finally:
                with state_lock:
                    state.update(phase='ready', last_finished_at=time.time())
                    save(run_dir / 'service-state.json', state)
                infer_lock.release()
            self.respond(status, result)

    server = ThreadingHTTPServer(address, Handler)
    server.daemon_threads = False
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    owner = (run_dir / 'server.lock').open('a')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = json.loads(args.config.read_text())
    state = {'phase': 'loading', 'pid': os.getpid(), 'started_at': time.time(),
             'requests_completed': 0, 'requests_failed': 0}
    state_path = run_dir / 'service-state.json'
    save(state_path, state)
    stopped = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.set())
    server = None
    try:
        checkpoint_sha256 = verify_assets(config)
        engine = SamEngine(config)
        engine.metadata.update(checkpoint_sha256=checkpoint_sha256,
                               checkpoint_hash_verified=True, runtime_assets_verified=True)
        engine.warmup()
        state.update(model=engine.metadata, warmup_passed=True)
        server = make_server(engine, (config['host'], int(config['port'])), run_dir, state)
        server.timeout = 0.5
        state.update(phase='ready', ready_at=time.time(), port=int(config['port']))
        save(state_path, state)
        print(json.dumps(state), flush=True)
        while not stopped.is_set():
            server.handle_request()
    except Exception:
        state.update(phase='failed', error=traceback.format_exc(), stopped_at=time.time())
        save(state_path, state)
        raise
    finally:
        if server:
            server.server_close()
        if state['phase'] != 'failed':
            state.update(phase='stopped', stopped_at=time.time())
            save(state_path, state)
        owner.close()


if __name__ == '__main__':
    main()
