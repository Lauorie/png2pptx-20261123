"""Paired discovery-only prompt ablation, preserving exact input image bytes."""
import argparse
import importlib.util
import json
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from transport.common import read_json as read, write_json as write, sha256
from transport.model import Model
from presentation import now, overlay

LOCK = threading.RLock()


def load_contract(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def publish(root):
    with LOCK:
        cfg = read(root / 'config.json')
        status = read(root / 'status.json') if (root / 'status.json').exists() else {'state': 'prepared'}
        dest = root / 'reports'; assets = dest / 'assets'; assets.mkdir(parents=True, exist_ok=True)
        lines = ['# 前置发现粒度：同图同配置提示对照', '', f'更新：{now()}；运行：{status["state"]}', '',
                 'A：v006 的发现提示（已有防整列合组说明）；B：明确一次几何任务的负担、文字简短不减框、局部延伸止于当前连接。',
                 '两边均重新请求。固定原图、视图、坐标、模型、默认 thinking 和 113000 输出预算。无修正/几何/去重调用。',
                 '历史 v005 的整列合组是旧证据，不冒充本次 A 重跑结果。框数不等于定位正确率。', '',
                 '[配置](../config.json) · [请求清单](../jobs.json) · [代码/输入哈希](../manifest.json)', '']
        completed = failed = 0
        for item in read(root / 'jobs.json'):
            cid, stage = item['case'], item['stage']
            lines += [f'## {cid} · {stage}', '', '| A：v006 发现提示 | B：本轮澄清提示 |', '| --- | --- |']
            columns = []
            for arm in ('before', 'after'):
                folder = root / 'arms' / arm / 'cases' / cid / stage
                state = read(folder / 'status.json') if (folder / 'status.json').exists() else {'state': 'queued'}
                cell = state['state']
                if (folder / 'result.json').exists():
                    result = read(folder / 'result.json')
                    cell += f'；{len(result["regions"])} 框；group {sum(r["grouping"] == "group" for r in result["regions"])}'
                    completed += 1
                failed += state['state'] == 'failed'
                img = folder / 'boxes.png'
                if img.exists():
                    target = assets / f'{cid}_{stage}_{arm}.png'
                    if not target.exists():shutil.copy2(img, target)
                    rel = f'assets/{target.name}'
                    cell += f'<br>[![定位]({rel})]({rel})'
                rel = f'../arms/{arm}/cases/{cid}/{stage}/request'
                if (folder / 'request/prompt.json').exists():cell += f'<br>[实际 prompt]({rel}/prompt.json)'
                columns.append(cell)
            lines += ['| ' + ' | '.join(columns) + ' |', '']
        lines[2] += f'；返回 {completed}/{2 * len(read(root / "jobs.json"))}；未恢复失败 {failed}'
        output = '\n'.join(lines) + '\n'
        tmp = dest / '发现粒度对照.tmp.md'; tmp.write_text(output); tmp.replace(dest / '发现粒度对照.md')
        (root / 'RESULTS.md').write_text('# 前置发现粒度实验\n\n' + lines[2] + '\n\n[实时两栏结果](reports/发现粒度对照.md)\n\n此实验尚未证明最终几何改善；按原图核对起终点覆盖、粒度、重复和共享分支。\n')


def run(root):
    for name, expected in read(root / 'manifest.json')['files'].items():
        if sha256(root / name) != expected:raise ValueError('Frozen file changed: ' + name)
    cfg = read(root / 'config.json')
    # One original 16-slot localization process must have ended before adding 4.
    # Existing geometry budgets (64+32) and the other localization (16) then
    # leave the total at or below 116, without changing the active experiments.
    gates = [Path(x) for x in cfg['wait_any_execution_finished']]
    while not any(f.exists() and read(f).get('state') == 'finished' for f in gates):
        write(root / 'status.json', {'state': 'waiting_api_budget', 'updated_at': now()}); publish(root)
        time.sleep(10)
    write(root / 'status.json', {'state': 'running', 'started_at': now()}); publish(root)
    modules = {a: load_contract(root / 'arms' / a / 'contracts.py', 'contracts_' + a) for a in ('before', 'after')}
    models = {a: Model(cfg, root / 'arms' / a, system=modules[a].SYSTEM) for a in modules}

    def one(item, arm):
        cid, stage = item['case'], item['stage']
        folder = root / 'arms' / arm / 'cases' / cid / stage
        C = modules[arm]; inp = root / 'inputs' / cid
        size = Image.open(inp / 'global.png').size
        images = [('GLOBAL clean — sole output coordinate frame', inp / 'global.png')]
        if stage == 'global':q = C.discovery_prompt(size)
        else:
            spec = next(v for v in read(inp / 'frames.json')['local_views'] if v['id'] == stage)
            q = C.local_prompt(size, spec['bounds_global'])
            images.append(('LOCAL clean — magnified evidence only, never output this frame', inp / spec['path']))
        write(folder / 'status.json', {'state': 'running', 'started_at': now()})
        try:
            answer = models[arm].ask(folder / 'request', q, images, lambda a: C.discovery_valid(a, size))
            write(folder / 'result.json', answer)
            overlay(inp / 'global.png', answer['regions'], folder / 'boxes.png')
            write(folder / 'status.json', {'state': 'completed', 'finished_at': now()})
        except Exception as exc:
            write(folder / 'status.json', {'state': 'failed', 'error': str(exc), 'finished_at': now()})
        publish(root)

    with ThreadPoolExecutor(max_workers=cfg['api_concurrency']) as pool:
        jobs = [pool.submit(one, item, arm) for item in read(root / 'jobs.json') for arm in modules]
        for job in as_completed(jobs):job.result()
    write(root / 'status.json', {'state': 'finished', 'finished_at': now()}); publish(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args(); run(args.run.resolve())
