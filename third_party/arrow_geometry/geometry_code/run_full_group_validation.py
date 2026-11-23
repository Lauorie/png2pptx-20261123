"""Full pipeline validation with shared request slots and saved-call dependencies."""
import argparse
import copy
import fcntl
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pipeline
import protocol
from transport.common import read_json as read, write_json as write, sha256 as sha
from transport.model import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    lock = (run / 'writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = read(run / 'code_manifest.json')
    if any(sha(run / 'code' / n) != h for n, h in manifest['files'].items()):
        raise ValueError('Frozen code changed')
    if sha(run / 'config.json') != manifest['config_sha256']:
        raise ValueError('Frozen config changed')
    if manifest.get('selection_sha256') and sha(run / 'validation_selection.json') != manifest['selection_sha256']:
        raise ValueError('Frozen selection changed')
    if any(sha(run / name) != digest for name, digest in manifest.get('input_files', {}).items()):
        raise ValueError('Frozen input changed')
    config = read(run / 'config.json')
    spec = read(run / 'validation_selection.json')
    ids = spec['cases']
    source = Path(config['global_predictions_run']) / 'manifest.json'
    known = {r['id'] for r in read(source)['images']}
    if len(ids) != len(set(ids)) or not set(ids) <= known:
        raise ValueError('Invalid validation cases')
    write(run / 'manifest.json', dict(ids=ids, source=str(source), source_sha256=sha(source),
                                     same_algorithm_all_cases=True, dependencies=spec.get('dependencies', {})))
    write(run / 'process.json', dict(pid=os.getpid(), started=time.time()))
    model = Model(config, run_dir=run, reuse_from=config.get('model_cache_from'), system=protocol.SYSTEM)
    locks = {k: threading.Semaphore(config[k + '_workers']) for k in ('cpu', 'sam')}
    states, done, errors = {}, {}, {}
    mutex = threading.RLock()
    started = time.monotonic()

    def report(cid=None, value=None):
        with mutex:
            if cid:
                states[cid] = copy.deepcopy(value)
            write(run / 'progress.json', dict(total=len(ids), completed=len(done), cases=states,
                                             errors=errors, seconds=time.monotonic() - started, updated=time.time()))
            lines = ['# '+spec.get('title','分组身份：整图完整流程'), '',
                     f'已结束 {len(done)}/{len(ids)}。密集线网以恢复连接结构为主；孤立箭头检查几何准确性。'
                     '原生渲染完成不等于视觉准确。重复暂不处理。', '',
                     '| case | 状态 | 原图 | 当前实例 | 样式／最终 PPT | 叠图 |',
                     '|---|---|---|---|---|---|']
            for case in ids:
                folder = run / 'cases' / case
                rel = 'cases/' + case
                names = ['clean.png', 'result/instances.png',
                         'native/styled.png' if (folder / 'native/styled.png').exists() else 'result/styled.png',
                         'result/overlap.png']
                images = [f'![]({rel}/{name})' if (folder / name).exists() else '等待' for name in names]
                title = f'[{case}]({rel}/RESULTS.md)' if (folder / 'RESULTS.md').exists() else case
                lines.append('| ' + ' | '.join([title, states.get(case, {}).get('stage', '等待'), *images]) + ' |')
            lines += ['', '[实验说明](实验说明.md) · [实时状态](progress.json) · [分组小实验](../RESULTS.md)',
                      '', '运行中图片是部分阶段结果；完整结论以 completed.json 和最终叠图审核为准。']
            tmp = run / 'RESULTS.md.tmp'
            tmp.write_text('\n'.join(lines) + '\n')
            tmp.replace(run / 'RESULTS.md')
            if spec.get('baselines'):
                comparison=['# 同阶段几何与最终渲染对照','',
                    '初始几何优先使用 stages/initial 快照；尚未生成时标为部分结果。'
                    '最终样式只展示实际完成的 native 渲染。执行成功与视觉准确分开判断。','',
                    '| case | 基线几何 | '+spec.get('comparison_label','本轮')+'几何 |','|---|---|---|']
                for case in ids:
                    cells=[]
                    for base in (Path(spec['baselines'][case]),run/'cases'/case):
                        path=base/'stages/initial/result/instances.png';stage='完整初始阶段'
                        if not path.exists():path=base/'result/instances.png';stage='当前部分结果'
                        cells.append(stage+'<br>![]('+os.path.relpath(path,run)+')' if path.exists() else '等待')
                    comparison.append('| '+case+' | '+' | '.join(cells)+' |')
                comparison+=['','| case | 基线实际PPT | '+spec.get('comparison_label','本轮')+'实际PPT |','|---|---|---|']
                for case in ids:
                    cells=[]
                    for base in (Path(spec['baselines'][case]),run/'cases'/case):
                        path=base/'native/styled.png'
                        cells.append('![]('+os.path.relpath(path,run)+')' if path.exists() else '渲染尚未完成／缺失')
                    comparison.append('| '+case+' | '+' | '.join(cells)+' |')
                temp=run/'COMPARISON.md.tmp';temp.write_text('\n'.join(comparison)+'\n');temp.replace(run/'COMPARISON.md')

    def work(cid):
        dependency = spec.get('dependencies', {}).get(cid)
        if dependency:
            waiting = time.monotonic()
            while not Path(dependency['finished']).exists():
                process = read(dependency['process'])
                try:
                    os.kill(process['pid'], 0)
                except ProcessLookupError:
                    # No active producer remains; fresh mainline requests may
                    # now run without duplicating an in-flight saved request.
                    write(run / ('dependency_' + cid + '.json'),
                          dict(state='producer_ended_without_result', source=dependency))
                    break
                report(cid, dict(stage='waiting_for_existing_group', seconds=time.monotonic() - waiting))
                time.sleep(5)
            else:
                write(run / ('dependency_' + cid + '.json'),
                      dict(state='ready', source=dependency, waited_seconds=time.monotonic() - waiting))
        write(run/'cases'/cid/'case_started.json',dict(started=time.time()))
        return pipeline.run_case(cid, config, run / 'cases' / cid, model, locks, report)

    report()
    monitor_stop=threading.Event()
    def monitor():
        import run_metrics
        while not monitor_stop.is_set():
            try:run_metrics.update(run,spec)
            except Exception as exc:write(run/'performance_error.json',dict(error=str(exc)))
            monitor_stop.wait(30)
    monitor_thread=None
    if spec.get('performance_baseline'):
        monitor_thread=threading.Thread(target=monitor,daemon=True);monitor_thread.start()
    with ThreadPoolExecutor(max_workers=config['case_workers']) as pool:
        jobs = {pool.submit(work, cid): cid for cid in ids}
        for future in as_completed(jobs):
            cid = jobs[future]
            try:
                done[cid] = future.result()
                print(cid, done[cid]['state'], flush=True)
            except Exception as exc:
                errors[cid] = str(exc)
                states[cid] = dict(stage='failed', error=str(exc))
                write(run / 'cases' / cid / 'error.json', dict(error=str(exc), traceback=traceback.format_exc()))
                print(cid, 'failed', str(exc), flush=True)
            report()
    write(run / 'finished.json', dict(total=len(ids), completed=len(done), cases=done, errors=errors,
                                      seconds=time.monotonic() - started))
    monitor_stop.set()
    if monitor_thread is not None:
        monitor_thread.join(timeout=5)
        import run_metrics
        run_metrics.update(run,spec)


if __name__ == '__main__':
    main()
