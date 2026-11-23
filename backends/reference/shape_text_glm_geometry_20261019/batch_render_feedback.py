"""Run render-feedback checks for every input with an existing text baseline."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--baseline-cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-root', type=Path)
    parser.add_argument('--concurrency', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 4:
        raise ValueError('API concurrency must be between 1 and 4')
    if not any(os.environ.get(k) for k in ('OpenAI_API', 'OPENAI_API_KEY', 'BOYUE_API_KEY')):
        raise ValueError('API key must be supplied through the environment')
    inputs = sorted(args.inputs.glob('*/original.png'))
    if not inputs:
        raise ValueError('No input images found')
    for image in inputs:
        baseline = args.baseline_cases / image.parent.name / 'texts'
        for name in ('scene.json', 'text.pptx', 'native.png', 'provenance.json', 'summary.json'):
            if not (baseline / name).is_file():
                raise ValueError(f'Missing baseline artifact: {baseline / name}')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'cases').mkdir()
    (args.output / 'logs').mkdir()
    progress = {'state': 'running', 'pid': os.getpid(), 'total': len(inputs),
                'started_at': datetime.now(timezone.utc).isoformat(),
                'inputs': str(args.inputs.resolve()),
                'baseline_cases': str(args.baseline_cases.resolve()),
                'policy': 'CPU score > 0.80 keeps original; otherwise localize SOURCE glyph bbox/color for every target, then reject any post-render score decrease',
                'scope': 'all flagged groups; reuse OCR, rerun render-feedback',
                'completed': 0, 'active_case': None, 'results': []}
    save(args.output / 'progress.json', progress)
    for image in inputs:
        ident = image.parent.name
        baseline = args.baseline_cases / ident / 'texts'
        output = args.output / 'cases' / ident
        progress['active_case'] = ident
        save(args.output / 'progress.json', progress)
        # No six-group pilot cap: the number of groups cannot exceed row count.
        groups = max(1, len(read(baseline / 'scene.json')['texts']))
        command = [sys.executable, '-B', str(ROOT / 'review_rendered_text.py'),
                   '--baseline', str(baseline), '--image', str(image),
                   '--output', str(output), '--max-groups', str(groups),
                   '--concurrency', str(args.concurrency)]
        if args.reuse_root and (args.reuse_root / ident / 'provenance.json').is_file():
            command.extend(['--reuse-review', str(args.reuse_root / ident)])
        print(f'{ident}: started', flush=True)
        with (args.output / 'logs' / f'{ident}.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record = {'case': ident, 'returncode': result.returncode}
        if (output / 'summary.json').is_file():
            summary = read(output / 'summary.json')
            for key in ('state', 'reviewed_groups', 'proposed_targets', 'accepted_targets',
                        'rejected_targets', 'deferred_ids', 'final_cpu_failed_ids', 'quality_status', 'errors'):
                record[key] = summary.get(key)
        else:
            record['state'] = 'failed'
        progress['results'].append(record)
        progress['completed'] += 1
        progress['active_case'] = None
        save(args.output / 'progress.json', progress)
        print(f'{ident}: finished, exit={result.returncode}, accepted={record.get("accepted_targets")}', flush=True)
    progress['state'] = 'completed_with_errors' if any(
        r.get('state') != 'completed' for r in progress['results']) else 'completed'
    progress['finished_at'] = datetime.now(timezone.utc).isoformat()
    save(args.output / 'progress.json', progress)


if __name__ == '__main__':
    main()
