"""Batch all originals, optionally reusing a complete historical detector run."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image

from ..frames import resized_size
from ..scene import save
from .clients import file_sha
from .report import build_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--legacy-run', type=Path); p.add_argument('--workers', type=int, default=3)
    p.add_argument('--resume', action='store_true'); p.add_argument('--report-output', type=Path)
    p.add_argument('--boundary-review-rounds', type=int, default=2)
    p.add_argument('--max-boundary-calls', type=int, default=64)
    p.add_argument('--boundary-max-tokens', type=int, default=1536)
    args = p.parse_args(); root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    if args.boundary_review_rounds < 1 or args.max_boundary_calls < 0 or args.boundary_max_tokens < 1:
        p.error('Boundary rounds/tokens must be positive and total calls nonnegative')
    paths = sorted(args.source.glob('*/original.png')) or sorted(args.source.glob('*.png'))
    if not paths: p.error('No source PNGs')
    legacy = json.loads((args.legacy_run/'recovery/assets.json').read_text()) if args.legacy_run else None
    cases = []
    for path in paths:
        ident = path.parent.name if path.name=='original.png' else path.stem
        source_sha = file_sha(path); matches = [a for a in legacy if a['source_sha256']==source_sha] if legacy is not None else []
        extra = []
        if legacy is not None:
            if matches:
                extra = ['--legacy-assets', str((args.legacy_run/'recovery/assets.json').resolve())]
            else:
                old = json.loads((args.legacy_run/'results'/ident/'scene.json').read_text())
                if old['state']!='model_review_complete' or old['objects'] or file_sha(old['source'])!=source_sha:
                    raise ValueError('Cannot treat unverified/mismatched historical scene as empty: '+ident)
                with Image.open(path) as im: size = list(resized_size(im.size, 2048))
                empty = root/'imported_empty'/(ident+'.json')
                save(empty, {'assets': [], 'uncertain': [], 'input_size': size, 'source_sha256': source_sha,
                             'coverage_status': 'imported_empty_inventory', 'provenance': str((args.legacy_run/'results'/ident/'scene.json').resolve())})
                extra = ['--candidates', str(empty)]
        cases.append({'id': ident, 'source': str(path.resolve()), 'source_sha256': source_sha, 'run': str(root/ident),
                      'expected_assets': [{'id': a['id'], 'source_box_half_open': a['box_source_half_open']} for a in matches], 'extra_args': extra})
    if len({c['id'] for c in cases}) != len(cases): raise ValueError('Duplicate case IDs')
    boundary_config = {k: getattr(args, k) for k in ('boundary_review_rounds', 'max_boundary_calls', 'boundary_max_tokens')}
    manifest = {'cases': cases, 'legacy_run': str(args.legacy_run.resolve()) if args.legacy_run else None,
                'boundary_review_config': boundary_config}
    prior = root/'batch_manifest.json'
    if prior.exists() and (not args.resume or json.loads(prior.read_text())!=manifest):
        raise ValueError('Batch exists; resume requires identical inputs')
    save(prior, manifest); records = []
    def work(case):
        command = [sys.executable, '-u', '-m', 'png2pptx.images', '--image', case['source'], '--output', case['run'], *case['extra_args']]
        for key, value in boundary_config.items(): command.extend(['--'+key.replace('_', '-'), str(value)])
        if args.resume and (Path(case['run'])/'config.json').exists(): command.append('--resume')
        with (root/(case['id']+'.log')).open('a') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        return {'id': case['id'], 'exit_code': result.returncode, 'run': case['run']}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for future in as_completed([pool.submit(work, c) for c in cases]):
            record = future.result(); records.append(record)
            save(root/'batch_status.json', {'state': 'running', 'completed': len(records), 'total': len(cases), 'cases': records})
            print(json.dumps(record), flush=True)
    save(root/'batch_status.json', {'state': 'finished', 'completed': len(records), 'total': len(cases), 'cases': records})
    summary = build_report(root, args.report_output or root/'review')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return int(any(r['exit_code'] for r in records))


if __name__ == '__main__': raise SystemExit(main())
