"""Run the VLM-content/OCR-localization pipeline on an input collection.

No changes to input images or older runs. Completed cases can be copied only
after source, policy and implementation hashes agree. API credentials stay in
the environment and are never written to the batch manifest.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_completed(folder, image, threshold, max_score_drop, recognition_mode='staged', glm_crop_refine=False):
    for name in ('summary.json', 'provenance.json', 'final/scene.json',
                 'final/text.pptx', 'final/native.png', 'comparison.png'):
        if not (folder / name).is_file():
            raise ValueError(f'Missing completed artifact: {folder / name}')
    provenance = read(folder / 'provenance.json')
    expected = {'image_sha256': sha(image), 'threshold': threshold,
                'max_score_drop': max_score_drop,'recognition_mode':recognition_mode}
    if glm_crop_refine:expected['glm_crop_refine']=True
    if provenance['parameters'] != expected:
        raise ValueError(f'Completed case source/policy mismatch: {folder}')
    for name, digest in provenance['code_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'Completed case code mismatch: {name}')
    return read(folder / 'summary.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'output', 'ocr-python', 'models'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--concurrency', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=.80)
    parser.add_argument('--max-score-drop', type=float, default=.05,
                        help='Absolute tolerance on the 0–1 scale; .05 means five points')
    parser.add_argument('--recognition-mode',choices=('staged','legacy'),default='staged')
    parser.add_argument('--ocr-mode',choices=('geometry','legacy'),default='geometry')
    parser.add_argument('--ocr-max-side',type=int,default=1536)
    parser.add_argument('--glm-crop-refine',action='store_true')
    parser.add_argument('--completed-case', action='append', default=[], metavar='ID=PATH')
    parser.add_argument('--evidence-cases',type=Path,help='Previous cases: read complete global responses without new API calls')
    parser.add_argument('--case',action='append',help='Optional subset of case IDs')
    args = parser.parse_args()
    if args.evidence_cases and args.recognition_mode!='legacy':
        raise ValueError('Old global response reuse requires --recognition-mode legacy')
    if not 1 <= args.concurrency <= 4:
        raise ValueError('Concurrency must be between one and four')
    if not 0 <= args.threshold <= 1 or not 0 <= args.max_score_drop <= 1:
        raise ValueError('Invalid score thresholds')
    images = {p.parent.name: p.resolve() for p in sorted(args.inputs.glob('*/original.png'))}
    if not images:
        raise ValueError('No original.png inputs')
    if args.case:
        if set(args.case)-set(images):raise ValueError('Unknown requested cases')
        images={i:images[i] for i in args.case}
    reuse = {}
    for spec in args.completed_case:
        ident, folder = spec.split('=', 1)
        if ident not in images or ident in reuse:
            raise ValueError(f'Unknown or duplicate completed case: {ident}')
        reuse[ident] = Path(folder).resolve()
        validate_completed(reuse[ident], images[ident], args.threshold, args.max_score_drop,args.recognition_mode,args.glm_crop_refine)
    if not args.evidence_cases and len(reuse) < len(images) and not any(os.environ.get(k) for k in
                                          ('OpenAI_API', 'OPENAI_API_KEY', 'BOYUE_API_KEY')):
        raise ValueError('API credential must be supplied through environment')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'cases').mkdir()
    (args.output / 'logs').mkdir()
    state = {'state': 'running', 'pid': os.getpid(), 'total': len(images),
             'started_at': datetime.now(timezone.utc).isoformat(),
             'concurrency': args.concurrency, 'completed': 0, 'failed': 0,
             'active_cases': [], 'results': [], 'inputs': str(args.inputs.resolve()),
             'recognition_mode':args.recognition_mode,'glm_crop_refine':args.glm_crop_refine,
             'ocr_mode':args.ocr_mode,'ocr_max_side':args.ocr_max_side,
             'policy': 'VLM transcription; OCR approximate localization; CPU score < 0.80 review; maximum absolute drop 0.02'}
    lock = threading.Lock()
    save(args.output / 'manifest.json', {'cases': [
        {'case': ident, 'image': str(image), 'image_sha256': sha(image),
         'completed_source': str(reuse[ident]) if ident in reuse else None}
        for ident, image in images.items()], 'batch_script_sha256': sha(Path(__file__))})
    save(args.output / 'progress.json', state)

    def work(ident, image):
        with lock:
            state['active_cases'].append(ident)
            save(args.output / 'progress.json', state)
        output = args.output / 'cases' / ident
        print(f'{ident}: started', flush=True)
        record = {'case': ident, 'reused': ident in reuse}
        try:
            if ident in reuse:
                shutil.copytree(reuse[ident], output, symlinks=True)
                summary = validate_completed(output, image, args.threshold, args.max_score_drop,args.recognition_mode,args.glm_crop_refine)
                record['returncode'] = 0
                save(output / 'batch_reuse.json', {'source': str(reuse[ident]),
                    'image_sha256': sha(image), 'new_api_calls': 0})
            else:
                command = [sys.executable, '-B', '-u', str(ROOT / 'run_global_ocr_review.py'),
                           '--image', str(image), '--output', str(output),
                           '--ocr-python', str(args.ocr_python.resolve()), '--models', str(args.models.resolve()),
                           '--threshold', str(args.threshold), '--max-score-drop', str(args.max_score_drop),
                           '--recognition-mode',args.recognition_mode,'--ocr-mode',args.ocr_mode,
                           '--ocr-max-side',str(args.ocr_max_side)]
                if args.glm_crop_refine:command.append('--glm-crop-refine')
                if args.evidence_cases:
                    evidence=(args.evidence_cases/ident).resolve()
                    for _ in range(5):
                        if (evidence/'global').is_dir():break
                        evidence=Path(read(evidence/'reuse_provenance.json')['run'])
                    responses=[]
                    for candidate in sorted(evidence.glob('global/calls/request_*/content.txt')):
                        status=read(candidate.parent/'stream_status.json')
                        if status.get('http_status')==200 and status.get('sse_done') and status.get('finish_reason')=='stop':
                            responses.append(candidate)
                    if not responses:raise ValueError('No complete global evidence; no automatic API fallback')
                    command.extend(['--evidence-run',str(evidence),'--saved-response',str(responses[-1])])
                with (args.output / 'logs' / f'{ident}.log').open('w') as log:
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                record['returncode'] = result.returncode
                if result.returncode:
                    raise RuntimeError(f'Pipeline failed, exit={result.returncode}; see logs/{ident}.log')
                summary = validate_completed(output, image, args.threshold, args.max_score_drop,args.recognition_mode,args.glm_crop_refine)
            record.update(state='completed', total_texts=summary['total_texts'],
                          final_texts=summary['final_texts'], accepted_ids=summary['accepted_ids'],
                          unresolved_multiline_ids=summary['unresolved_multiline_ids'],
                          unresolved_ids=summary['unresolved_ids'])
            record['quality_status']=summary.get('quality_status','not_assessed')
        except Exception as exc:
            record.update(state='failed', error=str(exc))
        with lock:
            state['active_cases'].remove(ident)
            state['results'].append(record)
            state['completed'] += int(record['state'] == 'completed')
            state['failed'] += int(record['state'] == 'failed')
            save(args.output / 'progress.json', state)
        print(f'{ident}: {record["state"]}', flush=True)
        return record

    # Reused results become available immediately, without waiting for API cases.
    for ident in reuse:
        work(ident, images[ident])
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(work, ident, image) for ident, image in images.items() if ident not in reuse]
        for future in as_completed(futures):
            future.result()
    state['state'] = 'completed_with_errors' if state['failed'] else 'completed'
    state['finished_at'] = datetime.now(timezone.utc).isoformat()
    save(args.output / 'progress.json', state)
    save(args.output / 'summary.json', state)
    print(state['state'], state['completed'], '/', state['total'], flush=True)


if __name__ == '__main__':
    main()
