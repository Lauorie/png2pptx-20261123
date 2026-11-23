"""Batch PNG-only text/math recovery, with shared CPU/render limits and resume."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def discover(root):
    """Read image paths only, never PDF, metadata or benchmark answer text."""
    if (root/'taskset.txt').is_file() and (root/'tasks').is_dir():
        ids=(root/'taskset.txt').read_text().splitlines()
        if len(ids)!=len(set(ids)) or any(not i or Path(i).name!=i for i in ids):
            raise ValueError('Invalid or duplicate task ID')
        images={i:root/'tasks'/i/'reference.png' for i in ids}
    else:
        files=sorted(root.glob('*/original.png')) or sorted(root.glob('*/reference.png'))
        images={p.parent.name:p for p in files}
    if not images or any(not p.is_file() for p in images.values()):
        raise ValueError('Missing input images')
    return {i:p.resolve() for i,p in images.items()}


def validate_completed(folder,image,code):
    for name in ('summary.json','provenance.json','final/scene.json','final/text.pptx','final/native.png','comparison.png'):
        if not (folder/name).is_file():raise ValueError('Missing output: '+name)
    p=read(folder/'provenance.json')['parameters']
    if p.get('image_sha256')!=sha(image) or p.get('geometry_backend')!='glm' or not p.get('glm_crop_refine'):
        raise ValueError('Completed source/backend mismatch')
    for name,digest in read(folder/'provenance.json')['code_sha256'].items():
        if code.get(name)!=digest:raise ValueError('Completed code mismatch: '+name)
    return read(folder/'summary.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--concurrency',type=int,default=8)
    parser.add_argument('--cpu-slots',type=int,default=2)
    parser.add_argument('--render-slots',type=int,default=2)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--case-list',type=Path,help='JSON list of selected case IDs; validates membership')
    parser.add_argument('--completed-delivery',type=Path,help='Select only case directories with combined.pptx in an existing delivery')
    parser.add_argument('--retry-from',type=Path,help='Recover only failed cases from a previous batch into a NEW output')
    parser.add_argument('--recover-from',type=Path,help='Restart selected cases in a fresh output, reusing available recognition evidence')
    args=parser.parse_args()
    if not 1<=args.concurrency<=64 or not 1<=args.cpu_slots<=16 or not 1<=args.render_slots<=16:
        raise ValueError('Invalid concurrency')
    if not any(os.environ.get(k) for k in ('OpenAI_API','OPENAI_API_KEY','BOYUE_API_KEY')):
        raise ValueError('API credential required in environment')
    images=discover(args.inputs.resolve());output=args.output.resolve()
    if args.completed_delivery:
        selected=sorted(p.parent.name for p in args.completed_delivery.glob('task_*/combined.pptx'))
        if not selected or not set(selected)<=set(images):
            raise ValueError('No completed deliveries, or unknown completed case IDs')
        images={ident:images[ident] for ident in selected}
    if args.case_list:
        selected=read(args.case_list)
        if not isinstance(selected,list) or len(set(selected))!=len(selected) or not set(selected)<=set(images):
            raise ValueError('Invalid selected case IDs')
        images={ident:images[ident] for ident in selected}
    recovery=None
    if args.recover_from:
        if args.retry_from:raise ValueError('Choose recover-from or retry-from')
        recovery=args.recover_from.resolve()
        if recovery==output:raise ValueError('Recovery output must be fresh')
        previous=read(recovery/'manifest.json')
        known={r['id']:r['sha256'] for r in previous['cases']}
        if any(known.get(i)!=sha(p) for i,p in images.items()):
            raise ValueError('Recovery input mismatch')
    if args.retry_from:
        recovery=args.retry_from.resolve()
        if recovery==output:raise ValueError('Retry output must differ from the original batch')
        previous=read(recovery/'progress.json')
        if previous.get('active_cases'):raise ValueError('Wait for original batch to finish')
        failed={i for i,v in previous['results'].items() if v['state']=='failed'}
        if not failed or not failed<=images.keys():raise ValueError('Missing or invalid failed case IDs')
        images={i:p for i,p in images.items() if i in failed}
    files=[*ROOT.glob('*.py'),ROOT/'config.json',*ROOT.joinpath('src').rglob('*.py'),ROOT/'src/shape_integration/texts/crop_review_rules.txt']
    code={str(p.relative_to(ROOT)):sha(p) for p in files}
    manifest={'inputs':str(args.inputs.resolve()),'cases':[{'id':i,'image':str(p),'sha256':sha(p)} for i,p in images.items()],
              'code_sha256':code,'geometry_backend':'glm','concurrency':args.concurrency,
              'cpu_slots':args.cpu_slots,'render_slots':args.render_slots,
              'max_api_concurrency':args.concurrency*read(ROOT/'config.json')['api_concurrency']}
    if recovery:manifest['recovery_source']=str(recovery)
    if output.exists():
        previous_manifest=read(output/'manifest.json')
        def resume_identity(value):
            value={k:v for k,v in value.items() if k not in (
                'concurrency','cpu_slots','render_slots','max_api_concurrency')}
            value['code_sha256']={k:v for k,v in value['code_sha256'].items()
                                  if k!=Path(__file__).name}
            return value
        if not args.resume or resume_identity(previous_manifest)!=resume_identity(manifest):
            raise ValueError('Use a fresh output or resume identical inputs/config/code')
        if previous_manifest!=manifest:
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            save(output/'resume_history'/f'manifest_{stamp}.json',previous_manifest)
            if (output/'progress.json').exists():
                save(output/'resume_history'/f'progress_{stamp}.json',read(output/'progress.json'))
            save(output/'manifest.json',manifest)
    else:
        output.mkdir(parents=True);save(output/'manifest.json',manifest)
    (output/'logs').mkdir(exist_ok=True);(output/'cases').mkdir(exist_ok=True)
    state={'state':'running','pid':os.getpid(),'started_at':now(),'total':len(images),'completed':0,'failed':0,
           'active_cases':{},'results':{},'concurrency':args.concurrency,'cpu_slots':args.cpu_slots,
           'render_slots':args.render_slots,'max_api_concurrency':manifest['max_api_concurrency']}
    lock=threading.Lock();cpus=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else []
    def publish():save(output/'progress.json',state)
    publish()
    def work(index,ident,image):
        folder=output/'cases'/ident;started=time.monotonic()
        record={'case':ident,'started_at':now()}
        with lock:
            state['active_cases'][ident]={'stage':'starting','started_at':now()};publish()
        try:
            if (folder/'summary.json').exists():
                summary=validate_completed(folder,image,code);record['reused']=True
            else:
                command=[sys.executable,'-B','-u',str(ROOT/'run_global_ocr_review.py'),
                         '--image',str(image),'--output',str(folder),'--geometry-backend','glm',
                         '--recognition-mode','staged','--glm-crop-refine','--ocr-template-validation','on']
                if folder.exists():command.append('--resume')
                if recovery and (recovery/'cases'/ident/'provenance.json').exists():
                    command.extend(['--recovery-run',str(recovery/'cases'/ident)])
                env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                         NUMEXPR_NUM_THREADS='1',MALLOC_ARENA_MAX='2',
                         SHAPE_BATCH_LIMIT_DIR=str(output/'.limits'),SHAPE_CPU_SLOTS=str(args.cpu_slots),
                         SHAPE_RENDER_SLOTS=str(args.render_slots))
                if cpus:env['SHAPE_CPU_AFFINITY']=','.join(str(cpus[(index*2+j)%len(cpus)]) for j in range(min(2,len(cpus))))
                with (output/'logs'/f'{ident}.log').open('a') as log:
                    proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    with lock:
                        state['active_cases'][ident].update(pid=proc.pid,log=str(output/'logs'/f'{ident}.log'));publish()
                    print(ident,'started',proc.pid,flush=True)
                    code_return=proc.wait()
                record['returncode']=code_return
                if code_return:raise RuntimeError(f'Pipeline exit={code_return}; see logs/{ident}.log')
                summary=validate_completed(folder,image,code)
            record.update(state='completed',total_texts=summary['total_texts'],final_texts=summary['final_texts'],
                          quality_status=summary['quality_status'],accepted=len(summary['accepted_ids']),
                          unresolved=len(summary['unresolved_ids']))
        except Exception as exc:record.update(state='failed',error=str(exc))
        record.update(finished_at=now(),seconds=round(time.monotonic()-started,2))
        with lock:
            state['active_cases'].pop(ident,None);state['results'][ident]=record
            state['completed']+=int(record['state']=='completed');state['failed']+=int(record['state']=='failed')
            publish()
        print(ident,record['state'],record.get('error',''),flush=True)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures=[pool.submit(work,i,ident,image) for i,(ident,image) in enumerate(images.items())]
        for future in as_completed(futures):future.result()
    state.update(state='completed_with_errors' if state['failed'] else 'completed',finished_at=now())
    publish();save(output/'summary.json',state)
    print(state['state'],state['completed'],'/',state['total'],flush=True)


if __name__=='__main__':main()
