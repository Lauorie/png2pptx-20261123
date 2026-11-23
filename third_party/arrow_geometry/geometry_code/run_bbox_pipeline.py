"""Unified prepare/run CLI: pinned bbox discovery through editable PPT."""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
from transport.common import read_json as read,write_json as write,sha256

def invoke(args):
    subprocess.run([sys.executable,'-B',*map(str,args)],check=True)

def prepare(root,discovery_run,images,config,cases,mode):
    root=Path(root).resolve()
    if root.exists():raise ValueError('Run exists; use run to resume, not prepare')
    cfg=read(config);cfg['discovery_provider']='bbox_handoff';cfg['pipeline_mode']=mode
    assert cfg.get('thinking')=={'type':'enabled'} and 'reasoning_effort' not in cfg
    root.mkdir(parents=True);code=root/'runner_code'
    shutil.copytree(Path(__file__).resolve().parent,code,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    if discovery_run:
        source=Path(discovery_run).resolve();origin='import_completed_handoff'
    else:
        source=root/'discovery';origin='new_bbox_discovery'
        invoke([code/'discovery_frontend/integrated_run.py','prepare','--run-dir',source,'--images',*images])
    cfg['bbox_discovery_run']=str(source);write(root/'reconstruction_config.json',cfg)
    write(root/'RUN.json',dict(discovery_mode=origin,discovery_run=str(source),cases=cases,mode=mode))
    if discovery_run:
        cmd=[code/'run.py','--run',root/'reconstruction','--config',root/'reconstruction_config.json','--freeze-only']
        if cases:cmd+=['--cases',*cases]
        invoke(cmd)
    files=[f for f in code.rglob('*') if f.is_file() and '__pycache__' not in f.parts]+[root/'RUN.json',root/'reconstruction_config.json']
    write(root/'bootstrap_manifest.json',{str(f.relative_to(root)):sha256(f) for f in files})
    (root/'README.md').write_text('# bbox发现重建管线\n\n状态以 reconstruction/progress.json 为准。\n\n[重建结果](reconstruction/RESULTS.md) · [运行参数](RUN.json) · [发现交接来源](reconstruction/discovery_manifest.json)\n\n运行：python runner_code/run_bbox_pipeline.py run --run-dir '+str(root)+'\n')
    print('Prepared without model calls:',root,flush=True)

def execute(root,prepare_only=False):
    root=Path(root).resolve()
    for rel,h in read(root/'bootstrap_manifest.json').items():
        if sha256(root/rel)!=h:raise ValueError('Frozen bootstrap changed: '+rel)
    meta=read(root/'RUN.json');code=root/'runner_code'
    if meta['discovery_mode']=='new_bbox_discovery':
        if prepare_only:raise ValueError('New-image discovery must run before reconstruction inputs can be prepared')
        invoke([code/'discovery_frontend/integrated_run.py','run','--run-dir',meta['discovery_run']])
    args=[code/'run.py','--run',root/'reconstruction','--config',root/'reconstruction_config.json']
    if meta['cases']:args+=['--cases',*meta['cases']]
    if prepare_only:args+=['--prepare-only']
    invoke(args)

def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--run-dir',type=Path,required=True)
    inputs=p.add_mutually_exclusive_group(required=True);inputs.add_argument('--discovery-run',type=Path);inputs.add_argument('--images',nargs='+',type=Path);inputs.add_argument('--input-dir',type=Path)
    p.add_argument('--cases',nargs='+');p.add_argument('--config',type=Path,default=Path(__file__).resolve().parent.parent/'config.json');p.add_argument('--mode',choices=['geometry','full'],default='full')
    r=sub.add_parser('run');r.add_argument('--run-dir',type=Path,required=True);r.add_argument('--prepare-only',action='store_true')
    a=parser.parse_args()
    if a.command=='prepare':
        images=a.images or (sorted(a.input_dir.glob('*.png')) if a.input_dir else [])
        if not a.discovery_run and not images:raise ValueError('No PNG input')
        prepare(a.run_dir,a.discovery_run,images,a.config,a.cases,a.mode)
    else:execute(a.run_dir,a.prepare_only)

if __name__=='__main__':main()

