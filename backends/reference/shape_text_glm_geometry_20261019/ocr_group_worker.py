"""Process crop groups sequentially with one persistent CPU OCR engine."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from compare_group_ocr import main as recognize


def save(path, data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    temporary.replace(path)


def attempt_sides(maximum):
    return list(dict.fromkeys([maximum,min(maximum,1024),min(maximum,768)]))


def supervise(manifest_path, manifest):
    """A native crash kills only the OCR child, never the reconstruction job."""
    attempts=[]
    for _ in range(3*len(manifest['jobs'])+1):
        result=subprocess.run([sys.executable,'-B','-u',__file__,'--manifest',str(manifest_path),'--child'])
        pointer=manifest_path.parent/'ocr_worker_current.json'
        record=json.loads(pointer.read_text()) if pointer.exists() else {}
        attempts.append({'returncode':result.returncode,'current':record})
        save(manifest_path.parent/'ocr_worker_restarts.json',attempts)
        if result.returncode==0:return
        # A crash before any recorded recognition attempt cannot be repaired by
        # repeatedly restarting the same interpreter/import environment.
        target=Path(record['output']) if record.get('output') else None
        if not target or not target.is_relative_to(manifest_path.parent):
            raise RuntimeError('OCR child failed before a recoverable attempt')
        if (target/'attempt_status.json').exists():
            old=json.loads((target/'attempt_status.json').read_text())
            if old.get('state')!='running':
                raise RuntimeError('OCR child failed without advancing an attempt')
        target.mkdir(parents=True,exist_ok=True)
        save(target/'attempt_status.json',{'state':'failed','returncode':result.returncode,
                                          'reason':'native_child_exit','pid':record.get('pid')})
    raise RuntimeError('OCR restart budget exhausted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--child',action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if not args.child:
        supervise(args.manifest.resolve(),manifest)
        return
    results = []
    restarts=args.manifest.parent/'ocr_worker_restarts.json'
    native_failed_sides=set()
    if restarts.exists():
        native_failed_sides={r['current'].get('side') for r in json.loads(restarts.read_text())
                             if r.get('returncode',0)<0}
    for job in manifest['jobs']:
        output = Path(job['output'])
        try:
            if not (output/'report.json').is_file():
                output.mkdir(parents=True,exist_ok=True)
                failures=[]
                for side in attempt_sides(job.get('max_side',1536)):
                    if side in native_failed_sides:
                        failures.append({'side':side,'reason':'native_backend_failed_at_this_resolution'})
                        continue
                    attempt=output/f'attempt_{side}'
                    if (attempt/'report.json').exists():
                        value=json.loads((attempt/'report.json').read_text())
                    elif attempt.exists():
                        failures.append({'side':side,'reason':'previous_failed_or_interrupted_attempt'})
                        continue
                    else:
                        save(args.manifest.parent/'ocr_worker_current.json',{
                            'group':job['group'],'side':side,'output':str(attempt.resolve()),'pid':os.getpid()})
                        # recognize creates its own directory; the supervisor
                        # writes native-crash status there if the child exits.
                        try:
                            recognize(['--crop-dir',job['crop_dir'],'--scene',manifest['scene'],
                                '--ids',*job['ids'],'--models',manifest['models'],
                                '--output',str(attempt),'--ocr-max-side',str(side),
                                '--rotations',*map(str,job['rotations'])])
                            value=json.loads((attempt/'report.json').read_text())
                        except Exception as exc:
                            attempt.mkdir(parents=True,exist_ok=True)
                            save(attempt/'attempt_status.json',{'state':'failed','reason':str(exc)})
                            failures.append({'side':side,'reason':str(exc)})
                            continue
                    value['resolution_attempts']=failures+[{'side':side,'state':'completed'}]
                    save(output/'report.json',value)
                    break
                else:raise RuntimeError('All bounded OCR resolution attempts failed')
            results.append({'group':job['group'],'state':'completed'})
        except Exception as exc:
            traceback.print_exc()
            results.append({'group':job['group'],'state':'failed','error':str(exc)})
        path = args.manifest.parent/'ocr_progress.json'
        save(path,{'completed':len(results),'total':len(manifest['jobs']),'results':results})
    print('OCR groups complete',len(results),flush=True)


if __name__ == '__main__':
    main()
