"""Join completed cases from one ongoing fresh experiment in canonical order."""
import argparse
from pathlib import Path
import time
from .merge_existing import execute,inspect_sources,read,save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    run=parser.parse_args().run_dir.resolve();settings=read(run/'settings.json')
    upstream=Path(settings['upstream_fresh_run'])
    items=read(run/'source_manifest.json')['images']
    def products():
        pending={row['id'] for row in items};records=[]
        while pending:
            state=read(upstream/'fresh_stage_status.json').get('branches',{})
            status=read(upstream/'status.json')
            if status.get('state') in ('failed','stopped'):
                raise RuntimeError('Upstream fresh experiment stopped before all products were committed')
            shape_progress=upstream/'shape/progress.json'
            shapes=read(shape_progress).get('cases',{}) if shape_progress.is_file() else {}
            emitted=False
            for item in items:
                cid=item['id']
                if cid not in pending:continue
                text=state.get('fresh_text_'+cid,{})
                if text.get('state')=='failed' or shapes.get(cid,{}).get('state')=='failed':
                    raise RuntimeError('Fresh source case failed: '+cid)
                if text.get('state')!='completed' or not (upstream/'shape/geometry/cases'/cid/'complete.json').is_file():continue
                ready,missing=inspect_sources(upstream,Path(settings['source_text_run']),
                    Path(settings['source_arrow_run']),[cid])
                if missing:continue
                records.extend(ready);pending.remove(cid);emitted=True
                save(run/'sources.json',{'ready':records,'missing':[],'pending':sorted(pending),
                    'upstream_fresh_run':str(upstream),'shape_text_inference_from_original_png':True,
                    'historical_shape_text_products':False,'arrow_policy':'explicit_user_selected_v049'})
                yield ready[0]
            if not emitted:time.sleep(1)
    execute(run,record_stream=products())


if __name__=='__main__':main()
