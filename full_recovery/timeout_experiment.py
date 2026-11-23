"""Paired critic-budget experiment on identical source/render inputs."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from .merge_existing import read,save


def run_experiment(run):
    spec=run/'timeout_experiment_inputs.json'
    if not spec.exists():return
    from shape_integration.common import Runtime
    from shape_integration.styling.contracts import rows
    settings=read(run/'settings.json');config=read(run/'shape/pipeline_config.json')
    results=[]
    for index,item in enumerate(read(spec)['requests']):
        request=read(item['request']);prompt=request['prompt'];expected=[r['id'] for r in prompt['targets']]
        job=read(item['job']);scene=read(item['scene'])
        table={a:scene['parts'][k] for a,k in job['mapping'].items()}
        images=[(r['label'],Path(r['path'])) for r in request['images']]
        def arm(name):
            optimized=name=='bounded'
            budget=min(request['parameters']['max_tokens'],max(item.get('bounded_min_tokens',8192),768*len(expected))) if optimized else request['parameters']['max_tokens']
            effort=item.get('bounded_effort','low') if optimized else request['parameters'].get('reasoning_effort','high')
            rt=Runtime({**config,'api_concurrency':2,'shape_schema_retries':0,
                'reasoning_effort':effort,
                'request_timeout_seconds':360},settings['glm_url'],'')
            folder=run/'timeout_experiment'/str(index)/name;started=time.monotonic()
            def validate(raw):
                out=rows(raw,expected,'critic',table)
                if out.get('_errors'):raise ValueError('Incomplete critic rows: '+str(out['_errors']))
                return out
            result={'arm':name,'requested_budget':budget,'reasoning_effort':effort,'targets':len(expected),'images':len(images)}
            try:
                value=rt.call(folder,'critic',request['system'],prompt,images,validate,budget)
                result.update(state='completed',checks=value['checks'],valid_rows=len(value['checks']))
            except Exception as exc:
                result.update(state='failed',error_type=type(exc).__name__,error=str(exc),valid_rows=0)
            result['elapsed_seconds']=time.monotonic()-started
            save(folder/'experiment_result.json',result)
            print('CRITIC_AB',index,name,result['state'],round(result['elapsed_seconds'],2),flush=True)
            return result
        with ThreadPoolExecutor(max_workers=2) as pool:pair=list(pool.map(arm,['baseline','bounded']))
        results.append({'request':item['request'],'arms':pair})
        save(run/'timeout_experiment/results.json',{'paired_same_prompt_and_images':True,'requests':results})
