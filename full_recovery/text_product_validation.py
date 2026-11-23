"""Verify the reference text product's actual provenance schema."""
from pathlib import Path
import time
from .prepare import read,save,sha


def verify(item,output):
    output=Path(output);final=output/'final';scene=read(final/'scene.json')
    provenance=read(final/'provenance.json');initial=read(output/'provenance.json')
    expected=item['source_sha256']
    if (sha(item['original'])!=expected or provenance.get('image_sha256')!=expected
            or initial.get('parameters',{}).get('image_sha256')!=expected):
        raise ValueError('Fresh text original-image provenance differs: '+item['id'])
    if scene.get('source_size')!=item['source_size']:
        raise ValueError('Fresh text coordinate grid differs: '+item['id'])
    parameters=initial.get('parameters',{})
    if parameters.get('baseline_run') is not None or parameters.get('base_pptx_sha256') is not None:
        raise ValueError('Original-only text recovery cannot use prior baseline/base products')
    names=['scene.json','text.pptx','native.pdf','native.png','provenance.json','quality.json']
    for name in names:
        if not (final/name).is_file() or not (final/name).stat().st_size:
            raise ValueError('Missing completed text product: '+name)
    return {'id':item['id'],'source_sha256':expected,'source_size':item['source_size'],
        'source_provenance_sha256':sha(output/'provenance.json'),
        'products_sha256':{name:sha(final/name) for name in names},
        'effective_review_policy':provenance.get('review_policy'),
        'startup_reported_review_policy':parameters.get('review_policy'),
        'fresh_original_inference':True,'historical_prediction_products_reused':False,
        'visual_accuracy_verified':False}


def index(run,destination):
    """An explicit read-only index cannot rewrite a failed worker's status."""
    run=Path(run).resolve();destination=Path(destination).resolve()
    if destination.exists():raise ValueError('Use a new validation index')
    items=read(run/'source_manifest.json')['images'];progress=read(run/'progress.json')['cases']
    results=[]
    for item in items:
        status=progress.get(item['id'],{});state=status.get('state')
        if state not in ('completed','failed'):raise ValueError('Text case may still be writing: '+item['id'])
        if state=='failed' and status.get('error')!="'source_sha256'":
            raise ValueError('A model/process failure cannot be finalized as a wrapper-only error: '+item['id'])
        output=run/'texts/cases'/item['id']
        result=verify(item,output)
        result['worker_case_status']=progress[item['id']];results.append(result)
    destination.mkdir(parents=True)
    for item in items:
        target=destination/'cases'/item['id'];target.parent.mkdir(exist_ok=True)
        target.symlink_to(run/'texts/cases'/item['id'],target_is_directory=True)
    save(destination/'manifest.json',{'cases':[{'id':i['id'],'sha256':i['source_sha256']} for i in items]})
    save(destination/'validation.json',{'source_run':str(run),'at':time.time(),'cases':results,
        'source_worker_status':read(run/'status.json'),'source_outputs_unchanged':True,
        'model_calls':0,'reason':'Verify actual image_sha256 provenance after wrapper-only source_sha256 lookup error'})
    return results
