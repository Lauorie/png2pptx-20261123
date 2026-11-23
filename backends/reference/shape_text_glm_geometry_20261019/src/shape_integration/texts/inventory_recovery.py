"""Salvage independently valid inventory objects; never invent missing boxes."""
from copy import deepcopy
import json

from ..common import read, save, sha
from . import model


def latest_complete(root):
    """Only same-image, complete stage-one responses qualify as evidence."""
    for path in sorted((root/'calls').glob('request_*/content.txt'),reverse=True):
        try:
            status=read(path.parent/'stream_status.json')
            identity=read(path.parent/'identity.json')
            if (status.get('http_status')!=200 or status.get('finish_reason')!='stop'
                    or not status.get('sse_done')):continue
            if not identity.get('prompt',{}).get('task','').startswith('Stage 1:'):continue
            if sha(root/'input/clean.png') not in [i['sha256'] for i in identity['images']]:continue
            text=path.read_text().strip()
            if text.startswith('```') and text.endswith('```'):text='\n'.join(text.splitlines()[1:-1])
            raw=json.loads(text)
            if isinstance(raw,dict) and isinstance(raw.get('texts'),list):return raw,path
        except (OSError,ValueError,KeyError,TypeError):continue
    return None,None


def partition(raw,size,opt):
    from .staged_formula import inventory_validate
    good=[];bad=[];regions=[];warnings=[]
    for index,row in enumerate(raw['texts']):
        try:
            value=inventory_validate({'texts':[row]},size,opt)
            good.extend(value['texts'])
        except (ValueError,TypeError,KeyError,AttributeError) as exc:
            bad.append({'index':index,'candidate':deepcopy(row),'reason':str(exc)})
    for index,region in enumerate(raw.get('regions',[]) if isinstance(raw.get('regions',[]),list) else []):
        try:
            value=inventory_validate({'texts':[],'regions':[region]},size,opt)
            regions.extend(value['regions'])
        except (ValueError,TypeError,KeyError,AttributeError) as exc:
            warnings.append({'stage':'inventory_region','index':index,'region':region,'reason':str(exc)})
    if not isinstance(raw.get('regions',[]),list):
        warnings.append({'stage':'inventory_region','reason':'Ignored malformed optional regions array'})
    uncertain=raw.get('uncertain',[])
    uncertain=[v for v in uncertain if isinstance(v,str)] if isinstance(uncertain,list) else []
    return {'texts':good,'regions':regions,'uncertain':uncertain},bad,warnings


def recover(root,size,opt,runtime,cause):
    from .staged_formula import inventory_validate,inventory_prompt
    raw,path=latest_complete(root)
    if raw is None:
        raise RuntimeError('No complete same-image inventory available for recovery: '+str(cause))
    value,bad,warnings=partition(raw,size,opt)
    repaired=[]
    for rejected in bad:
        index=rejected['index'];folder=root/'object_repairs'/f'item_{index:04d}'
        prompt=inventory_prompt(size,opt)
        prompt['task']='Repair ONLY one rejected inventory occurrence in the original full-image coordinates.'
        prompt['rejected_candidate']=rejected['candidate']
        prompt['validation_error']=rejected['reason']
        prompt['rules'] += [
            'Return only the visible occurrence represented by rejected_candidate. '
            'All other objects were already accepted; do not reproduce them. '
            'Read its tight bbox and style from the CLEAN image, not the invalid coordinates. '
            'Return regions=[]. If this was a spurious/empty occurrence, return texts=[] '
            'and explain this in uncertain. Never fabricate coordinates or characters.']
        try:
            reply=model.ask(runtime,folder,prompt,[('CLEAN full figure',root/'input/clean.png')],
                lambda v:inventory_validate(v,size,opt,local=True),opt)
            value['texts'].extend(reply['texts'])
            repaired.append({'index':index,'count':len(reply['texts']),'uncertain':reply['uncertain']})
            if not reply['texts'] or reply['uncertain']:
                warnings.append({'stage':'inventory_object','index':index,
                    'reason':'Targeted retry returned empty or uncertain evidence','details':reply['uncertain']})
        except Exception as exc:
            # Valid siblings survive; invalid geometry is never clamped or guessed.
            warnings.append({'stage':'inventory_object','index':index,'candidate':rejected['candidate'],
                'reason':str(exc),'status':'unresolved_no_valid_bbox'})
    save(root/'partial_recovery.json',{'source':str(path),'sha256':sha(path),'cause':str(cause),
        'valid_original_count':len(raw['texts'])-len(bad),'rejected':bad,'repaired':repaired,
        'warnings':warnings,'accepted_count':len(value['texts'])})
    if not value['texts']:
        raise RuntimeError('Inventory recovery has no valid text objects; refusing an empty success')
    value['recovery_warnings']=warnings
    return value
