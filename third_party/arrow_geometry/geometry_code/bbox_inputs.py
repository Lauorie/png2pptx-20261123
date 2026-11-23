"""Frozen bbox discovery handoff, independent of SAM instance identities."""
import copy,math,shutil
import numpy as np
from pathlib import Path
from PIL import Image
from transport.common import read_json as read,write_json as write,sha256 as sha
from discovery_frontend.semantic_hints import semantic_text

def cases(source):
    source=Path(source)
    ids=read(source/'cases.json')
    if not isinstance(ids,list) or not ids or len(ids)!=len(set(ids)):raise ValueError('Invalid discovery case list')
    return ids

def ready(source,cid):
    p=Path(source)/'cases'/cid
    return all((p/f).is_file() for f in ['finished.json','reconstruction_handoff.json','result.json'])

def freeze(run,config,selected=None):
    source=Path(config['bbox_discovery_run']).resolve();known=cases(source);ids=selected or known
    if len(ids)!=len(set(ids)) or not set(ids)<=set(known):raise ValueError('Unknown or repeated discovery cases')
    missing=[c for c in ids if not ready(source,c)]
    if missing:raise ValueError('Discovery handoff not finished; select completed cases explicitly or wait: '+', '.join(missing))
    manifest={'provider':'v26_bbox_handoff','source_run':str(source),'selected_cases':ids,'source_total':len(known),'files':{}}
    provider=read(source/'manifest.json') if (source/'manifest.json').exists() else {}
    if provider:
        dst=Path(run)/'discovery_inputs/provider_manifest.json';dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/'manifest.json',dst)
        manifest['files'][str(dst.relative_to(run))]={'source':str(source/'manifest.json'),'sha256':sha(dst)}
    for cid in ids:
        folder=Path(run)/'discovery_inputs'/cid;folder.mkdir(parents=True,exist_ok=True)
        for src,dest in [(source/'inputs'/cid/'source.png','clean.png'),(source/'inputs'/cid/'views.json','views.json'),
                         (source/'cases'/cid/'reconstruction_handoff.json','handoff.json'),(source/'cases'/cid/'result.json','upstream_result.json')]:
            before=sha(src);shutil.copy2(src,folder/dest)
            if sha(src)!=before or sha(folder/dest)!=before:raise ValueError('Discovery changed during freeze: '+str(src))
            manifest['files'][str((folder/dest).relative_to(run))]={'source':str(src),'sha256':before}
        with Image.open(folder/'clean.png') as image:size=image.size
        if read(folder/'views.json')['native_size']!=list(size):raise ValueError('Discovery native size mismatch')
        views=read(folder/'views.json');actual=sha(folder/'clean.png')
        if views['source_sha256']!=actual:
            files=provider.get('files',{})
            if files.get('inputs/'+cid+'/source.png')!=actual or files.get('inputs/'+cid+'/views.json')!=sha(folder/'views.json'):
                raise ValueError('Discovery input hash mismatch, not certified by provider manifest')
            whole=next((v for v in views.get('views',[]) if v.get('view_id')=='G'),None)
            if not whole or whole['transform']['crop_bounds_global_half_open']!=[0,0,*size]:raise ValueError('No full view to verify re-encoded source')
            original_view=source/whole['path']
            if sha(original_view)!=whole['sha256']:raise ValueError('Upstream model image changed')
            with Image.open(original_view) as im:reference=im.convert('RGB')
            with Image.open(folder/'clean.png') as im:recreated=im.convert('RGB').resize(reference.size,Image.Resampling.LANCZOS)
            if not np.array_equal(np.asarray(reference),np.asarray(recreated)):raise ValueError('Source pixels differ from upstream model view')
            shutil.copy2(original_view,folder/'verified_global_view.png')
            manifest['files'][str((folder/'verified_global_view.png').relative_to(run))]={'source':str(original_view),'sha256':whole['sha256']}
            write(folder/'source_validation.json',dict(policy='provider_manifest_and_exact_rendered_pixel_match',declared_source_sha256=views['source_sha256'],actual_source_sha256=actual,global_view_sha256=whole['sha256']))
        normalize(read(folder/'handoff.json'),size,cid)
    write(Path(run)/'discovery_manifest.json',manifest)
    return ids

def verify(run):
    manifest=read(Path(run)/'discovery_manifest.json')
    for rel,v in manifest['files'].items():
        if sha(Path(run)/rel)!=v['sha256']:raise ValueError('Frozen discovery input changed: '+rel)
    return manifest

def box(value,size):
    if not isinstance(value,list) or len(value)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in value):raise ValueError('Invalid bbox')
    x1,y1,x2,y2=map(float,value)
    if x2<=x1 or y2<=y1 or x2<0 or y2<0 or x1>size[0] or y1>size[1]:raise ValueError('Empty/outside bbox')
    clipped=[max(0,x1),max(0,y1),min(float(size[0]),x2),min(float(size[1]),y2)]
    if clipped[2]<=clipped[0] or clipped[3]<=clipped[1]:raise ValueError('Box has no visible extent')
    return clipped

def reference_box(ref,view,size):
    if 'bbox_native' in ref:return box(ref['bbox_native'],size)
    # Reference boxes are in the upstream rendered view; target bbox_native is not.
    if list(view.get('original_image_size_px',[]))!=list(size):raise ValueError('Reference frame original size mismatch')
    a=view['local_to_global'];b=view['global_to_original'];v=ref['bbox'];xy=[]
    for i in (0,2):
        xy.extend([(v[i+k]*a['scale_xy'][k]+a['offset_xy'][k])*b['scale_xy'][k]+b['offset_xy'][k] for k in (0,1)])
    return box(xy,size)

def normalize(raw,size,cid):
    if raw.get('case_id')!=cid or not isinstance(raw.get('tasks'),list):raise ValueError('Handoff case/schema mismatch')
    tasks=[];ids=set();issues=[];context=copy.deepcopy(raw.get('context_regions',[]))
    for item in raw['tasks']:
        tid=item.get('target_id')
        if not isinstance(tid,str) or not tid or tid in ids:raise ValueError('Invalid/duplicate target ID')
        ids.add(tid)
        if item.get('render_owner',True) is False:
            context.append(copy.deepcopy(item));continue
        obj=copy.deepcopy(item);obj['id']=tid;obj['bbox_native']=box(item['bbox_native'],size)
        obj['locator']=semantic_text(item.get('identity_hint',''))
        obj['residual_hint']=semantic_text(item.get('residual_scope') or '')
        obj['readonly']=[]
        for ref in item.get('readonly_identities',[]):
            r=copy.deepcopy(ref)
            try:r['bbox_native']=reference_box(ref,item['view'],size)
            except (KeyError,ValueError,TypeError) as exc:
                issues.append(dict(stage='handoff_reference',target=tid,reference=ref.get('id'),error=str(exc)));continue
            r['locator']=semantic_text(ref.get('locator_hint',''));obj['readonly'].append(r)
        tasks.append(obj)
    active={t['id'] for t in tasks}
    edges={t['id']:{r.get('owner_id') for r in t['readonly'] if r.get('owner_id') in active} for t in tasks}
    def reaches(start,target):
        todo=[start];seen=set()
        while todo:
            current=todo.pop()
            if current==target:return True
            if current in seen:continue
            seen.add(current);todo.extend(edges.get(current,[])-seen)
        return False
    for t in tasks:
        for r in t['readonly']:
            owner=r.get('owner_id')
            safe=owner in active and owner!=t['id'] and not reaches(owner,t['id'])
            r['exclusion_verified']=safe
            if not safe:issues.append(dict(stage='handoff_ownership',target=t['id'],owner=owner,policy='context_only_no_exclusion',reason='missing owner, self reference or ownership cycle'))
    deferred=[]
    if not isinstance(raw.get('deferred_candidates',[]),list):raise ValueError('Deferred candidates must be a list')
    for item in raw.get('deferred_candidates',[]):
        tid=item.get('target_id')
        if not isinstance(tid,str) or not tid or tid in ids:raise ValueError('Invalid/duplicate deferred target ID')
        ids.add(tid)
        if item.get('task_role')!='missing_candidate' or item.get('render_owner') is not False:
            raise ValueError('Deferred candidates cannot own initial geometry')
        obj=copy.deepcopy(item);obj['id']=tid;obj['bbox_native']=box(item['bbox_native'],size)
        deferred.append(obj)
    return {'case_id':cid,'tasks':tasks,'deferred_candidates':deferred,
            'context_regions':context,'issues':issues,'source_count':len(raw['tasks'])+len(deferred)}

def load_case(cid,out):
    out=Path(out);run=out.parents[1];folder=run/'discovery_inputs'/cid
    source=Image.open(folder/'clean.png').convert('RGB');out.mkdir(parents=True,exist_ok=True)
    source.save(out/'clean.png');raw=read(folder/'handoff.json');normalized=normalize(raw,source.size,cid)
    write(out/'discovery_handoff.json',raw);write(out/'discovery_plan.json',normalized)
    upstream=read(folder/'upstream_result.json')
    write(out/'discovery_provenance.json',{'provider':'bbox_handoff','source':read(run/'discovery_manifest.json')['source_run'],
        'handoff_sha256':sha(folder/'handoff.json'),'native_size':list(source.size),'mask_identity_executed':False,
        'upstream_state':upstream.get('state'),'upstream_pending':{k:upstream.get(k,0) for k in ['pending_identity_count','pending_relation_count','post_sam_pending_count']}})
    return source,normalized

