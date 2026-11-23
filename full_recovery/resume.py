"""Register verified completed artifacts from this workflow's stopped run.

Only completed stages are linked read-only by their consumers. Interrupted stages
start in fresh directories so changed budgets/prompts cannot collide with traces.
"""
import json
from pathlib import Path
from .compose import sha,COMPOSITION_PROTOCOL


def stage_completed(source, target, repair_shapes=(), *, reuse_stages=('arrow','text','shape','composition')):
    source=Path(source).resolve();target=Path(target).resolve()
    read=lambda p:json.loads(p.read_text())
    old={i['id']:i for i in read(source/'source_manifest.json')['images']}
    items=read(target/'source_manifest.json')['images'];records=[]
    def link(src,dst,kind,ident):
        if dst.exists():raise ValueError('Resume destination already exists: '+str(dst))
        dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(src,target_is_directory=True)
        records.append({'case':ident,'stage':kind,'source':str(src),'destination':str(dst)})
    for item in items:
        ident=item['id']
        if old.get(ident)!=item:raise ValueError('Resume source image provenance changed: '+ident)
        if sha(item['original'])!=item['source_sha256']:raise ValueError('Original image changed')
        arrow=source/'arrows/cases'/ident
        if 'arrow' in reuse_stages and (arrow/'completed.json').exists() and (arrow/'arrows.pptx').exists():
            if read(arrow/'source_manifest.json')['image_sha256']!=item['source_sha256']:raise ValueError('Arrow source mismatch')
            link(arrow,target/'arrows/cases'/ident,'arrow',ident)
        text=source/'texts'/ident
        if 'text' in reuse_stages and (text/'summary.json').exists() and (text/'final/text.pptx').exists():
            if read(text/'final/provenance.json')['image_sha256']!=item['source_sha256']:raise ValueError('Text source mismatch')
            link(text,target/'texts'/ident,'text',ident)
        shape=source/'shape/geometry/cases'/ident
        shape_done='shape' in reuse_stages and (shape/'complete.json').exists() and ident not in repair_shapes
        if shape_done:
            receipt=read(shape/'complete.json')
            if sha(receipt['pptx'])!=receipt['pptx_sha256']:raise ValueError('Shape output changed')
            link(shape,target/'shape/geometry/cases'/ident,'shape',ident)
            link(source/'shape/delivery'/ident,target/'shape/delivery'/ident,'shape_delivery',ident)
            # Composition extracts photos/icons from the completed shape's
            # own localization registry, not from its style receipt.
            link(source/'shape/localization/cases'/ident,target/'shape/localization/cases'/ident,'shape_localization',ident)
        whole=source/'cases'/ident
        if shape_done and 'composition' in reuse_stages and (whole/'completed.json').exists():
            receipt=read(whole/'completed.json')
            if receipt.get('composition_protocol')!=COMPOSITION_PROTOCOL:continue
            if sha(whole/'combined.pptx')!=receipt['pptx_sha256']:raise ValueError('PPT changed')
            link(whole,target/'cases'/ident,'composition',ident)
            link(source/'delivery'/ident,target/'delivery'/ident,'delivery',ident)
    result={'source':str(source),'source_status':read(source/'status.json'),
            'only_own_completed_checkpoints':True,'incomplete_stages_restarted':True,
            'repair_shapes':list(repair_shapes),'artifacts':records}
    result['reuse_stages']=list(reuse_stages)
    (target/'continuation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result
