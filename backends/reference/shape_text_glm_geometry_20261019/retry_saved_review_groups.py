"""Explicitly review selected saved crops and merge validated occurrence updates.

Uses the shared normative prompt, preserves original inputs and complete request
history, and never retries a network timeout automatically.
"""
import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'.deps'),str(ROOT/'src'),str(ROOT/'vendor')]
from PIL import Image
from shape_integration.common import Runtime,read,save,sha,usage,valid_box
from shape_integration.texts import options
from shape_integration.texts.controller import mapped_box
from shape_integration.texts.local_content import validate,validate_pixel_lines,collisions,FIELDS,STYLE_FIELDS
from shape_integration.texts.prompt_protocol import review,SYSTEM


def merge_result(rows,source_rows,result,frame):
    proposed=[];resolved=set();removed=set();unresolved=[]
    originals={r['id']:r for r in source_rows}
    for item in result['items']:
        ident=item['id'];original=originals[ident]
        if item['status']=='excluded_image_text':
            removed.add(ident);resolved.add(ident);continue
        if item['status'] not in ('located','split'):
            unresolved.append(ident);continue
        children=item['parts'] if item['status']=='split' else [item]
        for part in children:
            row=deepcopy(original)
            row.update({k:deepcopy(part[k]) for k in (*FIELDS,*STYLE_FIELDS) if k in part})
            row.update(id=part['id'],bbox_2d=mapped_box(part['bbox_2d'],frame,True))
            if not valid_box(row['bbox_2d'],frame['source_size']):raise ValueError('Mapped box outside source')
            if item['status']=='split':
                row.update(parent_id=ident,parent_geometry_prior_bbox=original['bbox_2d'])
            else:row['geometry_prior_bbox']=original['bbox_2d']
            row.setdefault('observations',[]).append({'origin':'explicit_saved_group_review','local':part})
            proposed.append(row)
        removed.add(ident);resolved.add(ident)
    retained=[r for r in rows if r['id'] not in removed and r.get('parent_id') not in removed]
    ids={r['id'] for r in proposed}
    if len(ids)!=len(proposed):raise ValueError('Duplicate proposed occurrence IDs')
    conflicts=[pair for pair in collisions([*retained,*proposed]) if ids.intersection(pair)]
    if conflicts:raise ValueError('Review introduced occurrence collisions: '+str(conflicts))
    return [*retained,*proposed],resolved,unresolved


def verify_source(image_path,review_root):
    previous_image=review_root.parent.parent/'original.png'
    if previous_image.is_file():
        with Image.open(previous_image) as old,Image.open(image_path) as new:
            if old.size!=new.size or old.convert('RGB').tobytes()!=new.convert('RGB').tobytes():
                raise ValueError('Source pixels differ from saved review')
        return
    for ancestor in review_root.parents:
        manifest=ancestor/'manifest.json'
        if not manifest.is_file():continue
        items={i['id']:i for i in read(manifest).get('images',[])}
        ident=next((p.name for p in review_root.parents if p.name in items),None)
        if ident:
            if items[ident]['source_sha256']!=sha(image_path):raise ValueError('Source hash differs from batch manifest')
            return
    raise ValueError('Cannot verify source identity from saved review or batch manifest')


def pending_ids(initial, fallback):
    """Continue the current recognition state, not the original failed batch."""
    if 'remaining_unresolved_ids' in initial:
        return set(initial['remaining_unresolved_ids'])
    if 'errors' not in initial:
        return set(fallback)
    pending=set()
    for error in initial['errors']:
        if error.startswith('Unresolved local transcription:'):
            values=ast.literal_eval(error.split(':',1)[1].strip())
            if not isinstance(values,list) or any(not isinstance(v,str) for v in values):
                raise ValueError('Invalid unresolved transcription IDs')
            pending.update(values)
    return pending


def verify_response_root(folder, image, input_rows):
    """Cached responses must originate from these exact source pixels/targets."""
    sources=read(folder/'provenance.json')['sources']
    for path in (image,input_rows):
        if sources.get(str(path.resolve()))!=sha(path):
            raise ValueError('Saved response provenance mismatch: '+str(path))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review-root',type=Path,required=True)
    p.add_argument('--recognition',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--groups',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--offline',action='store_true',help='Only revalidate complete saved responses; never call the API')
    p.add_argument('--response-root',type=Path,action='append',default=[],help='Additional source-verified selected-review cache')
    args=p.parse_args()
    if args.output.exists():raise ValueError('Use a fresh output directory')
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    cfg=read(ROOT/'config.json');cfg['api_concurrency']=1
    key=os.environ.get('OpenAI_API') or os.environ.get('OPENAI_API_KEY') or os.environ.get('BOYUE_API_KEY')
    if not key and not args.offline:raise ValueError('API key must be supplied by environment')
    runtime=None if args.offline else Runtime(cfg,cfg['expected_boyue_base_url'],key).for_stage(reasoning_effort='low',shape_schema_retries=1)
    opt=options(cfg);initial=read(args.recognition);rows=deepcopy(initial['texts'])
    originals={r['id']:r for r in read(args.review_root/'input_rows.json')}
    image=Image.open(args.image).convert('RGB')
    verify_source(args.image,args.review_root)
    for folder in args.response_root:
        verify_response_root(folder,args.image,args.review_root/'input_rows.json')
    inherited_pending=pending_ids(initial,read(args.review_root/'summary.json').get('unresolved_ids',[]))
    sources=[args.recognition,args.image,args.review_root/'input_rows.json']
    before={str(f.resolve()):sha(f) for f in sources}
    args.output.mkdir(parents=True)
    save(args.output/'provenance.json',{'sources':before,'groups':args.groups,
         'response_roots':[str(p.resolve()) for p in args.response_root],
         'script_sha256':sha(__file__),'policy':'explicit sequential selected-group review; no timeout retry'})
    records=[];resolved=set();successful=[];failures=[];ambiguous=set()
    for name in args.groups:
        if Path(name).name!=name:raise ValueError('Invalid group name')
        folder=args.review_root/name;prior=read(folder/'prompt.json');frame=read(folder/'input/frame.json')
        group=[originals[t['id']] for t in prior['targets']]
        prompt=review(frame['size'],prior['targets'],prior.get('context_read_only',[]))
        images=[('CLEAN source crop; all coordinates refer here.',folder/'input/clean.png'),
                ('Fallible OCR occurrence anchors; labels are not source content.',folder/'input/ids.png')]
        if (folder/'input/directions.png').exists():
            images.append(('Rotation evidence only, not bbox coordinates.',folder/'input/directions.png'))
        def validator(value):
            checked=validate_pixel_lines(validate(value,group,frame['size']),group,frame,image)
            merge_result(rows,group,checked,frame)
            return checked
        try:
            result=None;rejections=[];saved_source=None
            cached=list((folder/'calls').rglob('parsed_response.json'))
            for cache in args.response_root:
                cached.extend((cache/'calls'/name).rglob('parsed_response.json'))
            for path in sorted(cached,key=lambda p:p.stat().st_mtime,reverse=True):
                status_path=path.parent/'stream_status.json'
                if not status_path.is_file():continue
                status=read(status_path)
                if status.get('finish_reason')!='stop' or not status.get('sse_done'):continue
                try:result=validator(read(path));saved_source=path;break
                except (ValueError,KeyError,TypeError) as exc:
                    rejections.append({'path':str(path),'error':str(exc)})
            if result is None:
                if args.offline:raise ValueError('No complete saved response passes validation: '+str(rejections))
                result=runtime.call(args.output/'calls'/name,'review',SYSTEM,prompt,images,validator,opt['max_tokens'])
            rows,updated,unresolved=merge_result(rows,group,result,frame)
            resolved.update(updated);ambiguous.update(unresolved);successful.append(name)
            record={'group':name,'state':'validated','resolved_ids':sorted(updated),'unresolved_ids':unresolved}
            if saved_source:record.update(saved_response=str(saved_source.resolve()),saved_response_sha256=sha(saved_source),api_calls=0)
            save(args.output/'validated_groups'/(name+'.json'),result)
        except Exception as exc:
            record={'group':name,'state':'failed','error':str(exc)};failures.append(name+': '+str(exc))
        records.append(record)
        # Earlier unresolved-ID warnings remain auditable; only successfully
        # replaced group transport errors and their resolved IDs are removed.
        errors=[e for e in initial.get('errors',[]) if not any(e.startswith(g+':') for g in successful)
                and not e.startswith('Unresolved local transcription:')]
        remaining=(inherited_pending-resolved)|ambiguous
        if remaining:errors.append('Unresolved local transcription: '+str(sorted(remaining)))
        errors.extend(failures)
        save(args.output/'recognition_refined.json',{'texts':rows,'errors':errors,'inherited_errors':initial.get('errors',[]),
             'remaining_unresolved_ids':sorted(remaining)})
        save(args.output/'summary.json',{'groups':records,'resolved_ids':sorted(resolved),
             'remaining_unresolved_ids':sorted(remaining),'errors':errors,'visual_accuracy_verified':False})
        save(args.output/'usage.json',usage(args.output/'calls'))
        print(name,record['state'],flush=True)
    assert before=={str(f.resolve()):sha(f) for f in sources}


if __name__=='__main__':main()
