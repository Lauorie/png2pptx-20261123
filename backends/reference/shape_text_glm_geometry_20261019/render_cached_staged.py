"""Render a staged inventory and saved formula replies, strictly without APIs."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import threading

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'.deps'),str(ROOT/'src'),str(ROOT/'vendor')]
from PIL import Image
from shape_integration.common import read,save,sha
from shape_integration.texts import options,layout,export
from shape_integration.texts.staged_formula import formula_validate
from shape_integration.texts.formula_state import adopt,keep_previous
from shape_integration.texts.numeric_routing import inventory as route_inventory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('image','cache-run','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    source=args.image.resolve();cache=args.cache_run.resolve();output=args.output.resolve()
    inventory=read(cache/'staged/inventory.json')
    if inventory['source_sha256']!=sha(source):raise ValueError('Cached source does not match input')
    if output.exists():raise ValueError('Use a fresh output directory')
    output.mkdir(parents=True)
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    rows=deepcopy(inventory['texts']);by_id={r['id']:r for r in rows}
    accepted={};rejected=[];warnings=[]
    for group in sorted((cache/'staged/formulas').glob('F*')):
        clean_hash=sha(group/'input/clean.png')
        for response in sorted((group/'calls').glob('request_*/content.txt'),reverse=True):
            status=read(response.parent/'stream_status.json')
            identity=read(response.parent/'identity.json')
            if status.get('http_status')!=200 or status.get('finish_reason')!='stop' or not status.get('sse_done'):continue
            if clean_hash not in [im['sha256'] for im in identity['images']]:
                raise ValueError('Cached formula crop hash mismatch')
            targets={r['id'] for r in identity['prompt']['targets']}
            try:
                content=response.read_text().strip()
                if content.startswith('```') and content.endswith('```'):content='\n'.join(content.splitlines()[1:-1])
                items=json.loads(content)['items']
                for ident in targets:
                    if ident in accepted or ident not in by_id:continue
                    candidates=[i for i in items if isinstance(i,dict) and i.get('id')==ident]
                    try:
                        item=formula_validate({'items':candidates},[by_id[ident]])['items'][0]
                        if item['status']!='resolved':continue
                        updated=adopt(by_id[ident],item)
                        by_id[ident].update(updated)
                        accepted[ident]={'response':str(response),'sha256':sha(response)}
                    except (ValueError,TypeError,KeyError) as exc:
                        rejected.append({'id':ident,'response':str(response),'reason':str(exc)})
            except (ValueError,TypeError,KeyError) as exc:
                rejected.append({'response':str(response),'reason':str(exc)})
    for row in rows:
        if row['kind']=='formula' and row['id'] not in accepted:
            row.update(keep_previous(row,'No valid saved formula reply'))
            warnings.append({'id':row['id'],'reason':'Formula rendered using explicit fallback'})
        if row['kind']!='formula' and row.get('visual_line_count',1)>1:
            warnings.append({'id':row['id'],'reason':'Cached multiline object retained; no new line-bbox request'})
    routed=route_inventory({**inventory,'texts':rows},'cached_replay_before_layout')
    rows=routed['texts']
    save(output/'recognition.json',routed)
    save(output/'cache_replay.json',{'source_run':str(cache),'source_sha256':sha(source),
        'inventory_sha256':sha(cache/'staged/inventory.json'),'new_api_calls':0,
        'formula_replies':accepted,'rejected':rejected,'warnings':warnings,
        'policy':'latest_valid_cached_reply; no new GLM, OCR or color re-estimation',
        'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),*ROOT.joinpath('src').rglob('*.py')]}})
    opt=options(read(ROOT/'config.json'))
    opt.update(pixel_reflow_enabled=False,source_font_selection=False,glyph_registration_enabled=False)
    slots=threading.BoundedSemaphore(1)
    final=output/'final';final.mkdir()
    print('Cached inventory:',len(rows),'Formulas:',len(accepted),'Warnings:',len(warnings),flush=True)
    fitted,errors=layout.fit_all(rows,final/'typography',slots,opt,source=source)
    counts=export.final(fitted,source,final,slots,opt)
    with Image.open(source) as im:size=list(im.size)
    save(final/'scene.json',{'texts':fitted,'source_size':size,'errors':errors})
    summary={'counts':counts,'new_api_calls':0,'warnings':warnings,'errors':errors,
             'stage':'cached_native_render_complete','visual_accuracy_verified':False}
    save(output/'summary.json',summary)
    print(json.dumps({'stage':summary['stage'],'texts':len(fitted),'formulas':len(accepted),'new_api_calls':0}),flush=True)


if __name__=='__main__':main()
