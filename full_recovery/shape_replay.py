"""Replay native layer ordering while preserving fitted geometry and paint.

Completed observations and semantic styles are read-only inputs. Only a changed
native order needs a new render. This avoids rerunning every style critic for
geometry that has already been recovered.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import shutil
import threading
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree

from .compose import P,A,SLIDE,sha,xml_bytes
from .merge_existing import read,save,inspect_sources,execute


def reorder_native(source, target, scene, order):
    refs={scene['parts'][k]['object_id']+'/'+scene['parts'][k]['id']:i for i,k in enumerate(order)}
    with ZipFile(source) as archive:package={n:archive.read(n) for n in archive.namelist()}
    root=etree.fromstring(package[SLIDE]);changes=[]
    def ranks(node):
        names=node.xpath('.//*[local-name()="cNvPr"]/@name')
        return [refs[n.split(' [',1)[0]] for n in names if n.split(' [',1)[0] in refs]
    for group in root.iter('{'+P+'}grpSp'):
        children=[n for n in group if etree.QName(n).localname not in ('nvGrpSpPr','grpSpPr','extLst')]
        if len(children)<2:continue
        owners={n.split('/',1)[0] for child in children for n in child.xpath('.//*[local-name()="cNvPr"]/@name')
                if n.split(' [',1)[0] in refs}
        if len(owners)!=1:continue
        original={n:i for i,n in enumerate(children)}
        # Unknown leaves keep their relative slot, instead of moving a native
        # annotation or an unregistered part on the strength of a guessed ID.
        known=[n for n in children if ranks(n)]
        sorted_known=iter(sorted(known,key=lambda n:min(ranks(n))))
        ordered=[next(sorted_known) if n in known else n for n in children]
        if ordered==children:continue
        positions=[group.index(n) for n in children]
        for n in children:group.remove(n)
        for index,n in zip(positions,ordered):group.insert(index,n)
        changes.append({'owner':next(iter(owners)),'parts_reordered':len(known)})
    if changes:
        package[SLIDE]=xml_bytes(root)
        with ZipFile(target,'w',ZIP_DEFLATED) as archive:
            for n,data in package.items():archive.writestr(n,data)
    else:shutil.copy2(source,target)
    return changes


def replay(record, destination, render_slots):
    from shape_integration.styling import occlusion
    from shape_integration.styling.export import render_native
    item=record['item'];paths=record['paths'];cid=item['id'];target=destination/'shape/geometry/cases'/cid
    if (target/'complete.json').exists():return read(target/'complete.json')
    for key,path in paths.items():
        if sha(path)!=record['sha256'][key]:raise ValueError('Source branch changed: '+cid+'/'+key)
    scene=read(paths['shape_scene']);doc=read(paths['shape_styles']);styles={k:v['style'] for k,v in doc['parts'].items()}
    table={k:p for k,p in scene['parts'].items() if k in styles and p.get('status')=='ok'}
    source_style=Path(paths['shape_styles']).parent
    rel_path=source_style/'occlusion.json'
    geoms,discovered=occlusion.discover(table,{'geometry_tolerance_px':.2,'minimum_overlap_area_px':1.})
    relations=read(rel_path) if rel_path.exists() else discovered
    if relations.get('scope')!='components':relations=discovered
    relations=occlusion.order(scene,table,relations)
    relations=occlusion.order_internal_by_paint(scene,table,relations,styles,geoms,source=item['original'])
    fitted=target/'fitting';styled=target/'styling';styled.mkdir(parents=True,exist_ok=True)
    fitted.mkdir(parents=True,exist_ok=True)
    shutil.copy2(paths['shape_scene'],fitted/'scene.json')
    changes=reorder_native(paths['shape_pptx'],styled/'styled.pptx',scene,relations['flat_order'])
    if changes:
        with render_slots:render_native(styled/'styled.pptx',styled/'render',scene['source_size'],180)
    else:
        shutil.copy2(paths['shape_preview'],styled/'native.png')
        if (source_style/'native.pdf').exists():shutil.copy2(source_style/'native.pdf',styled/'native.pdf')
    doc=deepcopy(doc);doc['native_layer_replay']={'source':paths['shape_styles'],'source_sha256':record['sha256']['shape_styles'],
        'geometry_and_paint_preserved':True,'changes':changes,'new_semantic_style_calls':0}
    save(styled/'styles.json',doc);save(styled/'occlusion.json',relations)
    save(destination/'shape/localization/cases'/cid/'dedup/image_boxes_original.json',read(paths['image_registry']))
    done={'id':cid,'state':'completed_with_unresolved','geometry_sha256':sha(fitted/'scene.json'),
        'pptx_sha256':sha(styled/'styled.pptx'),'source_branch':record,'native_order_changes':changes,
        'geometry_and_paint_preserved':True,'new_semantic_style_calls':0,'visual_accuracy_verified':False}
    save(target/'complete.json',done);print('SHAPE_ORDER_REPLAYED',cid,len(changes),flush=True)
    return done


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);args=p.parse_args()
    run=args.run_dir.resolve();settings=read(run/'settings.json');records=read(run/'replay_sources.json')['ready']
    destination=run/'refreshed_shape';save(destination/'source_manifest.json',{'images':[r['item'] for r in records]})
    slots=threading.BoundedSemaphore(2)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda r:replay(r,destination,slots),records))
    ready,missing=inspect_sources(destination,Path(settings['source_text_run']),Path(settings['source_arrow_run']))
    if missing or len(ready)!=settings['requested']:raise ValueError('Replay did not bind every requested source')
    save(run/'sources.json',{'ready':ready,'missing':missing,'sam_requests':0,'new_arrow_recovery':False})
    execute(run)


if __name__=='__main__':main()
