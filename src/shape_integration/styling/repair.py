"""Reuse reviewed semantic styles and recover only numerical strokes and dot roles."""
import copy
from pathlib import Path
import time
import numpy as np
from PIL import Image
from shapely.geometry import box as shape_box
from shapely.ops import unary_union
from ..common import read,save,sha,digest,now
from .geometry import parts,raster
from .native_stroke import NativeFitter
from . import occlusion
from .export import export
from .width_limit import limit as limit_width


def run_case(scene,source,folder,runtime,options,pool,render_slots,*,assets=(),selected_objects=None,
             rules_only=False,pixel_slots=None,progress=lambda *a:None,line_evidence=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);source=Path(source);table=parts(scene)
    if selected_objects is not None:raise ValueError('Semantic repair uses whole cases; select cases with --ids')
    if sha(source)!=scene['source_sha256']:raise ValueError('Source does not match the geometry')
    root=Path(options['semantic_seed_root']);root=root/'cases' if (root/'cases').is_dir() else root
    seed=root/scene.get('case_id','unknown')/'styling'
    if read(seed/'scene_source.json')!=scene:raise ValueError('Seed style geometry differs from the current scene')
    seed_document=read(seed/'styles.json')
    if seed_document.get('state')!='finished':raise ValueError('Seed styles must be terminal')
    saved=seed_document['parts']
    if set(saved)!=set(table):raise ValueError('Seed style membership differs from the current scene')
    fitter=NativeFitter(source,table,line_evidence or options.get('sam_evidence_root'),scene.get('case_id','unknown'),folder)
    identity={'scene':digest(scene),'source':sha(source),'seed':sha(seed/'styles.json'),'relations':sha(seed/'occlusion.json'),
              'fitter':fitter.identity,'code':sha(Path(__file__)),'options':options,'mode':'semantic_preserving_native.1'}
    if (folder/'identity.json').exists() and read(folder/'identity.json')!=identity:raise ValueError('Repair identity changed; use a new run')
    save(folder/'identity.json',identity);save(folder/'scene_source.json',scene)
    if (folder/'summary.json').exists():
        result=read(folder/'summary.json')
        if result.get('styles_sha256')==sha(folder/'styles.json'):return result
    original={k:copy.deepcopy(v['style']) for k,v in saved.items()};styles=copy.deepcopy(original)
    relations=copy.deepcopy(read(seed/'occlusion.json'));geoms,_=occlusion.discover(table,options)
    records={};image=Image.open(source).convert('RGB');started=time.monotonic()
    asset_masks = ([] if options.get('image_ownership_stage') == 'composition' else
        [shape_box(*a['bbox_2d']) for a in assets if len(a.get('bbox_2d', [])) == 4])
    def blocked(geometry):
        if geometry.is_empty:return None
        b=geometry.bounds;box=[max(0,int(b[0])-1),max(0,int(b[1])-1),min(image.width,int(b[2])+2),min(image.height,int(b[3])+2)]
        if box[2]<=box[0] or box[3]<=box[1]:return None
        mask=raster(geometry,box)
        def lookup(xy):
            x=np.rint(xy[:,0]).astype(int)-box[0];y=np.rint(xy[:,1]).astype(int)-box[1]
            ok=(x>=0)&(x<mask.shape[1])&(y>=0)&(y<mask.shape[0]);result=np.zeros(len(xy),bool);result[ok]=mask[y[ok],x[ok]];return result
        return lookup
    for i,key in enumerate(reversed(relations['flat_order'])):
        hidden=unary_union([occlusion.exclusions(key,geoms,relations,styles),*[g for g in asset_masks if geoms[key].intersects(g)]])
        if pixel_slots:pixel_slots.acquire()
        try:styles[key],line=fitter.fit(key,original[key],excluded=blocked(hidden),visibility_id=digest(hidden.wkb_hex))
        finally:
            if pixel_slots:pixel_slots.release()
        if line.get('role')!='filled_compact' and styles[key].get('stroke'):
            if not line.get('width_reliable'):
                styles[key]['width_px']=original[key]['width_px'];line['kept_seed_width']=True
            if not line.get('dash_reliable'):
                styles[key]['dash']=original[key].get('dash','solid');styles[key].pop('dash_pattern_px',None)
                if original[key].get('dash_pattern_px'):styles[key]['dash_pattern_px']=original[key]['dash_pattern_px']
                line['kept_seed_pattern']=True
        if line.get('role')!='filled_compact' and any(styles[key].get(f)!=original[key].get(f) for f in ('fill','stroke')):
            raise AssertionError('A numerical repair changed semantic colours')
        styles[key], capped = limit_width(styles[key], image.width, options)
        if capped:line['width_limit'] = capped
        records[key]={'style':styles[key],'line':line,'state':'rules_only','origin':'semantic_seed_native_repair',
            'geometry_state':scene['objects'][table[key]['object_id']]['state'],
            'history':[{'stage':'semantic_seed','source':str(seed/'styles.json'),'style':original[key]},
                       {'stage':'native_numeric_repair','style':styles[key]}]}
        if i%10==0:progress('semantic_seed_native_fit',i+1,len(table))
    relations=occlusion.order_internal_by_paint(scene,table,relations,styles,geoms,options['geometry_tolerance_px'],source=image)
    save(folder/'occlusion.json',relations)
    export(scene,styles,relations,folder,source,options,render_slots)
    save(folder/'styles.json',{'schema_version':'shape_integration.styles.2','state':'finished','input':identity,
        'length_unit':'original_source_pixels','parts':records,'semantic_seed':str(seed),'visual_accuracy_verified':False})
    if sha(seed/'styles.json')!=identity['seed'] or sha(seed/'occlusion.json')!=identity['relations']:
        raise ValueError('Seed styles changed during repair')
    summary={'state':'rules_only','parts':len(table),'targeted_parts':len(table),'critic_accepted':0,'unresolved_parts':len(table),
        'renders':1,'model_calls':{},'source_colour_corrections':0,'native_fitted_strokes':sum(bool(r['line'].get('measured')) and bool(r['style']['stroke']) for r in records.values()),
        'filled_compact_parts':sum(r['line'].get('role')=='filled_compact' for r in records.values()),
        'styles_sha256':sha(folder/'styles.json'),'semantic_seed_sha256':sha(seed/'styles.json'),
        'semantic_colours_preserved_except_filled_dot_roles':True,'mode':'semantic_preserving_native.1',
        'wall_seconds':time.monotonic()-started,'updated_utc':now(),'visual_accuracy_verified':False}
    save(folder/'summary.json',summary);progress('completed',len(table),len(table));return summary
