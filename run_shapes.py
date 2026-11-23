"""One resumable shape + style workflow, from original PNGs to editable PPTX.

Prepare on the submit host; run inference, pixel fitting and rendering in rjob.
No historical shape/image detections, fitted scenes or semantic seeds are used.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ['SHAPE_REFERENCE_ROOT'] = str(ROOT)
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'vendor'), str(ROOT/'experiment/shape_sam_guided')]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def input_records(dataset, requested_ids=None):
    """Read the existing taskset format or a bounded package catalog."""
    dataset = Path(dataset).resolve()
    taskset = dataset/'taskset.txt'
    if taskset.is_file():
        ids = taskset.read_text().split()
        if requested_ids:
            unknown = set(requested_ids)-set(ids)
            if unknown: raise ValueError('Unknown task IDs: '+','.join(sorted(unknown)))
            ids = [ident for ident in ids if ident in set(requested_ids)]
        return [(ident, dataset/'tasks'/ident/'reference.png') for ident in ids]

    catalogs = [dataset/'catalog.json']
    catalogs.extend(path/'catalog.json' for path in sorted(dataset.iterdir()) if path.is_dir())
    catalog_path = next((path for path in catalogs if path.is_file()), None)
    if catalog_path is None:
        raise ValueError('Input root needs taskset.txt or a package catalog.json')
    catalog = read(catalog_path)
    rows = catalog.get('items') if isinstance(catalog, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError('Package catalog needs a nonempty items list')
    first = rows[0]
    relative = first.get('image') if isinstance(first, dict) else None
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('Catalog image paths must be relative to the package root')
    roots = [dataset, catalog_path.parent, dataset.parent]
    package_root = next((root.resolve() for root in roots if (root/relative).is_file()), None)
    if package_root is None:
        raise FileNotFoundError('Catalog image is missing: '+relative)

    records = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('image'), str):
            raise ValueError('Catalog item needs an image path')
        ident = row.get('case_key') or row.get('id_padded') or row.get('id')
        if ident is None: raise ValueError('Catalog item needs a stable case identifier')
        path = Path(row['image'])
        if path.is_absolute(): raise ValueError('Catalog image paths must be relative')
        resolved = package_root/path
        if not resolved.is_file(): raise FileNotFoundError(str(resolved))
        records.append((str(ident), resolved))
    ids = [ident for ident, _ in records]
    if len(set(ids)) != len(ids): raise ValueError('Catalog case identifiers must be unique')
    if requested_ids:
        unknown = set(requested_ids)-set(ids)
        if unknown: raise ValueError('Unknown catalog cases: '+','.join(sorted(unknown)))
        selected = set(requested_ids)
        records = [record for record in records if record[0] in selected]
    return records


def prepare(args):
    """Freeze only executable dependencies and an original-input manifest."""
    from PIL import Image
    run = args.run_dir.resolve(); dataset = args.input_root.resolve()
    if run.exists(): raise ValueError('Use a new run directory: '+str(run))
    records = input_records(dataset, args.ids)
    items = []
    for ident, path in records:
        with Image.open(path) as image: size = list(image.size)
        items.append({'id': ident, 'original': str(path), 'source_size': size, 'source_sha256': sha(path)})
    settings = read(ROOT/'experiment/shape_sam_guided/settings.recommended.json')
    settings.update(project_root=str(ROOT), input_root=str(dataset), case_workers=2,
                    namespace='ailab-llmmultimodal',charged_group='llmmultimodal_gpu_pool',
                    worker_cpu=32,
                    glm_tp=4, glm_max_tokens=100000, glm_concurrency=12,
                    glm_request_timeout_seconds=3600,
                    sam_cache_only=False, reuse_sam_indices=[], sam_cache_run=None,
                    worker_max_seconds=86400, idle_stop_seconds=1800,
                    sam_python='/data/dongdaozhu/conda/envs/sam3/bin/python',
                    glm_ports=[10088,10089], glm_max_model_len=262144, glm_max_num_seqs=24,
                    sam_checkpoint_sha256='1a1df9b2d12bb57a6c20038171ef6f1c2d831b9064f7af3aff11096d8f26f285',
                    maximum_stroke_width_pt=args.max_stroke_pt)
    settings['sam_checkpoint'] = str(Path(settings['training_run'])/'checkpoints/best.pt')
    config = read(ROOT/'config.json')
    config.update(case_workers=2, api_concurrency=12, fitting_enabled=True,
                  styling_enabled=False, arrows_enabled=False, texts_enabled=False, image_export_enabled=False)
    config['fitting'].update(max_revisions=1, workers=24, recover_clipped_partial=True, geometry_guard_enabled=True,
                             sam_radius_refinement=True,conditional_geometry_critic=True,projected_edge_refinement=True,
                             source_frame_refinement=True,
                             source_span_checks=False)
    config['styling'].update(maximum_stroke_width_pt=args.max_stroke_pt, minimum_stroke_width_px=0.,
                             workers=16, case_workers=2, pixel_workers=2, render_workers=2,
                              color_roles_enabled=True, max_tokens=100000,conditional_pattern_critic=True,
                              sam_visual_pattern=True)
    for name in ('jobs','services/logs','delivery'): (run/name).mkdir(parents=True, exist_ok=True)
    snapshot = run/'code'
    for directory in ('src','png2pptx','vendor','inputs','full_recovery'):
        shutil.copytree(ROOT/directory, snapshot/directory, symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for relative in ('run_shapes.py','config.json','scripts/shape_worker.py','tests'):
        src = ROOT/relative; dst = snapshot/relative; dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir(): shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__','*.pyc','tmp*','.pytest_cache'))
        else: shutil.copy2(src,dst)
    for src in ROOT.glob('run*.py'):shutil.copy2(src,snapshot/src.name)
    for directory in ('shape_sam_guided','style_line_sam'):
        target = snapshot/'experiment'/directory; target.mkdir(parents=True,exist_ok=True)
        for src in (ROOT/'experiment'/directory).glob('*.py'): shutil.copy2(src,target/src.name)
    shutil.copytree(ROOT/'experiment/style_line_sam/service',snapshot/'experiment/style_line_sam/service',
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # Regression controls import only these pure point/mask helpers. Their
    # optional API experiment imports stay inside main(), outside production.
    controls=snapshot/'experiments/final_shape_critic_20261020';controls.mkdir(parents=True,exist_ok=True)
    for name in ('native_corner_probe.py','native_corner_mask_fit.py'):
        shutil.copy2(ROOT/'experiments/final_shape_critic_20261020'/name,controls/name)
    files = {str(p.relative_to(snapshot)):sha(p) for p in sorted(snapshot.rglob('*')) if p.is_file() and not p.is_symlink()}
    save(run/'source_manifest.json',{'dataset':str(dataset),'images':items,'reference_png_only':True})
    save(run/'settings.json',settings); save(run/'pipeline_config.json',config)
    save(run/'code_sha256.json',files)
    save(run/'status.json',{'phase':'prepared','expected':len(items),'at':time.time()})
    print(json.dumps({'run':str(run),'samples':len(items),'max_stroke_pt':args.max_stroke_pt}))


def execute(run, *, resume_settings=None, selected_ids=None, report_root=None):
    if os.environ.get('STYLE_CLUSTER_WORKER') != '1': raise RuntimeError('Execute through scripts/submit_shapes.sh')
    from PIL import Image
    from models import Models
    from runtime import ForwardRuntime
    from evidence import frame_crop, normalize_predictions, group_duplicates
    from core import public_proposal
    from shape_integration import image_bridge, batch
    from shape_integration.shape import discover, run_shape
    from shape_integration.sam_fixed import FixedSamEvidence
    from shape_integration.pipeline import run_case, source_manifest
    from shape_integration.styling import options as style_options
    from shape_integration.styling.controller import run_case as style_case
    from shape_integration.styling.width_limit import SLIDE_WIDTH_PT
    sys.path.insert(0,str(ROOT/'experiment/style_line_sam'))
    from sam_tiles import run_case as line_evidence, STYLE_CATEGORIES
    import fitz
    import zipfile
    from lxml import etree

    class FreshRuntime(ForwardRuntime):
        def client(self, folder, budget):
            owner = self
            class Detector:
                root = Path(folder)
                base_url = owner.models.config['glm_url']
                def request_parameters(self):
                    return {'model':'GLM-5.3-Flash','max_tokens':min(budget,owner.models.config['glm_max_tokens']),
                            'temperature':0,'seed':owner.models.config['seed'],
                            'thinking':owner.config.get('thinking','on'),
                            'reasoning_effort':owner.config.get('reasoning_effort',owner.models.config.get('reasoning_effort','high'))}
                def call(self, tag, system, prompt, images, validator=None):
                    target = self.root/tag
                    try:
                        value = owner.call(target,'detect',system,prompt,images,validator or (lambda x:x),budget)
                    except (RuntimeError,ValueError) as exc:
                        save(target/'error_0.json',{'error':str(exc),'type':type(exc).__name__});raise
                    save(target/'result.json',value)
                    return value
            return Detector()

    settings = read(run/'settings.json'); config = read(run/'pipeline_config.json')
    if resume_settings is not None:
        overrides = read(resume_settings)
        allowed = {'case_slot_config', 'sam_url', 'sam_urls', 'case_workers', 'glm_concurrency',
                   'fitting_workers', 'styling_workers', 'checkpoint_path_root',
                   'resume_finished_geometry', 'resume_transport_cases_once'}
        if set(overrides) - allowed:
            raise ValueError('Resume may only override scheduling/service settings')
        settings.update(overrides)
        if 'glm_concurrency' in overrides:
            limit = overrides['glm_concurrency']
            if type(limit) is not int or limit < 1:
                raise ValueError('Positive API concurrency required')
            config['api_concurrency'] = limit
    report_root = Path(report_root) if report_root is not None else run
    report_root.mkdir(parents=True, exist_ok=True)
    settings['sam_output_root']=str(run)
    # Partition views have private aggregate manifests, but use canonical input
    # and SAM paths so an unchanged styling checkpoint keeps its exact identity.
    checkpoint_root = Path(settings.get('checkpoint_path_root', run))
    owner = (run/'.pipeline.lock').open('a'); fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    items = []; manifest = read(run/'source_manifest.json')
    if selected_ids is not None:
        selected = set(read(selected_ids))
        if not selected <= {item['id'] for item in manifest['images']}:
            raise ValueError('Resume selection contains unknown shape cases')
        manifest = {**manifest, 'images': [item for item in manifest['images'] if item['id'] in selected]}
    for original in manifest['images']:
        path = Path(original['original'])
        if sha(path) != original['source_sha256']: raise ValueError('Original changed: '+str(path))
        dest = checkpoint_root/'inputs'/(original['id']+'.png'); dest.parent.mkdir(exist_ok=True)
        if not dest.exists():
            with Image.open(path) as image:
                ratio = 2048/max(image.size)
                image.convert('RGB').resize(tuple(max(1,round(v*ratio)) for v in image.size),Image.Resampling.LANCZOS).save(dest)
        with Image.open(dest) as image: size = list(image.size)
        items.append({**original,'input':str(dest),'input_size':size,'input_sha256':sha(dest)})
    for stage in ('localization','geometry'):
        save(run/stage/'manifest.json',{'images':items,'config':config,'localization_source':str(run/'localization')})
    models = Models(settings); runtime = FreshRuntime(models,config,None)
    image_bridge.configure(runtime)
    render_slots = threading.BoundedSemaphore(2); pixel_slots = threading.BoundedSemaphore(2)
    results = {}; state_lock = threading.Lock(); pdf_lock = threading.Lock()

    def progress():
        save(report_root/'progress.json',{'done':len(results),'expected':len(items),
             'failed':sum(r.get('state')=='failed' for r in results.values()),'cases':results,'at':time.time()})

    def process_unlimited(item, fit_pool, style_pool, sam_pool):
        ident = item['id']; case = run/'geometry/cases'/ident; receipt = case/'complete.json'
        if receipt.exists():
            row = read(receipt)
            if sha(Path(row['pptx'])) != row['pptx_sha256']: raise ValueError('Completed output changed')
            return row
        local = run/'localization/cases'/ident; local.mkdir(parents=True,exist_ok=True)
        began = time.monotonic()
        save(case/'stage.json',{'stage':'discovery_and_images','at':time.time()})
        if not (local/'summary.json').exists():
            with ThreadPoolExecutor(max_workers=2) as first_pool:
                seed = first_pool.submit(discover,Path(item['original']),Path(item['input']),local/'shapes',runtime)
                assets_job = first_pool.submit(image_bridge.detect,Path(item['original']),local/'images',config)
                _, plan, _ = seed.result()
                # SAM depends on the layout, not image-asset review; overlap
                # the same evidence calls while retaining the ownership barrier.
                cache = run/'sam_geometry'/ident; index_path = cache/'index.json'
                if index_path.exists(): index = read(index_path)
                else:
                    with Image.open(item['original']) as opened: source = opened.convert('RGB')
                    views = [('full',[0,0,source.width-1,source.height-1])]
                    views += [(r['id'],[v*item['source_size'][i%2]/item['input_size'][i%2]
                              for i,v in enumerate(r['bbox_2d'])]) for r in plan['regions']]
                    def infer(view):
                        name,box = view; png,frame = frame_crop(source,box,cache/name)
                        pred = models.sam(png,cache/name/'masks')
                        return normalize_predictions(pred,frame,name,source.size)
                    predictions = []
                    for rows in sam_pool.map(infer,views): predictions.extend(rows)
                    groups = group_duplicates(predictions)
                    index = {'source_sha256':item['source_sha256'],'candidate_threshold':settings['sam_threshold'],
                             'groups':[{'id':g['id'],'representative':public_proposal(g['representative'])} for g in groups]}
                    save(index_path,index)
                inventory = assets_job.result()
            from png2pptx.frames import scale_box
            assets = [{**a,'bbox_2d':scale_box(a['bbox'],inventory['input_size'],item['source_size'])} for a in inventory['assets']]
            provider = FixedSamEvidence(index['groups'],assets,mode='critic',precheck=False,**settings['evidence'])
            current = FreshRuntime(models,config,None,provider)
            shapes = run_shape(Path(item['original']),Path(item['input']),local/'shapes',current)
            batch.finish_case(local,Path(item['original']),item,shapes,inventory,{},config,current)
        else:
            index = read(run/'sam_geometry'/ident/'index.json')
            assets = read(local/'dedup/image_boxes_original.json')['assets']
            provider = FixedSamEvidence(index['groups'],assets,mode='critic',precheck=False,**settings['evidence'])
            current = FreshRuntime(models,config,None,provider)
        save(case/'stage.json',{'stage':'geometry','at':time.time()})
        provenance = source_manifest(run/'localization',[item])
        scene_path = case/'fitting/scene.json';summary_path=case/'fitting/summary.json'
        cached_scene=read(scene_path) if scene_path.exists() else None
        finished_geometry=(settings.get('resume_finished_geometry',False) and cached_scene is not None
            and summary_path.exists() and read(summary_path).get('state') in ('completed','completed_with_unresolved')
            and cached_scene['source_sha256']==item['source_sha256']
            and not any(o.get('stop_reason') in ('observation_transport_failed','check_transport_failed')
                        for o in cached_scene['objects'].values()))
        geometry_summary=(read(summary_path) if finished_geometry else
            run_case(ROOT,run/'geometry',item,current,config,fit_pool,run/'localization',provenance,False))
        scene = read(scene_path); scene_hash = sha(scene_path)
        service_failures=[oid for oid,obj in scene['objects'].items() if obj.get('stop_reason') in
                          {'observation_transport_failed','check_transport_failed'}]
        if service_failures:
            raise RuntimeError('Incomplete model transport; resume this case after service recovery: '+', '.join(service_failures))
        save(case/'stage.json',{'stage':'native_sam_crops','at':time.time()})
        line_evidence({'id':ident,'original':item['original'],'scene':str(scene_path)},
                      {'side':96,'prompts':'both','categories':list(STYLE_CATEGORIES),
                       'score_threshold':.5},checkpoint_root/'sam_styles',models.sam,sam_pool)
        save(case/'stage.json',{'stage':'styling','at':time.time()})
        opt = style_options(config); opt['sam_evidence_root'] = str(checkpoint_root/'sam_styles')
        style_summary = style_case(scene,Path(item['original']),case/'styling',current,opt,style_pool,
                                   render_slots,assets=assets,pixel_slots=pixel_slots)
        if sha(scene_path) != scene_hash: raise AssertionError('Styles mutated geometry')
        styles = read(case/'styling/styles.json'); widths = [v['style']['width_px']*SLIDE_WIDTH_PT/item['source_size'][0]
                  for v in styles['parts'].values() if v['style'].get('stroke')]
        maximum_width = settings.get('maximum_stroke_width_pt')
        if maximum_width is not None and any(w>maximum_width+1e-8 for w in widths): raise AssertionError('Stroke limit exceeded')
        pptx = case/'styling/styled.pptx'
        with zipfile.ZipFile(pptx) as z:
            xml = etree.fromstring(z.read('ppt/slides/slide1.xml'))
            native_widths = [int(n.get('w'))/12700 for n in xml.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/main}ln')
                             if n.get('w') and n.find('{http://schemas.openxmlformats.org/drawingml/2006/main}noFill') is None]
            if maximum_width is not None and any(w>maximum_width+.0001 for w in native_widths): raise AssertionError('Native PPT exceeds width limit')
        deliver = run/'delivery'/ident; deliver.mkdir(exist_ok=True)
        for src,name in ((pptx,'shapes.pptx'),(case/'styling/native.png','preview.png'),
                         (case/'styling/comparison.png','comparison.png')): shutil.copy2(src,deliver/name)
        quality=('completed' if style_summary['state']=='completed' and geometry_summary['state']=='completed'
                 else 'completed_with_unresolved')
        row = {'id':ident,'state':quality,'geometry':geometry_summary['state'],'styling':style_summary['state'],
               'pptx':str(deliver/'shapes.pptx'),'pptx_sha256':sha(deliver/'shapes.pptx'),
               'parts':len(styles['parts']),'max_width_pt':max(widths,default=0),
               'distinct_widths':len(set(round(w,4) for w in widths)),
               'capped_parts':sum(any(h.get('stage')=='maximum_stroke_width' for h in v.get('history',[])) for v in styles['parts'].values()),
               'geometry_sha256':scene_hash,'wall_seconds':time.monotonic()-began}
        save(receipt,row); save(case/'stage.json',{'stage':'completed','at':time.time()}); print('CASE_COMPLETE',ident,row,flush=True)
        return row

    def process(item, fit_pool, style_pool, sam_pool):
        slot_config = settings.get('case_slot_config')
        if slot_config:
            from full_recovery.case_slots import CaseSlots
            with CaseSlots(slot_config, 'shape').acquire(item['id']):
                return process_unlimited(item, fit_pool, style_pool, sam_pool)
        return process_unlimited(item, fit_pool, style_pool, sam_pool)

    progress()
    with ThreadPoolExecutor(max_workers=settings.get('fitting_workers',24)) as fit_pool, \
         ThreadPoolExecutor(max_workers=settings.get('styling_workers',16)) as style_pool, \
         ThreadPoolExecutor(max_workers=8) as sam_pool, ThreadPoolExecutor(max_workers=settings.get('case_workers',2)) as case_pool:
        pending = {case_pool.submit(process,item,fit_pool,style_pool,sam_pool):item for item in items}
        for future in as_completed(pending):
            item = pending[future]
            try: row = future.result()
            except Exception as exc:
                row = {'id':item['id'],'state':'failed','error':str(exc),'traceback':traceback.format_exc()}
                save(run/'geometry/cases'/item['id']/'failure.json',row); print('CASE_FAILED',row,flush=True)
            results[item['id']] = row; progress()
        if settings.get('resume_transport_cases_once',False):
            def transient_transport(row):
                error=row.get('error','')
                return ('Incomplete model transport' in error or
                        'Model transport failed' in error and any(
                            f'HTTP {status}' in error for status in (408, 425, 429, 500, 502, 503, 504)))
            retry=[item for item in items if results[item['id']]['state']=='failed'
                   and transient_transport(results[item['id']])]
            # Retry only interrupted streams and transient provider statuses;
            # completed case receipts are hash-checked and skipped.
            retried={case_pool.submit(process,item,fit_pool,style_pool,sam_pool):item for item in retry}
            for future in as_completed(retried):
                item=retried[future]
                try:row=future.result()
                except Exception as exc:
                    row={'id':item['id'],'state':'failed','error':str(exc),'traceback':traceback.format_exc()}
                    save(run/'geometry/cases'/item['id']/'failure.json',row)
                results[item['id']]=row;progress()
    bundle_root = report_root/'delivery'
    bundle_root.mkdir(parents=True, exist_ok=True)
    pdf = fitz.open()
    for item in items:
        comparison = run/'delivery'/item['id']/'comparison.png'
        if comparison.exists():
            with Image.open(comparison) as im: w,h = im.size
            page = pdf.new_page(width=960,height=960*h/w+24)
            page.insert_text((10,16),item['id']); page.insert_image(fitz.Rect(0,24,960,960*h/w+24),filename=str(comparison))
    if len(pdf): pdf.save(bundle_root/'comparison_125.pdf',garbage=4,deflate=True)
    pdf.close()
    with zipfile.ZipFile(bundle_root/'editable_shapes.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for item in items:
            path = run/'delivery'/item['id']/'shapes.pptx'
            if path.exists(): archive.write(path,item['id']+'.pptx')
    failed = [k for k,v in results.items() if v['state']=='failed']
    final = {'state':'completed_with_failures' if failed else 'completed_with_unresolved' if any(v['state']!='completed' for v in results.values()) else 'completed',
             'expected':len(items),'exported':len(items)-len(failed),'failed':failed,'cases':results,
             'maximum_stroke_width_pt':settings['maximum_stroke_width_pt'],'scope':'shape_geometry_and_style',
             'original_inputs_unchanged':all(sha(i['original'])==i['source_sha256'] for i in items),
             'at':time.time(),'visual_accuracy_verified':False}
    save(report_root/'summary.json',final)
    if report_root == run:
        save(run/'delivery/validation.json',final)
    else:
        save(report_root/'validation.json',final)
    if failed: raise RuntimeError(str(len(failed))+' cases failed; checkpoints retained')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True); p.add_argument('--input-root',type=Path)
    p.add_argument('--ids',nargs='+'); p.add_argument('--max-stroke-pt',type=float,default=None,
        help='Optional physical stroke ceiling; default preserves source-measured widths')
    p.add_argument('--resume-settings',type=Path)
    p.add_argument('--selected-ids',type=Path)
    p.add_argument('--report-root',type=Path)
    p.add_argument('--prepare',action='store_true'); args = p.parse_args()
    if args.prepare:
        if args.input_root is None: p.error('--prepare needs --input-root')
        if args.max_stroke_pt is not None and not 0<args.max_stroke_pt<100: p.error('Invalid stroke ceiling')
        prepare(args)
    else: execute(args.run_dir.resolve(), resume_settings=args.resume_settings,
                  selected_ids=args.selected_ids, report_root=args.report_root)


if __name__ == '__main__': main()
