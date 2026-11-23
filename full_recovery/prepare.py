"""Freeze three independent implementations and register the original PNGs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def freeze_text(text_source,text,api_url='http://127.0.0.1:10089/v1'):
    """Freeze the explicitly selected text source, never a prior shape run."""
    text_source=Path(text_source).resolve();text.mkdir(parents=True)
    for source in text_source.glob('*.py'):shutil.copy2(source,text/source.name)
    for directory in ('src','tests','vendor'):
        shutil.copytree(text_source/directory,text/directory,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # Freeze the transport package behind the original absolute vendor link.
    package=text/'vendor/png2pptx'
    if package.is_symlink():
        original=package.resolve();package.unlink()
        shutil.copytree(original,package,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(text_source/'.deps',text/'.deps',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # The supplied pptbench125_glm_current_high_20261023_v1 results used
    # explicit high effort in both inventory and local geometry/attribute calls.
    # The source directory's current config omits these fields and falls back
    # to low. Copying its code alone therefore does not reproduce that run.
    text_config=read(text_source/'config.json')
    text_config.update(model='GLM-5.3-Flash',expected_boyue_base_url=api_url,
                       reasoning_effort='high')
    text_config.setdefault('texts',{})['geometry_reasoning_effort']='high'
    save(text/'config.json',text_config)
    entry=text/'run_global_ocr_review.py';original=entry.read_text()
    adapted=original.replace("if not key:raise ValueError('API key required in environment')",
        "if not key and os.environ.get('PPT_LOCAL_GLM')!='1':raise ValueError('API key required in environment')")
    adapted=adapted.replace("if not key:raise ValueError('API key required for GLM crop review')",
        "if not key and os.environ.get('PPT_LOCAL_GLM')!='1':raise ValueError('API key required for GLM crop review')")
    if adapted==original:raise ValueError('Text local-adapter guard did not match the source')
    entry.write_text(adapted)
    # An audit tile is allowed to see half an existing word, but that partial
    # view cannot replace the complete occurrence. The reference prompt says
    # this already; enforce it at application time without another model call.
    controller=text/'src/shape_integration/texts/controller.py'
    source=controller.read_text()
    marker="            # A lower-quality partial crop must not destroy a readable whole line."
    guard=("            if origin == 'inventory_visual_audit' and any(\n"
           "                    old['bbox_2d'][i] < f['box'][i] if i < 2 else\n"
           "                    old['bbox_2d'][i] > f['box'][i] for i in range(4)):\n"
           "                history.append({'action':'keep_complete_audit_occurrence',\n"
           "                    'id':old['id'],'origin':origin,'candidate':row,\n"
           "                    'reason':'Existing occurrence crosses audit crop boundary'})\n"
           "                continue\n")
    if source.count(marker)!=1:raise ValueError('Reference text audit merge no longer matches source')
    controller.write_text(source.replace(marker,guard+marker))
    # Resolve only well-supported reading reversals with the already measured
    # complete glyph ink, before the reference pipeline's existing final export.
    from .source_text_direction import PROTOCOL as source_direction_protocol
    from .text_occurrences import PROTOCOL as grounding_protocol
    crop=text/'src/shape_integration/texts/two_stage_crop.py'
    source=crop.read_text()
    marker="    save(output/'crop_recognition.json',{'texts':chosen,'policy':POLICY})"
    direction=("    from full_recovery.source_text_direction import correct as correct_direction\n"
               "    with Image.open(source) as direction_source:\n"
               "        chosen,direction=correct_direction(chosen,direction_source.convert('RGB'))\n"
               "    save(output/'source_direction.json',{'protocol':"
               +repr(source_direction_protocol)+",'observations':direction,\n"
               "        'source_boxes_and_font_sizes_unchanged':True,'extra_model_stages':0})\n")
    direction+=("    from full_recovery.text_occurrences import ground as ground_occurrences,refit_position\n"
               "    with Image.open(source) as occurrence_source:\n"
               "        chosen,grounded_ids,grounding=ground_occurrences(chosen,occurrence_source.convert('RGB'))\n"
               "    chosen=[refit_position(row,layout.fit)\n"
               "        if row['id'] in grounded_ids else row for row in chosen]\n"
               "    save(output/'source_occurrence_grounding.json',{'observations':grounding,\n"
               "        'changed_ids':grounded_ids,'extra_model_stages':0})\n")
    if source.count(marker)!=1:raise ValueError('Reference text final export no longer matches source')
    crop.write_text(source.replace(marker,direction+marker))
    # The existing full-source recovery can explicitly retire a false initial
    # occurrence already represented by a verified, unchanged source keeper.
    from .source_text_occurrences import adapt as adapt_occurrences,PROTOCOL as occurrence_protocol
    crop.write_text(adapt_occurrences(crop.read_text()))
    from .text_recovery_policy import install as install_recovery_policy
    occurrence_repair = install_recovery_policy(text)
    from .text_unit_layout import install as install_unit_layout
    install_unit_layout(text)
    from .text_diagnostic_repair import adapt as adapt_diagnostic_repair, PROTOCOL as diagnostic_protocol
    crop.write_text(adapt_diagnostic_repair(crop.read_text()))
    from .text_diagnostic_repair import adapt_inventory
    inventory_audit = text/'src/shape_integration/texts/inventory_audit.py'
    inventory_audit.write_text(adapt_inventory(inventory_audit.read_text()))
    # Assert the unchanged low-effort contract at the local template field.
    test=text/'tests/test_local_geometry.py';source=test.read_text()
    expected="assert parameters['reasoning_effort']=='low'"
    if expected in source:
        test.write_text(source.replace(expected,"assert parameters['chat_template_kwargs']['reasoning_effort']=='low'"))
    # The latest reference added a fifth format example and requires complete
    # frame metadata; synchronize its two stale test fixtures, not runtime code.
    fixture_fixes={
        'test_format_normalization.py':("assert len(parsed['texts'])==4","assert len(parsed['texts'])==5"),
        'test_layout_safety.py':("frame={'scale':1,'offset':[0,0],'box':[0,0,200,100]}",
                                "frame={'scale':1,'offset':[0,0],'box':[0,0,200,100],'size':[200,100],'source_size':[200,100]}")}
    if (text/'src/shape_integration/texts/single_crop.py').exists():
        for name,(before,after) in fixture_fixes.items():
            path=text/'tests'/name
            if path.exists():path.write_text(path.read_text().replace(before,after))
        path=text/'tests/test_format_normalization.py'
        data=path.read_text()
        data=data.replace("prompt=recognize([200,100]); examples=prompt['format_examples']['texts']",
                          "prompt=recognize([200,100]); examples=prompt['format_examples']['texts']; examples.sort(key=lambda r:r['text']!='k / v')")
        path.write_text(data)
        save(text/'test_fixture_adaptations.json',fixture_fixes)
    from .text_test_fixtures import synchronize
    synchronize(text)
    from .text_inventory import PROTOCOL as inventory_review_protocol
    from .source_text_attributes import PROTOCOL as source_attribute_protocol
    text_policy = {'protocol':'reference_text_pipeline_high', 'installed':True,
                   'reason':'Reference high profile; partial audit views cannot overwrite complete occurrences',
                   'reasoning_effort':'high', 'geometry_reasoning_effort':'high',
                   'reference_run':str(text_source/'runs/pptbench125_glm_current_high_20261023_v1'),
                   'audit_crop_boundary_guard':True,
                   'attribute_prompt_policy':source_attribute_protocol,
                   'reading_direction_policy':source_direction_protocol,
                   'source_ink_occurrence_placement':grounding_protocol,
                   'source_placement_retains_native_typography':True,
                   'source_occurrence_inventory_review':inventory_review_protocol,
                   'source_occurrence_recovery':occurrence_protocol,
                   'bounded_occurrence_repair':occurrence_repair,
                   'diagnostic_repair':diagnostic_protocol,
                   'extra_model_stages':0}
    save(text/'source_provenance.json',{'source':str(text_source),'sha256':{str(p.relative_to(text_source)):sha(p) for p in (text_source/'src').rglob('*.py')},
        'local_runtime_adaptations': text_policy})


def prepare(project,args):
    if getattr(args,'arrow_code_run',None) and getattr(args,'arrow_products_run',None):
        raise ValueError('Fresh arrow code and existing arrow products are mutually exclusive')
    shape_only=getattr(args,'shape_only',False)
    if shape_only and any(getattr(args,name,None) for name in
                          ('arrow_code_run','arrow_products_run','arrow_implementation')):
        raise ValueError('Shape-only preparation does not accept arrow inputs; compose them later.')
    implementation = getattr(args, 'arrow_implementation', None)
    bundled = project/'third_party/arrow_geometry'
    if not shape_only and not implementation and not getattr(args,'arrow_code_run',None) and (bundled/'source.json').is_file():
        implementation = bundled
    fresh_arrow_code = getattr(args,'arrow_code_run',None)
    if not shape_only and implementation and not getattr(args,'arrow_products_run',None) and not fresh_arrow_code:
        fresh_arrow_code = Path(implementation)
    run=args.run_dir.resolve()
    if run.exists():raise ValueError('Use a fresh integration run')
    import run_shapes
    run_shapes.prepare(SimpleNamespace(run_dir=run/'shape',input_root=args.input_root,ids=args.ids,max_stroke_pt=None))
    # The full compositor commits image representations after native recovery.
    # A registered crop is still a proposal during shape fitting and styling.
    shape_config = read(run/'shape/pipeline_config.json')
    shape_config['image_ownership_stage'] = 'composition'
    save(run/'shape/pipeline_config.json', shape_config)
    manifest=read(run/'shape/source_manifest.json');items=manifest['images']
    text_source=args.text_source.resolve();arrow_reference=args.arrow_reference.resolve()
    code=run/'code';code.mkdir()
    shutil.copy2(project/'run_ppt.py',code/'run_ppt.py')
    shutil.copytree(project/'full_recovery',code/'full_recovery',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(project/'scripts/ppt_worker.py',code/'ppt_worker.py')
    shutil.copy2(project/'scripts/lo_guard.py',code/'lo_guard.py')
    for name in ('test_full_composition.py','test_full_api.py','test_full_arrow_methods.py','test_full_text_occlusion.py','test_full_line_dedup.py','test_full_text_product.py'):
        shutil.copy2(project/'tests'/name,code/name)
    shutil.copy2(project/'tests/test_fresh_pipeline.py',code/'test_full_fresh_pipeline.py')
    from .fonts import freeze
    freeze(project/'.runtime/text_font_bundle_20261021',run/'backends/reference_fonts')
    text=run/'backends/text';freeze_text(text_source,text)
    text_config=read(text/'config.json')
    arrow=run/'backends/arrow';arrow.mkdir(parents=True)
    from .arrow_backend import freeze as freeze_arrow
    arrow_config,frozen=freeze_arrow(project,arrow_reference,args.arrow_delivery.resolve(),arrow)
    sam_source=text_source.parent/'arrow_sam_ckpt_20261015'
    sam_backend=run/'backends/arrow_sam';sam_backend.mkdir()
    for name in ('server.py','engine.py','verify_assets.py'):
        shutil.copy2(sam_source/'sam_service'/name,sam_backend/name)
    sam_config=read(sam_source/'runs/full40_best_20261016_111338/config.json')
    sam_repo=project/'deploy/sam/vendor/sam3'
    sam_config.update(training_run=str(project/'deploy/sam'),sam_repository=str(sam_repo),
        dataset_manifest=str(project/'deploy/sam/categories.json'),
        checkpoint_path=str(project/'weights/sam3_best_epoch7/best.pt'))
    for name,asset in sam_config['runtime_assets'].items():
        asset['path']=str(sam_repo/name)
    for name in ('baseline_run','comparison_runs'):sam_config.pop(name,None)
    sam_config.update(host='127.0.0.1',port=18107)
    # The frozen model/runtime are input assets; no prior predictions or GLM
    # outputs are copied or consulted by any stage in this run.
    sam_manifest=sam_backend/'dataset_manifest.json'
    shutil.copy2(sam_config['dataset_manifest'],sam_manifest);sam_config['dataset_manifest']=str(sam_manifest)
    save(sam_backend/'config.json',sam_config)
    arrow_run=run/'arrows';arrow_run.mkdir()
    arrow_config.update(base_url='http://127.0.0.1:10089/v1',model='GLM-5.3-Flash',
                        input_run=str(arrow_run/'inputs'),global_predictions_run=str(arrow_run/'global_input'),
                        model_cache_from=None,sam_cache_from=[],sam_url='http://127.0.0.1:18107')
    for name in ('style_resume_from','coverage_resume_from'):
        arrow_config.pop(name,None)
    save(arrow_run/'config.json',arrow_config)
    save(arrow_run/'scope.json',{'cases':[i['id'] for i in items]})
    save(arrow_run/'global_input/manifest.json',{'images':items})
    for item in items:
        path=arrow_run/'inputs/cases'/item['id']/'clean.png';path.parent.mkdir(parents=True,exist_ok=True);path.symlink_to(item['original'])
    settings=read(run/'shape/settings.json')
    settings['sam_ports']=settings['sam_ports'][:3]
    settings.update(worker_max_seconds=172800,
                    namespace='ailab-llmshare',charged_group='llmshare_gpu',worker_cpu=64,
                    glm_dp=1,colocated_glm=False,
                    glm_profiles={'glm_a':{'glm_tp':4,'glm_dp':1,'glm_max_model_len':262144,'glm_max_num_seqs':24,'glm_gpu_memory_utilization':.9},
                                  'glm_b':{'glm_tp':4,'glm_dp':1,'glm_max_model_len':262144,'glm_max_num_seqs':24,'glm_gpu_memory_utilization':.72}},
                    sam_devices=[4,5,6],arrow_sam_device=7,arrow_sam_port=18107,
                    text_workers=2,text_api_limit_per_case=text_config['api_concurrency'],
                    text_ocr_python='/opt/shape-cpu/bin/python',
                    text_models='/mnt/shared-storage-user/dongdaozhu/Anything2Fig/third_party/DrawAI/.local/drawai_runtime/models/paddlex/official_models',
                    text_recognition_mode='staged',text_geometry_backend='glm',text_ocr_mode='geometry',text_ocr_template_validation='on',
                    text_review_policy='reference_text_pipeline_high',
                    text_crop_refine=True,text_ocr_max_side=1536,text_threshold=.80,text_max_score_drop=.05,
                    arrow_reference=str(arrow_reference),arrow_delivery_method_reference=str(args.arrow_delivery.resolve()),text_source=str(text_source),
                    source_project=str(project),maximum_stroke_width_pt=None)
    save(run/'shape/settings.json',settings)
    save(run/'glm_profiles.json',{'shape':{'base_url':settings['glm_url'],'max_model_len':262144,'max_tokens':settings['glm_max_tokens'],'reasoning_effort':'high'},
         'arrow':{'base_url':arrow_config['base_url'],'max_tokens':arrow_config['max_tokens'],'thinking':arrow_config['thinking'],
                  'reasoning_effort':arrow_config.get('reasoning_effort'),'reference_config':str(arrow_reference/'config_point_geometry_mainline.json')},
         'text':{'base_url':text_config['expected_boyue_base_url'],'global_max_tokens':text_config['global_max_tokens'],
                 'reasoning_effort':text_config.get('reasoning_effort'),'stage_specific_effort':'preserved, including local low overrides',
                 'reference_config':str(text_source/'config.json')},'text_arrow_service_context_capacity':262144})
    settings.update(workflow_scope='fresh_full',arrow_driver='frozen_point_geometry',
                    fresh_arrow_root=str(arrow_run),requested=len(items),expected=len(items))
    save(run/'settings.json',settings);save(run/'source_manifest.json',manifest)
    if not shape_only and fresh_arrow_code:
        # Only executable code/configuration is frozen from this reference.
        # Its localization, masks, geometry and PPT products are never inputs.
        from .fresh_arrow import prepare as prepare_fresh_arrow
        prepare_fresh_arrow(fresh_arrow_code,run/'fresh_arrows',items)
        settings.update(workflow_scope='fresh_full',arrow_driver='frozen_geometry_first',
                        fresh_arrow_root=str(run/'fresh_arrows'),requested=len(items),expected=len(items))
        save(run/'settings.json',settings)
    external_arrows = getattr(args,'arrow_products_run',None)
    if external_arrows:
        from .external_arrow_products import bind
        bound = bind(external_arrows, items, run/'external_arrow_inputs.json',
                     pending_ids=getattr(args,'pending_arrow_ids',()))
        settings.update(arrow_driver='existing_products',fresh_arrow_root=bound['root'],
            external_arrow_inputs=str(run/'external_arrow_inputs.json'),arrow_inference_enabled=False,
            expected=len(bound['records']))
        save(run/'settings.json',settings)
    if not shape_only and implementation:
        from .arrow_implementation import freeze as freeze_implementation
        record = freeze_implementation(implementation, run/'backends/arrow_implementation')
        settings['arrow_implementation_manifest'] = str(record)
        save(run/'settings.json', settings)
    from .compose import COMPOSITION_PROTOCOL
    save(run/'provenance.json',{'shape':'latest integrated epoch18 + geometry guard + source-measured native styles',
         'text_source':str(text_source),'arrow_reference':str(arrow_reference),
         'arrow_method_source':read(arrow/'METHOD_SOURCE.json'),
         'text_review_policy':settings['text_review_policy'],'composition_logic_changed':True,
         'composition_protocol':COMPOSITION_PROTOCOL,'text_background_policy':'source_text_stroke_fill.3',
         'image_representation_ownership':'composition_commit_after_native_style',
         'shape_text_contract':'complete_supported_frames_then_source_grounded_text_fill',
         'native_cylinder_contract':'shared_sample_top_face_and_bounded_can_with_source_safe_paint',
         'external_results_reused':bool(external_arrows),
         'arrow_reference_files':frozen,'only_reference_png_used':not bool(external_arrows),
         'shape_text_inference_from_original_png':True,
         'external_arrow_products':str(external_arrows) if external_arrows else None,
         'adaptation':'Isolated processes; local GLM model/endpoint and thinking-template adapter; native OPC composition.',
         'scheduling':{'shape_cases':2,'shape_requests':12,'style_pixels':2,
                       'arrow_cases':arrow_config['case_workers'],'arrow_requests':arrow_config['api_concurrency'],
                       'arrow_groups':arrow_config['task_workers'],'text_cases':2,'text_requests_per_case':text_config['api_concurrency'],
                       'native_renders_global':2,'glm_replicas':2,'glm_tp_per_replica':4,'vllm_active_sequences_per_replica':24}})
    identities={}
    for directory in (code,run/'backends'):
        for path in sorted(directory.rglob('*')):
            if path.is_file() and not path.is_symlink():identities[str(path.relative_to(run))]=sha(path)
    if (run/'external_arrow_inputs.json').is_file():
        identities['external_arrow_inputs.json']=sha(run/'external_arrow_inputs.json')
    save(run/'code_manifest.json',identities)
    for directory in ('jobs','services/logs','texts','cases','delivery','tests_tmp'): (run/directory).mkdir(parents=True,exist_ok=True)
    save(run/'status.json',{'state':'prepared','samples':len(items),'local_glm':True})
    print(json.dumps({'run':str(run),'samples':len(items),'external_results_reused':bool(external_arrows)}))
