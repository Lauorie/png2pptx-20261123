"""Transport/resource adaptation of a newly frozen complete-PPT workflow."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
from .prepare import read,save,sha
from .service_endpoints import sam_base,image_edit_base


def sam_profile(url):
    """Pin the actual user-selected service instead of a fixed old epoch."""
    url=sam_base(url)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url.rstrip('/')+'/health',timeout=15) as response:
        health=json.load(response)
    model=health.get('model',{});checkpoint=model.get('checkpoint',{})
    digest=model.get('checkpoint_sha256') or checkpoint.get('sha256')
    if (health.get('phase') not in ('ready','busy') or not health.get('warmup_passed')
            or not model.get('checkpoint_hash_verified') or not isinstance(digest,str)
            or len(digest)!=64 or not isinstance(checkpoint.get('epoch'),int)):
        raise ValueError('Selected SAM service is not ready or its model identity is unverified')
    return {'url':url.rstrip('/'),'epoch':checkpoint['epoch'],'sha256':digest,
            'checkpoint_path':model['checkpoint_path'],'verified_at':time.time()}


def credential_sources(project,args):
    key_files=getattr(args,'api_key_file',None)
    if not key_files:
        raise ValueError('Provide --api-key-file explicitly; this handoff contains no credentials')
    if isinstance(key_files,Path):key_files=[key_files]
    return key_files


def stage_credentials(run,key_files):
    """Stage the selected credentials; historical products never select a key."""
    for key in key_files:
        if key.stat().st_mode & 0o077:raise ValueError('Source API key must have mode 0600')
    private=run/'private';private.mkdir(mode=0o700);private.chmod(0o700)
    credentials=list(dict.fromkeys(line.strip() for key in key_files for line in key.read_text().splitlines() if line.strip()))
    if not credentials or any(any(c.isspace() for c in value) for value in credentials):
        raise ValueError('Expected one credential per line in the protected key file')
    staged_keys=[]
    for index,value in enumerate(credentials):
        staged=private/f'api_key_{index}'
        fd=os.open(staged,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as stream:stream.write(value+'\n')
        staged_keys.append(str(staged))
    return staged_keys


def align_shape_limits(settings,config):
    """Use the allocated relay budget instead of the local-model default."""
    limit=next((row['api_global_concurrency'] for row in settings.get('api_relays',[])
                if row['role']=='shape'),settings['api_global_concurrency'])
    settings['glm_concurrency']=limit
    config.update(api_concurrency=limit,case_workers=settings['case_workers'])


def configure(project,args):
    run=args.run_dir.resolve()
    selected_sam=sam_profile(args.sam_url)
    sam_url=selected_sam['url']
    key_files=credential_sources(project,args)
    if args.api_concurrency<1:raise ValueError('Positive API concurrency required')
    for name in ('case_workers','text_workers'):
        value=getattr(args,name,None)
        if value is not None and value<1:raise ValueError('Positive '+name+' required')
    staged_keys=stage_credentials(run,key_files)
    staged=Path(staged_keys[0])
    base='http://127.0.0.1:18188/v1'
    checkpoint=selected_sam['sha256']
    settings=read(run/'settings.json')
    shape_only=getattr(args,'shape_only',False)
    external_arrows=settings.get('arrow_driver')=='existing_products' and not shape_only
    if shape_only:settings['workflow_scope']='shape'
    settings.update(glm_transport='api',glm_url=base,glm_api_model='glm-5.3-flash',
        glm_api_key_file=str(staged),glm_model='api:glm-5.3-flash',
        api_model='glm-5.3-flash',api_base_url=args.api_base_url.rstrip('/'),api_key_file=str(staged),api_key_files=staged_keys,
        api_credential_source_files=[str(p.resolve()) for p in key_files],
        api_gateway_port=18188,api_global_concurrency=args.api_concurrency,api_request_timeout_seconds=3600,
        glm_request_timeout_seconds=3600,sam_url=sam_url,sam_urls=[sam_url],
        external_sam_epoch=selected_sam['epoch'],external_sam_checkpoint_sha256=checkpoint,sam_cache_run=None,
        sam_checkpoint_sha256=checkpoint,
        sam_checkpoint=selected_sam['checkpoint_path'],sam_reuse_roots=[],
        namespace=getattr(args,'namespace','ailab-intern10'),
        charged_group=getattr(args,'charged_group','intern10_proxy'),gpu=0,
        worker_cpu=getattr(args,'worker_cpu',16),worker_memory_mib=getattr(args,'worker_memory_mib',65536),
        worker_max_seconds=345600,project_root=str(project),source_project=str(project))
    if getattr(args,'image_edit_url',None):
        settings['image_edit']={'enabled':True,'base_url':image_edit_base(args.image_edit_url),
            'model':args.image_edit_model,'concurrency':args.image_edit_concurrency,'timeout_seconds':180}
    if getattr(args,'arrow_native_renderer',None):
        settings['arrow_native_renderer']=str(args.arrow_native_renderer)
    if getattr(args,'native_renderer',None):
        settings['native_renderer']=str(args.native_renderer.resolve())
    for key in ('glm_profiles','glm_tp','glm_dp','glm_max_model_len','glm_ports','sam_devices','arrow_sam_device'):
        settings.pop(key,None)
    # Separate branch budgets prevent long arrow generations from occupying
    # every shape/text slot. No changes to stage-specific inference profiles.
    if not shape_only and len(staged_keys)>=3 and args.api_concurrency>=6:
        roles=('shape','text') if external_arrows else ('shape','arrow','text')
        limits=[args.api_concurrency//len(roles)]*len(roles)
        limits[-1]+=args.api_concurrency-sum(limits)
        ports={'shape':18188,'arrow':18189,'text':18190}
        settings['api_relays']=[{'role':role,'port':ports[role],'api_global_concurrency':limits[i],
            'api_key_files':staged_keys[i::len(roles)]} for i,role in enumerate(roles)]
    requested_case_workers = getattr(args, 'case_workers', None) or 8
    settings.update(case_workers=requested_case_workers, text_workers=8)
    if shape_only:
        settings.update(case_workers=requested_case_workers,glm_concurrency=args.api_concurrency,
                        fitting_workers=48,styling_workers=32)
    save(run/'settings.json',settings);save(run/'shape/settings.json',settings)
    text=read(run/'backends/text/config.json');text['expected_boyue_base_url']=('http://127.0.0.1:18190/v1' if settings.get('api_relays') else base)
    save(run/'backends/text/config.json',text)
    from .arrow_budget import arrow_base_url, to_settings as arrow_budget_settings
    settings=arrow_budget_settings(settings,getattr(args,'arrow_reasoning_effort',None),
                                   getattr(args,'arrow_max_tokens',None))
    arrow=read(run/'arrows/config.json');arrow.update(base_url=arrow_base_url('http://127.0.0.1:18189/v1' if settings.get('api_relays') else base),sam_url=args.sam_url.rstrip('/'),case_workers=8)
    save(run/'arrows/config.json',arrow)
    if settings.get('workflow_scope')=='fresh_full':
        if not external_arrows:
            from .fresh_arrow import configure as configure_arrow
            limits=next((r['api_global_concurrency'] for r in settings.get('api_relays',[]) if r['role']=='arrow'),args.api_concurrency)
            configure_arrow(settings['fresh_arrow_root'],arrow['base_url'],settings['sam_url'],limits)
        settings.update(text_workers=getattr(args,'text_workers',None) or 2,
                        case_workers=getattr(args,'case_workers',None) or 2,reuse_request_roots=[])
    shape_config=read(run/'shape/pipeline_config.json')
    from .inference_profile import apply as apply_profile
    settings,shape_config,text=apply_profile(settings,shape_config,text,
        model=getattr(args,'api_model','glm-5.3-flash'),
        effort=getattr(args,'force_reasoning_effort',None),
        profile=getattr(args,'critic_profile','full'))
    if getattr(args,'api_budget_config',None):
        settings['api_budget_config']=str(args.api_budget_config.resolve())
    save(run/'backends/text/config.json',text)
    align_shape_limits(settings,shape_config)
    save(run/'shape/pipeline_config.json',shape_config)
    save(run/'settings.json',settings);save(run/'shape/settings.json',settings)
    # These are renderer assets, never previous inference outputs.
    runtime=run/'renderer_runtime';runtime.mkdir()
    source=project/'.runtime/renderer_support'
    shutil.copytree(source/'glibc235',runtime/'glibc235')
    shutil.copy2(source/'lo_cpu.py',runtime/'lo_cpu.py')
    shutil.copy2(project/'tests/test_full_api.py',run/'code/test_full_api.py')
    save(run/'glm_profiles.json',{
        'provider':{'model':settings['api_model'],'base_url':args.api_base_url,'server_context_limit':'provider-managed; not locally configured'},
        'forced_reasoning_effort':settings.get('api_force_reasoning_effort'),
        'critic_profile':settings['critic_profile'],'critic_switches':settings['critic_switches'],
        'shape':{'max_tokens':settings['glm_max_tokens'],'reasoning_effort':'high'},
        'arrow':{k:arrow.get(k) for k in ('max_tokens','thinking','reasoning_effort')},
        'text':{'global_max_tokens':text['global_max_tokens'],'stage_specific_effort':settings.get('api_force_reasoning_effort') or 'reference preserved, including local low overrides'},
        'global_api_concurrency':args.api_concurrency,'transport':'Upstream SSE for all three branches'})
    provenance=read(run/'provenance.json')
    provenance.update(shape='latest integrated geometry + source-measured native styles + SAM visual pattern + selected service',
        sam_service=selected_sam,
        adaptation='Frozen reference stages; loopback streaming API relay and cross-process shared-SAM lock; no local GLM weights.',
        external_results_reused=external_arrows,only_reference_png_used=not external_arrows,
        experimental_numeric_L_alignment_enabled=False,experimental_global_critic_enabled=False,
        scheduling={**provenance['scheduling'],'shape_cases':settings['case_workers'],'text_cases':settings['text_workers'],
                    'arrow_cases':0 if external_arrows else arrow['case_workers'],
                    'global_api_requests':args.api_concurrency,'shape_requests':settings['glm_concurrency'],
                    'api_branch_budgets':settings.get('api_relays',[]),'glm_replicas':0,'gpu':0,
                    'worker_cpu':16,'worker_memory_mib':65536,'native_renders_global':2})
    for key in ('glm_tp_per_replica','vllm_active_sequences_per_replica'):
        provenance['scheduling'].pop(key,None)
    save(run/'provenance.json',provenance)
    save(run/'sam_service_binding.json',selected_sam)
    if shape_only:
        provenance.update(workflow_scope='shape_geometry_and_style',text_inference_enabled=False,
            arrow_inference_enabled=False,composition_enabled=False,
            later_composition='Use shape/geometry, shape/localization and source_manifest with supplied text/arrow products')
        provenance['scheduling'].update(shape_requests=args.api_concurrency,text_cases=0,arrow_cases=0)
        save(run/'provenance.json',provenance)
    from .fresh_arrow import describe as describe_arrow
    describe_arrow(run)
    identities={}
    for directory in (run/'code',run/'backends',runtime):
        for path in sorted(directory.rglob('*')):
            if path.is_file() and not path.is_symlink():identities[str(path.relative_to(run))]=sha(path)
    for path in (run/'settings.json',run/'shape/settings.json',run/'shape/pipeline_config.json',
                 run/'arrows/config.json',run/'source_manifest.json',run/'provenance.json',run/'glm_profiles.json'):
        identities[str(path.relative_to(run))]=sha(path)
    identities['sam_service_binding.json']=sha(run/'sam_service_binding.json')
    if (run/'external_arrow_inputs.json').is_file():
        identities['external_arrow_inputs.json']=sha(run/'external_arrow_inputs.json')
    save(run/'code_manifest.json',identities)
    (run/'logs').mkdir(exist_ok=True)
    save(run/'status.json',{'state':'prepared','samples':len(read(run/'source_manifest.json')['images']),
         'local_glm':False,'gpu':0,'transport':'hosted_api','at':time.time()})
