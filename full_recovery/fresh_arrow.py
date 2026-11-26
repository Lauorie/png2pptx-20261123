"""Freeze the selected geometry-first arrow implementation for new inference."""
import importlib.util
from pathlib import Path
import shutil

from .arrow_workers import widened
from .prepare import read,save,sha


def prepare(reference,root,items):
    reference=Path(reference).resolve();root=Path(root).resolve()
    if reference.name=='geometry_code':
        reference=reference.parent
    staging=root.parent/'arrow_code_inputs';staging.mkdir()
    working=staging/'working';working.mkdir()
    (working/'code').symlink_to(reference/'geometry_code',target_is_directory=True)
    config=read(reference/'config.json')
    # This entry prepares a fresh inference run. Historical comparison and
    # localization-reuse switches must not leak out of the frozen template.
    for key in ('localization_reuse_from','comparison_run'):
        config.pop(key,None)
    config.update(case_workers=2,api_concurrency=8,cpu_workers=2,
                  input_run=str(root/'localization/inputs'),
                  global_predictions_run=str(root/'sam/missing'),repair_reuse_cached_masks=False)
    save(working/'config.json',config)
    baseline=staging/'reference'
    ids=[r['id'] for r in items]
    save(baseline/'cases.json',ids);save(baseline/'localization/cases.json',ids)
    for name in ('localization/config.json','sam/missing/config.json'):
        data=read(reference/name)
        for key in ('comparison_run','frozen_global_from','replay_observations_from','model_cache_from','reused_review_from'):
            data.pop(key,None)
        save(baseline/name,data)
    for item in items:
        path=baseline/'localization/inputs'/item['id']/'source.png'
        path.parent.mkdir(parents=True);path.symlink_to(item['original'])
    spec=importlib.util.spec_from_file_location('frozen_arrow_preparation',reference/'code/prepare.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.prepare(root,working,baseline,ids)
    # Freeze the actual selected driver, including its recorded fixes.
    for path in (reference/'code').glob('*.py'):
        shutil.copy2(path,root/'code'/path.name)
    record=read(root/'provenance.json')
    record.update(algorithm_source=str(reference),historical_products_read=False,
                  original_images=[{'id':i['id'],'path':i['original'],'sha256':i['source_sha256']} for i in items],
                  source_code_sha256={str(p.relative_to(reference)):sha(p)
                    for directory in ('code','geometry_code') for p in (reference/directory).rglob('*.py')},
                  adaptations=['resource limits','loopback API transport','disable prior mask reuse',
                               'restore node naming in geometry replies (arrow_reply_repair)'])
    save(root/'provenance.json',record)
    # Keep replies whose only defect is unquoted or positional node naming.
    from .arrow_reply_repair import install as install_reply_repair
    install_reply_repair(root/'geometry_code')
    refresh(root)


def refresh(root):
    """Only a newly prepared run may call this, before execution starts."""
    root=Path(root)
    if (root/'execution.json').exists():raise ValueError('Cannot rewrite an executed arrow run')
    manifest=read(root/'manifest.json')
    manifest['files']={name:sha(root/name) for name in manifest['files']}
    for directory in ('code','geometry_code'):
        manifest['files'].update({str(p.relative_to(root)):sha(p)
                                  for p in (root/directory).rglob('*') if p.is_file()})
    save(root/'manifest.json',manifest)


def configure(root,base_url,sam_url,api_concurrency=8):
    root=Path(root)
    paths=[root/'config.json',root/'localization/config.json',root/'sam/missing/config.json',
           *sorted((root/'geometry/jobs').glob('*/config.json'))]
    for path in paths:
        data=read(path);data.update(base_url=base_url,sam_url=sam_url,
                                   api_concurrency=api_concurrency)
        save(path,widened(data,api_concurrency))
    manifest=root/'localization/manifest.json';data=read(manifest)
    data['files']={n:sha(root/'localization'/n) for n in data['files']};save(manifest,data)
    for job in (root/'geometry/jobs').iterdir():
        data=read(job/'manifest.json');data['files']={n:sha(job/n) for n in data['files']}
        save(job/'manifest.json',data)
    refresh(root)


def describe(run):
    """Report the driver actually executed, rather than an unused legacy copy."""
    run=Path(run);settings=read(run/'settings.json')
    if settings.get('arrow_driver')=='existing_products':
        manifest=read(Path(settings['external_arrow_inputs']))
        provenance=read(run/'provenance.json')
        provenance['unused_legacy_arrow_backend']=provenance['arrow_method_source']
        provenance['arrow_method_source']={'driver':'existing_products','input_run':manifest['root'],
            'input_manifest':settings['external_arrow_inputs'],'historical_products_read':True,
            'arrow_inference_enabled':False}
        provenance['scheduling'].update(arrow_cases=0,arrow_requests=0,arrow_groups=0)
        save(run/'provenance.json',provenance)
        profile=read(run/'glm_profiles.json')
        profile['arrow']={'inference_enabled':False,'input_run':manifest['root']}
        save(run/'glm_profiles.json',profile)
        return
    if settings.get('arrow_driver')!='frozen_geometry_first':return
    root=Path(settings['fresh_arrow_root']);config=read(root/'config.json');source=read(root/'provenance.json')
    provenance=read(run/'provenance.json')
    provenance['unused_legacy_arrow_backend']=provenance['arrow_method_source']
    provenance['arrow_method_source']={'driver':settings['arrow_driver'],
        'algorithm_source':source['algorithm_source'],'code_and_config_root':str(root),
        'historical_products_read':False}
    provenance['scheduling'].update(arrow_cases=config['case_workers'],
        arrow_requests=config['api_concurrency'],arrow_groups=config['task_workers'])
    save(run/'provenance.json',provenance)
    profile=read(run/'glm_profiles.json')
    profile['arrow']={k:config.get(k) for k in ('max_tokens','thinking','reasoning_effort',
        'style_profile','request_timeout_seconds','request_max_attempts')}
    profile['arrow']['algorithm_source']=source['algorithm_source']
    save(run/'glm_profiles.json',profile)
