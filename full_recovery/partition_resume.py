"""Disjoint, checkpoint-preserving branch views for a coordinated handoff.

The coordinator must own the original run locks before launching these views.
Each view writes distinct case directories; manifests and aggregate reports are
private to the view. Completed products keep their canonical original paths.
"""
from pathlib import Path
import shutil

from .atomic_json import save
from .prepare import read, sha


def validate_partition(universe, groups, protected=()):
    universe = set(universe)
    seen = set()
    for group in groups:
        if len(group) != len(set(group)) or seen.intersection(group):
            raise ValueError('Partition contains duplicate case writers')
        seen.update(group)
    if seen != universe:
        raise ValueError('Partition must cover the selected cases exactly')
    if not set(protected).issubset(groups[0]):
        raise ValueError('Active cases must remain in the original provider group')


def case_view(source, target, ids):
    source, target = Path(source), Path(target)
    if target.exists():
        raise ValueError('Use a fresh case view')
    all_items = read(source / 'source_manifest.json')['images']
    if not set(ids) <= {i['id'] for i in all_items}:
        raise ValueError('Unknown case IDs')
    items = [i for i in all_items if i['id'] in set(ids)]
    target.mkdir(parents=True)
    save(target / 'source_manifest.json', {'images': items})
    shape = target / 'shape'
    shape.mkdir()
    for name in ('settings.json', 'pipeline_config.json'):
        shutil.copy2(source / 'shape' / name, shape / name)
    original_manifest = read(source / 'shape/source_manifest.json')
    save(shape / 'source_manifest.json', {**original_manifest,
         'images': [i for i in original_manifest['images'] if i['id'] in set(ids)]})
    for parent in ('geometry/cases', 'localization/cases', 'sam_geometry',
                   'sam_styles', 'delivery', 'inputs'):
        (shape / parent).mkdir(parents=True, exist_ok=True)
    (target / 'texts/cases').mkdir(parents=True)
    for item in items:
        cid = item['id']
        for parent in ('geometry/cases', 'localization/cases', 'sam_geometry',
                       'sam_styles', 'delivery'):
            original = source / 'shape' / parent / cid
            (shape / parent / cid).symlink_to(original, target_is_directory=True)
        (shape / 'inputs' / (cid + '.png')).symlink_to(source / 'shape/inputs' / (cid + '.png'))
        (target / 'texts/cases' / cid).symlink_to(source / 'texts/cases' / cid, target_is_directory=True)
    for lock in ('worker.lock', 'fresh_branches.lock', 'shape/.pipeline.lock'):
        (target / lock).touch()
    save(target / 'ownership.json', {'canonical_source': str(source), 'case_ids': ids,
         'case_directories_shared_only_with_canonical_source': True,
         'requires_canonical_source_coordinator_lock': True})
    return items


def provision_view(target):
    """Create absent canonical case directories in the cluster worker's UID."""
    target = Path(target)
    ownership = read(target / 'ownership.json')
    source = Path(ownership['canonical_source'])
    for cid in ownership['case_ids']:
        for parent in ('geometry/cases', 'localization/cases', 'sam_geometry', 'sam_styles', 'delivery'):
            link = target / 'shape' / parent / cid
            expected = source / 'shape' / parent / cid
            if not link.is_symlink() or link.readlink() != expected:
                raise ValueError('Unexpected canonical case link')
            expected.mkdir(parents=True, exist_ok=True)


def prepare_partition(plan, role):
    from .prepare_branch_resume import prepare, freeze
    source = Path(plan['source_run'])
    previous = Path(plan['previous_run'])
    run = Path(plan[role + '_run'])
    ids = plan[role + '_ids']
    key = Path(plan['discovery_key_file']) if role == 'discovery' else previous / 'private/api_key'
    prepare(source, run, key)
    items = case_view(source, run / 'work', ids)
    previous_settings = read(previous / 'settings.json')
    settings = read(run / 'settings.json')
    count = int(plan.get(role + '_case_workers', 24 if role == 'discovery' else 4))
    limit = int(plan.get(role + '_api_concurrency',
                         plan.get('discovery_api_concurrency', 96) if role == 'discovery' else 6))
    fitting = limit
    slots = {'root': str(run / 'case_slots'), 'total': count,
             'shape_reserved': 16 if role == 'discovery' else 3}
    for name in ('api_base_url', 'api_model', 'api_force_reasoning_effort',
                 'api_service_tier', 'api_budget_config', 'sam_url', 'sam_urls'):
        settings[name] = previous_settings[name]
    settings.update(case_workers=count, text_workers=count, glm_concurrency=limit,
        fit_workers=fitting, fitting_workers=fitting, styling_workers=fitting,
        api_global_concurrency=limit, case_slot_config=slots, expected=len(items),
        requested=len(items), partition_role=role,
        partition_coordinator=plan['report'], canonical_source_run=str(source))
    settings.update(plan.get(role + '_settings', {}))
    if role == 'discovery':
        settings.update(api_base_url=plan['discovery_base_url'],
            api_model=plan['discovery_model'], api_service_tier=None,
            api_default_image_detail='auto', api_budget_config=None,
            **plan.get('discovery_rate_settings', {}),
            checkpoint_model_policy={
                'cached_model': previous_settings['api_model'],
                'new_upstream_model': plan['discovery_model'],
                'policy': 'Reuse source-bound completed stage checkpoints; new gateway responses record actual model. Legacy logical cache identity retained.'})
    else:
        shutil.copy2(previous / 'private/cpa-worker.crt', run / 'private/cpa-worker.crt')
    save(run / 'settings.json', settings)
    overrides = read(previous / 'shape_resume_settings.json')
    overrides.update(case_slot_config=slots, case_workers=count, glm_concurrency=limit,
                     sam_url=settings['sam_url'], sam_urls=settings['sam_urls'],
                     fitting_workers=fitting, styling_workers=fitting,
                     checkpoint_path_root=str(source / 'shape'))
    save(run / 'shape_resume_settings.json', overrides)
    selection = read(run / 'resume_selection.json')
    pending = [cid for cid in ids if not (source / 'shape/geometry/cases' / cid / 'complete.json').is_file()]
    selection.update(source_run=str(run / 'work'), pending_shape=pending, pending_text=[],
                     preserved={cid: v for cid, v in selection['preserved'].items() if cid in ids},
                     canonical_source_run=str(source), partition_role=role)
    save(run / 'resume_selection.json', selection)
    save(run / 'shape_pending_ids.json', pending)
    save(run / 'source_manifest.json', {'images': items})
    shutil.copy2(Path(__file__), run / 'code/full_recovery/partition_resume.py')
    (run / 'code/partition_worker.py').write_text(
        'from pathlib import Path\nimport runpy,sys\n'
        'from full_recovery.partition_resume import provision_view\n'
        'provision_view(Path(__file__).resolve().parents[1]/"work")\n'
        'runpy.run_module("full_recovery.api_worker",run_name="__main__")\n')
    freeze(run)
    save(run / 'status.json', {'state': 'prepared_partition',
                             'pending_shape': len(pending), 'pending_text': 0})
    return run


def all_shape_products(plan):
    records = []
    for base in (Path(plan['pilot_run']), Path(plan['source_run'])):
        for item in read(base / 'source_manifest.json')['images']:
            case = base / 'shape/geometry/cases' / item['id']
            marker = case / 'complete.json'
            if not marker.is_file():
                return None
            receipt = read(marker)
            if sha(receipt['pptx']) != receipt['pptx_sha256']:
                raise ValueError('Shape receipt failed hash verification: ' + item['id'])
            records.append((base, item))
    if len(records) != plan['merge_expected'] or len({item['id'] for _, item in records}) != len(records):
        raise ValueError('Composition source count differs from the approved scope')
    return records


def merge_collection(plan, target):
    records = all_shape_products(plan)
    if records is None:
        raise ValueError('Wait for all shape products before preparing composition')
    target = Path(target)
    if target.exists():
        raise ValueError('Use a fresh composition source collection')
    for name in ('shape/geometry/cases', 'shape/localization/cases', 'texts/cases'):
        (target / name).mkdir(parents=True, exist_ok=True)
    items = []
    for base, item in records:
        cid = item['id']; items.append(item)
        for rel in ('shape/geometry/cases', 'shape/localization/cases', 'texts/cases'):
            (target / rel / cid).symlink_to(base / rel / cid, target_is_directory=True)
    save(target / 'source_manifest.json', {'images': items})
    save(target / 'texts/manifest.json', {'cases': [{'id': i['id'], 'sha256': i['source_sha256']} for i in items]})
    save(target / 'selection.json', {'policy': 'Explicit pilot3 plus remaining56 current completed products',
                                   'read_only_symlinks': True, 'cases': len(items)})
    return target
