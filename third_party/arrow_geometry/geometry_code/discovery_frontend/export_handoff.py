#!/usr/bin/env python3
"""Export discovery ownership tasks, without executing geometry or PPT rendering."""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

from PIL import Image


def export(root):
    sys.path.insert(0, str(root/'code'))
    from core import crop_box
    from evidence import frame
    from region_partition import local_view
    from transport.common import read_json as read, write_json as write, sha256
    from visual_labels import render
    cfg = read(root/'config.json')
    case_ids = read(root/'cases.json')
    if not all((root/'cases'/cid/'finished.json').exists() for cid in case_ids):
        raise RuntimeError('All selected cases must finish before final handoff export')
    checks = []
    for cid in case_ids:
        case = root/'cases'/cid
        result = read(case/'result.json')
        baseline = {o['id']: o for o in read(case/'baseline_result.json')['objects']}
        source = root/'inputs'/cid/'source.png'
        size = Image.open(source).size
        active = result['objects']
        active_ids = {o['id'] for o in active}
        assert len(active_ids) == len(active)
        regions = []
        for ledger in result['region_ledger']:
            parent = baseline[ledger['parent_id']]
            assert set(ledger['active_owner_ids']) <= active_ids
            if ledger['commit']:
                assert parent['id'] not in active_ids
            else:
                assert parent['id'] in active_ids
            regions.append({'id': parent['id'], 'display_id': parent['display_id'],
                            'bbox_native': parent['bbox_native'], 'source_locator_hint': parent['locator'],
                            'role': ledger['parent_role'], 'owner_ids': ledger['active_owner_ids'],
                            'complete_transfer_accepted': ledger['commit'],
                            'remaining_conflicts': ledger['conflicts']})
        tasks, padded = [], []
        retained={x['parent_id']:x for x in result['region_ledger'] if not x['commit']}
        for o in active:
            b = o['bbox_native']
            focus = [max(0, b[0]-3), max(0, b[1]-3), min(size[0]-1, b[2]+3), min(size[1]-1, b[3]+3)]
            context = [0,0,*size] if o['id'] in retained else crop_box([focus], size, 48)
            view = local_view(size, context, cfg)
            assert 0 <= context[0] < context[2] <= size[0] and 0 <= context[1] < context[3] <= size[1]
            assert 1080 <= max(view.input_size) <= 2048
            parents = [r['id'] for r in regions if r['complete_transfer_accepted'] and o['id'] in r['owner_ids']]
            tasks.append({'target_id': o['id'], 'display_id': o['display_id'], 'kind': o['kind'],
                          'original_identity_bbox_native': b, 'focus_bbox_native': focus,
                          'focus_margin_native_px': 3, 'context_view': view.metadata(size),
                          'source_image': str(source), 'source_image_sha256': sha256(source),
                          'identity_hint': o.get('locator', ''),
                          'hint_coordinates_are_untrusted': True,
                          'context_region_ids': parents,
                          'retained_group_scope_hint': retained[o['id']]['plan']['target_scope'] if o['id'] in retained else None,
                          'retained_group_full_image_context': o['id'] in retained,
                          'render_owner': True, 'bbox_is_hint_not_pixel_clip': True,
                          'instruction': 'Reconstruct only this complete identity. Other visible lines and context regions are read-only. Use context pixels to retain the full shaft and heads; do not clip the path to the estimated bbox.'})
            padded.append({**copy.deepcopy(o), 'bbox_native': focus})
        write(case/'reconstruction_handoff.json', {'case_id': cid, 'schema_version': 'discovery-owner-handoff-v1',
              'tasks': tasks, 'regions': regions,
              'input_result_sha256': sha256(case/'result.json'),
              'geometry_executed': False, 'ppt_executed': False,
              'physical_completeness_and_duplicate_freedom_measured': False,
              'policy': 'Only tasks become reconstruction jobs. Context-only regions must never create another render object. Retained groups remain responsible until complete transfer is verified.'})
        global_view = frame(read(root/'inputs'/cid/'views.json')['views'][0]['transform'])
        render(root/'inputs'/cid/'G/target.png', padded, case/'handoff_focus.png', global_view)
        checks.append({'case_id': cid, 'task_count': len(tasks),
                       'context_only_regions': sum(r['role']=='context_only' for r in regions),
                       'retained_groups': sum(r['role']=='active_reconstruction_group' for r in regions),
                       'all_owner_refs_resolve': True, 'one_task_per_active_id': True,
                       'model_input_size_bounds_verified': True})
    write(root/'analysis/handoff_verification.json', {'cases': checks,
          'exporter_sha256': sha256(Path(__file__)), 'geometry_executed': False})
    print(json.dumps(checks, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    export(parser.parse_args().run_dir.resolve())
