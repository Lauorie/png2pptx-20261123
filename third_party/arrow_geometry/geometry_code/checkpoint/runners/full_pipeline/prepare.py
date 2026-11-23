"""Freeze a fresh complete run, using a reference run only for images/settings.

Preparation is offline: no model or SAM service is invoked.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Standalone frozen runners load the same verified copy implementation.
for _copy_root in (Path(__file__).resolve().parents[3], Path(__file__).resolve().parents[1]/'geometry_code'):
    if (_copy_root/'checked_copy.py').is_file():
        sys.path.insert(0,str(_copy_root));break
from checked_copy import copy_file


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_localization(root, source, ids, cfg):
    """Freeze only completed reviewed inputs; never execute discovery/review."""
    source=source.resolve(); destination=root/'localization'; records=[]
    loc=read(source/'config.json')
    for key in ('review_inherit_existing_type','review_omit_candidate_labels','review_defer_uncertain_to_missing'):
        if loc.get(key)!=cfg.get(key):
            raise ValueError('Reused localization policy does not match development version: '+key)
    for cid in ids:
        case=source/'cases'/cid
        status=read(case/'status.json')
        if status.get('state') not in ('completed','completed_with_issues'):
            raise ValueError('Localization is not ready: '+cid)
        files={'inputs/'+cid+'/source.png':source/'inputs'/cid/'source.png'}
        for name in ('result.json','status.json','reconstruction_handoff.json'):
            files['cases/'+cid+'/'+name]=case/name
        latest=case/'latest.json'
        if latest.exists():
            overlay=read(latest).get('overlay')
            if overlay and (source/overlay).is_file() and source in (source/overlay).resolve().parents:
                files['cases/'+cid+'/current.png']=source/overlay
        for relative,src in files.items():
            digest=sha(src);dst=destination/relative;dst.parent.mkdir(parents=True,exist_ok=True)
            copy_file(src,dst)
            if sha(dst)!=digest or sha(src)!=digest:raise ValueError('Input changed while freezing: '+str(src))
            records.append(dict(path=str(dst.relative_to(root)),source=str(src),sha256=digest))
        handoff=read(case/'reconstruction_handoff.json')
        if handoff.get('deferred_candidates') and not cfg.get('post_geometry_missing_enabled'):
            raise ValueError('Deferred candidates require the missing stage')
        preview=destination/'cases'/cid/'current.png'
        if preview.exists():write(preview.parent/'latest.json',{'overlay':str(preview.relative_to(destination))})
    loc.update(new_discovery_calls=False,new_global_discovery_calls=False,new_local_discovery_calls=False,
               new_local_review_calls=False,reused_review_from=str(source))
    write(destination/'config.json',loc);write(destination/'cases.json',ids)
    write(destination/'input_provenance.json',dict(source=str(source),files=records))
    write(destination/'manifest.json',dict(files={str(p.relative_to(destination)):sha(p) for p in destination.rglob('*') if p.is_file()}))
    return records


def prepare(root, working, reference, selected=None, reuse_localization=None,resume_after_dedup=None,
            resume_before_missing=None, config_overrides=None):
    root, working, reference = root.resolve(), working.resolve(), reference.resolve()
    if root.exists():
        raise ValueError('Choose a new run directory; existing runs are immutable')
    if resume_after_dedup and resume_before_missing:
        raise ValueError('Select exactly one continuation boundary')
    ids = selected or read(reference/'cases.json')
    if len(ids) != len(set(ids)) or not set(ids) <= set(read(reference/'cases.json')):
        raise ValueError('Unknown or duplicate cases')
    cfg = read(working/'config.json')
    if config_overrides:
        cfg.update(read(config_overrides))
    if cfg['thinking'] != {'type': 'enabled'} or cfg.get('reasoning_effort') is not None:
        raise ValueError('Preserve default thinking')
    cfg.pop('reasoning_effort', None)
    for key in ('resume_after_dedup_from', 'resume_before_missing_from'):
        cfg.pop(key, None)
    root.mkdir(parents=True)
    ignore = shutil.ignore_patterns('__pycache__', '.pytest_cache', '*.pyc')
    shutil.copytree(working/'code', root/'geometry_code', ignore=ignore,copy_function=copy_file)
    shutil.copytree(working/'code/checkpoint/runners/full_pipeline', root/'code', ignore=ignore,copy_function=copy_file)
    for name in ('api_slots', 'cpu_slots', 'sam_slots', 'geometry/cases', 'logs', 'case_status', 'reports'):
        (root/name).mkdir(parents=True, exist_ok=True)
    cfg.update(run_label=root.name, source_controller_run=str(root),
               source_provider='fresh_localization_geometry_then_global_and_single_sam_missing',
               source_localization=str(root/'localization'), bbox_discovery_run=str(root/'localization'),
               post_geometry_missing_enabled=True, post_geometry_sam_run=str(root/'sam/missing'),
               post_geometry_sam_scan_enabled=True,
               sam_existing_coverage_threshold=.9, coverage_rounds=0,
               source_geometry=None, shape_overrides=None, style_reuse_from=None,
               model_cache_from=None, sam_cache_from=[],
               request_slot_dir=str(root/'api_slots'), shared_cpu_slot_dir=str(root/'cpu_slots'),
               shared_sam_slot_dir=str(root/'sam_slots'))
    if reuse_localization:
        cfg.update(localization_reuse_from=str(reuse_localization.resolve()),
                   source_provider='reused_reviewed_localization_then_fresh_geometry_missing_style')
    if resume_after_dedup:
        if not reuse_localization:raise ValueError('Geometry continuation requires reused candidate review')
        old=read(resume_after_dedup/'config.json')
        cfg.update(resume_after_dedup_from=str(root/'resume_geometry'),case_workers=min(len(ids),cfg['case_workers']),
                   source_provider='completed_geometry_then_fresh_style_and_independent_heads')
        for key in ('request_slot_dir','shared_cpu_slot_dir','shared_sam_slot_dir'):
            cfg[key]=old[key]
    if resume_before_missing:
        if not reuse_localization or not cfg.get('missing_review_context_enabled'):
            raise ValueError('Review continuation requires reused localization and integrated context inputs')
        cfg.update(resume_before_missing_from=str(root/'resume_missing'),
                   post_geometry_sam_scan_enabled=False,
                   source_provider='frozen_initial_geometry_and_candidates_then_new_missing_review_to_ppt')
    shared = {k: cfg[k] for k in ('model', 'base_url', 'temperature', 'thinking', 'max_tokens',
              'request_timeout_seconds', 'stream_idle_timeout_seconds', 'api_concurrency',
              'case_workers', 'request_max_attempts', 'transport_retries', 'format_retries',
              'request_slot_dir', 'sam_url', 'sam_timeout_seconds')}
    loc = read(reference/'localization/config.json')
    for key in ('frozen_global_from', 'replay_observations_from', 'model_cache_from', 'reasoning_effort'):
        loc.pop(key, None)
    loc.update(shared, new_discovery_calls=True, new_global_discovery_calls=True,
               new_local_discovery_calls=True, new_local_review_calls=True,
               protocol_version='geometry_first_missing_single_bbox')
    # Review policy belongs to the selected development version, not the
    # historical run used for images and the remaining localization settings.
    for key in ('review_inherit_existing_type', 'review_omit_candidate_labels',
                'review_defer_uncertain_to_missing'):
        loc[key] = cfg.get(key, False)
    if reuse_localization:
        loc.update(new_discovery_calls=False,new_global_discovery_calls=False,new_local_discovery_calls=False,new_local_review_calls=False)
    write(root/'preflight/localization_config.json', loc)
    local_code = root/'geometry_code/checkpoint/upstream/localization/code'
    reused=[]
    if reuse_localization:
        reused=freeze_localization(root,reuse_localization,ids,cfg)
    else:
        subprocess.run([sys.executable, '-B', str(local_code/'run.py'), 'prepare', '--run-dir',
                        str(root/'localization'), '--baseline', str(reference/'localization'),
                        '--config', str(root/'preflight/localization_config.json'),
                        '--constraints', str(root/'geometry_code/checkpoint/constraints'),
                        '--cases', *ids], check=True)
    resume_files=[]
    if resume_after_dedup:
        from resume_checkpoint import freeze
        resume_files=freeze(resume_after_dedup,root,ids)
    if resume_before_missing:
        from resume_missing_checkpoint import freeze
        resume_files=freeze(resume_before_missing,root,ids, fresh_style_calls=not cfg.get("stop_after_reconciliation", False))
    sam = read(reference/'sam/missing/config.json')
    sam.update(shared, sam_existing_coverage_threshold=.9, dedup_iou=.9,
               reuse_frozen_sam=False, new_sam_calls=True, new_missing_glm_calls=False,
               new_addition_review_calls=False, sam_addition_scope_review_enabled=False,
               missing_mode='scan_only_after_geometry', bbox_coverage_rejection=True,
               protocol_version='sam_scan_coverage90_then_post_geometry_single_check')
    sam.pop('reasoning_effort', None)
    if resume_before_missing:
        sam.update(reuse_frozen_sam=True, new_sam_calls=False, missing_mode='frozen_candidate_queue')
    (root/'sam/inputs').mkdir(parents=True)
    (root/'sam/missing/cases').mkdir(parents=True)
    (root/'sam/missing/inputs').symlink_to('../inputs', target_is_directory=True)
    write(root/'sam/missing/config.json', sam)
    write(root/'sam/missing/cases.json', ids)
    write(root/'config.json', cfg); write(root/'cases.json', ids)
    for cid in ids:
        job = root/'geometry/jobs'/cid
        job.mkdir(parents=True)
        (job/'geometry_code').symlink_to('../../../geometry_code', target_is_directory=True)
        (job/'cases').symlink_to('../../cases', target_is_directory=True)
        write(job/'config.json', {**cfg, 'case_workers': 1})
        write(job/'cases.json', [cid])
        write(job/'manifest.json', {'files': {n: sha(job/n) for n in ('config.json', 'cases.json')}})
    write(root/'provenance.json', dict(working_code=str(working), reference_settings=str(reference),
          image_source=str((reuse_localization or reference/'localization')/'inputs'), case_count=len(ids),
          reviewed_localization_source=str(reuse_localization.resolve()) if reuse_localization else None,
          prior_glm_answers_reused=bool(reuse_localization), prior_geometry_reused=bool(resume_after_dedup or resume_before_missing), prior_sam_reused=bool(resume_after_dedup or resume_before_missing),
          resumed_geometry_source=str((resume_after_dedup or resume_before_missing).resolve()) if (resume_after_dedup or resume_before_missing) else None,
          fresh_glm_discovery=not bool(reuse_localization), fresh_geometry=not bool(resume_after_dedup or resume_before_missing), fresh_dedup=not bool(resume_after_dedup), fresh_style=not cfg.get('stop_after_reconciliation',False),
          fresh_missing_review=not bool(resume_after_dedup), fresh_missing_discovery=not bool(resume_after_dedup or resume_before_missing),
          preparation_only=True, existing_geometry_readonly_during_missing=True))
    write(root/'preflight/validation.json', dict(preparation_only=True, paid_requests=0,
          reviewed_localization_reused=bool(reuse_localization),
          global_missing_enabled=cfg.get('post_geometry_global_discovery_enabled',True), one_bbox_per_missing_check=True,
          sam_candidate_coverage_threshold=.9, sam_dedup_iou=.9,
          global_proposals_filtered_by_localization=False))
    # Mutable progress/reports must never become startup hash requirements.
    paths = [p for part in ('code','geometry_code','localization/code','localization/inputs',
                             'localization/constraints_snapshot')
             for p in (root/part).rglob('*') if p.is_file()]
    paths += [root/item['path'] for item in reused]
    paths += resume_files
    if reused:paths.append(root/'localization/input_provenance.json')
    paths += [root/n for n in ('config.json','cases.json','provenance.json',
              'localization/config.json','localization/cases.json','localization/manifest.json',
              'sam/missing/config.json','sam/missing/cases.json')]
    paths += list((root/'geometry/jobs').glob('*/config.json'))
    paths += list((root/'geometry/jobs').glob('*/cases.json'))
    paths += list((root/'geometry/jobs').glob('*/manifest.json'))
    project=working.parent.parent
    for name in ('AGENTS.md','README.md','当前进度.md','PROJECT.json','docs/TASK.md'):
        src=project/name
        if src.exists():
            dst=root/'constraints_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True)
            copy_file(src,dst);paths.append(dst)
    files = {str(p.relative_to(root)): sha(p) for p in paths}
    write(root/'manifest.json', {'files': files})
    subprocess.run([sys.executable, '-B', str(root/'code/run_full125.py'),
                    '--run-dir', str(root), '--report-only'], check=True)
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--working', required=True, type=Path)
    parser.add_argument('--reference-run', required=True, type=Path)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--reuse-localization', type=Path, help='Reuse a completed reviewed localization run; start at geometry')
    parser.add_argument('--resume-after-dedup',type=Path,help='Freeze completed geometry, missing and reconciliation; continue at rendering/style')
    parser.add_argument('--resume-before-missing',type=Path,help='Freeze initial geometry and discovery queues; continue at missing review through final PPT')
    parser.add_argument('--config-overrides',type=Path,help='Run-only non-secret configuration overrides')
    args = parser.parse_args()
    print(prepare(args.run_dir, args.working, args.reference_run, args.cases,args.reuse_localization,args.resume_after_dedup,args.resume_before_missing,args.config_overrides))
