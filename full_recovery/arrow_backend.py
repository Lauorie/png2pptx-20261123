"""Freeze current arrow methods; delivered PPTs are audit references, never inputs."""
from pathlib import Path
import shutil

from .prepare import read, save, sha


def replace_once(path, before, after):
    text = path.read_text()
    if text.count(before) != 1:
        raise ValueError('Arrow integration anchor changed: ' + str(path))
    path.write_text(text.replace(before, after, 1))


def freeze(project, source, delivery, target):
    source, delivery, target = map(Path, (source, delivery, target))
    config = read(source/'config_point_geometry_mainline.json')
    files = [*source.glob('*.py'), *source.joinpath('lib').glob('*.py'),
             *source.joinpath('transport').glob('*.py'), *source.joinpath('shape_core').rglob('*.py'),
             *source.joinpath('tests').glob('*.py')]
    originals = {}
    for path in files + [source/'BASELINE.json', source/'shape_core/SOURCE.json']:
        relative = path.relative_to(source)
        dest = target/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        originals[str(relative)] = sha(path)

    # The named delivery contains manual corrections, not an executable branch.
    # Its sibling frozen code supplies the actual whole-route identity method.
    dedup = delivery.parent/'code/dedup.py'
    manifest = read(delivery.parent/'code_manifest.json')
    if sha(dedup) != manifest['code/dedup.py']:
        raise ValueError('Delivered arrow identity source changed')
    shutil.copy2(dedup, target/'ownership_dedup.py')
    shutil.copy2(project/'full_recovery/arrow_finalize.py', target/'arrow_finalize.py')
    replace_once(target/'pipeline.py',
        "    with locks['cpu']:allrows=publish(clean,results,out)\n    rendered=False\n",
        "    if config.get('whole_path_identity_review',False):\n"
        "        import arrow_finalize\n"
        "        progress('whole_path_identity')\n"
        "        results,identity_issues=arrow_finalize.finish(clean,results,out,config,model,locks)\n"
        "        issues.extend(identity_issues)\n"
        "    with locks['cpu']:allrows=publish(clean,results,out)\n    rendered=False\n")
    # A total deadline is a transport failure just like an idle timeout. Keep
    # the existing bounded retry, rather than introducing a repair process.
    replace_once(target/'transport/client.py',
        'retryable=isinstance(exc,httpx.TransportError)) from None',
        'retryable=isinstance(exc,(httpx.TransportError,TimeoutError))) from None')
    config.update(whole_path_identity_review=True, identity_review_batch_size=3,
                  style_max_tokens=8192, style_reasoning_effort='low',
                  style_request_timeout_seconds=360, style_transport_retries=1)
    save(target/'METHOD_SOURCE.json', {
        'geometry_source': str(source), 'geometry_config': str(source/'config_point_geometry_mainline.json'),
        'source_sha256': originals,
        'whole_path_identity_source': str(dedup), 'whole_path_identity_sha256': sha(dedup),
        'delivery_audit_reference': str(delivery), 'delivery_manifest_sha256': sha(delivery/'manifest.json'),
        'manual_corrections': ['task_0931 endpoint survivor choice', 'task_0246 one-path style and independent triangle'],
        'manual_artifacts_imported': False, 'case_specific_parameters_added': False,
        'general_fixes': ['native endpoint evidence before deletion', 'preserve unique endpoint heads',
                          'bounded low-effort style requests', 'retry total and idle timeouts'],
        'composition_logic_changed': False,
    })
    frozen = {str(p.relative_to(target)): sha(p) for p in target.rglob('*') if p.is_file()}
    return config, frozen
