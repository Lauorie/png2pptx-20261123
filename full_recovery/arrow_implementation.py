"""Pin an arrow implementation independently from explicitly reused products."""
from pathlib import Path
import shutil

from .prepare import read, save, sha


def freeze(source, target):
    source, target = Path(source).resolve(), Path(target)
    if source.name == 'geometry_code':
        source = source.parent
    if not (source/'geometry_code').is_dir() or not (source/'code/prepare.py').is_file():
        raise ValueError('Arrow implementation needs geometry_code and its preparation driver')
    target.mkdir(parents=True)
    files = {}
    for name in ('geometry_code', 'code'):
        shutil.copytree(source/name, target/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
        for path in (target/name).rglob('*'):
            if path.is_file():
                digest = sha(source/path.relative_to(target))
                if sha(path) != digest:
                    raise ValueError('Arrow code changed while freezing')
                files[str(path.relative_to(target))] = digest
    for name in ('config.json', 'localization/config.json', 'sam/missing/config.json'):
        data = read(source/name)
        save(target/name, data)
    save(target/'source.json', {'implementation_source': str(source), 'files': files,
                               'inference_enabled': False, 'arrow_products_imported': False})
    return target/'source.json'
