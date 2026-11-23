"""Bind explicitly selected arrow products without importing other branch caches."""
from pathlib import Path

from .prepare import read, save, sha


def bind(root, items, output, *, pending_ids=()):
    from PIL import Image, ImageChops
    root = Path(root).resolve()
    base = (root/'geometry/cases' if (root/'geometry/cases').is_dir() else
            root/'cases' if (root/'cases').is_dir() else root)
    if not base.is_dir():
        raise ValueError('Selected arrow cases directory is missing')
    pending_ids=set(pending_ids)
    if not pending_ids <= {item['id'] for item in items}:
        raise ValueError('Pending arrow IDs must be requested source cases')
    records = [];pending=[]
    for item in items:
        case = base/item['id']
        files = {'arrow_scene': case/'scene.json', 'arrow_pptx': case/'arrows.pptx',
                 'arrow_clean': case/'clean.png'}
        missing=[name for name,path in files.items() if not path.is_file() or path.stat().st_size==0]
        if missing and item['id'] in pending_ids:
            pending.append({'id':item['id'],'state':'pending_arrow_input','missing':missing,
                            'source_sha256':item['source_sha256'],'selected_root':str(root)})
            continue
        for name, path in files.items():
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError('Selected arrow product missing: '+str(path))
            if not path.resolve().is_relative_to(root):
                raise ValueError('Selected arrow product resolves outside its declared run: '+str(path))
        if sha(item['original']) != item['source_sha256']:
            raise ValueError('Original changed before arrow binding: '+item['id'])
        if read(files['arrow_scene'])['size'] != item['source_size']:
            raise ValueError('Selected arrow coordinate grid differs: '+item['id'])
        with Image.open(item['original']) as original, Image.open(files['arrow_clean']) as clean:
            if original.size != clean.size or ImageChops.difference(
                    original.convert('RGB'), clean.convert('RGB')).getbbox() is not None:
                raise ValueError('Selected arrow source pixels differ: '+item['id'])
        records.append({'id': item['id'], 'source_sha256': item['source_sha256'],
            'paths': {name: str(path.resolve()) for name, path in files.items()},
            'sha256': {name: sha(path) for name, path in files.items()}})
    result = {'protocol': 'explicit_existing_arrow_products.1', 'root': str(root),
              'selection': 'explicit --arrow-products-run; no automatic latest selection',
              'arrow_inference_enabled': False, 'records': records, 'pending':pending}
    save(output, result)
    return result


def verify(manifest):
    root = Path(manifest['root']).resolve()
    for record in manifest['records']:
        for name, value in record['paths'].items():
            path = Path(value).resolve()
            if not path.is_relative_to(root) or sha(path) != record['sha256'][name]:
                raise ValueError('Explicit arrow input changed: '+record['id']+'/'+name)
