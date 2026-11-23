"""One global image inventory, or import existing detections without model tokens."""
import json
import math
import re
from pathlib import Path

from ..frames import Crop, scale_box
from ..scene import save
from .clients import digest, file_sha
from .prompts import CATEGORIES, DETECTION_SYSTEM, IMAGE_LABELS, detection_prompt, legacy_asset_scope


def validate_inventory(raw, size):
    if not isinstance(raw, dict) or not isinstance(raw.get('assets'), list) or not isinstance(raw.get('uncertain'), list):
        raise ValueError('Expected assets and uncertain arrays')
    seen = set()
    for item in raw['assets']:
        ident = item.get('id', '')
        if not isinstance(ident, str) or not re.fullmatch(r'I\d{4,}', ident) or ident in seen:
            raise ValueError('Image IDs must be unique I0001-style strings')
        seen.add(ident)
        if item.get('category') not in CATEGORIES or item.get('background') not in ('alpha', 'opaque_panel'):
            raise ValueError('Invalid image category/background')
        Crop.from_model(item['bbox'], size)
        for key in ('target', 'keep', 'remove'):
            if not isinstance(item.get(key), str) or not item[key].strip() or len(item[key]) > 700:
                raise ValueError('Missing or excessively long edit field: ' + key)
    return raw


def source_box(bbox, input_size, source_size):
    mapped = scale_box(bbox, input_size, source_size)
    return [max(0, math.floor(mapped[0])), max(0, math.floor(mapped[1])),
            min(source_size[0], math.ceil(mapped[2])+1), min(source_size[1], math.ceil(mapped[3])+1)]


def detect(client, image_path, size):
    prompt = detection_prompt(size)
    identity = {'system': DETECTION_SYSTEM, 'prompt': prompt, 'input_sha256': file_sha(image_path),
                'parameters': client.request_parameters(), 'endpoint': client.base_url}
    tag = 'inventory_' + digest(identity)
    folder = client.root/tag
    if folder.exists():
        if not (folder/'result.json').exists():
            raise RuntimeError('Prior inventory failed/interrupted; inspect calls and start a new run')
        if json.loads((folder/'identity.json').read_text()) != identity:
            raise ValueError('Inventory cache identity mismatch')
        return validate_inventory(json.loads((folder/'result.json').read_text()), size), str(folder.resolve())
    try:
        result = client.call(tag, DETECTION_SYSTEM, prompt, [(IMAGE_LABELS['inventory'], image_path)],
                             lambda raw: validate_inventory(raw, size))
    finally:
        if folder.exists(): save(folder/'identity.json', identity)
    return result, str(folder.resolve())


def import_legacy(path, source_sha, input_size, source_size):
    """Import only source-matched candidates; never import old method selections/masks."""
    items = json.loads(Path(path).read_text())
    assets = []
    for item in items:
        if item['source_sha256'] != source_sha: continue
        b = item['box_source_half_open']
        bbox = scale_box([b[0], b[1], b[2]-1, b[3]-1], source_size, input_size)
        bbox = [max(0, min(v, input_size[i % 2]-1)) for i, v in enumerate(bbox)]
        assets.append({'id': item['id'], 'category': item['category'], 'bbox': bbox,
                       'source_box_half_open': b, 'target': item['label'],
                       **legacy_asset_scope(item['category'], item.get('occlusion')),
                       'provenance': {'import_path': str(Path(path).resolve()), 'import_sha256': file_sha(path),
                                      'uid': item['uid'], 'boundary_review': item.get('boundary_review', {}).get('decision'),
                                      'uncertainty': item.get('uncertainty', '')}})
    if not assets:
        raise ValueError('No source-matched candidates in legacy manifest')
    return validate_inventory({'assets': assets, 'uncertain': [], 'imported': True}, input_size)
