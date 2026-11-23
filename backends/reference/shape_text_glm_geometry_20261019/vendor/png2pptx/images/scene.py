"""Image assets coexist with (and never masquerade as) fitted shape objects."""
import copy
from pathlib import Path


def merge_image_stage(scene, stage):
    if scene['schema_version'] != 'png2pptx.scene.1':
        raise ValueError('Expected geometry scene schema png2pptx.scene.1')
    for field in ('source_sha256', 'source_size'):
        if not scene['image'].get(field) or scene['image'][field] != stage['image'][field]:
            raise ValueError('Image stage source mismatch: ' + field)
    result = copy.deepcopy(scene)
    assets = result.setdefault('image_assets', {})
    for ident, asset in stage['assets'].items():
        if ident in assets and assets[ident] != asset:
            raise ValueError('Conflicting image identity: ' + ident)
        assets[ident] = copy.deepcopy(asset)
    result['image_stage'] = {k: copy.deepcopy(v) for k, v in stage.items() if k not in ('assets', 'image')}
    result['pending_categories']['images'] = stage['status']
    # No deletion/ownership inference from bbox overlap: the future router must decide that.
    return result


def exportable_assets(scene, accepted_only=False):
    for asset in scene.get('image_assets', {}).values():
        if not asset.get('export_enabled', True) or not asset.get('rgba'):
            continue
        if accepted_only and asset.get('status') != 'accepted_by_user':
            continue
        yield asset


def asset_path(scene, asset, asset_root=None):
    path = Path(asset['rgba'])
    return path if path.is_absolute() else Path(asset_root or scene['image_stage']['asset_root']) / path
