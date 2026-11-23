"""Verify the exact tested checkpoint and SAM runtime before loading."""
from pathlib import Path
import hashlib
import json

from engine import select_categories


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_assets(config):
    path = Path(config['checkpoint_path'])
    print('Verifying frozen checkpoint SHA256: ' + str(path), flush=True)
    actual = sha256(path)
    if actual != config['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint SHA256 mismatch')
    for name, asset in config['runtime_assets'].items():
        if sha256(asset['path']) != asset['sha256']:
            raise ValueError('Training runtime SHA256 mismatch: ' + name)
    manifest = json.loads(Path(config['dataset_manifest']).read_text())
    select_categories(config['categories'], manifest['categories'])
    return actual
