"""Rebuild per-part masks from saved selections, without rerunning SAM."""
import argparse
import copy
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from clients import read, save, sha


def repair(run, config):
    source = run / 'sam_lines' / config
    dest = run / 'sam_lines' / (config + '_part_owned')
    rows = []
    for path in sorted(source.glob('*/line_evidence.json')):
        data = copy.deepcopy(read(path))
        out = dest / data['case']
        masks_dir = out / 'target_masks'
        masks_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for key, part in data['parts'].items():
            for tile in part['tiles']:
                if not tile['target']:
                    continue
                prediction = read(Path(tile['dir']) / 'sam/predictions.json')
                selected = {m['id'] for m in tile['masks'] if m['target']}
                mask_path = masks_dir / (hashlib.sha256((key + tile['dir']).encode()).hexdigest() + '.png')
                if not mask_path.exists():
                    union = np.zeros((data['size'], data['size']), bool)
                    found = set()
                    for mask in prediction['predictions']:
                        if mask['id'] in selected:
                            with Image.open(mask['mask_path']) as im:
                                union |= np.asarray(im.convert('L')) > 127
                            found.add(mask['id'])
                    if found != selected or not union.any():
                        raise ValueError('Missing selected mask: ' + str(path))
                    Image.fromarray(union.astype(np.uint8) * 255).save(mask_path)
                tile['target_mask'] = str(mask_path)
                n += 1
        data['ownership_repair'] = {'source': str(path), 'source_sha256': sha(path),
                                    'method': 'union_of_saved_selected_prediction_ids'}
        save(out / 'line_evidence.json', data)
        rows.append({'case': data['case'], 'masks': n})
        print('REPAIRED', data['case'], n, flush=True)
    save(dest / 'completion.json', {'config': config, 'cases': rows})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--configs', nargs='+', default=['side96_both'])
    a = p.parse_args()
    from clients import require_cluster
    require_cluster()
    for config in a.configs:
        repair(a.run_dir.resolve(), config)
