"""Precompute SAM masks on upsampled square border tiles and build per-part line evidence.

``--configs`` lets a seeded pilot subset compare tile sides and prompt sets; the
chosen configuration is then run for every case. No case-specific branches.
Outputs per case: ``<out>/<config>/<case>/tiles/*`` (tile PNG, SAM predictions)
and ``line_evidence.json`` (per-part tile list + target mask audit).
"""
import argparse
import hashlib
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / 'src_v2'), str(HERE.parents[1] / 'vendor')]
from clients import read, save, SamClient  # noqa: E402
from shape_integration.styling.geometry import parts as scene_parts, part_key  # noqa: E402
from shape_integration.styling import lines, sam_lines  # noqa: E402

CLASS = {'rectangle': 'rectangle', 'rounded_rectangle': 'rectangle', 'ellipse': 'ellipse', 'path': 'polygon'}
STYLE_CATEGORIES = ('line', 'rectangle', 'circle', 'ellipse', 'polygon', 'other')
SIZE = 1008


def prompts_for(kind, mode):
    if mode == 'line':
        return [{'name': 'line'}]
    # A native ellipse also represents a circle. Query both detector families
    # without an aspect-ratio classification boundary; source/contour matching
    # still decides which returned masks support the actual native part.
    families = ['circle', 'ellipse'] if kind == 'ellipse' else [CLASS.get(kind, 'other')]
    if mode == 'class':
        return [{'name': name} for name in families]
    return [{'name': name} for name in ['line', *families]]


def masks_from(prediction, threshold, categories):
    out = []
    for p in prediction['predictions']:
        if p['score'] < threshold or p['category'] not in categories:
            continue
        with Image.open(p['mask_path']) as im:
            out.append({'id': p['id'], 'category': p['category'], 'score': p['score'],
                        'mask': np.asarray(im.convert('L')) > 127})
    return out


def run_case(case, config, root, sam, pool):
    folder = root / case['id']
    sam_config=getattr(getattr(sam,'__self__',None),'config',{})
    identity={'source':hashlib.sha256(Path(case['original']).read_bytes()).hexdigest(),
              'scene':hashlib.sha256(Path(case['scene']).read_bytes()).hexdigest(),'config':config,
              'sam':{k:sam_config.get(k) for k in ('sam_threshold','external_sam_epoch','external_sam_checkpoint_sha256','sam_checkpoint_sha256')},
              'protocol':'shared_source_tile_line_evidence.2'}
    path=folder/'line_evidence.json'
    if path.exists():
        cached=read(path)
        previous=cached.get('identity',{})
        if previous!=identity:
            # The aggregate depends on scene membership and contour planning.
            # Individual tile requests depend only on source pixels, crop and
            # SAM identity. Preserve the old index, then reuse exact tile calls
            # while rebuilding the new scene index under this single writer.
            if previous.get('source')!=identity['source'] or previous.get('sam')!=identity['sam']:
                raise ValueError('SAM source/model changed; use a fresh evidence directory')
            archive=folder/'previous_evidence'/hashlib.sha256(path.read_bytes()).hexdigest()
            archive.mkdir(parents=True,exist_ok=True)
            path.rename(archive/'line_evidence.json')
        else:
            for part in cached['parts'].values():
                for tile in part['tiles']:
                    if not (Path(tile['dir'])/'tile.png').is_file() or (tile.get('target_mask') and not Path(tile['target_mask']).is_file()):
                        raise ValueError('Committed SAM line evidence has a missing tile/mask')
            return {'case':case['id'],'parts':len(cached['parts']),'tiles':cached['sam_requests'],
                    'seconds':cached['sam_wall_seconds'],'reused_completed_evidence':True}
    image = Image.open(case['original']).convert('RGB')
    arr = np.asarray(image, float)
    scene = read(case['scene'])
    table = scene_parts(scene)
    jobs = {}
    plans = {}
    for key, part in table.items():
        guess = lines.measure(arr, part, guess_width=2.)
        width = guess.get('width_px') or 2.
        boxes = sam_lines.plan_tiles(part, image.size, config['side'])
        plans[key] = {'guess': {k: guess.get(k) for k in ('present', 'width_px', 'dash', 'coverage')},
                      'width': width, 'tiles': []}
        for box in boxes:
            name = f'{box[0]}_{box[1]}_{box[2]}_' + '-'.join(p['name'] for p in prompts_for(part['kind'], config['prompts']))
            tile = folder / 'tiles' / name
            plans[key]['tiles'].append({'box': box, 'dir': str(tile)})
            if name not in jobs:
                jobs[name] = (box, tile, prompts_for(part['kind'], config['prompts']))

    def infer(box, tile, prompts):
        tile.mkdir(parents=True, exist_ok=True)
        png = tile / 'tile.png'
        if not png.exists():
            sam_lines.tile_image(image, box, SIZE).save(png)
        return sam(png, tile / 'sam', prompts)
    started = time.monotonic()
    futures = {pool.submit(infer, *job): name for name, job in jobs.items()}
    results = {}
    for future in as_completed(futures):
        results[futures[future]] = future.result()
    elapsed = time.monotonic() - started
    evidence = {}
    for key, plan in plans.items():
        part = table[key]; half = max(1.5, plan['width'] / 2)
        rows = []
        for tile in plan['tiles']:
            name = Path(tile['dir']).name
            masks = masks_from(results[name], config['score_threshold'], config['categories'])
            distance = sam_lines.path_band(part, tile['box'], SIZE, half)
            union, audit = sam_lines.select_targets(masks, distance, half, SIZE / tile['box'][2])
            target_mask = Path(tile['dir']) / ('target_' + hashlib.sha256(key.encode()).hexdigest()[:16] + '.png')
            if union is not None:
                Image.fromarray(union.astype(np.uint8) * 255).save(target_mask)
            rows.append({**tile, 'target': union is not None, 'masks': audit,
                         'target_mask': str(target_mask) if union is not None else None})
        evidence[key] = {'guess': plan['guess'], 'tiles': rows}
    save(folder / 'line_evidence.json', {'case': case['id'], 'config': config, 'size': SIZE, 'identity':identity,
                                         'sam_requests': len(jobs), 'sam_wall_seconds': elapsed, 'parts': evidence})
    return {'case': case['id'], 'parts': len(evidence), 'tiles': len(jobs), 'seconds': elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', required=True, help='name in sam_configs.json')
    parser.add_argument('--pilot', type=int, default=0, help='seeded random subset size (0 = all cases)')
    parser.add_argument('--ids', nargs='*')
    args = parser.parse_args()
    from clients import require_cluster
    require_cluster()
    run = args.run_dir.resolve()
    config = read(run / 'sam_configs.json')[args.config]
    cases = read(run / 'inputs.json')['cases']
    if args.ids:
        cases = [c for c in cases if c['id'] in args.ids]
    elif args.pilot:
        cases = random.Random(20261015).sample(cases, args.pilot)
    root = run / 'sam_lines' / args.config
    sam = SamClient(run)
    rows = []
    with ThreadPoolExecutor(max_workers=sam.queue.qsize() * 2) as pool:
        for case in cases:
            rows.append(run_case(case, config, root, sam, pool))
            print('SAM_CASE', json.dumps(rows[-1]), flush=True)
    save(root / 'completion.json', {'config': config, 'cases': rows, 'completed_at': time.time()})


if __name__ == '__main__':
    main()
