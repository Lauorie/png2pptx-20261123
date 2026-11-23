"""Object-level ablation with fixed identity/context and a paired critic check."""
import argparse
import copy
import fcntl
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

from PIL import Image, ImageDraw
from core import read, save, sha
from run import sam_index
from models import Models
from runtime import ForwardRuntime
from report import render
from shape_integration.sam_guidance import GuidedSamEvidence
from shape_integration.fitting.control import process_object
from shape_integration.fitting.critic import check_object


def render_fit(path, source_size, crop):
    fit = read(path)
    scene = {'source_size': source_size, 'parts': {p['id']: p for p in fit['parts']}}
    return render(scene, crop, 900)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = read(args.config)
    base = Path(config['baseline_root'])
    items = {i['id']: i for i in read(base / 'inputs/manifest.json')['images']}
    selected = read(args.manifest)['objects']
    save(root / 'manifest.json', {'objects': selected, 'config': config})
    pipe = read(base / 'runs' / config['baseline_run'] / 'manifest.json')['config']
    pipe = copy.deepcopy(pipe)
    pipe['fitting'].update(max_revisions=1, recover_clipped_partial=True)
    models = Models(config)
    indices = {cid: sam_index(items[cid], config, models, root / 'sam_cache' / cid)
               for cid in sorted({v['case'] for v in selected})}

    def task(spec):
        cid, oid = spec['case'], spec['object']
        item = items[cid]
        saved = Path(spec['record'])
        record = read(saved)
        previous = read(saved.parent / 'revision_00/job.json')
        candidate = record['candidate']
        assets = read(saved.parent / 'image_assets.json')
        inventory = read(saved.parent.parent.parent / 'candidates.json')['objects']
        target = root / (cid + '_' + oid)
        target.mkdir(exist_ok=True)
        results = {}
        variants = config.get('probe_variants', [{'name': mode, 'mode': mode} for mode in ['none', 'bbox', 'mask']])
        for variant in variants:
            mode, label = variant['mode'], variant['name']
            evidence_options = {**config['evidence'], **variant.get('evidence', {})}
            evidence = None if mode == 'none' else GuidedSamEvidence(indices[cid]['groups'], assets, mode=mode, **evidence_options)
            if evidence:
                evidence.set_inventory(inventory)
            runtime = ForwardRuntime(models, pipe, saved.parent, evidence)
            folder = target / label
            started = time.monotonic()
            value = process_object(folder, candidate, item, base, runtime, pipe['fitting'],
                                   namespace='probe/' + target.name + '/' + label,
                                   input_context=previous.get('input_context', {}), image_assets=assets)
            results[label] = {'state': value['state'], 'revision': value['selected_revision'],
                             'seconds': time.monotonic() - started, 'record': str(folder / 'record.json')}
        # Isolate the critic information change on EXACTLY the same fitted input.
        critic_base = config.get('probe_critic_base', 'mask')
        mask_record = read(target / critic_base / 'record.json') if critic_base in results else {}
        chosen = mask_record.get('selected_revision')
        critics = {}
        if chosen:
            original = target / critic_base / chosen
            import shutil
            for mode in ['mask', 'critic']:
                dest = target / ('critic_' + mode)
                dest.mkdir(exist_ok=True)
                for name in ['job.json', 'clean.png', 'fit_local.json', 'fit_original.json', 'frame.json',
                             'critic_source.png', 'fit_overlay.png', 'inspection.json', 'image_assets.json']:
                    path = original / name
                    if path.exists() and not (dest / name).exists():
                        shutil.copy2(path, dest / name)
                if not (dest / 'image_assets.json').exists():
                    save(dest / 'image_assets.json', assets)
                provider = GuidedSamEvidence(indices[cid]['groups'], assets, mode=mode, **config['evidence'])
                provider.set_inventory(inventory)
                runtime = ForwardRuntime(models, pipe, saved.parent, provider)
                critics[mode] = check_object(read(original / 'job.json'), dest, runtime, pipe['fitting'])
        crop = [float('inf'), float('inf'), 0, 0]
        for mode, value in results.items():
            if value['revision']:
                b = read(target / mode / value['revision'] / 'job.json')['frame']['source_crop_half_open']
                crop = [min(crop[0], b[0]), min(crop[1], b[1]), max(crop[2], b[2]), max(crop[3], b[3])]
        if crop[2] > crop[0] and crop[3] > crop[1]:
            crop = [int(v) for v in crop]
            with Image.open(base / item['original']) as im:
                clean = im.convert('RGB').crop(crop)
            size = (900, max(2, round(900 * clean.height / clean.width)))
            images = [('Source', clean.resize(size, Image.Resampling.LANCZOS))]
            for mode, value in results.items():
                im = render_fit(target / mode / value['revision'] / 'fit_original.json', item['source_size'], crop) if value['revision'] else Image.new('RGB', size, 'white')
                images.append((mode, im.resize(size, Image.Resampling.LANCZOS)))
            board = Image.new('RGB', (size[0] * 2, (size[1] + 30) * ((len(images)+1)//2)), 'white')
            draw = ImageDraw.Draw(board)
            for i, (label, im) in enumerate(images):
                x, y = (i % 2) * size[0], (i // 2) * (size[1] + 30)
                board.paste(im, (x, y + 30)); draw.text((x + 4, y + 8), label, fill='black')
            board.save(target / 'comparison.png')
        receipt = {'case': cid, 'object': oid, 'label': candidate['label'], 'results': results, 'critic_ablation': critics}
        save(target / 'results.json', receipt)
        return receipt

    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(task, s): s for s in selected}
        for future in as_completed(pending):
            try:
                value = future.result()
            except Exception as exc:
                value = {**pending[future], 'error': str(exc)}
            results.append(value)
            save(root / 'progress.json', {'completed': len(results), 'total': len(selected), 'results': results})
            print('PROBE', value['case'], value['object'], value.get('error', 'done'), flush=True)
    save(root / 'completion.json', {'results': results, 'errors': sum('error' in r for r in results)})
    if any('error' in r for r in results):
        raise RuntimeError('Probe errors; see completion.json')


if __name__ == '__main__':
    main()
