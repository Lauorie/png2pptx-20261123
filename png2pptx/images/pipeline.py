"""Resumable image stage using the geometry project's frames and scene exporter."""
import argparse
from collections import Counter
import fcntl
import json
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

from ..frames import resized_size
from ..scene import new_scene, save
from .clients import EDIT_ENDPOINT, EDIT_MODEL, SenseNovaClient, digest, file_sha, make_detector
from .detect import import_legacy, source_box, validate_inventory
from .coverage import detect_with_coverage
from .boundary import review_boundaries
from .planning import plan_edit, NeedsContext
from .matte import checkerboard, choose_key, chroma_alpha
from .scene import merge_image_stage


def read(path):
    return json.loads(Path(path).read_text())


def prepare_edit(source, asset, folder, context_padding=.15):
    box = asset['source_box_half_open']
    margin = max(8, round(max(box[2]-box[0], box[3]-box[1]) * context_padding))
    context_box = [max(0, box[0]-margin), max(0, box[1]-margin),
                   min(source.width, box[2]+margin), min(source.height, box[3]+margin)]
    crop, context = source.crop(box), source.crop(context_box)
    crop.save(folder/'source_crop.png'); context.save(folder/'source_context.png')
    writer_size = resized_size(context.size, 2048)
    context.resize(writer_size, Image.Resampling.LANCZOS).save(folder/'writer_input_2048.png')
    key, key_info = choose_key(crop)
    resized = resized_size(context.size, 1024)
    canvas_size = tuple(max(64, math.ceil(v/16)*16) for v in resized)
    pad = [(canvas_size[i]-resized[i])//2 for i in (0, 1)]
    canvas = Image.new('RGB', canvas_size, tuple(key) if asset.get('background') == 'alpha' else 'white')
    canvas.paste(context.resize(resized, Image.Resampling.LANCZOS), tuple(pad))
    canvas.save(folder/'input.png')
    scale = [resized[i]/context.size[i] for i in (0, 1)]
    target = [pad[i % 2] + (v-context_box[i % 2])*scale[i % 2] for i, v in enumerate(box)]
    placement = [context_box[i]-pad[i]/scale[i] for i in (0, 1)]
    placement += [placement[i]+canvas_size[i]/scale[i] for i in (0, 1)]
    frame = {'source_box_half_open': box, 'context_box_source_half_open': context_box,
             'edit_input_size': list(canvas.size), 'resized_context_size': list(resized),
             'source_to_edit': {'scale_xy': scale, 'padding_xy': pad,
                                'formula': '(source-context_origin+0.5)*scale-0.5+padding'},
             'target_box_edit_half_open': target, 'placement_source_half_open': placement,
             'writer_input_size': list(writer_size), 'global_input_size': list(resized_size(source.size, 2048)),
             'context_padding_fraction': context_padding, 'minimum_context_padding_px': 8,
             'editor_long_edge': 1024, 'key_rgb': key, 'key_selection': key_info}
    save(folder/'transform.json', frame)
    return frame


def extract_one(source, asset, folder, editor, planner=None, context_padding=.15):
    folder.mkdir(parents=True, exist_ok=True)
    if planner is None:
        raise ValueError('A visual prompt writer is required; no generic-template fallback')
    for context_attempt in range(2):
        local = folder/f'context_{context_attempt:02d}'; local.mkdir(parents=True, exist_ok=True)
        padding = context_padding if context_attempt==0 else max(.45, context_padding*2)
        frame = prepare_edit(source, asset, local, padding)
        try:
            asset, prompt = plan_edit(planner, source, asset, frame, local)
        except NeedsContext:
            if context_attempt==0: continue
            raise
        for path in local.iterdir():
            if path.is_file(): shutil.copyfile(path, folder/path.name)
        break
    asset['provenance'] = {**asset.get('provenance', {}), 'context_attempt': context_attempt,
                           'context_padding_used': padding}
    (folder/'prompt.txt').write_text(prompt)
    raw_path, call = editor.edit(folder/'input.png', prompt)
    shutil.copyfile(raw_path, folder/'edit_raw.png')
    with Image.open(raw_path) as opened: raw = opened.copy()
    if asset['background'] == 'alpha':
        rgba, matte = chroma_alpha(raw, frame['key_rgb'], source.crop(asset['source_box_half_open']))
    else:
        # Whole panels are opaque. Only exterior canvas is transparent, including white content.
        rgba = raw.convert('RGBA')
        a = Image.new('L', rgba.size, 0)
        b = frame['target_box_edit_half_open']
        rect = [max(0, math.floor(b[0])), max(0, math.floor(b[1])),
                min(rgba.width, math.ceil(b[2])), min(rgba.height, math.ceil(b[3]))]
        ImageDraw.Draw(a).rectangle([rect[0], rect[1], rect[2]-1, rect[3]-1], fill=255)
        rgba.putalpha(a)
        pixels = np.asarray(rgba.crop(rect).convert('RGB'), dtype=float)
        key_fraction = float((np.linalg.norm(pixels-np.array(frame['key_rgb']), axis=2) < 50).mean())
        matte = {'method': 'opaque_panel_rectangle', 'panel_key_fraction': key_fraction,
                 'quality_flags': ['key_inside_opaque_panel'] if key_fraction > .005 else []}
    if frame['key_selection']['source_near_key_fraction'] > .001:
        matte['quality_flags'].append('source_color_close_to_key')
    if rgba.getbbox() is None:
        raise ValueError('Empty result after alpha extraction; raw response retained')
    rgba.save(folder/'rgba.png')
    checkerboard(rgba).save(folder/'alpha_preview.png')
    save(folder/'matte.json', matte)
    return {**asset, 'status': 'needs_review' if matte['quality_flags'] else 'extracted_unverified',
            'rgba': str((folder/'rgba.png').resolve()), 'rgba_sha256': file_sha(folder/'rgba.png'),
            'placement_source_half_open': frame['placement_source_half_open'],
            'export_enabled': True, 'visual_fidelity_verified': False,
            'quality_flags': matte['quality_flags'],
            'provenance': {**asset.get('provenance', {}), 'editor_call': call,
                           'input_sha256': file_sha(folder/'input.png'), 'prompt_sha256': file_sha(folder/'prompt.txt'),
                           'transform': str((folder/'transform.json').resolve()), 'alpha_method': matte['method']}}


def resolve_source(args, scene):
    if args.image: return args.image.resolve()
    original = Path(scene['image']['original'])
    if original.is_absolute(): return original
    roots = [args.scene_root] if args.scene_root else [args.scene.parent, *args.scene.parents]
    for root in roots:
        candidate = root/original
        if candidate.is_file(): return candidate.resolve()
    raise FileNotFoundError('Cannot resolve scene original image; provide --image or --scene-root')


def usage_summary(root):
    totals = Counter(); counts = {}; missing = 0
    for stage in ('detector', 'boundary_review', 'prompt_writer'):
        responses = 0
        for path in (root/'calls'/stage).glob('*/response_0.json'):
            raw = read(path); responses += 1
            usage = raw.get('usage')
            if not usage: missing += 1
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                if isinstance((usage or {}).get(key), int): totals[key] += usage[key]
        counts[stage+'_calls_with_response'] = responses
    return {**counts, 'usage_missing': missing, **dict(totals),
            'edit_requests': len(list((root/'calls/editor').glob('*/request.json'))),
            'editor_usage_note': 'Image editing service usage is retained per call, not assumed to be zero tokens.'}


def run(args):
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(args, root)


def run_locked(args, root):
    scene = read(args.scene) if args.scene else None
    source_path = resolve_source(args, scene)
    source_sha = file_sha(source_path)
    with Image.open(source_path) as im: source = im.convert('RGB')
    source_size = list(source.size); input_size = list(resized_size(source.size, 2048))
    if scene and (scene['image'].get('source_sha256') != source_sha or scene['image']['source_size'] != source_size):
        raise ValueError('Geometry scene does not match supplied source image')
    code_files = sorted(Path(__file__).parent.glob('*.py'))
    code_files += [Path(__file__).parents[1]/name for name in ('frames.py', 'render.py', 'scene.py')]
    code_files += sorted((Path(__file__).parents[1]/'api').glob('*.py'))
    code_hashes = {str(p.relative_to(Path(__file__).parents[1])): file_sha(p) for p in code_files}
    configuration = {'source_sha256': source_sha, 'source_size': source_size, 'input_size': input_size,
                     'model': args.model, 'max_tokens': args.max_tokens, 'edit_endpoint': args.edit_endpoint,
                     'edit_model': args.edit_model, 'steps': args.steps, 'seed': args.seed,
                     'image_guidance': args.image_guidance,
                     'context_padding': args.context_padding, 'prompt_max_tokens': args.prompt_max_tokens,
                     'coverage_rounds': args.coverage_rounds,
                     'max_local_calls': args.max_local_calls,
                     'boundary_review_rounds': args.boundary_review_rounds,
                     'max_boundary_calls': args.max_boundary_calls,
                     'boundary_max_tokens': args.boundary_max_tokens,
                     'only': args.only, 'detect_only': args.detect_only,
                     'scene_sha256': file_sha(args.scene) if args.scene else None,
                     'candidates_sha256': file_sha(args.candidates or args.legacy_assets) if args.candidates or args.legacy_assets else None,
                     'candidate_format': 'legacy' if args.legacy_assets else 'inventory' if args.candidates else 'detect',
                     'code_sha256': code_hashes}
    if (root/'config.json').exists():
        if not args.resume: raise FileExistsError('Run exists; use --resume with unchanged arguments')
        if read(root/'config.json') != configuration: raise ValueError('Frozen run configuration/code changed; choose a new output directory')
    else:
        save(root/'config.json', configuration)
        for path in code_files:
            dest = root/'code_snapshot'/path.relative_to(Path(__file__).parents[1]); dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    shutil.copyfile(source_path, root/'original.png')
    source.resize(input_size, Image.Resampling.LANCZOS).save(root/'input_2048.png')
    image_meta = {'id': scene['image']['id'] if scene else source_path.parent.name if source_path.name=='original.png' else source_path.stem,
                  'original': str(root/'original.png'), 'input': str(root/'input_2048.png'),
                  'source_size': source_size, 'input_size': input_size, 'source_sha256': source_sha,
                  'input_sha256': file_sha(root/'input_2048.png')}
    stage = {'schema_version': 'png2pptx.images.1', 'image': image_meta, 'assets': {},
             'asset_root': str(root), 'status': 'detecting', 'layer_policy': 'images_before_geometry',
             'coverage_verified': False, 'configuration_hash': digest(configuration)}
    save(root/'status.json', {'state': 'detecting'})
    try:
        if args.legacy_assets:
            inventory = import_legacy(args.legacy_assets, source_sha, input_size, source_size)
            origin = str(args.legacy_assets.resolve())
        elif args.candidates:
            payload = read(args.candidates)
            if payload.get('source_sha256') != source_sha or payload.get('input_size') != input_size:
                raise ValueError('Candidates must have matching source_sha256 and input_size')
            inventory = validate_inventory(payload, input_size); origin = str(args.candidates.resolve())
        else:
            client = make_detector(root/'calls/detector', args.model, args.max_tokens)
            inventory, origin = detect_with_coverage(client, root/'input_2048.png', input_size,
                                                     root/'detection', args.coverage_rounds,
                                                     source_path=root/'original.png', max_local_calls=args.max_local_calls)
        if args.only and not set(args.only).issubset({a['id'] for a in inventory['assets']}):
            raise ValueError('--only contains unknown image IDs')
        save(root/'inventory_before_boundary_review.json',
             {**inventory, 'source_sha256': source_sha, 'input_size': input_size, 'provenance': origin})
        stage['status'] = 'boundary_review'
        save(root/'image_stage.json', stage)
        save(root/'status.json', {'state': 'boundary_review', 'assets': len(inventory['assets'])})
        reviewer = (make_detector(root/'calls/boundary_review', args.model, args.boundary_max_tokens)
                    if inventory['assets'] and args.max_boundary_calls else None)
        inventory = review_boundaries(reviewer, root/'original.png', inventory, input_size,
                                      root/'boundary_review', max_calls=args.max_boundary_calls,
                                      calls_per_asset=args.boundary_review_rounds)
        save(root/'inventory.json', {**inventory, 'source_sha256': source_sha, 'input_size': input_size, 'provenance': origin})
        stage['inventory_provenance'] = origin; stage['uncertain'] = inventory['uncertain']
        stage['coverage_status'] = inventory.get('coverage_status', 'imported_inventory')
        stage['boundary_review'] = inventory['boundary_review']
        stage['status'] = 'detected' if args.detect_only else 'extracting'
        items = inventory['assets']
        all_ids = {a['id'] for a in items}
        if args.only and not set(args.only).issubset(all_ids): raise ValueError('--only contains unknown image IDs')
        stage['skipped_ids'] = sorted(all_ids-set(args.only)) if args.only else []
        items = [a for a in items if not args.only or a['id'] in args.only]
        editor = SenseNovaClient(root/'calls/editor', args.edit_endpoint, args.edit_model, args.steps, args.seed,
                                 image_guidance=args.image_guidance)
        planner = make_detector(root/'calls/prompt_writer', args.model, args.prompt_max_tokens) if not args.detect_only else None
        overlay = source.resize(input_size, Image.Resampling.LANCZOS); draw = ImageDraw.Draw(overlay)
        for item in items:
            ident = item['id']; asset = dict(item)
            # Raw model coordinates stay in inventory.json; transforms never overwrite them.
            asset['bbox_global_pixels'] = item['bbox']
            asset['source_box_half_open'] = item.get('source_box_half_open') or source_box(item['bbox'], input_size, source_size)
            asset['provenance'] = {**item.get('provenance', {}), 'inventory': origin}
            draw.rectangle(item['bbox'], outline='#ff00cc', width=3); draw.text(tuple(item['bbox'][:2]), ident, fill='#ff00cc')
            folder = root/'assets'/ident
            save(root/'status.json', {'state': 'prompt_and_edit', 'current_id': ident, 'completed': len(stage['assets']), 'total': len(items)})
            if args.detect_only:
                result = {**asset, 'status': 'detected', 'export_enabled': False}
            elif asset['boundary_review']['status'] != 'model_review_complete':
                result = {**asset, 'status': 'boundary_unresolved', 'export_enabled': False,
                          'error': asset['boundary_review']['reason']}
            else:
                try:
                    result = extract_one(source, asset, folder, editor, planner, args.context_padding)
                except Exception as exc:
                    result = {**asset, 'status': 'failed', 'error': str(exc), 'export_enabled': False}
            stage['assets'][ident] = result
            save(folder/'result.json', result)
            save(root/'image_stage.json', stage)
            print(json.dumps({'id': ident, 'status': result['status'], 'quality_flags': result.get('quality_flags', [])}), flush=True)
        overlay.save(root/'inventory_overlay.png')
        failures = sum(a['status']=='failed' for a in stage['assets'].values())
        boundary_pending = stage['boundary_review']['unresolved_ids']
        stage['status'] = ('partial' if failures or boundary_pending or stage['skipped_ids'] or stage['coverage_status']=='unresolved'
                           else 'detected' if args.detect_only else 'extracted_unverified')
        stage['usage'] = usage_summary(root)
        save(root/'image_stage.json', stage)
        scene = scene or new_scene(image_meta, [])
        scene = merge_image_stage(scene, stage)
        save(root/'scene.json', scene)
        if not args.detect_only:
            from ..render import svg, write_pptx
            import cairosvg
            content = svg(scene)
            (root/'recovered.svg').write_text(content)
            cairosvg.svg2png(bytestring=content.encode(), write_to=str(root/'recovered.png'))
            write_pptx(scene, root/'recovered.pptx')
        save(root/'status.json', {'state': ('finished_with_failures' if failures else
                                          'finished_with_unresolved' if boundary_pending else 'finished'),
                                 'assets': len(items), 'failures': failures, 'stage_status': stage['status'],
                                 'boundary_review': stage['boundary_review'],
                                 'visual_fidelity_verified': False, 'usage': stage['usage']})
        return 1 if failures else 0
    except Exception as exc:
        stage['status'] = 'failed'; stage['error'] = str(exc); stage['usage'] = usage_summary(root)
        save(root/'image_stage.json', stage)
        save(root/'status.json', {'state': 'failed', 'error': str(exc), 'usage': stage['usage']})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--scene', type=Path, help='Geometry scene to extend without changing fitted objects')
    parser.add_argument('--scene-root', type=Path)
    imports = parser.add_mutually_exclusive_group()
    imports.add_argument('--candidates', type=Path, help='Reuse a source-matched inventory.json')
    imports.add_argument('--legacy-assets', type=Path, help='Reuse old recovery/assets.json detections only')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='glm-5.3-flash', help='GLM-5.3 vision route (same family as geometry stage)')
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--prompt-max-tokens', type=int, default=1536)
    parser.add_argument('--coverage-rounds', type=int, default=2, help='Bounded local repair rounds after full coverage audits')
    parser.add_argument('--max-local-calls', type=int, default=16, help='Total local search requests per figure, excluding bounded schema retries')
    parser.add_argument('--boundary-review-rounds', type=int, default=2, help='Independent review calls per asset, including verification/context')
    parser.add_argument('--max-boundary-calls', type=int, default=64, help='Total independent review calls per figure; zero leaves targets unresolved')
    parser.add_argument('--boundary-max-tokens', type=int, default=1536, help='Output budget per independent boundary review')
    parser.add_argument('--context-padding', type=float, default=.15)
    parser.add_argument('--only', nargs='+', help='Explicit small pilot subset, recorded as incomplete coverage')
    parser.add_argument('--detect-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--edit-endpoint', default=EDIT_ENDPOINT)
    parser.add_argument('--edit-model', default=EDIT_MODEL)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--image-guidance', type=float, default=3.)
    args = parser.parse_args()
    if not args.image and not args.scene: parser.error('Provide --image or --scene')
    if min(args.max_tokens, args.prompt_max_tokens, args.boundary_max_tokens, args.steps) < 1:
        parser.error('Token and step limits must be positive')
    if args.boundary_review_rounds < 1 or args.max_boundary_calls < 0:
        parser.error('Boundary review rounds must be positive and total calls nonnegative')
    if args.coverage_rounds < 0 or args.max_local_calls < 1 or not math.isfinite(args.context_padding) or not 0 <= args.context_padding <= 1:
        parser.error('Invalid coverage or crop expansion setting')
    return run(args)
