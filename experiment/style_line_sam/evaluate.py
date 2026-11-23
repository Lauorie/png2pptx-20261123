"""Compare two styling arms on identical frozen geometry.

1. Pixel fidelity (model-free): along each fixed contour, the same stroke
   measurement is applied to SOURCE and to each arm's actual LibreOffice render;
   fill colour error is the median CIELAB distance inside each closed part.
   The stroke measurement is also what the new arm uses, so it is reported with
   that caveat next to the blind review.
2. Blind pairwise GLM review per case, both candidate orders, with zoom crops of
   the regions where the two renders differ most.
3. Cost from the request traces, and visual panels.
"""
import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / 'src_v2'), str(HERE.parents[1] / 'vendor')]
from clients import read, save, sha, StyleRuntime, Slots, Endpoints  # noqa: E402
from judge_parser import parse_judge, validate_judge  # noqa: E402
from shape_integration.styling import lines  # noqa: E402
from shape_integration.styling.geometry import parts as scene_parts, footprint, raster, bounds  # noqa: E402

FONT = os.environ.get('STYLE_FONT', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')


def lab(rgb):
    c = np.asarray(rgb, float) / 255.
    c = np.where(c > .04045, ((c + .055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[.4124, .3576, .1805], [.2126, .7152, .0722], [.0193, .1192, .9505]])
    xyz = c @ m.T / np.array([.95047, 1., 1.08883])
    f = np.where(xyz > .008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def case_render(run, arm, ident):
    folder = run / 'arms' / arm / 'cases' / ident / 'styling'
    return folder, folder / 'native.png'


def stroke_metrics(src_m, out_m):
    row = {'source_present': src_m.get('present'), 'render_present': out_m.get('present')}
    if src_m.get('present') and out_m.get('present'):
        ink_s = src_m['width_px'] * src_m['contrast']; ink_r = out_m['width_px'] * out_m['contrast']
        row['log_ink_ratio'] = abs(math.log(max(ink_r, 1e-3) / max(ink_s, 1e-3)))
        if src_m['width_px'] >= 2 and out_m['width_px'] >= 2:
            row['color_delta_e'] = float(np.linalg.norm(lab(lines.rgb(src_m['color'])) - lab(lines.rgb(out_m['color']))))
        if src_m.get('dash_reliable'):
            row['pattern_match'] = (src_m['dash'] == 'solid') == (out_m.get('dash') == 'solid')
            row['exact_pattern_match'] = src_m['dash'] == out_m.get('dash')
            if src_m['dash'] != 'solid' and 'coverage' in out_m:
                row['dashed_coverage_error'] = abs(src_m['coverage'] - out_m['coverage'])
            if src_m['dash'] != 'solid' and out_m.get('dash_pattern_px') and src_m.get('dash_pattern_px'):
                row['log_period_ratio'] = abs(math.log(sum(out_m['dash_pattern_px']) / sum(src_m['dash_pattern_px'])))
    return row


def pixel_case(run, arms, case):
    scene = read(case['scene']); table = scene_parts(scene)
    source = np.asarray(Image.open(case['original']).convert('RGB'), float)
    renders = {a: np.asarray(Image.open(case_render(run, a, case['id'])[1]).convert('RGB'), float) for a in arms}
    if any(r.shape != source.shape for r in renders.values()):
        raise ValueError('Native render dimensions differ from the source: ' + case['id'])
    src_lab = lab(source); rend_lab = {a: lab(r) for a, r in renders.items()}
    cache = run / 'evaluation_measurements' / case['id']
    common_identity = {'scene': sha(case['scene']), 'source': sha(case['original']),
                       'code': sha(Path(lines.__file__)), 'policy': 'source_guess_2_then_source_width.1'}

    def measured(name, pixels, path, guesses=None):
        record = cache / (name + '.json')
        identity = {**common_identity, 'image': sha(path)}
        if record.exists():
            saved = read(record)
            if saved['identity'] == identity:
                return saved['parts']
        values = {key: lines.measure(pixels, part, guess_width=(guesses[key].get('width_px') or 2.) if guesses else 2.)
                  for key, part in table.items()}
        save(record, {'identity': identity, 'parts': values})
        return values

    source_lines = measured('source', source, case['original'])
    render_lines = {a: measured('render_' + a, renders[a], case_render(run, a, case['id'])[1], source_lines) for a in arms}
    rows = {}
    for key, part in table.items():
        src_m = source_lines[key]
        row = {'object': part['object_id'], 'part': part['id']}
        for arm in arms:
            out_m = render_lines[arm][key]
            row[arm] = stroke_metrics(src_m, out_m)
        geometry, _ = footprint(part)
        if geometry.area > 16:
            b = [int(v) for v in bounds(part)]
            box = [max(0, b[0] - 2), max(0, b[1] - 2), min(source.shape[1], b[2] + 3), min(source.shape[0], b[3] + 3)]
            if box[2] > box[0] and box[3] > box[1]:
                inner = ndimage.binary_erosion(raster(geometry, box), iterations=4)
                if inner.sum() >= 20:
                    s = src_lab[box[1]:box[3], box[0]:box[2]][inner]
                    for arm in arms:
                        r = rend_lab[arm][box[1]:box[3], box[0]:box[2]][inner]
                        row[arm]['fill_delta_e'] = float(np.median(np.linalg.norm(s - r, axis=1)))
        rows[key] = row
    return rows


def stat(function, values):
    values = list(values)
    return float(function(values)) if values else None


def summarize_pixels(per_case, arms):
    out = {}
    for arm in arms:
        rows = [r[arm] for case in per_case.values() for r in case.values()]
        presence = [r for r in rows if r['source_present'] is not None and r['render_present'] is not None]
        pick = lambda field: [r[field] for r in rows if field in r]
        out[arm] = {
            'parts': len(rows),
            'eligible_counts': {'presence': len(presence), **{field: len(pick(field)) for field in
                                ('log_ink_ratio', 'color_delta_e', 'pattern_match', 'exact_pattern_match',
                                 'dashed_coverage_error', 'log_period_ratio', 'fill_delta_e')}},
            'stroke_presence_agreement': stat(np.mean, [bool(r['source_present']) == bool(r['render_present']) for r in presence]),
            'median_log_ink_ratio': stat(np.median, pick('log_ink_ratio')),
            'ink_within_25pct': stat(np.mean, [v <= math.log(1.25) for v in pick('log_ink_ratio')]),
            'median_stroke_delta_e': stat(np.median, pick('color_delta_e')),
            'solid_vs_dashed_agreement': stat(np.mean, pick('pattern_match')),
            'exact_line_kind_agreement': stat(np.mean, pick('exact_pattern_match')),
            'median_dashed_coverage_error': stat(np.median, pick('dashed_coverage_error')),
            'dashed_period_within_20pct': stat(np.mean, [v <= math.log(1.2) for v in pick('log_period_ratio')]),
            'median_fill_delta_e': stat(np.median, pick('fill_delta_e')),
            'fill_delta_e_over_10': stat(np.mean, [v > 10 for v in pick('fill_delta_e')]),
        }
    a, b = arms
    paired = {'ink': [0, 0, 0], 'fill': [0, 0, 0], 'dash_duty': [0, 0, 0]}
    for case in per_case.values():
        for r in case.values():
            for metric, field, tol in (('ink', 'log_ink_ratio', .05), ('fill', 'fill_delta_e', 1.),
                                       ('dash_duty', 'dashed_coverage_error', .05)):
                if field in r[a] and field in r[b]:
                    d = r[a][field] - r[b][field]
                    paired[metric][0 if d < -tol else 1 if d > tol else 2] += 1
    out['paired_parts'] = {k: {f'{a}_better': v[0], f'{b}_better': v[1], 'equal': v[2]} for k, v in paired.items()}
    return out


def diff_regions(a, b, count=3, pad=24):
    diff = np.abs(np.asarray(a, float) - np.asarray(b, float)).sum(axis=-1)
    diff = ndimage.gaussian_filter(diff, 3)
    mask = diff > 30
    labels, n = ndimage.label(mask)
    if not n:
        return []
    weights = ndimage.sum(diff, labels, range(1, n + 1))
    boxes = []
    for index in np.argsort(-weights)[:count]:
        ys, xs = np.nonzero(labels == index + 1)
        x0, y0, x1, y1 = xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad
        side = max(x1 - x0, y1 - y0, 96)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        boxes.append([int(max(0, cx - side / 2)), int(max(0, cy - side / 2)),
                      int(min(a.width, cx + side / 2)), int(min(a.height, cy + side / 2))])
    return boxes


def triplet(images, labels, box, height=420):
    tiles = []
    font = ImageFont.truetype(FONT, 18)
    for im, label in zip(images, labels):
        crop = im.crop(box); scale = height / crop.height
        crop = crop.resize((max(1, round(crop.width * scale)), height), Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', (crop.width, height + 28), 'white'); canvas.paste(crop, (0, 28))
        ImageDraw.Draw(canvas).text((4, 4), label, font=font, fill='black'); tiles.append(canvas)
    board = Image.new('RGB', (sum(t.width for t in tiles) + 10 * (len(tiles) - 1), tiles[0].height), (128, 128, 128))
    x = 0
    for t in tiles:
        board.paste(t, (x, 0)); x += t.width + 10
    return board


def fit(image, long_edge=2048):
    scale = min(1., long_edge / max(image.size))
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS) if scale < 1 else image


JUDGE = 'Compare two editable reconstructions of the same figure against the source. Return one JSON object only.'


def judge_prompt():
    return {'task': 'Both candidates share identical shape geometry; only shape styles differ. Text, photos and arrows are NOT rendered in either candidate. Decide which candidate better reproduces the SOURCE shape styles: fill presence/color/gradient and stroke presence/color/thickness/dash pattern.',
            'rules': ['Ignore missing text, photos, arrows and any geometry difference from the source; both candidates share them.',
                      'Weigh visible, material style errors (wrong fill color, missing or extra outline, wrong dash pattern, clearly wrong thickness) over tiny shade differences.',
                      'Zoom images show SOURCE | CANDIDATE 1 | CANDIDATE 2 for the regions where the candidates differ most.',
                      'Answer tie when the differences are negligible or mixed with similar importance.'],
            'output': {'preferred': 'candidate_1|candidate_2|tie', 'confidence': 'high|medium|low',
                       'evidence': 'short visible evidence', 'candidate_1_errors': ['short'], 'candidate_2_errors': ['short']}}


def review_case(run, out, arms, case, runtime):
    folder = out / 'pairwise' / case['id']
    source = Image.open(case['original']).convert('RGB')
    rendered = {a: Image.open(case_render(run, a, case['id'])[1]).convert('RGB') for a in arms}
    boxes = diff_regions(rendered[arms[0]], rendered[arms[1]])
    verdicts = []
    for order in (arms, arms[::-1]):
        tag = '_'.join(order)
        root = folder / tag; root.mkdir(parents=True, exist_ok=True)
        images = [('SOURCE (complete original figure)', source), ('CANDIDATE 1', rendered[order[0]]), ('CANDIDATE 2', rendered[order[1]])]
        paths = []
        for i, (label, im) in enumerate(images):
            path = root / f'image_{i}.png'
            if not path.exists():
                fit(im).save(path)
            paths.append((label, path))
        for i, box in enumerate(boxes):
            path = root / f'zoom_{i}.png'
            if not path.exists():
                triplet([source, rendered[order[0]], rendered[order[1]]], ['SOURCE', 'CANDIDATE 1', 'CANDIDATE 2'], box).save(path)
            paths.append((f'ZOOM {i + 1}: SOURCE | CANDIDATE 1 | CANDIDATE 2', path))
        try:
            saved_call = root / 'calls/judge_00'
            response = saved_call / 'response_0.json'
            if response.exists() and not (saved_call / 'result.json').exists():
                choice = read(response)['choices'][0]
                if choice.get('finish_reason') != 'length':
                    try:
                        recovered = parse_judge(choice['message'].get('content') or '')
                    except ValueError:
                        pass
                    else:
                        save(saved_call / 'result.json', recovered)
                        save(saved_call / 'parse_recovery.json', {
                            'parser': 'top_level_schema_valid_final.1', 'response_sha256': sha(response),
                            'at': time.time(), 'note': 'Reparsed final content; reasoning not used. Original error preserved.'})
            result = runtime.call(root / 'calls', 'judge', JUDGE, judge_prompt(), paths, validate_judge, 12000)
            winner = {'candidate_1': order[0], 'candidate_2': order[1], 'tie': 'tie'}[result['preferred']]
            verdicts.append({'order': list(order), 'winner': winner, 'confidence': result.get('confidence'),
                             'evidence': result['evidence'], 'raw': result})
        except Exception as exc:
            verdicts.append({'order': list(order), 'winner': None, 'error': str(exc)})
    winners = [v['winner'] for v in verdicts]
    outcome = winners[0] if winners[0] == winners[1] and winners[0] is not None else 'order_inconsistent'
    result = {'case': case['id'], 'outcome': outcome, 'verdicts': verdicts, 'zoom_boxes': boxes}
    save(folder / 'result.json', result)
    return result


def panel(run, arms, case, out):
    source = Image.open(case['original']).convert('RGB')
    rendered = [Image.open(case_render(run, a, case['id'])[1]).convert('RGB') for a in arms]
    board = triplet([source, *rendered], ['SOURCE', *arms], [0, 0, source.width, source.height], height=min(700, source.height))
    boxes = diff_regions(rendered[0], rendered[1])
    zooms = [triplet([source, *rendered], ['SOURCE', *arms], b, height=300) for b in boxes]
    width = max([board.width] + [z.width for z in zooms]); height = board.height + sum(z.height + 10 for z in zooms) + 40
    canvas = Image.new('RGB', (width, height), 'white')
    ImageDraw.Draw(canvas).text((6, 8), case['id'], font=ImageFont.truetype(FONT, 22), fill='black')
    canvas.paste(board, (0, 40)); y = 40 + board.height + 10
    for z in zooms:
        canvas.paste(z, (0, y)); y += z.height + 10
    canvas.save(out)
    return out


def costs(run, arm):
    root = run / 'arms' / arm / 'cases'
    rows = {'calls': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'service_seconds': 0., 'by_mode': {}}
    for timing in root.rglob('timing.json'):
        t = read(timing); usage = t.get('usage') or {}
        mode = next((p.split('_')[0] for p in timing.parts[::-1] if p.split('_')[0] in
                     ('select', 'critic', 'refine', 'relations', 'group')), 'other')
        rows['calls'] += 1; rows['prompt_tokens'] += usage.get('prompt_tokens', 0)
        rows['completion_tokens'] += usage.get('completion_tokens', 0); rows['service_seconds'] += t.get('service_seconds', 0)
        rows['by_mode'][mode] = rows['by_mode'].get(mode, 0) + 1
    summaries = [read(p) for p in root.glob('*/styling/summary.json')]
    rows['states'] = {}
    for s in summaries:
        rows['states'][s['state']] = rows['states'].get(s['state'], 0) + 1
    rows['critic_accepted'] = sum(s['critic_accepted'] for s in summaries)
    rows['targeted_parts'] = sum(s['targeted_parts'] for s in summaries)
    render_roots = [folder for p in root.glob('*/styling/summary.json')
                    for folder in [p.parent / 'rules_baseline', *(p.parent / 'snapshots').glob('snapshot_*')]]
    rows['renders'] = sum(1 for folder in render_roots if (folder / 'export.json').exists()
                          and read(folder / 'export.json').get('native_rendered'))
    walls = [read(p).get('arm_wall_seconds', 0) for p in root.glob('*/styling/arm_case.json')]
    rows['case_wall_seconds_sum'] = float(sum(walls))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--arms', nargs=2, default=['original', 'v2'])
    parser.add_argument('--skip-review', action='store_true')
    parser.add_argument('--ids', nargs='*')
    parser.add_argument('--tag')
    parser.add_argument('--pixel-workers', type=int, default=2)
    args = parser.parse_args()
    from clients import require_cluster
    require_cluster()
    run = args.run_dir.resolve(); arms = args.arms
    cases = read(run / 'inputs.json')['cases']
    if args.ids:
        cases = [c for c in cases if c['id'] in args.ids]
    out = run / ('evaluation_' + (args.tag or '_vs_'.join(arms))); out.mkdir(exist_ok=True)
    missing = {c['id']: [a for a in arms if not case_render(run, a, c['id'])[1].exists()] for c in cases}
    save(out / 'skipped_cases.json', {k: v for k, v in missing.items() if v})
    cases = [c for c in cases if not missing[c['id']]]
    if not cases:
        raise SystemExit('No case has a render in both arms')
    cache = out / 'pixel_cases'
    cache.mkdir(exist_ok=True)
    def measured(case):
        from clients import sha
        path = cache / (case['id'] + '.json')
        identity = {'scene': sha(case['scene']), 'source': sha(case['original']),
                    'measurement_code': sha(Path(lines.__file__)),
                    'metric_protocol': 'stroke_fill_pattern_duty.2',
                    'renders': {a: sha(case_render(run, a, case['id'])[1]) for a in arms}}
        if path.exists() and read(path)['identity'] == identity:
            return read(path)['parts']
        parts = pixel_case(run, arms, case)
        save(path, {'identity': identity, 'parts': parts})
        print('PIXEL_DONE', case['id'], flush=True)
        return parts
    with ThreadPoolExecutor(max_workers=args.pixel_workers) as pool:
        per_case = dict(zip([c['id'] for c in cases], pool.map(measured, cases)))
    save(out / 'pixel_parts.json', per_case)
    pixel = summarize_pixels(per_case, arms)
    save(out / 'pixel_summary.json', pixel)
    (out / 'panels').mkdir(exist_ok=True)
    pages = [panel(run, arms, case, out / 'panels' / f'{case["id"]}.png') for case in cases]
    images = [Image.open(p).convert('RGB') for p in pages]
    images[0].save(out / 'panels.pdf', save_all=True, append_images=images[1:], resolution=100)
    reviews = []
    if not args.skip_review:
        settings = {'concurrency': 12, 'max_tokens': 12000, 'timeout_seconds': 1200, 'seed': 42, 'reasoning_effort': 'high'}
        from shape_integration.transport import RequestTimeout
        runtime = StyleRuntime(run, settings, RequestTimeout, endpoints=Endpoints(run), slots=Slots(12))
        runtime = runtime.for_stage(shape_schema_retries=2)
        runtime.parse_response = parse_judge
        with ThreadPoolExecutor(max_workers=12) as pool:
            futures = [pool.submit(review_case, run, out, arms, c, runtime) for c in cases]
            for f in as_completed(futures):
                reviews.append(f.result())
    tally = {}
    for r in reviews:
        tally[r['outcome']] = tally.get(r['outcome'], 0) + 1
    save(out / 'summary.json', {'arms': arms, 'cases': [c['id'] for c in cases], 'pixel': pixel, 'pairwise': tally,
                                'costs': {a: costs(run, a) for a in arms}, 'created_at': time.time()})
    print(json.dumps({'pixel': pixel, 'pairwise': tally}, indent=1))


if __name__ == '__main__':
    main()
