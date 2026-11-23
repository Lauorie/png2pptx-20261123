"""Apply cached OCR localization, source-ink tightening and hard CPU rollback.

Generate the report with compare_group_ocr.py first. This CPU-only stage never
changes transcription, grouping, font family, rotation or colors. Fresh output
is required; baseline artifacts are immutable.
"""
import argparse
from pathlib import Path
import os
import shutil
import sys
import threading

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT/'.deps'), str(ROOT/'src'), str(ROOT/'vendor')]
import numpy as np
from PIL import Image, ImageDraw
from shape_integration.common import read, save, sha
from shape_integration.texts import options, layout, export, ocr_refinement
from shape_integration.texts import render_feedback as feedback
from shape_integration.texts.bbox_color_review import retain_nonregressing
from shape_integration.styling.views import resample_crop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'image', 'ocr-report', 'ocr-crop-dir', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--target-ids', nargs='+')
    parser.add_argument('--ink-threshold', type=float, default=.20)
    parser.add_argument('--min-confidence', type=float, default=.8)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    source = Image.open(args.image).convert('RGB')
    before = Image.open(args.baseline/'native.png').convert('RGB')
    if source.size != before.size:
        raise ValueError('Canvas mismatch')
    if read(args.baseline/'provenance.json')['image_sha256'] != sha(args.image):
        raise ValueError('Baseline belongs to a different source')
    report = read(args.ocr_report)
    for name, digest in report.get('input_sha256', {}).items():
        if name not in ('clean.png', 'frame.json') or sha(args.ocr_crop_dir/name) != digest:
            raise ValueError('OCR input provenance mismatch')
    frame = read(args.ocr_crop_dir/'frame.json')
    if report['frame'] != frame or frame['source_size'] != list(source.size):
        raise ValueError('OCR crop/frame mismatch')
    crop = resample_crop(source, frame['box'], frame['size'], frame['scale'], frame['offset'])
    clean = Image.open(args.ocr_crop_dir/'clean.png').convert('RGB')
    if not np.array_equal(np.asarray(crop), np.asarray(clean)):
        raise ValueError('OCR crop does not belong to supplied source')
    # Bind legacy cached reports (which lack hashes) to the actual resized OCR input.
    recorded = Image.open(args.ocr_report.parent/'input_000.png').convert('RGB')
    if not np.array_equal(np.asarray(clean.resize(recorded.size, Image.Resampling.LANCZOS)), np.asarray(recorded)):
        raise ValueError('Cached OCR input differs from source crop')
    scene = read(args.baseline/'scene.json'); rows = scene['texts']
    by_id = {r['id']:r for r in rows}
    wanted = args.target_ids or list(report['matches'])
    if len(set(wanted)) != len(wanted) or set(wanted)-set(by_id):
        raise ValueError('Invalid target IDs')
    paths = [args.image, args.baseline/'scene.json', args.baseline/'native.png',
             args.baseline/'text.pptx', args.ocr_report, args.ocr_crop_dir/'clean.png']
    hashes = {str(p.resolve()):sha(p) for p in paths}
    args.output.mkdir(parents=True)
    save(args.output/'provenance.json', {'sources':hashes, 'image_sha256':sha(args.image),
         'ocr_input_verified':True, 'target_ids':wanted, 'api_calls':0,
         'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in
                         [Path(__file__), ROOT/'src/shape_integration/texts/ocr_refinement.py']}})
    records = {}; raw = {}; tight = {}; errors = []; used = set()
    for ident in wanted:
        try:
            detection = report['matches'][ident]['chosen']
            ocr_refinement.validate_match(by_id[ident], detection, source.size, args.min_confidence)
            key = tuple(round(v, 2) for v in detection['source_bbox'])
            if key in used:
                raise ValueError('OCR occurrence reused by multiple targets')
            used.add(key)
            record, mask = ocr_refinement.tighten(source, detection['source_bbox'], args.ink_threshold)
            record['ocr_detection'] = detection; records[ident] = record
            folder = args.output/'localization'/ident; folder.mkdir(parents=True)
            Image.fromarray(mask.astype('uint8')*255).save(folder/'ink_mask.png')
            source.crop(record['roi']).save(folder/'source_crop.png')
            save(folder/'bounds.json', record)
            raw[ident] = [ocr_refinement.proposal(by_id[ident], detection['source_bbox'], record)]
            if record['status'] == 'located':
                tight[ident] = [ocr_refinement.proposal(by_id[ident], record['bbox_2d'], record)]
        except (ValueError, KeyError, TypeError) as exc:
            errors.append({'id':ident, 'reason':str(exc)})
    save(args.output/'localization.json', records)
    save(args.output/'localization_errors.json', errors)
    print('Source ink localization:', {i:r['status'] for i,r in records.items()}, flush=True)
    slots = threading.BoundedSemaphore(1)
    opt = options(read(ROOT/'config.json'))
    opt.update(pixel_reflow_enabled=False, source_font_selection=False, glyph_registration_enabled=False)
    def fit(updates, label):
        if not updates:
            return {}
        fitted, failures = layout.fit_all([r for rs in updates.values() for r in rs],
            args.output/'typography'/label, slots, opt, source=args.image)
        if failures:
            raise RuntimeError('Native font measurement failed: '+str(failures))
        return {r['id']:[r] for r in fitted}
    def render(updates, folder):
        print('Rendering', folder.name, 'edited targets', len(updates), flush=True)
        folder.mkdir(parents=True)
        result = feedback.replace(rows, updates)
        if updates:
            export.final(result, args.image, folder, slots, opt)
        else:
            for name in ('text.pptx', 'native.png', 'native.pdf', 'export.json'):
                if (args.baseline/name).is_file():
                    shutil.copy2(args.baseline/name, folder/name)
        save(folder/'scene.json', {**scene, 'texts':result})
        return Image.open(folder/'native.png').convert('RGB')
    raw = fit(raw, 'ocr_raw'); tight = fit(tight, 'ocr_tight')
    raw_image = render(raw, args.output/'raw_ocr_candidate')
    # Keep the unfiltered candidate for visual inspection, even if rolled back.
    tight_image = render(tight, args.output/'tight_candidate')
    collision_rejections = []
    for pair in feedback.new_collisions(rows, feedback.replace(rows, tight)):
        collision_rejections.extend(i for i in pair if i in tight)
    eligible = {i:r for i,r in tight.items() if i not in collision_rejections}
    rounds = []; state = {}
    def evaluate(updates):
        folder = args.output/'cpu_guard'/f'round_{len(rounds)+1:02d}'
        actual = render(updates, folder)
        metrics = {i:feedback.evidence(source, before, actual, [by_id[i]], r) for i,r in updates.items()}
        save(folder/'scores.json', metrics)
        rounds.append(folder); state.update(folder=folder, image=actual)
        return metrics
    accepted, history = retain_nonregressing(eligible, evaluate)
    save(args.output/'cpu_guard/history.json', history)
    shutil.copytree(state['folder'], args.output/'final')
    save(args.output/'final/provenance.json', {'image_sha256':sha(args.image), 'sources':hashes})
    # Common ROI for all four variants, so the ablation scores are comparable.
    scores = {}
    for ident in wanted:
        compare_rows = [*raw.get(ident, []), *tight.get(ident, [])]
        scores[ident] = {label:feedback.evidence(source, before, im, [by_id[ident]], compare_rows)
                        for label,im in [('raw_ocr',raw_image), ('tight',tight_image), ('final',state['image'])]}
    save(args.output/'scores.json', scores)
    roi = feedback.window([r for i in wanted for r in [by_id[i], *raw.get(i, []), *tight.get(i, [])]], source.size, 18)
    boxes = source.copy(); draw = ImageDraw.Draw(boxes)
    for ident, rec in records.items():
        draw.rectangle(rec['ocr_bbox'], outline='#00A060', width=1)
        if rec['bbox_2d']:
            draw.rectangle(rec['bbox_2d'], outline='#FF00AA', width=1)
    panels = [('Original',source), ('OCR green / tight pink',boxes), ('Before PPT',before),
              ('Raw OCR candidate',raw_image), ('Tight candidate',tight_image), ('Final after rollback',state['image'])]
    w = 450; h = round((roi[3]-roi[1])*w/(roi[2]-roi[0]))
    canvas = Image.new('RGB', (w*3, (h+34)*2), 'white'); d = ImageDraw.Draw(canvas)
    for index, (label, im) in enumerate(panels):
        x = index%3*w; y = index//3*(h+34)
        d.text((x+8,y+8), label, fill='black')
        canvas.paste(im.crop(roi).resize((w,h),Image.Resampling.LANCZOS), (x,y+34))
    canvas.save(args.output/'comparison.png')
    summary = {'state':'completed', 'api_calls':0, 'accepted_ids':sorted(accepted),
               'deferred_ids':sorted(set(wanted)-set(accepted)), 'collision_rejections':sorted(set(collision_rejections)),
               'errors':errors, 'rule':'after >= before on the identical union ROI; rerender after rollback',
               'metric_is_proxy_not_ground_truth':True, 'quality_status':'needs_visual_review'}
    save(args.output/'summary.json', summary)
    assert all(sha(Path(p)) == value for p,value in hashes.items()), 'Baseline/source modified'
    print(summary, flush=True)


if __name__ == '__main__':
    main()
