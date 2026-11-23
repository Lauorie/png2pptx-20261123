"""Read-only local OCR experiment with rotation-aware source-coordinate boxes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import unicodedata

OCR_ENGINES = {}


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def normalized(text):
    return ''.join(unicodedata.normalize('NFKC', text).split())


def unrotate(point, angle, width, height):
    x, y = point
    return {0: (x, y), 90: (width - 1 - y, x),
            180: (width - 1 - x, height - 1 - y),
            270: (y, height - 1 - x)}[angle]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--crop-dir', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--ids', nargs='+', required=True)
    p.add_argument('--llm-proposals', type=Path, help='Optional historical LLM boxes for visualization only; no API is required')
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rotations', type=int, nargs='+', default=[0, 90, 270])
    p.add_argument('--ocr-max-side', type=int, default=768)
    args = p.parse_args(argv)
    if any(a not in (0, 90, 180, 270) for a in args.rotations):
        raise ValueError('Use right-angle rotations only')
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
    args.output.mkdir(parents=True, exist_ok=False)
    from PIL import Image, ImageDraw, ImageFont
    from paddleocr import PaddleOCR
    source = Image.open(args.crop_dir / 'clean.png').convert('RGB')
    frame = json.loads((args.crop_dir / 'frame.json').read_text())
    rows = {r['id']: r for r in json.loads(args.scene.read_text())['texts']}
    targets = [rows[i] for i in args.ids]
    proposals = json.loads(args.llm_proposals.read_text()) if args.llm_proposals else {r['id']:[r] for r in targets}
    comparison_label = 'LLM' if args.llm_proposals else 'Before'
    def local_box(box):
        return [(v - frame['box'][i % 2] + .5) * frame['scale'] - .5 + frame['offset'][i % 2] for i, v in enumerate(box)]
    def source_box(box):
        return [(v + .5 - frame['offset'][i % 2]) / frame['scale'] + frame['box'][i % 2] - .5 for i, v in enumerate(box)]
    start = time.monotonic()
    cache_key = (str(args.models.resolve()), args.ocr_max_side)
    ocr = OCR_ENGINES.get(cache_key)
    if ocr is None:
        print('Loading local PP-OCRv5 server models; CPU threads=2', flush=True)
        ocr = PaddleOCR(ocr_version='PP-OCRv5',
        text_detection_model_name='PP-OCRv5_server_det',
        text_detection_model_dir=str(args.models / 'PP-OCRv5_server_det'),
        text_recognition_model_name='PP-OCRv5_server_rec',
        text_recognition_model_dir=str(args.models / 'PP-OCRv5_server_rec'),
        use_doc_orientation_classify=False, use_doc_unwarping=False,
        use_textline_orientation=False, text_det_limit_side_len=args.ocr_max_side,
        text_det_limit_type='max', device='cpu', cpu_threads=2,
            enable_hpi=False, enable_mkldnn=False)
        OCR_ENGINES[cache_key] = ocr
    print('Models ready', flush=True)
    detections = []
    for angle in args.rotations:
        image = source if angle == 0 else source.transpose({90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_270}[angle])
        rotated_size = image.size
        if max(image.size) > args.ocr_max_side:
            scale = args.ocr_max_side / max(image.size)
            image = image.resize(tuple(max(1,round(v*scale)) for v in image.size), Image.Resampling.LANCZOS)
        path = args.output / f'input_{angle:03d}.png'
        image.save(path)
        begin = time.monotonic()
        results = list(ocr.predict(str(path)))
        if len(results) != 1: raise ValueError('Expected a single image')
        raw = results[0].json
        if isinstance(raw, str): raw = json.loads(raw)
        save(args.output / f'raw_{angle:03d}.json', raw)
        data = raw.get('res', raw)
        for text, confidence, poly in zip(data['rec_texts'], data['rec_scores'], data['rec_polys']):
            original_poly = [((q[0]+.5)*rotated_size[0]/image.width-.5,
                              (q[1]+.5)*rotated_size[1]/image.height-.5) for q in poly]
            points = [unrotate(q, angle, source.width, source.height) for q in original_poly]
            b = [min(q[0] for q in points), min(q[1] for q in points), max(q[0] for q in points), max(q[1] for q in points)]
            detections.append({'index': len(detections), 'text': text, 'confidence': float(confidence),
                'rotation': angle, 'crop_bbox': b, 'source_bbox': source_box(b), 'crop_polygon': points})
        print('OCR', angle, 'boxes', len(data['rec_texts']), 'seconds', round(time.monotonic()-begin, 2), flush=True)
        save(args.output / 'progress.json', {'completed_rotations': [a for a in args.rotations if (args.output / f'raw_{a:03d}.json').exists()]})
    matches = {}; used = set()
    for row in targets:
        b = row['bbox_2d']; cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
        diag = max(10., math.hypot(b[2]-b[0], b[3]-b[1]))
        candidates = []
        for d in detections:
            if normalized(d['text']) != normalized(row['text']) or d['confidence'] < .5 or d['index'] in used: continue
            q = d['source_bbox']; distance = math.hypot((q[0]+q[2])/2-cx, (q[1]+q[3])/2-cy)
            if distance > max(32., diag): continue
            candidates.append({**d, 'rank_cost': distance/diag + .25*(1-d['confidence'])})
        candidates.sort(key=lambda d: d['rank_cost'])
        chosen = candidates[0] if candidates else None
        if chosen: used.add(chosen['index'])
        matches[row['id']] = {'text': row['text'], 'chosen': chosen, 'candidates': candidates}
    report = {'model': 'PP-OCRv5_server_det+PP-OCRv5_server_rec', 'device': 'cpu', 'threads': 2,
              'input_sha256':{name:hashlib.sha256((args.crop_dir/name).read_bytes()).hexdigest()
                              for name in ('clean.png','frame.json')},
              'frame': frame, 'ocr_max_side':args.ocr_max_side,'detections': detections, 'matches': matches,
              'elapsed_seconds': time.monotonic()-start,
              'note': 'OCR boxes are detector bounds, not guaranteed tight ink or font evidence. Exact normalized content plus spatial association; no source modification.'}
    save(args.output / 'report.json', report)
    boxes = [local_box(r['bbox_2d']) for r in targets]
    boxes += [m['chosen']['crop_bbox'] for m in matches.values() if m['chosen']]
    pad = 24 * frame['scale']
    roi = [max(0, int(min(b[0] for b in boxes)-pad)), max(0, int(min(b[1] for b in boxes)-pad)),
           min(source.width, math.ceil(max(b[2] for b in boxes)+pad)), min(source.height, math.ceil(max(b[3] for b in boxes)+pad))]
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
    panels = [source.copy() for _ in range(3)]
    for row in targets:
        ident = row['id']
        for canvas, box, color, label in [
            (panels[1], local_box(proposals[ident][0]['bbox_2d']), '#C00090', ident+' '+comparison_label),
            *([(panels[2], matches[ident]['chosen']['crop_bbox'], '#008650', ident+' OCR')] if matches[ident]['chosen'] else [])]:
            draw = ImageDraw.Draw(canvas); draw.rectangle(box, outline=color, width=4)
            xy = (box[0], max(0, box[1]-28)); draw.rectangle(draw.textbbox(xy,label,font=font),fill='white'); draw.text(xy,label,font=font,fill=color)
    w = 620; h = round((roi[3]-roi[1])*w/(roi[2]-roi[0]))
    board = Image.new('RGB',(w*3,h+42),'white'); draw = ImageDraw.Draw(board)
    for i,(panel,label) in enumerate(zip(panels,['Original',comparison_label+' bbox','OCR matched bbox'])):
        board.paste(panel.crop(roi).resize((w,h),Image.Resampling.LANCZOS),(i*w,42))
        draw.text((i*w+8,8),label,font=font,fill='black')
    board.save(args.output/'comparison.png')
    print('DONE', json.dumps({i: {'text':m['text'],'matched':bool(m['chosen']), 'box':m['chosen']['source_bbox'] if m['chosen'] else None} for i,m in matches.items()},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
