"""Audit identical SAM inputs and create a seeded mask comparison sample."""
import argparse
import json
import os
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from clients import read, save, require_cluster


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    run = p.parse_args().run_dir.resolve()
    require_cluster()
    root = run / 'delivery/sam_comparison'
    root.mkdir(parents=True, exist_ok=True)
    cases = read(run / 'inputs.json')['cases']
    checked = 0
    candidates = []
    for case in cases:
        old = read(run / 'sam_lines/side96_both_part_owned' / case['id'] / 'line_evidence.json')
        new = read(run / 'sam_lines/side96_both_new' / case['id'] / 'line_evidence.json')
        if set(old['parts']) != set(new['parts']) or old['config'] != new['config']:
            raise ValueError('SAM geometry/config differs: ' + case['id'])
        seen = set()
        for key, before in old['parts'].items():
            after = {tuple(t['box']): t for t in new['parts'][key]['tiles']}
            for tile in before['tiles']:
                other = after[tuple(tile['box'])]
                old_dir, new_dir = Path(tile['dir']), Path(other['dir'])
                if old_dir not in seen:
                    if read(old_dir / 'sam/request.json') != read(new_dir / 'sam/request.json'):
                        raise ValueError('SAM image or prompt differs: ' + str(new_dir))
                    if read(new_dir / 'sam/predictions.json')['best_checkpoint']['epoch'] != 7:
                        raise ValueError('Unexpected new SAM checkpoint: ' + str(new_dir))
                    checked += 1
                    seen.add(old_dir)
                if tile['target'] or other['target']:
                    candidates.append((case['id'], key, tile, other))
    chosen = random.Random(20261016).sample(candidates, min(24, len(candidates)))
    font = ImageFont.truetype(os.environ['STYLE_FONT'], 16)
    rows = []
    for index, (ident, key, a, b) in enumerate(chosen):
        with Image.open(Path(a['dir']) / 'tile.png') as im:
            source = im.convert('RGB')
        overlays = []
        masks = []
        for tile in (a, b):
            mask = np.zeros((source.height, source.width), bool)
            if tile['target']:
                with Image.open(tile['target_mask']) as im:
                    mask = np.asarray(im.convert('L')) > 127
            masks.append(mask)
            rgb = np.asarray(source, float)
            rgb[mask] = rgb[mask] * .55 + np.array([224, 0, 160]) * .45
            overlays.append(Image.fromarray(np.uint8(rgb)))
        canvas = Image.new('RGB', (1024, 380), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), ident + ' ' + key + ' box=' + str(a['box']), font=font, fill='black')
        for n, (label, im) in enumerate(zip(('SOURCE', 'OLD SAM selected mask', 'NEW SAM selected mask'), (source, *overlays))):
            canvas.paste(im.resize((336, 336), Image.Resampling.LANCZOS), (n * 344, 42))
            draw.text((n * 344 + 4, 23), label, font=font, fill='black')
        path = root / f'{index:02d}_{ident}.png'
        canvas.save(path)
        union = (masks[0] | masks[1]).sum()
        rows.append({'case': ident, 'part': key, 'box': a['box'], 'panel': path.name,
                     'mask_iou': float((masks[0] & masks[1]).sum() / union) if union else None})
    save(root / 'audit.json', {'identical_input_requests': checked, 'cases': len(cases),
                              'old_epoch': 18, 'new_epoch': 7, 'seeded_samples': rows,
                              'note': 'Mask IoU measures agreement between models, not correctness.'})
    (root / 'INDEX.md').write_text('# 新旧 SAM 部件 mask 随机样本\n\n输入切块与提示逐一核对相同。粉色为所选目标 mask；模型间 IoU 只代表一致性。\n\n' +
                                   '\n\n'.join(f'![{r["case"]} {r["part"]}]({r["panel"]})' for r in rows) + '\n')
    print('SAM_INPUTS_VERIFIED', checked, 'visual samples', len(rows), flush=True)


if __name__ == '__main__':
    main()
