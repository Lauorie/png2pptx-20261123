"""Pilot comparison of SAM tile configurations for contour line ownership.

For each configuration and part: how much of the measured stroke the target
line masks cover, how many samples SAM vetoes, and whether the pixel stroke
measurement changes when the SAM support is applied.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / 'src_v2'), str(HERE.parents[1] / 'vendor')]
from clients import read, save  # noqa: E402
from shape_integration.styling import lines, sam_lines  # noqa: E402
from shape_integration.styling.geometry import parts as scene_parts  # noqa: E402


def support_for(row, size):
    tiles = []
    for tile in row['tiles']:
        union = None
        if tile['target']:
            with Image.open(tile.get('target_mask', Path(tile['dir']) / 'target_union.png')) as im:
                union = np.asarray(im.convert('L')) > 127
        tiles.append((tile['box'], union))
    return sam_lines.Support(tiles, size)


def changed(a, b):
    if bool(a.get('present')) != bool(b.get('present')):
        return 'presence'
    if not a.get('present'):
        return None
    if a.get('dash') != b.get('dash'):
        return 'dash'
    if abs(np.log(max(a['width_px'], 1e-3) / max(b['width_px'], 1e-3))) > np.log(1.1):
        return 'width'
    if np.linalg.norm(lines.rgb(a['color']) - lines.rgb(b['color'])) > 20:
        return 'color'
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--configs', nargs='+', required=True)
    args = parser.parse_args()
    from clients import require_cluster
    require_cluster()
    run = args.run_dir.resolve(); cases = {c['id']: c for c in read(run / 'inputs.json')['cases']}
    report = {}
    for name in args.configs:
        root = run / 'sam_lines' / name
        stats = {'parts': 0, 'parts_with_target': 0, 'tiles': 0, 'tiles_with_target': 0, 'masks': 0, 'target_masks': 0,
                 'present_samples_covered': [], 'vetoed_samples': 0, 'changes': {}, 'examples': []}
        for path in sorted(root.glob('*/line_evidence.json')):
            data = read(path); case = cases[data['case']]
            image = np.asarray(Image.open(case['original']).convert('RGB'), float)
            table = scene_parts(read(case['scene']))
            for key, row in data['parts'].items():
                stats['parts'] += 1; stats['tiles'] += len(row['tiles'])
                stats['tiles_with_target'] += sum(t['target'] for t in row['tiles'])
                stats['masks'] += sum(len(t['masks']) for t in row['tiles'])
                stats['target_masks'] += sum(m['target'] for t in row['tiles'] for m in t['masks'])
                support = support_for(row, data['size'])
                if not support:
                    continue
                stats['parts_with_target'] += 1
                plain = lines.measure(image, table[key], guess_width=row['guess'].get('width_px') or 2.)
                with_sam = lines.measure(image, table[key], support=support, guess_width=row['guess'].get('width_px') or 2.)
                stats['vetoed_samples'] += with_sam.get('vetoed_samples', 0)
                if plain.get('present'):
                    rows = lines.arc_samples(table[key], 3.)
                    pts = np.concatenate([r['points'] for r in rows])
                    known, inside = support(pts)
                    if known.any():
                        stats['present_samples_covered'].append(float(inside[known].mean()))
                kind = changed(plain, with_sam)
                if kind:
                    stats['changes'][kind] = stats['changes'].get(kind, 0) + 1
                    if len(stats['examples']) < 12:
                        stats['examples'].append({'case': data['case'], 'part': table[key]['object_id'] + '/' + table[key]['id'],
                                                  'change': kind, 'plain': {k: plain.get(k) for k in ('present', 'dash', 'width_px', 'color')},
                                                  'sam': {k: with_sam.get(k) for k in ('present', 'dash', 'width_px', 'color', 'vetoed_samples')}})
        covered = stats.pop('present_samples_covered')
        stats['median_path_coverage_where_known'] = float(np.median(covered)) if covered else None
        stats['sam_requests'] = sum(read(p)['sam_requests'] for p in root.glob('*/line_evidence.json'))
        stats['sam_wall_seconds'] = sum(read(p)['sam_wall_seconds'] for p in root.glob('*/line_evidence.json'))
        report[name] = stats
    save(run / 'sam_lines' / 'pilot_report.json', report)
    print(json.dumps(report, indent=1, ensure_ascii=False)[:6000])


if __name__ == '__main__':
    main()
