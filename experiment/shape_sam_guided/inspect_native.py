"""Assemble readable saved native comparisons for completed cases, no inference."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
from core import read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = read(args.run_dir / 'manifest.json')['config']
    base = Path(config['baseline_root'])
    args.output.mkdir(parents=True, exist_ok=True)
    for item in read(base / 'inputs/manifest.json')['images']:
        cid = item['id']
        target = args.output / (cid + '.png')
        paths = [base / item['original'], *[args.run_dir / 'native_review' / cid / name / 'native.png'
                                          for name in ['historical', *config['arms']]]]
        if target.exists() or not all(p.exists() for p in paths):
            continue
        w, h = item['source_size']; factor = min(1000/w, 850/h)
        size = round(w*factor), round(h*factor)
        board = Image.new('RGB', (size[0]*3, (size[1]+30)*2), 'white')
        draw = ImageDraw.Draw(board)
        for j, (label, path) in enumerate(zip(['Source', 'Historical', *config['arms']], paths)):
            x, y = j%3*size[0], j//3*(size[1]+30)
            with Image.open(path) as im:
                board.paste(im.convert('RGB').resize(size, Image.Resampling.LANCZOS), (x, y+30))
            draw.text((x+5, y+7), cid+' '+label, fill='black')
        board.save(target)
        print(target, flush=True)


if __name__ == '__main__':
    main()
