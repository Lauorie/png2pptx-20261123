"""Render completed immutable PPTs serially while the GPU benchmark continues."""
import argparse
import fcntl
from pathlib import Path
import time

from PIL import Image, ImageDraw
from core import read, save, sha
from shape_integration.styling.export import render_native


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--historical-native-root', type=Path)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    root, config = args.run_dir, read(args.config)
    base = Path(config['baseline_root'])
    items = read(base / 'inputs/manifest.json')['images']
    arms = config['arms']
    native = root / 'native_review'
    native.mkdir(parents=True, exist_ok=True)
    lock = (native / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        count = 0
        for item in items:
            cid = item['id']
            for method in ['historical', *arms]:
                if method == 'historical':
                    ppt = base / 'runs' / config['baseline_run'] / 'cases' / cid / 'fitting/geometry_final.pptx'
                else:
                    folder = root / method / 'cases' / cid
                    if not (folder / 'forward_complete.json').exists():
                        continue
                    ppt = folder / 'fitting/geometry_final.pptx'
                target = native / cid / method
                marker = target / 'verified.json'
                digest = sha(ppt)
                if marker.exists():
                    if read(marker)['ppt_sha256'] != digest:
                        raise ValueError('Rendered PPT changed: ' + str(ppt))
                elif method == 'historical' and args.historical_native_root:
                    old = args.historical_native_root / cid
                    if read(old / 'completed.json')['ppt_sha256']['historical'] != digest:
                        raise ValueError('Historical native render identity changed')
                    target.mkdir(parents=True, exist_ok=True)
                    for name in ['native.png', 'native.pdf']:
                        path = target / name
                        if not path.exists():
                            path.symlink_to((old / 'historical' / name).resolve())
                    save(marker, {'ppt_sha256': digest, 'reused_verified_native': str(old / 'historical')})
                else:
                    render_native(ppt, target / 'render', item['source_size'])
                    save(marker, {'ppt_sha256': digest, 'completed_at': time.time()})
                    print('NATIVE', cid, method, flush=True)
                count += 1
        expected = len(items) * (len(arms) + 1)
        save(native / 'progress.json', {'rendered': count, 'expected': expected, 'updated_at': time.time()})
        if count == expected:
            break
        if not args.wait:
            return
        time.sleep(15)
    pages = []
    for item in items:
        cid = item['id']
        paths = [base / item['original'], *[native / cid / method / 'native.png' for method in ['historical', *arms]]]
        with Image.open(paths[0]) as im:
            ratio = min(900 / im.width, 800 / im.height)
            size = (round(im.width * ratio), round(im.height * ratio))
        columns = 3
        rows = (len(paths) + columns - 1) // columns
        board = Image.new('RGB', (size[0] * columns, (size[1] + 28) * rows), 'white')
        draw = ImageDraw.Draw(board)
        for i, (label, path) in enumerate(zip(['Source', 'Historical', *arms], paths)):
            x, y = i % columns * size[0], i // columns * (size[1] + 28)
            with Image.open(path) as im:
                board.paste(im.convert('RGB').resize(size, Image.Resampling.LANCZOS), (x, y + 28))
            draw.text((x + 5, y + 7), cid + ': ' + label, fill='black')
        board.save(native / cid / 'comparison.png')
        pages.append(board)
    pages[0].save(native / 'overview.pdf', save_all=True, append_images=pages[1:], resolution=140.)
    save(native / 'completion.json', {'cases': len(items), 'renders': expected, 'completed_at': time.time()})


if __name__ == '__main__':
    main()
