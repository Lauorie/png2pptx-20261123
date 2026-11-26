"""Rewrite one figure's arrow products with stroke styles measured from the source.

Run as ``python -m full_recovery.arrow_restyle_stage`` so that the arrow
branch's own writer (in ``--geometry-code``) produces the PPTX exactly as it
does for the branch's output; only the style of each row differs.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

from .arrow_stroke_style import restyle_all

logger = logging.getLogger(__name__)


def run(geometry_code: Path, scene_path: Path, source: Path, output: Path) -> Dict[str, int]:
    """Write ``output/scene.json`` and ``output/arrows.pptx``; return counts by style state."""
    sys.path[:0] = [str(geometry_code), str(geometry_code / 'lib')]
    from lib import native  # the arrow branch's writer
    scene = json.loads(scene_path.read_text())
    with Image.open(source) as opened:
        pixels = np.asarray(opened.convert('RGB'))
    if [pixels.shape[1], pixels.shape[0]] != list(scene['size']):
        raise ValueError('Arrow scene and source figure differ in size')
    rows = restyle_all(pixels, scene['arrows'])
    output.mkdir(parents=True, exist_ok=True)
    native.write(tuple(scene['size']), rows, [row['style'] for row in rows], output / 'arrows.pptx')
    (output / 'scene.json').write_text(json.dumps({**scene, 'arrows': rows}, ensure_ascii=False, indent=2) + '\n')
    counts: Dict[str, int] = {}
    for row in rows:
        state = str(row.get('style', {}).get('state'))
        counts[state] = counts.get(state, 0) + 1
    (output / 'receipt.json').write_text(json.dumps({'protocol': 'arrow_stroke_from_pixels.1', 'rows': len(rows),
                                                     'states': counts, 'model_calls': 0}, indent=2) + '\n')
    return counts


def main() -> int:
    """Command-line entry."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    for name in ('--geometry-code', '--scene', '--source', '--output'):
        parser.add_argument(name, type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger.info('%s', run(args.geometry_code, args.scene, args.source, args.output))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
