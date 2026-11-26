"""Swap a figure's arrow products for copies styled from the source pixels.

Called at the composition boundary. The arrow branch's own files are left
untouched; the restyled copies live in the figure's composition folder. Any
failure keeps the original products, so this step can never lose a figure.
"""
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

logger = logging.getLogger(__name__)


def restyled(paths: Mapping[str, Path], source: Path, folder: Path, settings: Mapping[str, Any],
             code: Path) -> Dict[str, Path]:
    """Return replacement ``arrow_scene`` and ``arrow_pptx`` paths, or ``{}``.

    Args:
        paths: The figure's branch products; ``arrow_scene`` is read.
        source: The source figure.
        folder: Where the restyled copies are written.
        settings: Run settings; ``fresh_arrow_root`` locates the arrow writer.
        code: Directory that contains the ``full_recovery`` package.

    Returns:
        New paths for the two arrow products, or an empty mapping when the
        arrow writer is unavailable or the restyle step failed.
    """
    if settings.get('arrow_stroke_from_pixels', True) is False or not settings.get('fresh_arrow_root'):
        return {}
    geometry_code = Path(settings['fresh_arrow_root']) / 'geometry_code'
    if not (geometry_code / 'lib/native.py').is_file():
        return {}
    done = subprocess.run([sys.executable, '-B', '-m', 'full_recovery.arrow_restyle_stage',
                           '--geometry-code', str(geometry_code), '--scene', str(paths['arrow_scene']),
                           '--source', str(source), '--output', str(folder)],
                          cwd=code, capture_output=True, text=True)
    products = {'arrow_scene': folder / 'scene.json', 'arrow_pptx': folder / 'arrows.pptx'}
    if done.returncode or not all(path.is_file() and path.stat().st_size for path in products.values()):
        logger.warning('Arrow restyle failed; keeping branch styles: %s', done.stderr[-500:])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'failed.log').write_text(done.stderr[-4000:])
        return {}
    return products
