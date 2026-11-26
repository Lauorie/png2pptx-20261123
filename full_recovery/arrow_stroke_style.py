"""Give arrow rows the stroke style measured from the source figure.

The arrow branch hands over geometry with a placeholder style. Colour, width
and dash rhythm are measured here from the pixels under each path; a row whose
pixels cannot be read keeps the style it came with.
"""
from typing import Any, Dict, List

import numpy as np

from .stroke_measure import colour, measure

Row = Dict[str, Any]
PEER_COLOUR = 48.


def _path(row: Row) -> np.ndarray:
    """Return the drawn route of ``row`` as (x, y) points."""
    return np.asarray([row['start'], *row.get('via', []), row['end']], dtype=float)


def restyle(pixels: np.ndarray, row: Row) -> Row:
    """Return ``row`` with its stroke style measured from ``pixels``.

    Args:
        pixels: The source figure, height by width by 3.
        row: One arrow row with ``kind``, ``start``, ``via``, ``end`` and ``style``.

    Returns:
        A new row; the input is not modified. Heads are kept. Block arrows and
        rows that cannot be measured are returned unchanged.
    """
    if row.get('kind') == 'block_arrow':
        return row
    stroke = measure(pixels, _path(row))
    if stroke is None:
        return row
    dash: Dict[str, Any] = {'type': 'solid', 'preset': 'solid'}
    if stroke.dash:
        mark, gap = (round(value, 2) for value in stroke.dash)
        dash = {'type': 'dashed', 'preset': 'dash', 'pattern_px': [mark, gap], 'mark_px': mark, 'gap_px': gap}
    style = {**row.get('style', {}), 'color': stroke.colour, 'width_px': round(stroke.width_px, 2),
             'dash': dash, 'state': 'pixel_measured'}
    return {**row, 'style': style}


def _rgb(value: str) -> np.ndarray:
    """Return #RRGGBB as three numbers."""
    return np.asarray([int(value[index:index + 2], 16) for index in (1, 3, 5)], dtype=float)


def restyle_all(pixels: np.ndarray, rows: List[Row]) -> List[Row]:
    """Return every row restyled from ``pixels``.

    A figure draws its connectors in a few line styles. A row too short to
    measure fully takes its colour from its own pixels and its width from the
    measured row of the closest colour (within ``PEER_COLOUR``), or else the
    median measured width. With nothing measured in the figure, or no ink
    under the row, the row is returned unchanged.
    """
    first = [restyle(pixels, row) for row in rows]
    peers = [row['style'] for row in first if row.get('style', {}).get('state') == 'pixel_measured']
    if not peers:
        return first
    usual = float(np.median([peer['width_px'] for peer in peers]))
    result: List[Row] = []
    for row in first:
        measured = row.get('style', {}).get('state') == 'pixel_measured'
        found = None if measured or row.get('kind') == 'block_arrow' else colour(pixels, _path(row))
        if found is None:
            result.append(row)
            continue
        distance = [float(np.abs(_rgb(peer['color']) - _rgb(found)).max()) for peer in peers]
        nearest = int(np.argmin(distance))
        width = peers[nearest]['width_px'] if distance[nearest] <= PEER_COLOUR else round(usual, 2)
        result.append({**row, 'style': {**row['style'], 'color': found, 'width_px': width,
                                        'dash': {'type': 'solid', 'preset': 'solid'},
                                        'state': 'pixel_colour_peer_width'}})
    return result
