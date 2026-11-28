"""Read lettering again that neither the text branch nor a retained picture carries.

The text branch leaves out lettering it takes to be inside a screenshot or a
photograph, on the assumption that the picture branch keeps that area as a
picture. The two branches run side by side and neither sees what the other
kept, so lettering inside a drawn page whose photographs alone became
pictures is lost by both.

Ownership is settled here from pixels instead: text-like source ink that the
slide without text does not show and that no text row covers belongs to
nobody. Where such ink is dense it is gathered into a region, the region is
read once more, and every line read takes its box and colour from the ink.
"""
import logging
import math
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

PROTOCOL = 'unowned_text_regions.1'
# Fewest text-like marks that make a region worth one reading.
MINIMUM_MARKS = 5
# Most regions read for one slide.
REGION_LIMIT = 8
# A text line is wider than tall; a dashed vertical rule is not.
MINIMUM_WIDTH_IN_HEIGHTS = 3.
# How far, in mark heights, two marks may be apart along and across a line.
REACH_ALONG = 1.5
REACH_ACROSS = .8
# How far apart, in mark heights, two groups of marks may be and still be read in one view.
GROUP_REACH = 4.
# Height in pixels a mark is enlarged to for reading, and the bounds on that.
READING_HEIGHT = 24.
MAXIMUM_SCALE = 4.
MAXIMUM_SIDE = 1600
# Share of a line's letters that must be found as source marks under it.
INK_PER_LETTER = .25
MAXIMUM_LINE_LENGTH = 200
# A line of several letters whose ink is this much taller than wide runs up or down the page.
UPRIGHT_HEIGHT_TO_WIDTH = 1.5


def _height(mark: Dict[str, Any]) -> float:
    return float(mark['bbox_2d'][3]-mark['bbox_2d'][1]+1)


def _linked(first: Dict[str, Any], second: Dict[str, Any]) -> bool:
    a, b = first['bbox_2d'], second['bbox_2d']
    tall = max(_height(first), _height(second))
    if not .4 < _height(first)/_height(second) < 2.5:
        return False
    along = max(a[0]-b[2], b[0]-a[2], 0)
    across = max(a[1]-b[3], b[1]-a[3], 0)
    return along <= REACH_ALONG*tall and across <= REACH_ACROSS*tall


def _extent(members: Sequence[int], marks: Sequence[Dict[str, Any]]) -> Tuple[List[float], float]:
    boxes = np.asarray([marks[i]['bbox_2d'] for i in members], float)
    return ([boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()],
            float(np.median([_height(marks[i]) for i in members])))


def _joined(groups: List[List[int]], marks: Sequence[Dict[str, Any]]) -> List[List[int]]:
    """Join groups of marks that lie close enough to be pieces of one block of lettering.

    A connector crossing a paragraph, or a wide gap between words, splits its
    marks into several groups. Read one by one, each view cuts words in half.
    """
    groups = [list(g) for g in groups]
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                (a, tall_a), (b, tall_b) = _extent(groups[i], marks), _extent(groups[j], marks)
                reach = GROUP_REACH*max(tall_a, tall_b)
                if (.4 < tall_a/tall_b < 2.5 and max(a[0]-b[2], b[0]-a[2], 0) <= reach
                        and max(a[1]-b[3], b[1]-a[3], 0) <= reach):
                    groups[i] += groups.pop(j)
                    changed = True
                    break
            if changed:
                break
    return groups


def regions(marks: Sequence[Dict[str, Any]], size: Tuple[int, int], *, minimum: int = MINIMUM_MARKS,
            limit: int = REGION_LIMIT) -> List[Dict[str, Any]]:
    """Gather unread text-like marks into the regions dense enough to hold lettering.

    Args:
        marks: Source ink components no text row covers and the slide does not show; each has ``bbox_2d``.
        size: Width and height of the source image.
        minimum: Fewest marks that make a region.
        limit: Most regions returned.

    Returns:
        Regions with most marks first: ``bbox_2d`` padded by one mark height, ``marks``, ``mark_height``.
    """
    parent = list(range(len(marks)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i, first in enumerate(marks):
        for j in range(i+1, len(marks)):
            if _linked(first, marks[j]):
                parent[root(i)] = root(j)
    groups: Dict[int, List[int]] = {}
    for index in range(len(marks)):
        groups.setdefault(root(index), []).append(index)
    found = []
    for members in _joined([g for g in groups.values() if len(g) > 1], marks):
        if len(members) < minimum:
            continue
        boxes = np.asarray([marks[i]['bbox_2d'] for i in members], float)
        tall = float(np.median([_height(marks[i]) for i in members]))
        x0, y0, x1, y1 = boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()
        if x1-x0+1 < MINIMUM_WIDTH_IN_HEIGHTS*tall:
            continue
        found.append({'bbox_2d': [int(max(0, x0-tall)), int(max(0, y0-tall)),
                                  int(min(size[0]-1, x1+tall)), int(min(size[1]-1, y1+tall))],
                      'marks': len(members), 'mark_height': tall})
    found.sort(key=lambda region: (-region['marks'], region['bbox_2d']))
    return found[:limit]


def view(source: Image.Image, region: Dict[str, Any], rows: Iterable[Dict[str, Any]]) -> Tuple[Image.Image, Dict[str, Any]]:
    """Return the enlarged region with already recovered text blanked, and how it maps back.

    Args:
        source: The source image.
        region: One region from :func:`regions`.
        rows: Text rows already recovered; their boxes are painted over so they are not read twice.

    Returns:
        The picture to read and its frame: ``origin`` in source pixels and ``scale``.
    """
    x0, y0, x1, y1 = region['bbox_2d']
    crop = np.asarray(source.convert('RGB').crop((x0, y0, x1+1, y1+1))).copy()
    ground = np.median(crop.reshape(-1, 3), axis=0).astype('uint8')
    for row in rows:
        box = row.get('bbox_2d', [])
        if len(box) != 4:
            continue
        left, top = max(0, math.floor(box[0]-2-x0)), max(0, math.floor(box[1]-2-y0))
        right, bottom = min(crop.shape[1], math.ceil(box[2]+3-x0)), min(crop.shape[0], math.ceil(box[3]+3-y0))
        if right > left and bottom > top:
            crop[top:bottom, left:right] = ground
    scale = min(MAXIMUM_SCALE, max(1., READING_HEIGHT/max(1., region['mark_height'])),
                MAXIMUM_SIDE/max(crop.shape[0], crop.shape[1]))
    picture = Image.fromarray(crop)
    picture = picture.resize((max(1, round(picture.width*scale)), max(1, round(picture.height*scale))),
                             Image.Resampling.LANCZOS)
    return picture, {'origin': [x0, y0], 'scale': float(scale)}


def validate(value: Any, size: Tuple[int, int]) -> Dict[str, Any]:
    """Check one reading of a region.

    Args:
        value: The model's reply.
        size: Width and height of the picture that was read.

    Returns:
        The reply.

    Raises:
        ValueError: When the reply has another form or a line lies outside the picture.
    """
    if not isinstance(value, dict) or not isinstance(value.get('lines'), list):
        raise ValueError('Return {"lines": [...]}; an empty list when the picture holds no lettering')
    for line in value['lines']:
        text = line.get('text') if isinstance(line, dict) else None
        box = line.get('bbox_2d') if isinstance(line, dict) else None
        if (not isinstance(text, str) or not 1 <= len(text.strip()) <= MAXIMUM_LINE_LENGTH
                or any(unicodedata.category(c).startswith('C') for c in text)):
            raise ValueError('Each line needs its visible text on one line')
        if (not isinstance(box, list) or len(box) != 4 or not all(isinstance(v, (int, float)) for v in box)
                or not (-2 <= box[0] < box[2] <= size[0]+2 and -2 <= box[1] < box[3] <= size[1]+2)):
            raise ValueError('Each line needs bbox_2d [xmin,ymin,xmax,ymax] inside the picture')
        if type(line.get('bold')) is not bool or type(line.get('italic')) is not bool:
            raise ValueError('Each line needs bold and italic as true or false')
    return value


def line_rows(lines: Sequence[Dict[str, Any]], frame: Dict[str, Any], marks: Sequence[Dict[str, Any]], *,
              used: Set[str], font: str, claimed: Optional[Set[int]] = None) -> List[Dict[str, Any]]:
    """Turn the lines read in one region into text rows placed on the source ink.

    A line keeps only if unread source marks lie under it, and it takes its box
    and colour from those marks. Each mark supports one line. Lettering that
    runs up or down the page is not added.

    Args:
        lines: Validated lines in the coordinates of the enlarged picture.
        frame: ``origin`` and ``scale`` of that picture.
        marks: All unread source marks of the slide.
        used: Row identifiers already taken; the new ones are added to it.
        font: Font family given to the rows.
        claimed: Indices of marks already supporting a line; extended here.

    Returns:
        One row per supported line.
    """
    claimed = set() if claimed is None else claimed
    rows = []
    for line in lines:
        box = [line['bbox_2d'][i]/frame['scale']+frame['origin'][i % 2] for i in range(4)]
        members = [i for i, m in enumerate(marks) if i not in claimed
                   and box[0] <= (m['bbox_2d'][0]+m['bbox_2d'][2])/2 <= box[2]
                   and box[1] <= (m['bbox_2d'][1]+m['bbox_2d'][3])/2 <= box[3]]
        letters = sum(not c.isspace() for c in line['text'])
        if len(members) < max(1, math.ceil(INK_PER_LETTER*letters)):
            continue
        boxes = np.asarray([marks[i]['bbox_2d'] for i in members], float)
        wide, tall = boxes[:, 2].max()-boxes[:, 0].min()+1, boxes[:, 3].max()-boxes[:, 1].min()+1
        if letters > 1 and tall > UPRIGHT_HEIGHT_TO_WIDTH*wide:
            # Rows written here are horizontal; rotated lettering is left for the text branch.
            continue
        claimed.update(members)
        index = len(used)+1
        while 'REGION_TEXT_%04d' % index in used:
            index += 1
        ident = 'REGION_TEXT_%04d' % index
        used.add(ident)
        text = line['text'].strip()
        rows.append({'id': ident, 'text': text, 'kind': 'text',
                     'bbox_2d': [float(boxes[:, 0].min()), float(boxes[:, 1].min()),
                                 float(boxes[:, 2].max()), float(boxes[:, 3].max())],
                     'font': font, 'bold': line['bold'], 'italic': line['italic'],
                     'color': Counter(marks[i]['color'] for i in members).most_common(1)[0][0],
                     'rotation_deg': 0., 'layout_mode': 'single_line',
                     'runs': [{'text': text, 'script': 'normal'}],
                     'source_region': {'protocol': PROTOCOL, 'source_marks': len(members)}})
    return rows


def observe(source: Image.Image, marks: Sequence[Dict[str, Any]], rows: Sequence[Dict[str, Any]], folder: Path,
            runtime: Any, save: Callable[[Path, Any], None]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Read every dense region of unowned lettering of one slide.

    Args:
        source: The source image.
        marks: Source ink no text row covers and the slide without text does not show.
        rows: Text rows already recovered.
        folder: Where requests and the receipt are kept.
        runtime: Model access with ``call``.
        save: Writes one JSON receipt.

    Returns:
        The added rows and the receipt.
    """
    folder = Path(folder)
    marks = [m for m in marks if 'complete_label_reference' not in m]
    found = regions(marks, source.size)
    used = {r['id'] for r in rows}
    claimed: Set[int] = set()
    additions: List[Dict[str, Any]] = []
    receipts = []
    for number, region in enumerate(found, 1):
        local = folder/('region_%02d' % number)
        local.mkdir(parents=True, exist_ok=True)
        picture, frame = view(source, region, rows)
        picture.save(local/'view.png')
        receipt = {'region': region, 'frame': frame, 'lines': 0, 'added': []}
        try:
            value = runtime.call(local/'calls', 'region', 'Read lettering from the picture. Return only JSON.', {
                'protocol': PROTOCOL,
                'task': 'Transcribe every legible line of lettering in this enlarged part of a figure.',
                'coordinates': {'image_size_px': list(picture.size), 'bbox_2d': '[xmin,ymin,xmax,ymax] in pixels of this picture'},
                'rules': [
                    'This area was not kept as a picture on the slide, so its lettering is in scope even when it sits inside a drawn page, card, window or screenshot.',
                    'One entry per visible line. Copy the characters exactly; do not translate, complete or correct them.',
                    'Leave out lettering too small or blurred to read with confidence, lettering cut by the border, and anything that is not lettering: icons, rules, placeholder bars, arrows.',
                    'Blank patches cover text that is already recovered; do not guess what was there.',
                    'If nothing legible remains return lines=[].'],
                'output': {'lines': [{'text': 'visible line', 'bbox_2d': [0, 0, 10, 10], 'bold': False, 'italic': False}]}},
                [('Enlarged part of the source figure', local/'view.png')],
                lambda reply: validate(reply, picture.size), 4096)
        except (RuntimeError, ValueError, requests.RequestException) as error:
            # Optional additions never stop the export of the text already recovered.
            logger.warning('Unowned text region %d was not read: %s', number, error)
            receipt['error'] = {'type': type(error).__name__, 'message': str(error)[:500]}
            receipts.append(receipt)
            continue
        added = line_rows(value['lines'], frame, marks, used=used, font='Arial', claimed=claimed)
        receipt.update(lines=len(value['lines']), added=[r['id'] for r in added])
        receipts.append(receipt)
        additions.extend(added)
    audit = {'protocol': PROTOCOL, 'unread_marks': len(marks), 'regions': len(found), 'model_calls': len(found),
             'added_text': additions, 'receipts': receipts}
    save(folder/'regions.json', audit)
    return additions, audit
