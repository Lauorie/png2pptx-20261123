"""Validate meaningful text fields; allow harmless additional output metadata."""
import math
import re
from ..common import valid_box
from . import FONTS


def recognition(raw, size, *, known=(), local=False, default_font=FONTS[0]):
    if not isinstance(raw, dict) or not isinstance(raw.get('texts'), list):
        raise ValueError('texts must be an array')
    rows = []; replaced = set()
    for i, row in enumerate(raw['texts']):
        prefix = f'texts.{i}'
        if not isinstance(row, dict) or not isinstance(row.get('text'), str):
            raise ValueError(prefix + '.text must be a string')
        if not valid_box(row.get('bbox_2d'), size):
            raise ValueError(prefix + '.bbox_2d must be finite, nondegenerate and inside this image')
        if any(c in row['text'] for c in '\n\r\v'):
            raise ValueError(prefix + ': split visual lines into separate items')
        if any(ord(c) < 32 and c != '\t' for c in row['text']):
            raise ValueError(prefix + ': XML control character in text')
        font = row.get('font', default_font)
        color = row.get('color', '#000000')
        if font not in FONTS or not isinstance(color, str) or not re.fullmatch(r'#[0-9a-fA-F]{6}', color):
            raise ValueError(prefix + ': font must be allowed and color must be #RRGGBB')
        angle = row.get('rotation_deg', 0)
        if type(angle) not in (float, int) or not math.isfinite(angle):
            raise ValueError(prefix + '.rotation_deg must be a finite clockwise angle')
        angle=(angle+180)%360-180
        for key in ('bold', 'italic'):
            if type(row.get(key, False)) is not bool:
                raise ValueError(prefix + '.' + key + ' must be boolean')
        legibility = row.get('legibility', 'clear')
        if legibility not in ('clear', 'partial', 'unreadable') or (legibility == 'clear' and not row['text'].strip()):
            raise ValueError(prefix + ': invalid legibility or empty clear text')
        runs = row.get('runs') or [{'text': row['text'], 'script': 'normal'}]
        if not isinstance(runs, list) or any(not isinstance(r, dict) or not isinstance(r.get('text'), str)
                or r.get('script') not in ('normal', 'sub', 'super') for r in runs):
            raise ValueError(prefix + '.runs must contain text and normal/sub/super script')
        if ''.join(r['text'] for r in runs) != row['text']:
            raise ValueError(prefix + '.runs text concatenation must equal text')
        replaces = row.get('replaces', [])
        if not isinstance(replaces, list) or any(not isinstance(v, str) or v not in known for v in replaces):
            raise ValueError(prefix + '.replaces references an unknown text ID')
        if len(replaces) > 1 or (replaces and not local):
            raise ValueError(prefix + ': replace one existing line at a time')
        if replaced.intersection(replaces):
            raise ValueError(prefix + ': an existing line cannot be replaced twice in one response')
        replaced.update(replaces)
        rows.append(dict(text=row['text'], bbox_2d=list(row['bbox_2d']), font=font,
                         color=color.upper(), bold=row.get('bold', False), italic=row.get('italic', False),
                         rotation_deg=angle, legibility=legibility,
                         runs=[{k:r[k] for k in ('text', 'script')} for r in runs], replaces=replaces))
    regions = raw.get('regions', [])
    if not isinstance(regions, list):
        raise ValueError('regions must be an array')
    for region in regions:
        if not isinstance(region, dict) or not valid_box(region.get('bbox_2d'), size) or not isinstance(region.get('reason'), str):
            raise ValueError('Each region needs a valid bbox_2d and reason')
    if local and regions:
        raise ValueError('Local review has no further region pass; return regions=[]')
    uncertain = raw.get('uncertain', [])
    if not isinstance(uncertain, list) or any(not isinstance(v, str) for v in uncertain):
        raise ValueError('uncertain must be a list of strings')
    return {'texts': rows, 'regions': regions, 'uncertain': uncertain}
