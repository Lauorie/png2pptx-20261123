"""Validate meaningful text fields; allow harmless additional output metadata."""
import math
import re
from ..common import valid_box
from . import FONTS


def recognition(raw, size, *, known=(), local=False, default_font=FONTS[0]):
    if not isinstance(raw, dict) or not isinstance(raw.get('texts'), list):
        raise ValueError('texts must be an array')
    rows = []; replaced = set(); repairs=[]
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
        # Background is CPU-owned. Ignore even unsolicited/cached model fills.
        background={'background_fill':None,'background_evidence':''}
        color = row.get('color', '#000000')
        if font not in FONTS or not isinstance(color, str) or not re.fullmatch(r'#[0-9a-fA-F]{6}', color):
            raise ValueError(prefix + ': font must be allowed and color must be #RRGGBB')
        angle = row.get('rotation_deg', 0)
        if type(angle) not in (float, int) or not math.isfinite(angle):
            raise ValueError(prefix + '.rotation_deg must be a finite clockwise angle')
        angle=(angle+180)%360-180
        for key in ('bold', 'italic', 'underline'):
            if type(row.get(key, False)) is not bool:
                raise ValueError(prefix + '.' + key + ' must be boolean')
        legibility = row.get('legibility', 'clear')
        if legibility not in ('clear', 'partial', 'unreadable') or (legibility == 'clear' and not row['text'].strip()):
            raise ValueError(prefix + ': invalid legibility or empty clear text')
        runs = row.get('runs') or [{'text': row['text'], 'script': 'normal'}]
        if not isinstance(runs, list) or any(not isinstance(r, dict) or not isinstance(r.get('text'), str)
                or r.get('script') not in ('normal', 'sub', 'super') for r in runs):
            raise ValueError(prefix + '.runs must contain text and normal/sub/super script')
        run_text=''.join(r['text'] for r in runs)
        text=row['text']
        if run_text != text:
            from .display_equivalence import equivalent
            if run_text.replace(' ','')==text.replace(' ','') and text.replace(' ',''):
                from .format_normalization import reconcile_run_spaces
                repaired=reconcile_run_spaces(text,runs)
                repairs.append({'index':i,'field':'runs','rule':'ascii_spaces_only_text_authoritative',
                                'before':runs,'after':repaired})
                runs=repaired
            elif row.get('kind')=='formula':
                if equivalent(text,runs):text=run_text
                else:
                    from .formula_protocol import reconcile
                    reconciled=reconcile(text,runs,row.get('latex',''))
                    if reconciled is None:raise ValueError(prefix + '.runs text concatenation must equal text')
                    repairs.append({'index':i,'field':'text','rule':'latex_and_runs_agree_display_metadata_reconciled',
                                    'before':text,'after':reconciled,'latex_preserved':row.get('latex')})
                    text=reconciled
            else:raise ValueError(prefix + '.runs text concatenation must equal text')
        for run in runs:
            if 'color' in run and (not isinstance(run['color'], str) or not re.fullmatch(r'#[0-9a-fA-F]{6}', run['color'])):
                raise ValueError(prefix + '.runs.color must be #RRGGBB')
            for key in ('bold', 'italic', 'underline'):
                if key in run and type(run[key]) is not bool:
                    raise ValueError(prefix + '.runs.' + key + ' must be boolean')
        kind = row.get('kind', 'text')
        line_count=row.get('visual_line_count',1)
        if type(line_count) is not int or not 1<=line_count<=16:
            raise ValueError(prefix + '.visual_line_count must be an integer from 1 to 16')
        latex = row.get('latex', '')
        if kind not in ('text', 'formula') or not isinstance(latex, str) or len(latex) > 16000:
            raise ValueError(prefix + ': invalid kind/latex')
        if kind == 'formula':
            from .omml import latex_to_omml, normalize_latex_input
            if not latex.strip():
                raise ValueError(prefix + ': native formula requires complete LaTeX')
            # Style commands alone do not make a natural-language phrase math.
            structural=re.search(r'[_^=+<>∑∫√×÷]|\\(?:frac|sqrt|sum|int|hat|tilde|alpha|beta|theta|epsilon|varepsilon|mathbb|mathcal)\b',latex)
            words=re.findall(r'[A-Za-z]{3,}',row['text'])
            if not structural and len(words)>=2:
                raise ValueError(prefix + ': prose must use kind=text and colored runs, not formula LaTeX')
            normalized=normalize_latex_input(latex)
            if normalized!=latex:
                repairs.append({'index':i,'field':'latex','rule':'supported_equivalent_latex_syntax',
                                'before':latex,'after':normalized})
            latex=normalized
            latex_to_omml(latex, color=color)
        replaces = row.get('replaces', [])
        if not isinstance(replaces, list) or any(not isinstance(v, str) or v not in known for v in replaces):
            raise ValueError(prefix + '.replaces references an unknown text ID')
        if len(replaces) > 1 or (replaces and not local):
            raise ValueError(prefix + ': replace one existing line at a time')
        if replaced.intersection(replaces):
            raise ValueError(prefix + ': an existing line cannot be replaced twice in one response')
        replaced.update(replaces)
        rows.append(dict(text=text, bbox_2d=list(row['bbox_2d']), font=font, **background,
                         color=color.upper(), bold=row.get('bold', False), italic=row.get('italic', False), underline=row.get('underline',False),
                         rotation_deg=angle, legibility=legibility,
                         runs=[{'text':r['text'], 'script':r['script'],
                                **{k:r[k] for k in ('bold','italic','underline') if k in r},
                                **({'color':r['color'].upper()} if 'color' in r else {})} for r in runs],
                         kind=kind, latex=latex.strip(), replaces=replaces,
                         visual_line_count=line_count,
                         line_layout_status=('requires_split' if kind=='text' and line_count>1
                                             else 'native_formula' if kind=='formula' else 'single_baseline')))
    regions = raw.get('regions', [])
    if not isinstance(regions, list):
        raise ValueError('regions must be an array')
    checked_regions=[]
    for index,region in enumerate(regions):
        if isinstance(region,dict):
            b=region.get('bbox_2d')
            if (isinstance(b,list) and len(b)==4 and all(type(v) in (int,float) and math.isfinite(v) for v in b)
                    and all(-1<=v<=size[i%2] for i,v in enumerate(b))):
                bounded=[max(0,min(v,size[i%2]-1)) for i,v in enumerate(b)]
                if bounded!=b:
                    repairs.append({'index':index,'field':'regions.bbox_2d','rule':'one_pixel_boundary_convention',
                                    'before':b,'after':bounded})
                    region={**region,'bbox_2d':bounded}
        if not isinstance(region, dict) or not valid_box(region.get('bbox_2d'), size) or not isinstance(region.get('reason'), str):
            raise ValueError('Each region needs a valid bbox_2d and reason')
        checked_regions.append(region)
    regions=checked_regions
    if local and regions:
        raise ValueError('Local review has no further region pass; return regions=[]')
    uncertain = raw.get('uncertain', [])
    if not isinstance(uncertain, list) or any(not isinstance(v, str) for v in uncertain):
        raise ValueError('uncertain must be a list of strings')
    return {'texts': rows, 'regions': regions, 'uncertain': uncertain,
            **({'format_repairs':repairs} if repairs else {})}
