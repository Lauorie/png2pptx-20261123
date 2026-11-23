"""Separate visual text inventory from mathematical syntax generation.

Stage one establishes occurrence identity, geometry and initial text/style hints.
Stage two reads enlarged clean pixels and emits LaTeX for fixed formula IDs;
its valid content and colors supersede earlier hints. It cannot move, drop,
duplicate or restyle ordinary text.
"""
from copy import deepcopy
import json
import re

from PIL import Image

from ..common import read, save, sha
from . import contracts, model
from .prompt_protocol import RECOVERY_SCOPE, coordinates
from .controller import frame, merge, local_hints
from .local_geometry import compact_groups, union
from .omml import latex_to_omml, normalize_latex_input
from .numeric_routing import inventory as route_inventory


def inventory_prompt(size, opt, *, local=False, known=()):
    rules = [r for r in RECOVERY_SCOPE if not r.startswith((
        'CONTENT:', 'FORMAT CHECK:', 'FORMULA DISPLAY METADATA:', 'FORMULA COLOR SYNTAX:'))]
    rules = [r.replace('and formula LaTeX textcolor on the corresponding subexpressions',
                       'on the corresponding character spans') for r in rules]
    rules += [
        'STAGE 1 ONLY: identify and transcribe text occurrences and their visual style. '
        'Do NOT generate LaTeX. Set kind=formula for mathematical variables, scripts, '
        'operators, fractions, radicals, matrices and equations requiring Office Math; '
        'set kind=text for ordinary labels/prose even if they are italic or colored.',
        'For formulas, text/runs are a readable linear inventory, NOT a formula parser. '
        'Include visible numerator/denominator symbols, scripts and accents where '
        'readable. Mathematical hierarchy will be read independently from the same '
        'source in stage 2. Never invent a flattened formula just to guess its hierarchy.',
        'Return runs as the SINGLE transcription authority. Omit the redundant '
        'text field; the program concatenates runs into text. Preserve spaces '
        'inside runs. Split runs at EVERY visible '
        'foreground color change, even within one word or one line. Each run has '
        'text, script=normal/sub/super and color=#RRGGBB. Background fill is not ink color.',
        'Inspect glyph interiors separately from their surrounding node/border. '
        'A pale or inactive-looking node may still contain BLACK text. Never '
        'infer gray text from gray arrows, borders, absence of fill or semantic role.',
        'No font size estimate. Report a tight glyph-ink bbox; sizing is measured '
        'using actual native rendering downstream.',
        'A mathematical expression is ONE unit including all scripts, fraction bars '
        'and radicals. Ordinary multiline labels remain one unit per visible baseline.',
        'INSTANCE BOUNDARIES: an ordinary sentence on one baseline remains ONE '
        'text object, with separate colored runs rather than separate word boxes. '
        'However, spatially independent tags aligned above nodes, words or columns '
        'are separate objects even when they share a baseline. Do not imitate their '
        'positions by inserting repeated spaces into one artificial line. Give '
        'each independent tag its own tight bbox and its own observed ink color. '
        'Keep attached mathematical scripts with their base expression.',
        'LAYOUT SPACING: never encode horizontal positioning with tabs, repeated '
        'padding spaces or wide Unicode spaces. Example: three independent labels '
        'A, B and C above three columns require THREE texts entries, each with its '
        'own bbox and runs; neither one run "A    B    C" nor three runs in one '
        'text box is valid. Do not merely collapse the spaces to "A B C". '
        'A normal sentence such as "Call Pete tomorrow" remains ONE entry, '
        'including when its words have different colors.',
    ]
    if local:
        rules.append('Use replaces=[existing ID] for corrections, [] for new occurrences. '
                     'Preserve occurrence identity; return regions=[].')
        rules.append('If an old entry incorrectly combined independent labels, '
                     'return separate entries with separately observed bboxes. '
                     'Only the first uses replaces=[old ID]; additional labels '
                     'use replaces=[]. Never repeat the replacement ID.')
    else:
        if opt.get('inventory_local_enabled',True):
            rules.append(f'Propose at most {opt["max_regions"]} regions needing enlarged reading.')
        else:
            rules.append('Return the complete global text inventory and regions=[]. '
                         'Ordinary text crop review is selected later by CPU scores.')
    return {'task': 'Stage 1: text inventory, formula routing and colored style runs.',
            'coordinates': coordinates(size), 'existing_ids': list(known),
            'fonts': ['Times New Roman', 'Arial', 'Calibri', 'Courier New'],
            'font_guidance': 'Use Courier New for monospaced/typewriter/code lettering; do not replace it with Arial.',
            'rules': rules,
            'example': {'texts': [{'kind': 'formula',
                'runs': [{'text': 'S', 'script': 'normal', 'color': '#000000'},
                         {'text': '1', 'script': 'sub', 'color': '#000000'}],
                'bbox_2d': [0, 0, size[0]-1, size[1]-1], 'font': 'Times New Roman',
                'color': '#000000', 'bold': False, 'italic': True, 'underline': False,
                'rotation_deg': 0, 'visual_line_count': 1, 'legibility': 'clear',
                'replaces': []}], 'regions': [], 'uncertain': []},
            'example_notice': 'Syntax example only; do not copy its content or bbox.'}


def marked_runs_equal(text, runs):
    """Accept explicit script markers only where the SAME runs mark scripts.

    Unlike deleting all underscores, this preserves ordinary literal content.
    Adjacent same-script spans are grouped independently of their color splits.
    """
    from .display_equivalence import normalize, equivalent
    if equivalent(text,runs):return True
    groups=[]
    for r in runs:
        value=normalize(r['text'])
        if not value:continue
        if groups and groups[-1][0]==r['script']:
            groups[-1][1]+=value
        else:groups.append([r['script'],value])
    pattern=[]
    for kind,value in groups:
        literal=re.escape(value)
        if kind=='normal':pattern.append(literal)
        else:
            marker='_' if kind=='sub' else r'\^'
            pattern.append('(?:'+literal+'|'+marker+literal+'|'+marker+r'\{'+literal+r'\})')
    return re.fullmatch(''.join(pattern),normalize(text)) is not None


def inventory_validate(raw, size, opt, *, local=False, known=()):
    value = deepcopy(raw)
    if not isinstance(value, dict) or not isinstance(value.get('texts'), list):
        raise ValueError('Expected texts array')
    kinds = []; repairs=[]
    for row in value['texts']:
        kind = row.get('kind')
        if kind not in ('text', 'formula'):
            raise ValueError('Explicit kind=text/formula routing is required')
        if row.get('latex'):
            raise ValueError('Stage 1 must not generate LaTeX')
        runs=row.get('runs')
        if not isinstance(runs,list) or not runs or any(not isinstance(r,dict)
                or not isinstance(r.get('text'),str) or r.get('script') not in ('normal','sub','super') for r in runs):
            raise ValueError('Stage 1 requires explicit transcription runs')
        joined=''.join(r['text'] for r in runs)
        # This catches explicit layout padding, including padding split across
        # colored runs. It does not guess child coordinates or split prose by
        # words: the retry must read independent boxes from source pixels.
        if kind=='text' and re.search(r'\t|[ \u00a0]{3,}|[\u2000-\u2003\u3000]',joined):
            raise ValueError('Layout padding in texts.'+str(len(kinds))+': '
                'do not position independent labels with whitespace or multiple runs '
                'inside one box. Re-read the image and return a separate texts entry '
                'and tight bbox per independent label. Do not just collapse spaces; '
                'keep genuine continuous prose together.')
        if 'text' not in row:row['text']=joined
        elif kind=='formula' and row['text']!=joined and marked_runs_equal(row['text'],runs):
            repairs.append({'index':len(kinds),'field':'text','rule':'explicit_script_markers_agree_with_runs',
                            'before':row['text'],'after':joined})
            row['text']=joined
        kinds.append(kind)
        # Reuse all geometry, color, run and rotation checks, without requiring
        # formula syntax in this deliberately syntax-free stage.
        row.update(kind='text', latex='')
    result = contracts.recognition(value, size, local=local, known=known,
                                   default_font=opt['default_font'])
    result['format_repairs']=result.get('format_repairs',[])+repairs
    for row, kind in zip(result['texts'], kinds):
        row['kind'] = kind
        if kind == 'formula':
            row.update(formula_status='pending', line_layout_status='pending_formula')
    return route_inventory(result,'inventory_validation')


def completed_inventory(root, size, opt):
    """Recover full same-image stage-one responses without re-requesting text.

    Only this separate stage-one protocol qualifies, never old LaTeX responses.
    Preserve original requests/errors and write an explicit revalidation audit.
    """
    for cached in sorted((root/'calls').glob('request_*/content.txt'),reverse=True):
        status_path=cached.parent/'stream_status.json'
        identity_path=cached.parent/'identity.json'
        if not status_path.exists() or not identity_path.exists():continue
        status=read(status_path);identity=read(identity_path)
        if status.get('http_status')!=200 or status.get('finish_reason')!='stop' or not status.get('sse_done'):continue
        if not identity.get('prompt',{}).get('task','').startswith('Stage 1:'):continue
        if sha(root/'input/clean.png') not in [i['sha256'] for i in identity['images']]:continue
        text=cached.read_text().strip()
        if text.startswith('```') and text.endswith('```'):text='\n'.join(text.splitlines()[1:-1])
        try:result=inventory_validate(json.loads(text),size,opt)
        except (ValueError,TypeError,KeyError):continue
        save(root/'revalidation.json',{'source':str(cached),'sha256':sha(cached),
             'new_api_calls':0,'format_repairs':result.get('format_repairs',[])})
        return result
    return None


def formula_prompt(rows, size):
    return {'task': 'Stage 2: read native mathematical structure for fixed occurrences.',
        'coordinates': coordinates(size),
        'targets': [{k: r[k] for k in ('id', 'text', 'runs', 'color', 'rotation_deg')}
                    for r in rows],
        'rules': [
            'Read CLEAN pixels; ID annotations only associate occurrences. They are '
            'not part of the source. Output each target ID exactly once; no new IDs.',
            'Return complete LaTeX for the entire expression including accents, '
            'fractions, radicals, matrices and attached scripts. Do not simplify '
            'mathematics or replace a fraction by a slash. Use unambiguous braced arguments.',
            'Do not return bbox, runs, font size, ordinary text or new text objects. '
            'Stage 1 fixes identity; its text and colors are fallible reading hints, '
            'not authoritative mathematical hierarchy. Resolve hierarchy from the source.',
            r'Re-read actual glyph colors and content from the clean crop. The latest '
            r'valid response overrides earlier text/color estimates when they disagree. '
            r'Preserve observed glyph colors on their exact subexpressions using '
            r'\textcolor{#RRGGBB}{expression}. For uniform color use the supplied '
            r'default color unless the pixels contradict it; explicitly color corrected spans. '
            r'Do not use named colors or optional [rgb]/[HTML] syntax.',
            'Use status=unreadable and latex="" with a reason if the expression '
            'cannot be read. Never fabricate plausible notation.',
            r'Examples: S_1; \frac{a}{\sqrt{b}}; '
            r'\textcolor{#E02040}{x}_{\textcolor{#2255CC}{1}}. '
            'Escape every backslash in JSON. No Markdown code fence.',
        ], 'response_format': {'items': [{'id': 'target ID', 'status': 'resolved/unreadable',
                                          'latex': 'complete expression', 'reason': 'visible evidence'}]}}


def formula_validate(raw, rows):
    if not isinstance(raw, dict) or not isinstance(raw.get('items'), list):
        raise ValueError('Expected formula items array')
    by_id = {r['id']: r for r in rows}
    items = raw['items']
    if (len(items) != len(by_id) or any(not isinstance(i, dict) for i in items)
            or {i.get('id') for i in items} != set(by_id)):
        raise ValueError('Return each formula ID exactly once')
    checked = []
    for item in items:
        if set(item)-{'id', 'status', 'latex', 'reason'}:
            raise ValueError('Formula replies use only id/status/latex/reason; encode content and color corrections in LaTeX')
        status, latex = item.get('status'), item.get('latex')
        if status not in ('resolved', 'unreadable') or not isinstance(latex, str):
            raise ValueError('Invalid formula status/latex')
        if status == 'unreadable':
            if latex.strip() or not item.get('reason'):
                raise ValueError('Unreadable formula needs empty LaTeX and a reason')
        else:
            if not latex.strip() or len(latex) > 16000:
                raise ValueError('Resolved formula needs bounded complete LaTeX')
            latex = normalize_latex_input(latex)
            row = by_id[item['id']]
            tree = latex_to_omml(latex, color=row['color'])
            if not tree.xpath('.//*[local-name()="t"]/text()'):
                raise ValueError('Formula has no visible content')
        checked.append({**item, 'latex': latex})
    return {'items': checked}


def recognize(source, folder, runtime, opt, progress=lambda *a, **k: None):
    """Latest valid formula reply wins; unresolved targets retain explicit fallbacks."""
    folder.mkdir(parents=True, exist_ok=True)
    inventory_file = folder/'inventory.json'
    if inventory_file.exists():
        inventory = read(inventory_file)
    else:
        root = folder/'global'
        f = frame(source, root/'input', opt)
        progress('text_inventory')
        value = completed_inventory(root,f['size'],opt)
        if value is None:
            try:
                value = model.ask(runtime, root, inventory_prompt(f['size'], opt),
                    [('CLEAN original figure', root/'input/clean.png')],
                    lambda v: inventory_validate(v, f['size'], opt), opt)
            except Exception as exc:
                from .inventory_recovery import recover
                value = recover(root,f['size'],opt,runtime,exc)
        rows, history = [], []
        merge(rows, value['texts'], f, 'staged_global', history, opt)
        pending = list(value.get('recovery_warnings',[]))
        from .controller import mapped_box
        regions=value['regions'][:opt['max_regions']] if opt.get('inventory_local_enabled',True) else []
        for index, region in enumerate(regions):
            local = folder/'regions'/f'R{index:02d}'
            lf = frame(source, local/'input', opt, mapped_box(region['bbox_2d'], f, True))
            known = local_hints(local/'input/clean.png', rows, lf, local/'input/ids.png')
            try:
                reply = model.ask(runtime, local, inventory_prompt(lf['size'], opt, local=True, known=known),
                    [('CLEAN enlarged crop', local/'input/clean.png'),
                     ('Identity hints only', local/'input/ids.png')],
                    lambda v: inventory_validate(v, lf['size'], opt, local=True, known=known), opt)
                merge(rows, reply['texts'], lf, 'staged_local', history, opt)
            except Exception as exc:
                pending.append({'region': region, 'error': str(exc)})
        pending += value['regions'][len(regions):]
        inventory = {'texts': rows, 'merge_history': history, 'uncertain': value['uncertain'],
                     'regions_needing_review': pending, 'source_sha256': sha(source)}
        save(inventory_file, inventory)
    if inventory['source_sha256'] != sha(source):
        raise ValueError('Inventory source mismatch')
    from .inventory_audit import audit as audit_inventory
    inventory = audit_inventory(inventory, source, folder/'visual_audit', runtime, opt, progress)
    inventory=route_inventory(inventory,'cached_inventory_before_formula')
    from .multiline_retry import repair as repair_multiline
    if opt.get('multiline_before_baseline',True):
        result = repair_multiline(inventory,source,folder/'multiline_retry',runtime,opt,progress)
    else:
        # Known multiline failures are resolved jointly in the single low-score crop.
        result = deepcopy(inventory)
    save(folder/'line_checked_inventory.json',result)
    with Image.open(source) as im:
        groups = compact_groups([r for r in result['texts'] if r['kind']=='formula'], im.size, 6)
    by_id = {r['id']: r for r in result['texts']}
    failures = []
    from .formula_state import adopt, keep_previous
    for index, group in enumerate(groups):
        progress('formula_structure', group=index+1, total=len(groups))
        root = folder/'formulas'/f'F{index:03d}'
        f = frame(source, root/'input', {**opt, 'crop_min_padding_px': 32}, union(group))
        local_hints(root/'input/clean.png', group, f, root/'input/ids.png')
        try:
            reply = model.ask(runtime, root, formula_prompt(group, f['size']),
                [('CLEAN enlarged formula crop', root/'input/clean.png'),
                 ('Formula IDs only; do not read annotations', root/'input/ids.png')],
                lambda v: formula_validate(v, group), opt)
            for item in reply['items']:
                row = by_id[item['id']]
                if item['status']=='resolved':row.update(adopt(row,item))
                else:
                    failures.append(item)
                    row.update(keep_previous(row,item.get('reason','unreadable formula')))
        except Exception as exc:
            # A group request may fail because of one item. Recover each valid
            # completed item independently, newest first, without another API.
            recovered={}
            for saved in sorted((root/'calls').glob('request_*/content.txt'),reverse=True):
                status_path=saved.parent/'stream_status.json'
                if not status_path.exists():continue
                status=read(status_path)
                if status.get('http_status')!=200 or status.get('finish_reason')!='stop' or not status.get('sse_done'):continue
                try:
                    content=saved.read_text().strip()
                    if content.startswith('```'):content='\n'.join(content.splitlines()[1:-1])
                    items=json.loads(content).get('items',[])
                    for row in group:
                        candidates=[i for i in items if isinstance(i,dict) and i.get('id')==row['id']]
                        if row['id'] in recovered or len(candidates)!=1:continue
                        try:
                            item=formula_validate({'items':candidates},[row])['items'][0]
                            if item['status']=='resolved':recovered[row['id']]=adopt(row,item)
                        except (ValueError,TypeError,KeyError):continue
                except (ValueError,TypeError,AttributeError):continue
            for row in group:
                if row['id'] in recovered:
                    row.update(recovered[row['id']]);continue
                failures.append({'id':row['id'],'error':str(exc)})
                row.update(keep_previous(row,str(exc)))
    save(folder/'formula_summary.json', {'formulas': sum(len(g) for g in groups),
         'groups': len(groups), 'unresolved': failures, 'conflict_policy':'latest_valid_reply_wins'})
    result['regions_needing_review']+= [{'stage':'formula_structure',**f} for f in failures]
    save(folder/'resolved_inventory.json', result)
    return result
