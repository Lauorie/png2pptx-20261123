"""Conservative whole-occurrence routing: simple baseline numbers are text."""
from copy import deepcopy
import re


# No whitespace joining, substring extraction, or Unicode compatibility folding:
# those would turn separate labels, scripts, or mathematical alphabets into digits.
NUMBER = re.compile(r'[+\-−]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)%?')
RULE = 'whole_baseline_number_to_text_v1'
SYMBOLS = set('+-−=<>≤≥≠×÷±∓[](){}|:;,!?')


def route(row, stage):
    """Return a copy; never split an expression or infer missing geometry/style.

    Unknown LaTeX commands remain formulas. Even apparently decorative commands
    may carry style not represented by runs; preserving them is safer than
    stripping syntax and losing accents, scripts, color or emphasis.
    """
    result=deepcopy(row)
    if row.get('kind')!='formula':return result
    text=row.get('text');runs=row.get('runs')
    if not isinstance(text,str):return result
    numeric = NUMBER.fullmatch(text) is not None
    symbol = len(text)==1 and text in SYMBOLS
    if not (numeric or symbol):return result
    if not isinstance(runs,list) or not runs or any(
            not isinstance(r,dict) or r.get('script')!='normal'
            or not isinstance(r.get('text'),str) for r in runs):return result
    if ''.join(r['text'] for r in runs)!=text:return result
    if row.get('visual_line_count',1)!=1:return result
    if row.get('legibility','clear')!='clear':return result
    # Explicitly identified formula fragments must stay with their expression.
    if row.get('formula_parent_id') or row.get('part_of_formula'):return result
    latex=row.get('latex','')
    if not isinstance(latex,str):return result
    # Stage 1 has no LaTeX. Later stages must agree on the ENTIRE expression.
    # Reject braces, commands, script markers, or mismatched display summaries.
    if latex.strip() and latex.strip()!=text:return result
    previous={k:deepcopy(row.get(k)) for k in ('kind','latex','formula_status','line_layout_status')}
    result.update(kind='text',latex='',formula_status='not_applicable',
                  line_layout_status='single_baseline')
    result.setdefault('routing_audit',[]).append({'rule':RULE if numeric else 'whole_baseline_symbol_to_text_v1','stage':stage,
        'reason':'Complete numeric or literal symbol occurrence; normal runs; no LaTeX structure',
        'before':previous,'after':{'kind':'text','latex':''}})
    return result


def inventory(value, stage):
    result=deepcopy(value)
    result['texts']=[route(row,stage) for row in result['texts']]
    return result
