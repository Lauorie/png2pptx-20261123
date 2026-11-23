"""Reconcile redundant display metadata only when structured math agrees.

LaTeX remains unchanged. This limited parser never guesses a mathematical
expression from OCR or silently drops conflicting subscripts/accents.
"""
import unicodedata
from lxml import etree
from latex2mathml.converter import convert
from .omml import normalize_latex_input
from .display_equivalence import normalize, runs_structure


ACCENTS = {'^': '\u0302', 'ˆ': '\u0302', '~': '\u0303', '˜': '\u0303',
           '¯': '\u0304', '˙': '\u0307', '¨': '\u0308'}


def simple_structure(latex):
    """Single base with optional accent and scripts; reject richer syntax."""
    try:root = etree.fromstring(convert(normalize_latex_input(latex)).encode())
    except Exception as exc:raise ValueError('Unparseable math evidence') from exc
    def parse(node):
        tag = etree.QName(node).localname
        if tag in ('math', 'mrow', 'mstyle'):
            children = list(node)
            if len(children) == 1:
                return parse(children[0])
            parts = [parse(c) for c in children]
            if any(s for _, s in parts):
                raise ValueError('Multiple scripted bases')
            return ''.join(b for b, _ in parts), {}
        if tag in ('mi', 'mn', 'mo', 'mtext'):
            return node.text or '', {}
        if tag == 'mover' and len(node) == 2:
            base, scripts = parse(node[0]); accent, extra = parse(node[1])
            if scripts or extra or accent not in ACCENTS or len(base) != 1:
                raise ValueError('Unsupported accent')
            return unicodedata.normalize('NFC', base + ACCENTS[accent]), {}
        if tag in ('msub', 'msup', 'msubsup'):
            base, scripts = parse(node[0]); scripts = dict(scripts)
            keys = ('sub', 'super') if tag == 'msubsup' else (('sub',) if tag == 'msub' else ('super',))
            for key, child in zip(keys, list(node)[1:]):
                value, extra = parse(child)
                if extra or key in scripts:
                    raise ValueError('Nested or duplicate script')
                scripts[key] = normalize(value)
            return base, scripts
        raise ValueError('Unsupported structure')
    base, scripts = parse(root)
    return normalize(base), scripts


def reconcile(text, runs, latex):
    try:
        base, scripts = simple_structure(latex)
    except (ValueError, TypeError, etree.XMLSyntaxError):
        return None
    observed = runs_structure(runs)
    if observed is None:
        return None
    run_base, run_scripts = observed
    if scripts != run_scripts:
        return None
    bare = ''.join(c for c in unicodedata.normalize('NFD', base) if not unicodedata.combining(c))
    # Omitted accent in runs is allowed because LaTeX preserves it. A different
    # accent or different base is a genuine semantic conflict, never normalized.
    if unicodedata.normalize('NFC', run_base) not in (unicodedata.normalize('NFC', base), bare):
        return None
    joined = ''.join(r['text'] for r in runs)
    suffix = ''.join(r['text'] for r in runs if r['script'] != 'normal')
    forms = {normalize(base), normalize(bare), normalize(base + suffix),
             normalize(joined), normalize(normalize_latex_input(latex))}
    if normalize(unicodedata.normalize('NFC', text)) not in forms:
        return None
    return joined
