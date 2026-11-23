"""Lossless schema reconciliation; never invent or replace non-space characters."""
from copy import deepcopy


def reconcile_run_spaces(text, runs):
    """Preserve authoritative text and every non-space character's run style.

    Only ASCII spaces may differ. Case, symbols, tabs, order, scripts and colors
    are not normalized. Spaces at a run boundary inherit the preceding run.
    """
    joined=''.join(r['text'] for r in runs)
    if joined==text:return deepcopy(runs)
    if joined.replace(' ','')!=text.replace(' ','') or not text.replace(' ',''):
        raise ValueError('Non-space transcription differs')
    owners=[i for i,r in enumerate(runs) for c in r['text'] if c!=' ']
    repaired=deepcopy(runs)
    for r in repaired:r['text']=''
    position=0;previous=owners[0]
    for char in text:
        if char!=' ':previous=owners[position];position+=1
        repaired[previous]['text']+=char
    assert ''.join(r['text'] for r in repaired)==text
    return repaired
