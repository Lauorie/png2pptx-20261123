"""Keep mathematical letter styling in degraded plain-run previews."""
from copy import deepcopy
import unicodedata

def letter(char):
    name=unicodedata.name(char,'')
    for prefix,style in [('MATHEMATICAL BOLD ITALIC ','bi'),
                         ('MATHEMATICAL BOLD ','b'),('MATHEMATICAL ITALIC ','i')]:
        if name.startswith(prefix) and name[len(prefix):].startswith(('CAPITAL ','SMALL ','DIGIT ')):
            parts=unicodedata.decomposition(char).split()
            if len(parts)==2 and parts[0]=='<font>':return chr(int(parts[1],16)),style
    return char,None


def preview_row(row):
    """Use base glyphs plus weight/slant only in the disposable fallback.

    The recognized row and native LaTeX/OMML stay untouched. Arial contains K
    but lacks the supplementary mathematical bold K; drawing the latter with
    a plain-run font silently produces a box instead of the label.
    """
    result=deepcopy(row);runs=[];changes=[]
    for run in row['runs']:
        chunks=[]
        for char in run['text']:
            value,style=letter(char)
            if chunks and chunks[-1][1]==style:chunks[-1]=(chunks[-1][0]+value,style)
            else:chunks.append((value,style))
            if style:changes.append({'before':char,'after':value,'style':style})
        for value,style in chunks:
            native={**run,'text':value}
            if style:native.update(bold='b' in style,italic='i' in style)
            runs.append(native)
    result['runs']=runs;result['text']=''.join(r['text']for r in runs)
    return result,changes
