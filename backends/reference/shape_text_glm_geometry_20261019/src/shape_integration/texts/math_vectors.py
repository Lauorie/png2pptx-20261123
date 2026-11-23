"""Recognize explicitly separated numerical vectors without guessing symbols."""
from copy import deepcopy
import re
from lxml import etree

M='http://www.w3.org/1998/Math/MathML'
q=lambda name:'{'+M+'}'+name


def space(node):
    if node.tag==q('mtext'):return bool(node.text) and node.text.isspace()
    if node.tag==q('mspace'):
        width=node.get('width','')
        match=re.match(r'^([+-]?(?:\d+(?:\.\d*)?|\.\d+))',width)
        return float(match[1])>0 if match else bool(width and not width.startswith('negative'))
    return False


def numeric(nodes):
    allowed={q(v) for v in ('mn','mo','mstyle','mrow')}
    if any(n.tag not in allowed for root in nodes for n in root.iter()):return False
    value=''.join(''.join(n.itertext()) for n in nodes).strip()
    return bool(re.fullmatch(r'[+−-]?(?:\d+(?:\.\d*)?|\.\d+)',value))


def restructure(tree, leading_gap_em=None):
    """Replace [number SPACE number ...] by one editable one-row matrix.

    Only explicit positive whitespace separates cells. [1,2], [a b], indexed
    expressions, intervals and existing fraction/radical/matrix structures are
    not inferred or rewritten. Colors/styles remain inside their own cells.
    """
    count=0
    for parent in list(tree.iter(q('mrow'))):
        children=list(parent);i=0
        while i<len(children):
            start=children[i]
            if start.tag!=q('mo') or start.text!='[':i+=1;continue
            end=next((j for j in range(i+1,len(children)) if
                children[j].tag==q('mo') and children[j].text==']'),None)
            if end is None:i+=1;continue
            cells=[];cell=[]
            for node in children[i+1:end]:
                if space(node):
                    if cell:cells.append(cell);cell=[]
                else:cell.append(node)
            if cell:cells.append(cell)
            if not 2<=len(cells)<=32 or not all(numeric(c) for c in cells):i=end+1;continue
            group=etree.Element(q('mrow'));group.append(deepcopy(start))
            table=etree.SubElement(group,q('mtable'));row=etree.SubElement(table,q('mtr'))
            for cell in cells:
                target=etree.SubElement(row,q('mtd'))
                for node in cell:target.append(deepcopy(node))
            group.append(deepcopy(children[end]))
            if leading_gap_em is not None and i and space(children[i-1]):
                gap=etree.Element(q('mspace'));gap.set('width',str(leading_gap_em)+'em')
                parent.replace(children[i-1],gap)
            parent.insert(parent.index(start),group)
            for node in children[i:end+1]:parent.remove(node)
            count+=1;i=end+1
    return count
