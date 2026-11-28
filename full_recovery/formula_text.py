"""Parse balanced text/style groups before handing math syntax to latex2mathml.

The converter's text-token handling is not brace-aware for nested formatting.
Only text-mode groups are replaced; mathematical structure is left untouched.
"""
import re
from lxml import etree

MATHML='http://www.w3.org/1998/Math/MathML'
COMMANDS={'text','textbf','textit','textrm','textnormal','texttt'}
# Accept these style aliases only inside a literal text group. Mathematical
# expressions outside text mode still go through the original math parser.
ALIASES={'mathbf':'textbf','mathit':'textit','mathrm':'textrm','mathtt':'texttt'}
# Text-mode boxes that carry no style of their own.
BOXES={'mbox','hbox'}
PROTOCOL='balanced_formula_text.3'


def text_group(command: str) -> bool:
    """Whether ``command`` opens a text-mode group: the ``\\text...`` family and plain boxes.

    A style this parser has no rendering for (``\\textsf``, ``\\textsc``...)
    still opens text mode: its content is text and inherits the surrounding
    style. Left to the math converter, such a wrapper printed the placeholder
    of the group inside it.
    """
    return command in COMMANDS or command in BOXES or (command!='textcolor' and re.fullmatch(r'text[a-z]+',command) is not None)


def braced(source, index):
    while index<len(source) and source[index].isspace():index+=1
    if index>=len(source) or source[index]!='{':raise ValueError('Expected braced text argument')
    start=index+1;depth=1;index=start
    while index<len(source):
        if source[index]=='\\':index+=2;continue
        if source[index]=='{':depth+=1
        elif source[index]=='}':
            depth-=1
            if depth==0:return source[start:index],index+1
        index+=1
    raise ValueError('Unclosed text group')


def spans(source, bold=None, italic=None, color=None, family=None):
    result=[];buffer=[]
    def flush():
        if buffer:
            result.append((''.join(buffer),bold,italic,color,family));buffer.clear()
    i=0
    while i<len(source):
        char=source[i]
        if char=='{':
            flush();inner,i=braced(source,i);result.extend(spans(inner,bold,italic,color,family));continue
        if char=='}':raise ValueError('Unexpected text closing brace')
        if char=='\\':
            if i+1<len(source) and source[i+1] in '{}%_&#$ ':
                buffer.append(source[i+1]);i+=2;continue
            match=re.match(r'\\([A-Za-z]+)',source[i:])
            if match is None:raise ValueError('Unsupported text escape')
            command=match[1];i+=len(match[0]);flush()
            if command=='textcolor':
                value,i=braced(source,i)
                if not re.fullmatch(r'#[0-9A-Fa-f]{6}',value):raise ValueError('Text color must be #RRGGBB')
                inner,i=braced(source,i);result.extend(spans(inner,bold,italic,value.upper(),family));continue
            command=ALIASES.get(command,command)
            if not text_group(command):raise ValueError('Unsupported nested text command: '+command)
            inner,i=braced(source,i)
            b=False if command=='textnormal' else True if command=='textbf' else bold
            it=False if command in ('textnormal','textrm','texttt') else True if command=='textit' else italic
            face='Courier New' if command=='texttt' else None if command in ('textnormal','textrm') else family
            result.extend(spans(inner,b,it,color,face));continue
        buffer.append('\u00a0' if char=='~' else char);i+=1
    flush();return result


def prepare(source):
    """Return math-preserving placeholders plus their styled text leaves."""
    marker='A2FTextSlot'
    while marker in source:marker+='X'
    output=[];replacements={};i=0
    while i<len(source):
        if source[i]!='\\':output.append(source[i]);i+=1;continue
        match=re.match(r'\\([A-Za-z]+)',source[i:])
        if match is None:output.append(source[i:i+2]);i+=2;continue
        command=match[1]
        if not text_group(command):output.append(match[0]);i+=len(match[0]);continue
        inner,end=braced(source,i+len(match[0]))
        key=marker+str(len(replacements))
        replacements[key]=spans(inner,True if command=='textbf' else False if command=='textnormal' else None,
                               True if command=='textit' else False if command in ('textnormal','textrm','texttt') else None,
                               family='Courier New' if command=='texttt' else None)
        output.append('\\text{'+key+'}');i=end
    return ''.join(output),replacements


def restore(tree, replacements):
    for node in list(tree.iter('{'+MATHML+'}mtext')):
        key=''.join(node.itertext())
        if key not in replacements:continue
        inherited=node.get('mathvariant','normal')
        group=etree.Element('{'+MATHML+'}mrow')
        for text,bold,italic,color,family in replacements[key]:
            bold='bold' in inherited if bold is None else bold
            italic='italic' in inherited if italic is None else italic
            leaf=etree.SubElement(group,'{'+MATHML+'}mtext')
            leaf.text=text;leaf.set('mathvariant','bold-italic' if bold and italic else 'bold' if bold else 'italic' if italic else 'normal')
            if color:leaf.set('mathcolor',color)
            if family:leaf.set('fontfamily',family)
        node.getparent().replace(node,group)
    return tree


def install():
    """Use the same balanced parser in future runs without editing old freezes."""
    from shape_integration.texts import math_text, omml
    math_text.prepare=prepare
    math_text.restore=restore
    if getattr(omml._MathMLConverter,'_balanced_text_protocol',None)==PROTOCOL:
        return
    original=omml._MathMLConverter._convert
    def convert(self,node):
        runs=original(self,node)
        if node.tag=='{'+MATHML+'}mtext' and node.get('fontfamily'):
            for run in runs:
                latin=run.find('{'+omml.DRAWINGML_NS+'}rPr/{'+omml.DRAWINGML_NS+'}latin')
                if latin is not None:latin.set('typeface',node.get('fontfamily'))
        return runs
    omml._MathMLConverter._convert=convert
    omml._MathMLConverter._balanced_text_protocol=PROTOCOL
