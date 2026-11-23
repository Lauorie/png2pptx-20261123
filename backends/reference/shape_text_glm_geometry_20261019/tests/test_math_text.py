import pytest
from lxml import etree
from latex2mathml.converter import convert
from shape_integration.texts.math_text import prepare,restore,MATHML


def parse(value):
    rewritten,table=prepare(value)
    return restore(etree.fromstring(convert(rewritten).encode()),table)


def test_nested_text_style_preserves_content_and_math_boundaries():
    tree=parse(r'\text{prefix \textbf{bold \textit{both}} suffix}+x_2')
    leaves=list(tree.iter('{'+MATHML+'}mtext'))
    assert ''.join(x.text for x in leaves)=='prefix bold both suffix'
    assert [x.get('mathvariant') for x in leaves]==['normal','bold','bold-italic','normal']
    assert len(list(tree.iter('{'+MATHML+'}msub')))==1


def test_nested_color_and_escaped_braces_are_literal_text():
    tree=parse(r'\text{\textcolor{#AB1234}{\textbf{A\{B\}}}}')
    leaf=next(tree.iter('{'+MATHML+'}mtext'))
    assert leaf.text=='A{B}' and leaf.get('mathcolor')=='#AB1234'
    assert leaf.get('mathvariant')=='bold'


def test_unsupported_or_unbalanced_text_fails_instead_of_printing_commands():
    for value in [r'\text{\unknown{word}}',r'\text{missing']:
        with pytest.raises(ValueError):prepare(value)


def test_outer_math_style_is_inherited_and_explicit_normal_can_reset_it():
    tree=parse(r'\mathbf{\text{Bold \textnormal{plain}}}')
    leaves=list(tree.iter('{'+MATHML+'}mtext'))
    assert [x.get('mathvariant') for x in leaves]==['bold','normal']


def test_native_omml_keeps_nested_style_color_and_literal_content():
    from shape_integration.texts.omml import latex_to_omml
    from shape_integration.texts.equations import NS
    tree=latex_to_omml(r'\textcolor{#EF1234}{\text{\textbf{Label, }}}x_1')
    text=''.join(tree.xpath('.//m:t/text()',namespaces=NS))
    assert text=='Label, x1' and '\\' not in text
    assert tree.xpath('.//a:rPr[@b="1"]/a:solidFill/a:srgbClr/@val',namespaces=NS)==['EF1234']
    assert tree.find('.//m:sSub',NS) is not None


def test_real_preview_accepts_script_signs_and_nested_styled_text(tmp_path):
    import threading
    import fitz
    from shape_integration.texts.equations import measure
    rows=[{'id':'charge','latex':r'\hat{q}^{+}','color':'#000000'},
          {'id':'label','latex':r'\textcolor{#EF1234}{\text{\textbf{Label, }}}\mathcal{J}_n','color':'#EF1234'}]
    measure(rows,tmp_path,threading.BoundedSemaphore(1))
    with fitz.open(tmp_path/'black/black.pdf') as document:
        text=''.join(page.get_text() for page in document)
    assert '¿' not in text and '+' in text and 'Label' in text and '\\textbf' not in text
