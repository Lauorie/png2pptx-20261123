"""Native formula regression: nested styling must never become visible TeX."""
import pytest

from full_recovery.formula_text import install


@pytest.fixture(autouse=True)
def converter():
    install()


def test_colored_bold_typewriter_group_retains_text_styles_and_scripts():
    from shape_integration.texts.omml import latex_to_omml, OMML_NS, DRAWINGML_NS
    tree=latex_to_omml(r'z_{j}^{4}\ \texttt{[0.2\ \textcolor{#12ab34}{\mathbf{0.8}}]}')
    ns={'m':OMML_NS,'a':DRAWINGML_NS}
    visible=''.join(tree.xpath('.//m:t/text()',namespaces=ns))
    assert '\\' not in visible and 'textcolor' not in visible
    assert '[0.2' in visible and '0.8]' in visible
    assert tree.xpath('.//m:sub//m:t/text()',namespaces=ns)==['j']
    assert tree.xpath('.//m:sup//m:t/text()',namespaces=ns)==['4']
    colored=tree.xpath('.//m:r[a:rPr/a:solidFill/a:srgbClr/@val="12AB34"]',namespaces=ns)
    assert ''.join(r.find('m:t',ns).text for r in colored)=='0.8'
    assert all(r.find('a:rPr',ns).get('b')=='1' for r in colored)
    assert all(r.find('a:rPr/a:latin',ns).get('typeface')=='Courier New' for r in colored)


def test_nested_braces_escaped_characters_and_ordinary_math_survive():
    from shape_integration.texts.omml import latex_to_omml, OMML_NS
    tree=latex_to_omml(r'\frac{x^2}{y}+\texttt{\{a\_b\}\ \textit{C}}')
    ns={'m':OMML_NS}
    assert tree.find('.//m:f',ns) is not None
    assert '{a_b}' in ''.join(tree.xpath('.//m:t/text()',namespaces=ns))
    assert tree.find('.//m:sSup',ns) is not None


@pytest.mark.parametrize('latex',[
    r'\texttt{\textcolor{#12345}{x}}',r'\texttt{\unknown{x}}',r'\texttt{unclosed'])
def test_bad_text_groups_fail_instead_of_leaking_commands(latex):
    from shape_integration.texts.omml import latex_to_omml
    with pytest.raises(ValueError):latex_to_omml(latex)
