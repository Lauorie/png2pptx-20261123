from copy import deepcopy
import pytest
from shape_integration.texts.format_normalization import reconcile_run_spaces
from shape_integration.texts import contracts
from shape_integration.texts.omml import normalize_latex_input,latex_to_omml,FormulaConversionError
from shape_integration.texts.prompt_protocol import recognize


def test_spaces_repaired_without_mutation_or_style_loss():
    runs=[{'text':'k','script':'normal','color':'#FF0000'},
          {'text':'/','script':'normal'},{'text':'v','script':'sub','color':'#0000FF'}]
    old=deepcopy(runs)
    result=reconcile_run_spaces(' k / v ',runs)
    assert ''.join(r['text'] for r in result)==' k / v '
    assert result[2]['script']=='sub' and result[2]['color']=='#0000FF'
    assert runs==old


@pytest.mark.parametrize('text',['k - v','K / v','k / w','v / k','k\t/v'])
def test_character_changes_are_never_normalized(text):
    with pytest.raises(ValueError):reconcile_run_spaces(text,[{'text':'k/v','script':'normal'}])


@pytest.mark.parametrize('latex,expected',[
    (r'\textcolor[rgb]{0.9,0.1,0.3}{x_1}',r'\textcolor{#E61A4C}{x_1}'),
    (r'\textcolor[RGB]{224,32,64}{x_1}',r'\textcolor{#E02040}{x_1}'),
    (r'\textcolor[HTML]{e02040}{x_1}',r'\textcolor{#E02040}{x_1}')])
def test_explicit_color_models_canonicalize(latex,expected):
    assert normalize_latex_input(latex)==expected
    assert latex_to_omml(latex) is not None


@pytest.mark.parametrize('spec',['[rgb]{2,0,0}','[rgb]{nan,0,0}','[RGB]{1.2,2,3}','[HTML]{bad}','[rgb]{0,0}'])
def test_invalid_colors_still_fail(spec):
    with pytest.raises(FormulaConversionError):normalize_latex_input('\\textcolor'+spec+'{x}')


def test_overescaped_control_spaces_are_restored_outside_multiline_math():
    source=r'e^2_0\\ [0.00\\ 0.00\\ \textcolor{#00AA55}{1.00}]'
    expected=r'e^2_0\ [0.00\ 0.00\ \textcolor{#00AA55}{1.00}]'
    assert normalize_latex_input(source)==expected
    assert normalize_latex_input(expected)==expected
    from shape_integration.texts.equations import NS
    assert any(t.isspace() for t in latex_to_omml(source).xpath('.//m:t/text()',namespaces=NS))


@pytest.mark.parametrize('environment',['bmatrix','array','aligned','align','gathered','split'])
def test_intentional_multiline_separators_are_not_rewritten(environment):
    source='\\begin{'+environment+r'}1 & 2\\ 3 & 4\end{'+environment+'}'
    assert normalize_latex_input(source)==source


def test_prompt_examples_validate_and_spaces_audit():
    prompt=recognize([200,100]); examples=prompt['format_examples']['texts']
    parsed=contracts.recognition({'texts':examples},[200,100])
    assert len(parsed['texts'])==4
    examples[0]['runs'][1]['text']='/'
    original=deepcopy(examples)
    parsed=contracts.recognition({'texts':examples},[200,100])
    assert parsed['texts'][0]['text']=='k / v'
    assert ''.join(r['text'] for r in parsed['texts'][0]['runs'])=='k / v'
    assert parsed['format_repairs'][0]['rule']=='ascii_spaces_only_text_authoritative'
    assert examples==original
