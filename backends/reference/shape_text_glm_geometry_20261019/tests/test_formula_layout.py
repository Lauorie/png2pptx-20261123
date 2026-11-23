from copy import deepcopy
from PIL import Image
import pytest

from shape_integration.texts import equations,formula_layout
from shape_integration.texts.omml import latex_to_omml


def vector(**kwargs):
    return {'id':'v','text':'e02 [0.0 1.0]',
            'latex':r'e^2_0\ [0.0\ \textcolor{#00AA55}{\mathbf{1.0}}]',
            'kind':'formula','bbox_2d':[10,10,210,40],'color':'#000000',
            'font':'Arial','bold':False,'italic':False,'underline':False,
            'rotation_deg':0,'runs':[{'text':'e','script':'normal'}],**kwargs}


def test_native_numeric_vector_preserves_order_color_scripts_and_bold():
    row=vector();before=deepcopy(row)
    baseline=latex_to_omml(row['latex'])
    tree=latex_to_omml(row['latex'],numeric_vectors=True,vector_gap_em=1,
                       numeric_font_family='Arial')
    assert baseline.find('.//m:m',equations.NS) is None
    assert len(tree.findall('.//m:m/m:mr/m:e',equations.NS))==2
    assert tree.find('.//m:sSubSup',equations.NS) is not None
    assert '00AA55' in tree.xpath('.//a:srgbClr/@val',namespaces=equations.NS)
    assert tree.find('.//m:m//a:rPr[@b="1"]',equations.NS) is not None
    assert tree.find('.//m:m//a:latin[@typeface="Arial"]',equations.NS) is not None
    normalize=lambda t:''.join(''.join(t.xpath('.//m:t/text()',namespaces=equations.NS)).split())
    # Native matrix brackets move from m:t into structural delimiter metadata.
    assert normalize(baseline).replace('[','').replace(']','')==normalize(tree)
    delimiter=tree.find('.//m:d/m:dPr',equations.NS)
    assert delimiter.find('m:begChr',equations.NS).get('{'+equations.OMML_NS+'}val')=='['
    assert delimiter.find('m:endChr',equations.NS).get('{'+equations.OMML_NS+'}val')==']'
    assert row==before


@pytest.mark.parametrize('latex',[
    r'[a\ b]',r'[1,2]',r'[0.0,1.0]',r'[x_1\ x_2]',
    r'[\frac{1}{2}\ 3]',r'[\sqrt{2}\ 3]',r'[12]',r'[1\! 2]'])
def test_nonvector_expressions_are_not_inferred(latex):
    assert latex_to_omml(latex,numeric_vectors=True).find('.//m:m',equations.NS) is None


def test_text_and_explicit_gaps_are_literal_but_math_variables_stay_math():
    tree=latex_to_omml(r'\text{Train both }\theta\text{ and }\omega\quad x')
    for run in tree.findall('.//m:r',equations.NS):
        value=run.find('m:t',equations.NS).text
        if 'and' in value or value.isspace():assert run.find('m:rPr/m:nor',equations.NS) is not None
        if value in ('θ','ω','x'):assert run.find('m:rPr/m:nor',equations.NS) is None


def test_word_preview_and_native_ppt_share_the_same_layout_profile():
    row=vector(native_formula_layout={'numeric_vectors':True,'vector_gap_em':1,
                                      'numeric_font_family':'Arial'})
    tree=equations.native_tree(row)
    word=equations.word_math(row)
    assert len(tree.findall('.//m:m/m:mr/m:e',equations.NS))==2
    assert len(word.findall('.//m:m/m:mr/m:e',equations.NS))==2
    ns={**equations.NS,'w':equations.W}
    assert word.find('.//m:m//w:rFonts[@w:ascii="Arial"]',ns) is not None


@pytest.mark.parametrize('improves',[True,False])
@pytest.mark.parametrize('low_proxy',[True,False])
def test_cpu_selection_keeps_boxes_latex_content_and_styles(tmp_path,monkeypatch,improves,low_proxy):
    row=vector();before=deepcopy(row);source=tmp_path/'source.png'
    Image.new('RGB',(240,80),'white').save(source)
    metric={'width':60,'height':20,'reference_size_px':24}
    monkeypatch.setattr(formula_layout,'eligible',lambda *a:True)
    def score(candidate,metric,image):
        modified='native_formula_layout' in candidate
        base=.15 if low_proxy else .5
        return {'score':base+.1 if modified and improves else base-.1 if modified else base,
                'source_mass':40,'predicted_ink_size_px':[150,30]}
    monkeypatch.setattr(formula_layout,'preview_score',score)
    from shape_integration.texts import equation_recovery
    monkeypatch.setattr(equation_recovery,'measure_resilient',
        lambda candidates,*a:({r['id']:metric for r in candidates},[]))
    selected,metrics=formula_layout.select([row],{'v':metric},source,tmp_path/'fit',None,
                                           {'render_timeout_seconds':180})
    assert row==before
    assert selected[0]['bbox_2d']==before['bbox_2d']
    for key in ('latex','text','runs','color','font','bold','italic','rotation_deg'):
        assert selected[0][key]==before[key]
    assert ('native_formula_layout' in selected[0])==improves
