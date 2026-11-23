from copy import deepcopy
from zipfile import ZipFile

from lxml import etree
import pytest

from shape_integration.texts import options
from shape_integration.texts.staged_formula import (
    inventory_prompt, inventory_validate, formula_validate,
)
from shape_integration.texts.omml import latex_to_omml


def row(kind='formula'):
    return {'id': 'T0001', 'text': 'S1', 'kind': kind,
            'bbox_2d': [10, 10, 40, 40], 'font': 'Times New Roman',
            'color': '#E02040', 'runs': [
                {'text': 'S', 'script': 'normal', 'color': '#E02040'},
                {'text': '1', 'script': 'sub', 'color': '#2255CC'}]}


def test_inventory_routes_formula_without_requiring_latex():
    original = row()
    value = inventory_validate({'texts': [original]}, [100,100], options({}))
    result = value['texts'][0]
    assert result['kind']=='formula' and result['latex']==''
    assert result['formula_status']=='pending'
    assert result['runs']==original['runs']


def test_runs_are_single_authority_when_text_is_omitted():
    original=row();original.pop('text')
    result=inventory_validate({'texts':[original]},[100,100],options({}))['texts'][0]
    assert result['text']=='S1'


def test_structured_marker_compatibility_never_drops_literal_underscores():
    from shape_integration.texts.staged_formula import marked_runs_equal
    runs=[{'text':'x','script':'normal'},{'text':'T','script':'sub'},
          {'text':'+x','script':'normal'},{'text':'0','script':'sub'}]
    assert marked_runs_equal('x_T+x_{0}',runs)
    assert not marked_runs_equal('x_1+x_0',runs)
    assert not marked_runs_equal('x_T+x_0extra',runs)
    assert not marked_runs_equal('literal_name',[{'text':'literalname','script':'normal'}])


def test_inventory_rejects_latex_and_missing_routing():
    for changed in ({**row(),'latex':'S_1'}, {**row(),'kind':None}):
        with pytest.raises(ValueError):
            inventory_validate({'texts':[changed]}, [100,100], options({}))


def test_inventory_preserves_multicolor_ordinary_line():
    original = row('text')
    original['runs'][1]['script']='normal'
    result = inventory_validate({'texts':[original]}, [100,100], options({}))['texts'][0]
    assert result['runs']==original['runs']
    assert result['kind']=='text'


def test_inventory_prompt_distinguishes_sentence_runs_from_independent_tags():
    rules=' '.join(inventory_prompt([100,100],options({}))['rules'])
    assert 'sentence on one baseline remains ONE' in rules
    assert 'spatially independent tags' in rules
    assert 'Do not imitate their positions by inserting repeated spaces' in rules
    assert 'Keep attached mathematical scripts with their base expression' in rules


@pytest.mark.parametrize('text',['A    B    C','A\tB','A\u2003B','A\u3000B'])
def test_layout_padding_requests_separate_boxes_instead_of_guessing_geometry(text):
    original=row('text');original.pop('text')
    original['runs']=[{'text':text,'script':'normal','color':'#000000'}]
    before=deepcopy(original)
    with pytest.raises(ValueError,match='separate texts entry'):
        inventory_validate({'texts':[original]},[100,100],options({}))
    assert original==before


def test_padding_split_across_colored_runs_is_still_rejected():
    original=row('text');original.pop('text')
    original['runs']=[{'text':'A  ','script':'normal','color':'#CC0000'},
                      {'text':' B','script':'normal','color':'#0000CC'}]
    with pytest.raises(ValueError,match='Layout padding'):
        inventory_validate({'texts':[original]},[100,100],options({}))


def test_sentence_spaces_and_formula_inventory_are_not_split():
    original=row('text');original.pop('text')
    original['runs']=[{'text':'Call ','script':'normal','color':'#CC0000'},
                      {'text':'Pete tomorrow','script':'normal','color':'#0000CC'}]
    result=inventory_validate({'texts':[original]},[100,100],options({}))
    assert len(result['texts'])==1
    assert result['texts'][0]['runs']==original['runs']
    formula=row();formula.pop('text')
    formula['runs'][0]['text']='S   '
    assert inventory_validate({'texts':[formula]},[100,100],options({}))['texts'][0]['kind']=='formula'


def test_separate_tags_survive_validation_and_local_replacement():
    entries=[]
    for i,letter in enumerate('ABC'):
        item=row('text');item.pop('text')
        item.update(bbox_2d=[10+i*25,10,25+i*25,30],
                    runs=[{'text':letter,'script':'normal','color':'#CC0000'}],
                    replaces=['T0001'] if i==0 else [])
        entries.append(item)
    result=inventory_validate({'texts':entries},[100,100],options({}),local=True,known=['T0001'])
    assert [r['text'] for r in result['texts']]==list('ABC')
    assert len({tuple(r['bbox_2d']) for r in result['texts']})==3


def test_formula_stage_preserves_colors_and_exports_math_structure():
    original=row(); before=deepcopy(original)
    latex=r'\textcolor{#E02040}{S}_{\textcolor{#2255CC}{1}}'
    result=formula_validate({'items':[{'id':'T0001','status':'resolved','latex':latex}]},[original])
    assert original==before
    tree=latex_to_omml(result['items'][0]['latex'])
    assert len(tree.xpath('.//*[local-name()="sSub"]'))==1
    assert {'E02040','2255CC'} <= set(tree.xpath('.//*[local-name()="srgbClr"]/@val'))


@pytest.mark.parametrize('change',[
    {'bbox_2d':[1,2,3,4]}, {'id':'UNKNOWN'},
    {'status':'unreadable','latex':'S_1'},
])
def test_formula_rejects_color_loss_geometry_change_and_identity_errors(change):
    item={'id':'T0001','status':'resolved',
          'latex':r'\textcolor{#E02040}{S}_{\textcolor{#2255CC}{1}}'}
    with pytest.raises(ValueError):formula_validate({'items':[{**item,**change}]},[row()])


def test_fraction_radical_are_native_omml_not_script_runs():
    target=row();target['text']='a b';target['color']='#000000'
    target['runs']=[{'text':'a b','script':'normal','color':'#000000'}]
    result=formula_validate({'items':[{'id':'T0001','status':'resolved',
        'latex':r'\frac{a}{\sqrt{b}}'}]},[target])
    tree=latex_to_omml(result['items'][0]['latex'])
    assert tree.xpath('.//*[local-name()="f"]')
    assert tree.xpath('.//*[local-name()="rad"]')


def test_formula_duplicate_ids_fail():
    item={'id':'T0001','status':'unreadable','latex':'','reason':'clipped'}
    with pytest.raises(ValueError):formula_validate({'items':[item,item]},[row()])


def test_latest_formula_may_correct_old_colors_and_content():
    from shape_integration.texts.formula_state import adopt
    latex=r'\textcolor{#2255CC}{Q}_{\textcolor{#E02040}{2}}'
    checked=formula_validate({'items':[{'id':'T0001','status':'resolved','latex':latex}]},[row()])
    result=adopt(row(),checked['items'][0])
    assert result['text']=='Q2' and result['runs'][0]['color']=='#2255CC'
    assert result['runs'][1]['color']=='#E02040'
    assert result['formula_revision']['before']['text']=='S1'


def test_formula_failure_retains_prior_syntax_or_explicit_runs_fallback():
    from shape_integration.texts.formula_state import keep_previous
    original=row();original['latex']=r'S_1'
    result=keep_previous(original,'HTTP failure')
    assert result['latex']==original['latex'] and result['runs']==original['runs']
    result=keep_previous(row(),'no valid reply')
    assert result['kind']=='formula' and result['formula_status']=='fallback_runs'
    tree=latex_to_omml(result['latex'])
    assert tree.xpath('.//*[local-name()="sSub"]')
    assert {'E02040','2255CC'} <= set(tree.xpath('.//*[local-name()="srgbClr"]/@val'))


def test_formula_api_failure_does_not_abort_inventory(tmp_path,monkeypatch):
    from PIL import Image
    from shape_integration.common import save,sha
    from shape_integration.texts import staged_formula as staged
    from shape_integration.texts import multiline_retry
    source=tmp_path/'source.png';Image.new('RGB',(100,100),'white').save(source)
    target={**row(),'rotation_deg':0,'formula_status':'pending'}
    save(tmp_path/'stage/inventory.json',{'texts':[target], 'source_sha256':sha(source),
                                        'regions_needing_review':[]})
    monkeypatch.setattr(multiline_retry,'repair',lambda inventory,*a:deepcopy(inventory))
    def fail(*a,**kw):raise RuntimeError('HTTP 400 upstream error')
    monkeypatch.setattr(staged.model,'ask',fail)
    result=staged.recognize(source,tmp_path/'stage',None,options({}))
    assert len(result['texts'])==1
    assert result['texts'][0]['formula_status']=='fallback_runs'
    assert result['texts'][0]['bbox_2d']==target['bbox_2d']
    assert result['regions_needing_review'][0]['id']==target['id']


def test_real_render_of_explicit_runs_fallback(tmp_path):
    from shape_integration.texts.formula_state import keep_previous
    from shape_integration.texts.equations import measure
    import threading
    target=keep_previous(row(),'no valid model reply')
    metrics=measure([target],tmp_path,threading.BoundedSemaphore(1))
    assert metrics['T0001']['width']>0 and metrics['T0001']['height']>0


def test_inventory_prompt_does_not_request_formula_syntax():
    prompt=inventory_prompt([100,100],options({}))
    assert 'latex' not in prompt['example']['texts'][0]
    assert not any('complete structural LaTeX' in s for s in prompt['rules'])


def test_ppt_keeps_multicolor_runs_and_native_math(tmp_path):
    from shape_integration.texts.export import write
    text={**row('text'),'rotation_deg':0,'bold':False,'italic':False,
          'font_size_px':20,'frame_px':[10,10,60,30], 'resolved_font':'Times New Roman'}
    formula={**row(),'id':'T0002','rotation_deg':0,'bold':False,'italic':False,
        'font_size_px':20,'frame_px':[10,50,60,30],'resolved_font':'Times New Roman',
        'latex':r'\frac{\textcolor{#E02040}{a}}{\sqrt{\textcolor{#2255CC}{b}}}'}
    from shape_integration.texts.equations import measure
    import threading
    metric=measure([formula],tmp_path/'metrics',threading.BoundedSemaphore(1))['T0002']
    formula['layout']={'measurement':metric}
    write([text,formula],[100,100],tmp_path/'sample.pptx')
    with ZipFile(tmp_path/'sample.pptx') as archive:
        tree=etree.fromstring(archive.read('ppt/slides/slide1.xml'))
    assert tree.xpath('.//*[local-name()="oMath"]')
    assert tree.xpath('.//*[local-name()="rad"]')
    shapes=tree.xpath('.//*[local-name()="sp"]')
    assert {'E02040','2255CC'} <= set(shapes[0].xpath('.//*[local-name()="srgbClr"]/@val'))
