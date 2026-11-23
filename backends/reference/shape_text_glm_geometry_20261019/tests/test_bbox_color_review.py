from copy import deepcopy

import pytest
from PIL import Image

from shape_integration.texts import bbox_color_review as review


FRAME={'size':[300,200],'source_size':[300,200],'box':[0,0,300,200],'scale':1,'offset':[0,0]}


def row(ident='T1'):
    return {'id':ident,'text':'target','bbox_2d':[30,40,100,60],'font':'Arial',
            'bold':False,'italic':True,'underline':False,'color':'#AAAAAA','rotation_deg':0,
            'kind':'text','latex':'','runs':[{'text':'target','script':'normal','color':'#AAAAAA'}]}


def diagnosis(causes=(1,2),status='located'):
    return {'id':'T1','causes':list(causes),
            'text_checks':{key:{'status':('mismatch' if 2 in causes else 'uncertain' if not causes else 'match') if key=='size' else 'match',
                                'evidence':'SOURCE and PPT glyph extents differ.' if key=='size' and 2 in causes else 'Compared SOURCE and PPT glyph property.'} for key in review.TEXT_CHECKS},
            'background_elements':['A gray diagonal arrow crosses the word.'] if 1 in causes else [],
            'text_issues':['The predicted right boundary clips the last two letters.'] if 2 in causes else [],
            'localization_status':status,
            'localization_evidence':'Source first and last glyph extents and ascenders/descenders are visible.' if status=='located' else 'Source glyph boundaries are obscured by crossing graphics.',
            'bbox_2d':[25,38,120,62] if status=='located' else None,
            'color':'#B0B0B0' if status=='located' else None}


def test_cpu_above_80_bypasses_vlm_even_with_weak_registration(monkeypatch):
    scores={'T1':.800001,'T2':.80,'T3':.79}
    monkeypatch.setattr(review,'evidence',lambda source,before,after,rows:{'before':{'score':scores[rows[0]['id']]}})
    rows=[row(i) for i in scores];im=Image.new('RGB',(300,200))
    groups,selection=review.select(rows,im,im)
    assert selection['automatic_ids']==['T1']
    assert {r['id'] for g in groups for r in g}=={'T2','T3'}


def test_background_only_still_generates_source_bbox_proposal():
    original=row();saved=deepcopy(original)
    raw={'items':[diagnosis((1,))]}
    assert review.validate(raw,[original],FRAME)==raw
    assert review.proposals(raw,[original],FRAME)['T1'][0]['bbox_2d']==[25,38,120,62]
    assert original==saved


def test_bare_list_is_normalized_without_weakening_item_validation():
    items=[diagnosis((1,))]
    assert review.validate(items,[row()],FRAME)=={'items':items}
    assert 'T1' in review.proposals(items,[row()],FRAME)
    with pytest.raises(ValueError):
        review.validate([diagnosis((1,),'keep')],[row()],FRAME)
    with pytest.raises(ValueError):
        review.validate(items+items,[row()],FRAME)


@pytest.mark.parametrize('causes',[(2,),(1,2)])
def test_text_error_repairs_only_geometry_and_color(causes):
    original=row();original['layout']={'stale':True};original['font_size_px']=10
    result=review.proposals({'items':[diagnosis(causes)]},[original],FRAME)['T1'][0]
    for field in ('text','font','bold','italic','rotation_deg','kind','latex'):
        assert result[field]==original[field]
    assert result['runs'][0]['text']==original['runs'][0]['text']
    assert result['bbox_2d']==[25,38,120,62] and result['color']=='#B0B0B0'
    assert 'layout' not in result and 'font_size_px' not in result


@pytest.mark.parametrize('item',[
    diagnosis((1,),'keep'),diagnosis((2,),'repair'),
    {**diagnosis(),'background_elements':[]},
    {**diagnosis(),'text_issues':[]},
    {**diagnosis(),'font':'Calibri'},
    {**diagnosis((1,),'uncertain'),'bbox_2d':[25,38,120,62]},
    {**diagnosis(),'bbox_2d':None},
    {**diagnosis(),'localization_evidence':''},
    {**diagnosis(),'bbox_2d':[-1,1,100,20]},
])
def test_inconsistent_diagnosis_or_unrequested_changes_rejected(item):
    with pytest.raises(ValueError):review.validate({'items':[item]},[row()],FRAME)


def test_mixed_run_colors_preserved_or_updated_explicitly():
    original=row();original['runs']=[{'text':'tar','script':'normal','color':'#AAAAAA'},
        {'text':'get','script':'normal','color':'#FF0000'}]
    raw={'items':[diagnosis()]}
    result=review.proposals(raw,[original],FRAME)['T1'][0]
    assert [r['color'] for r in result['runs']]==['#B0B0B0','#FF0000']
    raw['items'][0]['color']=['#000000','#0000FF']
    result=review.proposals(raw,[original],FRAME)['T1'][0]
    assert [r['color'] for r in result['runs']]==['#000000','#0000FF']


def test_unlocalizable_source_is_preserved():
    raw={'items':[diagnosis((),'uncertain')]}
    assert review.validate(raw,[row()],FRAME)==raw
    assert review.proposals(raw,[row()],FRAME)=={}


def test_prompt_explains_scores_causes_and_concrete_evidence():
    value=review.prompt([300,200],[{'id':'T1','cpu_similarity_score_0_to_100':25.06}],[])
    assert value['targets'][0]['cpu_similarity_score_0_to_100']==25.06
    rules=' '.join(value['rules'])
    assert 'NOT a whole-crop score' in rules
    assert 'causes=[1]' in rules and 'causes=[2]' in rules and 'causes=[1,2]' in rules
    assert 'background_elements' in rules and 'text_issues' in rules
    assert value['protocol']=='source_ink_bbox_color.5'
    assert 'There is no keep decision' in rules
    assert 'no intentional outer padding' in rules
    assert 'Each of the four edges must touch' in rules
    assert 'Crop padding is context only' in rules
    assert 'internal word spaces, character holes and empty corners are normal' in rules


@pytest.mark.parametrize('status',['match','uncertain'])
def test_similarity_judgment_does_not_suppress_source_localization(status):
    item=diagnosis((1,));item['text_checks']['size']['status']=status
    assert 'T1' in review.proposals({'items':[item]},[row()],FRAME)


def test_localization_does_not_change_unsupported_properties():
    item=diagnosis((2,));item['text_checks']['orientation']['status']='mismatch'
    result=review.proposals({'items':[item]},[row()],FRAME)['T1'][0]
    assert result['rotation_deg']==row()['rotation_deg']
    assert result['font']==row()['font']
    assert result['text']==row()['text']


def test_tight_bbox_can_expand_to_recover_clipped_glyphs():
    item=diagnosis((1,));item['bbox_2d']=[20,35,130,65]
    result=review.proposals({'items':[item]},[row()],FRAME)['T1'][0]
    assert result['bbox_2d']==[20,35,130,65]


def test_missing_comparative_evidence_and_cause_inconsistency_are_rejected():
    item=diagnosis();item['text_checks']['color']['evidence']=' '
    with pytest.raises(ValueError):review.validate({'items':[item]},[row()],FRAME)
    item=diagnosis();item['text_checks']['size']['status']='match'
    with pytest.raises(ValueError):review.validate({'items':[item]},[row()],FRAME)


@pytest.mark.parametrize('before,after,accepted',[(.2,.3,True),(.2,.2,True),(.9,.85,False),(.2,.199999,False),(.2,float('nan'),False),(float('inf'),.9,False)])
def test_explicit_strict_post_render_score_must_not_decrease(before,after,accepted):
    assert review.nonregression_gate({'before':{'score':before},'after':{'score':after}},0.)[0] is accepted


@pytest.mark.parametrize('before,after,accepted',[
    (.80,.75,True),(.60,.55,True),(.60,.551,True),(.60,.549999,False),
    (.20,.16,True),(.20,.149,False),(.20,.30,True),
    (.00,.00,False),(.60,float('nan'),False)])
def test_default_accepts_at_most_five_absolute_points(before,after,accepted):
    assert review.DEFAULT_MAX_SCORE_DROP==.05
    assert review.nonregression_gate({'before':{'score':before},'after':{'score':after}})[0] is accepted


def test_rollbacks_are_rerendered_and_rechecked_until_stable():
    calls=[]
    def evaluate(updates):
        calls.append(set(updates))
        return {ident:{'before':{'score':.5},'after':{'score':.4 if ident=='A' or (ident=='B' and 'A' not in updates) else .6}} for ident in updates}
    original={'A':[row('A')],'B':[row('B')],'C':[row('C')]}
    accepted,history=review.retain_nonregressing(original,evaluate)
    assert set(accepted)=={'C'} and set(original)=={'A','B','C'}
    assert calls==[{'A','B','C'},{'B','C'},{'C'}]
    assert [r['rejected_ids'] for r in history]==[['A'],['B'],[]]


def test_all_rejected_renders_exact_baseline():
    calls=[]
    def evaluate(updates):
        calls.append(set(updates))
        return {ident:{'before':{'score':.5},'after':{'score':.4}} for ident in updates}
    accepted,history=review.retain_nonregressing({'A':[row()]},evaluate)
    assert accepted=={} and calls==[{'A'},set()]
