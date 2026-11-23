from copy import deepcopy
from PIL import Image,ImageDraw
import pytest
from shape_integration.texts.ocr_multiline import match_lines,propose_lines,regroup_fitted,conflict_parents
from shape_integration.texts.contracts import recognition
from shape_integration.texts.bbox_color_review import retain_nonregressing


def row(**kw):
    return dict({'id':'T1','text':'Alpha Beta','kind':'text','visual_line_count':2,
        'bbox_2d':[10,10,80,60],'rotation_deg':0,'font':'Arial','color':'#000000',
        'runs':[{'text':'Alpha ','script':'normal','color':'#FF0000'},
                {'text':'Beta','script':'normal','color':'#0000FF'}]},**kw)


def detections():
    return [{'index':0,'text':'Alpha','confidence':.99,'rotation':0,'source_bbox':[12,12,70,28]},
            {'index':1,'text':'Beta','confidence':.99,'rotation':0,'source_bbox':[15,36,65,55]}]


def test_contract_preserves_line_count_and_pending_status():
    parsed=recognition({'texts':[row()]},[200,200])['texts'][0]
    assert parsed['visual_line_count']==2 and parsed['line_layout_status']=='requires_split'
    for count in (0,True,2.5,17):
        with pytest.raises(ValueError):recognition({'texts':[row(visual_line_count=count)]},[200,200])


def test_complete_lines_only_and_formulas_excluded():
    assert match_lines(row(),detections(),[200,200])[0]
    assert match_lines(row(visual_line_count=1),detections(),[200,200])[0]
    for values in (detections()[:1],[detections()[0],{**detections()[1],'text':'Be'}],
                   [detections()[0],{**detections()[1],'source_bbox':[140,140,190,160]}],
                   [detections()[0],{**detections()[1],'source_bbox':[100,12,150,28]}]):
        assert match_lines(row(),values,[200,200])[0] is None
    assert match_lines(row(kind='formula'),detections(),[200,200])[0] is None
    assert match_lines(row(),detections(),[200,200],used={1})[0] is None


def test_approximate_lines_keep_vlm_text_and_require_all_lines():
    parent=row(text='SS Modeling',runs=[{'text':'SS Modeling','script':'normal','color':'#000000'}])
    values=[{**detections()[0],'text':'ss'}, {**detections()[1],'text':'Modelin'}]
    parts,audit=match_lines(parent,values,[200,200])
    assert [parent['text'][p['span'][0]:p['span'][1]] for p in parts]==['SS','Modeling']
    assert [p['association']['mode'] for p in parts]==['casefold','approximate']
    assert match_lines(parent,values[:1],[200,200])[0] is None
    assert match_lines(parent,values+[dict(values[1],index=2,source_bbox=[15,57,65,74])],[200,200])[0] is None


def test_inexact_empty_source_never_uses_raw_partial_bbox():
    parent=row(text='Alpha Modeling',runs=[{'text':'Alpha Modeling','script':'normal','color':'#000000'}])
    values=[detections()[0],dict(detections()[1],text='Modelin')]
    children,audit=propose_lines(parent,values,Image.new('RGB',(200,200),'white'))
    assert children is None and audit['status']=='inexact_ocr_support_unresolved'


def test_ambiguous_repeated_occurrences_deferred():
    values=detections()+[{**detections()[1],'index':2,'source_bbox':[15,57,65,74]}]
    assert match_lines(row(),values,[200,200])[1]['status']=='ambiguous_ocr_line_assignment'


def test_children_preserve_styles_and_grouped_rollback():
    parent=row();original=deepcopy(parent)
    image=Image.new('RGB',(200,200),'white');draw=ImageDraw.Draw(image)
    draw.rectangle((20,16,60,24),fill='red');draw.rectangle((20,40,60,50),fill='blue')
    children,audit=propose_lines(parent,detections(),image)
    assert [c['text'] for c in children]==['Alpha','Beta']
    assert [c['runs'][0]['color'] for c in children]==['#FF0000','#0000FF']
    assert all(c['parent_id']=='T1' and c['visual_line_count']==1 for c in children)
    assert parent==original
    updates={'T1':children}
    assert regroup_fitted(updates,list(reversed(children)))==updates
    assert conflict_parents(updates,[['T1.L02','T9']])=={'T1'}
    def evaluate(parts):
        return {i:{'before':{'score':.5},'after':{'score':.4}} for i in parts}
    accepted,history=retain_nonregressing(updates,evaluate,.02)
    assert accepted=={} and history[0]['rejected_ids']==['T1']
