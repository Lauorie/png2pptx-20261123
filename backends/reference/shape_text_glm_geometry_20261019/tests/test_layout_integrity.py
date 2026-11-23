from copy import deepcopy
import pytest

from shape_integration.texts.layout_integrity import attribute_layout, reconcile, stacked_runs
from shape_integration.texts.two_stage_crop import validate_attributes, attribute_prompt
from test_two_stage_crop import original


def row(ident,box,text='same'):
    return {**original(),'id':ident,'text':text,'bbox_2d':box,
            'runs':[{'text':text,'script':'normal','color':'#000000'}]}


def test_new_collisions_rollback_whole_parents_without_deletion():
    a=row('A',[0,0,20,10]);b=row('B',[0,20,20,30]);before=deepcopy([a,b])
    updates={'A':[{**a,'bbox_2d':[0,18,20,28]}],'B':[{**b}]}
    result,warnings=reconcile([a,b],updates)
    assert result=={} and {r['id'] for r in warnings}=={'A','B'}
    assert [a,b]==before and len(updates)==2


def test_read_only_neighbor_cannot_be_claimed_by_another_target():
    a=row('A',[0,0,20,10]);b=row('B',[0,20,20,30])
    result,warnings=reconcile([a,b],{'A':[{**a,'bbox_2d':b['bbox_2d']}]})
    assert result=={} and warnings[0]['id']=='A'


def test_preexisting_formula_overlap_and_identical_words_remain():
    a=row('A',[0,0,20,20]);b=row('B',[10,0,30,20])
    updates={'A':[a],'B':[b]}
    assert reconcile([a,b],updates)==(updates,[])
    b['bbox_2d']=[40,0,60,20]
    assert reconcile([a,b],{'A':[a],'B':[b]})==({'A':[a],'B':[b]},[])


def test_overlapping_new_child_lines_rollback_parent():
    a=row('A',[0,0,100,100])
    children=[{**a,'id':'A.G01','bbox_2d':[0,0,50,20]},
              {**a,'id':'A.G02','bbox_2d':[0,10,50,30]}]
    assert reconcile([a],{'A':children})[0]=={}


def test_rotated_multiline_is_not_forced_to_one_line():
    source=row('A',[0,0,20,90],'TRANSPOSED CONVOLUTION Stride = 2')
    unit={**source,'rotation_deg':-90,'layout_mode':'multiline',
          'visual_line_count':2,'direction_status':'confirmed','reason':'two parallel baselines'}
    locked={'A':{'bbox_2d':[0,0,20,90]}}
    got=validate_attributes({'items':[{'id':'A','status':'resolved','units':[unit]}]},[source],locked,[100,100])['items'][0]
    assert got['visual_line_count']==2 and got['layout_mode']=='multiline'
    assert got['rotation_deg']==-90 and got['text']==source['text']


def test_confirmed_multiline_can_repair_missing_join_separator():
    from shape_integration.texts.multiline_retry import validate
    a={**row('A',[0,0,100,90],'TRANSPOSED CONVOLUTIONStride = 2'),
       'visual_line_count':2,'layout_mode':'multiline','rotation_deg':90}
    raw={'parent_id':'A','lines':[
        {'text':'TRANSPOSED CONVOLUTION','bbox_2d':[10,10,20,80]},
        {'text':'Stride = 2','bbox_2d':[30,10,40,80]}]}
    checked=validate(raw,a,[100,100]);assert len(checked['lines'])==2
    assert ''.join(a['text'][lo:hi] for lo,hi in (r['span'] for r in checked['lines']))==a['text']
    a.pop('layout_mode')
    with pytest.raises(ValueError,match='split a word'):validate(raw,a,[100,100])


def test_stacked_glyphs_are_not_rotation_or_word_splitting():
    a=row('A',[0,0,20,150],'MODULE 1');before=deepcopy(a)
    glyphs=stacked_runs(a)
    assert ''.join(r[0]['text'] for r in glyphs)=='MODULE1' and len(glyphs)==7
    assert a==before
    assert attribute_layout({'layout_mode':'stacked_vertical','rotation_deg':0},'text')==('stacked_vertical',1)
    with pytest.raises(ValueError):attribute_layout({'layout_mode':'stacked_vertical','rotation_deg':90},'text')
    accented=row('A',[0,0,20,150],'A\u0301 B')
    assert len(stacked_runs(accented))==2


def test_stacked_label_native_export_and_measurement(tmp_path):
    from shape_integration.texts.layout import pillow_metrics, fit
    from shape_integration.texts.export import write
    from pptx import Presentation
    a={**row('A',[10,10,30,200],'MODULE 1'),'layout_mode':'stacked_vertical'}
    measurement=pillow_metrics(a);horizontal=pillow_metrics({**a,'layout_mode':'single_line'})
    assert measurement['height']>horizontal['height']*5
    fitted=fit(a,measurement)
    assert fitted['font_size_px']>15
    path=tmp_path/'stacked.pptx';write([fitted],[200,220],path)
    shape=Presentation(path).slides[0].shapes[0]
    assert shape.rotation==0 and shape.text=='M\vO\vD\vU\vL\vE\v1'
    assert {r.font.size for r in shape.text_frame.paragraphs[0].runs}=={shape.text_frame.paragraphs[0].font.size}


def test_new_schema_explicitly_exposes_layout_not_background():
    prompt=attribute_prompt([100,100],[],[])
    assert prompt['response_format']['items'][0]['units'][0]['layout_mode']=='single_line/multiline/stacked_vertical'
    assert any('ENTIRE original target' in s for s in prompt['rules'])


def test_rejected_attributes_keep_entire_original_geometry(tmp_path,monkeypatch):
    from PIL import Image
    from shape_integration.common import save,read
    from shape_integration.texts import two_stage_crop as pipeline, options
    source=tmp_path/'source.png';Image.new('RGB',(100,100),'white').save(source)
    old={**original(),'font_size_px':20,'resolved_font':'Arial','frame_px':[1,1,20,30]}
    baseline=tmp_path/'baseline';save(baseline/'scene.json',{'texts':[old]})
    Image.new('RGB',(100,100),'white').save(baseline/'native.png')
    output=tmp_path/'result';output.mkdir()
    class Runtime:
        def for_stage(self,**kwargs):return self
        def call(self,folder,tag,system,prompt,images,validator,budget):
            if prompt['stage']=='1_bbox_only':
                raw={'id':old['id'],'status':'resolved','units':[{'bbox_2d':[300,300,500,500]}]}
            else:raw={'id':old['id'],'status':'ambiguous','units':[],'reason':'Box covers only a connector'}
            return validator({'items':[raw]})
    monkeypatch.setattr(pipeline.export,'final',lambda rows,source,folder,*args:
        Image.new('RGB',(100,100),'white').save(folder/'native.png'))
    summary=pipeline.finish(source,output,baseline,options({'max_tokens':1000}),Runtime())
    selected=read(output/'final/scene.json')['texts'][0]
    for key in ('bbox_2d','font_size_px','rotation_deg','text','runs'):
        assert selected[key]==old[key]
    assert summary['accepted_ids']==[] and summary['unresolved_ids']==[old['id']]
