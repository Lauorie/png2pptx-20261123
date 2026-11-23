from copy import deepcopy
import pytest

from shape_integration.texts.response_validation import numeric_box, retry_invalid
from shape_integration.texts.two_stage_crop import validate_geometry, validate_attributes
from shape_integration.texts.typography_audit import audit
from shape_integration.texts.multiline_retry import fallback_lines


def item(ident, box=None):
    return {'id':ident,'status':'resolved','units':[{'bbox_2d':box or [1,2,20,30]}]}


@pytest.mark.parametrize('box',[["1","2","20","3e1"],[1,2.,20,30]])
def test_numeric_wire_coordinates(box):
    result=validate_geometry({'items':[item('T1',box)]},['T1'],[100,100])
    assert result['items'][0]['units'][0]['bbox_2d']==[1,2,20,30]


@pytest.mark.parametrize('value',[True,'NaN','Infinity','1+2','xmin',float('nan'),float('inf')])
def test_non_numeric_or_non_finite_never_repaired(value):
    with pytest.raises(ValueError):numeric_box([value,2,20,30])


def test_bad_neighbor_missing_duplicate_unknown_are_isolated():
    raw={'items':[item('T1'),item('T2',[40,1,20,30]),item('T3'),item('T3'),item('UNKNOWN')]}
    result=validate_geometry(raw,['T1','T2','T3','T4'],[100,100],partial=True)
    assert [r['status'] for r in result['items']]==['resolved','invalid','invalid','invalid']
    assert result['warnings'] and len(result['items'])==4
    assert raw['items'][0]['units'][0]['bbox_2d']==[1,2,20,30]


def test_only_bad_ids_retried_and_good_result_cannot_be_overwritten():
    first=validate_geometry({'items':[item('T1'),item('T2',[99,1,100,30])]},['T1','T2'],[100,100],partial=True)
    prompt={'targets':[{'id':'T1'},{'id':'T2'}]};before=deepcopy(prompt);calls=[]
    def call(retry,pending):
        calls.append((retry,pending))
        return validate_geometry({'items':[item('T2'),item('T1',[1,1,2,2])]},pending,[100,100],partial=True)
    result=retry_invalid(first,prompt,call)
    assert len(calls)==1 and calls[0][0]['targets']==[{'id':'T2'}]
    assert result['items'][0]==first['items'][0]
    assert result['items'][1]['status']=='resolved' and prompt==before


def test_failed_repair_retains_good_observations():
    first=validate_geometry({'items':[item('T1')]},['T1','T2'],[100,100],partial=True)
    def call(*args):raise RuntimeError('timeout')
    result=retry_invalid(first,{'targets':[{'id':'T1'},{'id':'T2'}]},call)
    assert result['items']==first['items'] and result['retry_error']=='timeout'


def test_multiline_attributes_are_preserved_for_explicit_line_repair():
    from test_two_stage_crop import original
    a=original();b={**original(),'id':'T0002'}
    unit={**a,'direction_status':'confirmed','visual_line_count':1,'reason':'Visible letter'}
    second={**unit,'visual_line_count':2}
    locked={r['id']:{'bbox_2d':[2,3,15,25],'background_fill':None} for r in (a,b)}
    result=validate_attributes({'items':[
        {'id':a['id'],'status':'resolved','units':[unit]},
        {'id':b['id'],'status':'resolved','units':[second]}]},[a,b],locked,[100,100],partial=True)
    assert result['items'][0]['status']=='located'
    assert result['items'][1]['status']=='located'
    assert result['items'][1]['visual_line_count']==2
    assert result['items'][1]['layout_mode']=='multiline'


def test_multiline_fallback_is_render_only_and_keeps_inline_styles():
    row={'text':'Hello brave World','visual_line_count':2,'runs':[
        {'text':'Hello ','script':'normal','color':'#CC0000'},
        {'text':'brave World','script':'normal','color':'#0000CC'}]}
    before=deepcopy(row);lines=fallback_lines(row)
    assert len(lines)==2 and row==before
    assert ' '.join(''.join(r['text'] for r in line) for line in lines)==row['text']
    assert lines[0][0]['color']=='#CC0000' and lines[-1][-1]['color']=='#0000CC'


def test_typography_audit_only_flags_conservative_outliers_never_restyles():
    rows=[{'id':str(i),'text':str(128+i),'kind':'text','font':'Arial','rotation_deg':0,
           'bbox_2d':[i*50,0,i*50+35,20],'font_size_px':32 if i==3 else 20}
          for i in range(4)]
    before=deepcopy(rows)
    assert [r['id'] for r in audit(rows)]==['3'] and rows==before
    assert audit([{**r,'kind':'formula'} for r in rows])==[]


def test_export_fallback_has_explicit_breaks_and_original_frame(tmp_path):
    from test_two_stage_crop import original
    from shape_integration.texts.export import write
    from pptx import Presentation
    row={**original(),'text':'Hello World','visual_line_count':2,
        'runs':[{'text':'Hello World','script':'normal','color':'#000000'}],
        'bbox_2d':[10,10,110,90],'frame_px':[10,10,200,20],
        'font_size_px':20,'resolved_font':'Arial','multiline_render_fallback':True}
    path=tmp_path/'test.pptx';record=write([row],[200,120],path)
    shape=Presentation(path).slides[0].shapes[0]
    assert shape.text=='Hello\vWorld'
    assert shape.text_frame.word_wrap is True
    assert record['unit_mapping'][0]['multiline_render_fallback'] is True


def test_finish_isolates_repairs_and_splits_remaining_multiline(tmp_path,monkeypatch):
    from PIL import Image
    from test_two_stage_crop import original
    from shape_integration.common import save,read
    from shape_integration.texts import two_stage_crop as pipeline, multiline_retry, options
    source=tmp_path/'source.png';Image.new('RGB',(100,100),'white').save(source)
    a={**original(),'visual_line_count':1}
    b={**original(),'id':'T0002','text':'Hello World','visual_line_count':2,
       'runs':[{'text':'Hello World','script':'normal','color':'#000000'}]}
    baseline=tmp_path/'baseline';save(baseline/'scene.json',{'texts':[a,b]})
    Image.new('RGB',(100,100),'white').save(baseline/'native.png')
    output=tmp_path/'result';output.mkdir();calls=[]
    monkeypatch.setattr(pipeline,'compact_groups',lambda *args:[[a,b]])
    class Runtime:
        def for_stage(self,**kwargs):return self
        def call(self,folder,tag,system,prompt,images,validator,budget):
            targets=[r['id'] for r in prompt['targets']];calls.append((str(folder),targets))
            if prompt['stage']=='1_bbox_only':
                raw=[item(i,['100','100','200','300']) for i in targets]
                if 'repair' not in folder.name:raw=[raw[0]]  # missing b only
            else:
                raw=[]
                for ident in targets:
                    row=a if ident==a['id'] else b
                    unit={**row,'direction_status':'confirmed','reason':'Visible text'}
                    raw.append({'id':ident,'status':'resolved','units':[unit]})
            return validator({'items':raw})
    def ask(runtime,folder,prompt,images,validator,opt):
        w,h=prompt['image_size']
        return validator({'parent_id':b['id'],'lines':[
            {'text':'Hello','bbox_2d':[w*.1,h*.1,w*.5,h*.3]},
            {'text':'World','bbox_2d':[w*.1,h*.4,w*.5,h*.6]}]})
    monkeypatch.setattr(multiline_retry.model,'ask',ask)
    def fit(rows,*args,**kwargs):
        return [{**r,'font_size_px':10,'resolved_font':'Arial','frame_px':[1,1,20,20]} for r in rows],[]
    monkeypatch.setattr(pipeline.layout,'fit_all',fit)
    monkeypatch.setattr(pipeline.export,'final',lambda rows,source,folder,*args:
        Image.new('RGB',(100,100),'white').save(folder/'native.png'))
    summary=pipeline.finish(source,output,baseline,options({'max_tokens':1000}),Runtime())
    result=read(output/'final/scene.json')['texts']
    assert [r['text'] for r in result]==['a','Hello','World']
    assert summary['unresolved_multiline_ids']==[]
    assert all(targets==[b['id']] for folder,targets in calls if 'repair' in folder.rsplit('/',1)[-1])
    assert any(folder.endswith('/geometry_repair_calls') for folder,targets in calls)
    assert summary['cpu_acceptance_gate'] is False
