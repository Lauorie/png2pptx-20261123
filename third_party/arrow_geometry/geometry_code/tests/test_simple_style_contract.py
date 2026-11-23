import copy,json,math,threading,zipfile
from pathlib import Path
import pytest
from PIL import Image
from simple_style_contract import Contract,merge_style
import style_pipeline

def row():
    return dict(id='a',kind='straight',start=[10.,20.],via=[],end=[90.,20.],beziers=[],heads=['none','triangle'],geometry_heads_reported=True,
        style=dict(color='#666666',width_px=2.,dash=dict(type='solid',preset='solid'),start=None,end=dict(type='triangle',width='lg',length='sm'),state='geometry_preview'))

def style(**extra):
    return dict(id='a',color='#112233',width_px=.25,dash_px=[],start='circle',end='none',**extra)

def test_source_pixel_floor_and_head_reclassification():
    r=row();before=copy.deepcopy(r);view={'view_to_global':[[.25,0,0],[0,.25,0],[0,0,1]]};c=Contract(view)
    answer={'styles':[style()]};raw=copy.deepcopy(answer);c.validate(answer,[r])
    actual=style_pipeline.apply_styles([r],answer,c)[0]
    assert actual['style']['width_px']==8. and actual['style']['policy']['widths'][0]['effective_source_px']==2.
    assert actual['heads']==['oval','none'] and actual['style']['start']==dict(type='oval',width='med',length='med')
    assert answer==raw and r==before and c.geometry_signature([r])==c.geometry_signature([actual])

def test_limits_apply_to_native_export(tmp_path):
    from lib.native import write
    r=Contract().enforce_rows([row()])[0];write((100,40),[r],[r['style']],tmp_path/'x.pptx')
    import xml.etree.ElementTree as ET
    with zipfile.ZipFile(tmp_path/'x.pptx') as z:root=ET.fromstring(z.read('ppt/slides/slide1.xml'))
    heads=root.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd')
    assert heads and heads[0].attrib=={'type':'triangle','w':'med','len':'med'}

def test_2048px_canvas_keeps_one_pixel_stroke_with_bounded_renderer_error():
    from native_scale import plan
    r=Contract().enforce_rows([row()])[0];r['style']['width_px']=1.
    assert not plan((2048,1000),[r],[r['style']])['safe']
    allowed=plan((2048,1000),[r],[r['style']],head_floor_tolerance=.02)
    assert allowed['safe'] and 0<allowed['permitted_renderer_head_inflation']<.02
    assert max(allowed['required_inches'])<=56 and r['style']['width_px']==1.
    assert not plan((4096,1000),[r],[r['style']],head_floor_tolerance=.02)['safe']

def test_outline_floor_does_not_add_missing_shape_stroke():
    r=row();r.update(kind='block_arrow',shape_parts=[dict(id='p',commands=[['M',0,0],['L',20,0],['L',20,20],['Z']])])
    c=Contract();v=dict(id='a',parts=[dict(id='p',fill='#AAAAAA',stroke=None,width_px=0,dash='solid')])
    s=c.validate_style(v,r);assert s['shape_parts']['p']['stroke'] is None and s['shape_parts']['p']['width_px']==0
    v['parts'][0].update(stroke='#000000',width_px=.1)
    assert c.validate_style(v,r)['shape_parts']['p']['width_px']==1.

def test_bad_numbers_and_extra_head_types_are_rejected():
    for value in [math.nan,math.inf,'1']:
        v=style();v['width_px']=value
        with pytest.raises(ValueError):Contract().validate_style(v,row())
    v=style();v['end']={'type':'triangle','width':'sm','length':'lg'}
    with pytest.raises(ValueError):Contract().validate_style(v,row())
    with pytest.raises(ValueError):Contract().checks({'checks':[dict(id='a',verdict='accept',geometry_action='mask')]},[row()])

def test_commit_preserves_geometry_and_legacy_contract():
    import style_contract
    original=row();updated=copy.deepcopy(original);updated['start'][0]+=1.e-12
    updated['style']=Contract().validate_style(style(),original);updated['heads']=['oval','none']
    merged=merge_style(original,updated)
    assert merged['start']==original['start'] and merged['heads']==['oval','none']
    legacy=dict(id='a',color='#112233',width_px=2,dash_px=[],start=None,end={'type':'triangle','width':'lg','length':'sm'})
    assert style_contract.validate_style(legacy,original)['end']['width']=='lg'

def test_style_estimate_critic_use_same_policy_without_regeneration(tmp_path,monkeypatch):
    class Model:
        def __init__(self):self.calls=[]
        def ask(self,folder,q,images,validator):
            self.calls.append((q,images))
            return validator({'checks':[dict(id='a',verdict='accept',reason='')]} if 'checks' in q['output_schema'] else {'styles':[style()]})
    renders=[]
    def render(rows,out,size,view,locks):
        out=Path(out);out.mkdir(parents=True,exist_ok=True);Image.new('RGB',tuple(view['size']),'white').save(out/'render.png');renders.append(rows);return out/'render.png'
    monkeypatch.setattr(style_pipeline,'render',render)
    model=Model();r=row();before=copy.deepcopy(r)
    rows,issues=style_pipeline.recover(Image.new('RGB',(100,40),'white'),[r],{},tmp_path,dict(style_profile='three_heads_mid_min2',style_refine_rounds=1),model,{'cpu':threading.Semaphore(1)})
    assert not issues and len(model.calls)==2 and len(renders)==1
    assert rows[0]['heads']==['oval','none'] and rows[0]['style']['width_px']==2.
    assert all(q['protocol']=='style_three_heads_mid_min2.0' for q,_ in model.calls)
    assert r==before and (tmp_path/'final/render.png').exists()
