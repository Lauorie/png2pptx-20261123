import copy
import numpy as np
import pytest
from PIL import Image,ImageDraw
from pptx import Presentation
from full_recovery.quality_choice import candidates,nominate
from full_recovery.shape_duplicates import duplicates
from full_recovery.image_edit import transparent_exterior,replace_objects
from full_recovery.text_bounds import clamp_small_text_overflow
from shape_integration.styling.controller import commit_refinement
from shape_integration.common import sha


def test_transparency_removes_neutral_paper_grain_but_keeps_enclosed_white():
    im=Image.new('RGB',(64,64),(248,247,249));draw=ImageDraw.Draw(im)
    draw.line((0,63,63,63),fill=(223,225,227),width=2)
    im.putpixel((0,0),(217,221,220))
    draw.rectangle((16,16,47,47),fill='#7597FF')
    draw.rectangle((24,24,39,39),fill='white')
    result=transparent_exterior(im)
    assert result.getchannel('A').getbbox()==(16,16,48,48)
    assert result.getpixel((30,30))==(255,255,255,255)


def test_shape_stage_effort_is_forwarded_without_changing_other_stages(tmp_path,monkeypatch):
    import importlib.util,threading
    from pathlib import Path
    from types import SimpleNamespace
    root=Path(__file__).resolve().parents[1]/'experiment/shape_sam_guided'
    monkeypatch.syspath_prepend(str(root))
    spec=importlib.util.spec_from_file_location('stage_runtime_fixture',root/'runtime.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sent=[]
    def post(url,body,**kwargs):
        sent.append(body)
        return {'choices':[{'finish_reason':'stop','message':{'content':'{"decision":"accept"}'}}]}
    models=SimpleNamespace(config={'seed':0,'glm_max_tokens':32000,'glm_url':'http://unused',
                                  'reasoning_effort':'high'},slots=threading.Semaphore(1),post=post)
    runtime=module.ForwardRuntime(models,{'shape_schema_retries':0},None)
    checker=runtime.for_stage(reasoning_effort='low')
    checker.call(tmp_path/'critic','check','visual check',{},[],lambda x:x,16384)
    runtime.call(tmp_path/'observer','observe','observe',{},[],lambda x:x,32000)
    assert [r['chat_template_kwargs']['reasoning_effort'] for r in sent]==['low','high']
    assert models.config['reasoning_effort']=='high'


def test_original_image_boundary_does_not_schedule_full_context_reconstruction():
    from shape_integration.fitting.control import needs_more_source
    job={'frame':{'actual_crop_input_size_px':[1000,400],
                  'source_crop_half_open':[300,0,1300,400],'original_image_size_px':[1300,400]}}
    assert not needs_more_source([50,0,999,399],job)
    assert needs_more_source([0,20,900,370],job)


def test_all_native_arrow_head_vocabularies_enter_branch_selection():
    scene={'parts':{'line':{'id':'line','object_id':'n','kind':'path','status':'ok',
        'commands':[['M',10,20],['L',100,20]],'bbox_2d':[10,20,100,20]}}}
    styles={'line':{'fill':None,'stroke':'#000000','width_px':1,'dash':'solid'}}
    for head in ['arrow','unknown','stealth','diamond','open','triangle','oval','none']:
        arrow={'id':'a','start':[10,20],'end':[100,20],'heads':['none',head],'style':{'color':'#000000'}}
        rows=candidates(scene,styles,[arrow])
        assert len(nominate(rows))==1


def test_unaccepted_refinement_preserves_previous_style_and_accept_commits():
    old={'node':{'fill':'#00FF00'},'line':{'stroke':'#000000'}}
    proposed={'node':{'fill':'#000000'},'line':{'stroke':'#222222'}}
    result,rejected=commit_refinement(old,proposed,{'node':{'value':{'verdict':'revise'}},'line':{'value':{'verdict':'accept'}}})
    assert result=={'node':old['node'],'line':proposed['line']}
    assert rejected==['node'] and proposed['node']['fill']=='#000000'


def test_duplicate_rule_preserves_real_parallel_source_strokes():
    image=Image.new('RGB',(130,70),'white');draw=ImageDraw.Draw(image)
    draw.line((10,30,110,30),fill='black',width=2)
    def row(name,y):return {'id':name,'owner':'n','layer':'shape','heads':['none','none'],
        'length':100.,'points':np.array([[10,y],[110,y]],float),'style':{'stroke':'#000000','width_px':1,'dash':'solid'}}
    a,b=row('a',30),row('b',31)
    assert len(duplicates([a,b],np.asarray(image))[0])==1
    draw.line((10,34,110,34),fill='black',width=2)
    assert not duplicates([a,row('c',34)],np.asarray(image))[0]


def test_transparency_removes_exterior_white_without_erasing_white_object_interior():
    image=Image.new('RGBA',(60,60),'white');draw=ImageDraw.Draw(image)
    draw.rectangle((10,10,49,49),fill='blue');draw.rectangle((20,20,39,39),fill='white')
    image.putpixel((11,11),(0,0,255,100))
    result=transparent_exterior(image)
    assert result.getpixel((0,0))[3]==0
    assert result.getpixel((25,25))==(255,255,255,255)
    assert result.getpixel((11,11))[3]==100
    with pytest.raises(ValueError):transparent_exterior(Image.new('RGB',(20,20),'white'))


def test_replacement_stays_at_background_owner_layer(tmp_path):
    from pptx.enum.shapes import MSO_SHAPE
    p=Presentation();p.slide_width=p.slide_height=1000000;s=p.slides.add_slide(p.slide_layouts[6])
    cloud=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,900000,900000);cloud.name='cloud/body'
    node=s.shapes.add_shape(MSO_SHAPE.OVAL,100000,100000,200000,200000);node.name='network/node'
    xml=node._element.xml;p.save(tmp_path/'source.pptx')
    image=tmp_path/'asset.png';Image.new('RGBA',(50,50),(0,0,255,128)).save(image)
    replace_objects(tmp_path/'source.pptx',tmp_path/'result.pptx',[{'owner':'cloud','source_size':[100,100],
        'path':str(image),'sha256':sha(image),'placement_source_half_open':[0,0,90,90]}])
    result=Presentation(tmp_path/'result.pptx')
    assert [s.name for s in result.slides[0].shapes]==['IMG EDIT cloud','network/node']
    assert result.slides[0].shapes[1]._element.xml==xml


def test_measured_ascender_overflow_is_bounded_and_does_not_resize_text(tmp_path):
    p=Presentation();p.slide_width=p.slide_height=1000000
    sp=p.slides.add_slide(p.slide_layouts[6]).shapes.add_textbox(100000,-30000,400000,200000)
    sp.name='TXT t';sp.text='heading';path=tmp_path/'text.pptx';p.save(path)
    row={'id':'t','kind':'text','bbox_2d':[10,2,50,15],'font_size_px':16,
         'layout':{'measurement':{'method':'native_ppt_atlas','dy':6,'reference_size_px':24}}}
    assert not clamp_small_text_overflow(path,[100,100])
    assert len(clamp_small_text_overflow(path,[100,100],texts=[row]))==1
    sp=Presentation(path).slides[0].shapes[0]
    assert (sp.top,sp.width,sp.height,sp.text)==(0,400000,200000,'heading')


def test_editor_json_api_data_uri_and_cache_do_not_repeat_generation(tmp_path,monkeypatch):
    import requests,base64
    from io import BytesIO
    from full_recovery.image_edit import Editor
    calls=[]
    class Response:
        def raise_for_status(self):pass
        def json(self):
            image=Image.new('RGB',(128,128),'white');ImageDraw.Draw(image).rectangle((20,30,105,97),fill='blue')
            buf=BytesIO();image.save(buf,format='PNG')
            return {'data':[{'b64_json':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()}]}
    def post(self,url,**kw):calls.append((url,kw));return Response()
    monkeypatch.setattr(requests.Session,'post',post)
    editor=Editor({'base_url':'http://example.invalid/v1','image_size':128})
    crop=Image.new('RGB',(80,60),'white');path=tmp_path/'edit.png'
    first=editor.edit(crop,'isolate the blue background',path)
    assert first==editor.edit(crop,'isolate the blue background',path) and len(calls)==1
    assert calls[0][1]['json']['images'][0]['image_url'].startswith('data:image/png;base64,')
    # Complete output may extend beyond the old crop; it must not be cut off
    # when converting the padded editor canvas back to source coordinates.
    assert first['source_extent_wh'][0]>crop.width
    assert Image.open(path).getpixel((0,0))[3]==0
    icon=editor.edit(crop,'Keep the icon and its radial arrows',tmp_path/'icon.png',role='icon')
    assert icon['role']=='icon'
    instruction=calls[-1][1]['json']['prompt']
    assert 'intrinsic arrows' in instruction and 'Remove all foreground nodes' not in instruction


def test_editor_local_scope_and_label_boxes_follow_letterbox_transform(tmp_path,monkeypatch):
    import requests,base64,json
    from io import BytesIO
    from full_recovery.image_edit import Editor
    sent=[]
    class Response:
        def raise_for_status(self):pass
        def json(self):
            im=Image.new('RGB',(128,128),'white')
            ImageDraw.Draw(im).rectangle((40,50,70,80),fill='blue')
            out=BytesIO();im.save(out,format='PNG')
            return {'data':[{'b64_json':base64.b64encode(out.getvalue()).decode()}]}
    def post(self,url,**kwargs):sent.append(kwargs['json']['prompt']);return Response()
    monkeypatch.setattr(requests.Session,'post',post)
    request={'target':'blue circular icon','remove_overlays':['external gray line'],
        'edit_box':[10,5,30,20],'native_labels':[{'text':'A','box':[12,6,18,15]}]}
    editor=Editor({'base_url':'http://example.invalid/v1','image_size':128})
    editor.edit(Image.new('RGB',(80,40),'white'),request,tmp_path/'edited.png',role='icon')
    # The 80x40 source occupies [32,48,96,80], not [0,0,80,40].
    assert '[40.0, 52.0, 56.0, 64.0]' in sent[0]
    assert json.dumps([41.6,52.8,46.4,60.0]) in sent[0]
    assert sent[0].count('"text": "A"')==1
