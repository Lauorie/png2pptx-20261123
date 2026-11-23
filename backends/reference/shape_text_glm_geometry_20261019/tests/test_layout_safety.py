from concurrent.futures import ThreadPoolExecutor
from zipfile import ZipFile
import json
import pytest
from PIL import Image
from lxml import etree

from shape_integration.texts import contracts, options, export
from shape_integration.texts.local_content import validate, refine
from shape_integration.texts.local_content import validate_pixel_lines


def row():
    return dict(id='T1',text='First Second',runs=[{'text':'First Second','script':'normal'}],
                bbox_2d=[10,10,90,65],kind='text',latex='',rotation_deg=0,font='Arial',
                color='#000000',bold=False,italic=False,underline=False,legibility='clear')


def part(text,y):
    return dict(text=text,runs=[{'text':text,'script':'normal'}],kind='text',latex='',
                bbox_2d=[20,y,120,y+20],rotation_deg=0,direction_status='confirmed',
                font='Calibri',underline=True,reason='Separate visible baseline')


def test_colored_prose_cannot_use_math_mode():
    a=row();a.update(kind='formula',latex=r'\textcolor{#FF0000}{First}\ \textcolor{#0000FF}{Second}')
    with pytest.raises(ValueError,match='prose'):
        contracts.recognition({'texts':[a]},[200,100])
    a.update(kind='text',latex='')
    assert contracts.recognition({'texts':[a]},[200,100])['texts'][0]['kind']=='text'


def test_split_children_keep_unique_ids_styles_and_nonoverlapping_geometry():
    item={'id':'T1','status':'split','reason':'Two visual lines','parts':[part('First',10),part('Second',40)]}
    result=validate({'items':[item]},[row()],[200,100])['items'][0]
    assert [x['id'] for x in result['parts']]==['T1.L01','T1.L02']
    assert all(x['font']=='Calibri' and x['underline'] for x in result['parts'])
    item['parts'][1]['bbox_2d']=item['parts'][0]['bbox_2d']
    with pytest.raises(ValueError,match='overlapping'):
        validate({'items':[item]},[row()],[200,100])


def test_id_keyed_responses_are_normalized_without_identity_changes():
    item={**part('First',10),'status':'located'}
    result=validate({'T1':item},[row()],[200,100])
    assert result['items'][0]['id']=='T1'
    with pytest.raises(ValueError,match='disagrees'):
        validate({'T1':{**item,'id':'T2'}},[row()],[200,100])


def test_split_replaces_parent_in_pipeline_and_keeps_provenance(tmp_path):
    class Runtime:
        def for_stage(self,**kw):return self
        def call(self,*args,**kw):
            return args[5]({'items':[{'id':'T1','status':'split','reason':'Two separate baselines',
                                      'parts':[part('First',400),part('Second',800)]}]})
    source=tmp_path/'source.png';Image.new('RGB',(100,100),'white').save(source)
    with ThreadPoolExecutor(max_workers=1) as pool:
        rows,errors=refine([row()],source,tmp_path/'review',Runtime(),options({}),pool)
    assert not errors and len(rows)==2
    assert all(x['parent_id']=='T1' for x in rows)
    assert json.loads((tmp_path/'review/summary.json').read_text())['split_parent_ids']==['T1']


def test_underline_is_editable_native_style(tmp_path):
    a=row();a.update(underline=True,frame_px=[10,10,90,30],font_size_px=20,resolved_font='Arial')
    export.write([a],[200,100],tmp_path/'text.pptx')
    with ZipFile(tmp_path/'text.pptx') as z:xml=etree.fromstring(z.read('ppt/slides/slide1.xml'))
    assert xml.xpath('count(//*[local-name()="rPr"][@u="sng"])')==1


def test_pixel_baselines_check_split_children_not_just_parent(monkeypatch):
    item={'id':'T1','status':'split','reason':'Two visual units',
          'parts':[part('First',10),part('Second',40)]}
    checked=validate({'items':[item]},[row()],[200,100])
    seen=[]
    def bands(candidate,image):
        seen.append(candidate['id'])
        return [[20,40,120,46],[20,54,120,60]] if candidate['id']=='T1.L02' else []
    monkeypatch.setattr('shape_integration.texts.local_content.line_bands',bands)
    frame={'scale':1,'offset':[0,0],'box':[0,0,200,100]}
    with pytest.raises(ValueError,match='T1.L02.*multiple separated'):
        validate_pixel_lines(checked,[row()],frame,Image.new('RGB',(200,100)))
    assert seen==['T1.L01','T1.L02']
