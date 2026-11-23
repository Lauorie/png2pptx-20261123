from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from zipfile import ZipFile

from lxml import etree
from PIL import Image
import pytest

from shape_integration.texts import local_content,export,single_crop
from shape_integration.texts.prompt_protocol import review
from shape_integration.texts.equations import NS


def row(angle=0):
    return {'id':'T1','text':'Label','kind':'text','latex':'',
            'runs':[{'text':'Label','script':'normal','color':'#000000'}],
            'bbox_2d':[20,20,80,45],'rotation_deg':angle,'font':'Arial','color':'#000000',
            'bold':False,'italic':False,'underline':False,'legibility':'clear',
            'visual_line_count':1,'line_layout_status':'single_baseline'}


def test_crop_prompt_explicitly_allows_absolute_angle_corrections():
    prompt=review([200,120],[row(90)],[dict(row(),id='ReadOnly')])
    rule=next(r for r in prompt['rules'] if r.startswith('CROP ORIENTATION:'))
    for phrase in ('rotation_deg is editable','NEW ABSOLUTE clockwise','old +90 may become -90',
                   'logical character order','ORIGINAL CLEAN crop axes','Context IDs remain read-only'):
        assert phrase in rule
    assert 'rotation_deg' in prompt['response_format']['items'][0]['units'][0]


@pytest.mark.parametrize('old,new',[(0,90),(90,-90),(180,0),(0,180),(20,35.5)])
def test_rotation_only_crop_change_is_validated_and_selected_for_refitting(old,new):
    original=row(old);before=deepcopy(original)
    unit={**deepcopy(original),'rotation_deg':new,'direction_status':'confirmed',
          'reason':'Source glyphs support the corrected reading direction'}
    checked=local_content.validate({'items':[{'id':'T1','status':'resolved','units':[unit]}]},
                                   [original],[200,120])['items'][0]
    expected=(new+180)%360-180
    assert checked['rotation_deg']==expected
    for key in ('text','runs','bbox_2d','font','color','bold','italic','underline'):
        assert checked[key]==original[key]
    revised={**original,**{k:checked[k] for k in local_content.FIELDS+local_content.STYLE_FIELDS}}
    updates=single_crop.changes([original],[revised],['T1'])
    assert updates['T1'][0]['rotation_deg']==expected
    assert original==before


def test_uncertain_direction_does_not_create_an_angle_update():
    original=row(90)
    checked=local_content.validate({'items':[{'id':'T1','status':'ambiguous','units':[],
                                            'reason':'Glyph direction is unreadable'}]},
                                   [original],[200,120])
    assert checked['items'][0]['rotation_deg'] is None
    assert original['rotation_deg']==90


def test_crop_angle_flows_through_model_reply_to_source_row(tmp_path):
    source=tmp_path/'source.png';Image.new('RGB',(200,120),'white').save(source)
    original=row(90);before=deepcopy(original);calls=[]
    class Runtime:
        def for_stage(self,**kwargs):return self
        def call(self,folder,name,system,prompt,images,validator,max_tokens,**kwargs):
            calls.append(prompt)
            unit={**row(-90),'bbox_2d':prompt['targets'][0]['rough_bbox_2d'],
                  'direction_status':'confirmed','reason':'Bottom-to-top glyph direction'}
            return validator({'items':[{'id':'T1','status':'resolved','units':[unit]}]})
    opt={'geometry_max_targets':6,'crop_min_padding_px':8,'crop_padding':.04,
         'crop_scale':3.,'global_long_edge':200,'crop_max_long_edge':400,'max_tokens':1000}
    with ThreadPoolExecutor(max_workers=1) as pool:
        revised,errors=local_content.refine([original],source,tmp_path/'review',Runtime(),opt,pool,
                                            target_ids=['T1'],collision_passes=0)
    assert len(calls)==1 and not errors
    assert revised[0]['rotation_deg']==-90
    assert revised[0]['text']=='Label'
    assert original==before


def test_corrected_angles_are_written_into_native_ppt_transforms(tmp_path):
    rows=[]
    for index,angle in enumerate((90,-90,180)):
        unit=row(angle)
        unit.update(id=f'T{index}',frame_px=[20+index*80,20,60,25],font_size_px=18,
                    resolved_font='Arial')
        rows.append(unit)
    target=tmp_path/'rotations.pptx';export.write(rows,[300,120],target)
    with ZipFile(target) as archive:tree=etree.fromstring(archive.read('ppt/slides/slide1.xml'))
    shapes=tree.findall('.//p:sp',NS)
    assert [int(shape.find('p:spPr/a:xfrm',NS).get('rot','0')) for shape in shapes]==[
        90*60000,270*60000,180*60000]
    assert tree.xpath('.//a:r/a:t/text()',namespaces=NS)==['Label']*3
