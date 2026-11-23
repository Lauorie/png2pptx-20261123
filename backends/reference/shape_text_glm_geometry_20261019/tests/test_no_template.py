from copy import deepcopy
from PIL import Image,ImageDraw
import pytest
from shape_integration.texts import ocr_geometry
from shape_integration.texts.ocr_refinement import proposal


def fixture():
    source=Image.new('RGB',(150,80),'white')
    ImageDraw.Draw(source).rectangle((42,25,50,44),fill='black')
    row={'id':'T1','text':'3','kind':'text','bbox_2d':[35,20,58,50],
         'color':'#123456','rotation_deg':0,'font':'Arial','bold':True,
         'runs':[{'text':'3','script':'normal','color':'#123456'}]}
    d={'index':0,'text':'3','confidence':.99,'rotation':0,'source_bbox':[39,22,53,47]}
    return source,row,d


def test_off_never_renders_or_matches_a_template(monkeypatch):
    source,row,d=fixture();before=deepcopy(row)
    def forbidden(*a,**k):raise AssertionError('Template must not be used')
    monkeypatch.setattr(ocr_geometry,'_register_at',forbidden)
    record=ocr_geometry.select(row,[d],source,template_validation=False)
    result=proposal(row,record['bbox_2d'],record)
    assert record['template_validation'] is False and record['bbox_2d']==[42,25,50,44]
    assert result['text']==row['text'] and result['runs']==row['runs'] and row==before
    assert result['color']==row['color'] and result['bold']==row['bold']


def test_wrong_ocr_transcription_does_not_rewrite_content():
    source,row,d=fixture();d['text']='wrong'
    record=ocr_geometry.select(row,[d],source,template_validation=False)
    assert proposal(row,record['bbox_2d'],record)['text']=='3'


def test_blank_and_consumed_detection_have_no_fake_ocr_success():
    source,row,d=fixture()
    with pytest.raises(ValueError):ocr_geometry.select(row,[d],Image.new('RGB',source.size,'white'),template_validation=False)
    result=ocr_geometry.select(row,[d],source,used=[0],template_validation=False)
    assert result['detection_indices']==[] and result['proposal_type']=='prior_ink_only_no_ocr'
