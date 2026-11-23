import numpy as np
from PIL import Image, ImageDraw, ImageFont

from shape_integration.texts.registration import register, best_match


def test_cross_axis_exception_requires_whole_phrase_and_strong_pixels():
    from shape_integration.texts.registration import supported_cross_axis_repair
    candidate = {'score': .92, 'flow_coverage': .9, 'cross_axis_ratio': .4}
    assert supported_cross_axis_repair({'kind': 'text'}, candidate)
    assert not supported_cross_axis_repair({'kind': 'formula'}, candidate)
    assert not supported_cross_axis_repair({'kind': 'text'}, {**candidate, 'score': .7})
    assert not supported_cross_axis_repair({'kind': 'text'}, {**candidate, 'flow_coverage': .4})


def test_oversized_cross_axis_box_can_recover_rotation(tmp_path):
    mask = Image.new('L', (500, 80), 0)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 32)
    ImageDraw.Draw(mask).text((4, 4), 'Object Detection', font=font, fill=255)
    mask = mask.crop(mask.getbbox())
    template = Image.new('RGBA', mask.size, (0, 0, 0, 0))
    template.putalpha(mask)
    path = tmp_path / 'ink.png'
    template.save(path)
    rotated = template.rotate(90, expand=True)
    source = Image.new('RGB', (180, 400), (220, 235, 250))
    source.paste(rotated, (70, 40), rotated)
    row = {'id': 'T', 'text': 'Object Detection', 'kind': 'text',
           'rotation_deg': 90, 'color': '#000000',
           'bbox_2d': [45, 35, 119, 45 + rotated.height]}
    out, _, audit = register(row, {'ink_template': str(path)}, source)
    assert audit['status'] == 'accepted'
    assert out['rotation_deg'] == -90
    assert audit['after']['narrow_extent']
    assert abs(out['bbox_2d'][0] - 70) <= 2


def test_foreground_match_does_not_reward_blank_background():
    template=np.zeros((9,12),np.float32);template[2:7,3:8]=1
    image=np.zeros((40,50),np.float32);image[13:22,17:29]=template
    score,x,y=best_match(image,template)
    assert score>.999 and (x,y)==(17,13)
    assert best_match(image,np.zeros((3,4),np.float32)) is None


def test_source_glyphs_recover_rotated_direction_and_tight_box(tmp_path):
    mask=Image.new('L',(300,80),0)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',32)
    ImageDraw.Draw(mask).text((4,4),'Attention',font=font,fill=255)
    mask=mask.crop(mask.getbbox())
    template=Image.new('RGBA',mask.size,(0,0,0,0));template.putalpha(mask)
    path=tmp_path/'ink.png';template.save(path)
    rotated=template.rotate(-90,expand=True)
    source=Image.new('RGB',(150,250),(220,235,250));source.paste(rotated,(60,40),rotated)
    row={'id':'T','text':'Attention','kind':'text','rotation_deg':-90,'color':'#000000',
         'bbox_2d':[55,30,60+rotated.width+5,40+rotated.height+10]}
    out,_,audit=register(row,{'ink_template':str(path)},source)
    assert audit['status']=='accepted'
    assert out['rotation_deg']==90
    assert abs(out['bbox_2d'][0]-60)<=2 and abs(out['bbox_2d'][1]-40)<=2


def test_blank_source_does_not_change_geometry(tmp_path):
    template=Image.new('RGBA',(10,20),(0,0,0,255));path=tmp_path/'ink.png';template.save(path)
    row={'id':'T','kind':'text','rotation_deg':0,'color':'#000000','bbox_2d':[10,10,30,40]}
    out,_,audit=register(row,{'ink_template':str(path)},Image.new('RGB',(60,60),'white'))
    assert audit['status']=='low_score' and out==row
