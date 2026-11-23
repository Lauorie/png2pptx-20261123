import numpy as np
import pytest
from PIL import Image,ImageDraw

from full_recovery.source_ellipsis import observe,clear_image_copies


def test_pixel_quantization_keeps_one_round_dot_group_and_retires_native_copies(tmp_path):
    from full_recovery.source_dot_ownership import resolve
    source=Image.new('RGB',(100,60),'white');draw=ImageDraw.Draw(source)
    for b in [(30,30,31,31),(36,30,37,31),(42,30,42,31)]:draw.rectangle(b,fill='black')
    rows=[{'id':'p','text':'...','color':'#000000','bbox_2d':[29,28,44,34]}]
    marks=observe(source,rows)
    assert len(marks)==1 and len(marks[0]['dots'])==3
    scene={'parts':{}};styles={}
    for i,dot in enumerate(marks[0]['dots']):
        x,y=dot['center'];d=dot['diameter'];key=f'd{i}'
        scene['parts'][key]={'object_id':'owner','status':'ok',
            'commands':[['M',x-d/2,y-d/2],['L',x+d/2,y-d/2],['L',x+d/2,y+d/2],['L',x-d/2,y+d/2],['Z']]}
        styles[key]={'fill':'#000000','stroke':None}
    path=tmp_path/'source.png';source.save(path)
    removed,audit=resolve(scene,styles,rows,path)
    assert removed==['d0','d1','d2'] and len(audit)==1
    # A three-pixel long stroke is still distinct from one-pixel sampling error.
    draw.rectangle((42,30,42,31),fill='white');draw.rectangle((42,29,42,31),fill='black')
    assert not observe(source,rows)


@pytest.mark.parametrize('scale',[1,2])
@pytest.mark.parametrize('mode',['complete','missing_dot','textured_background'])
def test_raster_cleanup_uses_complete_source_occurrence_and_retains_background(scale,mode):
    source=Image.new('RGB',(150,100),'#AACCEE');draw=ImageDraw.Draw(source)
    for x in (70,82,94):draw.ellipse((x-2,68,x+2,72),fill='black')
    rows=[{'id':'p','text':'...','color':'#000000','bbox_2d':[66,66,99,75]}]
    marks=observe(source,rows);assert len(marks)==1
    image=source.crop((50,50,130,100)).convert('RGBA');draw=ImageDraw.Draw(image)
    draw.rectangle((60,3,70,10),fill='red')
    if mode=='missing_dot':draw.rectangle((30,17,34,23),fill='#AACCEE')
    if mode=='textured_background':
        for x in range(15,50,2):draw.line((x,14,x,26),fill='#223344')
    image=image.resize((80*scale,50*scale),Image.Resampling.NEAREST)
    before=np.asarray(image).copy()
    result,audit=clear_image_copies(image,[50,50,130,100],marks)
    if mode!='complete':
        assert not audit and np.array_equal(result,before)
    else:
        assert len(audit)==1
        for x in (20,32,44):assert result.getpixel((x*scale,20*scale))==(170,204,238,255)
        assert result.getpixel((65*scale,6*scale))==(255,0,0,255)
        assert np.array_equal(np.asarray(result)[:12*scale],before[:12*scale])
