"""Small-target evidence keeps target identity separate from neighbouring paint."""
import copy
import numpy as np
from PIL import Image,ImageDraw
from shape_integration.styling.views import target_details,contour_identity


def test_triangle_detail_owns_its_contour_and_preserves_clean_pixels(tmp_path):
    source=tmp_path/'source.png';im=Image.new('RGB',(160,100),'white');draw=ImageDraw.Draw(im)
    draw.ellipse((20,30,50,60),fill='#30E050')
    draw.polygon([(57,36),(73,45),(57,54)],fill='black');im.save(source)
    table={'h':{'kind':'path','bbox_2d':[57,36,73,54],
                    'commands':[['M',57,36],['L',73,45],['L',57,54],['Z']]}}
    before=copy.deepcopy(table)
    images,records=target_details(source,table,{'P7':'h'},tmp_path/'detail',2)
    assert len(images)==1 and set(records)=={'P7'}
    assert records['P7']['contour']=='triangle' and contour_identity(table['h'])=='triangle'
    with Image.open(images[0][1]) as sheet:
        clean=np.asarray(sheet)[40:216,2:178]
        assert np.any(np.max(clean,axis=2)<20)  # Black source target survives amplification.
        assert not np.any(np.all(clean==[0,113,188],axis=2))  # No locator ink on CLEAN.
    assert table==before


def test_large_visible_target_does_not_add_an_extra_detail_view(tmp_path):
    source=tmp_path/'source.png';Image.new('RGB',(200,200),'white').save(source)
    table={'p':{'kind':'path','bbox_2d':[10,10,190,190],
                'commands':[['M',10,10],['L',190,190]]}}
    assert target_details(source,table,{'P1':'p'},tmp_path/'detail',2)==([], {})
