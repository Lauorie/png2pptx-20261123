import copy
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent/'lib'))
import numpy as np
from PIL import Image, ImageDraw
import coverage_visual as cv


def row():
    return dict(id='test',kind='straight',start=[10,20],via=[],end=[60,20],
        style=dict(color='#000000',width_px=1,start=None,end=None,
            dash=dict(type='dashed',preset='dash',mark_px=4,gap_px=6,phase_px=0)))


def test_local_transform_scales_geometry_and_dash_without_mutation():
    r=row();r['beziers']=[[[10,20],[20,20],[40,20],[60,20]]];before=copy.deepcopy(r)
    out=cv.project([r],{'global_to_view':[[2,0,-10],[0,2,-20],[0,0,1]]})[0]
    assert out['start']==[10,20] and out['end']==[110,20]
    assert out['beziers'][0][-1]==[110,20]
    assert out['style']['width_px']==2 and out['style']['dash']['gap_px']==12
    assert r==before


def test_gray_source_is_not_blue_and_dash_gaps_stay_open(tmp_path):
    source=Image.new('RGB',(100,100),'white');ImageDraw.Draw(source).line([(0,70),(99,70)],fill='black',width=2)
    r=row();before=copy.deepcopy(r);out=tmp_path/'current.png'
    cv.render(source,[r],{'global_to_view':np.eye(3).tolist()},out,False)
    image=np.asarray(Image.open(out));assert image.shape==(100,100,3)
    assert image[20,12,2]>image[20,12,0]+100
    assert min(image[20,17])>240  # explicit dash gap, not a continuous guide
    assert image[70,50,0]==image[70,50,2] and 100<image[70,50,0]<220
    assert r==before


def test_filled_target_remains_filled(tmp_path):
    r=row();r.update(kind='block_arrow',start=[10,10],via=[[40,10],[40,40]],end=[10,40])
    out=tmp_path/'filled.png';cv.render(Image.new('RGB',(60,60),'white'),[r],{'global_to_view':np.eye(3).tolist()},out,False)
    pixel=np.asarray(Image.open(out))[25,25]
    assert pixel[2]>pixel[0]+100
