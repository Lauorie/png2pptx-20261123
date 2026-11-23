"""Known-width raster and scale regressions independent of the estimator."""
import numpy as np
import pytest
from PIL import Image, ImageDraw
from lib import fit, style


@pytest.mark.parametrize('scale',[1,2,5])
@pytest.mark.parametrize('width',[2,4,8])
def test_resampling_ringing_does_not_split_a_colored_shaft(scale,width):
    original=Image.new('RGB',(260,140),'white');d=ImageDraw.Draw(original)
    d.line([(20,70),(200,70)],fill='#334477',width=width)
    d.polygon([(200,70),(182,50),(182,90)],fill='#334477')
    mask=Image.fromarray(np.any(np.array(original)<200,axis=-1))
    image=original.resize((260*scale,140*scale),Image.Resampling.LANCZOS)
    row=dict(id='x',kind='straight',start=[20.5*scale-.5,70.5*scale-.5],end=[200.5*scale-.5,70.5*scale-.5],
        via=[],heads=['none','triangle'],start_head=False,end_head=True,end_head_type='triangle',line_style='solid')
    measured=style.recover(np.asarray(image),np.asarray(mask.resize(image.size,Image.Resampling.NEAREST)),row,adaptive_heads=True)
    assert abs(measured['width_px']/scale-width)<.55


def test_local_width_twenty_maps_to_four_without_altering_native_measurement():
    row=dict(id='x',kind='straight',start=[0,0],via=[],end=[100,0],style=dict(width_px=20,dash=dict(mark_px=30,gap_px=20)))
    view=dict(view_to_global=[[.2,0,10],[0,.2,20],[0,0,1]],size=[100,100],bbox_global=[10,20,30,40])
    result=fit.transform(row,view)
    assert result['style']['width_px']==4 and row['style']['width_px']==20
    assert result['style']['dash']['mark_px']==6 and result['style']['dash']['gap_px']==4
    assert result['style']['evidence']['width_conversion']['local_width_px']==20
