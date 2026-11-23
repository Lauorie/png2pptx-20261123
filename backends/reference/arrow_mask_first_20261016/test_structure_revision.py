import numpy as np
import pytest
import shared_repair as sr
import curve_sampling as cs
from lib import style


def test_no_mask_block_contour_remains_filled_and_has_no_synthetic_head():
    points=[['a',10,10],['b',30,10],['c',30,30],['d',40,30],['e',20,50],['f',0,30],['g',10,30]]
    route=dict(id='r',kind='block_arrow',line_style='solid',masks=[],contour=[n[0] for n in points],segments=[])
    a=sr.validate(dict(status='repair',replace=[],nodes=points,routes=[route]),[100,100],[100,100],[],[],True)
    geometry=sr.block_geometry(a['routes'][0],a['nodes'],{})
    assert geometry['kind']=='block_arrow' and geometry['start']==[10,10]
    im=np.full((100,100,3),[40,130,210],dtype=np.uint8)
    result=style.filled(im,None,dict(geometry,id='r'))
    assert result['representation']=='filled_contour' and result['width_px'] is None
    assert result['start'] is None and result['end'] is None and result['color']=='#2882d2'


def test_nomination_topology_does_not_enter_repair_request():
    q=sr.prompt([100,100],[100,100],dict(bbox=[1,2,30,40],description='two invented arrows from the prior trunk',
        regions=[dict(bbox=[2,3,20,30],description='must connect hidden rail')]),[],[],True)
    assert q['focus']==dict(bbox=[1,2,30,40],regions=[dict(bbox=[2,3,20,30])])
    baseline=sr.prompt([100,100],[100,100],dict(bbox=[1,2,30,40],description='visual locator'),[],[],True,retain_locator_text=True)
    assert baseline['focus']['description']=='visual locator'


def test_zero_weight_points_cannot_pull_spline_to_neighbor():
    t=np.linspace(0,100,50);true=np.c_[t,10*np.sin(t/100*np.pi)]
    obs=np.concatenate([true,true+np.array([0,80])]);times=np.r_[t,t]
    weights=np.r_[np.ones(50),np.zeros(50)]
    curves,_=cs._joint_curve(obs,times,np.array([[0,0],[100,0]]),3,100,weights)
    p=cs._sample_curves(curves)
    assert p[:,1].max()<12 and p[:,1].min()>-1
    np.testing.assert_allclose(p[[0,-1]],[[0,0],[100,0]],atol=1e-8)
    with pytest.raises(ValueError):cs._joint_curve(obs,times,[[0,0],[100,0]],3,100,np.zeros(100))


def test_tiles_cover_portrait_and_landscape_without_creating_target_ids():
    import tile_coverage as tc
    for size in [(320,900),(1500,350),(800,800)]:
        cover=np.zeros(size[::-1],bool)
        for x0,y0,x1,y1 in tc.windows(size):cover[y0:y1,x0:x1]=True
        assert cover.all()


def test_tile_failure_is_explicit_and_successful_regions_keep_global_frame(tmp_path):
    import threading
    from PIL import Image
    import tile_coverage as tc
    class Model:
        def ask(self,path,prompt,images,validate):
            if path.parent.name=='T3':raise RuntimeError('simulated tile failure')
            w,h=prompt['size']
            return validate(dict(regions=[dict(bbox=[0,0,w,h],description='local region')]))
    cfg=dict(task_workers=6,context_padding=0,min_crop_side=1,display_side=200)
    result=tc.discover(Image.new('RGB',(600,400),'white'),[],tmp_path,Model(),cfg,dict(cpu=threading.Semaphore(2)))
    assert result['state']=='partial' and len(result['regions'])==5 and len(result['errors'])==1
    for r in result['regions']:
        x0,y0,x1,y1=r['bbox'];assert 0<=x0<x1<=600 and 0<=y0<y1<=400
