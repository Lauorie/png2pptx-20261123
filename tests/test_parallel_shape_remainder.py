from copy import deepcopy
import numpy as np
from PIL import Image,ImageDraw
from shape_integration.fitting.source_primitives import register,update
from full_recovery.source_route_nomination import nominate


def nodes_fixture():
    source=Image.new('RGB',(240,250),'white');draw=ImageDraw.Draw(source);parts=[]
    for i,y in enumerate((35,85,135,190)):
        draw.rectangle((170,y-9,188,y+9),fill='#FFCCED',outline='#FF78C7',width=1)
        if i==2:continue
        p={'id':f'p{i}','object_id':'owner','key':['test','owner',f'p{i}'],'status':'ok','kind':'rectangle',
            'parameters':{'center':[179.,y],'width':18.,'height':18.,'angle_radians':0.},
            'diagnostics':{},'observed_point_ids':[]}
        update(p);parts.append(p)
    job={'frame':{'source_crop_half_open':[0,0,240,250],'actual_crop_input_size_px':[240,250]},
         'identity_bbox_local':[80,10,155,225]}
    return {'parts':parts},job,source


def test_observed_owned_family_extent_overrides_cut_locator_without_losing_member():
    local,job,source=nodes_fixture();audit=register(local,job,source)
    assert len(local['parts'])==4
    addition=[r for r in audit if r.get('added')]
    assert len(addition)==1 and addition[0]['measured_owned_family_extent'][0]>job['identity_bbox_local'][2]
    assert sorted(round(p['parameters']['center'][1])for p in local['parts'])==[35,85,135,190]
    before=[p['id']for p in local['parts']];register(local,job,source)
    assert [p['id']for p in local['parts']]==before


def test_closed_family_does_not_copy_an_independently_owned_image_or_new_column():
    local,job,source=nodes_fixture();register(local,job,source,owned_regions=[[167,123,191,146]])
    assert len(local['parts'])==3
    draw=ImageDraw.Draw(source)
    for y in(35,85,135,190):draw.rectangle((207,y-9,225,y+9),fill='#FFCCED',outline='#FF78C7',width=1)
    local,job,_=nodes_fixture();audit=register(local,job,source)
    assert len(local['parts'])==4
    assert all(abs(p['parameters']['center'][0]-179)<1 for p in local['parts'])


def line(ident,layer,start,end):
    return {'id':ident,'key':ident,'layer':layer,'points':np.linspace(start,end,80),
        'length':float(np.linalg.norm(np.array(end)-start)),'heads':['none','none'],
        'endpoint_nodes':('u','v'),'style':{'stroke':'#4CAF50','color':'#4CAF50','width_px':1}}


def test_pale_antialiased_source_ridge_nominates_complete_same_endpoint_group():
    image=Image.new('RGB',(180,140),'white');draw=ImageDraw.Draw(image)
    pigment=np.array([76,175,80]);pale=tuple(np.rint(255-(255-pigment)*.20).astype(int))
    draw.line((30,50,145,50),fill=pale,width=1)
    a=line('long','shape',[30,45],[145,45]);b=line('short','arrows',[45,51],[125,51])
    original=deepcopy([a,b]);pairs=nominate([a,b],np.asarray(image))
    assert len(pairs)==1 and pairs[0]['scope']=='unique_endpoint_headless_source_route'
    for r,old in zip([a,b],original):np.testing.assert_equal(r['points'],old['points'])
    # Gray ink has a different source pigment direction and must not substitute.
    gray=Image.new('RGB',image.size,'white');ImageDraw.Draw(gray).line((30,50,145,50),fill='#CCCCCC',width=1)
    assert nominate([a,b],np.asarray(gray))==[]
    # Real pale parallel lines map to different source bands.
    draw.line((30,60,145,60),fill=pale,width=1);b['points'][:,1]=60
    assert nominate([a,b],np.asarray(image))==[]
