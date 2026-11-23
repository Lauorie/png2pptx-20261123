import json
import numpy as np
from PIL import Image
import evidence as ev
import retained_evidence as retained
from transport.common import write_json


def source(tmp_path,wrong_size=False):
    old=tmp_path/'old';old.mkdir()
    view=ev.identity_view([100,100]);write_json(old/'transform.json',view)
    m=np.zeros((100,100),bool);m[49:52,10:91]=True
    if wrong_size:m=m[:50]
    Image.fromarray(m.astype('uint8')*255).save(old/'mask.png')
    row=dict(id='G1:R1',short_id='E1',source_mask_paths=[str(old/'mask.png')],local_input=str(old/'clean.png'))
    return old,row,m


def test_reuse_projects_known_native_frame_and_preserves_source(tmp_path):
    old,row,m=source(tmp_path);raw=(old/'mask.png').read_bytes()
    clean=Image.new('RGB',(100,100),'white')
    crop,view=ev.frame(clean,[],tmp_path/'new',dict(context_padding=0,min_crop_side=20,display_side=100),box=[25,25,75,75])
    selected=retained.load([row],view,tmp_path/'retained')
    assert len(selected)==1 and selected[0]['id']=='M1'
    expected=ev.project(dict(path=old/'mask.png',view=ev.identity_view([100,100])),view)
    assert np.array_equal(selected[0]['mask'],expected)
    y,x=np.nonzero(expected)
    assert x.min()==0 and x.max()==99 and y.min()==48 and y.max()==53
    assert selected[0]['retained_from']['existing_short_ids']==['E1']
    assert (old/'mask.png').read_bytes()==raw


def test_shared_native_evidence_is_loaded_once_but_keeps_all_attributions(tmp_path):
    old,row,m=source(tmp_path);row2={**row,'id':'G2:R1','short_id':'E2'}
    selected=retained.load([row,row2],ev.identity_view([100,100]),tmp_path/'retained')
    assert len(selected)==1
    assert selected[0]['retained_from']['existing_ids']==['G1:R1','G2:R1']
    assert selected[0]['retained_from']['existing_short_ids']==['E1','E2']


def test_invalid_frame_is_reported_not_silently_guessed(tmp_path):
    old,row,m=source(tmp_path,wrong_size=True)
    selected=retained.load([row],ev.identity_view([100,100]),tmp_path/'retained')
    assert selected==[]
    audit=json.loads((tmp_path/'retained/manifest.json').read_text())
    assert 'dimensions differ' in audit['issues'][0]['error']
    assert audit['new_sam_calls']==0 and not audit['geometry_modified']
