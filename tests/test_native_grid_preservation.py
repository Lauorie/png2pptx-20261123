"""Composition cannot infer a texture from incomplete structural strokes."""
import copy
from PIL import Image
from full_recovery.compose import sha
from full_recovery.image_assets import review


def test_detached_internal_strokes_do_not_create_a_grid_pattern(tmp_path):
    source=tmp_path/'source.png';Image.new('RGB',(200,150),'white').save(source)
    parts={}
    for name,commands in {
        'face':[['M',10,10],['L',190,10],['L',190,140],['L',10,140],['Z']],
        'a':[['M',40,40],['L',160,40]],
        'b':[['M',40,90],['L',160,90]],
    }.items():
        parts[name]={'id':name,'object_id':'owner','kind':'path','status':'ok',
                     'bbox_2d':[10,10,190,140],'commands':commands}
    scene={'source_size':[200,150],'source_sha256':sha(source),'parts':parts,
           'objects':{'owner':{'candidate':{'label':'panel','kind':'rectangle',
              'bbox_2d':[10,10,190,140]},'part_keys':list(parts)}}}
    before=copy.deepcopy(scene)
    result=review(scene,{},[],[],source,source,tmp_path/'review',None,selected=[])
    assert result['native_patterns']==[]
    assert result['drop_shape_keys']==[]
    assert scene==before
