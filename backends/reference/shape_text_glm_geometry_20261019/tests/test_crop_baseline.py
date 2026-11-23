from copy import deepcopy

import pytest

from shape_integration.common import save
from shape_integration.texts import local_geometry


def inventory():
    return {'texts':[{'id':'T1','text':'x1','kind':'formula','latex':r'x_1',
        'bbox_2d':[1,2,20,30],'rotation_deg':0,
        'runs':[{'text':'x','color':'#FF0000','script':'normal'},
                {'text':'1','color':'#0000FF','script':'sub'}]}],
        'regions_needing_review':[{'reason':'existing warning'}]}


def test_crop_baseline_enables_latest_content_and_color_review(tmp_path,monkeypatch):
    original=inventory();before=deepcopy(original)
    def refine(rows,source,folder,runtime,opt,pool,progress):
        assert opt['content_review_enabled'] is True
        assert opt['geometry_max_targets']==6
        rows[0]['bbox_2d']=[2,3,18,29]
        save(folder/'summary.json',{'unresolved_geometry':0})
        return rows,[]
    monkeypatch.setattr(local_geometry,'refine',refine)
    result=local_geometry.refine_inventory(original,None,tmp_path,None,
        {'content_review_enabled':True,'geometry_max_targets':99})
    assert original==before
    assert result['texts'][0]['latex']==before['texts'][0]['latex']
    assert result['texts'][0]['runs']==before['texts'][0]['runs']
    assert result['texts'][0]['bbox_2d']==[2,3,18,29]
    assert result['regions_needing_review']==original['regions_needing_review']


def test_crop_failure_propagates_as_review_warning(tmp_path,monkeypatch):
    def refine(rows,*args):
        save(tmp_path/'summary.json',{'unresolved_geometry':1})
        return rows,['request timeout']
    monkeypatch.setattr(local_geometry,'refine',refine)
    result=local_geometry.refine_inventory(inventory(),None,tmp_path,None,{})
    assert result['texts']==inventory()['texts']
    assert len(result['regions_needing_review'])==3


def test_crop_failure_preserves_original_inventory(tmp_path,monkeypatch):
    def refine(rows,*args):
        rows[0]['text']='wrong'
        raise ValueError('invalid reply')
    monkeypatch.setattr(local_geometry,'refine',refine)
    original=inventory()
    result=local_geometry.refine_inventory(original,None,tmp_path,None,{})
    assert result['texts']==original['texts']
    assert result['regions_needing_review'][-1]['policy']=='previous_valid_inventory'
