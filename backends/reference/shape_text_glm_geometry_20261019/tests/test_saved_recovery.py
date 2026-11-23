import runpy
from pathlib import Path

import pytest


recover=runpy.run_path(str(Path(__file__).resolve().parents[1]/'recover_saved_recognition.py'))['recover']


def test_invalid_item_does_not_remove_valid_siblings_or_clip_coordinates():
    good={'text':'Valid','bbox_2d':[5,5,35,20],'font':'Arial','color':'#000000'}
    bad={**good,'text':'Outside','bbox_2d':[95,5,110,20]}
    accepted,rejected=recover({'texts':[good,bad]},[100,40])
    assert len(accepted)==1 and accepted[0]['text']=='Valid'
    assert len(rejected)==1 and rejected[0]['response_index']==1
    assert rejected[0]['proposal']['bbox_2d']==[95,5,110,20]
    assert bad['bbox_2d']==[95,5,110,20]


def test_all_invalid_is_not_an_empty_success():
    with pytest.raises(ValueError,match='No individually valid'):
        recover({'texts':[{'text':'Bad','bbox_2d':[-1,0,5,5]}]},[100,40])
