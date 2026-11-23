from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import pytest

from shape_integration.texts import options
from shape_integration.texts.prompts import recognize
from shape_integration.texts.text_scope import RECOVERY_SCOPE
from shape_integration.texts.local_content import validate, refine


def target():
    return {'id':'T1','text':'sample','bbox_2d':[10,10,80,30], 'rotation_deg':0,
            'kind':'text','latex':'','runs':[{'text':'sample','script':'normal'}],
            'font':'Arial','color':'#000000','bold':False,'italic':False,'legibility':'clear'}


def excluded():
    return {'id':'T1','status':'excluded_image_text','bbox_2d':None,'rotation_deg':None,
            'direction_status':'ambiguous','reason':'Lettering is inside a photographed sign, not a diagram label.'}


def test_global_and_local_share_scope_policy():
    for local in [False,True]:
        prompt=recognize([100,100],local=local)
        assert all(rule in prompt['rules'] for rule in RECOVERY_SCOPE)


def test_exclusion_requires_evidence_and_has_no_geometry():
    assert validate({'items':[excluded()]},[target()],[100,100])['items'][0]['status']=='excluded_image_text'
    item=excluded();item['reason']=''
    with pytest.raises(ValueError,match='evidence'):
        validate({'items':[item]},[target()],[100,100])
    item=excluded();item['bbox_2d']=[1,1,10,10]
    with pytest.raises(ValueError,match='null'):
        validate({'items':[item]},[target()],[100,100])


def test_excluded_image_text_is_not_exported(tmp_path):
    class Runtime:
        def for_stage(self,**kw):return self
        def call(self,*args,**kw):return args[5]({'items':[excluded()]})
    source=tmp_path/'source.png';Image.new('RGB',(100,100),'white').save(source)
    with ThreadPoolExecutor(max_workers=1) as pool:
        rows,errors=refine([target()],source,tmp_path/'review',Runtime(),options({}),pool)
    assert rows==[] and errors==[]
    import json
    summary=json.loads((tmp_path/'review/summary.json').read_text())
    assert summary['excluded_image_text_ids']==['T1'] and summary['retained_targets']==0
