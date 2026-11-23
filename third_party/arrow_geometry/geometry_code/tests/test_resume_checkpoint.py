import json
from pathlib import Path
from PIL import Image
import pytest

import resume_checkpoint


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def test_freeze_keeps_geometry_and_copies_crop_frame(tmp_path):
    old=tmp_path/'old';root=tmp_path/'new';cid='task_1';source=old/'geometry/cases'/cid
    source.mkdir(parents=True)
    Image.new('RGB',(80,60),'white').save(source/'clean.png')
    save(source/'reconstruction_handoff.json',{'tasks':[]})
    image=root/'localization/inputs'/cid/'source.png';image.parent.mkdir(parents=True);image.write_bytes((source/'clean.png').read_bytes())
    save(root/'localization/cases'/cid/'reconstruction_handoff.json',{'tasks':[]})
    crop=source/'geometry/input/clean.png';crop.parent.mkdir(parents=True)
    Image.new('RGB',(120,100),'white').save(crop)
    frame={'view_to_global':[[.5,0,1],[0,.5,2],[0,0,1]]}
    save(crop.parent/'transform.json',frame)
    groups=[{'group':'G1','rows':[{'id':'P1','start':[1.25,2.5],'end':[40.25,40.5],'local_input':str(crop)}],
             'issues':[{'stage':'geometry_critic','state':'budget_exhausted'}]}]
    for name in ['initial_geometry_results.json','before_reconciliation.json','geometry_results.json']:save(source/name,groups)
    save(source/'post_geometry_missing/audit.json',{'issues':[]})
    save(source/'reconciliation/audit.json',{'issues':[]})
    files=resume_checkpoint.freeze(old,root,[cid])
    restored,before,final,issues=resume_checkpoint.load(root/'resume_geometry'/cid)
    assert restored==before==groups
    row=final[0]['rows'][0];original=groups[0]['rows'][0]
    assert {k:v for k,v in row.items() if k!='local_input'}=={k:v for k,v in original.items() if k!='local_input'}
    copied=Path(row['local_input']);assert copied.read_bytes()==crop.read_bytes()
    assert json.loads((copied.parent/'transform.json').read_text())==frame
    assert issues==groups[0]['issues']
    assert all(p.is_file() for p in files)


def test_resume_rejects_different_source_image(tmp_path):
    old=tmp_path/'old';root=tmp_path/'new';cid='task_1'
    a=old/'geometry/cases'/cid/'clean.png';a.parent.mkdir(parents=True);a.write_bytes(b'old')
    b=root/'localization/inputs'/cid/'source.png';b.parent.mkdir(parents=True);b.write_bytes(b'changed')
    with pytest.raises(ValueError,match='source image mismatch'):resume_checkpoint.freeze(old,root,[cid])
