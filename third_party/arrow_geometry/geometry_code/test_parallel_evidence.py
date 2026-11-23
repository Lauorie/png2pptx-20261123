import threading
from pathlib import Path
import numpy as np
from PIL import Image
import evidence as ev,observation_pool as op,parallel_repairs as pr
from transport.common import read_json


def row(rid,points,parent=None):
    return dict(id=rid,parent_id=parent or rid,kind='straight',start=points[0],end=points[1],via=[])


def test_shared_context_does_not_darken_duplicate_masks_or_change_target(tmp_path):
    image=Image.new('RGB',(100,100),'white');view=ev.identity_view(image.size)
    m=np.zeros((100,100),dtype='uint8');m[45:48,10:90]=255
    source=tmp_path/'native.png';Image.fromarray(m).save(source);before=source.read_bytes()
    native=dict(path=str(source),view=view,observation='G2',id='M1',bbox_global=[10,45,90,48])
    own=np.zeros_like(m,bool);own[45:48,10:40]=True
    arrays=[]
    for name,pool in [('one',[native]),('twice',[native,{**native,'id':'M2'}])]:
        out=tmp_path/name;(out/'input').mkdir(parents=True);image.save(out/'input/proposals.png')
        op.overlay(image,view,[dict(id='M1',mask=own)],'G1',pool,out)
        arrays.append(np.array(Image.open(out/'input/proposals.png')))
    assert np.array_equal(*arrays) and source.read_bytes()==before
    assert np.all(arrays[0][46,20]==255) and not np.all(arrays[0][46,60]==255)


def test_task_bundling_preserves_separate_foci_without_guessing_a_junction():
    tasks=[dict(bbox=[0,0,20,20],description='left'),dict(bbox=[30,0,50,20],description='right')]
    assert len(pr.bundle(tasks,[]))==2
    rows=[row('a',[[0,10],[15,10]],'network'),row('b',[[35,10],[50,10]],'network')]
    grouped=pr.bundle(tasks,rows)
    assert len(grouped)==1 and len(grouped[0]['focus_regions'])==2
    assert [x['description'] for x in grouped[0]['focus_regions']]==['left','right']


def test_independent_repairs_use_same_snapshot_and_run_concurrently(tmp_path,monkeypatch):
    gate=threading.Barrier(2);seen=[]
    existing=[row('old',[[0,90],[90,90]])]
    def recover(clean,task,current,pid,folder,*args,**kw):
        seen.append([r['id'] for r in current]);gate.wait(timeout=3)
        return [],dict(group=pid,replace=[],rows=[row(pid+':new',[[task['bbox'][0],10],[task['bbox'][2],10]])],state='completed')
    monkeypatch.setattr(pr.repair,'recover',recover)
    tasks=[dict(bbox=[0,0,20,20],description='one'),dict(bbox=[50,0,70,20],description='two')]
    patches,issues=pr.execute(None,tasks,existing,'P',tmp_path,dict(task_workers=2),None,{})
    assert len(patches)==2 and not issues and seen==[['old'],['old']]
    assert existing==[row('old',[[0,90],[90,90]])]


def test_actual_id_write_conflict_returns_to_model_with_committed_state(tmp_path,monkeypatch):
    seen=[]
    def recover(clean,task,current,pid,folder,*args,**kw):
        seen.append((Path(folder).name,[r['id'] for r in current]))
        # A model can discover a needed replacement outside its original focus.
        return [],dict(group=pid,replace=[current[0]['id']],rows=[row(pid+':new',[[0,90],[90,90]])],state='completed',repair_calls=1,geometry_calls=1,head_calls=1)
    monkeypatch.setattr(pr.repair,'recover',recover)
    tasks=[dict(bbox=[0,0,20,20],description='one'),dict(bbox=[50,0,70,20],description='two')]
    patches,issues=pr.execute(None,tasks,[row('old',[[0,90],[90,90]])],'P',tmp_path,dict(task_workers=2),None,{})
    assert not issues and len(seen)==3 and seen[-1]==('rebase',['P1:new'])
    assert patches[-1]['replace']==['P1:new'] and patches[-1]['repair_calls']==2
    assert read_json(tmp_path/'parallel_summary.json')['write_conflicts'][0]['resolved_by']=='fresh_model_decision'
