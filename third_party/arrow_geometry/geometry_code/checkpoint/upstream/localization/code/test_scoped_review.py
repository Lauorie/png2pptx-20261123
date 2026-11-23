import copy
import pytest
from PIL import Image
import scoped_review as S


def row(key, box):
    return dict(id=key,bbox=box,render_type='linework',grouping='group',label=key,extent='complete')


def test_scope_control_cannot_commit_edits_or_unknown_neighbor_ids():
    focus=[row('a',[0,0,20,20])]; nearby=[row('b',[22,0,30,20])]
    raw=dict(frame='LOCAL',decision='include',include_ids=['b'],reason='same visible bundle',replacements=[{'bad':'speculative edit'}])
    assert 'replacements' not in S.validate_scope(raw,(100,100),focus,nearby)
    assert 'replacements' in raw
    with pytest.raises(ValueError): S.validate_scope({**raw,'include_ids':['missing']},(100,100),focus,nearby)


def test_scope_follows_parallel_replacements_and_keeps_requested_scope_whole():
    rows=[row(str(i),[i*30,0,i*30+20,20]) for i in range(9)]
    original=copy.deepcopy(rows)
    tasks,deferred=S.choose_jobs(rows,[{'kind':'included_scope','ids':['old','8']}],{'old':[str(i) for i in range(8)]},set(),6,False)
    assert len(tasks)==1 and len(tasks[0]['ids'])==9 and not deferred
    assert rows==original
    tasks,deferred=S.choose_jobs(rows,[{'kind':'included_scope','ids':['0','1']},{'kind':'change_check','ids':['1','2']}],{},set(),6,False)
    assigned=[key for j in tasks for key in j['ids']]
    assert len(assigned)==len(set(assigned)) and len(deferred)==1


def test_live_include_resolves_parallel_edit_then_rechecks_merge(tmp_path,monkeypatch):
    cid='synthetic'
    inp=tmp_path/'inputs'/cid;inp.mkdir(parents=True)
    Image.new('RGB',(400,400),'white').save(inp/'source.png')
    monkeypatch.setattr(S,'latest',lambda *a:None)
    rows=[row('a',[200,200,500,500]),row('b',[520,200,600,500])]
    seen=[]
    class Model:
        def ask(self,folder,q,images,validator):
            focus=q['candidates']; ids=[r['id'] for r in focus];seen.append(ids)
            answer=dict(frame='LOCAL',decision='edit',keep_ids=ids,replacements=[],discard=[],inspect=[])
            if ids==['a']:
                answer=dict(frame='LOCAL',decision='include',include_ids=['b'],reason='nearby continuation')
            elif ids==['b']:
                answer['keep_ids']=[];answer['replacements']=[dict(source_ids=ids,regions=[{**focus[0],'id':'new'}],reason='bbox adjustment')]
            elif len(ids)==2:
                box=[min(r['bbox'][i] for r in focus) if i<2 else max(r['bbox'][i] for r in focus) for i in range(4)]
                answer['keep_ids']=[];answer['replacements']=[dict(source_ids=ids,regions=[row('whole',box)],reason='manageable related bundle')]
            return validator(answer)
    result,issues=S.run(tmp_path,cid,rows,Model(),dict(local_review_rounds=3,local_workers=2),lambda stage:None)
    assert not issues and len(result)==1
    assert any(len(ids)==2 and 'b' not in ids and 'a' in ids for ids in seen)
    assert len(seen)==4  # two first requests, joined scope, post-merge check
