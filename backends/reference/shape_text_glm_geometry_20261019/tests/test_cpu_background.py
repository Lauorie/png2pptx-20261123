from copy import deepcopy
import json
import subprocess
import threading

import numpy as np
from PIL import Image

from shape_integration.texts.background_selection import score_pair, select, batches, roi


def row(ident='T1', x=10, angle=0):
    return {'id':ident,'bbox_2d':[x,10,x+20,20], 'frame_px':[x,10,20,10],
            'rotation_deg':angle,'text':'and','background_fill':'#FF0000',
            'background_evidence':'stale LLM output'}


def test_same_roi_white_transparent_and_tie():
    white=Image.new('RGB',(40,30),'white')
    pixels=np.asarray(white).copy();pixels[14:16,:]=(80,0,0)
    line=Image.fromarray(pixels);box=[5,5,35,25]
    assert score_pair(white,line,white,box)['selected']=='white'
    assert score_pair(line,line,white,box)['selected']=='transparent'
    assert score_pair(white,white,white,box)['selected']=='transparent'
    assert score_pair(line,line,white,box)['transparent_score']==1


def test_rotated_footprints_not_batched_together():
    rows=[row(),row('T2',18,90),row('T3',65)]
    groups=batches(rows,[100,60])
    assert len(groups)==2
    assert [r[0] for r in groups[0]]==['T1','T3']
    assert roi(rows[1],[100,60])[1]<rows[1]['bbox_2d'][1]


def test_no_base_clears_cached_fill_without_model_or_render(tmp_path):
    source=tmp_path/'source.png';Image.new('RGB',(100,60),'white').save(source)
    original=[row()];snapshot=deepcopy(original)
    result=select(original,source,tmp_path/'result',None,{})
    assert original==snapshot
    assert result[0]['background_fill'] is None
    assert not result[0]['background_selection']['identifiable']
    report=json.loads((tmp_path/'result/selection.json').read_text())
    assert report['renders']==0 and report['base_sha256'] is None


def test_failed_candidate_keeps_transparent(tmp_path,monkeypatch):
    from shape_integration.texts import export, equations
    source=tmp_path/'source.png';Image.new('RGB',(100,60),'white').save(source)
    base=tmp_path/'base.pptx';base.write_bytes(b'test')
    monkeypatch.setattr(export,'write',lambda *a: None)
    def fail(*a):raise subprocess.TimeoutExpired('libreoffice',180)
    monkeypatch.setattr(equations,'render_preview',fail)
    result=select([row()],source,tmp_path/'result',threading.BoundedSemaphore(1),
                  {'render_timeout_seconds':180},base)
    assert result[0]['background_fill'] is None
    assert result[0]['background_selection']['reason']=='candidate_render_failed'


def test_disjoint_batch_selects_each_object_independently(tmp_path,monkeypatch):
    from shape_integration.texts import export, equations
    pixels=np.full((60,100,3),255,dtype=np.uint8);pixels[15:17,:]=(180,0,0)
    reference=pixels.copy();reference[10:20,10:30]=255
    source=tmp_path/'source.png';Image.fromarray(reference).save(source)
    base=tmp_path/'base.pptx';base.write_bytes(b'test')
    trials={}
    def write(rows,size,target,base):trials[str(target)]=deepcopy(rows)
    def render(target,output,size,timeout):
        image=pixels.copy()
        for r in trials[str(target)]:
            if r.get('background_fill'):
                x,y,w,h=r['frame_px'];image[y:y+h,x:x+w]=255
        output.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(image).save(output.parent/'native.png')
    monkeypatch.setattr(export,'write',write)
    monkeypatch.setattr(equations,'render_preview',render)
    rows=[row(),row('T2',65)]
    result=select(rows,source,tmp_path/'result',threading.BoundedSemaphore(1),
                  {'render_timeout_seconds':180},base)
    assert [r['background_fill'] for r in result]==['#FFFFFF',None]
    assert len(trials)==2
    assert [r['text'] for r in result]==[r['text'] for r in rows]
