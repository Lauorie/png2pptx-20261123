import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import critic_context_ablation as experiment
import geometry_critic


def document():
    return dict(nodes=[['n1',1.,2.],['n2',10.,12.]],routes=[dict(id='R1',paths=[dict(id='p1',segments=[dict(kind='line',nodes=['n1','n2'])])])])


def test_original_control_is_exact():
    original=geometry_critic.prompt(document(),[100,80]);saved=copy.deepcopy(original)
    q,images=experiment.build(original,Path('/tmp/input'),'A_local_current',{})
    assert q==original==saved
    assert len(images)==2


def test_context_changes_only_context_fields():
    original=geometry_critic.prompt(document(),[100,80]);ctx=dict(size=[500,400],clean_bbox=[20,30,80,90])
    for local,context in [('A_local_current','B_context_current'),('C_local_structural','D_context_structural')]:
        a,ai=experiment.build(original,Path('/tmp/input'),local,ctx)
        b,bi=experiment.build(original,Path('/tmp/input'),context,ctx)
        assert b.pop('context')==ctx
        assert len(b['instructions'])==len(a['instructions'])+1
        b['instructions'].pop()
        assert b==a and bi[:2]==ai and len(bi)==3


def test_structural_variant_keeps_coordinates_and_protocol():
    original=geometry_critic.prompt(document(),[100,80]);q=experiment.structural_prompt(original)
    for key in ('size','current','output_schema'):assert q[key]==original[key]
    assert q['task']!=original['task']


def test_global_bbox_uses_pixel_boundaries():
    # Local image is a 2x upsampled crop whose global origin is (100, 200).
    view=dict(global_size=[1000,800],view_to_global=[[.5,0,99.75],[0,.5,199.75],[0,0,1]])
    frame=dict(bbox=[20,40,60,80])
    assert np.allclose(experiment.global_crop_bbox(frame,view,[500,400]),[55,110,65,120])


def test_context_not_used_as_output_frame():
    original=geometry_critic.prompt(document(),[100,80])
    q,_=experiment.build(original,Path('/tmp/input'),'D_context_structural',dict(size=[1080,720],clean_bbox=[1,2,5,6]))
    assert q['size']==[100,80]
    assert 'never CONTEXT coordinates' in q['instructions'][-1]


def test_roundtrip_and_accept_preserve_points():
    doc=document();frame=dict(bbox=[3,4,30,40],scale=[2.4,3.1])
    moved=geometry_critic.transform(geometry_critic.transform(doc,frame),frame,inverse=True)
    assert np.allclose([n[1:] for n in doc['nodes']],[n[1:] for n in moved['nodes']])
    assert geometry_critic.validate(dict(decision='accept'),[100,80])==dict(decision='accept')


def test_boundary_followup_is_single_instruction_change():
    original=geometry_critic.prompt(document(),[100,80])
    c,ci=experiment.build(original,Path('/tmp/input'),'C_local_structural',{})
    e,ei=experiment.build(original,Path('/tmp/input'),'E_local_structural_boundary',{})
    assert 'touching strokes' in e['instructions'].pop()
    assert e==c and ei==ci


def test_tolerance_followup_keeps_original_endpoint_checks():
    original=geometry_critic.prompt(document(),[100,80])
    a,ai=experiment.build(original,Path('/tmp/input'),'A_local_current',{})
    f,fi=experiment.build(original,Path('/tmp/input'),'F_local_current_tolerance',{})
    assert 'Accept small pixel offsets' in f['instructions'].pop()
    assert f==a and fi==ai
