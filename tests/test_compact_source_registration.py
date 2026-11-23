from copy import deepcopy
import numpy as np
from PIL import Image,ImageDraw
from shape_integration.fitting.source_boxes import register_compact
from shape_integration.fitting.paths import box_commands


def fixture():
    clean=Image.new('RGB',(150,220),'white');draw=ImageDraw.Draw(clean);parts=[]
    for i,y in enumerate((20,70,130,180)):
        draw.rectangle((100,y,119,y+19),fill='#E4A3D3',outline='#A04B8D',width=2)
        p={'center':[79.5,y+9.5],'width':19,'height':19,'angle_radians':0.}
        parts.append({'id':str(i),'kind':'rectangle','status':'ok','parameters':p,
            'bbox_2d':[70,y,89,y+19],'diagnostics':{},'commands':box_commands([*p['center'],19,19,0.])})
    return {'parts':parts},{'frame':{'source_crop_half_open':[0,0,150,220],'actual_crop_input_size_px':[150,220]}},clean


def test_compact_column_measures_source_edges_after_bounded_translation():
    local,job,source=fixture();before=deepcopy(local);audit=register_compact(local,job,source)
    assert sum(x.get('applied',False) for x in audit)==4
    assert all(99<part['bbox_2d'][0]<102 for part in local['parts'])
    assert all(x['reason']=='four_source_stroke_centerlines' for x in audit if x['applied'])
    assert before!=local


def test_missing_source_and_two_color_columns_do_not_invent_geometry():
    local,job,source=fixture();before=deepcopy(local)
    assert register_compact(local,job,Image.new('RGB',source.size,'white'))==[] and local==before
    draw=ImageDraw.Draw(source)
    for y in (20,70,130,180):draw.rectangle((50,y,69,y+19),fill='#9BC5E0',outline='#347AA0',width=2)
    register_compact(local,job,source)
    assert local==before
    local,job,source=fixture();before=deepcopy(local)
    assert register_compact(local,job,source.convert('L').convert('RGB'))==[] and local==before
    local,job,source=fixture()
    for p in local['parts']:
        p['bbox_2d'][0]-=55;p['bbox_2d'][2]-=55;p['parameters']['center'][0]-=55
    before=deepcopy(local);register_compact(local,job,source)
    assert local==before


def test_count_conflict_corrects_existing_identities_without_inventing_a_member():
    local,job,source=fixture();local['parts'].pop(1)
    ids=[p['id'] for p in local['parts']];audit=register_compact(local,job,source)
    assert any(a.get('reason')=='compact_source_count_conflict' and
               a['source_components']==4 and a['observed_parts']==3 for a in audit)
    assert [p['id'] for p in local['parts']]==ids
    assert sum(a.get('applied',False) for a in audit)==3
    assert np.allclose([p['parameters']['center'][1] for p in local['parts']],[29.5,139.5,189.5],atol=1)


def test_open_sides_gray_background_and_large_displacement_abstain():
    local,job,source=fixture();before=deepcopy(local)
    draw=ImageDraw.Draw(source)
    for y in (20,70,130,180):draw.rectangle((100,y,119,y+2),fill='white')
    register_compact(local,job,source)
    assert local==before


def test_count_conflict_survives_scene_assembly_as_incomplete(tmp_path):
    from shape_integration.common import save
    from shape_integration.fitting.scene import assemble
    fit={'object_id':'o','coordinate_frame':'original_pixel_centers','observation_status':'ok',
         'fit_status':'ok','frame':{},'completeness_status':'ok','shared_point_diagnostics':{},'parts':[],'points':[]}
    save(tmp_path/'objects/o/revision_00/fit_original.json',fit)
    conflict={'reason':'compact_source_count_conflict','observed_parts':3,'source_components':4}
    record={'o':{'state':'accepted_by_model','selected_revision':'revision_00','history':[],
                 'source_registration_conflicts':[conflict]}}
    item={'id':'case','source_size':[200,200],'source_sha256':'source'}
    scene=assemble(tmp_path,item,[{'id':'o','origin':'discovery'}],record,{},[])
    assert scene['objects']['o']['completeness_status']=='source_count_unresolved'
    assert scene['objects']['o']['source_registration_conflicts']==[conflict]
    assert any(r['state']=='source_count_unresolved' for r in scene['unresolved'])
