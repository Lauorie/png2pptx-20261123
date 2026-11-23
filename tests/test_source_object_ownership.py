"""Source identities survive geometric errors without collapsing real parallels."""
import copy
import numpy as np
import pytest
from PIL import Image, ImageDraw

from full_recovery.source_ownership import scopes, validate_inventory, validate_mapping, select_mapping, summary
from full_recovery.quality_choice import apply_choices
from full_recovery.image_edit import source_locked_icon
from shape_integration.styling.native_stroke import fit_profiles
from shape_integration.fitting.source_boxes import register
from shape_integration.fitting.scene import assemble
from shape_integration.fitting.paths import box_commands
from shape_integration.common import save


def route(ident, branch, y):
    points = np.c_[np.linspace(20, 140, 60), np.full(60, y)]
    return {'id': ident, 'key': ident, 'keys': [ident] if branch == 'shape' else [], 'layer': branch, 'points': points,
            'length': 120., 'refs': [], 'style': {'stroke': '#000000', 'width_px': 1}}


def source(ident, kind='line'):
    return {'id': ident, 'kind': kind, 'bbox': [20, 20, 140, 23],
            'visible_count': 1, 'description': 'visible horizontal connection'}


def test_source_identity_can_replace_displaced_fit_and_keep_real_parallel():
    rows = [route('shape:wrong', 'shape', 38), route('arrow:upper', 'arrows', 30),
            route('arrow:lower', 'arrows', 45)]
    regions, outside = scopes(rows, (180, 100))
    assert len(regions) == 1 and not outside
    aliases = {'C1': rows[0], 'C2': rows[1], 'C3': rows[2]}
    objects = [source('S1'), source('S2')]
    mapping = {'objects': [
        {'id': 'S1', 'candidate_ids': ['C1', 'C2'], 'keep_ids': ['C2'],
         'status': 'resolved', 'reason': 'same source occurrence despite eight pixel displacement'},
        {'id': 'S2', 'candidate_ids': ['C3'], 'keep_ids': ['C3'],
         'status': 'resolved', 'reason': 'separate physical parallel'}], 'unassigned': []}
    validate_mapping(mapping, objects, aliases, {})
    choices = [{'candidate_ids': [aliases[k]['id'] for k in obj['candidate_ids']],
                'keep_ids': [aliases[k]['id'] for k in obj['keep_ids']], 'reason': obj['reason']}
               for obj in mapping['objects']]
    result = apply_choices(rows, choices)
    assert result['drop_shape_keys'] == ['shape:wrong']
    assert not result['drop_arrow_names']


def test_every_native_and_missing_source_need_explicit_receipts():
    aliases = {'C1': route('shape:x', 'shape', 20)}
    missing = {'objects': [{'id': 'S1', 'candidate_ids': [], 'keep_ids': [],
                            'status': 'incomplete', 'reason': 'no recovery'}],
               'unassigned': [{'candidate_id': 'C1', 'classification': 'uncertain', 'reason': 'unresolved fit'}]}
    validate_mapping(missing, [source('S1')], aliases, {})
    invalid = copy.deepcopy(missing); invalid['objects'][0]['status'] = 'resolved'
    with pytest.raises(ValueError, match='incomplete'):
        validate_mapping(invalid, [source('S1')], aliases, {})
    invalid = copy.deepcopy(missing); invalid['unassigned'] = []
    with pytest.raises(ValueError, match='Every supplied'):
        validate_mapping(invalid, [source('S1')], aliases, {})


def test_uncertainty_and_image_content_cannot_delete_external_connections():
    aliases = {'C1': route('shape:x', 'shape', 20)}
    mapping = {'objects': [{'id': 'S1', 'candidate_ids': ['C1'], 'keep_ids': [],
                           'status': 'resolved', 'reason': 'inside icon'}], 'unassigned': []}
    obj = {**source('S1', 'image_content'), 'asset_id': 'I1'}
    with pytest.raises(ValueError, match='external native connection'):
        validate_mapping(mapping, [obj], aliases, {'I1': {'source_box_half_open': [60, 10, 100, 40]}})
    mapping['objects'][0]['status'] = 'uncertain'
    with pytest.raises(ValueError, match='uncertainty'):
        validate_mapping(mapping, [source('S1')], aliases, {})


def test_invalid_image_decision_preserves_only_its_native_content_and_keeps_valid_neighbours():
    aliases={k:route(k,'shape',y) for k,y in [('C1',20),('C2',35),('C3',33)]}
    objects=[{**source('S1','image_content'),'asset_id':'I1'},source('S2')]
    raw={'objects':[{'id':'S1','candidate_ids':['C1'],'keep_ids':[],
            'status':'resolved','reason':'incorrect image ownership'},
        {'id':'S2','candidate_ids':['C2','C3'],'keep_ids':['C3'],
            'status':'resolved','reason':'valid independent source choice'}],'unassigned':[]}
    result=validate_mapping(raw,objects,aliases,{'I1':{'source_box_half_open':[60,10,100,40]}},allow_omissions=True)
    assert result['objects'][0]['keep_ids']==['C1']
    assert result['objects'][0]['status']=='uncertain'
    assert result['objects'][1]['keep_ids']==['C3'] and result['objects'][1]['status']=='resolved'
    assert len(result['mapping_normalizations'])==1


def test_invalid_native_relations_preserve_real_objects_without_accepting_invented_ids():
    aliases={k:route(k,'shape',20) for k in ('C1','C2','C3','C4')}
    raw={'objects':[{'id':'S1','candidate_ids':['C1'],'keep_ids':['C2','invented',{}],
        'status':'resolved','reason':'invalid IDs'},
        {'id':'S2','candidate_ids':['C3','C4'],'keep_ids':['C4'],'status':'resolved','reason':'valid'}],
        'unassigned':[]}
    result=validate_mapping(raw,[source('S1'),source('S2')],aliases,{},allow_omissions=True)
    assert set(result['objects'][0]['keep_ids'])=={'C1','C2'}
    assert result['objects'][0]['status']=='uncertain'
    assert result['objects'][1]['keep_ids']==['C4']


def test_shared_native_conflict_keeps_content_and_marks_both_sources_incomplete():
    aliases={k:route(k,'shape',20) for k in ('C1','C2')}
    raw={'objects':[{'id':'S1','candidate_ids':['C1','C2'],'keep_ids':['C2'],
        'status':'resolved','reason':'first'},
        {'id':'S2','candidate_ids':['C1'],'keep_ids':['C1'],'status':'resolved','reason':'second'}],
        'unassigned':[]}
    result=validate_mapping(raw,[source('S1'),source('S2')],aliases,{},allow_omissions=True)
    assert all('C1' in row['keep_ids'] and row['status']=='incomplete' for row in result['objects'])


def test_inventory_uses_valid_source_pixels_and_registered_assets():
    validate_inventory({'objects': [source('S1')]}, (180, 100), set())
    invalid = {**source('S1', 'image_content'), 'asset_id': 'invented'}
    with pytest.raises(ValueError, match='existing asset'):
        validate_inventory({'objects': [invalid]}, (180, 100), {'I1'})


def test_partial_unregistered_image_owner_preserves_content_without_losing_valid_sources():
    import copy
    invalid={**source('S2','image_content'),'asset_id':'invented'}
    observed=validate_inventory({'objects':[source('S1'),invalid]},(180,100),set(),allow_partial=True)
    assert observed['objects'][0]['kind']=='line'
    assert observed['objects'][1]['kind']=='uncertain' and observed['inventory_normalizations']
    aliases={k:route(k,'shape',20) for k in ('C1','C2','C3')}
    raw={'objects':[{'id':'S1','candidate_ids':['C1','C2'],'keep_ids':['C2'],
        'status':'resolved','reason':'complete same source connection'}],
        'unassigned':[{'candidate_id':'C3','classification':'image_owned','asset_id':'invented','reason':'pixels'}]}
    with pytest.raises(ValueError,match='registered asset'):
        validate_mapping(copy.deepcopy(raw),[source('S1')],aliases,{})
    result=validate_mapping(raw,[source('S1')],aliases,{},allow_omissions=True)
    assert result['objects'][0]['keep_ids']==['C2']
    assert result['unassigned'][0]['classification']=='uncertain' and result['mapping_normalizations']
    assert validate_mapping(copy.deepcopy(result),[source('S1')],aliases,{},allow_omissions=True)==result


def test_dense_source_counts_can_abstain_instead_of_inventing_a_network_template():
    uncertain = {**source('S1', 'bundle'), 'visible_count': None, 'count_status': 'uncertain', 'members': []}
    validate_inventory({'objects': [uncertain]}, (180, 100), set())
    invented = {**uncertain, 'visible_count': 49}
    with pytest.raises(ValueError, match='null'):
        validate_inventory({'objects': [invented]}, (180, 100), set())
    exact = {**uncertain, 'visible_count': 3, 'count_status': 'exact',
             'members': ['top visible route', 'middle visible route', 'bottom visible route']}
    validate_inventory({'objects': [exact]}, (180, 100), set())
    exact['members'] = []
    with pytest.raises(ValueError, match='explicit source description'):
        validate_inventory({'objects': [exact]}, (180, 100), set())
    aliases = {'C1': route('shape:x', 'shape', 20)}
    mapping = {'objects': [{'id': 'S1', 'candidate_ids': ['C1'], 'keep_ids': ['C1'],
                           'status': 'resolved', 'reason': 'usable survivor'}], 'unassigned': []}
    with pytest.raises(ValueError, match='remains incomplete'):
        validate_mapping(mapping, [uncertain], aliases, {})
    mapping['objects'][0]['status'] = 'incomplete'
    validate_mapping(mapping, [uncertain], aliases, {})
    partial = validate_inventory({'objects': [{**source('S2', 'bundle'), 'visible_count': 3, 'members': []}]},
                                 (180, 100), set(), allow_partial=True)
    assert partial['objects'][0]['visible_count'] is None
    assert partial['objects'][0]['count_status'] == 'uncertain'
    assert partial['objects'][0]['source_count_normalization']['proposed_count'] == 3


def test_horizontal_source_locators_and_canvas_edge_conventions_do_not_trigger_model_retries():
    row = {**source('S1'), 'bbox': [0, 20, 180, 20]}
    result = validate_inventory({'objects': [row]}, (180, 100), set())
    assert result['objects'][0]['bbox'] == [0, 20, 179, 20]
    assert result['objects'][0]['source_bbox_normalization']['reason'] == 'one_pixel_canvas_edge_convention'
    assert validate_inventory(result, (180, 100), set()) == result
    reversed_box = {**source('S2'), 'bbox': [140, 23, 20, 20]}
    assert validate_inventory({'objects': [reversed_box]}, (180, 100), set())['objects'][0]['bbox'] == [20, 20, 140, 23]


def test_partial_model_receipts_preserve_omissions_as_explicit_uncertainty():
    aliases = {'C1': route('shape:a', 'shape', 20), 'C2': route('shape:b', 'shape', 30)}
    objects = [source('S1'), source('S2')]
    result = validate_mapping({'objects': [{'id': 'S1', 'candidate_ids': ['C1'], 'keep_ids': ['C1'],
        'status': 'resolved', 'reason': 'source supported'}], 'unassigned': []}, objects, aliases, {}, allow_omissions=True)
    assert result['objects'][1]['status'] == 'incomplete'
    assert result['unassigned'][0]['candidate_id'] == 'C2'
    assert result['unassigned'][0]['classification'] == 'uncertain'
    assert validate_mapping(result, objects, aliases, {}) == result
    redundant = {'objects': [{'id': 'S1', 'candidate_ids': ['C1'], 'keep_ids': ['C1'],
        'status': 'resolved', 'reason': 'source supported'}], 'unassigned': [{'candidate_id': 'C1',
        'classification': 'context', 'reason': 'listed again as context'}]}
    normalized = validate_mapping(redundant, [source('S1')], {'C1': aliases['C1']}, {}, allow_omissions=True)
    assert not normalized['unassigned'] and normalized['mapping_normalizations']
    assert validate_mapping(normalized, [source('S1')], {'C1': aliases['C1']}, {}) == normalized


def test_residual_batches_revisit_displaced_natives_and_preserve_one_source_survivor(tmp_path):
    import json
    aliases = {'C%d' % i: route('shape:%d' % i, 'shape', 180) for i in range(1, 99)}
    objects = [{**source('S1'), 'bbox': [20, 10, 140, 12]}]
    prompt = {'source_objects': objects, 'candidates': [summary(r, k, [0, 0, 600, 200], np.ones(2)) for k, r in aliases.items()],
              'rules': [], 'output': {'objects': [{}]}}
    class Mapper:
        residual = 0
        def call(self, folder, tag, system, payload, images, validator, budget):
            json.dumps(payload, allow_nan=False)  # Actual wire payload, including source-detail windows.
            ids = [r['id'] for r in payload['candidates']]
            keep = []; dropped = []; status = 'incomplete'
            if 'prior_source_selection' in payload:
                self.residual += 1
                prior = payload['prior_source_selection'][0]['kept_ids']
                if self.residual <= 2:
                    keep = ids[:1]; dropped = prior
                status = 'resolved'
            return validator({'objects': [{'id': 'S1', 'candidate_ids': ids, 'keep_ids': keep,
                    'drop_prior_ids': dropped, 'status': status, 'reason': 'source occurrence inspected'}], 'unassigned': []})
    mapper = Mapper()
    clean = tmp_path/'clean.png'; Image.new('RGB', (600, 200), 'white').save(clean)
    mapped, calls = select_mapping(mapper, tmp_path, prompt, [('CLEAN', clean)], objects, aliases, {}, [0, 0, 600, 200], np.ones(2))
    assert calls == 4 and mapper.residual == 3
    assert mapped['objects'][0]['keep_ids'] == ['C49']
    assert len(mapped['objects'][0]['candidate_ids']) == 98
    validate_mapping(mapped, objects, aliases, {})


def test_native_route_shared_across_buckets_gets_one_complete_source_ledger():
    left = route('shape:left', 'shape', 40); left['points'][:, 0] += 70
    right = route('shape:right', 'shape', 40); right['points'][:, 0] += 230
    whole = route('arrow:whole', 'arrows', 40)
    whole['points'][:, 0] = np.linspace(90, 370, len(whole['points']))
    regions, outside = scopes([left, right, whole], (600, 200))
    assert len(regions) == 1 and not outside
    assert {r['id'] for r in regions[0]['rows']} == {'shape:left', 'shape:right', 'arrow:whole'}


def test_cleanup_cannot_redesign_fills_edges_or_move_an_icon():
    original = Image.new('RGB', (80, 80), 'white'); draw = ImageDraw.Draw(original)
    draw.ellipse((10, 10, 69, 69), fill='#CCCCCC', outline='black', width=2)
    draw.line((15, 35, 65, 35), fill='black', width=1)
    original_pixels = np.asarray(original)
    result, audit = source_locked_icon(original, Image.new('RGBA', (160, 160), 'red'),
                                        [-10, -10, 100, 100],
                                        {'native_labels': [{'box': [32, 45, 40, 50]}]})
    actual = np.asarray(result)
    unchanged = np.ones((80, 80), bool); unchanged[44:52, 31:42] = False
    assert np.array_equal(actual[:, :, :3][unchanged], original_pixels[unchanged])
    assert tuple(actual[35, 20]) == (0, 0, 0, 255)
    assert result.size == original.size and audit['source_pixels_preserved']


def profiles(coverage, offset):
    offsets = np.linspace(-8, 8, 161)
    alpha = .4 * np.maximum(0, 1 - np.abs(offsets-offset)/.6)
    rgb = np.full((100, len(offsets), 3), 255.)
    rgb[:coverage] -= alpha[None, :, None] * 255
    return fit_profiles(rgb, offsets, np.zeros(rgb.shape[:2]), np.zeros(100, bool), [0, 0, 0])


def test_displaced_weak_fragments_do_not_become_a_reliable_hairline():
    result = profiles(25, 4)
    assert result['measured'] and not result['width_reliable']
    assert 'weak_or_displaced_colour_fragments' in result['width_reliability_reasons']


def test_aligned_faint_full_stroke_is_not_rejected_by_a_width_floor():
    result = profiles(100, 0)
    assert result['measured'] and result['width_reliable']
    assert result['width_px'] < 1


def repeated_scene(count=3):
    clean = Image.new('RGB', (200, 260), 'white'); draw = ImageDraw.Draw(clean)
    parts = []
    for i, y in enumerate((35, 105, 190)):
        draw.rectangle((40, y, 159, y+34), fill='#F0C8A0', outline='#804020', width=2)
        if i >= count:
            continue
        p = {'center': [99.5, y+17+12], 'width': 119., 'height': 34., 'angle_radians': 0.}
        parts.append({'id': str(i), 'status': 'ok', 'kind': 'rectangle', 'parameters': p,
                      'bbox_2d': [40, y+12, 159, y+46], 'diagnostics': {},
                      'commands': box_commands([*p['center'], p['width'], p['height'], 0.], 0)})
    job = {'frame': {'source_crop_half_open': [0, 0, 200, 260], 'actual_crop_input_size_px': [200, 260]}}
    return {'parts': parts}, job, clean


def test_repeated_boxes_follow_source_irregular_spacing_instead_of_a_grid():
    local, job, clean = repeated_scene()
    audit = register(local, job, clean)
    assert sum(bool(a['applied']) for a in audit) == 3
    ys = [p['parameters']['center'][1] for p in local['parts']]
    assert np.allclose(ys, [52, 122, 207], atol=1)
    assert abs((ys[2]-ys[1])-(ys[1]-ys[0])-15) < 1


def test_repeated_box_identity_abstains_when_source_count_differs():
    local, job, clean = repeated_scene()
    ImageDraw.Draw(clean).rectangle((40, 190, 159, 224), fill='white')
    previous = copy.deepcopy(local)
    audit = register(local, job, clean)
    assert audit[0]['reason'] == 'source_primitive_count_ambiguous'
    assert local == previous


def test_source_owned_boxes_measure_visible_sides_under_foreground_overlays():
    local, job, clean = repeated_scene()
    # Foreground arrows/symbols obscure part of the sides without changing the
    # independent coloured source component or authorizing a closed frame.
    draw = ImageDraw.Draw(clean)
    for y in (35, 105, 190):
        draw.rectangle((145, y+11, 170, y+23), fill='#3060C0')
    audit = register(local, job, clean)
    assert sum(bool(a['applied']) for a in audit) == 3
    assert any(s['source_visible_fill_fraction'] < .8 for a in audit for s in a['sides'])
    assert np.allclose([p['parameters']['center'][1] for p in local['parts']], [52, 122, 207], atol=1)


def test_unverified_partial_coverage_proposal_is_not_published_as_a_frame(tmp_path):
    fit = {'object_id': 'o', 'coordinate_frame': 'original_pixel_centers',
           'observation_status': 'partial', 'fit_status': 'ok', 'frame': {},
           'completeness_status': 'partial', 'shared_point_diagnostics': {}, 'parts': [], 'points': []}
    save(tmp_path/'objects/o/revision_00/fit_original.json', fit)
    item = {'id': 'case', 'source_size': [200, 200], 'source_sha256': 'original'}
    record = {'o': {'state': 'unresolved', 'selected_revision': 'revision_00', 'history': [],
                    'stop_reason': 'context_budget_exhausted'}}
    coverage = assemble(tmp_path, item, [{'id': 'o', 'origin': 'fitting_coverage'}], record, {}, [])
    assert coverage['objects']['o']['export_excluded_reason'] == 'unverified_crop_fragment'
    existing = assemble(tmp_path, item, [{'id': 'o', 'origin': 'discovery'}], record, {}, [])
    assert 'export_excluded_reason' not in existing['objects']['o']
    assert existing['objects']['o']['fit_status'] == 'ok'


def test_uncertain_residual_preserves_only_its_candidates_not_previous_duplicates(tmp_path):
    aliases={'C%d'%i:route('shape:%d'%i,'shape',20) for i in range(1,66)}
    objects=[source('S1')]
    prompt={'source_objects':objects,'candidates':[summary(r,k,[0,0,200,100],np.ones(2))
        for k,r in aliases.items()],'rules':[],'output':{'objects':[{}]}}
    class Mapper:
        def call(self,folder,tag,system,payload,images,validator,budget):
            ids=[r['id'] for r in payload['candidates']]
            residual='prior_source_selection' in payload
            return validator({'objects':[{'id':'S1','candidate_ids':ids,
                'keep_ids':ids if residual else ids[:1],
                'status':'uncertain' if residual else 'resolved',
                'reason':'Residual source identity unclear' if residual else 'Verified one physical source and survivor'}],
                'unassigned':[]})
    mapped,calls=select_mapping(Mapper(),tmp_path,prompt,[],objects,aliases,{},[0,0,200,100],np.ones(2))
    assert calls==2
    result=mapped['objects'][0]
    assert result['keep_ids']==['C1','C65']
    assert len(result['candidate_ids'])==65
    assert result['status']=='incomplete'
    assert [r['status'] for r in result['batch_review_statuses']]==['resolved','uncertain']
    assert validate_mapping(mapped,objects,aliases,{})==mapped


def test_later_resolved_batch_does_not_hide_earlier_uncertainty():
    from full_recovery.source_ownership import merge_mapping_receipt
    target={'id':'S1','candidate_ids':['C1'],'keep_ids':['C1'],
        'status':'uncertain','reason':'First native identity unclear'}
    merge_mapping_receipt(target,{'id':'S1','candidate_ids':['C2','C3'],'keep_ids':['C2'],
        'status':'resolved','reason':'C3 duplicates supported C2'})
    assert target['status']=='incomplete'
    assert target['keep_ids']==['C1','C2']
