from copy import deepcopy
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw
import pytest

from full_recovery.source_line_coverage import guard
from full_recovery.source_ownership import review, validate_mapping


def row(ident, branch, y):
    return {'id': ident, 'key': ident, 'keys': [ident] if branch == 'shape' else [],
        'refs': [ident], 'layer': branch, 'heads': ['none', 'none'], 'length': 120.,
        'points': np.c_[np.linspace(20, 140, 80), np.full(80, y)],
        'style': {'stroke': '#000000', 'width_px': 1}}


@pytest.mark.parametrize('branch', ['shape', 'arrows'])
def test_visible_unique_native_content_is_protected_in_both_branches(branch):
    source = Image.new('RGB', (180, 100), 'white')
    ImageDraw.Draw(source).line((20, 40, 140, 40), fill='black')
    route = row('only', branch, 40)
    choices, audit = guard([route], [{'candidate_ids': ['only'], 'keep_ids': []}], source)
    assert choices[0]['keep_ids'] == ['only'] and audit[0]['source_visible_samples'] >= 8


def test_unknown_source_coverage_is_not_a_zero_coverage_veto():
    source = Image.new('RGB', (180, 100), 'white')
    routes = [row('old', 'arrows', 40), row('chosen', 'shape', 45)]
    choices, audit = guard(routes, [{'candidate_ids': ['old', 'chosen'], 'keep_ids': ['chosen']}], source)
    assert choices[0]['keep_ids'] == ['chosen']
    assert audit[0]['decision'] == 'abstain' and not audit[0]['deletion_veto']


def certificate(ident):
    return {'certified': True, 'candidate_id': ident,
        'source_visible_points': np.c_[np.arange(30, 131), np.full(101, 40)].tolist()}


def test_one_proven_source_connection_does_not_become_two_when_both_fits_are_displaced():
    source = Image.new('RGB', (180, 100), 'white')
    ImageDraw.Draw(source).line((20, 40, 140, 40), fill='black')
    routes = [row('first', 'shape', 42), row('second', 'arrows', 43)]
    choices, audit = guard(routes, [{'candidate_ids': ['first', 'second'], 'keep_ids': ['first'],
        'source_ridge_certificate': certificate('first')}], source)
    assert choices[0]['keep_ids'] == ['first']
    assert audit[0]['decision'] == 'retain_single_source_representation' and not audit[0]['deletion_veto']


@pytest.mark.parametrize('proposed_branch', ['shape', 'arrows'])
def test_native_visibility_can_atomically_replace_the_winner_without_adding_a_duplicate(proposed_branch):
    source = Image.new('RGB', (180, 100), 'white')
    ImageDraw.Draw(source).line((20, 40, 140, 40), fill='black')
    other = 'arrows' if proposed_branch == 'shape' else 'shape'
    routes = [row('proposed', proposed_branch, 40), row('visible', other, 40)]
    native = {proposed_branch: np.full((100, 180, 3), 255, dtype=np.uint8), other: np.array(source)}
    choices, audit = guard(routes, [{'candidate_ids': ['proposed', 'visible'], 'keep_ids': ['proposed'],
        'source_ridge_certificate': certificate('proposed')}], source, native_views=native)
    assert choices[0]['keep_ids'] == ['visible'] and not audit
    assert choices[0]['source_native_arbitration']['proposed_keep_ids'] == ['proposed']


@pytest.mark.parametrize('kind', ['line', 'frame', 'uncertain'])
def test_physical_survivor_witness_not_a_kind_label_authorizes_duplicate_selection(kind):
    source = [{'id': 'S1', 'kind': kind, 'visible_count': 1}]
    aliases = {ident: row(ident, 'shape', 40) for ident in ('C1', 'C2')}
    raw = {'objects': [{'id': 'S1', 'candidate_ids': ['C1'], 'keep_ids': ['C1'],
        'status': 'resolved', 'reason': 'one physical source occurrence'}],
        'unassigned': [{'candidate_id': 'C2', 'classification': 'unsupported',
            'reason': 'second recovery of the same source',
            'duplicate_of': {'source_id': 'S1', 'candidate_ids': ['C1']}}]}
    result = validate_mapping(raw, source, aliases, {}, allow_omissions=True)
    assert result['objects'][0]['candidate_ids'] == ['C1', 'C2']
    assert result['objects'][0]['keep_ids'] == ['C1'] and not result['unassigned']
    validate_mapping(deepcopy(result), source, aliases, {})


def test_absent_native_survivor_cannot_authorize_deleting_a_shape_line():
    source = [{'id': 'S1', 'kind': 'line', 'visible_count': 1}]
    aliases = {'C1': row('C1', 'shape', 40)}
    raw = {'objects': [{'id': 'S1', 'candidate_ids': ['C1'], 'keep_ids': [],
        'status': 'incomplete', 'reason': 'no usable recovery'}], 'unassigned': []}
    result = validate_mapping(raw, source, aliases, {}, allow_omissions=True)
    assert result['objects'][0]['keep_ids'] == ['C1']
    assert result['objects'][0]['status'] == 'uncertain'


def test_native_coverage_views_include_routes_already_selected_before_model_review(tmp_path):
    routes = [row('shape:known', 'shape', 30), row('arrow:known', 'arrows', 30),
              row('shape:residual', 'shape', 70), row('arrow:residual', 'arrows', 70)]
    source = Image.new('RGB', (180, 110), 'white')
    for y in (30, 70): ImageDraw.Draw(source).line((20, y, 140, y), fill='black')
    path = tmp_path/'source.png'; source.save(path)
    certified = {'choices': [{'candidate_ids': ['shape:known', 'arrow:known'], 'keep_ids': ['shape:known']}],
        'receipts': [], 'claimed_ids': {'shape:known', 'arrow:known'}, 'source_objects': []}
    seen = []
    def native_views(scene, styles, supplied, shape, arrow, clean, folder, renderer):
        seen.extend(r['id'] for r in supplied)
        result = {}
        folder.mkdir(parents=True)
        for branch in ('shape', 'arrows'):
            image = Image.new('RGB', source.size, 'white')
            for r in supplied:
                if r['layer'] == branch:
                    ImageDraw.Draw(image).line([tuple(r['points'][0]), tuple(r['points'][-1])], fill='black')
            target = folder/(branch+'.png'); image.save(target); result[branch] = target
        return result
    class Runtime:
        def call(self, folder, tag, system, prompt, images, validator, budget):
            if tag == 'inventory': return validator({'objects': []})
            return validator({'objects': [], 'unassigned': [{'candidate_id': r['id'],
                'classification': 'context', 'reason': 'unrelated source context'} for r in prompt['candidates']]})
    with patch('full_recovery.source_straight_routes.resolve', return_value=certified), \
         patch('full_recovery.native_line_views.render', side_effect=native_views):
        result = review({'source_size': [180, 110], 'parts': {}}, {}, routes, path, None, None,
                        tmp_path/'review', Runtime())
    assert set(seen) == {r['id'] for r in routes}
    assert result['drop_arrow_names'] == ['arrow:known']
    assert not result['source_coverage_guards']


@pytest.mark.parametrize('kind',['line','arrow','bundle','frame','uncertain'])
@pytest.mark.parametrize('surviving_branch',['shape','arrows'])
@pytest.mark.parametrize('proof',['replacement_witness','source_count'])
def test_same_source_witness_keeps_one_recovery_despite_head_or_position_error(kind,surviving_branch,proof):
    source=Image.new('RGB',(180,100),'white')
    ImageDraw.Draw(source).line((20,40,140,40),fill='black')
    other='arrows' if surviving_branch=='shape' else 'shape'
    routes=[row('rejected',other,40),row('survivor',surviving_branch,44)]
    routes[0]['heads']=['none','triangle']
    choices=[{'candidate_ids':['rejected','survivor'],'keep_ids':['survivor'],
        'source_object_id':'source_a','source_kind':kind,
        'source_replacement_witnesses':[{'candidate_id':'rejected',
            'survivor_ids':['survivor'],'source_object_id':'source_a'}]}]
    if proof=='source_count':
        choices[0].pop('source_replacement_witnesses')
        choices[0]['source_object_selection']={'source_object_id':'source_a',
            'physical_source_count':1,'count_status':'exact','mapping_status':'resolved',
            'survivor_ids':['survivor']}
    selected,audit=guard(routes,choices,source)
    assert selected[0]['keep_ids']==['survivor']
    assert audit[0]['decision']=='retain_selected_source_representation'
    assert audit[0]['replacement_coverage']<.9 and not audit[0]['deletion_veto']


@pytest.mark.parametrize('missing', ['survivor','source'])
def test_invalid_same_source_witness_cannot_remove_unique_native_content(missing):
    source=Image.new('RGB',(180,100),'white')
    ImageDraw.Draw(source).line((20,40,140,40),fill='black')
    routes=[row('rejected','shape',40),row('survivor','arrows',44)]
    witness={'candidate_id':'rejected','survivor_ids':['survivor'],'source_object_id':'source_a'}
    if missing=='survivor':witness['survivor_ids']=['absent']
    else:witness['source_object_id']='different_source'
    selected,audit=guard(routes,[{'candidate_ids':['rejected','survivor'],
        'keep_ids':['survivor'],'source_object_id':'source_a',
        'source_replacement_witnesses':[witness]}],source)
    assert set(selected[0]['keep_ids'])=={'rejected','survivor'}
    assert audit[0]['reason']=='replacement_does_not_cover_original_visible_native_content'


@pytest.mark.parametrize('count,status',[(2,'exact'),(None,'uncertain')])
def test_parallel_or_uncounted_source_is_not_forced_to_one_representation(count,status):
    source=Image.new('RGB',(180,100),'white')
    ImageDraw.Draw(source).line((20,40,140,40),fill='black')
    routes=[row('other','shape',40),row('selected','arrows',44)]
    selected,audit=guard(routes,[{'candidate_ids':['other','selected'],'keep_ids':['selected'],
        'source_object_id':'source_a','source_object_selection':{'source_object_id':'source_a',
            'physical_source_count':count,'count_status':status,'mapping_status':'resolved',
            'survivor_ids':['selected']}}],source)
    assert set(selected[0]['keep_ids'])=={'other','selected'}
