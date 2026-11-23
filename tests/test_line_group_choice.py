"""Model output cannot split one complete implementation or lose independent paths."""
import pytest

from full_recovery.line_group_choice import validate, eligible


def matched_network(edges):
    """Two fitted implementations of each edge, with common node context."""
    rows, pairs = [], []
    for index, (nodes, regions) in enumerate(edges):
        names = []
        for layer in ('shape', 'arrows'):
            name = f'{layer}_{index}'; names.append(name)
            rows.append({'id': name, 'layer': layer, 'endpoint_nodes': nodes,
                         'endpoint_regions': regions})
        pairs.append({'id': f'p{index}', 'a': names[0], 'b': names[1],
                      'nomination': 'same_endpoint_nodes', 'unique_endpoint_pair': True,'p90_px':8.})
    return rows, pairs


def test_shared_destinations_do_not_transitively_merge_a_dense_network():
    from full_recovery.quality_choice import comparison_groups
    boxes = {'u': [0, 0, 10, 10], 'v': [30, 0, 40, 10],
             'x': [0, 50, 10, 60], 'y': [12, 50, 22, 60]}
    edges = [(('u', 'x'), [boxes['u'], boxes['x']]),
             (('u', 'y'), [boxes['u'], boxes['y']]),
             (('v', 'x'), [boxes['v'], boxes['x']]),
             (('v', 'y'), [boxes['v'], boxes['y']])]
    rows, pairs = matched_network(edges)
    assert comparison_groups(pairs, rows) == [[p] for p in pairs]
    # Input endpoint order is an identifier convention, not route direction.
    for row in rows:
        row['endpoint_nodes'] = tuple(reversed(row['endpoint_nodes']))
        row['endpoint_regions'] = list(reversed(row['endpoint_regions']))
    assert comparison_groups(pairs, rows) == [[p] for p in pairs]


def test_ambiguous_parallel_or_uncertain_pairs_are_not_absorbed_into_a_fan():
    from full_recovery.quality_choice import comparison_groups
    rows, pairs = matched_network([
        (('u', 'x'), [[0, 0, 10, 10], [0, 50, 10, 60]]),
        (('u', 'y'), [[0, 0, 10, 10], [12, 50, 22, 60]])])
    pairs[1]['unique_endpoint_pair'] = False
    assert comparison_groups(pairs, rows) == [[pairs[0]], [pairs[1]]]
    pairs[1]['unique_endpoint_pair'] = True
    pairs[1]['native_endpoint_identity_uncertain'] = True
    assert comparison_groups(pairs, rows) == [[pairs[0]], [pairs[1]]]


def test_one_to_many_native_dependency_is_never_split_to_form_endpoint_fans():
    from full_recovery.quality_choice import comparison_groups
    rows, pairs = matched_network([
        (('u', 'x'), [[0, 0, 10, 10], [0, 50, 10, 60]]),
        (('u', 'y'), [[0, 0, 10, 10], [12, 50, 22, 60]])])
    whole = {'id': 'whole', 'a': 'shape_0', 'b': 'arrow_whole'}
    assert comparison_groups([*pairs, whole], rows) == [[pairs[0], whole], [pairs[1]]]


def test_unopposed_route_at_the_same_node_stays_outside_the_group_and_deletion_set():
    from full_recovery.quality_choice import comparison_groups, apply_choices
    rows, pairs = matched_network([
        (('u', 'x'), [[0, 0, 10, 10], [0, 50, 10, 60]]),
        (('u', 'y'), [[0, 0, 10, 10], [12, 50, 22, 60]])])
    rows.append({'id': 'arrows_unique', 'layer': 'arrows', 'endpoint_nodes': ('u', 'z'),
                 'endpoint_regions': [[0, 0, 10, 10], [60, 50, 70, 60]]})
    for row in rows:
        row.update(key=row['id'], keys=[row['id']], refs=[row['id']])
    groups = comparison_groups(pairs, rows)
    assert groups == [[p] for p in pairs]
    members = sorted({p[k] for group in groups for p in group for k in ('a', 'b')})
    plan = apply_choices(rows, [{'candidate_ids': members,
                                'keep_ids': [n for n in members if n.startswith('shape')]}])
    assert set(plan['drop_arrow_names']) == {'arrows_0', 'arrows_1'}
    assert 'arrows_unique' not in members


def test_common_origin_does_not_couple_distant_targets_or_bypasses():
    from full_recovery.quality_choice import comparison_groups
    rows,pairs=matched_network([
        (('u','x'),[[0,0,10,10],[0,50,10,60]]),
        (('u','y'),[[0,0,10,10],[200,50,210,60]])])
    assert comparison_groups(pairs,rows)==[[pairs[0]],[pairs[1]]]


def test_local_route_choices_do_not_merge_when_fit_disagreement_grows():
    from full_recovery.quality_choice import comparison_groups
    rows,pairs=matched_network([
        (('u','x'),[[0,0,10,10],[0,50,10,60]]),
        (('u','y'),[[0,0,10,10],[20,50,30,60]])])
    assert comparison_groups(pairs,rows)==[[pairs[0]],[pairs[1]]]
    pairs[0]['p90_px']=12.
    assert comparison_groups(pairs,rows)==[[p] for p in pairs]
    for pair in pairs:pair.pop('p90_px')
    assert comparison_groups(pairs,rows)==[[pairs[0]],[pairs[1]]]


def test_complete_group_batching_uses_evidence_groups_instead_of_line_count(tmp_path):
    import numpy as np
    from unittest.mock import patch
    from PIL import Image
    from full_recovery.line_group_choice import compare
    table, groups, contexts, native = {}, [], {}, {}
    for index in range(7):
        ident=f'p{index}'; nodes=[]
        for layer in ('arrows', 'shape'):
            for edge in range(7):
                name=f'{ident}:{layer}:{edge}';nodes.append(name)
                table[name]={'id':name,'layer':layer,'points':np.array([[0.,0.],[10.,10.]]),
                             'endpoint_nodes':('u','v')}
            native[(ident,layer)]=None
        groups.append(([{'id':ident}],nodes))
        contexts[ident]={'N1':{'native_key':'u','bbox_2d':[0,0,1,1]},
                         'N2':{'native_key':'v','bbox_2d':[9,9,10,10]}}
    class Model:
        config={'mapping_workers':1}
        batch_sizes=[]
        def for_stage(self,**overrides):return self
        def call(self,folder,tag,system,prompt,images,validator,budget):
            self.batch_sizes.append(len(prompt['groups']))
            for entry in prompt['groups']:
                assert all(c['endpoint_context_ids']==['N1','N2'] for c in entry['candidate_geometry'].values())
            return validator({'groups':[{'id':entry['id'],'status':'selected','keep_branch':'B2',
                                         'reason':'SOURCE supports the complete matched local group'}
                                       for entry in prompt['groups']]})
    model=Model()
    with patch('full_recovery.line_group_choice.aligned_view',return_value=None):
        choices,calls=compare(groups,table,Image.new('RGB',(20,20)),native,
            lambda batch,nodes:([],[(0,0,20,20)]),contexts,tmp_path,model)
    assert model.batch_sizes==[6,1] and calls==2
    assert len(choices)==7 and all(len(c['keep_ids'])==7 for c in choices.values())


def group():
    return {'g': {'aliases': {'C1': 'a', 'C2': 's'}, 'required': [],
                  'branches': {'C1': 'B1', 'C2': 'B2'},
                  'implementations': {'B1': ['C1'], 'B2': ['C2']}}}


def reply(status, branch, **extra):
    return {'groups': [{'id': 'g', 'status': status, 'keep_branch': branch,
                         'reason': 'visible source endpoints and complete route', **extra}]}


def test_complete_choice_cannot_keep_both_via_extra_source_ids():
    raw = reply('selected', 'B2', source_objects=[{'id': 'S1'}, {'id': 'S2'}])
    assert validate(raw, group())['groups'][0]['keep_ids'] == ['C2']
    with pytest.raises(ValueError):
        validate(reply('selected', 'B2', keep_ids=['C1', 'C2']), group())
    with pytest.raises(ValueError):
        validate(reply('selected', 'B2', keep_ids=[]), group())


def test_distinct_paths_keep_union_and_uncertainty_has_no_deletion_authority():
    separate=validate(reply('independent', None), group())['groups'][0]
    assert separate['keep_ids'] == ['C1', 'C2'] and separate['status']=='unresolved'
    assert validate(reply('unresolved', None), group())['groups'][0]['keep_ids'] == ['C1', 'C2']
    with pytest.raises(ValueError):
        validate(reply('unresolved', 'B1'), group())


def test_immutable_native_member_cannot_be_dropped():
    prepared = group(); prepared['g']['required'] = ['C2']
    with pytest.raises(ValueError):
        validate(reply('selected', 'B1'), prepared)


def test_group_receipts_cannot_be_duplicated_or_omitted():
    raw = reply('selected', 'B1')
    raw['groups'] *= 2
    with pytest.raises(ValueError):
        validate(raw, group())
    with pytest.raises(ValueError):
        validate({'groups': []}, group())


def test_output_envelope_normalizes_without_relaxing_any_group_constraint():
    assert validate({'output': reply('selected', 'B2')}, group()) == validate(reply('selected', 'B2'), group())
    with pytest.raises(ValueError):
        validate({'output': reply('selected', 'B2', keep_ids=['C1', 'C2'])}, group())
    with pytest.raises(ValueError):
        validate({'groups': [], 'output': reply('selected', 'B2')}, group())


def test_uncertain_endpoint_identity_stays_with_detailed_source_matching():
    rows = [{'layer': 'shape'}, {'layer': 'arrows'}]
    assert eligible(rows, [{'nomination': 'same_endpoint_nodes'}])
    assert eligible(rows, [{'nomination': 'shared_network_span', 'shared_native_span': True}])
    assert eligible(rows, [{'native_endpoint_identity_uncertain': True}])


def test_uncertain_nomination_uses_existing_comparison_without_automatic_deletion(tmp_path):
    import numpy as np
    from unittest.mock import patch
    from PIL import Image
    from full_recovery.quality_choice import review
    points=np.linspace([10,20],[50,20],64)
    rows=[{'id': name,'layer': layer,'key': name,'owner':'o','keys':[],
           'refs':[name],'points':points,'length':40.,'heads':['none','none'],
           'style':{'stroke':'#000000','width_px':1}}
          for name,layer in [('s','shape'),('a','arrows')]]
    source=tmp_path/'source.png';Image.new('RGB',(80,60),'white').save(source)
    pair={'id':'p','a':'s','b':'a','nomination':'unpaired_node_route','native_endpoint_identity_uncertain':True}
    class Forbidden:
        def call(self,*args):raise AssertionError('No additional existence vote')
    seen=[]
    def compare(groups,*args):
        seen.extend(groups)
        return {'p':{'candidate_ids':['a','s'],'keep_ids':['a','s'],
                     'pairs':['p'],'selection_unit':'unresolved_competitor_group',
                     'source_relation':'unresolved','whole_group_branch':None,
                     'reason':'SOURCE does not resolve which native candidate is complete.'}},1
    with patch('full_recovery.quality_choice.candidates',return_value=rows), \
         patch('full_recovery.quality_choice.nominate',return_value=[pair]), \
         patch('full_recovery.quality_choice.selection_subset'), \
         patch('full_recovery.line_group_choice.compare',side_effect=compare), \
         patch('shape_integration.styling.export.render_native',return_value=source):
        result=review({'parts':{},'source_size':[80,60]},{},[],source,None,None,tmp_path/'review',Forbidden())
    assert len(seen)==1 and result['model_calls']==1 and not result['removed']
    assert not result['line_ownership_complete'] and len(result['unresolved_competitor_groups'])==1


@pytest.mark.parametrize('other,expected', [([[33, 58], [21, 42]], 0), ([[47, 58], [47, 38]], 1)])
def test_shared_node_uncertainty_allows_translation_but_not_an_unrelated_path(other, expected):
    import numpy as np
    from full_recovery.quality_choice import nominate
    def route(ident, layer, endpoints):
        points = np.linspace(*np.asarray(endpoints, float), 64)
        return {'id': ident, 'layer': layer, 'points': points,
                'length': float(np.linalg.norm(points[-1]-points[0])),
                'style': {'width_px': 1}, 'heads': ['none', 'none'],
                'network_nodes': ['shared_box'], 'node_radius_px': 25}
    pairs = nominate([route('a', 'arrows', [[40, 80], [40, 60]]), route('s', 'shape', other)])
    assert len(pairs) == expected
    if expected:assert pairs[0]['native_endpoint_identity_uncertain']


def test_complementary_recoveries_are_rendered_together_and_selected_once(tmp_path):
    from unittest.mock import patch
    from lxml import etree
    from PIL import Image
    from pptx import Presentation
    from pptx.enum.shapes import MSO_CONNECTOR
    from full_recovery.compose import read_package, P, SLIDE
    from full_recovery.quality_choice import review
    shape_deck = Presentation(); slide = shape_deck.slides.add_slide(shape_deck.slide_layouts[6])
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, 10, 20, 110, 20); line.name = 'owner/long'
    shape_path = tmp_path/'shape.pptx'; shape_deck.save(shape_path)
    arrow_deck = Presentation(); slide = arrow_deck.slides.add_slide(arrow_deck.slide_layouts[6])
    for name, start, end in [('a', 10, 60), ('b', 60, 110)]:
        line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, start, 20, end, 20); line.name = name
    arrow_path = tmp_path/'arrows.pptx'; arrow_deck.save(arrow_path)
    scene = {'source_size': [120, 60], 'parts': {'line': {'object_id': 'owner', 'id': 'long',
        'status': 'ok', 'kind': 'path', 'commands': [['M', 10, 20], ['L', 110, 20]],
        'bbox_2d': [10, 20, 110, 20]}}}
    style = {'stroke': '#000000', 'fill': None, 'width_px': 1}
    arrows = [{'id': name, 'start': [start, 20], 'end': [end, 20], 'heads': ['none', 'none'],
               'style': {'color': '#000000', 'width_px': 1}} for name, start, end in [('a', 10, 60), ('b', 60, 110)]]
    source = tmp_path/'source.png'; Image.new('RGB', (120, 60), 'white').save(source)
    rendered = []
    def render(path, folder, size, timeout):
        tree = etree.fromstring(read_package(path)[SLIDE])
        names = {p.get('name') for p in tree.iter('{'+P+'}cNvPr')} - {''}
        rendered.append(names)
        folder.mkdir(parents=True, exist_ok=True)
        target = folder/'native.png'; Image.new('RGB', size, 'white').save(target)
        return target
    class Model:
        config = {'mapping_workers': 1}
        calls = 0
        def for_stage(self, **overrides):return self
        def call(self, folder, tag, system, prompt, images, validator, budget):
            self.calls += 1
            assert tag == 'review' and len(prompt['groups']) == 1
            current = prompt['groups'][0]
            assert current['branch_implementations'] == {'B1': ['C1', 'C2'], 'B2': ['C3']}
            return validator({'groups': [{'id': current['id'], 'status': 'selected', 'keep_branch': 'B1',
                                         'reason': 'Both complementary source spans are retained together'}]})
    model = Model()
    with patch('shape_integration.styling.export.render_native', render):
        result = review(scene, {'line': style}, arrows, source, shape_path, arrow_path, tmp_path/'review', model)
    assert model.calls == result['model_calls'] == 1
    assert len(rendered) == 2 and any({'a', 'b'} <= names for names in rendered)
    assert result['drop_shape_keys'] == ['line'] and not result['drop_arrow_names']
    assert result['branch_choices'][0]['keep_ids'] == ['arrow:a', 'arrow:b']

def test_decision_list_transport_keeps_exact_group_and_complete_branch_constraints():
    response=reply('selected','B2')['groups']
    assert validate(response,group())==validate({'groups':response},group())
    with pytest.raises(ValueError):validate(response*2,group())
    with pytest.raises(ValueError):validate([],group())
    with pytest.raises(ValueError):validate(reply('selected','B2',keep_ids=['C1','C2'])['groups'],group())
