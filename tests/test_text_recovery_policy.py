from copy import deepcopy

from full_recovery.text_recovery_policy import new_conflicts, reconcile, inventory_repair_prompt


def test_rejected_repeated_inventory_proposal_uses_actual_accepted_ledger():
    initial={'task':'inventory','rules':['read original pixels']}
    accepted=[row('first',[0,0,30,10])]
    rejected={'candidate':{'text':'repeated label','replaces':['invented1','invented2']},
              'reason':'unknown replacement IDs'}
    before=deepcopy((initial,accepted,rejected))
    prompt=inventory_repair_prompt(initial,rejected,accepted)
    assert prompt['accepted_source_occurrences']==[{'text':'repeated label','bbox_2d':[0,0,30,10],'kind':'text'}]
    assert 'one ordinary baseline per entry' in ' '.join(prompt['rules'])
    assert 'return every missing occurrence' in ' '.join(prompt['rules'])
    assert 'Return replaces=[]' in ' '.join(prompt['rules'])
    assert (initial,accepted,rejected)==before


def row(ident, box, kind='text'):
    return {'id': ident, 'bbox_2d': box, 'kind': kind, 'text': 'repeated label'}


def test_existing_collision_requests_joint_review_without_deleting_occurrences():
    rows = [row('first', [0, 0, 30, 10]), row('second', [0, 8, 30, 18])]
    updates = {r['id']: [deepcopy(r)] for r in rows}
    before = deepcopy((rows, updates))
    events = new_conflicts(rows, updates)
    assert len(events) == 1 and set(events[0]['parents']) == {'first', 'second'}
    assert events[0]['baseline_overlap'] == .2
    assert reconcile(rows, updates) == (updates, [])
    assert (rows, updates) == before


def test_bad_baseline_does_not_exempt_a_worsened_collision():
    rows = [row('first', [0, 0, 30, 10]), row('second', [0, 8, 30, 18])]
    updates = {'first': [{**rows[0], 'bbox_2d': [0, 7, 30, 17]}], 'second': [rows[1]]}
    kept, warnings = reconcile(rows, updates)
    assert kept == {} and {r['id'] for r in warnings} == {'first', 'second'}
    assert len(rows) == 2 and rows[0]['bbox_2d'] == [0, 0, 30, 10]


def test_resolved_collision_keeps_new_positions_and_identical_strings():
    rows = [row('first', [0, 0, 30, 10]), row('second', [0, 8, 30, 18])]
    updates = {'first': [rows[0]], 'second': [{**rows[1], 'bbox_2d': [0, 20, 30, 30]}]}
    assert not new_conflicts(rows, updates)
    assert reconcile(rows, updates) == (updates, [])


def test_formula_children_keep_the_existing_script_overlap_contract():
    parent = row('formula', [0, 0, 30, 20], 'formula')
    updates = {'formula': [{**parent, 'id': 'formula.G1'}, {**parent, 'id': 'formula.G2'}]}
    assert not new_conflicts([parent], updates)
    assert reconcile([parent], updates) == (updates, [])
